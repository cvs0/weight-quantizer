"""Unsigned integer quantization of a weight matrix.

Codes are integers q in [0, qmax], qmax = 2^bits - 1. Reconstruction is

    W_hat[m, k] = scale[m, g] * (q[m, k] - zero[m, g]) * channel_scale[k]

with g = k // group_size. A missing channel_scale is the constant 1.
`torch.round` is round-half-to-even.

Symmetric (the GPTQ zero point):

    amax = max(abs(W)) over the reduction axes
    scale = 2 * amax / qmax     (scale = 1 if amax = 0)
    zero = (qmax + 1) / 2
    q = clip(round(W / scale) + zero, 0, qmax)

Asymmetric, with the range expanded so it contains 0:

    xmin = min(min(W), 0)
    xmax = max(max(W), 0)
    scale = (xmax - xmin) / qmax
    zero = clip(round(-xmin / scale), 0, qmax)
    q = clip(round(W / scale) + zero, 0, qmax)

If xmin = xmax = 0 the block is replaced by [-1, 1] before the scale is formed,
so the zero tensor still has a defined affine map.

Reduction:

    group_size is None: one scale per output row (per-channel), group_size stored as K
    group_size == 0: one scale for the whole matrix (per-tensor)
    group_size > 0: contiguous groups along K, last group shorter when K is not a multiple

The positive end of the symmetric grid stops at (qmax - zero) * scale = amax * (qmax - 1) / qmax,
which is half a bin below amax. Values taken from the same block that produced the
scale therefore satisfy |W - W_hat| <= scale / 2, up to fp32 rounding.
Asymmetric zero points are themselves rounded, so the bound used here is one bin:
|W - W_hat| <= scale, again up to fp32 rounding.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import torch

from ptqkit.pack import pack_int4, unpack_int4

Granularity = Literal["tensor", "channel", "group"]


class Method(str, enum.Enum):
    RTN = "rtn"
    GPTQ = "gptq"
    AWQ = "awq"


@dataclass(frozen=True)
class QuantConfig:
    """Weight-only quantization configuration. Activations stay fp32."""

    bits: int = 4
    symmetric: bool = True
    group_size: int | None = 128
    method: Method = Method.RTN
    percdamp: float = 0.01
    blocksize: int = 128
    awq_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if self.group_size is not None and self.group_size < 0:
            raise ValueError("group_size must be None, 0, or positive")
        if self.percdamp < 0:
            raise ValueError("percdamp must be non-negative")
        if self.blocksize <= 0:
            raise ValueError("blocksize must be positive")
        if not self.awq_alphas:
            raise ValueError("awq_alphas must be non-empty")


def granularity_of(config: QuantConfig) -> Granularity:
    if config.group_size == 0:
        return "tensor"
    if config.group_size is None:
        return "channel"
    return "group"


def qmax_of(bits: int) -> int:
    return (1 << bits) - 1


@dataclass(frozen=True)
class QuantizedWeight:
    """Quantized [out, in] weight.

    `codes` is uint8 [out, in] for int8, or packed uint8 [out, ceil(in/2)] for int4.
    `scales` and `zeros` are float32 [scale_rows, groups] with scale_rows in {1, out}.
    `group_size` is the K-stride of a group. Per-tensor and per-channel use K.
    `channel_scale` is an optional float32 [in] multiplier applied after the affine map.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int
    group_size: int
    in_features: int
    out_features: int
    symmetric: bool
    packed: bool
    granularity: Granularity
    channel_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.packed != (self.bits == 4):
            raise ValueError("only int4 storage is packed")
        if self.scales.shape != self.zeros.shape:
            raise ValueError("scales and zeros must have the same shape")
        if self.scales.ndim != 2:
            raise ValueError("scales must have shape [rows, groups]")
        rows, groups = int(self.scales.shape[0]), int(self.scales.shape[1])
        if rows not in (1, self.out_features):
            raise ValueError("scale rows must be 1 or out_features")
        n_groups = (self.in_features + self.group_size - 1) // self.group_size
        if groups != n_groups:
            raise ValueError("scale groups do not match group_size")
        if self.channel_scale is not None and int(self.channel_scale.numel()) != self.in_features:
            raise ValueError("channel_scale must have one entry per input channel")


def absolute_error_bound(scale: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
    """Per-element RTN bound. See the module docstring."""
    factor = 0.5 if symmetric else 1.0
    return scale * factor


def qparams(
    block: torch.Tensor,
    *,
    symmetric: bool,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scale, zero with shape [rows, 1] for `block` of shape [rows, cols]."""
    if block.ndim != 2:
        raise ValueError("block must have shape [rows, cols]")
    qmax = qmax_of(bits)
    if symmetric:
        amax = block.detach().abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, (2 * amax) / qmax, torch.ones_like(amax))
        zero = torch.full_like(scale, float(qmax + 1) / 2.0)
        return scale, zero
    xmin = block.detach().amin(dim=-1, keepdim=True)
    xmax = block.detach().amax(dim=-1, keepdim=True)
    xmin = torch.minimum(xmin, torch.zeros_like(xmin))
    xmax = torch.maximum(xmax, torch.zeros_like(xmax))
    degenerate = (xmin == 0) & (xmax == 0)
    xmin = torch.where(degenerate, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(degenerate, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = torch.round(-xmin / scale)
    zero = torch.clamp(zero, 0, qmax)
    return scale, zero


def integer_codes(
    block: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    qmax: int,
) -> torch.Tensor:
    """Round `block` onto the affine grid. Returns int32 codes."""
    q = torch.round(block / scale) + zero
    q = torch.clamp(q, 0, qmax)
    return q.round().to(dtype=torch.int32)


def assemble_quantized(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    granularity: Granularity,
    channel_scale: torch.Tensor | None = None,
) -> QuantizedWeight:
    """Pack `codes` [out, in] and attach fp32 scale/zero tensors."""
    if codes.ndim != 2:
        raise ValueError("codes must have shape [out, in]")
    out, inn = int(codes.shape[0]), int(codes.shape[1])
    packed = bits == 4
    stored_codes = pack_int4(codes) if packed else codes.to(dtype=torch.uint8)
    stored_codes = stored_codes.contiguous()
    cs: torch.Tensor | None
    if channel_scale is None:
        cs = None
    else:
        cs = channel_scale.detach().to(dtype=torch.float32).contiguous()
    return QuantizedWeight(
        codes=stored_codes,
        scales=scales.detach().to(dtype=torch.float32).contiguous(),
        zeros=zeros.detach().to(dtype=torch.float32).contiguous(),
        bits=bits,
        group_size=group_size,
        in_features=inn,
        out_features=out,
        symmetric=symmetric,
        packed=packed,
        granularity=granularity,
        channel_scale=cs,
    )


def quantize_weight(
    weight: torch.Tensor,
    config: QuantConfig,
    *,
    pre_scale: torch.Tensor | None = None,
) -> QuantizedWeight:
    """Round-to-nearest quantize `weight` [out, in].

    If `pre_scale` [in] is set, the matrix that is quantized is W diag(pre_scale),
    and the stored channel_scale is 1/pre_scale. Reconstruction then equals
    dequant(RTN(W diag(s))) diag(s)^{-1}.
    """
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out, in]")
    w = weight.detach().to(dtype=torch.float32)
    if not torch.isfinite(w).all():
        raise ValueError("weight must be finite")
    stored_channel: torch.Tensor | None = None
    if pre_scale is not None:
        s = pre_scale.detach().to(dtype=torch.float32, device=w.device).reshape(-1)
        if int(s.numel()) != int(w.shape[1]):
            raise ValueError("pre_scale must have one entry per input channel")
        if not torch.isfinite(s).all() or bool(torch.any(s <= 0).item()):
            raise ValueError("pre_scale must be finite and positive")
        w = w * s.unsqueeze(0)
        stored_channel = (1.0 / s).contiguous()

    out, inn = int(w.shape[0]), int(w.shape[1])
    qmax = qmax_of(config.bits)
    granularity = granularity_of(config)
    if granularity == "tensor":
        scale, zero = qparams(w.reshape(1, -1), symmetric=config.symmetric, bits=config.bits)
        codes = integer_codes(w, scale, zero, qmax)
        index_group = inn
    elif granularity == "channel":
        scale, zero = qparams(w, symmetric=config.symmetric, bits=config.bits)
        codes = integer_codes(w, scale, zero, qmax)
        index_group = inn
    else:
        assert config.group_size is not None
        gs = int(config.group_size)
        codes = torch.empty(out, inn, dtype=torch.int32)
        scale_cols: list[torch.Tensor] = []
        zero_cols: list[torch.Tensor] = []
        for start in range(0, inn, gs):
            end = min(start + gs, inn)
            block = w[:, start:end]
            sc, zz = qparams(block, symmetric=config.symmetric, bits=config.bits)
            codes[:, start:end] = integer_codes(block, sc, zz, qmax)
            scale_cols.append(sc)
            zero_cols.append(zz)
        scale = torch.cat(scale_cols, dim=1)
        zero = torch.cat(zero_cols, dim=1)
        index_group = gs

    return assemble_quantized(
        codes,
        scale,
        zero,
        bits=config.bits,
        group_size=index_group,
        symmetric=config.symmetric,
        granularity=granularity,
        channel_scale=stored_channel,
    )


def dequantize_weight(qw: QuantizedWeight) -> torch.Tensor:
    """Reconstruct an fp32 [out, in] matrix."""
    if qw.packed:
        codes = unpack_int4(qw.codes, qw.in_features).to(dtype=torch.float32)
    else:
        codes = qw.codes.to(dtype=torch.float32)
    idx = torch.arange(qw.in_features, device=codes.device) // qw.group_size
    if int(qw.scales.shape[0]) == 1:
        scale = qw.scales[0, idx].unsqueeze(0)
        zero = qw.zeros[0, idx].unsqueeze(0)
    else:
        scale = qw.scales[:, idx]
        zero = qw.zeros[:, idx]
    hat = (codes - zero) * scale
    if qw.channel_scale is not None:
        hat = hat * qw.channel_scale.unsqueeze(0)
    return hat


def element_scales(qw: QuantizedWeight) -> torch.Tensor:
    """Broadcast group scales to [out, in] (channel_scale is not included)."""
    idx = torch.arange(qw.in_features, device=qw.scales.device) // qw.group_size
    if int(qw.scales.shape[0]) == 1:
        return qw.scales[0, idx].unsqueeze(0).expand(qw.out_features, qw.in_features)
    return qw.scales[:, idx]
