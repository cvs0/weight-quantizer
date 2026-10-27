"""GPTQ column quantization.

Objective, for one linear map and a calibration matrix X [n, K]:

    min_q || X W^T - X W_hat(q)^T ||_F^2

The Hessian of that quadratic in the columns of W is H = X^T X / n, up to a
positive constant that cancels in the damped inverse below. Columns are
quantized left to right. After column i is replaced by its grid point, the
remaining columns are updated by the OBS step used in GPTQ:

    err = (w_i - w_hat_i) / Hinv[i, i]
    W[:, i:] -= err * Hinv[i, i:]

Hinv is the upper Cholesky factor of H^{-1}, not the dense inverse. Damping

    H_ii += percdamp * mean(diag(H))

is applied before the factorization. A diagonal entry that is exactly zero
(an input channel that never fired) is isolated: its row and column are
cleared, the diagonal is set to 1, and the weight column is set to 0.

Group scales are computed from the updated float weights at the start of each
group. Per-channel and per-tensor scales are computed once, before the sweep.
The grid is the same RTN grid as `quantize_weight`.
"""

from __future__ import annotations

import torch

from ptqkit.quant import (
    Method,
    QuantConfig,
    QuantizedWeight,
    assemble_quantized,
    granularity_of,
    integer_codes,
    qmax_of,
    qparams,
)


def _factor_inverse(hessian: torch.Tensor, percdamp: float) -> torch.Tensor:
    """Return the upper Cholesky factor of the damped inverse."""
    h = 0.5 * (hessian + hessian.transpose(0, 1))
    diag = torch.diag(h)
    dead = diag == 0
    if bool(torch.any(dead).item()):
        h = h.clone()
        h[dead, :] = 0
        h[:, dead] = 0
        # Boolean indexing on both axes writes the full sub-block. Write the diagonal.
        idx = torch.arange(h.shape[0], device=h.device)[dead]
        h[idx, idx] = 1
    damp = percdamp * torch.mean(torch.diag(h))
    if not bool(torch.isfinite(damp).item()) or float(damp.item()) < 0:
        damp = torch.zeros((), dtype=h.dtype, device=h.device)
    eye = torch.arange(h.shape[0], device=h.device)
    h = h.clone()
    h[eye, eye] = h[eye, eye] + damp
    last_error: Exception | None = None
    for _ in range(6):
        try:
            chol = torch.linalg.cholesky(h)
            inv = torch.cholesky_inverse(chol)
            factor = torch.linalg.cholesky(inv, upper=True)
            if not isinstance(factor, torch.Tensor):
                raise TypeError("Cholesky did not return a tensor")
            return factor
        except RuntimeError as exc:
            last_error = exc
            bump = torch.mean(torch.diag(h)) * 0.1 + 1e-6
            h[eye, eye] = h[eye, eye] + bump
    raise RuntimeError("damped Hessian Cholesky failed") from last_error


def _row_vectors(
    scale: torch.Tensor,
    zero: torch.Tensor,
    rows: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    sc = scale.reshape(-1).to(dtype=dtype)
    zz = zero.reshape(-1).to(dtype=dtype)
    if sc.numel() == 1:
        return sc.expand(rows), zz.expand(rows)
    if sc.numel() != rows:
        raise ValueError("scale rows do not match the weight")
    return sc, zz


def _obs_block(
    weight: torch.Tensor,
    hinv: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    qmax: int,
    start: int,
    end: int,
) -> None:
    """Quantize columns [start, end) and correct columns end: in place."""
    count = end - start
    block = weight[:, start:end].clone()
    hinv_block = hinv[start:end, start:end]
    err_block = torch.zeros_like(block)
    sc, zz = _row_vectors(scale, zero, int(weight.shape[0]), weight.dtype)
    for i in range(count):
        column = block[:, i]
        d = hinv_block[i, i]
        q = integer_codes(column.unsqueeze(1), sc.unsqueeze(1), zz.unsqueeze(1), qmax).squeeze(1)
        deq = sc * (q.to(dtype=weight.dtype) - zz)
        codes[:, start + i] = q
        denom = torch.clamp(d, min=torch.tensor(1e-8, dtype=weight.dtype, device=weight.device))
        err = (column - deq) / denom
        if not bool(torch.isfinite(err).all().item()):
            err = torch.zeros_like(column)
        tail = hinv_block[i, i:]
        block[:, i:] = block[:, i:] - err.unsqueeze(1) * tail.unsqueeze(0)
        err_block[:, i] = err
    # Within-block corrections stay in `block` (they already affected later
    # columns of this block). Only columns past `end` are written back, which
    # is the GPTQ tail update.
    if end < weight.shape[1]:
        weight[:, end:] = weight[:, end:] - err_block @ hinv[start:end, end:]


def gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    config: QuantConfig,
) -> QuantizedWeight:
    """Quantize `weight` [out, in] with Hessian `hessian` [in, in]."""
    if config.method not in (Method.GPTQ, Method.RTN):
        raise ValueError("gptq_quantize expects a GPTQ or RTN grid config")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out, in]")
    out, inn = int(weight.shape[0]), int(weight.shape[1])
    if hessian.shape != (inn, inn):
        raise ValueError("hessian must have shape [in, in]")
    work = weight.detach().to(dtype=torch.float64).clone()
    h = hessian.detach().to(dtype=torch.float64)
    dead = torch.diag(h) == 0
    if bool(torch.any(dead).item()):
        work[:, dead] = 0
    hinv = _factor_inverse(h, config.percdamp)
    qmax = qmax_of(config.bits)
    codes = torch.zeros(out, inn, dtype=torch.int32)
    granularity = granularity_of(config)

    if granularity == "group":
        assert config.group_size is not None
        gs = int(config.group_size)
        scale_cols: list[torch.Tensor] = []
        zero_cols: list[torch.Tensor] = []
        for start in range(0, inn, gs):
            end = min(start + gs, inn)
            sc, zz = qparams(work[:, start:end], symmetric=config.symmetric, bits=config.bits)
            _obs_block(work, hinv, codes, sc, zz, qmax, start, end)
            scale_cols.append(sc)
            zero_cols.append(zz)
        scales = torch.cat(scale_cols, dim=1)
        zeros = torch.cat(zero_cols, dim=1)
        index_group = gs
    else:
        if granularity == "tensor":
            scales, zeros = qparams(
                work.reshape(1, -1),
                symmetric=config.symmetric,
                bits=config.bits,
            )
        else:
            scales, zeros = qparams(work, symmetric=config.symmetric, bits=config.bits)
        for start in range(0, inn, config.blocksize):
            end = min(start + config.blocksize, inn)
            _obs_block(work, hinv, codes, scales, zeros, qmax, start, end)
        index_group = inn

    if not bool(torch.isfinite(codes.to(dtype=torch.float32)).all().item()):
        raise RuntimeError("GPTQ produced non-finite codes")
    return assemble_quantized(
        codes,
        scales,
        zeros,
        bits=config.bits,
        group_size=index_group,
        symmetric=config.symmetric,
        granularity=granularity,
        channel_scale=None,
    )
