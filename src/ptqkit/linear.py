"""Drop-in quantized replacement for nn.Linear and GPT-2 Conv1D.

Forward is y = x @ W_hat^T + bias, with W_hat the dequantized integer weight.
`impl="kernel"` calls the C++ dequant-matmul. `materialize` caches W_hat so a
later forward is a plain fp32 GEMM. That cache is for evaluation throughput of
the quantized weights. It is cleared by `_apply` (device or dtype moves).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ptqkit.gemm import dequant_gemm
from ptqkit.quant import Granularity, QuantizedWeight


class QuantizedLinear(nn.Module):
    codes: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: torch.Tensor | None
    channel_scale: torch.Tensor | None

    def __init__(
        self,
        qw: QuantizedWeight,
        bias: torch.Tensor | None,
        *,
        impl: str = "reference",
    ) -> None:
        super().__init__()
        if impl not in ("reference", "kernel"):
            raise ValueError("impl must be 'reference' or 'kernel'")
        self.in_features = qw.in_features
        self.out_features = qw.out_features
        self.bits = qw.bits
        self.group_size = qw.group_size
        self.symmetric = qw.symmetric
        self.packed = qw.packed
        self.granularity: Granularity = qw.granularity
        self.impl = impl
        self.awq_alpha: float | None = None
        self._cached: torch.Tensor | None = None
        self.register_buffer("codes", qw.codes.detach().contiguous())
        self.register_buffer("scales", qw.scales.detach().contiguous())
        self.register_buffer("zeros", qw.zeros.detach().contiguous())
        if qw.channel_scale is not None:
            self.register_buffer("channel_scale", qw.channel_scale.detach().contiguous())
        else:
            self.channel_scale = None
        if bias is not None:
            if int(bias.numel()) != qw.out_features:
                raise ValueError("bias does not match out_features")
            self.register_buffer("bias", bias.detach().to(dtype=torch.float32).contiguous())
        else:
            self.bias = None

    def quantized_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            codes=self.codes,
            scales=self.scales,
            zeros=self.zeros,
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            symmetric=self.symmetric,
            packed=self.packed,
            granularity=self.granularity,
            channel_scale=self.channel_scale,
        )

    def materialize(self) -> None:
        from ptqkit.quant import dequantize_weight

        self._cached = dequantize_weight(self.quantized_weight())

    def clear_cache(self) -> None:
        self._cached = None

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> QuantizedLinear:
        self._cached = None
        super()._apply(fn, recurse=recurse)  # type: ignore[no-untyped-call]
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected in_features={self.in_features}, got {x.shape[-1]}")
        lead = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        if self._cached is not None:
            y = nn.functional.linear(flat, self._cached, self.bias)
        else:
            y = dequant_gemm(flat, self.quantized_weight(), impl=self.impl)
            if self.bias is not None:
                y = y + self.bias
        return y.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        alpha = "" if self.awq_alpha is None else f", awq_alpha={self.awq_alpha}"
        return (
            f"in={self.in_features}, out={self.out_features}, bits={self.bits}, "
            f"group={self.group_size}, sym={self.symmetric}, impl={self.impl}{alpha}"
        )
