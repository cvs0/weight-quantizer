"""Reference and CPU-kernel dequantize-matmul.

Both compute y = x @ W_hat^T with W_hat from `dequantize_weight`. The kernel
keeps the integer matrix packed and dequantizes one output row at a time.
"""

from __future__ import annotations

from typing import Any

import torch

from ptqkit.quant import QuantizedWeight, dequantize_weight


def _extension() -> Any:
    from ptqkit import _kernels

    return _kernels


def kernel_info() -> dict[str, bool]:
    info: dict[str, bool] = _extension().kernel_info()
    return info


def dequant_gemm(
    x: torch.Tensor,
    qw: QuantizedWeight,
    *,
    impl: str = "reference",
    nthreads: int = 0,
) -> torch.Tensor:
    """Apply a quantized weight. `x` is [..., in]. Output is [..., out]."""
    if impl not in ("reference", "kernel"):
        raise ValueError("impl must be 'reference' or 'kernel'")
    if x.shape[-1] != qw.in_features:
        raise ValueError(f"x has trailing dim {x.shape[-1]}, weight expects {qw.in_features}")
    lead = x.shape[:-1]
    flat = x.detach().reshape(-1, qw.in_features).to(dtype=torch.float32)
    if impl == "reference":
        y = flat @ dequantize_weight(qw).transpose(0, 1)
    else:
        if flat.device.type != "cpu":
            raise RuntimeError("the dequant kernel runs on CPU")
        ext = _extension()
        channel = None if qw.channel_scale is None else qw.channel_scale.detach().cpu().numpy()
        y_np = ext.dequant_gemm(
            flat.contiguous().numpy(),
            qw.codes.detach().contiguous().cpu().numpy(),
            qw.scales.detach().contiguous().cpu().numpy(),
            qw.zeros.detach().contiguous().cpu().numpy(),
            channel,
            int(qw.group_size),
            int(qw.bits),
            int(nthreads),
        )
        y = torch.from_numpy(y_np)
    return y.reshape(*lead, qw.out_features)
