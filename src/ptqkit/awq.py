"""Activation-aware weight scaling, then RTN.

For calibration activations X [n, K] and weights W [out, K], let

    m_k = mean_i |X_ik|
    s(alpha)_k = m_k^alpha / mean(m^alpha)

The search picks alpha from `config.awq_alphas` to minimize

    || X W^T - X W_hat(alpha)^T ||_F^2
    W_hat(alpha) = dequant(RTN(W diag(s(alpha)))) diag(s(alpha))^{-1}

on a deterministic stride subsample of at most 2048 rows. Channel magnitudes
use every captured row. alpha = 0 yields s = 1, which is plain RTN, so the
selected search loss is at most the RTN search loss.

The stored channel_scale is 1/s. A global positive rescaling of s cancels
between the quantizer range and the stored multiplier, so only the relative
channel scales matter.
"""

from __future__ import annotations

import torch

from ptqkit.quant import QuantConfig, QuantizedWeight, dequantize_weight, quantize_weight

_SEARCH_ROWS = 2048


def _subsample(activations: torch.Tensor, limit: int) -> torch.Tensor:
    n = int(activations.shape[0])
    if n <= limit:
        return activations
    step = max(n // limit, 1)
    return activations[::step][:limit]


def awq_quantize(
    weight: torch.Tensor,
    activations: torch.Tensor,
    config: QuantConfig,
    *,
    mean_abs: torch.Tensor | None = None,
) -> tuple[QuantizedWeight, float]:
    """Return the quantized weight and the alpha that won the search."""
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out, in]")
    if activations.ndim != 2 or int(activations.shape[1]) != int(weight.shape[1]):
        raise ValueError("activations must have shape [n, in]")
    w = weight.detach().to(dtype=torch.float32)
    x_all = activations.detach().to(dtype=torch.float32)
    if mean_abs is None:
        mag = x_all.abs().mean(dim=0).clamp(min=1e-8)
    else:
        mag = mean_abs.detach().to(dtype=torch.float32).reshape(-1).clamp(min=1e-8)
        if int(mag.numel()) != int(w.shape[1]):
            raise ValueError("mean_abs must have one entry per input channel")
    search = _subsample(x_all, _SEARCH_ROWS)
    target = search @ w.transpose(0, 1)
    best: QuantizedWeight | None = None
    best_alpha = float(config.awq_alphas[0])
    best_mse = float("inf")
    for alpha in config.awq_alphas:
        a = float(alpha)
        if a == 0.0:
            pre = torch.ones(w.shape[1], dtype=torch.float32)
        else:
            pre = mag.pow(a)
            pre = pre / pre.mean().clamp(min=1e-8)
        qw = quantize_weight(w, config, pre_scale=pre)
        hat = dequantize_weight(qw)
        err = search @ hat.transpose(0, 1) - target
        mse = float(err.square().mean().item())
        if mse < best_mse:
            best_mse = mse
            best = qw
            best_alpha = a
    if best is None:
        raise RuntimeError("AWQ search produced no candidate")
    return best, best_alpha
