import torch

from ptqkit.gptq import gptq_quantize
from ptqkit.quant import Method, QuantConfig, dequantize_weight, quantize_weight


def _mse(weight: torch.Tensor, hat: torch.Tensor, inputs: torch.Tensor) -> float:
    err = inputs @ (hat - weight).transpose(0, 1)
    return float(err.square().mean().item())


def test_gptq_matches_rtn_on_identity_hessian() -> None:
    # H = I and percdamp = 0 does not couple columns. Scale math runs in fp64
    # inside GPTQ and fp32 inside RTN, so a bin edge can move by one code.
    torch.manual_seed(0)
    weight = torch.randn(8, 32)
    inputs = torch.randn(64, 32)
    hessian = torch.eye(32, dtype=torch.float64)
    config = QuantConfig(
        bits=4,
        symmetric=True,
        group_size=16,
        method=Method.GPTQ,
        percdamp=0.0,
    )
    gptq = dequantize_weight(gptq_quantize(weight, hessian, config))
    rtn = dequantize_weight(
        quantize_weight(weight, QuantConfig(bits=4, symmetric=True, group_size=16))
    )
    assert int((gptq - rtn).abs().gt(1e-4).sum()) <= 2
    assert _mse(weight, gptq, inputs) <= _mse(weight, rtn, inputs) * 1.05 + 1e-6


def test_gptq_reduces_output_mse() -> None:
    torch.manual_seed(0)
    out, k, n = 8, 32, 2048
    latent = torch.randn(n, 2) @ torch.randn(2, k)
    latent[:, 0] *= 5
    weight = torch.randn(out, k) * 0.1
    weight[:, 3] = 6
    hessian = latent.double().transpose(0, 1) @ latent.double() / n
    config = QuantConfig(
        bits=4,
        symmetric=True,
        group_size=32,
        method=Method.GPTQ,
        percdamp=0.01,
    )
    gptq = dequantize_weight(gptq_quantize(weight, hessian, config))
    rtn_cfg = QuantConfig(bits=4, symmetric=True, group_size=32)
    rtn = dequantize_weight(quantize_weight(weight, rtn_cfg))
    assert _mse(weight, gptq, latent) < 0.5 * _mse(weight, rtn, latent)
    assert torch.isfinite(gptq).all()


def test_dead_column_is_finite() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(64, 16)
    inputs[:, 3] = 0
    weight = torch.randn(4, 16)
    hessian = inputs.double().transpose(0, 1) @ inputs.double() / inputs.shape[0]
    config = QuantConfig(bits=4, symmetric=False, group_size=None, method=Method.GPTQ)
    hat = dequantize_weight(gptq_quantize(weight, hessian, config))
    assert torch.isfinite(hat).all()
