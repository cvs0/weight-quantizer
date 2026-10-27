import torch

from ptqkit.awq import awq_quantize
from ptqkit.quant import Method, QuantConfig, dequantize_weight, quantize_weight


def _mse(weight: torch.Tensor, hat: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return (inputs @ (hat - weight).transpose(0, 1)).square().mean()


def test_alpha_zero_is_rtn() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 32)
    inputs = torch.randn(40, 32)
    config = QuantConfig(
        bits=4,
        symmetric=True,
        group_size=16,
        method=Method.AWQ,
        awq_alphas=(0.0,),
    )
    quantized, alpha = awq_quantize(weight, inputs, config)
    rtn = quantize_weight(weight, QuantConfig(bits=4, symmetric=True, group_size=16))
    assert alpha == 0.0
    assert torch.allclose(dequantize_weight(quantized), dequantize_weight(rtn), atol=1e-5)


def test_search_is_not_worse_than_rtn_on_search_rows() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 32) * 0.1
    weight[:, 1] = 3.0
    inputs = torch.randn(48, 32)
    inputs[:, 0] *= 20
    config = QuantConfig(bits=4, symmetric=True, group_size=16, method=Method.AWQ)
    quantized, _alpha = awq_quantize(weight, inputs, config)
    rtn = quantize_weight(weight, QuantConfig(bits=4, symmetric=True, group_size=16))
    awq_mse = _mse(weight, dequantize_weight(quantized), inputs)
    rtn_mse = _mse(weight, dequantize_weight(rtn), inputs)
    assert float(awq_mse) <= float(rtn_mse) + 1e-6
