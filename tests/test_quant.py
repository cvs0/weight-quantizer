import torch

from ptqkit.quant import (
    Method,
    QuantConfig,
    absolute_error_bound,
    dequantize_weight,
    element_scales,
    quantize_weight,
)


def _bound(weight: torch.Tensor, config: QuantConfig) -> None:
    quantized = quantize_weight(weight, config)
    hat = dequantize_weight(quantized)
    err = (weight - hat).abs()
    scale = element_scales(quantized)
    limit = absolute_error_bound(scale, symmetric=config.symmetric) + 1e-4
    assert torch.all(err <= limit)
    assert hat.shape == weight.shape
    assert torch.isfinite(hat).all()


def test_rtn_error_bounds() -> None:
    torch.manual_seed(0)
    tensors = [
        torch.randn(17, 32),
        torch.randn(4, 7),
        torch.randn(8, 128),
        torch.linspace(-1.5, 2.0, 128).reshape(4, 32),
        torch.zeros(4, 8),
        torch.full((3, 16), 0.5),
    ]
    for weight in tensors:
        for bits in (4, 8):
            for symmetric in (True, False):
                for group_size in (0, None, 16):
                    config = QuantConfig(
                        bits=bits,
                        symmetric=symmetric,
                        group_size=group_size,
                        method=Method.RTN,
                    )
                    _bound(weight, config)


def test_scale_shapes() -> None:
    weight = torch.randn(6, 40)
    tensor = quantize_weight(weight, QuantConfig(8, True, 0))
    channel = quantize_weight(weight, QuantConfig(8, False, None))
    group = quantize_weight(weight, QuantConfig(4, True, 16))
    assert tensor.scales.shape == (1, 1)
    assert tensor.granularity == "tensor"
    assert tensor.group_size == 40
    assert channel.scales.shape == (6, 1)
    assert channel.granularity == "channel"
    assert group.scales.shape == (6, 3)
    assert group.packed
    assert group.codes.dtype == torch.uint8
    assert group.codes.shape == (6, 20)


def test_pre_scale_matches_manual_rescale() -> None:
    torch.manual_seed(1)
    weight = torch.randn(4, 16)
    pre = torch.rand(16) + 0.2
    config = QuantConfig(bits=4, symmetric=True, group_size=8)
    folded = quantize_weight(weight, config, pre_scale=pre)
    manual = dequantize_weight(quantize_weight(weight * pre, config)) / pre
    assert torch.allclose(dequantize_weight(folded), manual, atol=1e-5, rtol=1e-5)
