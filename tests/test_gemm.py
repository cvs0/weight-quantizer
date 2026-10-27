import torch

from ptqkit.gemm import dequant_gemm, kernel_info
from ptqkit.quant import Method, QuantConfig, QuantizedWeight, assemble_quantized, quantize_weight


def test_kernel_info() -> None:
    info = kernel_info()
    assert info["openmp"] is True
    assert isinstance(info["avx2"], bool)


def test_nibble_dot() -> None:
    codes = torch.tensor([[1, 2], [15, 0]], dtype=torch.int32)
    scales = torch.tensor([[1.0], [1.0]])
    zeros = torch.zeros(2, 1)
    qw = assemble_quantized(
        codes,
        scales,
        zeros,
        bits=4,
        group_size=2,
        symmetric=True,
        granularity="channel",
    )
    assert isinstance(qw, QuantizedWeight)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ref = dequant_gemm(x, qw, impl="reference")
    got = dequant_gemm(x, qw, impl="kernel", nthreads=1)
    expect = torch.tensor([[1.0, 15.0], [2.0, 0.0], [3.0, 15.0]])
    assert torch.allclose(ref, expect, atol=1e-6)
    assert torch.allclose(got, expect, atol=1e-6)


def test_kernel_matches_reference() -> None:
    torch.manual_seed(0)
    cases = [
        (4, 8, 1, 8, True, None, False),
        (5, 7, 3, 8, True, 0, False),
        (8, 16, 4, 4, True, 8, False),
        (3, 13, 1, 4, False, 4, False),
        (6, 32, 5, 4, True, 16, True),
        (32, 128, 2, 4, True, 128, False),
        (4, 1, 2, 8, True, None, False),
        (7, 17, 8, 8, False, None, False),
    ]
    generator = torch.Generator().manual_seed(0)
    for out, inn, batch, bits, symmetric, group, use_pre in cases:
        weight = torch.randn(out, inn, generator=generator)
        pre = torch.rand(inn, generator=generator) + 0.25 if use_pre else None
        config = QuantConfig(
            bits=bits,
            symmetric=symmetric,
            group_size=group,
            method=Method.RTN,
        )
        qw = quantize_weight(weight, config, pre_scale=pre)
        x = torch.randn(batch, inn, generator=generator)
        ref = dequant_gemm(x, qw, impl="reference")
        for threads in (1, 2):
            got = dequant_gemm(x, qw, impl="kernel", nthreads=threads)
            assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (out, inn, bits, group, threads)
        a = dequant_gemm(x, qw, impl="kernel", nthreads=1)
        b = dequant_gemm(x, qw, impl="kernel", nthreads=2)
        assert torch.equal(a, b)
