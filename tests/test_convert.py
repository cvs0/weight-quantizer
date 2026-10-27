import os
import tempfile

import torch
from torch import nn

from ptqkit.convert import (
    compressed_state,
    matrix_nbytes,
    quantize_model,
    save_compressed,
)
from ptqkit.quant import Method, QuantConfig


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class Conv1D(nn.Module):
    def __init__(self, nf: int, nx: int) -> None:
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class Wrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = Conv1D(8, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_rtn_and_gptq_and_awq_on_tiny_mlp() -> None:
    batches = [torch.randn(4, 16) for _ in range(3)]
    specs = [
        QuantConfig(bits=8, symmetric=True, group_size=None, method=Method.RTN),
        QuantConfig(bits=4, symmetric=True, group_size=8, method=Method.GPTQ),
        QuantConfig(bits=4, symmetric=True, group_size=8, method=Method.AWQ),
    ]
    for config in specs:
        model = Tiny()
        quantize_model(
            model,
            config,
            batches,
            sequential=config.method is not Method.RTN,
        )
        out = model(torch.randn(2, 16))
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()


def test_conv1d_replacement() -> None:
    model = Wrap()
    quantize_model(model, QuantConfig(bits=8, symmetric=False, group_size=None))
    out = model(torch.randn(3, 16))
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_int4_payload_is_smaller_and_roundtrips() -> None:
    layer = nn.Linear(128, 256, bias=False)
    before = int(layer.weight.numel()) * 4
    quantized = quantize_model(layer, QuantConfig(bits=4, symmetric=True, group_size=128))
    assert matrix_nbytes(quantized) < before
    state = compressed_state(quantized)
    assert any(key.endswith("codes") for key in state["tensors"])
    zeros = next(tensor for key, tensor in state["tensors"].items() if key.endswith("zeros"))
    assert zeros.dtype == torch.float16
    handle, path = tempfile.mkstemp(suffix=".pt")
    os.close(handle)
    try:
        nbytes = save_compressed(quantized, path)
        assert nbytes == os.path.getsize(path)
        loaded = torch.load(path, weights_only=False)
    finally:
        os.remove(path)
    assert loaded["version"] == 1
    assert any(key.endswith("codes") for key in loaded["tensors"])
