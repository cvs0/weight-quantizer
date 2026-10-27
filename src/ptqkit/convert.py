"""Replace nn.Linear and GPT-2 Conv1D modules with QuantizedLinear.

Conv1D stores weight as [in, out] and computes x @ W. The quantizer stores
[out, in] for both layouts, so the forward is always x @ W_hat^T + bias.

`sequential=False` (fp32 activation order) runs the calibration set once and
builds every Hessian from those inputs. `sequential=True` quantizes modules in
`named_modules` order and reruns calibration after each replacement, so later
Hessians see already-quantized earlier layers. During that rerun the quantized
layers use a cached fp32 reconstruction, which matches the integer grid and
keeps the forwards in the fp32 GEMM.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ptqkit.awq import awq_quantize
from ptqkit.gptq import gptq_quantize
from ptqkit.linear import QuantizedLinear
from ptqkit.quant import Method, QuantConfig, QuantizedWeight, quantize_weight

Batch = torch.Tensor | dict[str, torch.Tensor]
_SAMPLE_ROWS = 2048


class ActivationStats:
    def __init__(self) -> None:
        self.xtx: torch.Tensor | None = None
        self.abs_sum: torch.Tensor | None = None
        self.count: int = 0
        self.sample: torch.Tensor | None = None

    def update(self, hidden: torch.Tensor) -> None:
        x = hidden.detach().reshape(-1, hidden.shape[-1]).to(dtype=torch.float32, device="cpu")
        x64 = x.double()
        gram = x64.transpose(0, 1) @ x64
        self.xtx = gram if self.xtx is None else self.xtx + gram
        absolute = x64.abs().sum(dim=0)
        self.abs_sum = absolute if self.abs_sum is None else self.abs_sum + absolute
        self.count += int(x.shape[0])
        if self.sample is None:
            self.sample = x[:_SAMPLE_ROWS].clone()
        elif int(self.sample.shape[0]) < _SAMPLE_ROWS:
            need = _SAMPLE_ROWS - int(self.sample.shape[0])
            self.sample = torch.cat((self.sample, x[:need].clone()), dim=0)

    def hessian(self) -> torch.Tensor:
        if self.xtx is None or self.count == 0:
            raise RuntimeError("no calibration rows")
        return self.xtx / self.count

    def mean_abs(self) -> torch.Tensor:
        if self.abs_sum is None or self.count == 0:
            raise RuntimeError("no calibration rows")
        return (self.abs_sum / self.count).to(dtype=torch.float32)

    def rows(self) -> torch.Tensor:
        if self.sample is None:
            raise RuntimeError("no calibration rows")
        return self.sample


def _is_conv1d(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")


def iter_targets(model: nn.Module) -> list[tuple[str, nn.Module]]:
    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            continue
        if isinstance(module, nn.Linear) or _is_conv1d(module):
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                found.append((name, module))
    return found


def weight_out_in(module: nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"unsupported module {module.__class__.__name__}")
    if isinstance(module, nn.Linear):
        return weight.detach()
    if _is_conv1d(module):
        return weight.detach().transpose(0, 1).contiguous()
    raise TypeError(f"unsupported module {module.__class__.__name__}")


def _bias_of(module: nn.Module) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    if not isinstance(bias, torch.Tensor):
        return None
    detached: torch.Tensor = bias.detach()
    return detached


def _replace(root: nn.Module, name: str, new: nn.Module) -> None:
    parent: nn.Module = root
    parts = name.split(".")
    for part in parts[:-1]:
        child = getattr(parent, part)
        if not isinstance(child, nn.Module):
            raise TypeError(f"{part} is not a module")
        parent = child
    setattr(parent, parts[-1], new)


def _forward(model: nn.Module, batch: Batch) -> None:
    if isinstance(batch, dict):
        model(**batch)
    else:
        model(batch)


def _check_cpu(model: nn.Module) -> None:
    for param in model.parameters():
        if param.device.type != "cpu":
            raise RuntimeError("quantize_model expects CPU parameters")


def _prepare(model: nn.Module) -> None:
    model.eval()
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "use_cache", None) is not None:
        config.use_cache = False


def _as_batches(batches: Sequence[Batch]) -> list[Batch]:
    if len(batches) == 0:
        raise ValueError("calibration_batches is empty")
    return list(batches)


def _capture_all(
    model: nn.Module,
    names: list[tuple[str, nn.Module]],
    batches: Sequence[Batch],
) -> dict[str, ActivationStats]:
    stats = {name: ActivationStats() for name, _ in names}
    handles = []
    for name, module in names:

        def hook(mod: nn.Module, args: tuple[Any, ...], *, name: str = name) -> None:
            del mod
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError(f"{name} did not receive a tensor input")
            stats[name].update(args[0])

        handles.append(module.register_forward_pre_hook(hook))
    try:
        for batch in batches:
            _forward(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return stats


def _capture_one(
    model: nn.Module,
    name: str,
    module: nn.Module,
    batches: Sequence[Batch],
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []

    def hook(mod: nn.Module, args: tuple[Any, ...]) -> None:
        del mod
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError(f"{name} did not receive a tensor input")
        hidden = args[0].detach().reshape(-1, args[0].shape[-1])
        chunks.append(hidden.to(dtype=torch.float32, device="cpu"))

    handle = module.register_forward_pre_hook(hook)
    try:
        for batch in batches:
            _forward(model, batch)
    finally:
        handle.remove()
    if not chunks:
        raise RuntimeError(f"no activations captured for {name}")
    return torch.cat(chunks, dim=0)


def _module_from(
    qw: QuantizedWeight,
    source: nn.Module,
    *,
    kernel: bool,
    alpha: float | None,
) -> QuantizedLinear:
    layer = QuantizedLinear(qw, _bias_of(source), impl="kernel" if kernel else "reference")
    layer.awq_alpha = alpha
    return layer


def _quantize_one(
    weight: torch.Tensor,
    config: QuantConfig,
    stats: ActivationStats | None,
    rows: torch.Tensor | None,
) -> tuple[QuantizedWeight, float | None]:
    if config.method is Method.RTN:
        return quantize_weight(weight, config), None
    if config.method is Method.GPTQ:
        if stats is None and rows is None:
            raise ValueError("GPTQ requires calibration activations")
        if rows is not None and stats is None:
            hessian = rows.double().transpose(0, 1) @ rows.double() / float(rows.shape[0])
        else:
            assert stats is not None
            hessian = stats.hessian()
        return gptq_quantize(weight, hessian, config), None
    if config.method is Method.AWQ:
        if rows is None and stats is None:
            raise ValueError("AWQ requires calibration activations")
        if stats is not None and rows is None:
            sample = stats.rows()
            mean_abs = stats.mean_abs()
        else:
            assert rows is not None
            sample = rows[:_SAMPLE_ROWS]
            mean_abs = rows.abs().mean(dim=0)
        qw, alpha = awq_quantize(weight, sample, config, mean_abs=mean_abs)
        return qw, alpha
    raise ValueError(f"unknown method {config.method}")


def quantize_model(
    model: nn.Module,
    config: QuantConfig,
    calibration_batches: Sequence[Batch] | None = None,
    *,
    sequential: bool = False,
    kernel: bool = False,
    cache_weights: bool = False,
) -> nn.Module:
    """Quantize every Linear and Conv1D in `model` in place."""
    _check_cpu(model)
    _prepare(model)
    targets = iter_targets(model)
    if config.method is not Method.RTN and calibration_batches is None:
        raise ValueError(f"{config.method.value} requires calibration_batches")
    batches = [] if calibration_batches is None else _as_batches(calibration_batches)

    with torch.inference_mode():
        if len(targets) == 1 and targets[0][0] == "":
            module = targets[0][1]
            stats = None
            rows = None
            if config.method is not Method.RTN:
                if sequential:
                    rows = _capture_one(model, "root", module, batches)
                else:
                    stats = _capture_all(model, [("root", module)], batches)["root"]
            qw, alpha = _quantize_one(weight_out_in(module), config, stats, rows)
            layer = _module_from(qw, module, kernel=kernel, alpha=alpha)
            if cache_weights:
                layer.materialize()
            return layer
        if config.method is Method.RTN or not sequential:
            stats_map: dict[str, ActivationStats] | None = None
            if config.method is not Method.RTN:
                stats_map = _capture_all(model, targets, batches)
            for name, module in targets:
                stats = None if stats_map is None else stats_map[name]
                qw, alpha = _quantize_one(weight_out_in(module), config, stats, None)
                _replace(model, name, _module_from(qw, module, kernel=kernel, alpha=alpha))
        else:
            for name, module in targets:
                rows = _capture_one(model, name, module, batches)
                qw, alpha = _quantize_one(weight_out_in(module), config, None, rows)
                layer = _module_from(qw, module, kernel=kernel, alpha=alpha)
                # Cached reconstruction keeps later calibration passes on the fp32 GEMM.
                layer.materialize()
                _replace(model, name, layer)

        if cache_weights:
            for module in model.modules():
                if isinstance(module, QuantizedLinear):
                    module.materialize()
        else:
            for module in model.modules():
                if isinstance(module, QuantizedLinear):
                    module.clear_cache()
    return model


def compressed_state(model: nn.Module) -> dict[str, Any]:
    """Integer codes, fp16 scales/zeros, fp32 biases, and untouched parameters.

    fp16 holds every integer zero in [0, 255] exactly. Scales are stored in fp16
    only in this payload. Runtime dequantization uses the fp32 buffers.
    """
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, dict[str, Any]] = {}
    quantized_buffers: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        meta[name] = {
            "bits": module.bits,
            "group_size": module.group_size,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "symmetric": module.symmetric,
            "packed": module.packed,
            "granularity": module.granularity,
            "awq_alpha": module.awq_alpha,
        }
        prefix = f"{name}." if name else ""
        tensors[prefix + "codes"] = module.codes.detach().cpu()
        tensors[prefix + "scales"] = module.scales.detach().to(dtype=torch.float16).cpu()
        tensors[prefix + "zeros"] = module.zeros.detach().to(dtype=torch.float16).cpu()
        if module.channel_scale is not None:
            channel = module.channel_scale.detach().to(dtype=torch.float16).cpu()
            tensors[prefix + "channel_scale"] = channel
        if module.bias is not None:
            tensors[prefix + "bias"] = module.bias.detach().to(dtype=torch.float32).cpu()
        for buffer_name, _buffer in module.named_buffers(recurse=False):
            key = f"{name}.{buffer_name}" if name else buffer_name
            quantized_buffers.add(key)
    for param_name, param in model.named_parameters():
        tensors["param." + param_name] = param.detach().cpu()
    for buffer_name, buffer in model.named_buffers():
        if buffer_name in quantized_buffers:
            continue
        tensors["buffer." + buffer_name] = buffer.detach().cpu()
    return {"version": 1, "meta": meta, "tensors": tensors}


def payload_nbytes(state: dict[str, Any]) -> int:
    total = 0
    tensors = state["tensors"]
    for tensor in tensors.values():
        total += int(tensor.nbytes)
    return total


def matrix_nbytes(model: nn.Module) -> int:
    """Packed codes plus fp16 scales, zeros, and channel scales. No biases."""
    total = 0
    for module in model.modules():
        if not isinstance(module, QuantizedLinear):
            continue
        total += int(module.codes.nbytes)
        total += int(module.scales.numel()) * 2
        total += int(module.zeros.numel()) * 2
        if module.channel_scale is not None:
            total += int(module.channel_scale.numel()) * 2
    return total


def save_compressed(model: nn.Module, path: str) -> int:
    state = compressed_state(model)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(state, path)
    return os.path.getsize(path)


def state_dict_nbytes(model: nn.Module) -> int:
    total = 0
    for tensor in model.state_dict().values():
        total += int(tensor.nbytes)
    return total
