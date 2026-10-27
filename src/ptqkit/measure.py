"""WikiText-2 perplexity, compressed size, and matmul/forward timing.

Perplexity is the fixed-context sliding window:

    PPL = exp(sum_i nll_i / N)

Windows are [b, e) for b = 0, S, 2S, ... and e = min(b + C, T). A window's
labels in the context prefix (positions before the previous e) are set to -100.
The model drops labels[:, 0] and averages the cross-entropy of labels[:, 1:].
N is the sum of those counts. The returned mean loss is multiplied by that
count before the tokens are averaged.

When S < C, each token index in 1..T-1 is a label in labels[:, 1:] of exactly
one window. When S = C the first token of each chunk is labels[:, 0] and is
not scored.

Perplexity forwards use a cached fp32 reconstruction of the quantized weights,
so the number is the weight error, not the integer kernel. Kernel latency is
measured separately. The kernel is checked against the reference dequant GEMM
on one forward; that max-abs logit gap is recorded.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from ptqkit.convert import (
    compressed_state,
    matrix_nbytes,
    payload_nbytes,
    quantize_model,
    save_compressed,
    state_dict_nbytes,
    weight_out_in,
)
from ptqkit.gemm import dequant_gemm, kernel_info
from ptqkit.linear import QuantizedLinear
from ptqkit.quant import Method, QuantConfig, quantize_weight


@dataclass(frozen=True)
class EvalConfig:
    model: str = "gpt2"
    output: str = "results/gpt2_wikitext2.json"
    context: int = 1024
    stride: int = 512
    calib_sequences: int = 64
    calib_seq_len: int = 128
    calib_batch: int = 8
    sequential: bool = True
    max_windows: int = 0
    skip_ppl: bool = False
    skip_bench: bool = False
    methods: str = "all"
    fresh: bool = False


def method_grid() -> list[tuple[str, QuantConfig | None]]:
    rtn = Method.RTN
    return [
        ("fp32", None),
        ("int8_tensor_sym_rtn", QuantConfig(8, True, 0, rtn)),
        ("int8_tensor_asym_rtn", QuantConfig(8, False, 0, rtn)),
        ("int8_channel_sym_rtn", QuantConfig(8, True, None, rtn)),
        ("int8_channel_asym_rtn", QuantConfig(8, False, None, rtn)),
        ("int4_channel_sym_rtn", QuantConfig(4, True, None, rtn)),
        ("int4_channel_asym_rtn", QuantConfig(4, False, None, rtn)),
        ("int4_g128_sym_rtn", QuantConfig(4, True, 128, rtn)),
        ("int4_g128_asym_rtn", QuantConfig(4, False, 128, rtn)),
        ("int4_g128_sym_gptq", QuantConfig(4, True, 128, Method.GPTQ)),
        ("int4_g128_sym_awq", QuantConfig(4, True, 128, Method.AWQ)),
    ]


def hardware_info() -> dict[str, Any]:
    cpu_model = ""
    family = ""
    model_id = ""
    stepping = ""
    flags: list[str] = []
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "model name" and not cpu_model:
                cpu_model = value
            elif key == "cpu family" and not family:
                family = value
            elif key == "model" and not model_id:
                model_id = value
            elif key == "stepping" and not stepping:
                stepping = value
            elif key == "flags" and not flags:
                flags = value.split()
    mem: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                mem[parts[0].rstrip(":")] = int(parts[1])
    blas_line = ""
    config_text = torch.__config__.show()
    for line in config_text.splitlines():
        lower = line.lower()
        if "blas" in lower or "mkl" in lower or "onednn" in lower or "openblas" in lower:
            blas_line = line.strip()
            break
    if not blas_line and config_text:
        blas_line = config_text.splitlines()[0].strip()
    return {
        "cpu_model": cpu_model,
        "cpu_family": family,
        "cpu_model_id": model_id,
        "cpu_stepping": stepping,
        "logical_cpus": os.cpu_count(),
        "isa": [flag for flag in ("avx2", "fma", "avx512f") if flag in flags],
        "mem_total_kib": mem.get("MemTotal"),
        "mem_available_kib": mem.get("MemAvailable"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "blas": blas_line,
        "torch_config": config_text,
    }


def _configure_threads() -> int:
    threads = os.cpu_count() or 1
    torch.set_num_threads(threads)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    return threads


def _median_ms(fn: Callable[[], object], *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def bench_matmul(threads: int) -> list[dict[str, Any]]:
    """Median latency of fp32 GEMM vs the int8 and int4 kernels.

    Shapes are the GPT-2 small Conv1D maps: attention QKV, attention out,
    MLP up, MLP down. GFLOP/s counts 2*B*M*K matmul flops and ignores dequant.
    """
    shapes = (
        ("attn_qkv", 2304, 768),
        ("attn_out", 768, 768),
        ("mlp_up", 3072, 768),
        ("mlp_down", 768, 3072),
    )
    batches = (1, 16, 128)
    int8 = QuantConfig(bits=8, symmetric=True, group_size=None, method=Method.RTN)
    int4 = QuantConfig(bits=4, symmetric=True, group_size=128, method=Method.RTN)
    rows: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(0)
    for name, out_features, in_features in shapes:
        weight = torch.randn(out_features, in_features, generator=generator)
        qw8 = quantize_weight(weight, int8)
        qw4 = quantize_weight(weight, int4)
        for batch in batches:
            activations = torch.randn(batch, in_features, generator=generator)
            iters = 20 if batch == 1 else 8
            warmup = 4 if batch == 1 else 2

            def run_fp32(
                act: torch.Tensor = activations,
                mat: torch.Tensor = weight,
            ) -> torch.Tensor:
                return nn.functional.linear(act, mat)

            def run_int8(
                act: torch.Tensor = activations,
                quantized: Any = qw8,
            ) -> torch.Tensor:
                return dequant_gemm(act, quantized, impl="kernel", nthreads=threads)

            def run_int4(
                act: torch.Tensor = activations,
                quantized: Any = qw4,
            ) -> torch.Tensor:
                return dequant_gemm(act, quantized, impl="kernel", nthreads=threads)

            fp32_ms = _median_ms(run_fp32, iters=iters, warmup=warmup)
            int8_ms = _median_ms(run_int8, iters=iters, warmup=warmup)
            int4_ms = _median_ms(run_int4, iters=iters, warmup=warmup)
            flops = 2.0 * batch * out_features * in_features
            for kind, elapsed in (("fp32", fp32_ms), ("int8", int8_ms), ("int4", int4_ms)):
                rows.append(
                    {
                        "layer": name,
                        "out": out_features,
                        "in": in_features,
                        "batch": batch,
                        "kind": kind,
                        "median_ms": elapsed,
                        "gflops": flops / (elapsed / 1e3) / 1e9,
                        "iters": iters,
                    }
                )
    return rows


@torch.inference_mode()
def perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    context: int,
    stride: int,
    max_windows: int = 0,
) -> dict[str, float]:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("input_ids must have shape [1, T]")
    if context < 2 or stride < 1:
        raise ValueError("context must be >= 2 and stride must be >= 1")
    length = int(input_ids.shape[1])
    nll_sum = 0.0
    n_tokens = 0
    windows = 0
    prev = 0
    for begin in range(0, length, stride):
        end = min(begin + context, length)
        target_len = end - prev
        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-target_len] = -100
        outputs = model(chunk, labels=labels)
        loss = outputs.loss
        # labels[:, 0] is dropped by the causal shift. Count only what the loss averages.
        n_loss = int((labels[:, 1:] != -100).sum().item())
        if n_loss > 0 and loss is not None:
            nll_sum += float(loss.item()) * n_loss
            n_tokens += n_loss
        windows += 1
        prev = end
        if end == length or (max_windows > 0 and windows >= max_windows):
            break
    if n_tokens == 0:
        raise RuntimeError("perplexity accumulated no tokens")
    mean_nll = nll_sum / n_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "nll": mean_nll,
        "tokens": float(n_tokens),
        "windows": float(windows),
    }


def _fp32_matrix_bytes(model: nn.Module) -> int:
    from ptqkit.convert import iter_targets

    total = 0
    for _name, module in iter_targets(model):
        total += int(weight_out_in(module).numel()) * 4
    return total


def _file_bytes_fp32(model: nn.Module) -> int:
    handle, path = tempfile.mkstemp(suffix=".pt")
    os.close(handle)
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path)
    finally:
        os.remove(path)


def _file_bytes_quantized(model: nn.Module) -> int:
    handle, path = tempfile.mkstemp(suffix=".pt")
    os.close(handle)
    try:
        return save_compressed(model, path)
    finally:
        os.remove(path)


def _set_impl(model: nn.Module, impl: str) -> None:
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.impl = impl
            module.clear_cache()


def _forward_ms(model: nn.Module, seq: int, vocab: int, *, iters: int) -> float:
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, vocab, (1, seq), generator=generator)

    def run() -> None:
        model(ids)

    return _median_ms(run, iters=iters, warmup=2)


def _protocol(cfg: EvalConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "join": "\n\n",
        "calibration_split": "train",
        "eval_split": "test",
        "context": cfg.context,
        "stride": cfg.stride,
        "calib_sequences": cfg.calib_sequences,
        "calib_seq_len": cfg.calib_seq_len,
        "calib_batch": cfg.calib_batch,
        "activation_order": "sequential" if cfg.sequential else "fp32",
        "max_windows": cfg.max_windows,
        "ppl_forward": "cached fp32 reconstruction of quantized weights",
        "disk_scales": "fp16",
    }


def _write(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def _select(cfg: EvalConfig) -> list[tuple[str, QuantConfig | None]]:
    grid = method_grid()
    if cfg.methods == "all":
        return grid
    wanted = {part.strip() for part in cfg.methods.split(",") if part.strip()}
    chosen = [item for item in grid if item[0] in wanted]
    missing = wanted.difference(name for name, _ in chosen)
    if missing:
        raise ValueError(f"unknown methods: {sorted(missing)}")
    return chosen


def _meta(name: str, config: QuantConfig | None) -> dict[str, Any]:
    if config is None:
        return {
            "name": name,
            "bits": 32,
            "symmetric": None,
            "group_size": None,
            "granularity": None,
            "method": "fp32",
        }
    from ptqkit.quant import granularity_of

    return {
        "name": name,
        "bits": config.bits,
        "symmetric": config.symmetric,
        "group_size": config.group_size,
        "granularity": granularity_of(config),
        "method": config.method.value,
    }


def _load_split_ids(model_name: str, split: str) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("tokenizer did not return a tensor")
    return input_ids


def _calibration_batches(flat: torch.Tensor, cfg: EvalConfig) -> list[torch.Tensor]:
    usable = int(flat.numel()) // cfg.calib_seq_len
    count = min(cfg.calib_sequences, usable)
    if count < 1:
        raise RuntimeError("not enough calibration tokens")
    seqs = flat[: count * cfg.calib_seq_len].view(count, cfg.calib_seq_len)
    return [seqs[i : i + cfg.calib_batch].contiguous() for i in range(0, count, cfg.calib_batch)]


def _record_size(model: nn.Module, quantized: bool) -> dict[str, int]:
    if quantized:
        state = compressed_state(model)
        return {
            "payload_bytes": payload_nbytes(state),
            "file_bytes": _file_bytes_quantized(model),
            "matrix_bytes": matrix_nbytes(model),
        }
    return {
        "payload_bytes": state_dict_nbytes(model),
        "file_bytes": _file_bytes_fp32(model),
        "matrix_bytes": _fp32_matrix_bytes(model),
    }


def run_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    threads = _configure_threads()
    torch.manual_seed(0)
    protocol = _protocol(cfg)
    document: dict[str, Any] = {
        "complete": False,
        "protocol": protocol,
        "hardware": hardware_info(),
        "kernel": kernel_info(),
        "methods": [],
        "matmul": [],
        "forward": [],
        "kernel_logit_max_abs": {},
    }
    if not cfg.fresh and os.path.exists(cfg.output):
        with open(cfg.output, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("protocol") == protocol:
            document["methods"] = list(previous.get("methods", []))
    done = {
        str(row["name"])
        for row in document["methods"]
        if "perplexity" in row and "error" not in row
    }
    if not cfg.skip_bench:
        print("matmul bench", flush=True)
        document["matmul"] = bench_matmul(threads)
        document["hardware"]["torch_num_threads"] = torch.get_num_threads()
        _write(cfg.output, document)

    if cfg.skip_ppl and cfg.skip_bench:
        document["complete"] = True
        _write(cfg.output, document)
        return document

    from transformers import AutoModelForCausalLM

    print("loading model and wikitext-2", flush=True)
    base = cast(nn.Module, AutoModelForCausalLM.from_pretrained(cfg.model))
    base.eval()
    base.to("cpu")
    model_config = getattr(base, "config", None)
    if model_config is None:
        raise RuntimeError("model has no config")
    if getattr(model_config, "use_cache", None) is not None:
        model_config.use_cache = False
    test_ids = _load_split_ids(cfg.model, "test")
    train_ids = _load_split_ids(cfg.model, "train")
    batches = _calibration_batches(train_ids.view(-1), cfg)
    vocab = int(model_config.vocab_size)
    selected = _select(cfg)

    if not cfg.skip_ppl:
        for name, config in selected:
            if name in done:
                print(f"skip {name}", flush=True)
                continue
            print(f"eval {name}", flush=True)
            row = _meta(name, config)
            start = time.perf_counter()
            try:
                if config is None:
                    model: nn.Module = base
                    quantized = False
                    cache = False
                else:
                    import copy

                    model = copy.deepcopy(base)
                    quantize_model(
                        model,
                        config,
                        batches if config.method is not Method.RTN else None,
                        sequential=cfg.sequential,
                        kernel=False,
                        cache_weights=True,
                    )
                    quantized = True
                    cache = True
                    alphas = [
                        module.awq_alpha
                        for module in model.modules()
                        if isinstance(module, QuantizedLinear) and module.awq_alpha is not None
                    ]
                    if alphas:
                        row["awq_alpha_mean"] = float(sum(alphas) / len(alphas))
                stats = perplexity(
                    model,
                    test_ids,
                    context=cfg.context,
                    stride=cfg.stride,
                    max_windows=cfg.max_windows,
                )
                row.update(stats)
                row.update(_record_size(model, quantized))
                row["seconds"] = time.perf_counter() - start
                row["cached_fp32"] = cache or config is None
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["seconds"] = time.perf_counter() - start
                document["methods"] = [item for item in document["methods"] if item["name"] != name]
                document["methods"].append(row)
                _write(cfg.output, document)
                raise
            document["methods"] = [item for item in document["methods"] if item["name"] != name]
            document["methods"].append(row)
            _write(cfg.output, document)
            print(
                f"{name} ppl={row['perplexity']:.4f} file_bytes={row['file_bytes']} "
                f"({row['seconds']:.1f}s)",
                flush=True,
            )
            if config is not None:
                del model

    if not cfg.skip_bench:
        import copy

        probe = torch.randint(0, vocab, (1, 8))
        for label, config in (
            ("int8_channel_sym_rtn", QuantConfig(8, True, None, Method.RTN)),
            ("int4_g128_sym_rtn", QuantConfig(4, True, 128, Method.RTN)),
        ):
            cloned = copy.deepcopy(base)
            quantize_model(cloned, config, kernel=True, cache_weights=False)
            _set_impl(cloned, "reference")
            ref = cloned(probe).logits
            _set_impl(cloned, "kernel")
            got = cloned(probe).logits
            gap = float((ref - got).abs().max().item())
            gaps = document["kernel_logit_max_abs"]
            if not isinstance(gaps, dict):
                gaps = {}
                document["kernel_logit_max_abs"] = gaps
            gaps[label] = gap
            for seq, iters in ((1, 30), (128, 10)):
                _set_impl(cloned, "kernel")
                elapsed = _forward_ms(cloned, seq, vocab, iters=iters)
                document["forward"].append(
                    {
                        "model": label,
                        "seq": seq,
                        "median_ms": elapsed,
                        "iters": iters,
                        "impl": "kernel",
                    }
                )
            del cloned
        for seq, iters in ((1, 30), (128, 10)):
            elapsed = _forward_ms(base, seq, vocab, iters=iters)
            document["forward"].append(
                {"model": "fp32", "seq": seq, "median_ms": elapsed, "iters": iters, "impl": "fp32"}
            )
        _write(cfg.output, document)

    document["complete"] = True
    _write(cfg.output, document)
    return document


def calibration_batches_for_test(rows: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Test helper so the batching contract stays importable without datasets."""
    return list(rows)
