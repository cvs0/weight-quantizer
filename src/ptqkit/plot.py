"""Plot measured perplexity and compressed size. Numbers come from the json."""

from __future__ import annotations

import json
from typing import Any


def load_results(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _ordered_methods(data: dict[str, Any]) -> list[dict[str, Any]]:
    from ptqkit.measure import method_grid

    order = [name for name, _ in method_grid()]
    rows = [row for row in data.get("methods", []) if "perplexity" in row and "error" not in row]
    rank = {name: i for i, name in enumerate(order)}
    rows.sort(key=lambda row: rank.get(str(row["name"]), 10_000))
    return rows


def render_markdown(data: dict[str, Any]) -> str:
    hw = data.get("hardware", {})
    protocol = data.get("protocol", {})
    lines = [
        f"CPU: {hw.get('cpu_model')} family {hw.get('cpu_family')} "
        f"model {hw.get('cpu_model_id')} stepping {hw.get('cpu_stepping')}, "
        f"{hw.get('logical_cpus')} threads, ISA {hw.get('isa')}.",
        f"MemTotal {hw.get('mem_total_kib')} KiB, MemAvailable {hw.get('mem_available_kib')} KiB "
        f"at measurement start.",
        f"Python {hw.get('python')}, torch {hw.get('torch')}, BLAS `{hw.get('blas')}`.",
        f"Kernel build: {data.get('kernel')}.",
        f"Protocol: model `{protocol.get('model')}`, "
        f"{protocol.get('dataset')} {protocol.get('subset')} test, "
        f"context {protocol.get('context')}, stride {protocol.get('stride')}, "
        f"calibration {protocol.get('calib_sequences')} x {protocol.get('calib_seq_len')} "
        f"from train, activation order `{protocol.get('activation_order')}`.",
    ]
    rows = _ordered_methods(data)
    token_counts = {int(row["tokens"]) for row in rows if "tokens" in row}
    window_counts = {int(row["windows"]) for row in rows if "windows" in row}
    if len(token_counts) == 1 and len(window_counts) == 1:
        lines.append(
            f"Scored tokens: {next(iter(token_counts))}. Windows: {next(iter(window_counts))}."
        )
    lines.extend(
        [
            "",
            "| method | bits | sym | group | ppl | payload bytes | file bytes | matrix bytes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        sym = row.get("symmetric")
        sym_s = "" if sym is None else str(bool(sym)).lower()
        group = row.get("group_size")
        group_s = "" if group is None else str(group)
        lines.append(
            f"| {row['name']} | {row.get('bits')} | {sym_s} | {group_s} | "
            f"{float(row['perplexity']):.4f} | {int(row['payload_bytes'])} | "
            f"{int(row['file_bytes'])} | {int(row['matrix_bytes'])} |"
        )
    lines.extend(
        [
            "",
            "Matmul median latency. GFLOP/s = 2*B*M*K / time, dequant not counted.",
            "",
            "| layer | M | K | B | kind | median ms | GFLOP/s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in data.get("matmul", []):
        lines.append(
            f"| {row['layer']} | {row['out']} | {row['in']} | {row['batch']} | {row['kind']} | "
            f"{float(row['median_ms']):.4f} | {float(row['gflops']):.2f} |"
        )
    lines.extend(
        [
            "",
            "| model | seq | impl | median ms |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in data.get("forward", []):
        lines.append(
            f"| {row['model']} | {row['seq']} | {row['impl']} | {float(row['median_ms']):.4f} |"
        )
    gaps = data.get("kernel_logit_max_abs") or {}
    if isinstance(gaps, dict) and gaps:
        lines.extend(["", "Max abs logit gap, kernel vs reference dequant, shape [1, 8]:"])
        for name, value in gaps.items():
            lines.append(f"- {name}: {float(value):.6e}")
    lines.append("")
    return "\n".join(lines)


def plot_results(json_path: str, png_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_results(json_path)
    rows = _ordered_methods(data)
    if not rows:
        raise RuntimeError("no perplexity rows to plot")
    names = [str(row["name"]) for row in rows]
    ppl = [float(row["perplexity"]) for row in rows]
    mib = [int(row["file_bytes"]) / (1024 * 1024) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].bar(range(len(names)), ppl, color="#222222")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("perplexity, log scale")
    axes[1].bar(range(len(names)), mib, color="#222222")
    axes[1].set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    axes[1].set_ylabel("torch.save file MiB")
    hw = data.get("hardware", {})
    protocol = data.get("protocol", {})
    caption = (
        f"{hw.get('cpu_model')} x{hw.get('logical_cpus')}  "
        f"{protocol.get('model')}  context {protocol.get('context')} "
        f"stride {protocol.get('stride')}"
    )
    fig.text(0.01, 0.01, caption, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
