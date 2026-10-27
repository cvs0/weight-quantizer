"""Command line entry for the WikiText-2 measurement run."""

from __future__ import annotations

import argparse
import json

from ptqkit.measure import EvalConfig, run_evaluation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ptqkit",
        description="Quantize linear weights and measure WikiText-2 perplexity, size, and speed.",
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--output", default="results/gpt2_wikitext2.json")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--calib-sequences", type=int, default=64)
    parser.add_argument("--calib-seq-len", type=int, default=128)
    parser.add_argument("--calib-batch", type=int, default=8)
    parser.add_argument("--activation-order", choices=("sequential", "fp32"), default="sequential")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--plot", default="")
    args = parser.parse_args(argv)
    cfg = EvalConfig(
        model=args.model,
        output=args.output,
        context=args.context,
        stride=args.stride,
        calib_sequences=args.calib_sequences,
        calib_seq_len=args.calib_seq_len,
        calib_batch=args.calib_batch,
        sequential=args.activation_order == "sequential",
        max_windows=args.max_windows,
        skip_ppl=args.skip_ppl,
        skip_bench=args.skip_bench,
        methods=args.methods,
        fresh=args.fresh,
    )
    run_evaluation(cfg)
    plot_path = args.plot
    if not plot_path and not args.skip_ppl:
        if args.output.endswith(".json"):
            plot_path = args.output[: -len(".json")] + ".png"
        else:
            plot_path = args.output + ".png"
    if plot_path and not args.skip_ppl:
        from ptqkit.plot import plot_results, render_markdown

        plot_results(args.output, plot_path)
        with open(args.output, encoding="utf-8") as handle:
            print(render_markdown(json.load(handle)))
