"""Command-line entry point: ``mlx-quant-bench``.

Subcommands:

- ``inspect``  Show hardware info and model-fit estimates.
- ``run``      Run benchmarks against one or more models.
- ``report``   Render a markdown report from a JSONL results file.
- ``presets``  List available presets.

Examples::

    mlx-quant-bench inspect
    mlx-quant-bench run --preset tinyllama
    mlx-quant-bench run --preset mistral7b-prebuilt --reset
    mlx-quant-bench run \\
        --model 4bit=mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --model 8bit=mlx-community/Mistral-7B-Instruct-v0.3-8bit
    mlx-quant-bench report results/runs.jsonl --md REPORT.md --show-outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlx_quant_bench import __version__
from mlx_quant_bench.compare import (
    load_results,
    print_summary,
    render_markdown_report,
)
from mlx_quant_bench.spec import ModelSpec, expand_preset, list_presets, parse_spec


DEFAULT_RESULTS = "results/runs.jsonl"


# -------- Subcommand implementations ---------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    # Lazy import: MLX is only needed when actually inspecting hardware
    from mlx_quant_bench.inspect import print_inspection_report
    print_inspection_report()
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    print("Available presets:")
    for name in list_presets():
        specs = expand_preset(name)
        print(f"  {name}")
        for s in specs:
            kind = "local" if s.is_local else "hub"
            print(f"    - {s.label} [{kind}: {s.source}]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Lazy import: MLX is only needed when actually running benchmarks
    from mlx_quant_bench.benchmark import (
        append_results,
        benchmark_model,
        reset_results,
    )

    # Build the spec list from --preset and/or --model
    specs: list[ModelSpec] = []
    if args.preset:
        specs.extend(expand_preset(args.preset))
    if args.model:
        specs.extend(parse_spec(m) for m in args.model)
    if not specs:
        print(
            "ERROR: provide at least one --preset or --model.\n"
            "Run 'mlx-quant-bench presets' to see available presets.",
            file=sys.stderr,
        )
        return 2

    # Optionally reset the results file
    if args.reset:
        reset_results(args.output)
        print(f"Reset results file: {args.output}")

    # Print plan
    print(f"\nWill benchmark {len(specs)} model(s):")
    for s in specs:
        print(f"  - {s}")
    print()

    # Run each model
    for spec in specs:
        try:
            result = benchmark_model(
                spec,
                max_tokens=args.max_tokens,
                verbose=True,
            )
        except Exception as e:
            print(f"\nERROR benchmarking {spec.label}: {e}", file=sys.stderr)
            print("Skipping and continuing with remaining models.", file=sys.stderr)
            continue
        append_results(result, args.output)
        print(f"  → appended {len(result.prompt_results)} records to {args.output}")

    # Quick summary
    if Path(args.output).exists():
        records = load_results(args.output)
        print_summary(records)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    records = load_results(args.input)
    print_summary(records)

    if args.md:
        md = render_markdown_report(
            records,
            show_outputs=args.show_outputs,
            title=args.title,
        )
        Path(args.md).write_text(md)
        print(f"Markdown report written: {args.md}")
    return 0


# -------- Argparse setup ---------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-quant-bench",
        description="Quantization benchmarks for LLMs on Apple Silicon via MLX.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # inspect
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Show hardware info and model-fit estimates.",
    )
    p_inspect.set_defaults(func=cmd_inspect)

    # presets
    p_presets = subparsers.add_parser(
        "presets",
        help="List available benchmark presets.",
    )
    p_presets.set_defaults(func=cmd_presets)

    # run
    p_run = subparsers.add_parser(
        "run",
        help="Run benchmarks against one or more models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run benchmarks against one or more models.\n\n"
            "Provide models via --preset (named bundle) and/or --model (label=source).\n"
            "Both flags can be combined; specs accumulate.\n\n"
            "Examples:\n"
            "  mlx-quant-bench run --preset tinyllama\n"
            "  mlx-quant-bench run --preset mistral7b-prebuilt --reset\n"
            "  mlx-quant-bench run \\\n"
            "      --model 4bit=mlx-community/Mistral-7B-Instruct-v0.3-4bit \\\n"
            "      --model 8bit=mlx-community/Mistral-7B-Instruct-v0.3-8bit"
        ),
    )
    p_run.add_argument(
        "--preset",
        choices=list_presets(),
        help="A named preset that expands to a list of model specs.",
    )
    p_run.add_argument(
        "--model",
        action="append",
        metavar="LABEL=SOURCE",
        help=(
            "A model spec in the form label=source. Source can be a Hugging Face "
            "repo ID or a local path. Repeatable."
        ),
    )
    p_run.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per prompt (default: 256).",
    )
    p_run.add_argument(
        "--output",
        "-o",
        default=DEFAULT_RESULTS,
        help=f"JSONL results file (default: {DEFAULT_RESULTS}).",
    )
    p_run.add_argument(
        "--reset",
        action="store_true",
        help="Delete the results file before running (start fresh).",
    )
    p_run.set_defaults(func=cmd_run)

    # report
    p_report = subparsers.add_parser(
        "report",
        help="Render a comparison report from a JSONL results file.",
    )
    p_report.add_argument(
        "input",
        help="Path to the JSONL results file.",
    )
    p_report.add_argument(
        "--md",
        help="Write a markdown report to this path.",
    )
    p_report.add_argument(
        "--show-outputs",
        action="store_true",
        help="Include per-prompt outputs side-by-side in the markdown report.",
    )
    p_report.add_argument(
        "--title",
        default="MLX Quantization Benchmark — Results",
        help="Title for the markdown report.",
    )
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
