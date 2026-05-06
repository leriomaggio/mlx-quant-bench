"""Read benchmark results from JSONL and produce comparison reports.

Two output formats:

- Plain text (printed to stdout): a quick at-a-glance summary table.
- Markdown (optional): a detailed report with per-prompt outputs side-by-side
  for visual quality assessment.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load all JSONL records from a results file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_by_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by model_label, preserving insertion order."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r["model_label"]].append(r)
    return dict(groups)


def _summary_per_model(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-model averages."""
    groups = _group_by_model(records)
    summary = {}
    for label, group in groups.items():
        n = len(group)
        avg_ttft = sum(r["ttft_s"] for r in group) / n
        avg_tps = sum(r["decode_tps"] for r in group) / n
        avg_tokens = sum(r["output_tokens"] for r in group) / n
        summary[label] = {
            "source": group[0]["model_source"],
            "load_time_s": group[0]["load_time_s"],
            "peak_memory_gb": group[0]["peak_memory_gb"],
            "n_prompts": n,
            "avg_ttft_s": round(avg_ttft, 3),
            "avg_decode_tps": round(avg_tps, 2),
            "avg_output_tokens": round(avg_tokens, 1),
        }
    return summary


# -------- Plain text summary ------------------------------------------------

def print_summary(records: list[dict[str, Any]]) -> None:
    """Print a quick summary table to stdout."""
    summary = _summary_per_model(records)

    print()
    print("=" * 78)
    print(f"{'Model':<12} {'Memory':>10} {'Load':>8} {'Avg TTFT':>10} {'Avg tok/s':>10} {'Prompts':>8}")
    print("-" * 78)
    for label, s in summary.items():
        print(
            f"{label:<12} "
            f"{s['peak_memory_gb']:>9.2f}G "
            f"{s['load_time_s']:>7.1f}s "
            f"{s['avg_ttft_s']:>9.2f}s "
            f"{s['avg_decode_tps']:>10.1f} "
            f"{s['n_prompts']:>8d}"
        )
    print("=" * 78)
    print()


# -------- Markdown report ---------------------------------------------------

def render_markdown_report(
    records: list[dict[str, Any]],
    show_outputs: bool = False,
    title: str = "MLX Quantization Benchmark — Results",
) -> str:
    """Render a full markdown report from the records."""
    summary = _summary_per_model(records)
    groups = _group_by_model(records)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Model | Source | Peak memory | Load time | Avg TTFT | Avg tok/s | Prompts |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    for label, s in summary.items():
        lines.append(
            f"| **{label}** | `{s['source']}` "
            f"| {s['peak_memory_gb']:.2f} GB "
            f"| {s['load_time_s']:.1f}s "
            f"| {s['avg_ttft_s']:.2f}s "
            f"| {s['avg_decode_tps']:.1f} "
            f"| {s['n_prompts']} |"
        )
    lines.append("")

    # Memory ratio analysis (if 2+ models)
    if len(summary) >= 2:
        lines.append("## Memory comparison")
        lines.append("")
        # Use the model with the largest peak memory as baseline so all ratios
        # are meaningful "X× smaller than baseline" statements.
        baseline_label = max(summary, key=lambda k: summary[k]["peak_memory_gb"])
        baseline_mem = summary[baseline_label]["peak_memory_gb"]
        lines.append(
            f"Baseline: **{baseline_label}** at {baseline_mem:.2f} GB "
            f"(largest of the compared models)."
        )
        lines.append("")
        for label, s in summary.items():
            if label == baseline_label:
                continue
            mem = s["peak_memory_gb"]
            ratio = baseline_mem / mem if mem > 0 else 0
            lines.append(
                f"- **{label}**: {mem:.2f} GB "
                f"({ratio:.2f}× smaller than {baseline_label})"
            )
        lines.append("")

    # Per-prompt outputs side-by-side
    if show_outputs:
        lines.append("## Side-by-side outputs")
        lines.append("")

        # Get ordered list of prompt IDs (using the first model's order)
        first_model = next(iter(groups))
        prompt_ids = [r["prompt_id"] for r in groups[first_model]]

        # Build a (prompt_id, model_label) -> record lookup
        lookup = {(r["prompt_id"], r["model_label"]): r for r in records}

        for pid in prompt_ids:
            # Get the prompt category from any record matching this id
            category = None
            for r in records:
                if r["prompt_id"] == pid:
                    category = r["category"]
                    break

            lines.append(f"### `{pid}` ({category})")
            lines.append("")
            for label in summary:
                r = lookup.get((pid, label))
                if r is None:
                    continue
                lines.append(f"**{label}** ({r['decode_tps']:.1f} tok/s, {r['output_tokens']} tokens):")
                lines.append("")
                lines.append("```")
                lines.append(r["output"].strip())
                lines.append("```")
                lines.append("")

    return "\n".join(lines)
