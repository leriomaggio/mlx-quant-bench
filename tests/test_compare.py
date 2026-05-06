"""Tests for mlx_quant_bench.compare."""

from __future__ import annotations

from mlx_quant_bench.compare import (
    _group_by_model,
    _summary_per_model,
    render_markdown_report,
)


def _fake_records() -> list[dict]:
    """Build a small synthetic results list for testing."""
    return [
        {
            "model_label": "4bit",
            "model_source": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "load_time_s": 5.2,
            "peak_memory_gb": 4.5,
            "prompt_id": "capital_france",
            "category": "factual",
            "output": "The capital of France is Paris.",
            "output_tokens": 8,
            "ttft_s": 0.15,
            "decode_tps": 45.0,
        },
        {
            "model_label": "4bit",
            "model_source": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "load_time_s": 5.2,
            "peak_memory_gb": 4.5,
            "prompt_id": "speed_of_light",
            "category": "factual",
            "output": "About 300,000 km/s.",
            "output_tokens": 6,
            "ttft_s": 0.16,
            "decode_tps": 47.0,
        },
        {
            "model_label": "8bit",
            "model_source": "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
            "load_time_s": 7.0,
            "peak_memory_gb": 9.0,
            "prompt_id": "capital_france",
            "category": "factual",
            "output": "Paris is the capital of France.",
            "output_tokens": 8,
            "ttft_s": 0.20,
            "decode_tps": 30.0,
        },
        {
            "model_label": "8bit",
            "model_source": "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
            "load_time_s": 7.0,
            "peak_memory_gb": 9.0,
            "prompt_id": "speed_of_light",
            "category": "factual",
            "output": "Approximately 299,792 km/s.",
            "output_tokens": 8,
            "ttft_s": 0.21,
            "decode_tps": 31.0,
        },
    ]


class TestSummary:
    def test_groups_by_model(self) -> None:
        groups = _group_by_model(_fake_records())
        assert set(groups.keys()) == {"4bit", "8bit"}
        assert len(groups["4bit"]) == 2

    def test_summary_aggregates_per_model(self) -> None:
        summary = _summary_per_model(_fake_records())
        assert summary["4bit"]["peak_memory_gb"] == 4.5
        assert summary["4bit"]["n_prompts"] == 2
        # avg ttft = (0.15 + 0.16) / 2
        assert summary["4bit"]["avg_ttft_s"] == 0.155
        # avg tps = (45.0 + 47.0) / 2
        assert summary["4bit"]["avg_decode_tps"] == 46.0


class TestMarkdownReport:
    def test_renders_summary_table(self) -> None:
        md = render_markdown_report(_fake_records())
        assert "## Summary" in md
        assert "**4bit**" in md
        assert "**8bit**" in md

    def test_show_outputs_includes_outputs(self) -> None:
        md = render_markdown_report(_fake_records(), show_outputs=True)
        assert "Side-by-side outputs" in md
        assert "The capital of France is Paris." in md
        assert "Paris is the capital of France." in md

    def test_no_outputs_by_default(self) -> None:
        md = render_markdown_report(_fake_records())
        assert "Side-by-side outputs" not in md

    def test_memory_comparison_section(self) -> None:
        md = render_markdown_report(_fake_records())
        assert "Memory comparison" in md
        # 4bit should be 2x smaller than 8bit (4.5 GB vs 9.0 GB)
        assert "2.00×" in md or "2.0×" in md
