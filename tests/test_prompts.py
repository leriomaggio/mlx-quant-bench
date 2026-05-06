"""Tests for mlx_quant_bench.prompts."""

from __future__ import annotations

from mlx_quant_bench.prompts import PROMPTS, by_category, categories


class TestPrompts:
    def test_all_prompts_have_unique_ids(self) -> None:
        ids = [p.id for p in PROMPTS]
        assert len(ids) == len(set(ids))

    def test_all_prompts_have_nonempty_text(self) -> None:
        for p in PROMPTS:
            assert p.text.strip()

    def test_all_prompts_have_known_category(self) -> None:
        valid = {"factual", "reasoning", "instruction"}
        for p in PROMPTS:
            assert p.category in valid

    def test_each_category_has_multiple_prompts(self) -> None:
        # We want at least 2 prompts per category to make precision-induced
        # quality variation visible (a single prompt can be misleading).
        for cat in categories():
            assert len(by_category(cat)) >= 2

    def test_by_category_returns_only_matching(self) -> None:
        for cat in categories():
            for p in by_category(cat):
                assert p.category == cat
