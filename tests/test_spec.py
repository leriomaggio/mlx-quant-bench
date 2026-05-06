"""Tests for mlx_quant_bench.spec."""

from __future__ import annotations

import pytest

from mlx_quant_bench.spec import (
    ModelSpec,
    expand_preset,
    list_presets,
    parse_spec,
)


class TestParseSpec:
    def test_basic_hub_spec(self) -> None:
        s = parse_spec("4bit=mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        assert s.label == "4bit"
        assert s.source == "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        assert not s.is_local

    def test_local_path_spec(self) -> None:
        s = parse_spec("bf16=./models/foo")
        assert s.label == "bf16"
        assert s.source == "./models/foo"
        assert s.is_local

    def test_absolute_local_path(self) -> None:
        s = parse_spec("custom=/tmp/my-model")
        assert s.is_local

    def test_strips_whitespace(self) -> None:
        s = parse_spec("  4bit  =  some/repo  ")
        assert s.label == "4bit"
        assert s.source == "some/repo"

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'label=source'"):
            parse_spec("4bit-some/repo")

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty label"):
            parse_spec("=some/repo")

    def test_empty_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty source"):
            parse_spec("4bit=")

    def test_label_with_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot contain whitespace"):
            parse_spec("4 bit=some/repo")

    def test_source_with_equals_in_url(self) -> None:
        # Source might contain '=' (e.g., URL params); we only split on the first '='
        s = parse_spec("custom=path?with=equals")
        assert s.label == "custom"
        assert s.source == "path?with=equals"


class TestPresets:
    def test_list_presets_nonempty(self) -> None:
        presets = list_presets()
        assert len(presets) > 0
        assert "tinyllama" in presets
        assert "mistral7b-prebuilt" in presets
        assert "mistral7b-full" in presets

    def test_expand_known_preset(self) -> None:
        specs = expand_preset("tinyllama")
        assert len(specs) == 1
        assert isinstance(specs[0], ModelSpec)

    def test_mistral_prebuilt_has_two_models(self) -> None:
        specs = expand_preset("mistral7b-prebuilt")
        assert len(specs) == 2
        labels = {s.label for s in specs}
        assert labels == {"4bit", "8bit"}

    def test_mistral_full_has_four_precisions(self) -> None:
        specs = expand_preset("mistral7b-full")
        assert len(specs) == 4
        labels = {s.label for s in specs}
        assert labels == {"bf16", "8bit", "4bit", "3bit"}

    def test_mistral_full_has_local_paths_for_bf16_and_3bit(self) -> None:
        specs = expand_preset("mistral7b-full")
        by_label = {s.label: s for s in specs}
        # bf16 and 3bit must be locally converted (mlx-community doesn't ship them)
        assert by_label["bf16"].source.startswith("./")
        assert by_label["3bit"].source.startswith("./")
        # 4bit and 8bit are pre-published
        assert by_label["4bit"].source.startswith("mlx-community/")
        assert by_label["8bit"].source.startswith("mlx-community/")

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown preset"):
            expand_preset("does-not-exist")

    def test_expand_returns_fresh_list(self) -> None:
        # Mutating returned list must not affect later calls
        specs_a = expand_preset("tinyllama")
        specs_a.clear()
        specs_b = expand_preset("tinyllama")
        assert len(specs_b) == 1


class TestModelSpec:
    def test_repr_distinguishes_hub_and_local(self) -> None:
        hub = ModelSpec("4bit", "mlx-community/foo")
        local = ModelSpec("bf16", "./models/foo")
        assert "hub" in str(hub)
        assert "local" in str(local)

    def test_frozen_dataclass(self) -> None:
        s = ModelSpec("4bit", "some/repo")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            s.label = "changed"  # type: ignore
