"""Model specification: pairing a precision label with a model location.

A *spec* is a (label, source) pair where:

- label is a free-form precision tag for reporting ('bf16', '4bit', '8bit', etc.)
- source is either a Hugging Face repo ID or a filesystem path

This decoupling exists because the `mlx-community` org doesn't follow a uniform
naming pattern across precision variants. Some models have `-4bit` / `-8bit`
suffixes (Mistral 7B Instruct v0.3); some have a single `-mlx` repo only
(TinyLlama); bf16 / 3-bit variants for many models are not pre-published at
all and must be locally converted with ``mlx_lm.convert``.

Specs from the CLI look like ``label=source``::

    --model bf16=./models/Mistral-7B-Instruct-v0.3-bf16
    --model 4bit=mlx-community/Mistral-7B-Instruct-v0.3-4bit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """One model to benchmark.

    Attributes:
        label: Precision label for reporting (e.g. ``'bf16'``, ``'4bit'``).
            Must be a non-empty string with no whitespace.
        source: Either a Hugging Face repo ID (e.g.
            ``'mlx-community/Mistral-7B-Instruct-v0.3-4bit'``)
            or a local filesystem path (e.g. ``'./models/foo'``).
    """

    label: str
    source: str

    @property
    def is_local(self) -> bool:
        """True if ``source`` looks like a local path (starts with / or ./ or ../)."""
        return self.source.startswith(("/", "./", "../")) or Path(self.source).exists()

    def __str__(self) -> str:
        kind = "local" if self.is_local else "hub"
        return f"{self.label} [{kind}: {self.source}]"


# -------- Preset definitions ------------------------------------------------

# Plan A: pre-existing mlx-community repos. Works without local conversion;
# ideal for a fast benchmark run with no extra setup. Only 4-bit and 8-bit
# variants of Mistral 7B are pre-published, so the comparison is narrow but
# still informative (~2x memory drop, ~2x param-storage cost).
#
# Plan B: full precision ladder (bf16 / 8-bit / 4-bit / 3-bit). bf16 and
# 3-bit must be locally converted from the original Mistral release using
# ``scripts/convert_models.sh``. Yields a richer comparison and a complete
# memory ladder, but requires ~14 GB extra disk and ~10 minutes of conversion.

_PRESETS: dict[str, list[ModelSpec]] = {
    # Plan A: just the pre-existing pieces from mlx-community
    "mistral7b-prebuilt": [
        ModelSpec("8bit", "mlx-community/Mistral-7B-Instruct-v0.3-8bit"),
        ModelSpec("4bit", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
    ],
    # Plan B: full ladder (assumes you've run scripts/convert_models.sh)
    "mistral7b-full": [
        ModelSpec("bf16", "./models/Mistral-7B-Instruct-v0.3-bf16"),
        ModelSpec("8bit", "mlx-community/Mistral-7B-Instruct-v0.3-8bit"),
        ModelSpec("4bit", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
        ModelSpec("3bit", "./models/Mistral-7B-Instruct-v0.3-3bit"),
    ],
    # Sanity check: tiny model that downloads in seconds, useful for verifying
    # the pipeline before committing to bigger runs
    "tinyllama": [
        ModelSpec("default", "mlx-community/TinyLlama-1.1B-Chat-v1.0-mlx"),
    ],
}


def list_presets() -> list[str]:
    """Names of all available presets."""
    return sorted(_PRESETS.keys())


def expand_preset(name: str) -> list[ModelSpec]:
    """Look up a preset by name and return its model specs.

    Raises:
        KeyError: If ``name`` is not a known preset.
    """
    if name not in _PRESETS:
        available = ", ".join(list_presets())
        raise KeyError(f"Unknown preset {name!r}. Available: {available}")
    return list(_PRESETS[name])


# -------- Spec parsing -----------------------------------------------------

def parse_spec(raw: str) -> ModelSpec:
    """Parse a ``label=source`` string into a :class:`ModelSpec`.

    Examples:
        >>> parse_spec("4bit=mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        ModelSpec(label='4bit', source='mlx-community/Mistral-7B-Instruct-v0.3-4bit')
        >>> parse_spec("bf16=./models/foo")
        ModelSpec(label='bf16', source='./models/foo')

    Raises:
        ValueError: If ``raw`` is malformed (missing ``=``, empty label, etc.).
    """
    if "=" not in raw:
        raise ValueError(
            f"Model spec must be 'label=source', got: {raw!r}.\n"
            f"Example: --model 4bit=mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        )
    label, _, source = raw.partition("=")
    label = label.strip()
    source = source.strip()
    if not label:
        raise ValueError(f"Empty label in spec: {raw!r}")
    if not source:
        raise ValueError(f"Empty source in spec: {raw!r}")
    if any(c.isspace() for c in label):
        raise ValueError(f"Label cannot contain whitespace: {label!r}")
    return ModelSpec(label=label, source=source)
