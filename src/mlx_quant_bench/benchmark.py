"""Benchmark execution: load a model, run prompts, record metrics.

For each prompt we record:

- ``ttft_s``: time to first token (seconds)
- ``decode_tps``: decode throughput (tokens per second after the first)
- ``output_tokens``: how many tokens the model produced
- ``output``: the full text response (for eyeball/quality assessment)

Per-model we also record:

- ``load_time_s``: how long the model took to load from disk
- ``peak_memory_gb``: peak MLX memory observed during the prompt batch

These are the metrics that map to real production decisions: TTFT bounds
chat-app responsiveness, decode throughput bounds long-form generation cost,
peak memory bounds what fits on the GPU at inference time.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load

from mlx_quant_bench.prompts import PROMPTS, Prompt
from mlx_quant_bench.spec import ModelSpec


# -------- Result types ------------------------------------------------------

@dataclass
class PromptResult:
    """Per-prompt measurement."""
    prompt_id: str
    category: str
    output: str
    output_tokens: int
    ttft_s: float
    decode_tps: float


@dataclass
class ModelResult:
    """Per-model measurement, including all prompt results."""
    label: str
    source: str
    load_time_s: float
    peak_memory_gb: float
    prompt_results: list[PromptResult]

    def to_jsonl_records(self) -> list[dict[str, Any]]:
        """Flatten into one JSONL record per prompt for easy downstream analysis."""
        records = []
        for pr in self.prompt_results:
            record = {
                "model_label": self.label,
                "model_source": self.source,
                "load_time_s": self.load_time_s,
                "peak_memory_gb": self.peak_memory_gb,
                **asdict(pr),
            }
            records.append(record)
        return records


# -------- Memory utilities --------------------------------------------------

def _peak_memory_gb() -> float:
    """Current peak MLX memory in GB."""
    return mx.metal.get_peak_memory() / 1e9


def _reset_peak_memory() -> None:
    """Reset MLX's peak memory counter."""
    mx.metal.reset_peak_memory()


def _clear_cache() -> None:
    """Release MLX internal cache (helpful between models)."""
    mx.metal.clear_cache()


# -------- Single-prompt benchmark ------------------------------------------

def _benchmark_single_prompt(
    model: Any,
    tokenizer: Any,
    prompt: Prompt,
    max_tokens: int = 256,
) -> PromptResult:
    """Run one prompt and time it."""
    messages = [{"role": "user", "content": prompt.text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # First-token timing: measure separately from the rest of generation.
    # We do this by generating one token, marking time, then continuing.
    # Note: mlx_lm.generate doesn't expose per-token callbacks across all
    # versions, so we approximate with two calls.
    t0 = time.perf_counter()
    first_response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=1,
        verbose=False,
    )
    ttft = time.perf_counter() - t0

    # Now generate the full response and time the decode phase
    t1 = time.perf_counter()
    full_response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        verbose=False,
    )
    decode_time = time.perf_counter() - t1

    # Approximate output token count from the tokenizer
    output_tokens = len(tokenizer.encode(full_response))
    decode_tps = (output_tokens - 1) / decode_time if decode_time > 0 and output_tokens > 1 else 0.0

    return PromptResult(
        prompt_id=prompt.id,
        category=prompt.category,
        output=full_response,
        output_tokens=output_tokens,
        ttft_s=round(ttft, 4),
        decode_tps=round(decode_tps, 2),
    )


# -------- Per-model benchmark ----------------------------------------------

def benchmark_model(
    spec: ModelSpec,
    prompts: list[Prompt] | None = None,
    max_tokens: int = 256,
    verbose: bool = True,
) -> ModelResult:
    """Load a model and run the full prompt suite against it.

    Args:
        spec: Model to benchmark (label + source).
        prompts: Prompts to run. Defaults to the full :data:`PROMPTS` list.
        max_tokens: Maximum tokens to generate per prompt.
        verbose: Print progress to stdout.

    Returns:
        :class:`ModelResult` with all measurements.
    """
    if prompts is None:
        prompts = list(PROMPTS)

    if verbose:
        print(f"\n=== {spec} ===")
        print(f"Loading from: {spec.source}")

    _reset_peak_memory()

    t0 = time.perf_counter()
    model, tokenizer = load(spec.source)
    load_time = time.perf_counter() - t0

    if verbose:
        print(f"Loaded in {load_time:.1f}s. Running {len(prompts)} prompts...")

    prompt_results = []
    for i, prompt in enumerate(prompts, 1):
        if verbose:
            print(f"  [{i}/{len(prompts)}] {prompt.id} ({prompt.category})", end=" ")
        result = _benchmark_single_prompt(model, tokenizer, prompt, max_tokens=max_tokens)
        prompt_results.append(result)
        if verbose:
            print(f"→ {result.decode_tps:.1f} tok/s, ttft={result.ttft_s:.2f}s")

    peak_gb = _peak_memory_gb()
    if verbose:
        print(f"Peak memory: {peak_gb:.2f} GB")

    # Drop the model and clear MLX's cache before the next run to avoid
    # cross-contamination of memory measurements.
    del model, tokenizer
    _clear_cache()

    return ModelResult(
        label=spec.label,
        source=spec.source,
        load_time_s=round(load_time, 2),
        peak_memory_gb=round(peak_gb, 3),
        prompt_results=prompt_results,
    )


# -------- JSONL output -----------------------------------------------------

def append_results(result: ModelResult, path: str | Path) -> None:
    """Append a model's prompt results to a JSONL file (one record per prompt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in result.to_jsonl_records():
            f.write(json.dumps(record) + "\n")


def reset_results(path: str | Path) -> None:
    """Delete the results file if it exists."""
    p = Path(path)
    if p.exists():
        p.unlink()
