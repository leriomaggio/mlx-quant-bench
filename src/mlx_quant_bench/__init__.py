"""mlx-quant-bench — quantization benchmarks for LLMs on Apple Silicon.

Compare loaded memory, time-to-first-token, throughput, and output quality
across precision levels (bf16, 8-bit, 4-bit, 3-bit) using Apple's MLX framework.

Note: top-level imports are deliberately minimal so ``import mlx_quant_bench``
works without MLX installed. Only :mod:`mlx_quant_bench.benchmark` and
:mod:`mlx_quant_bench.inspect` actually require MLX.
"""

__version__ = "0.1.0"

from mlx_quant_bench.spec import ModelSpec, expand_preset, list_presets, parse_spec

__all__ = [
    "ModelSpec",
    "parse_spec",
    "expand_preset",
    "list_presets",
]
