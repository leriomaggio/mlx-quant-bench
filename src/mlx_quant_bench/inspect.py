"""Hardware inspection: report what MLX sees as the available GPU and memory.

Useful before launching a benchmark to confirm which models will fit and
to record platform context alongside the benchmark numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class HardwareInfo:
    architecture: str
    total_memory_gb: float
    recommended_budget_gb: float
    max_buffer_gb: float


def get_hardware_info() -> HardwareInfo:
    """Query MLX for the device it's running on."""
    info = mx.metal.device_info()
    return HardwareInfo(
        architecture=info.get("architecture", "unknown"),
        total_memory_gb=round(info.get("memory_size", 0) / 1e9, 2),
        recommended_budget_gb=round(
            info.get("max_recommended_working_set_size", 0) / 1e9, 2
        ),
        max_buffer_gb=round(info.get("max_buffer_length", 0) / 1e9, 2),
    )


# Rough "loaded memory" estimates for popular models. The constant 1.3
# accounts for activations + KV cache overhead beyond the raw weight bytes.
_MODEL_FIT_TABLE: list[tuple[str, float]] = [
    ("TinyLlama 1.1B", 1.1),
    ("Mistral 7B", 7.0),
    ("Llama 3 8B", 8.0),
    ("Mixtral 8x7B", 47.0),
    ("Llama 3 70B", 70.0),
]


def estimate_loaded_memory_gb(params_billions: float, bits: int) -> float:
    """Estimate loaded memory for a model given parameter count and precision."""
    weights_gb = params_billions * bits / 8
    return weights_gb * 1.3  # +30% for activations + KV cache (rough)


def print_inspection_report() -> None:
    """Print hardware info and a model-fit estimation table."""
    hw = get_hardware_info()
    print()
    print(f"Architecture:        {hw.architecture}")
    print(f"Total memory:        {hw.total_memory_gb:.1f} GB")
    print(f"Recommended budget:  {hw.recommended_budget_gb:.1f} GB")
    print(f"Max single buffer:   {hw.max_buffer_gb:.1f} GB")
    print()
    print(
        "Model fit estimates (loaded memory ≈ weights × 1.3 for activations + KV cache):"
    )
    print()

    budget = hw.recommended_budget_gb
    bit_widths = [16, 8, 4, 3]

    # Header
    header = f"  {'Model':<18s}  " + "  ".join(
        f"{b}-bit:".rjust(12) for b in bit_widths
    )
    print(header)

    for name, pb in _MODEL_FIT_TABLE:
        cells = []
        for b in bit_widths:
            est = estimate_loaded_memory_gb(pb, b)
            fit = "OK" if est < budget else "X"
            cells.append(f"{est:>5.1f}GB {fit}".rjust(12))
        print(f"  {name:<18s}  " + "  ".join(cells))

    print()
    print(f"  (OK = fits comfortably under {budget:.1f} GB recommended budget)")
    print()
