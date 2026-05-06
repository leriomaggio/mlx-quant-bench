# mlx-quant-bench

Quantization benchmarks for LLMs on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Compare loaded memory, time-to-first-token, decode throughput, and output quality across precision levels (bf16 / 8-bit / 4-bit / 3-bit) on a single Mac. Works with Mistral 7B Instruct v0.3 and any model in the `mlx-community` org on Hugging Face.

## Why this exists

Quantization is the single most-effective production optimisation for self-hosted LLMs: typically a 4× memory reduction at a marginal quality cost. But the trade-offs are model and task-specific, and the canonical tutorials don't actually show you the numbers on your hardware. This repo runs the comparison end-to-end on your Mac and produces a report you can read in five minutes.

The benchmark distinguishes three prompt categories: factual, reasoning, instruction-following - because precision degradation hits each one differently. 
Factual recall holds up to 4-bit easily; multi-step reasoning starts breaking earlier; instruction-following format compliance is often the first thing to go at 3-bit.

## Example output (Mistral 7B Instruct v0.3 on M3 Pro)

| Model | Peak memory | Load time | Avg TTFT | Avg tok/s |
|---|---|---|---|---|
| **8bit** | 9.0 GB | 7.0s | 0.20s | 30.5 |
| **4bit** | 4.5 GB | 5.2s | 0.15s | 46.0 |

> Numbers are illustrative — your run will differ based on hardware, prompt length, and MLX version. See `docs/REPORT_example.md` for a full annotated example.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/leriomaggio/mlx-quant-bench.git
cd mlx-quant-bench
pip install -e .

# 2. Set up Hugging Face access (one-time, see "Hugging Face setup" below)
hf auth login

# 3. Inspect your hardware
mlx-quant-bench inspect

# 4. Sanity-check with TinyLlama (~3 GB download, ~2 minute run)
mlx-quant-bench run --preset tinyllama --reset
mlx-quant-bench report results/runs.jsonl

# 5. The headline run: Mistral 7B at two precisions (~12 GB download, ~10 min)
mlx-quant-bench run --preset mistral7b-prebuilt --reset
mlx-quant-bench report results/runs.jsonl --md REPORT.md --show-outputs
```

That's the full Plan A workflow. The report markdown file is what you'd commit alongside results.

## Two benchmark plans

### Plan A — pre-built models (default, fastest)

Uses the 4-bit and 8-bit Mistral 7B variants pre-published on `mlx-community`. No conversion step, no special setup beyond Hugging Face login. The downside: only two precisions, which gives you a meaningful but narrow comparison.

```bash
mlx-quant-bench run --preset mistral7b-prebuilt --reset
```

This is the recommended path for first-run, for portfolio demonstrations, and for anyone who wants results in under 15 minutes.

### Plan B — full precision ladder (bf16, 8-bit, 4-bit, 3-bit)

Covers four precisions across the full quantization spectrum. The catch: `mlx-community` doesn't pre-publish bf16 or 3-bit variants for Mistral 7B — they need to be locally converted from the original Mistral release using `mlx_lm.convert`. The repo includes a helper script that does this in one command:

```bash
# One-time: locally convert bf16 and 3-bit (~10 min, ~28 GB disk)
bash scripts/convert_models.sh

# Run the full four-precision benchmark
mlx-quant-bench run --preset mistral7b-full --reset
mlx-quant-bench report results/runs.jsonl --md REPORT.md --show-outputs
```

Plan B is what you'd use if you want a complete memory ladder for a blog post or a thorough quality assessment. The conversion is a one-time cost; subsequent benchmarks reuse the cached models.

## Custom model specifications

If neither preset fits, specify models directly with `--model LABEL=SOURCE`. Each spec pairs a precision label (free-form, used in the report) with a source (Hugging Face repo ID or local path). Repeatable:

```bash
mlx-quant-bench run \
  --model 4bit=mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --model 8bit=mlx-community/Mistral-7B-Instruct-v0.3-8bit \
  --model my-experiment=./models/my-fine-tune
```

This is also the way to benchmark non-Mistral models — point at any `mlx-community` repo or a local MLX-format directory.

## Hugging Face setup

The benchmark downloads models from Hugging Face. Even though the `mlx-community` org is fully open, **Hugging Face requires authenticated access for downloads** as of late 2025.

### Step 1: Create an account

If you don't have one: [huggingface.co/join](https://huggingface.co/join). Free tier; no payment details needed.

### Step 2: Generate an access token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Click **"Create new token"**.
3. Choose **Read** as the token type (minimum required, safest).
4. Give it a memorable name (e.g. `mlx-quant-bench`).
5. **Copy the token immediately** — it starts with `hf_...` and is shown only once. Treat it like a password.

### Step 3: Log in via CLI

```bash
hf auth login
```

Paste the token when prompted (it won't show on screen — that's normal). The "Add token as git credential?" prompt: either answer is fine for this workflow.

> The CLI is `hf` as of late 2025. The older `huggingface-cli` binary is deprecated.

### Step 4: Verify

```bash
hf auth whoami
```

Should print your username. You're set — `mlx_lm.load()` will pick up the token automatically.

### Alternative: environment variable

If you'd rather not run `hf auth login`:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

Add to `~/.zshrc` for persistence.

### Optional: pre-download models

To avoid the model download happening during the benchmark itself:

```bash
# TinyLlama (sanity check, ~2 GB)
hf download mlx-community/TinyLlama-1.1B-Chat-v1.0-mlx

# Mistral 7B variants for Plan A (~12 GB total)
hf download mlx-community/Mistral-7B-Instruct-v0.3-4bit
hf download mlx-community/Mistral-7B-Instruct-v0.3-8bit
```

Models cache in `~/.cache/huggingface/hub/`. To redirect the cache to a larger disk:

```bash
export HF_HOME="/Volumes/BigDisk/.cache/huggingface"
```

Set this **before** both downloading and running the benchmark.

## Hardware requirements

This is Apple Silicon only — MLX uses Metal and won't run on Intel Macs. Approximate memory budgets for full benchmarks:

| Mac config | Plan A (4bit + 8bit) | Plan B (full ladder) |
|---|---|---|
| 16 GB unified memory | ✅ tight, close other apps | ❌ won't fit bf16 |
| 24 GB unified memory | ✅ comfortable | ⚠️ tight on bf16 |
| 36+ GB unified memory | ✅ comfortable | ✅ comfortable |

Run `mlx-quant-bench inspect` to see your specific budget and a model-fit table.

## Repository structure

```
mlx-quant-bench/
├── src/mlx_quant_bench/
│   ├── __init__.py        # Public API: ModelSpec, parse_spec, expand_preset
│   ├── cli.py             # `mlx-quant-bench` entry point
│   ├── spec.py            # Model spec parsing and presets
│   ├── prompts.py         # Benchmark prompt suite (factual/reasoning/instruction)
│   ├── benchmark.py       # Load model, run prompts, record metrics
│   ├── compare.py         # Read JSONL, render reports
│   └── inspect.py         # Hardware introspection
├── scripts/
│   └── convert_models.sh  # Plan B helper: locally convert bf16 + 3-bit
├── tests/
│   ├── test_spec.py       # Spec parsing and preset definitions
│   ├── test_prompts.py    # Prompt suite invariants
│   └── test_compare.py    # Report rendering
├── pyproject.toml
├── LICENSE
└── README.md (this file)
```

## What gets measured

For each prompt:
- **TTFT (time to first token)** — bounds chat-app responsiveness.
- **Decode throughput (tok/s)** — bounds long-form generation cost.
- **Output tokens** — for normalising throughput across prompts of different lengths.
- **Full output text** — for eyeball quality assessment.

Per-model:
- **Load time** — time to read weights from disk and initialise.
- **Peak GPU memory** — bounds what fits during inference.

All metrics land in a JSONL file, one record per prompt. The `report` subcommand turns this into a markdown summary with optional side-by-side outputs for visual quality comparison.

## Implementation notes

A few choices worth flagging:

- **MLX is the only backend.** No PyTorch, no `bitsandbytes`. This is a Mac-native benchmark for Mac-native deployment patterns.
- **TTFT is approximated** by running two `generate()` calls — one with `max_tokens=1` for the first-token timing, then a full generation for decode throughput. This isn't perfect (it double-counts the prefill), but it's robust across `mlx-lm` versions.
- **MLX peak memory is reset between models** via `mx.metal.reset_peak_memory()` to prevent cross-contamination.
- **The model spec system decouples "where to load from" from "what to call it".** This matters because `mlx-community` doesn't follow a uniform `-{precision}` suffix pattern across models — Mistral has 4-bit and 8-bit variants pre-published, TinyLlama has only a single MLX-format repo, bf16 / 3-bit Mistral need to be locally converted. The CLI handles all three cases through one interface.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The tests cover spec parsing, preset definitions, prompt invariants, and report rendering. They don't require MLX (those modules are imported lazily), so the test suite runs on any Python ≥3.10.

## License

MIT — see [LICENSE](LICENSE).

## Author

Valerio Maggio, Senior Technical Advocate / Developer Advocate.
[github.com/leriomaggio](https://github.com/leriomaggio) · [linkedin.com/in/valeriomaggio](https://linkedin.com/in/valeriomaggio)
