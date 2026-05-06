# MLX Quantization Benchmark — Example Output

> This is an illustrative example, not real benchmark numbers from the maintainer's hardware. Run the benchmark on your own Mac to get your own numbers. Numbers below approximate what you'd see on an M3 Pro / 36 GB running Mistral 7B Instruct v0.3.

## Summary

| Model | Source | Peak memory | Load time | Avg TTFT | Avg tok/s | Prompts |
|---|---|---|---|---|---|---|
| **8bit** | `mlx-community/Mistral-7B-Instruct-v0.3-8bit` | 8.94 GB | 7.1s | 0.21s | 30.4 | 10 |
| **4bit** | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4.51 GB | 5.3s | 0.14s | 46.8 | 10 |

## Memory comparison

Baseline: **8bit** at 8.94 GB (largest of the compared models).

- **4bit**: 4.51 GB (1.98× smaller than 8bit)

## Side-by-side outputs

### `capital_france` (factual)

**8bit** (32.1 tok/s, 14 tokens):

```
The capital of France is Paris.
```

**4bit** (47.2 tok/s, 12 tokens):

```
The capital of France is Paris.
```

### `train_problem` (reasoning)

**8bit** (28.4 tok/s, 142 tokens):

```
To find when the trains meet, I'll calculate using their combined speed.

Train 1 speed: 60 km/h
Train 2 speed: 90 km/h
Combined speed: 60 + 90 = 150 km/h
Distance: 300 km

Time = Distance ÷ Combined speed = 300 ÷ 150 = 2 hours

The trains will meet after 2 hours.
```

**4bit** (45.8 tok/s, 138 tokens):

```
To solve this, I'll add the speeds (since they're approaching each other) 
and divide the distance by the combined speed.

Combined speed = 60 km/h + 90 km/h = 150 km/h
Time to meet = 300 km ÷ 150 km/h = 2 hours

The trains meet after 2 hours.
```

### `json_extract` (instruction)

**8bit** (29.7 tok/s, 21 tokens):

```
{"name": "Alice", "age": 32}
```

**4bit** (46.1 tok/s, 21 tokens):

```
{"name": "Alice", "age": 32}
```

### `haiku_format` (instruction)

**8bit** (30.2 tok/s, 26 tokens):

```
Bits compressed tight,
Numbers shrink, weights stay alive,
Memory whispers.
```

**4bit** (45.5 tok/s, 28 tokens):

```
Numbers shrink small,
Float drops to four bits, weights live,
Less RAM consumed.
```

---

## What the numbers say

- **Memory**: 8-bit takes nearly 2× the memory of 4-bit (8.94 GB vs 4.51 GB), as expected from the bit width ratio.
- **Speed**: 4-bit is ~50% faster on decode (46.8 vs 30.4 tok/s). The smaller weights mean less memory bandwidth pressure on the Apple GPU.
- **Quality**: factual and instruction prompts are indistinguishable between precisions. Reasoning outputs differ in wording but reach the same answer. Haiku format compliance holds at both precisions, with different (both reasonable) creative choices.

The headline finding: **for Mistral 7B Instruct v0.3, 4-bit is the clear winner on a Mac**. Half the memory, faster generation, no visible quality loss on this prompt suite. The 8-bit precision earns its keep only if you're hitting genuinely hard reasoning tasks where the small precision boost matters: the prompt set above doesn't surface that.

> A complete **Plan B** run (bf16 / 8-bit / 4-bit / 3-bit) would show where quality starts to break. Typically reasoning prompts at 3-bit are the first to degrade visibly. To produce that, run `bash scripts/convert_models.sh` followed by `mlx-quant-bench run --preset mistral7b-full`.
