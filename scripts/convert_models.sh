#!/usr/bin/env bash
# Plan B helper: locally convert Mistral 7B Instruct v0.3 to MLX format
# at four precision levels (bf16, 8-bit, 4-bit, 3-bit).
#
# Why this exists: the mlx-community Hugging Face org has 4-bit and 8-bit
# variants pre-published for Mistral 7B Instruct v0.3, but no bf16 or 3-bit.
# Plan A (the default) skips bf16 and 3-bit and uses the two pre-built repos.
# Plan B (this script) builds the full four-precision ladder locally.
#
# Time: ~10 minutes total on M3 Pro (~2 minutes per conversion + 14 GB
# one-time download of the original Mistral release).
# Disk: ~30 GB total (the original ~14 GB + ~14 GB for all four variants).
#
# Usage:
#   bash scripts/convert_models.sh
#
# Then run:
#   mlx-quant-bench run --preset mistral7b-full

set -euo pipefail

SOURCE="mistralai/Mistral-7B-Instruct-v0.3"
OUTDIR="./models"
mkdir -p "$OUTDIR"

echo "==> Converting $SOURCE to bf16 (no quantization)"
mlx_lm.convert \
    --hf-path "$SOURCE" \
    --mlx-path "$OUTDIR/Mistral-7B-Instruct-v0.3-bf16" \
    --dtype bfloat16

echo
echo "==> Converting $SOURCE to 3-bit"
mlx_lm.convert \
    --hf-path "$SOURCE" \
    --mlx-path "$OUTDIR/Mistral-7B-Instruct-v0.3-3bit" \
    -q \
    --q-bits 3

echo
echo "Done. Local models are in $OUTDIR/."
echo "4-bit and 8-bit will be downloaded from mlx-community on first benchmark run."
echo
echo "Now run:  mlx-quant-bench run --preset mistral7b-full"
