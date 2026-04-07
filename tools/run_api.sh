#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-apple2orange}"
EXPERIMENT="${EXPERIMENT:-apple2orange}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DEVICE="${DEVICE:-cpu}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export MODEL_DATASET_NAME="$DATASET"
export MODEL_EXPERIMENT_NAME="$EXPERIMENT"
export MODEL_CHECKPOINT_ROOT="checkpoints/pytorch"
export MODEL_OUTPUT_ROOT="outputs"
export MODEL_IMAGE_SIZE="$IMAGE_SIZE"
export MODEL_DEVICE="$DEVICE"

python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
