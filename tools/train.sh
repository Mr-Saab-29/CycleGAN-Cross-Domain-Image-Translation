#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-apple2orange}"
EXPERIMENT="${EXPERIMENT:-apple2orange}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DEVICE="${DEVICE:-cpu}"

python scripts/train.py --dataset-name "$DATASET" --experiment-name "$EXPERIMENT" --image-size "$IMAGE_SIZE" --device "$DEVICE"
