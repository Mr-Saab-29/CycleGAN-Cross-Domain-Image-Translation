#!/usr/bin/env bash
set -euo pipefail

: "${INPUT:?Set INPUT=/path/to/image.jpg}"
: "${OUTPUT:?Set OUTPUT=/path/to/output.png}"
: "${DIRECTION:?Set DIRECTION=x2y or DIRECTION=y2x}"

DATASET="${DATASET:-apple2orange}"
EXPERIMENT="${EXPERIMENT:-apple2orange}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DEVICE="${DEVICE:-cpu}"

python scripts/infer_image.py --dataset-name "$DATASET" --experiment-name "$EXPERIMENT" --image-size "$IMAGE_SIZE" --device "$DEVICE" --direction "$DIRECTION" --input "$INPUT" --output "$OUTPUT"
