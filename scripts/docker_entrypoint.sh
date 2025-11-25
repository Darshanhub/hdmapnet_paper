#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/data}
IMAGE_DIR=${IMAGE_DIR:-}
MODEL_PATH=${MODEL_PATH:-/checkpoints/model.pt}
OUTPUT_DIR=${OUTPUT_DIR:-/outputs}
FORCE_CPU=${FORCE_CPU:-1}

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --dataroot "${DATA_ROOT}"
  --modelf "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${IMAGE_DIR}" ]]; then
  ARGS+=(--image_dir "${IMAGE_DIR}")
fi

if [[ "${FORCE_CPU}" == "1" ]]; then
  ARGS+=(--force-cpu)
fi

ARGS+=("$@")

python scripts/run_inference.py "${ARGS[@]}"
