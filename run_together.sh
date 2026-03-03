#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="./src:${PYTHONPATH:-}"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

: "${TOGETHER_API_KEY:?Please set TOGETHER_API_KEY before running this script.}"

MODEL="zai-org/GLM-5"
BASE_OUTPUT_DIR="outputs/together/glm-5"

TASK_TYPES=(
  # "gen_eng_literal"
  # "gen_eng_fig"
  "gen_swa_literal"
  "gen_swa_fig"
)

for TASK_TYPE in "${TASK_TYPES[@]}"; do
  UV_CACHE_DIR=.uv-cache uv run -m proverb.commands.evaluate --config configs/default.yaml \
    --inference_backend together \
    --model_name_or_path "${MODEL}" \
    --task_type "${TASK_TYPE}" \
    --output_dir "${BASE_OUTPUT_DIR}/${TASK_TYPE}"
done
