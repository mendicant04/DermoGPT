#!/usr/bin/env bash
set -euo pipefail

MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-8B-Instruct}"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:?Set SOURCE_MODEL_PATH}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:?Set MERGED_MODEL_PATH}"
MERGE_SCRIPT="${MERGE_SCRIPT:-src/merge_lora_weights.py}"

export PYTHONPATH="src:${PYTHONPATH:-}"

python "${MERGE_SCRIPT}" \
    --model-path "${SOURCE_MODEL_PATH}" \
    --model-base "${MODEL_BASE}" \
    --save-model-path "${MERGED_MODEL_PATH}" \
    --safe-serialization
