#!/usr/bin/env bash
set -euo pipefail

source /mnt/raid5/xujing/miniconda3/etc/profile.d/conda.sh
conda activate kg_GAUGE

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 cuda:N|cpu [run_name] [epochs] [batch_size] [eval_batch_size]" >&2
  exit 2
fi

DEVICE="$1"
RUN_NAME="${2:-full_terminal_specificity_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${3:-}"
BATCH_SIZE="${4:-}"
EVAL_BATCH_SIZE="${5:-}"

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_REL="configs/terminal_specificity_additive.yaml"
EXTRA_ARGS=(--config "$CONFIG_REL" --device "$DEVICE" --run-name "$RUN_NAME")
if [[ -n "$EPOCHS" ]]; then
  EXTRA_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
  EXTRA_ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi

echo "[START] benchmark=$(basename "$BENCH_DIR") config=$CONFIG_REL"
echo "[START] device=${DEVICE} run_name=${RUN_NAME} epochs=${EPOCHS:-config} batch_size=${BATCH_SIZE:-config} eval_batch_size=${EVAL_BATCH_SIZE:-config/auto}"

time python "${BENCH_DIR}/scripts/run.py" "${EXTRA_ARGS[@]}"

echo "[DONE] results_dir=${BENCH_DIR}/results/${RUN_NAME}"
