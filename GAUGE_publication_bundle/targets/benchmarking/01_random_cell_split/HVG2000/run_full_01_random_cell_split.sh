#!/usr/bin/env bash
set -euo pipefail

source /mnt/raid5/xujing/miniconda3/etc/profile.d/conda.sh
conda activate kg_GAUGE

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 cuda:N|cpu [epochs] [batch_size] [run_name] [eval_batch_size]" >&2
  exit 2
fi

DEVICE="$1"
EPOCHS="${2:-}"
BATCH_SIZE="${3:-}"
RUN_NAME="${4:-full_$(date +%Y%m%d_%H%M%S)}"
EVAL_BATCH_SIZE="${5:-}"

if [[ "$DEVICE" != "cpu" && "$DEVICE" != cuda:* ]]; then
  echo "Invalid device: $DEVICE (use cpu or cuda:N)" >&2
  exit 2
fi

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="${BENCH_DIR}/results/${RUN_NAME}"
EXTRA_ARGS=(--device "$DEVICE" --run-name "$RUN_NAME")
if [[ -n "$EPOCHS" ]]; then
  EXTRA_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  EXTRA_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
  EXTRA_ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi

echo "[START] benchmark=$(basename "$BENCH_DIR")"
echo "[START] device=${DEVICE} epochs=${EPOCHS:-config} batch_size=${BATCH_SIZE:-config} eval_batch_size=${EVAL_BATCH_SIZE:-config/auto}"
echo "[START] run_name=${RUN_NAME}"

time python "${BENCH_DIR}/scripts/run.py" \
  "${EXTRA_ARGS[@]}"

echo "[DONE] run_name=${RUN_NAME}"
echo "[DONE] results_dir=${RESULT_DIR}"
