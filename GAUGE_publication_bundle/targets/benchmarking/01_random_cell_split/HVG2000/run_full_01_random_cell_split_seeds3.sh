#!/usr/bin/env bash
set -euo pipefail

source /mnt/raid5/xujing/miniconda3/etc/profile.d/conda.sh
conda activate kg_GAUGE

SEEDS=(79 50 98 17 41)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 cuda:N|cpu [epochs] [batch_size] [base_run_name] [eval_batch_size]" >&2
  exit 2
fi

DEVICE="$1"
EPOCHS="${2:-}"
BATCH_SIZE="${3:-}"
BASE_RUN_NAME="${4:-full_$(date +%Y%m%d_%H%M%S)_pid$$}"
EVAL_BATCH_SIZE="${5:-}"

if [[ "$DEVICE" != "cpu" && "$DEVICE" != cuda:* ]]; then
  echo "Invalid device: $DEVICE (use cpu or cuda:N)" >&2
  exit 2
fi

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_REL="configs/default.yaml"
LOCK_DIR="${BENCH_DIR}/results/.device_locks"
LOCK_NAME="${DEVICE//:/_}.lock"
LOCK_PATH="${LOCK_DIR}/${LOCK_NAME}"

COMMON_ARGS=(--config "$CONFIG_REL" --gdsc-source-mode v2 --device "$DEVICE")
[[ -n "$EPOCHS" ]] && COMMON_ARGS+=(--epochs "$EPOCHS")
[[ -n "$BATCH_SIZE" ]] && COMMON_ARGS+=(--batch-size "$BATCH_SIZE")
[[ -n "$EVAL_BATCH_SIZE" ]] && COMMON_ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")

mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "[ERROR] device lock busy: ${DEVICE}" >&2
  echo "[ERROR] lock_path=${LOCK_PATH}" >&2
  echo "[ERROR] another ${BENCH_DIR} run is already using ${DEVICE}" >&2
  exit 3
fi

echo "[START] benchmark=$(basename "$BENCH_DIR")"
echo "[START] pid=$$  device=${DEVICE}  seeds=${SEEDS[*]}"
echo "[START] epochs=${EPOCHS:-config}  batch_size=${BATCH_SIZE:-config}  eval_batch_size=${EVAL_BATCH_SIZE:-config/auto}"
echo "[START] base_run_name=${BASE_RUN_NAME}"
echo "[START] lock_path=${LOCK_PATH}"

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="${BASE_RUN_NAME}_seed${SEED}"
  RESULT_DIR="${BENCH_DIR}/results/${RUN_NAME}"
  echo "[SEED] seed=${SEED}  run_name=${RUN_NAME}"
  time python "${BENCH_DIR}/scripts/run.py" \
    "${COMMON_ARGS[@]}" \
    --split-seed "$SEED" \
    --seed "$SEED" \
    --run-name "$RUN_NAME"
  echo "[DONE] seed=${SEED}  results_dir=${RESULT_DIR}"
done

echo "[ALL DONE] base_run_name=${BASE_RUN_NAME}"
