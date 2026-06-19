#!/usr/bin/env bash
set -euo pipefail

source /mnt/raid5/xujing/miniconda3/etc/profile.d/conda.sh
conda activate kg_GAUGE

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 cuda:N|cpu [run options]" >&2
  exit 2
fi

DEVICE="$1"
shift

CACHE_ARGS=()
if [[ -n "${GAUGE_CACHE_DIR:-}" ]]; then
  CACHE_ARGS=(--cache-dir "$GAUGE_CACHE_DIR")
fi

python -m GAUGE \
  --out-dir /mnt/raid5/xujing/KG/GAUGE_runs/$(date +%Y%m%d_%H%M%S) \
  --device "$DEVICE" \
  "${CACHE_ARGS[@]}" \
  run \
  "$@"
