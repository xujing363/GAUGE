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

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
python "${BENCH_DIR}/scripts/run.py" --device "$DEVICE" "$@"
