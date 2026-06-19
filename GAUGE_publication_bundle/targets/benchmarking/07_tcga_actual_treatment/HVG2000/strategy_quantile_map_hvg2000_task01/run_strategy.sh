#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_ROOT="${KGPUB_PY_ROOT:-/mnt/raid5/xujing/KG}"
PYTHONPATH="${PY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python strategy_quantile_map_hvg2000_task01/scripts/run_quantile_map_task01.py --config strategy_quantile_map_hvg2000_task01/config.yaml
python strategy_quantile_map_hvg2000_task01/task_01_dataset_profile_and_prediction_export/run_task.py
