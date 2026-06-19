#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$KGPUB_TARGETS/benchmarking/07_tcga_actual_treatment/HVG2000"
bash ./strategy_quantile_map_hvg2000_task01/run_strategy.sh "$@"
