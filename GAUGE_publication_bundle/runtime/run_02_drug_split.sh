#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$KGPUB_TARGETS/benchmarking/02_drug_split/HVG2000"
python ./scripts/run.py "$@"
