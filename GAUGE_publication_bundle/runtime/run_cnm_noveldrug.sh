#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
BASE="$KGPUB_TARGETS/DrugDesign_Sec/cnm_novelDrug"
cd "$BASE"
python scripts/n01_kg_gap_analysis.py
python scripts/n02_kg_proxy_scoring.py
python scripts/n03_figures.py
