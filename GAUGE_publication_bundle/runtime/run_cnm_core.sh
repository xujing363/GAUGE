#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
BASE="$KGPUB_TARGETS/DrugDesign_Sec/cnm"
cd "$BASE"
python scripts/01_drug_split_validation.py
python scripts/02_tcga_predict_drugsplit_model.py
python scripts/02_tcga_screening_value_hat.py
python scripts/03_tcga_indication_recovery.py
python scripts/04_drug_generation_brics_scoring.py
python scripts/05_chembl_moa_validation.py
python scripts/06_figures.py
python scripts/07_target_expression_validation.py
