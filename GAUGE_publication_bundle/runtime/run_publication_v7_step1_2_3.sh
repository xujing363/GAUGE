#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
BASE="$KGPUB_TARGETS/Combined/multicancer_contextual_v2/publication_submission/publication_v7"

python "$BASE/step1_nci_cellline/scripts/01_allcell_score_computation.py"
python "$BASE/step1_nci_cellline/scripts/02_nci_validation_allcell.py"
python "$BASE/step1_nci_cellline/scripts/03_kg_advantage_analysis.py"
python "$BASE/step2_ctrdb_patient/scripts/01_ctrdb_inference_and_combo_validation.py"
python "$BASE/step2_ctrdb_patient/scripts/run_two_drug_v5_complementarity.py"
python "$BASE/step3_tcga_screening/scripts/run_all_patient_analysis.py"
