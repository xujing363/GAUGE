#!/usr/bin/env bash
# Scenario C — Run all steps sequentially
# Usage: bash run_all.sh [--step N]  (default: run all)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

STEP="${1:-all}"

if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
    log "=== Step 1: Drug-gene perturbation (2000 HVGs × 56 drugs) ==="
    python3 C01_drug_gene_perturbation.py
    log "Step 1 done."
fi

if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
    log "=== Step 2: Cell-target perturbation (190 cells × targets) ==="
    python3 C02_cell_target_perturbation.py
    log "Step 2 done."
fi

if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
    log "=== Step 3: Synthetic lethality prediction + SynLeth validation ==="
    python3 C03_synthetic_lethality.py
    log "Step 3 done."
fi

if [[ "$STEP" == "all" || "$STEP" == "4" ]]; then
    log "=== Step 4: Cancer-type specific analysis ==="
    python3 C04_cancer_type_analysis.py
    log "Step 4 done."
fi

if [[ "$STEP" == "all" || "$STEP" == "5" ]]; then
    log "=== Step 5: Publication figures ==="
    python3 C05_publication_figures.py
    log "Step 5 done."
fi

log "=== All steps complete. Results in results/ and figures/ ==="
