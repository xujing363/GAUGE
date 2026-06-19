#!/bin/bash
# Scenario A: Drug Resistance Mechanism Exploration
# ==================================================
# Cell-line split test set (190 unseen cell lines x 283 drugs)
# Perturbation: Transcriptome silencing, fixed drug
# Model: /results/true (best cell-line split world model)
#
# Usage: bash run_all.sh [--skip-01] [--figures-only]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SKIP_01=false
FIGURES_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-01) SKIP_01=true ;;
        --figures-only) FIGURES_ONLY=true ;;
    esac
done

run_script() {
    local name="$1"
    local script="$2"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Running: $name"
    echo "════════════════════════════════════════════════════════════"
    python "$SCRIPT_DIR/scripts/$script" 2>&1 | tee "$LOG_DIR/${script%.py}.log"
    echo "  ✓ $name complete"
}

if [ "$FIGURES_ONLY" = false ]; then
    if [ "$SKIP_01" = false ]; then
        run_script "01: Global Perturbation Landscape" "01_global_perturbation_landscape.py"
    fi
    run_script "02: World Model Gate Dynamics"     "02_world_model_gate_dynamics.py"
    run_script "03: Cancer-Type Perturbation"      "03_cancer_type_perturbation.py"
    run_script "04: Drug Family Deep Dive"         "04_drug_family_deep_dive.py"
    run_script "05: Three-Network Attribution"     "05_three_network_attribution.py"
fi

run_script "06: Publication Figures"           "06_publication_figures.py"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Scenario A Complete"
echo "  Results: $SCRIPT_DIR/results/"
echo "  Figures: $SCRIPT_DIR/figures/"
echo "════════════════════════════════════════════════════════════"
