#!/bin/bash
# Scenario B: Drug Repurposing Analysis
# Run all analysis scripts in order.
# B01 and B02 use pre-computed data only (fast).
# B03 and B04 require model loading + GPU (slow, ~30-60 min each).
# B05 uses pre-computed data only (fast, but needs B03 delta data for figures).
# B06 assembles publication figures from all results.

set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
RESULTS_DIR="$(pwd)/results"

echo "======================================================================"
echo "Scenario B: Drug Repurposing via Drug Network Perturbation"
echo "Working directory: $(pwd)"
echo "======================================================================"

# Fast analyses (no model needed)
echo ""
echo "--- B01: Global Overview (pre-computed data) ---"
$PYTHON scripts/B01_global_overview.py

echo ""
echo "--- B02: Dynamic Alpha / Static→Dynamic (pre-computed data) ---"
$PYTHON scripts/B02_dynamic_alpha_static_to_dynamic.py

echo ""
echo "--- B05: Repurposing Discovery (pre-computed data) ---"
$PYTHON scripts/B05_repurposing_discovery.py

# Model-dependent analyses (require GPU)
echo ""
echo "--- B03: Drug Network Ablation (requires GPU model) ---"
$PYTHON scripts/B03_drug_network_ablation.py

echo ""
echo "--- B04: Cancer-Type Specificity (requires GPU model) ---"
$PYTHON scripts/B04_cancer_specificity.py

# Publication figures (requires all results)
echo ""
echo "--- B06: Publication Figures ---"
$PYTHON scripts/B06_publication_figures.py

echo ""
echo "======================================================================"
echo "All Scenario B analyses complete."
echo "Results: $RESULTS_DIR"
echo "Figures: $(pwd)/figures"
echo "======================================================================"
