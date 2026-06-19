#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
REPORT="$KGPUB_ROOT/runtime/smoke_report.json"

ok=true
check() {
  local name="$1"
  local cmd="$2"
  if bash -lc "$cmd" >/tmp/kgpub_smoke_${name}.log 2>&1; then
    echo "[PASS] $name"
  else
    echo "[FAIL] $name"
    ok=false
  fi
}

check py_import "python -c \"import GAUGE; print('ok')\""
check link_01_true "test -e '$KGPUB_TARGETS/benchmarking/01_random_cell_split/HVG2000/results/true/metrics.csv'"
check link_02_true "test -e '$KGPUB_TARGETS/benchmarking/02_drug_split/HVG2000/results/true/metrics.csv'"
check link_07_results "test -e '$KGPUB_TARGETS/benchmarking/07_tcga_actual_treatment/HVG2000/strategy_quantile_map_hvg2000_task01/results/predictions.csv'"
check script_01_help "python '$KGPUB_TARGETS/benchmarking/01_random_cell_split/HVG2000/scripts/run.py' --help"
check script_02_help "python '$KGPUB_TARGETS/benchmarking/02_drug_split/HVG2000/scripts/run.py' --help"
check peru_a_exists "test -f '$KGPUB_TARGETS/Peru/A/scripts/01_global_perturbation_landscape.py'"
check pub_v7_step1_exists "test -f '$KGPUB_TARGETS/Combined/multicancer_contextual_v2/publication_submission/publication_v7/step1_nci_cellline/scripts/01_allcell_score_computation.py'"

status="pass"
$ok || status="fail"
printf '{\n  "status": "%s"\n}\n' "$status" > "$REPORT"
echo "Smoke report: $REPORT"
$ok
