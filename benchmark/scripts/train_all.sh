#!/usr/bin/env bash
# Train all 3 splits x 5 seeds, spread over 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=(0 1 6 7)
SEEDS=(5 7 43 91 98)
SPLITS=(drug_split scaffold_split chemcluster_split)
OUT=${OUT:-runs}
mkdir -p "$OUT/logs"

i=0
for split in "${SPLITS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    python -m gauge_bench.train --split "$split" --seed "$seed" \
      --device "cuda:$gpu" --out-dir "$OUT/${split}_seed${seed}" \
      > "$OUT/logs/${split}_seed${seed}.log" 2>&1 &
    i=$((i + 1))
    if (( i % ${#GPUS[@]} == 0 )); then wait; fi
  done
done
wait

python scripts/summarize.py --runs "$OUT" --out "$OUT/summary"
