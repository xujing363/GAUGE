#!/usr/bin/env python3
"""Aggregate run directories into per-seed and per-split metric tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

AGG_COLS = ["test_overall_pcc", "test_within_drug_pcc", "test_within_drug_spearman",
            "test_within_cell_pcc", "test_raw_auc_rmse"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of run subdirectories")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for meta_path in sorted(Path(args.runs).glob("*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        metrics = pd.read_csv(meta_path.parent / "gdsc_metrics.csv")
        test = metrics.loc[metrics["split"].eq("test")].iloc[0]
        rows.append({
            "split": meta.get("split", ""), "seed": meta["seed"],
            "best_epoch": meta["best_epoch"],
            "selected_fusion_weight": meta["selected_fusion_weight"],
            "test_overall_pcc": test["overall_pcc"],
            "test_within_drug_pcc": test["within_drug_pcc_mean"],
            "test_within_drug_spearman": test["within_drug_spearman_mean"],
            "test_within_cell_pcc": test["within_cell_pcc_mean"],
            "test_raw_auc_rmse": test["raw_auc_rmse"],
            "test_value_spearman": test["value_spearman"],
        })
    if not rows:
        raise SystemExit(f"no run_meta.json found under {args.runs}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows).sort_values(["split", "seed"])
    per_seed.to_csv(out / "benchmark_per_seed.csv", index=False)
    per_seed.groupby("split")[AGG_COLS].agg(["mean", "std"]).round(4).to_csv(
        out / "benchmark_summary.csv")
    print(per_seed.to_string(index=False))


if __name__ == "__main__":
    main()
