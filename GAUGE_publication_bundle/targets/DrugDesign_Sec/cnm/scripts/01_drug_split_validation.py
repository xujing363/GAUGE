#!/usr/bin/env python3
"""
Drug-split holdout validation.

Key claim: The model predicts drug sensitivity for completely unseen drugs
(held out by drug identity during training) using molecular fingerprints alone.
This is validated against actual PRISM AUC measurements.

Output:
  results/drug_split_validation.csv      — per-drug PCC for val/test splits
  results/drug_split_summary.json        — aggregate statistics
  results/drug_split_topK.csv            — top held-out cancer drugs by predictability
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

ROOT = Path(__file__).resolve().parents[3]
PRED_CSV = ROOT / "PRISM/Secondary/cheml35/results/20260524_224312/predictions.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PRISM secondary screen treatment info (for drug class annotation)
REPURP_TREAT = ROOT / "KG_GAUGE_PublicData/repurposing/secondary/secondary-screen-replicate-collapsed-treatment-info.csv"


def load_predictions() -> pd.DataFrame:
    cols = ["DRUG_ID", "DRUG_NAME", "SANGER_MODEL_ID", "split",
            "AUC", "relative_value", "auc_hat", "value_hat", "uncertainty"]
    preds = pd.read_csv(PRED_CSV, usecols=cols)
    return preds


def per_drug_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (drug_id, drug_name, split), grp in preds.groupby(["DRUG_ID", "DRUG_NAME", "split"]):
        n = len(grp)
        if n < 10:
            continue
        # value_hat ↔ relative_value
        try:
            pcc_vh, _ = pearsonr(grp["value_hat"], grp["relative_value"])
        except Exception:
            pcc_vh = float("nan")
        # auc_hat ↔ AUC
        try:
            pcc_ah, _ = pearsonr(grp["auc_hat"], grp["AUC"])
        except Exception:
            pcc_ah = float("nan")
        try:
            rho_vh, _ = spearmanr(grp["value_hat"], grp["relative_value"])
        except Exception:
            rho_vh = float("nan")
        try:
            rho_ah, _ = spearmanr(grp["auc_hat"], grp["AUC"])
        except Exception:
            rho_ah = float("nan")
        rows.append({
            "drug_id": int(drug_id),
            "drug_name": drug_name,
            "split": split,
            "n_cells": n,
            "pcc_value_hat": round(float(pcc_vh), 6),
            "pcc_auc_hat": round(float(pcc_ah), 6),
            "spearman_value_hat": round(float(rho_vh), 6),
            "spearman_auc_hat": round(float(rho_ah), 6),
        })
    return pd.DataFrame(rows)


def annotate_drug_class(drug_df: pd.DataFrame) -> pd.DataFrame:
    """Attach drug class from PRISM treatment info."""
    if not REPURP_TREAT.exists():
        drug_df["moa"] = ""
        drug_df["target"] = ""
        return drug_df
    treat = pd.read_csv(REPURP_TREAT)
    treat_dedup = treat.drop_duplicates("name")
    name2moa = treat_dedup.set_index("name")["moa"].to_dict()
    name2tgt = treat_dedup.set_index("name")["target"].to_dict()
    lower_map_moa = {k.lower(): v for k, v in name2moa.items()}
    lower_map_tgt = {k.lower(): v for k, v in name2tgt.items()}
    drug_df["moa"] = drug_df["drug_name"].str.lower().map(lower_map_moa).fillna("")
    drug_df["target"] = drug_df["drug_name"].str.lower().map(lower_map_tgt).fillna("")
    return drug_df


def compute_summary(df: pd.DataFrame) -> dict:
    out = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        out[split] = {
            "n_drugs": len(sub),
            "mean_pcc_value_hat": round(float(sub["pcc_value_hat"].mean()), 4),
            "median_pcc_value_hat": round(float(sub["pcc_value_hat"].median()), 4),
            "mean_pcc_auc_hat": round(float(sub["pcc_auc_hat"].mean()), 4),
            "median_pcc_auc_hat": round(float(sub["pcc_auc_hat"].median()), 4),
            "mean_spearman_value_hat": round(float(sub["spearman_value_hat"].mean()), 4),
            "mean_spearman_auc_hat": round(float(sub["spearman_auc_hat"].mean()), 4),
            "frac_pcc_gt_0": round(float((sub["pcc_value_hat"] > 0).mean()), 4),
            "frac_pcc_gt_0.3": round(float((sub["pcc_value_hat"] > 0.3).mean()), 4),
        }

    # Generalization gap: val vs train
    train_m = df[df["split"] == "train"]["pcc_value_hat"].mean()
    val_m = df[df["split"] == "val"]["pcc_value_hat"].mean()
    test_m = df[df["split"] == "test"]["pcc_value_hat"].mean()
    out["generalization_gap_train_vs_val"] = round(float(train_m - val_m), 4)
    out["generalization_gap_train_vs_test"] = round(float(train_m - test_m), 4)

    # MWU: is val/test significantly different from train?
    train_pccs = df[df["split"] == "train"]["pcc_value_hat"].dropna()
    val_pccs   = df[df["split"] == "val"]["pcc_value_hat"].dropna()
    test_pccs  = df[df["split"] == "test"]["pcc_value_hat"].dropna()
    if len(val_pccs) >= 5:
        _, p_val = mannwhitneyu(val_pccs, train_pccs, alternative="less")
        out["p_val_lt_train"] = round(float(p_val), 6)
    if len(test_pccs) >= 5:
        _, p_test = mannwhitneyu(test_pccs, train_pccs, alternative="less")
        out["p_test_lt_train"] = round(float(p_test), 6)

    # One-sample t-test: is mean PCC significantly > 0? (null = no predictive power)
    for split_name, pccs in [("val", val_pccs), ("test", test_pccs)]:
        if len(pccs) >= 5:
            _, p_gt_zero = ttest_1samp(pccs, 0.0, alternative="greater")
            out[f"p_{split_name}_pcc_gt_zero"] = float(p_gt_zero)
            out[f"tstat_{split_name}_pcc_gt_zero"] = round(
                float((pccs.mean()) / (pccs.std() / len(pccs) ** 0.5)), 4
            )
    return out


def identify_top_holdout_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """Top 30 held-out (val+test) drugs by value_hat PCC."""
    holdout = df[df["split"].isin(["val", "test"])].copy()
    holdout = holdout.sort_values("pcc_value_hat", ascending=False)
    return holdout.head(30)


def main():
    print("Loading predictions...")
    preds = load_predictions()
    print(f"  {len(preds):,} rows, {preds['DRUG_ID'].nunique()} drugs")

    print("Computing per-drug metrics...")
    drug_df = per_drug_metrics(preds)
    drug_df = annotate_drug_class(drug_df)
    drug_df.to_csv(OUT_DIR / "drug_split_validation.csv", index=False)
    print(f"  Saved {len(drug_df)} drug-split rows")

    summary = compute_summary(drug_df)
    with open(OUT_DIR / "drug_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:")
    print(json.dumps(summary, indent=2))

    top = identify_top_holdout_drugs(drug_df)
    top.to_csv(OUT_DIR / "drug_split_topK.csv", index=False)
    print(f"\nTop held-out drugs (PCC value_hat):")
    print(top[["drug_name", "split", "n_cells", "pcc_value_hat", "pcc_auc_hat", "moa"]].to_string(index=False))


if __name__ == "__main__":
    main()
