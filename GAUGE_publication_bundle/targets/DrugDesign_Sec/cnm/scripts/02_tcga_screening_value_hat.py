#!/usr/bin/env python3
"""
TCGA Virtual Drug Screening with value_hat.

value_hat predicts relative_value — a GLOBAL rank-normalized sensitivity score
(each drug's mean relative_value ≈ 0.5, std ≈ 0.289 across all cell lines).
This makes value_hat cross-drug comparable: for a given patient, higher value_hat
means the model predicts that patient is more sensitive to that drug relative
to the global population.

Key analyses:
  1. Per-cancer indication recovery: AUROC of value_hat ranking for known indications
  2. Pan-cancer mean AP and MRR (from existing task_03 outputs)
  3. Top-ranked drugs per cancer type with drug class annotation
  4. Drug ranking landscape (heatmap data for LUAD)
  5. Null permutation comparison

Outputs (all in cnm/results/):
  tcga_indication_recovery_by_cancer.csv
  tcga_top_drugs_per_cancer.csv
  tcga_screening_summary.json
  tcga_luad_patient_drug_matrix.csv  (subset: LUAD, top 50 drugs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.multitest import fdrcorrection

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TCGA_PREDS = (
    ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000"
    / "strategy_quantile_map_hvg2000_task01/results/predictions.csv"
)
TASK03_PAN = (
    ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000"
    / "strategy_quantile_map_hvg2000_task01/task_03_indication_recovery/outputs"
    / "indication_recovery_patient_metrics_primary_pan_cancer.csv"
)
TASK03_BY_PROJECT = (
    ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000"
    / "strategy_quantile_map_hvg2000_task01/task_03_indication_recovery/outputs"
    / "indication_recovery_summary_primary_by_project.csv"
)
INDICATION_TRUTH = (
    ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000"
    / "strategy_quantile_map_hvg2000_task01/task_03_indication_recovery/outputs"
    / "indication_truth_drug_map.csv"
)
REPURP_TREAT = ROOT / "KG_GAUGE_PublicData/repurposing/secondary/secondary-screen-replicate-collapsed-treatment-info.csv"


def load_drug_class() -> dict[int, str]:
    """Map DRUG_ID → drug class."""
    if not REPURP_TREAT.exists():
        return {}
    treat = pd.read_csv(REPURP_TREAT)
    treat = treat.drop_duplicates("broad_id").dropna(subset=["broad_id"])

    def classify(row):
        t = str(row.get("target", "")).lower()
        m = str(row.get("moa", "")).lower()
        if any(k in t for k in ["egfr", "erbb"]):
            return "EGFR/ErbB_inhibitor"
        if any(k in t + m for k in ["mek", "braf", "kras"]):
            return "MEK/BRAF_inhibitor"
        if "cdk" in m or "cyclin" in m:
            return "CDK_inhibitor"
        if any(k in m for k in ["pi3k", "akt", "mtor"]):
            return "PI3K/AKT/mTOR_inhibitor"
        if "jak" in t or "stat" in t:
            return "JAK/STAT_inhibitor"
        if "topoisomerase" in m or "topoisomerase" in t:
            return "Topoisomerase_inhibitor"
        if "tubulin" in m or "microtubule" in m:
            return "Tubulin_inhibitor"
        if "glucocorticoid" in m:
            return "Glucocorticoid"
        if "androgen" in m:
            return "Androgen_receptor"
        return "Other"

    id_lookup: dict[int, str] = {}
    return id_lookup  # name-based below is easier


def load_name_drug_class() -> dict[str, str]:
    if not REPURP_TREAT.exists():
        return {}
    treat = pd.read_csv(REPURP_TREAT)
    treat_dedup = treat.drop_duplicates("name")

    def classify(row):
        t = str(row.get("target", "")).lower()
        m = str(row.get("moa", "")).lower()
        if any(k in t for k in ["egfr", "erbb"]):
            return "EGFR/ErbB"
        if any(k in t + m for k in ["mek", "braf", "kras"]):
            return "MEK/BRAF"
        if "cdk" in m or "cyclin" in m:
            return "CDK"
        if any(k in m for k in ["pi3k", "akt", "mtor"]):
            return "PI3K/AKT/mTOR"
        if "topoisomerase" in m or "topo" in t:
            return "Topoisomerase"
        if "tubulin" in m or "microtubule" in m:
            return "Tubulin"
        if "glucocorticoid" in m:
            return "Glucocorticoid"
        if "androgen" in m:
            return "Androgen"
        return "Other"

    return {row["name"].lower(): classify(row) for _, row in treat_dedup.iterrows()}


def compute_per_cancer_top_drugs(preds: pd.DataFrame, drug_class_map: dict) -> pd.DataFrame:
    """For each cancer type, rank drugs by mean value_hat and return top 20."""
    rows = []
    for project_id, grp in preds.groupby("project_id"):
        drug_means = grp.groupby(["DRUG_ID", "DRUG_NAME"])["value_hat"].mean().reset_index()
        drug_means = drug_means.sort_values("value_hat", ascending=False)
        drug_means["rank"] = range(1, len(drug_means) + 1)
        drug_means["project_id"] = project_id
        drug_means["drug_class"] = drug_means["DRUG_NAME"].str.lower().map(
            lambda n: drug_class_map.get(n, "Other")
        )
        rows.append(drug_means.head(20))
    return pd.concat(rows, ignore_index=True)


def value_hat_indication_auroc(preds: pd.DataFrame, truth_map: pd.DataFrame) -> pd.DataFrame:
    """
    For each (project, drug) pair where an indication truth exists,
    compute AUROC: does value_hat rank that drug higher than non-indicated drugs?

    This is a WITHIN-PATIENT test: for a single patient, does the model rank
    the correct indication drug higher than incorrect drugs?
    """
    # Build truth: project_id -> set of true indication DRUG_IDs
    project_true_drugs: dict[str, set[int]] = {}
    for _, row in truth_map.iterrows():
        proj = str(row["project_id"])
        if proj not in project_true_drugs:
            project_true_drugs[proj] = set()
        project_true_drugs[proj].add(int(row["DRUG_ID"]))

    rows = []
    for project_id, proj_preds in preds.groupby("project_id"):
        true_drugs = project_true_drugs.get(str(project_id), set())
        if not true_drugs:
            continue
        all_drug_ids = proj_preds["DRUG_ID"].unique()
        n_true = len(true_drugs & set(all_drug_ids))
        if n_true == 0:
            continue
        n_patients = proj_preds["entity_id"].nunique()

        # Per-patient AUROC
        auroc_list = []
        for _, pat_grp in proj_preds.groupby("entity_id"):
            if pat_grp["DRUG_ID"].nunique() < 5:
                continue
            y_true = pat_grp["DRUG_ID"].isin(true_drugs).astype(int).values
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            try:
                auroc = roc_auc_score(y_true, pat_grp["value_hat"].values)
                auroc_list.append(auroc)
            except Exception:
                pass

        if len(auroc_list) < 5:
            continue
        rows.append({
            "project_id": project_id,
            "n_patients": n_patients,
            "n_true_drugs": n_true,
            "n_total_drugs": len(all_drug_ids),
            "mean_auroc": round(float(np.mean(auroc_list)), 4),
            "median_auroc": round(float(np.median(auroc_list)), 4),
            "std_auroc": round(float(np.std(auroc_list)), 4),
            "n_evaluated_patients": len(auroc_list),
            "frac_auroc_gt_0.6": round(float(np.mean(np.array(auroc_list) > 0.6)), 4),
        })
    return pd.DataFrame(rows).sort_values("mean_auroc", ascending=False)


def compute_null_auroc(preds: pd.DataFrame, truth_map: pd.DataFrame,
                       n_permutations: int = 100) -> float:
    """Estimate null AUROC by shuffling value_hat within each patient."""
    project_true_drugs: dict[str, set[int]] = {}
    for _, row in truth_map.iterrows():
        proj = str(row["project_id"])
        if proj not in project_true_drugs:
            project_true_drugs[proj] = set()
        project_true_drugs[proj].add(int(row["DRUG_ID"]))

    rng = np.random.default_rng(42)
    null_aurocs = []
    sample_patients = preds["entity_id"].unique()[:200]  # subsample for speed

    for pat_id in sample_patients:
        pat_grp = preds[preds["entity_id"] == pat_id]
        proj = pat_grp["project_id"].iloc[0]
        true_drugs = project_true_drugs.get(str(proj), set())
        y_true = pat_grp["DRUG_ID"].isin(true_drugs).astype(int).values
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        for _ in range(n_permutations):
            shuffled = rng.permutation(pat_grp["value_hat"].values)
            try:
                null_aurocs.append(roc_auc_score(y_true, shuffled))
            except Exception:
                pass
    return float(np.mean(null_aurocs)) if null_aurocs else 0.5


def main():
    print("Loading TCGA predictions...")
    preds = pd.read_csv(TCGA_PREDS)
    print(f"  {len(preds):,} rows | {preds['entity_id'].nunique():,} patients | "
          f"{preds['DRUG_NAME'].nunique()} drugs | {preds['project_id'].nunique()} cancer types")

    print("\n=== value_hat distribution (should be globally distributed, cross-drug comparable) ===")
    per_drug_std = preds.groupby("DRUG_NAME")["value_hat"].std()
    print(f"  Global std: {preds['value_hat'].std():.4f}")
    print(f"  Per-drug std (median): {per_drug_std.median():.4f}")
    print(f"  Top 5 most discriminative drugs:")
    print(per_drug_std.nlargest(5).round(4).to_string())

    # Drug class mapping
    drug_class_map = load_name_drug_class()

    # Top drugs per cancer type
    print("\nComputing top drugs per cancer type...")
    top_drugs = compute_per_cancer_top_drugs(preds, drug_class_map)
    top_drugs.to_csv(OUT_DIR / "tcga_top_drugs_per_cancer.csv", index=False)

    # Load indication truth
    print("Loading indication truth...")
    truth_map = pd.read_csv(INDICATION_TRUTH) if INDICATION_TRUTH.exists() else pd.DataFrame()
    print(f"  {len(truth_map)} drug-indication pairs")

    # AUROC analysis
    if len(truth_map) > 0:
        print("Computing per-cancer AUROC...")
        auroc_df = value_hat_indication_auroc(preds, truth_map)
        auroc_df.to_csv(OUT_DIR / "tcga_indication_auroc_by_cancer.csv", index=False)
        print(auroc_df.to_string(index=False))

        print("\nEstimating null AUROC...")
        null_auroc = compute_null_auroc(preds, truth_map, n_permutations=50)
        print(f"  Null AUROC (shuffle): {null_auroc:.4f}")
        overall_mean_auroc = auroc_df["mean_auroc"].mean()
        print(f"  Overall mean AUROC: {overall_mean_auroc:.4f}")
    else:
        auroc_df = pd.DataFrame()
        null_auroc = 0.5

    # Load existing task_03 outputs
    pan = pd.read_csv(TASK03_PAN) if TASK03_PAN.exists() else pd.DataFrame()
    by_project = pd.read_csv(TASK03_BY_PROJECT) if TASK03_BY_PROJECT.exists() else pd.DataFrame()

    # LUAD patient drug matrix (top 50 discriminative drugs)
    print("\nBuilding LUAD patient drug matrix...")
    luad_preds = preds[preds["project_id"] == "TCGA-LUAD"].copy()
    # Top 50 drugs by std (most discriminative)
    top50_drugs = (luad_preds.groupby("DRUG_NAME")["value_hat"].std()
                   .nlargest(50).index.tolist())
    luad_matrix = (luad_preds[luad_preds["DRUG_NAME"].isin(top50_drugs)]
                   .pivot_table(index="entity_id", columns="DRUG_NAME",
                                values="value_hat", aggfunc="mean"))
    luad_matrix.to_csv(OUT_DIR / "tcga_luad_drug_matrix_top50.csv")
    print(f"  LUAD matrix: {luad_matrix.shape}")

    # Summary
    summary = {
        "n_patients": int(preds["entity_id"].nunique()),
        "n_drugs": int(preds["DRUG_NAME"].nunique()),
        "n_cancer_types": int(preds["project_id"].nunique()),
        "global_value_hat_std": round(float(preds["value_hat"].std()), 4),
        "median_per_drug_std": round(float(per_drug_std.median()), 4),
        "indication_recovery": {
            "pan_cancer_mean_ap": round(float(pan["ap"].mean()), 4) if len(pan) else None,
            "pan_cancer_mean_mrr": round(float(pan["mrr"].mean()), 4) if len(pan) else None,
            "pan_cancer_hit_at_10_rate": round(float(pan["hit_at_10"].mean()), 4) if len(pan) else None,
        } if len(pan) > 0 else {},
        "auroc_analysis": {
            "n_cancer_types_with_truth": len(auroc_df),
            "mean_auroc_across_cancers": round(float(auroc_df["mean_auroc"].mean()), 4) if len(auroc_df) else None,
            "null_auroc": round(float(null_auroc), 4),
            "auroc_lift_over_null": round(float(auroc_df["mean_auroc"].mean() - null_auroc), 4) if len(auroc_df) else None,
        },
    }
    if len(by_project) > 0:
        summary["top_cancer_types_by_hit_at_10"] = (
            by_project[by_project["hit_at_10"] > 0]
            .nlargest(5, "hit_at_10")[["project_id", "hit_at_10", "mean_ap"]]
            .to_dict(orient="records")
        )
    with open(OUT_DIR / "tcga_screening_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nScreening summary:")
    print(json.dumps(summary, indent=2))

    # Also copy the per-project results from task_03
    if len(by_project) > 0:
        by_project.to_csv(OUT_DIR / "tcga_indication_recovery_by_cancer.csv", index=False)
        print(f"\nPer-cancer indication recovery (top 10 by hit@10):")
        print(by_project.nlargest(10, "hit_at_10")[
            ["project_id", "n_patients", "mean_ap", "mean_mrr", "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10"]
        ].to_string(index=False))

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
