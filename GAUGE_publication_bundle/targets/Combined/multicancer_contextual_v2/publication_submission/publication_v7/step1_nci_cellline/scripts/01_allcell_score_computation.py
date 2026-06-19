#!/usr/bin/env python3
"""
All-Cell-Line Complementarity Score Computation
================================================
GAUGE drug combination discovery — full cell-line coverage.

SCIENTIFIC RATIONALE FOR USING ALL CELL LINES:
  The GDSC train/val/test split is defined for evaluating the SINGLE-DRUG
  prediction model (i.e., how well the model generalises to unseen cell lines
  when predicting drug IC50 / AUC).  It does NOT impose any restriction on
  which cell lines can be used for *drug-combination discovery*.

  Drug combination synergy labels (NCI-ALMANAC) are completely independent of
  the GDSC split.  The model was never trained on NCI-ALMANAC data, so using
  train or validation GDSC cell lines to score drug pairs introduces NO label
  leakage with respect to the NCI validation target.

  BENEFIT: using all cancer-type cell lines (train+val+test) gives a more
  stable (lower-variance) estimate of each drug's activity profile, which in
  turn improves the complementarity score.

WHAT THIS SCRIPT DOES:
  1. Loads predictions.csv (all splits) from the frozen GAUGE source model.
  2. For each of 4 cancer types, selects ALL cells annotated as that cancer.
  3. Computes base_score = value_hat − 0.1 × uncertainty  (matches scorer_v2.py).
  4. For the same 851 KG-guided candidate pairs used in the test-only analysis,
     recomputes:
       - pcp              = median_c[A(c) × B(c)]
       - pearson_r        = Pearson(A_profile, B_profile)
       - complementarity  = pcp × (1 − pearson_r)   ← primary score
       - within-drug shuffle control (cell-permuted A × B)
  5. Saves:
       - allcell_per_cell_scores_{cancer}.csv   (long format: one row per pair × cell)
       - allcell_pair_scores_{cancer}.csv       (wide format: one row per pair)
       - comparison_cellcount_{cancer}.csv      (test-only vs all-cell n_cells)
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]          # multicancer_contextual_v2 root
MULTI_ROOT = ROOT                                    # alias
KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))
SOURCE_PRED_PATH = (
    KG_ROOT / "Combined/results"
    / "combined_melanoma_v1_20260524_130336"
    / "predictions.csv"
)
MODEL_LIST_PATH = KG_ROOT / "KG_GAUGE_PublicData/GDSC/model_list_20260420.csv"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_U = 0.1
SEED = 42
rng = np.random.RandomState(SEED)

# ── Cancer configuration ──────────────────────────────────────────────────────
# Maps our cancer label → NCI panel name, model-list keyword, candidate-pairs path,
# and the test-only per-cell CSV (for comparison).
CANCER_CONFIGS = {
    "Melanoma": {
        "nci_panel": "Melanoma",
        "model_list_keyword": "melanoma",
        "candidate_pairs_path": (
            "Melanoma/results/contextual_combined_v2_auto_kg_margin_gate"
            "/contextual_candidate_pairs.csv"
        ),
        "testonly_pc_path": (
            "Melanoma/results/contextual_combined_v2_auto_kg_margin_gate"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
    "NSCLC": {
        "nci_panel": "Non-Small Cell Lung Cancer",
        "model_list_keyword": "non-small cell lung",
        "candidate_pairs_path": (
            "Non_Small_Cell_Lung_Carcinoma/results/contextual_combined_v2_auto_kg_probe_fix1"
            "/contextual_candidate_pairs.csv"
        ),
        "testonly_pc_path": (
            "Non_Small_Cell_Lung_Carcinoma/results/contextual_combined_v2_auto_kg_probe_fix1"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
    "Breast": {
        "nci_panel": "Breast Cancer",
        "model_list_keyword": "breast",
        "candidate_pairs_path": (
            "Breast_Carcinoma/results/contextual_combined_v2_auto_kg_pubv1_gpu1"
            "/contextual_candidate_pairs.csv"
        ),
        "testonly_pc_path": (
            "Breast_Carcinoma/results/contextual_combined_v2_auto_kg_pubv1_gpu1"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
    "Ovarian": {
        "nci_panel": "Ovarian Cancer",
        "model_list_keyword": "ovarian",
        "candidate_pairs_path": (
            "Ovarian_Carcinoma/results/contextual_combined_v2_auto_kg_pubv1_gpu1"
            "/contextual_candidate_pairs.csv"
        ),
        "testonly_pc_path": (
            "Ovarian_Carcinoma/results/contextual_combined_v2_auto_kg_pubv1_gpu1"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
}

# ── Load source predictions (all splits) ─────────────────────────────────────
print("=" * 70)
print("Loading GAUGE predictions (all splits)...")
print(f"  Source: {SOURCE_PRED_PATH}")
pred_all = pd.read_csv(SOURCE_PRED_PATH)
print(f"  Rows: {len(pred_all):,}  |  Splits: {pred_all['split'].value_counts().to_dict()}")

# Compute base_score — matches _base_score() in scorer_v2.py
pred_all["base_score"] = pred_all["value_hat"] - LAMBDA_U * pred_all["uncertainty"]

# ── Load model list ───────────────────────────────────────────────────────────
model_list = pd.read_csv(MODEL_LIST_PATH)
print(f"  Model list: {len(model_list):,} cell lines")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cancer_cells(keyword: str) -> list[str]:
    """Return all SANGER_MODEL_IDs annotated as a given cancer type (all splits)."""
    cancer_rows = model_list[
        model_list["cancer_type"].str.lower().str.contains(keyword, na=False)
    ]
    # Intersect with cells that have predictions
    all_pred_cells = set(pred_all["SANGER_MODEL_ID"].unique())
    cells = sorted(set(cancer_rows["model_id"].astype(str)) & all_pred_cells)
    return cells


def compute_pair_scores(
    pair_df: pd.DataFrame,
    base_scores_by_drug: dict[int, pd.Series],
    cells: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-cell and aggregated complementarity scores for each pair.

    Returns:
        per_cell_df: long-format DataFrame (one row per pair × cell)
        agg_df:      wide-format DataFrame (one row per pair)
    """
    per_cell_rows = []
    agg_rows = []
    rng_state = np.random.RandomState(SEED)

    for _, pair in pair_df.iterrows():
        drug_a_id = int(pair["drug_A_id"])
        drug_b_id = int(pair["drug_B_id"])

        if drug_a_id not in base_scores_by_drug or drug_b_id not in base_scores_by_drug:
            continue

        # Per-cell base scores (aligned to the same cell order)
        scores_a = base_scores_by_drug[drug_a_id]
        scores_b = base_scores_by_drug[drug_b_id]

        # Some drugs may not have data for all cells — intersect
        common_cells = sorted(set(scores_a.index) & set(scores_b.index) & set(cells))
        if len(common_cells) < 2:
            continue

        A = scores_a.loc[common_cells].values.astype(float)
        B = scores_b.loc[common_cells].values.astype(float)
        n_cells = len(A)

        # ── Per-cell product (co-activity)
        pcp = float(np.median(A * B))

        # ── Pearson correlation (profile similarity / mechanism redundancy)
        if np.std(A) > 1e-9 and np.std(B) > 1e-9:
            pearson_r = float(pearsonr(A, B)[0])
        else:
            pearson_r = 0.0

        # ── v5 complementarity score (primary metric)
        complementarity = pcp * (1.0 - pearson_r)

        # ── Aggregate product (v4 baseline)
        agg_product = float(np.median(A)) * float(np.median(B))

        # ── Within-drug shuffle control (permute A's cell assignment)
        rng_idx = rng_state.permutation(n_cells)
        A_shuf = A[rng_idx]
        pcp_shuf = float(np.median(A_shuf * B))
        if np.std(A_shuf) > 1e-9 and np.std(B) > 1e-9:
            pearson_shuf = float(pearsonr(A_shuf, B)[0])
        else:
            pearson_shuf = 0.0
        complementarity_shuf = pcp_shuf * (1.0 - pearson_shuf)

        # ── Per-cell rows
        for i, cell in enumerate(common_cells):
            per_cell_rows.append({
                "unordered_pair_key": pair["unordered_pair_key"],
                "SANGER_MODEL_ID": cell,
                "drug_A_id": drug_a_id,
                "drug_A_name": pair["drug_A_name"],
                "drug_B_id": drug_b_id,
                "drug_B_name": pair["drug_B_name"],
                "base_score_A": A[i],
                "base_score_B": B[i],
            })

        agg_rows.append({
            "unordered_pair_key": pair["unordered_pair_key"],
            "drug_A_id": drug_a_id,
            "drug_A_name": pair["drug_A_name"],
            "drug_B_id": drug_b_id,
            "drug_B_name": pair["drug_B_name"],
            "n_cells": n_cells,
            "pcp": pcp,
            "agg_product": agg_product,
            "pearson_r": pearson_r,
            "inv_pearson": 1.0 - pearson_r,
            "complementarity": complementarity,
            "pcp_shuf": pcp_shuf,
            "complementarity_shuf": complementarity_shuf,
            "kg_support_sources": pair.get("kg_support_sources", ""),
            "kg_support_source_count": pair.get("kg_support_source_count", 0),
            "kg_support_score": pair.get("kg_support_score", 0.0),
        })

    return pd.DataFrame(per_cell_rows), pd.DataFrame(agg_rows)


# ── Main per-cancer loop ──────────────────────────────────────────────────────
cancer_summary = []

for cancer, cfg in CANCER_CONFIGS.items():
    print("\n" + "=" * 70)
    print(f"Cancer: {cancer}")
    print("=" * 70)

    # ── Identify ALL cancer-type cells ──────────────────────────────────────
    all_cells = get_cancer_cells(cfg["model_list_keyword"])
    print(f"  All cancer cells (train+val+test): {len(all_cells)}")

    # Breakdown by split
    cancer_pred = pred_all[pred_all["SANGER_MODEL_ID"].isin(all_cells)]
    split_counts = cancer_pred.groupby("split")["SANGER_MODEL_ID"].nunique().to_dict()
    print(f"  Split breakdown: {split_counts}")

    # ── Build base_score lookup: drug_id → Series(cell → base_score) ────────
    # Each drug entry: pivot predictions for cells in all_cells
    cancer_pred_sub = cancer_pred[cancer_pred["SANGER_MODEL_ID"].isin(all_cells)]
    base_scores_by_drug: dict[int, pd.Series] = {}
    for drug_id, grp in cancer_pred_sub.groupby("DRUG_ID"):
        s = grp.set_index("SANGER_MODEL_ID")["base_score"]
        base_scores_by_drug[int(drug_id)] = s

    print(f"  Drugs with predictions in these cells: {len(base_scores_by_drug)}")

    # ── Test-only cells for comparison ────────────────────────────────────────
    test_cells = sorted(
        cancer_pred[cancer_pred["split"] == "test"]["SANGER_MODEL_ID"].unique()
    )
    print(f"  Test-only cells: {len(test_cells)}")

    # ── Load candidate pairs ───────────────────────────────────────────────
    candidate_pairs = pd.read_csv(MULTI_ROOT / cfg["candidate_pairs_path"])
    print(f"  KG candidate pairs: {len(candidate_pairs)}")

    # ── Compute scores: ALL cells ──────────────────────────────────────────
    print("  Computing complementarity scores (all cells)...")
    per_cell_allcell, agg_allcell = compute_pair_scores(
        candidate_pairs, base_scores_by_drug, all_cells
    )
    agg_allcell["cell_set"] = "all_cells"
    agg_allcell["cancer"] = cancer
    print(f"    Scored pairs: {len(agg_allcell)}  |  Per-cell rows: {len(per_cell_allcell)}")
    print(f"    Mean n_cells per pair: {agg_allcell['n_cells'].mean():.1f}")

    # ── Compute scores: TEST-ONLY cells (for comparison) ───────────────────
    print("  Computing complementarity scores (test-only cells)...")
    per_cell_testonly, agg_testonly = compute_pair_scores(
        candidate_pairs, base_scores_by_drug, test_cells
    )
    agg_testonly["cell_set"] = "test_only"
    agg_testonly["cancer"] = cancer

    # ── Save outputs ───────────────────────────────────────────────────────
    per_cell_allcell.to_csv(
        RESULTS_DIR / f"allcell_per_cell_scores_{cancer.lower()}.csv", index=False
    )
    agg_allcell.to_csv(
        RESULTS_DIR / f"allcell_pair_scores_{cancer.lower()}.csv", index=False
    )

    # Comparison: test-only vs all-cell statistics for key metrics
    comp_rows = []
    for col in ["pcp", "complementarity", "pearson_r"]:
        comp_rows.append({
            "cancer": cancer,
            "metric": col,
            "allcell_mean": agg_allcell[col].mean(),
            "allcell_std": agg_allcell[col].std(),
            "allcell_n_cells_mean": agg_allcell["n_cells"].mean(),
            "testonly_mean": agg_testonly[col].mean(),
            "testonly_std": agg_testonly[col].std(),
            "testonly_n_cells_mean": agg_testonly["n_cells"].mean(),
        })
    pd.DataFrame(comp_rows).to_csv(
        RESULTS_DIR / f"comparison_stats_{cancer.lower()}.csv", index=False
    )

    cancer_summary.append({
        "cancer": cancer,
        "n_cells_allcell": len(all_cells),
        "n_cells_testonly": len(test_cells),
        "n_cells_fold_increase": len(all_cells) / len(test_cells),
        "n_pairs_scored": len(agg_allcell),
        "complementarity_mean_allcell": agg_allcell["complementarity"].mean(),
        "complementarity_std_allcell": agg_allcell["complementarity"].std(),
        "complementarity_mean_testonly": agg_testonly["complementarity"].mean(),
        "complementarity_std_testonly": agg_testonly["complementarity"].std(),
        "split_breakdown": str(split_counts),
    })

    print(f"  Saved: allcell_pair_scores_{cancer.lower()}.csv")
    print(f"  Complementarity (all-cell): mean={agg_allcell['complementarity'].mean():.4f} "
          f"std={agg_allcell['complementarity'].std():.4f}")
    print(f"  Complementarity (test-only): mean={agg_testonly['complementarity'].mean():.4f} "
          f"std={agg_testonly['complementarity'].std():.4f}")

# ── Save summary ──────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(cancer_summary)
summary_df.to_csv(RESULTS_DIR / "01_allcell_computation_summary.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY: Cell line expansion")
print("=" * 70)
print(summary_df[["cancer", "n_cells_testonly", "n_cells_allcell", "n_cells_fold_increase"]].to_string(index=False))
print(f"\nAll outputs saved to: {RESULTS_DIR}")
print("[DONE] Script 01 complete.")
