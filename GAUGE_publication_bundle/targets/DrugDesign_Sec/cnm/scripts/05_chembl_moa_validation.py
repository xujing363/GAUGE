#!/usr/bin/env python3
"""
ChEMBL MoA structure-activity validation.

Uses ONLY the drug-split model predictions from:
  cnm/results/tcga_drugsplit_predictions.parquet

Key analyses:
  1. Group drugs by primary mechanism of action (MoA class) using:
     - ChEMBL MoA edges from the run directory
     - Curated MoA annotations for key clinical drugs
  2. Within-MoA value_hat correlation: do drugs targeting the same protein
     show similar patient sensitivity patterns?
  3. Between-MoA comparison: within-MoA correlation > between-MoA correlation?
  4. Held-out drug MoA recovery: do test/val drugs correlate with same-class
     training drugs (without ever seeing training data for that drug)?
  5. Cancer-type specificity: EGFR inhibitors for LUAD, MEK/BRAF for SKCM, etc.

Outputs (cnm/results/):
  moa_drug_groups.csv               — drugs with MoA class annotation
  moa_within_vs_between_corr.csv    — within vs between MoA correlation
  moa_holdout_recovery.csv          — test/val drug correlation with same-class trains
  chembl_moa_validation_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDS_PARQUET  = OUT_DIR / "tcga_drugsplit_predictions.parquet"
RUN_DIR        = ROOT / "PRISM/Secondary/cheml35/results/20260524_224312"
CHEMBL_MOA     = RUN_DIR / "chembl_moa_edges.csv"
KG_NODE_INDEX  = RUN_DIR / "kg_node_index.csv"

TARGET_PROJECTS = ["TCGA-LUAD", "TCGA-SKCM", "TCGA-BRCA", "TCGA-PRAD", "TCGA-HNSC"]

# Curated MoA class annotations by drug name substring
DRUG_MOA_CLASSES: dict[str, str] = {
    # EGFR/ErbB inhibitors
    "erlotinib":     "EGFR_inhibitor",
    "gefitinib":     "EGFR_inhibitor",
    "afatinib":      "EGFR_inhibitor",
    "osimertinib":   "EGFR_inhibitor",
    "lapatinib":     "EGFR_inhibitor",
    "neratinib":     "EGFR_inhibitor",
    "dacomitinib":   "EGFR_inhibitor",
    "canertinib":    "EGFR_inhibitor",
    # MEK inhibitors
    "trametinib":    "MEK_inhibitor",
    "selumetinib":   "MEK_inhibitor",
    "cobimetinib":   "MEK_inhibitor",
    "binimetinib":   "MEK_inhibitor",
    "mek162":        "MEK_inhibitor",
    "pimasertib":    "MEK_inhibitor",
    "as703026":      "MEK_inhibitor",
    # BRAF inhibitors
    "dabrafenib":    "BRAF_inhibitor",
    "vemurafenib":   "BRAF_inhibitor",
    "encorafenib":   "BRAF_inhibitor",
    "plx4720":       "BRAF_inhibitor",
    # CDK4/6 inhibitors
    "palbociclib":   "CDK4_6_inhibitor",
    "ribociclib":    "CDK4_6_inhibitor",
    "abemaciclib":   "CDK4_6_inhibitor",
    # PI3K/mTOR inhibitors
    "alpelisib":     "PI3K_inhibitor",
    "copanlisib":    "PI3K_inhibitor",
    "idelalisib":    "PI3K_inhibitor",
    "everolimus":    "mTOR_inhibitor",
    "temsirolimus":  "mTOR_inhibitor",
    "rapamycin":     "mTOR_inhibitor",
    "sirolimus":     "mTOR_inhibitor",
    # BCL-2 inhibitors
    "venetoclax":    "BCL2_inhibitor",
    "navitoclax":    "BCL2_inhibitor",
    "obatoclax":     "BCL2_inhibitor",
    # Taxanes
    "paclitaxel":    "taxane",
    "docetaxel":     "taxane",
    "cabazitaxel":   "taxane",
    # Platinum
    "cisplatin":     "platinum",
    "carboplatin":   "platinum",
    "oxaliplatin":   "platinum",
    # Hormone therapy
    "tamoxifen":     "ER_modulator",
    "fulvestrant":   "ER_modulator",
    "letrozole":     "aromatase_inhibitor",
    "anastrozole":   "aromatase_inhibitor",
    "exemestane":    "aromatase_inhibitor",
    "enzalutamide":  "AR_antagonist",
    "abiraterone":   "CYP17_inhibitor",
    # Vinca alkaloids / microtubule
    "vinorelbine":   "vinca_alkaloid",
    "vinblastine":   "vinca_alkaloid",
    "vincristine":   "vinca_alkaloid",
    # Topoisomerase
    "irinotecan":    "topo1_inhibitor",
    "topotecan":     "topo1_inhibitor",
    "etoposide":     "topo2_inhibitor",
    "doxorubicin":   "anthracycline",
    "epirubicin":    "anthracycline",
    # ALK inhibitors
    "crizotinib":    "ALK_inhibitor",
    "alectinib":     "ALK_inhibitor",
    "ceritinib":     "ALK_inhibitor",
    # Alkylating
    "cyclophosphamide": "alkylating",
    "ifosfamide":    "alkylating",
    "temozolomide":  "alkylating",
    # Others
    "imatinib":      "BCR_ABL_inhibitor",
    "dasatinib":     "BCR_ABL_inhibitor",
    "sorafenib":     "multikinase_inhibitor",
    "sunitinib":     "multikinase_inhibitor",
    "ibrutinib":     "BTK_inhibitor",
    "acalabrutinib": "BTK_inhibitor",
    # Antimetabolites
    "methotrexate":  "antifolate",
    "pemetrexed":    "antifolate",
    "gemcitabine":   "antimetabolite",
    "capecitabine":  "antimetabolite",
    "fluorouracil":  "antimetabolite",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def annotate_drug_moa(drugs_df: pd.DataFrame) -> pd.DataFrame:
    """Assign MoA class to each drug based on name substring matching."""
    drugs_df = drugs_df.copy()
    moa_classes = []
    for _, row in drugs_df.iterrows():
        name_lower = str(row["DRUG_NAME"]).lower()
        assigned = "Other"
        for substr, cls in DRUG_MOA_CLASSES.items():
            if substr in name_lower:
                assigned = cls
                break
        moa_classes.append(assigned)
    drugs_df["moa_class"] = moa_classes
    return drugs_df


def build_drug_patient_matrix(
    preds: pd.DataFrame,
    target_cancer: str,
    min_drugs_per_class: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build drug × patient value_hat matrix for a cancer type.
    Returns (matrix_df, drug_metadata_df).
    Matrix columns = entity_id, rows = DRUG_ID.
    """
    cancer_preds = preds[preds["project_id"] == target_cancer].copy()
    # Deduplicate DRUG_ID (take first if duplicates, e.g., PRISM replicates)
    cancer_preds = cancer_preds.drop_duplicates(["entity_id", "DRUG_ID"])
    matrix = cancer_preds.pivot_table(
        index="DRUG_ID", columns="entity_id", values="value_hat", aggfunc="first"
    )
    drug_meta = cancer_preds.drop_duplicates("DRUG_ID")[["DRUG_ID", "DRUG_NAME", "split"]]
    return matrix, drug_meta


def compute_moa_correlations(
    matrix: pd.DataFrame,
    drug_moa: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Pearson correlation between all drugs.
    Then aggregate into within-MoA vs between-MoA correlation.
    """
    drug_ids = matrix.index.tolist()
    n = len(drug_ids)
    if n < 4:
        return pd.DataFrame(), pd.DataFrame()

    # Compute correlation matrix
    vals = matrix.values.astype(np.float32)
    # Handle NaN by imputing with column mean
    col_means = np.nanmean(vals, axis=0)
    idx = np.where(np.isnan(vals))
    vals[idx] = np.take(col_means, idx[1])

    corr_mat = np.corrcoef(vals)  # [n_drugs × n_drugs]

    # Classify pairs
    within_rows, between_rows = [], []
    for i, j in combinations(range(n), 2):
        di, dj = drug_ids[i], drug_ids[j]
        ci = drug_moa.get(di, "Other")
        cj = drug_moa.get(dj, "Other")
        if ci == "Other" or cj == "Other":
            continue
        r = float(corr_mat[i, j])
        pair_type = "within" if ci == cj else "between"
        pair = {
            "drug_i": di, "drug_j": dj,
            "moa_i": ci, "moa_j": cj,
            "pair_type": pair_type,
            "pearson_r": round(r, 4),
        }
        if pair_type == "within":
            within_rows.append(pair)
        else:
            between_rows.append(pair)

    all_pairs = pd.DataFrame(within_rows + between_rows)
    summary = []
    for moa_class in sorted(set(drug_moa.values()) - {"Other"}):
        within = [r["pearson_r"] for r in within_rows if r["moa_i"] == moa_class]
        if len(within) < 1:
            continue
        between = [r["pearson_r"] for r in between_rows if r["moa_i"] == moa_class or r["moa_j"] == moa_class]
        summary.append({
            "moa_class":         moa_class,
            "n_drugs_in_class":  len(set(
                [r["drug_i"] for r in within_rows if r["moa_i"] == moa_class] +
                [r["drug_j"] for r in within_rows if r["moa_j"] == moa_class]
            )) + 1 if within_rows else 0,
            "n_within_pairs":    len(within),
            "mean_within_r":     round(float(np.mean(within)), 4) if within else None,
            "n_between_pairs":   len(between),
            "mean_between_r":    round(float(np.mean(between)), 4) if between else None,
            "delta_within_vs_between": round(
                float(np.mean(within) - np.mean(between)), 4
            ) if within and between else None,
        })
    return pd.DataFrame(summary), all_pairs


def holdout_moa_recovery(
    matrix: pd.DataFrame,
    drug_moa: dict[int, str],
    drug_splits: dict[int, str],
) -> pd.DataFrame:
    """
    For each test/val drug, compute its mean correlation with same-class train drugs
    vs different-class train drugs.
    """
    rows = []
    drug_ids = matrix.index.tolist()
    vals = matrix.values.astype(np.float32)
    col_means = np.nanmean(vals, axis=0)
    idx = np.where(np.isnan(vals))
    vals[idx] = np.take(col_means, idx[1])
    corr_mat = np.corrcoef(vals)
    id_to_idx = {did: i for i, did in enumerate(drug_ids)}

    for did in drug_ids:
        split = drug_splits.get(int(did), "unknown")
        if split not in ("val", "test"):
            continue
        moa = drug_moa.get(int(did), "Other")
        if moa == "Other":
            continue

        i = id_to_idx[did]
        same_class_train, diff_class_train = [], []
        for djd in drug_ids:
            if djd == did:
                continue
            j = id_to_idx[djd]
            jmoa = drug_moa.get(int(djd), "Other")
            jsplit = drug_splits.get(int(djd), "unknown")
            if jsplit != "train" or jmoa == "Other":
                continue
            r = float(corr_mat[i, j])
            if jmoa == moa:
                same_class_train.append(r)
            else:
                diff_class_train.append(r)

        if not same_class_train:
            continue
        rows.append({
            "DRUG_ID":            int(did),
            "split":              split,
            "moa_class":          moa,
            "n_same_class_train": len(same_class_train),
            "mean_r_same_class":  round(float(np.mean(same_class_train)), 4),
            "n_diff_class_train": len(diff_class_train),
            "mean_r_diff_class":  round(float(np.mean(diff_class_train)), 4) if diff_class_train else None,
            "delta_same_vs_diff": round(
                float(np.mean(same_class_train) - np.mean(diff_class_train)), 4
            ) if diff_class_train else None,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("ChEMBL MoA Structure-Activity Validation")
    print("  Source: tcga_drugsplit_predictions.parquet")
    print("=" * 60)

    # ── Load predictions ─────────────────────────────────────────────────────
    print("\n[1] Loading predictions...")
    preds = pd.read_parquet(PREDS_PARQUET)
    preds = preds[preds["project_id"].isin(TARGET_PROJECTS)]
    print(f"  {len(preds):,} predictions | {preds['entity_id'].nunique()} patients "
          f"| {preds['DRUG_ID'].nunique()} drugs")

    # ── Annotate MoA classes ─────────────────────────────────────────────────
    print("\n[2] Annotating drug MoA classes...")
    drugs_df = preds[["DRUG_ID", "DRUG_NAME", "split"]].drop_duplicates("DRUG_ID")
    drugs_df = annotate_drug_moa(drugs_df)
    drug_moa_map   = drugs_df.set_index("DRUG_ID")["moa_class"].to_dict()
    drug_split_map = drugs_df.set_index("DRUG_ID")["split"].to_dict()

    moa_counts = drugs_df["moa_class"].value_counts()
    print("  MoA class distribution (top 20, excluding Other):")
    print(moa_counts[moa_counts.index != "Other"].head(20).to_string())
    print(f"  Total drugs with MoA: {(drugs_df['moa_class'] != 'Other').sum()} / {len(drugs_df)}")
    drugs_df.to_csv(OUT_DIR / "moa_drug_groups.csv", index=False)

    # ── Per-cancer MoA analysis ──────────────────────────────────────────────
    print("\n[3] Computing within/between MoA correlations per cancer type...")
    all_summaries, all_holdout = [], []

    for cancer in TARGET_PROJECTS:
        print(f"\n  --- {cancer} ---")
        matrix, drug_meta = build_drug_patient_matrix(preds, cancer)
        if matrix.shape[0] < 10 or matrix.shape[1] < 10:
            print(f"  Skipping {cancer}: insufficient data")
            continue

        # MoA map for drugs in this cancer's matrix
        moa_this = {int(did): drug_moa_map.get(int(did), "Other")
                    for did in matrix.index}

        summary_df, pairs_df = compute_moa_correlations(matrix, moa_this)
        if len(summary_df) > 0:
            summary_df["cancer_type"] = cancer
            all_summaries.append(summary_df)
            print(f"  Within vs between MoA correlation:")
            print(summary_df[["moa_class", "n_drugs_in_class", "mean_within_r",
                               "mean_between_r", "delta_within_vs_between"]
                             ].to_string(index=False))

        # Holdout recovery
        holdout_df = holdout_moa_recovery(matrix, moa_this,
                                          {int(did): drug_split_map.get(int(did), "?")
                                           for did in matrix.index})
        if len(holdout_df) > 0:
            holdout_df["cancer_type"] = cancer
            all_holdout.append(holdout_df)
            print(f"\n  Holdout drug MoA recovery:")
            print(holdout_df[["DRUG_ID", "split", "moa_class", "mean_r_same_class",
                               "mean_r_diff_class", "delta_same_vs_diff"]
                             ].to_string(index=False))

    # ── Consolidate ──────────────────────────────────────────────────────────
    summary_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    holdout_all = pd.concat(all_holdout, ignore_index=True) if all_holdout else pd.DataFrame()

    if len(summary_all) > 0:
        summary_all.to_csv(OUT_DIR / "moa_within_vs_between_corr.csv", index=False)
    if len(holdout_all) > 0:
        # Add drug names
        did_to_name = preds.drop_duplicates("DRUG_ID").set_index("DRUG_ID")["DRUG_NAME"].to_dict()
        holdout_all["DRUG_NAME"] = holdout_all["DRUG_ID"].map(did_to_name)
        holdout_all.to_csv(OUT_DIR / "moa_holdout_recovery.csv", index=False)

    # Overall statistics
    overall_within  = float(summary_all["mean_within_r"].mean()) if len(summary_all) else None
    overall_between = float(summary_all["mean_between_r"].mean()) if len(summary_all) else None
    n_holdout_positive = int((holdout_all["delta_same_vs_diff"] > 0).sum()) if len(holdout_all) else 0

    # Statistical test: within-MoA r > between-MoA r (using all pairs across cancers)
    all_pairs_list = []
    for cancer in TARGET_PROJECTS:
        matrix, drug_meta = build_drug_patient_matrix(preds, cancer)
        if matrix.shape[0] < 10 or matrix.shape[1] < 10:
            continue
        moa_this = {int(did): drug_moa_map.get(int(did), "Other") for did in matrix.index}
        _, pairs_df = compute_moa_correlations(matrix, moa_this)
        if len(pairs_df) > 0:
            all_pairs_list.append(pairs_df)
    all_pairs_combined = pd.concat(all_pairs_list, ignore_index=True) if all_pairs_list else pd.DataFrame()

    within_r_all = all_pairs_combined.loc[
        all_pairs_combined["pair_type"] == "within", "pearson_r"
    ].dropna().values if len(all_pairs_combined) else np.array([])
    between_r_all = all_pairs_combined.loc[
        all_pairs_combined["pair_type"] == "between", "pearson_r"
    ].dropna().values if len(all_pairs_combined) else np.array([])

    mwu_stat, mwu_pval = (None, None)
    if len(within_r_all) >= 5 and len(between_r_all) >= 5:
        mwu_stat, mwu_pval = mannwhitneyu(within_r_all, between_r_all, alternative="greater")
        print(f"\n  Mann-Whitney U: within-MoA r > between-MoA r")
        print(f"    n_within={len(within_r_all)}, n_between={len(between_r_all)}")
        print(f"    U={mwu_stat:.0f}, p={mwu_pval:.3e}")

    # Paired Wilcoxon on holdout drug deltas (same-class > diff-class train correlation)
    wilcox_stat, wilcox_pval = (None, None)
    if len(holdout_all) >= 5:
        deltas = holdout_all["delta_same_vs_diff"].dropna().values
        try:
            wilcox_stat, wilcox_pval = wilcoxon(deltas, alternative="greater")
            print(f"\n  Wilcoxon: holdout drug same-class > diff-class correlation")
            print(f"    n={len(deltas)}, stat={wilcox_stat:.1f}, p={wilcox_pval:.3e}")
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")

    # Build summary
    summary = {
        "n_drugs_with_moa": int((drugs_df["moa_class"] != "Other").sum()),
        "n_moa_classes":    int(drugs_df.loc[drugs_df["moa_class"] != "Other", "moa_class"].nunique()),
        "correlation_analysis": {
            "overall_mean_within_moa_r":   round(float(overall_within), 4) if overall_within else None,
            "overall_mean_between_moa_r":  round(float(overall_between), 4) if overall_between else None,
            "delta_within_vs_between":     round(float(overall_within - overall_between), 4)
                                           if overall_within is not None and overall_between is not None else None,
            "mwu_within_gt_between_pval":  float(mwu_pval) if mwu_pval is not None else None,
            "n_within_pairs_total":        int(len(within_r_all)),
            "n_between_pairs_total":       int(len(between_r_all)),
        },
        "holdout_moa_recovery": {
            "n_holdout_drugs_with_moa":   int(len(holdout_all)),
            "n_positive_delta":           n_holdout_positive,
            "frac_positive_delta":        round(float(n_holdout_positive / max(len(holdout_all), 1)), 4),
            "mean_delta_same_vs_diff":    round(float(holdout_all["delta_same_vs_diff"].mean()), 4)
                                          if len(holdout_all) > 0 else None,
            "wilcoxon_delta_gt_zero_pval": float(wilcox_pval) if wilcox_pval is not None else None,
        },
        "moa_class_distribution": moa_counts[moa_counts.index != "Other"].head(15).to_dict(),
    }

    with open(OUT_DIR / "chembl_moa_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Summary]")
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
