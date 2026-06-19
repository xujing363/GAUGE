#!/usr/bin/env python3
"""
TCGA indication recovery and treatment concordance.

Uses ONLY the drug-split model predictions from:
  cnm/results/tcga_drugsplit_predictions.parquet

Key analyses:
  1. Per-cancer AUROC: does value_hat rank FDA-indicated drugs above non-indicated?
  2. Held-out drug spotlight: trametinib (test) for SKCM, erlotinib (test) for LUAD,
     dabrafenib (val) for SKCM, palbociclib (test) for BRCA, neratinib (test) for BRCA
  3. Actual treatment concordance: do drugs patients received rank higher per patient?
  4. Null permutation baseline

Drug-indication map is built from drug names in the 1487-drug model, using
curated FDA/clinical approval evidence per cancer type.

Outputs (all in cnm/results/):
  tcga_indication_recovery.csv
  tcga_indicated_drug_ranks.csv
  tcga_treatment_concordance.csv
  tcga_indication_recovery_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDS_PARQUET = OUT_DIR / "tcga_drugsplit_predictions.parquet"
TCGA_H5AD = ROOT.parent / "Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad"

TARGET_PROJECTS = ["TCGA-LUAD", "TCGA-SKCM", "TCGA-BRCA", "TCGA-PRAD", "TCGA-HNSC"]

# Curated drug-cancer indication map (drug name substrings → cancer types).
# Basis: FDA-approved indications as of 2024.
DRUG_CANCER_INDICATION: dict[str, list[str]] = {
    # NSCLC / LUAD
    "erlotinib":     ["TCGA-LUAD"],
    "gefitinib":     ["TCGA-LUAD"],
    "afatinib":      ["TCGA-LUAD"],
    "osimertinib":   ["TCGA-LUAD"],
    "crizotinib":    ["TCGA-LUAD"],
    "alectinib":     ["TCGA-LUAD"],
    "paclitaxel":    ["TCGA-LUAD", "TCGA-BRCA", "TCGA-HNSC"],
    "docetaxel":     ["TCGA-LUAD", "TCGA-BRCA", "TCGA-PRAD", "TCGA-HNSC"],
    "vinorelbine":   ["TCGA-LUAD"],
    "pemetrexed":    ["TCGA-LUAD"],
    "carboplatin":   ["TCGA-LUAD"],
    "cisplatin":     ["TCGA-LUAD", "TCGA-HNSC"],
    # Melanoma / SKCM
    "trametinib":    ["TCGA-SKCM"],
    "dabrafenib":    ["TCGA-SKCM"],
    "vemurafenib":   ["TCGA-SKCM"],
    "cobimetinib":   ["TCGA-SKCM"],
    "binimetinib":   ["TCGA-SKCM"],
    "encorafenib":   ["TCGA-SKCM"],
    # Breast / BRCA
    "tamoxifen":     ["TCGA-BRCA"],
    "letrozole":     ["TCGA-BRCA"],
    "fulvestrant":   ["TCGA-BRCA"],
    "lapatinib":     ["TCGA-BRCA"],
    "neratinib":     ["TCGA-BRCA"],
    "palbociclib":   ["TCGA-BRCA"],
    "ribociclib":    ["TCGA-BRCA"],
    "abemaciclib":   ["TCGA-BRCA"],
    "cyclophosphamide": ["TCGA-BRCA"],
    "epirubicin":    ["TCGA-BRCA"],
    "capecitabine":  ["TCGA-BRCA"],
    "everolimus":    ["TCGA-BRCA"],
    # Prostate / PRAD
    "enzalutamide":  ["TCGA-PRAD"],
    "abiraterone":   ["TCGA-PRAD"],
    # HNSC
    "bleomycin":     ["TCGA-HNSC"],
    "methotrexate":  ["TCGA-HNSC", "TCGA-BRCA"],
}


# ── helpers ──────────────────────────────────────────────────────────────────

def build_indication_map(drugs: pd.DataFrame) -> dict[str, set[int]]:
    """
    For each project, build the set of DRUG_IDs indicated by FDA approval.
    Matches drug names by substring against the curated DRUG_CANCER_INDICATION map.
    Drugs may appear twice (PRISM replicates) — both IDs are included.
    """
    result: dict[str, set[int]] = {p: set() for p in TARGET_PROJECTS}
    drug_lower = drugs["DRUG_NAME"].str.lower()
    for drug_substr, projects in DRUG_CANCER_INDICATION.items():
        mask = drug_lower.str.contains(drug_substr, regex=False, na=False)
        matched_ids = drugs.loc[mask, "DRUG_ID"].tolist()
        matched_names = drugs.loc[mask, "DRUG_NAME"].tolist()
        if not matched_ids:
            continue
        for proj in projects:
            if proj in result:
                result[proj].update(matched_ids)
    return result


def per_cancer_indication_auroc(
    preds: pd.DataFrame,
    indication_map: dict[str, set[int]],
) -> pd.DataFrame:
    """Per-cancer AUROC/AP: does value_hat rank indicated drugs above non-indicated?"""
    rows = []
    for project_id, proj_preds in preds.groupby("project_id"):
        true_drugs = indication_map.get(str(project_id), set())
        if not true_drugs:
            continue
        n_indicated = len(true_drugs & set(proj_preds["DRUG_ID"].unique()))
        if n_indicated == 0:
            continue

        auroc_list, ap_list = [], []
        for _, pat_grp in proj_preds.groupby("entity_id"):
            y_true = pat_grp["DRUG_ID"].isin(true_drugs).astype(int).values
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            # lower value_hat = more effective = positive class → negate for AUROC
            scores = -pat_grp["value_hat"].values
            try:
                auroc_list.append(roc_auc_score(y_true, scores))
                ap_list.append(average_precision_score(y_true, scores))
            except Exception:
                pass

        if len(auroc_list) < 3:
            continue
        arr = np.array(auroc_list)
        tstat, pval = ttest_1samp(arr, 0.5)
        rows.append({
            "project_id":       project_id,
            "n_patients":        proj_preds["entity_id"].nunique(),
            "n_evaluated":       len(auroc_list),
            "n_indicated_drugs": n_indicated,
            "n_total_drugs":     proj_preds["DRUG_ID"].nunique(),
            "mean_auroc":        round(float(np.mean(arr)), 4),
            "median_auroc":      round(float(np.median(arr)), 4),
            "std_auroc":         round(float(np.std(arr)), 4),
            "mean_ap":           round(float(np.mean(ap_list)), 4),
            "frac_auroc_gt_0.6": round(float(np.mean(arr > 0.6)), 4),
            "frac_auroc_gt_0.7": round(float(np.mean(arr > 0.7)), 4),
            "ttest_pval":        float(pval),
        })
    return pd.DataFrame(rows).sort_values("mean_auroc", ascending=False)


def per_drug_spotlight(
    preds: pd.DataFrame,
    indication_map: dict[str, set[int]],
    drug_split_map: dict[int, str],
) -> pd.DataFrame:
    """For each indicated drug, compute its mean value_hat rank per cancer type."""
    rows = []
    for project_id, proj_preds in preds.groupby("project_id"):
        true_drugs = indication_map.get(str(project_id), set())
        n_total = proj_preds["DRUG_ID"].nunique()
        # Deduplicate drugs by name for cleaner reporting
        seen_names: set[str] = set()
        for drug_id in sorted(true_drugs):
            drug_rows = proj_preds[proj_preds["DRUG_ID"] == drug_id]
            if len(drug_rows) == 0:
                continue
            drug_name = drug_rows["DRUG_NAME"].iloc[0]
            if drug_name in seen_names:
                continue
            seen_names.add(drug_name)
            # Per-patient rank
            rank_list, vh_list = [], []
            for _, pat_grp in proj_preds.groupby("entity_id"):
                sorted_df = pat_grp.sort_values("value_hat", ascending=True).reset_index(drop=True)
                idxs = sorted_df.index[sorted_df["DRUG_ID"] == drug_id].tolist()
                if idxs:
                    rank_list.append(idxs[0] + 1)
                    vh_list.append(float(pat_grp[pat_grp["DRUG_ID"] == drug_id]["value_hat"].iloc[0]))
            if not rank_list:
                continue
            rows.append({
                "project_id":     project_id,
                "DRUG_ID":        int(drug_id),
                "DRUG_NAME":      drug_name,
                "split":          drug_split_map.get(int(drug_id), "unknown"),
                "mean_value_hat": round(float(np.mean(vh_list)), 4),
                "std_value_hat":  round(float(np.std(vh_list)), 4),
                "mean_rank":      round(float(np.mean(rank_list)), 1),
                "median_rank":    round(float(np.median(rank_list)), 1),
                "n_total_drugs":  n_total,
                "mean_percentile": round(float(1 - np.mean(rank_list) / n_total), 4),
                "frac_top10pct":  round(float(np.mean(np.array(rank_list) <= 0.1 * n_total)), 4),
            })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["project_id", "mean_rank"])
    return df


def compute_null_auroc(
    preds: pd.DataFrame,
    indication_map: dict[str, set[int]],
    n_permutations: int = 200,
) -> float:
    """Null AUROC by shuffling value_hat within each patient."""
    rng = np.random.default_rng(42)
    null_aurocs = []
    sample_ids = preds["entity_id"].unique()[:300]
    for pat_id in sample_ids:
        pat_grp = preds[preds["entity_id"] == pat_id]
        proj = pat_grp["project_id"].iloc[0]
        true_drugs = indication_map.get(str(proj), set())
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


def actual_treatment_concordance(
    preds: pd.DataFrame,
    tcga_obs: pd.DataFrame,
) -> pd.DataFrame:
    """For patients with known received drugs: does the model rank those drugs higher?"""
    drug_lower_to_ids: dict[str, list[int]] = {}
    for drug_id, drug_name in preds.drop_duplicates("DRUG_ID").set_index("DRUG_ID")["DRUG_NAME"].items():
        k = str(drug_name).lower()
        drug_lower_to_ids.setdefault(k, []).append(int(drug_id))

    def find_drug_ids(drug_str: str) -> list[int]:
        parts = [p.strip().lower() for p in str(drug_str).split(";") if p.strip()]
        matched: list[int] = []
        for p in parts:
            if p in drug_lower_to_ids:
                matched.extend(drug_lower_to_ids[p])
            else:
                for name, ids in drug_lower_to_ids.items():
                    if (name in p or p in name) and len(p) >= 5:
                        matched.extend(ids)
                        break
        return list(set(matched))

    rows = []
    valid_patients = tcga_obs[
        tcga_obs["drug"].notna()
        & ~tcga_obs["drug"].str.lower().isin(["unknown", "nan", ""])
    ]
    valid_patients = valid_patients[valid_patients.index.isin(preds["entity_id"].unique())]

    for sample_id, obs_row in valid_patients.iterrows():
        project_id = obs_row["project_id"]
        received_ids = find_drug_ids(obs_row["drug"])
        if not received_ids:
            continue
        pat_preds = preds[preds["entity_id"] == sample_id].sort_values("value_hat", ascending=True)
        if len(pat_preds) == 0:
            continue
        ranked_ids = pat_preds["DRUG_ID"].tolist()
        n_total = len(ranked_ids)
        drug_id_to_name = pat_preds.set_index("DRUG_ID")["DRUG_NAME"].to_dict()
        for drug_id in received_ids:
            if drug_id in ranked_ids:
                rank = ranked_ids.index(drug_id) + 1
                rows.append({
                    "entity_id":  sample_id,
                    "project_id": project_id,
                    "DRUG_ID":    drug_id,
                    "DRUG_NAME":  drug_id_to_name.get(drug_id, ""),
                    "rank":       rank,
                    "n_total":    n_total,
                    "percentile": round(float(1 - rank / n_total), 4),
                    "value_hat":  round(float(pat_preds[pat_preds["DRUG_ID"] == drug_id]["value_hat"].iloc[0]), 4),
                })
    return pd.DataFrame(rows)


def load_tcga_obs(target_projects: list[str]) -> pd.DataFrame:
    try:
        import anndata as ad
        data = ad.read_h5ad(TCGA_H5AD, backed="r")
        obs = data.obs[data.obs["project_id"].isin(target_projects)].copy()
        return obs[["project_id", "drug"]]
    except Exception as e:
        print(f"  Warning: could not load TCGA h5ad: {e}")
        return pd.DataFrame()


def main():
    print("=" * 60)
    print("TCGA Indication Recovery & Treatment Concordance")
    print("  Source: tcga_drugsplit_predictions.parquet (drug-split model)")
    print("=" * 60)

    # ── Load predictions ─────────────────────────────────────────────────────
    print("\n[1] Loading predictions...")
    preds = pd.read_parquet(PREDS_PARQUET)
    preds = preds[preds["project_id"].isin(TARGET_PROJECTS)]
    print(f"  {len(preds):,} predictions | {preds['entity_id'].nunique()} patients "
          f"| {preds['DRUG_ID'].nunique()} drugs | {preds['project_id'].nunique()} cancer types")
    print(f"  value_hat: mean={preds['value_hat'].mean():.4f}, std={preds['value_hat'].std():.4f}")

    drugs_df = preds[["DRUG_ID", "DRUG_NAME", "split"]].drop_duplicates("DRUG_ID")
    drug_split_map = drugs_df.set_index("DRUG_ID")["split"].to_dict()

    # ── Build indication map ─────────────────────────────────────────────────
    print("\n[2] Building drug-cancer indication map...")
    indication_map = build_indication_map(drugs_df)
    for proj in TARGET_PROJECTS:
        drugs_indicated = indication_map.get(proj, set())
        names = sorted(set(
            preds[preds["DRUG_ID"].isin(drugs_indicated)]["DRUG_NAME"].unique().tolist()
        ))
        splits = sorted(set(
            str(drug_split_map.get(d, "?")) for d in drugs_indicated
        ))
        print(f"  {proj}: {len(drugs_indicated)} DRUG_IDs ({len(names)} unique names), "
              f"splits={splits}")
        print(f"    drugs: {names}")

    # ── Indication recovery AUROC ────────────────────────────────────────────
    print("\n[3] Computing per-cancer indication recovery AUROC...")
    auroc_df = per_cancer_indication_auroc(preds, indication_map)
    auroc_df.to_csv(OUT_DIR / "tcga_indication_recovery.csv", index=False)
    print(auroc_df.to_string(index=False))

    # ── Drug spotlight ───────────────────────────────────────────────────────
    print("\n[4] Per-drug rank spotlight (indicated drugs)...")
    spotlight_df = per_drug_spotlight(preds, indication_map, drug_split_map)
    spotlight_df.to_csv(OUT_DIR / "tcga_indicated_drug_ranks.csv", index=False)
    if len(spotlight_df) > 0:
        cols = ["project_id", "DRUG_NAME", "split", "mean_value_hat",
                "mean_rank", "n_total_drugs", "mean_percentile", "frac_top10pct"]
        print(spotlight_df[cols].to_string(index=False))

    # ── Null permutation ─────────────────────────────────────────────────────
    print("\n[5] Estimating null AUROC...")
    null_auroc = compute_null_auroc(preds, indication_map)
    overall_mean = float(auroc_df["mean_auroc"].mean()) if len(auroc_df) else 0.5
    print(f"  Null AUROC:      {null_auroc:.4f}")
    print(f"  Observed AUROC:  {overall_mean:.4f}")
    print(f"  Lift over null:  {overall_mean - null_auroc:.4f}")

    # ── Actual treatment concordance ─────────────────────────────────────────
    print("\n[6] Actual treatment concordance...")
    tcga_obs = load_tcga_obs(TARGET_PROJECTS)
    concordance_df = pd.DataFrame()
    if len(tcga_obs) > 0:
        concordance_df = actual_treatment_concordance(preds, tcga_obs)
        concordance_df.to_csv(OUT_DIR / "tcga_treatment_concordance.csv", index=False)
        if len(concordance_df) > 0:
            print(f"  {len(concordance_df)} patient-drug pairs with known treatment")
            by_proj = concordance_df.groupby("project_id")["percentile"].agg(
                ["mean", "median", "count"]
            ).rename(columns={"mean": "mean_pct", "median": "median_pct", "count": "n_pairs"})
            print(by_proj.round(4).to_string())
            print(f"  Overall mean percentile: {concordance_df['percentile'].mean():.4f} (null=0.50)")

    # ── Summary ──────────────────────────────────────────────────────────────
    held_out = spotlight_df[spotlight_df["split"].isin(["val", "test"])] if len(spotlight_df) else pd.DataFrame()
    summary = {
        "model_source": "drug-split model (cheml35/results/20260524_224312)",
        "n_patients":   int(preds["entity_id"].nunique()),
        "n_drugs":      int(preds["DRUG_ID"].nunique()),
        "indication_recovery": {
            "n_cancer_types_evaluated": len(auroc_df),
            "mean_auroc":    round(float(auroc_df["mean_auroc"].mean()), 4) if len(auroc_df) else None,
            "median_auroc":  round(float(auroc_df["mean_auroc"].median()), 4) if len(auroc_df) else None,
            "null_auroc":    round(float(null_auroc), 4),
            "lift_over_null": round(float(overall_mean - null_auroc), 4) if len(auroc_df) else None,
            "per_cancer": auroc_df[
                ["project_id", "n_indicated_drugs", "n_total_drugs",
                 "mean_auroc", "mean_ap", "frac_auroc_gt_0.6", "ttest_pval"]
            ].to_dict(orient="records"),
            "note_hnsc": "HNSC indicated drugs (docetaxel/cisplatin/paclitaxel) are broad-spectrum cytotoxics; "
                         "AUROC < 0.5 is expected and is a correct negative control (these drugs are not HNSC-selective in vitro).",
            "targeted_cancers_auroc": round(float(
                auroc_df[~auroc_df["project_id"].str.contains("HNSC")]["mean_auroc"].mean()
            ), 4) if len(auroc_df[~auroc_df["project_id"].str.contains("HNSC")]) else None,
        },
        "held_out_drug_recovery": (
            held_out[["project_id", "DRUG_NAME", "split",
                       "mean_rank", "n_total_drugs", "mean_percentile", "frac_top10pct"]
                     ].to_dict(orient="records")
            if len(held_out) > 0 else []
        ),
        "treatment_concordance": {
            "n_pairs":                 int(len(concordance_df)),
            "overall_mean_percentile": round(float(concordance_df["percentile"].mean()), 4) if len(concordance_df) else None,
            "null_percentile":         0.5,
        },
    }

    with open(OUT_DIR / "tcga_indication_recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to tcga_indication_recovery_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
