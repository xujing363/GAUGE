#!/usr/bin/env python3
"""
Publication v5 — Unsupervised Baseline Comparison.

Compares GAUGE profile_complementarity against three classes of published
unsupervised drug combination prediction methods (none trained on combination data):

  BL-1  Chemical structure dissimilarity (Tanimoto on Morgan fingerprints)
        Published basis: Preuer et al. (2018), O'Neil et al. (2016).
        Hypothesis: structurally dissimilar drugs target different mechanisms
        → more likely to be synergistic (complementary coverage).
        Score: 1 − Tanimoto(fingerprint_A, fingerprint_B)

  BL-2  GDSC global drug sensitivity anti-correlation
        Published basis: He et al. (2018) sensitivity-based combination prediction.
        Hypothesis: drugs with anti-correlated response profiles across all cell lines
        attack different cellular dependencies → Bliss synergy.
        Score: 1 − Pearson(IC50_A_all_cells, IC50_B_all_cells)

  BL-3  GDSC cancer-type drug sensitivity anti-correlation
        Same as BL-2 but restricted to cancer-type-matched GDSC cells.
        This is the naive version of our approach: same cell subset, raw IC50.
        Score: 1 − Pearson(IC50_A_cancer_cells, IC50_B_cancer_cells)

  BL-4  Target pathway diversity (rule-based)
        Published basis: Bayat Mokhtari et al. (2017) pathway-based combination.
        Hypothesis: drugs targeting different signalling pathways achieve orthogonal
        inhibition → more synergistic. Binary score: 1 if pathways differ, 0 if same.
        Source: GDSC screened_compounds TARGET_PATHWAY annotation.

These baselines represent the standard approaches used before deep learning:
pure chemistry (BL-1), population-level pharmacology (BL-2/3), and pathway logic (BL-4).
GAUGE profile_complementarity uses a KG-trained neural network to compute CANCER-TYPE-
SPECIFIC activity profiles — the comparison isolates the value of the neural model.

Run: python publication_v5_complementarity_multicancer/scripts/run_v5_baseline_comparison.py
"""
from __future__ import annotations
import json, re, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add KG modules to path
sys.path.insert(0, "/mnt/raid5/xujing/KG/GAUGE")
sys.path.insert(0, "/mnt/raid5/xujing/KG")

MULTI_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MULTI_ROOT / "publication_v5_complementarity_multicancer" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GDSC_DIR = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/GDSC")
ARTIFACTS_PKL = Path("/mnt/raid5/xujing/KG/Combined/results/combined_melanoma_v1_20260524_130336/artifacts.pkl")
SEED = 42
rng = np.random.RandomState(SEED)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def make_pair_key(a: str, b: str) -> str:
    return "||".join(sorted([norm(a), norm(b)]))


def tanimoto_binary(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Tanimoto (Jaccard) similarity for binary fingerprints."""
    a = fp_a > 0
    b = fp_b > 0
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 0.0
    return float(intersection / union)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())


def compute_metrics(y_true, y_score, y_cont, label, n_boot=3000, n_perm=5000):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_cont = np.asarray(y_cont, dtype=float)
    n = len(y_true)
    if y_true.sum() == 0 or y_true.sum() == n:
        return {"label": label, "n": n, "error": "no variance"}
    auc_val = roc_auc_score(y_true, y_score)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if 0 < yt.sum() < len(yt):
            aucs.append(roc_auc_score(yt, ys))
    auc_ci = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (0.0, 0.0)
    null_aucs = [roc_auc_score(y_true, rng.permutation(y_score)) for _ in range(n_perm)]
    p_auc = max(1 / n_perm, float((np.array(null_aucs) >= auc_val).mean()))
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    prauc = pr_auc(rec, prec)
    rho, p_rho = spearmanr(y_score, y_cont)
    return {
        "label": label, "n": int(n), "pos_rate": float(y_true.mean()),
        "auc": float(auc_val), "auc_ci": list(auc_ci), "auc_perm_p": float(p_auc),
        "prauc": float(prauc), "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "p5": float(precision_at_k(y_true, y_score, 5)),
        "p10": float(precision_at_k(y_true, y_score, 10)),
        "p15": float(precision_at_k(y_true, y_score, 15)),
        "p25": float(precision_at_k(y_true, y_score, 25)),
    }


def print_metrics(m, indent="  "):
    if "error" in m:
        print(f"{indent}[{m['label']}] ERROR: {m['error']}")
        return
    ci = m.get("auc_ci", [0, 0])
    print(f"{indent}[{m['label']}] n={m['n']} pos={m['pos_rate']:.3f}")
    print(f"{indent}  AUC={m['auc']:.4f} (95%CI {ci[0]:.3f}–{ci[1]:.3f}, p={m['auc_perm_p']:.4f})")
    print(f"{indent}  PRAUC={m['prauc']:.4f}  Spear={m['spearman_rho']:+.4f}(p={m['spearman_p']:.3f})")
    print(f"{indent}  P@5={m['p5']:.2f}  P@10={m['p10']:.2f}  P@15={m['p15']:.2f}  P@25={m['p25']:.2f}")


# ── Load drug data ────────────────────────────────────────────────────────────

print("=" * 70)
print("Loading drug data...")

# Fingerprints from model artifacts
import pickle
with open(ARTIFACTS_PKL, "rb") as f:
    artifacts = pickle.load(f)
drug_table = artifacts.drug_table
norm_to_drug = {}
for _, row in drug_table.iterrows():
    norm_to_drug[norm(row["DRUG_NAME"])] = row
print(f"  Drug table: {len(drug_table)} drugs, fingerprint dim={len(drug_table.iloc[0]['fingerprint'])}")

# GDSC2 IC50 profiles
print("Loading GDSC2 dose-response...")
gdsc2 = pd.read_excel(
    GDSC_DIR / "GDSC2_fitted_dose_response_27Oct23.xlsx",
    usecols=["DRUG_ID", "DRUG_NAME", "CANCER_TYPE", "SANGER_MODEL_ID", "LN_IC50"],
)
gdsc2 = gdsc2.dropna(subset=["LN_IC50"])
# Global pivot: drug × cell (mean over replicates)
gdsc_global = (gdsc2.groupby(["DRUG_NAME", "SANGER_MODEL_ID"])["LN_IC50"]
               .mean().unstack(fill_value=np.nan))
print(f"  GDSC2 global matrix: {gdsc_global.shape[0]} drugs × {gdsc_global.shape[1]} cells")

# GDSC target pathway
gdsc_compounds = pd.read_csv(GDSC_DIR / "screened_compounds_rel_8.5.csv")
drug_pathway = {}
for _, row in gdsc_compounds.iterrows():
    drug_pathway[norm(str(row["DRUG_NAME"]))] = str(row.get("TARGET_PATHWAY", "Unknown"))
print(f"  GDSC target pathways loaded: {len(drug_pathway)} drugs")

# ── Cancer type configs (matching v5) ─────────────────────────────────────────

CANCER_CONFIGS = {
    "Melanoma": {
        "gdsc_cancer": "Melanoma",
        "nci_matched": "nci_matched_melanoma.csv",
    },
    "NSCLC": {
        "gdsc_cancer": "Non-Small Cell Lung Carcinoma",
        "nci_matched": "nci_matched_nsclc.csv",
    },
    "Breast": {
        "gdsc_cancer": "Breast Carcinoma",
        "nci_matched": "nci_matched_breast.csv",
    },
    "Ovarian": {
        "gdsc_cancer": "Ovarian Carcinoma",
        "nci_matched": "nci_matched_ovarian.csv",
    },
}

# ── Process each cancer type ──────────────────────────────────────────────────

all_results = {}
all_pooled: dict[str, list] = {
    "y_true": [], "y_cont": [],
    "bl1_tanimoto": [], "bl2_gdsc_global": [], "bl3_gdsc_cancer": [], "bl4_pathway": [],
    "GAUGE_pcp": [], "GAUGE_complementarity": [],
}

for cancer, cfg in CANCER_CONFIGS.items():
    print(f"\n{'=' * 70}")
    print(f"  CANCER: {cancer}")
    print(f"{'=' * 70}")

    # Load pre-computed NCI-matched pairs (from v5)
    nci_df = pd.read_csv(RESULTS_DIR / cfg["nci_matched"])
    print(f"  Loaded {len(nci_df)} NCI-matched pairs")

    # Cancer-type GDSC cell subset
    cancer_cells = gdsc2[gdsc2["CANCER_TYPE"] == cfg["gdsc_cancer"]]["SANGER_MODEL_ID"].unique()
    gdsc_cancer = (gdsc2[gdsc2["CANCER_TYPE"] == cfg["gdsc_cancer"]]
                   .groupby(["DRUG_NAME", "SANGER_MODEL_ID"])["LN_IC50"]
                   .mean().unstack(fill_value=np.nan))
    print(f"  GDSC cancer cells: {len(cancer_cells)}; cancer matrix: {gdsc_cancer.shape}")

    rows = []
    n_fp_miss = 0
    n_gdsc_miss = 0

    for _, pair in nci_df.iterrows():
        da = pair["drug_A_name"]
        db = pair["drug_B_name"]
        nkey = norm(da)
        nkeyb = norm(db)

        # ── BL-1: Chemical Tanimoto dissimilarity ──────────────────────────────
        if nkey in norm_to_drug and nkeyb in norm_to_drug:
            fp_a = norm_to_drug[nkey]["fingerprint"]
            fp_b = norm_to_drug[nkeyb]["fingerprint"]
            bl1 = 1.0 - tanimoto_binary(np.asarray(fp_a), np.asarray(fp_b))
        else:
            bl1 = np.nan
            n_fp_miss += 1

        # ── BL-2: GDSC global anti-correlation ────────────────────────────────
        # Use DRUG_NAME index in gdsc_global (may have multiple GDSC names)
        def _get_profile(df_pivot, name_norm):
            # try exact norm match first
            for row_name in df_pivot.index:
                if norm(row_name) == name_norm:
                    return df_pivot.loc[row_name].values
            return None

        prof_a_global = _get_profile(gdsc_global, nkey)
        prof_b_global = _get_profile(gdsc_global, nkeyb)
        if prof_a_global is not None and prof_b_global is not None:
            mask = ~(np.isnan(prof_a_global) | np.isnan(prof_b_global))
            if mask.sum() >= 10 and np.std(prof_a_global[mask]) > 1e-9 and np.std(prof_b_global[mask]) > 1e-9:
                r_global = float(pearsonr(prof_a_global[mask], prof_b_global[mask])[0])
                bl2 = 1.0 - r_global
            else:
                bl2 = np.nan
        else:
            bl2 = np.nan
            n_gdsc_miss += 1

        # ── BL-3: GDSC cancer-type anti-correlation ───────────────────────────
        prof_a_cancer = _get_profile(gdsc_cancer, nkey)
        prof_b_cancer = _get_profile(gdsc_cancer, nkeyb)
        if prof_a_cancer is not None and prof_b_cancer is not None:
            mask = ~(np.isnan(prof_a_cancer) | np.isnan(prof_b_cancer))
            if mask.sum() >= 5 and np.std(prof_a_cancer[mask]) > 1e-9 and np.std(prof_b_cancer[mask]) > 1e-9:
                r_cancer = float(pearsonr(prof_a_cancer[mask], prof_b_cancer[mask])[0])
                bl3 = 1.0 - r_cancer
            else:
                bl3 = np.nan
        else:
            bl3 = np.nan

        # ── BL-4: Target pathway diversity ────────────────────────────────────
        pw_a = drug_pathway.get(nkey, "Unknown")
        pw_b = drug_pathway.get(nkeyb, "Unknown")
        if pw_a != "Unknown" and pw_b != "Unknown":
            bl4 = float(pw_a != pw_b)  # 1 = different pathways (predicted synergistic)
        else:
            bl4 = np.nan

        rows.append({
            "drug_A": da, "drug_B": db,
            "is_syn": int(pair["is_syn"]), "nci_score": float(pair["nci_score"]),
            "bl1_tanimoto_dissim": bl1,
            "bl2_gdsc_global_anticorr": bl2,
            "bl3_gdsc_cancer_anticorr": bl3,
            "bl4_pathway_diversity": bl4,
            "GAUGE_pcp": float(pair["pcp"]),
            "GAUGE_complementarity": float(pair["complementarity"]),
        })

    df_bl = pd.DataFrame(rows)
    df_bl.to_csv(RESULTS_DIR / f"baseline_comparison_{cancer.lower()}.csv", index=False)
    n_pairs = len(df_bl)
    print(f"  Fingerprint misses: {n_fp_miss}, GDSC profile misses: {n_gdsc_miss}")
    print(f"  BL-1 coverage: {df_bl['bl1_tanimoto_dissim'].notna().sum()}/{n_pairs}")
    print(f"  BL-2 coverage: {df_bl['bl2_gdsc_global_anticorr'].notna().sum()}/{n_pairs}")
    print(f"  BL-3 coverage: {df_bl['bl3_gdsc_cancer_anticorr'].notna().sum()}/{n_pairs}")
    print(f"  BL-4 coverage: {df_bl['bl4_pathway_diversity'].notna().sum()}/{n_pairs}")

    y_true = df_bl["is_syn"].values
    y_cont = df_bl["nci_score"].values

    cancer_metrics = {}
    for bl_col, bl_label in [
        ("bl1_tanimoto_dissim", "BL-1: Chemical dissimilarity (Tanimoto)"),
        ("bl2_gdsc_global_anticorr", "BL-2: GDSC global anti-corr (all cells)"),
        ("bl3_gdsc_cancer_anticorr", "BL-3: GDSC cancer anti-corr (cancer cells)"),
        ("bl4_pathway_diversity", "BL-4: Target pathway diversity (rule-based)"),
        ("GAUGE_pcp", "GAUGE pcp (v4, cell co-activity)"),
        ("GAUGE_complementarity", "GAUGE complementarity (v5, NEW)"),
    ]:
        valid = df_bl[bl_col].notna()
        if valid.sum() < 10:
            print(f"  Skipping {bl_col}: too few valid pairs ({valid.sum()})")
            continue
        yt = y_true[valid.values]
        ys = df_bl[bl_col][valid].values
        yc = y_cont[valid.values]
        m = compute_metrics(yt, ys, yc, label=bl_label)
        cancer_metrics[bl_col] = m
        print_metrics(m)

        # Accumulate pooled (only pairs with all scores available for pooled)
        if bl_col.startswith("GAUGE"):
            all_pooled[bl_col.replace("GAUGE_", "GAUGE_")].extend(ys.tolist())
            if bl_col == "GAUGE_pcp":
                all_pooled["y_true"].extend(yt.tolist())
                all_pooled["y_cont"].extend(yc.tolist())

    all_results[cancer] = cancer_metrics

# ── Pooled multi-cancer baseline analysis ─────────────────────────────────────
print(f"\n{'=' * 70}")
print("POOLED (4 cancers, n=476 total per mode)...")
print(f"{'=' * 70}")

# For pooled, rebuild from per-cancer CSVs so we have consistent coverage
pooled_rows = []
for cancer, cfg in CANCER_CONFIGS.items():
    bl_df = pd.read_csv(RESULTS_DIR / f"baseline_comparison_{cancer.lower()}.csv")
    pooled_rows.append(bl_df)
pooled_df = pd.concat(pooled_rows, ignore_index=True)
pooled_df.to_csv(RESULTS_DIR / "baseline_comparison_pooled.csv", index=False)

print(f"  Total pooled rows: {len(pooled_df)}")
y_true_pool = pooled_df["is_syn"].values
y_cont_pool = pooled_df["nci_score"].values

pooled_metrics = {}
for bl_col, bl_label in [
    ("bl1_tanimoto_dissim", "BL-1: Chemical dissimilarity (Tanimoto)"),
    ("bl2_gdsc_global_anticorr", "BL-2: GDSC global anti-corr (all cells)"),
    ("bl3_gdsc_cancer_anticorr", "BL-3: GDSC cancer anti-corr (cancer cells)"),
    ("bl4_pathway_diversity", "BL-4: Target pathway diversity (rule-based)"),
    ("GAUGE_pcp", "GAUGE pcp (v4, cell co-activity)"),
    ("GAUGE_complementarity", "GAUGE complementarity (v5, NEW)"),
]:
    valid = pooled_df[bl_col].notna()
    if valid.sum() < 20:
        continue
    yt = y_true_pool[valid.values]
    ys = pooled_df[bl_col][valid].values
    yc = y_cont_pool[valid.values]
    m = compute_metrics(yt, ys, yc, label=bl_label, n_perm=10000)
    pooled_metrics[bl_col] = m
    print_metrics(m)

all_results["POOLED"] = pooled_metrics

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("SUMMARY TABLE (AUC, p, P@10)")
print(f"{'=' * 70}")

cols = [
    ("bl1_tanimoto_dissim", "BL-1 (Tanimoto)"),
    ("bl2_gdsc_global_anticorr", "BL-2 (GDSC global)"),
    ("bl3_gdsc_cancer_anticorr", "BL-3 (GDSC cancer)"),
    ("bl4_pathway_diversity", "BL-4 (Pathway)"),
    ("GAUGE_pcp", "GAUGE-pcp"),
    ("GAUGE_complementarity", "GAUGE-Compl"),
]

header = f"{'Cancer':<12}" + "".join(f"{c[1]:>18}" for c in cols)
print(header)
print("-" * (12 + 18 * len(cols)))

for cancer_name in list(CANCER_CONFIGS.keys()) + ["POOLED"]:
    cmr = all_results.get(cancer_name, {})
    row = f"{cancer_name:<12}"
    for bl_col, _ in cols:
        m = cmr.get(bl_col, {})
        if not m or "error" in m:
            row += f"{'N/A':>18}"
        else:
            sig = "*" if m["auc_perm_p"] < 0.05 else " "
            row += f"  {m['auc']:.3f}(p={m['auc_perm_p']:.3f}){sig}".rjust(18)
    print(row)

# ── Save JSON ─────────────────────────────────────────────────────────────────

with open(RESULTS_DIR / "baseline_comparison_report.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}")
print("\n=== BASELINE COMPARISON COMPLETE ===")
