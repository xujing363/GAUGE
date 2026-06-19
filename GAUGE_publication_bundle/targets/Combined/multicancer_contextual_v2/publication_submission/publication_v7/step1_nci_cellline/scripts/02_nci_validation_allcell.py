#!/usr/bin/env python3
"""
NCI-ALMANAC Validation — All-Cell-Line Analysis
================================================
Validates GAUGE complementarity scores (computed using ALL cancer-type cell
lines) against the independent NCI-ALMANAC drug-combination synergy dataset.

STRUCTURE OF EVIDENCE CHAIN:
  1. Input:  per-pair complementarity scores from script 01 (all cells).
  2. Outcome: NCI-ALMANAC Bliss synergy score (independent of GDSC training).
  3. Metric:  AUC-ROC (binary synergy threshold: SCORE > 0), permutation p-value.
  4. Comparison: all-cell vs test-only (from existing per-cell CSVs).

KEY SCIENTIFIC CLAIMS TESTED:
  A. Using all GDSC cancer cell lines improves AUC vs test-only (more stable
     drug activity profiles → better discrimination).
  B. The profile complementarity score (v5) outperforms simpler baselines.
  C. Results are consistent across 4 independent cancer types.

OUTPUTS:
  - table1_allcell_per_cancer_auc.csv        (main Table 1: per-cancer AUC)
  - table2_allcell_vs_testonly_comparison.csv (cell-line expansion benefit)
  - table3_allcell_baseline_comparison.csv   (scoring mode comparison)
  - nci_matched_allcell_all_cancers.csv      (raw matched data, all cancers)
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))
NCI_PATH = KG_ROOT / "KG_GAUGE_PublicData/NCI_ALMANAC"

SEED = 42
rng = np.random.RandomState(SEED)

# ── NCI loading helpers ───────────────────────────────────────────────────────
nsc_map = pd.read_csv(NCI_PATH / "nsc_name_map.csv")
nsc_map["norm_name"] = nsc_map["drug_name"].apply(lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower()))
norm_to_nscs: dict[str, list[int]] = {}
for _, r in nsc_map.iterrows():
    norm_to_nscs.setdefault(r["norm_name"], []).append(int(r["nsc"]))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def make_pair_key(a: str, b: str) -> str:
    return "||".join(sorted([norm(a), norm(b)]))


print("Loading NCI-ALMANAC...")
with zipfile.ZipFile(NCI_PATH / "ComboDrugGrowth_Nov2017.zip") as zf:
    with zf.open("ComboDrugGrowth_Nov2017.csv") as f:
        nci_all = pd.read_csv(f, usecols=["PANEL", "NSC1", "NSC2", "SCORE"], low_memory=False)
nci_all = nci_all.dropna(subset=["SCORE"])
nci_all["NSC1"] = nci_all["NSC1"].astype(int)
nci_all["NSC2"] = nci_all["NSC2"].astype(int)
print(f"  {len(nci_all):,} NCI records across {nci_all['PANEL'].nunique()} panels")


def build_nci_lookup(panel: str) -> dict:
    sub = nci_all[nci_all["PANEL"] == panel]
    grouped = sub.groupby(["NSC1", "NSC2"])["SCORE"].mean()
    return {tuple(sorted([int(k[0]), int(k[1])])): float(v) for k, v in grouped.items()}


def match_nci(drug_a: str, drug_b: str, lookup: dict):
    for na in norm_to_nscs.get(norm(drug_a), []):
        for nb in norm_to_nscs.get(norm(drug_b), []):
            k = tuple(sorted([na, nb]))
            if k in lookup:
                return lookup[k]
    return None


# ── Metric computation ────────────────────────────────────────────────────────

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_cont: np.ndarray,
    label: str,
    n_boot: int = 3000,
    n_perm: int = 5000,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_cont = np.asarray(y_cont, dtype=float)
    n = len(y_true)

    if y_true.sum() == 0 or y_true.sum() == n:
        return {"label": label, "n": n, "error": "no_variance"}

    auc_val = roc_auc_score(y_true, y_score)

    # Bootstrap CI
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if 0 < yt.sum() < len(yt):
            aucs.append(roc_auc_score(yt, ys))
    ci_lo, ci_hi = np.percentile(aucs, [2.5, 97.5])

    # Permutation p-value
    null_aucs = [roc_auc_score(y_true, rng.permutation(y_score)) for _ in range(n_perm)]
    p_perm = max(1 / n_perm, float((np.array(null_aucs) >= auc_val).mean()))

    # PRAUC
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    prauc = pr_auc(rec, prec)

    # Spearman with continuous score
    rho, p_rho = spearmanr(y_score, y_cont)

    return {
        "label": label,
        "n": int(n),
        "pos_rate": float(y_true.mean()),
        "auc": float(auc_val),
        "auc_ci_lo": float(ci_lo),
        "auc_ci_hi": float(ci_hi),
        "auc_perm_p": float(p_perm),
        "prauc": float(prauc),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "p5": precision_at_k(y_true, y_score, 5),
        "p10": precision_at_k(y_true, y_score, 10),
        "p15": precision_at_k(y_true, y_score, 15),
        "p25": precision_at_k(y_true, y_score, 25),
    }


def print_metrics(m: dict, indent: str = "  ") -> None:
    if "error" in m:
        print(f"{indent}[{m['label']}] ERROR: {m['error']}")
        return
    print(f"{indent}[{m['label']}] n={m['n']} pos={m['pos_rate']:.3f}")
    print(f"{indent}  AUC={m['auc']:.4f} (95%CI {m['auc_ci_lo']:.3f}–{m['auc_ci_hi']:.3f}) "
          f"perm_p={m['auc_perm_p']:.4f}")
    print(f"{indent}  PRAUC={m['prauc']:.4f}  Spear={m['spearman_rho']:+.4f}(p={m['spearman_p']:.3f})")
    print(f"{indent}  P@10={m['p10']:.2f}  P@25={m['p25']:.2f}")


# ── Cancer configurations ─────────────────────────────────────────────────────
CANCER_CONFIGS = {
    "Melanoma": {
        "nci_panel": "Melanoma",
        "testonly_pc_path": (
            "Melanoma/results/contextual_combined_v2_auto_kg_margin_gate"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
    "NSCLC": {
        "nci_panel": "Non-Small Cell Lung Cancer",
        "testonly_pc_path": (
            "Non_Small_Cell_Lung_Carcinoma/results/contextual_combined_v2_auto_kg_probe_fix1"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
    "Breast": {
        "nci_panel": "Breast Cancer",
        "testonly_pc_path": (
            "Breast_Carcinoma/results/contextual_combined_v2_auto_kg_pubv1_gpu1"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
    "Ovarian": {
        "nci_panel": "Ovarian Cancer",
        "testonly_pc_path": (
            "Ovarian_Carcinoma/results/contextual_combined_v2_auto_kg_pubv1_gpu1"
            "/contextual_combination_predictions_per_cell.csv"
        ),
    },
}


def compute_testonly_scores(pc: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the complementarity score from an existing per-cell CSV."""
    rows = []
    for pair_key, grp in pc.groupby("unordered_pair_key"):
        A = grp["base_score_A"].values
        B = grp["base_score_B"].values
        n_cells = len(A)
        drug_A = grp["drug_A_name"].iloc[0]
        drug_B = grp["drug_B_name"].iloc[0]

        pcp = float(np.median(A * B))
        agg = float(np.median(A)) * float(np.median(B))
        if np.std(A) > 1e-9 and np.std(B) > 1e-9:
            pearson_r = float(pearsonr(A, B)[0])
        else:
            pearson_r = 0.0
        complementarity = pcp * (1.0 - pearson_r)

        rows.append({
            "unordered_pair_key": pair_key,
            "drug_A_name": drug_A,
            "drug_B_name": drug_B,
            "n_cells": n_cells,
            "pcp": pcp,
            "agg_product": agg,
            "pearson_r": pearson_r,
            "complementarity": complementarity,
        })
    return pd.DataFrame(rows)


# ── Per-cancer analysis ───────────────────────────────────────────────────────
all_allcell_rows = []
all_testonly_rows = []
cancer_results = []
table1_rows = []
table2_rows = []

for cancer, cfg in CANCER_CONFIGS.items():
    print("\n" + "=" * 70)
    print(f"Cancer: {cancer}")
    print("=" * 70)

    # Load all-cell pair scores from script 01
    allcell_path = RESULTS_DIR / f"allcell_pair_scores_{cancer.lower()}.csv"
    if not allcell_path.exists():
        print(f"  MISSING: {allcell_path} — run script 01 first")
        continue
    agg_allcell = pd.read_csv(allcell_path)

    # Load test-only per-cell CSV and recompute scores
    testonly_path = ROOT / cfg["testonly_pc_path"]
    pc_testonly = pd.read_csv(testonly_path)
    agg_testonly = compute_testonly_scores(pc_testonly)

    n_allcell_cells = agg_allcell["n_cells"].max()
    n_testonly_cells = agg_testonly["n_cells"].max()
    print(f"  All-cell: {n_allcell_cells} cells | Test-only: {n_testonly_cells} cells")

    # Build NCI lookup
    nci_lookup = build_nci_lookup(cfg["nci_panel"])
    n_nci_bg = len(nci_lookup)
    n_nci_syn = sum(1 for v in nci_lookup.values() if v > 0)
    print(f"  NCI background: {n_nci_syn}/{n_nci_bg} = {n_nci_syn/n_nci_bg:.1%} synergistic")

    # Match all-cell pairs to NCI
    matched_allcell = []
    for _, row in agg_allcell.iterrows():
        nci_score = match_nci(row["drug_A_name"], row["drug_B_name"], nci_lookup)
        if nci_score is not None:
            r = row.to_dict()
            r["nci_score"] = nci_score
            r["is_syn"] = int(nci_score > 0)
            r["cancer"] = cancer
            matched_allcell.append(r)

    # Match test-only pairs to NCI
    matched_testonly = []
    for _, row in agg_testonly.iterrows():
        nci_score = match_nci(row["drug_A_name"], row["drug_B_name"], nci_lookup)
        if nci_score is not None:
            r = row.to_dict()
            r["nci_score"] = nci_score
            r["is_syn"] = int(nci_score > 0)
            r["cancer"] = cancer
            matched_testonly.append(r)

    df_allcell = pd.DataFrame(matched_allcell)
    df_testonly = pd.DataFrame(matched_testonly)

    if len(df_allcell) < 20:
        print(f"  SKIP: only {len(df_allcell)} matched pairs (all-cell)")
        continue

    n_matched = len(df_allcell)
    n_syn = int(df_allcell["is_syn"].sum())
    print(f"  Matched: {n_matched} pairs | Synergistic: {n_syn}/{n_matched} = {n_syn/n_matched:.1%}")

    all_allcell_rows.extend(matched_allcell)
    all_testonly_rows.extend(matched_testonly)

    y_t = df_allcell["is_syn"].values.astype(int)
    y_nci = df_allcell["nci_score"].values.astype(float)

    # ── Scoring mode comparison ────────────────────────────────────────────
    print(f"\n  Scoring mode comparison (all-cell, n={n_matched} pairs):")
    modes = [
        ("complementarity", "GAUGE complementarity (v5, all-cell) — PRIMARY"),
        ("pcp",             "Per-cell product (v4, all-cell)"),
        ("agg_product",     "Aggregate product (baseline)"),
        ("inv_pearson",     "1 − Pearson (mechanism diversity only)"),
    ]
    mode_metrics = {}
    for col, label in modes:
        if col not in df_allcell.columns:
            continue
        ys = df_allcell[col].values.astype(float)
        m = compute_metrics(y_t, ys, y_nci, label)
        print_metrics(m)
        mode_metrics[col] = m

    # ── All-cell vs test-only comparison ──────────────────────────────────
    print(f"\n  All-cell vs Test-only comparison (complementarity score):")
    df_to_common = df_testonly[df_testonly["unordered_pair_key"].isin(df_allcell["unordered_pair_key"])]

    m_allcell = compute_metrics(y_t, df_allcell["complementarity"].values, y_nci, f"{cancer} all-cell")
    print_metrics(m_allcell, indent="    all-cell: ")

    if not df_to_common.empty:
        y_t_to = df_to_common["is_syn"].values.astype(int)
        y_nci_to = df_to_common["nci_score"].values.astype(float)
        m_testonly = compute_metrics(y_t_to, df_to_common["complementarity"].values, y_nci_to, f"{cancer} test-only")
        print_metrics(m_testonly, indent="    test-only: ")
        auc_delta = m_allcell.get("auc", 0) - m_testonly.get("auc", 0)
        print(f"    ΔAUC (all-cell − test-only): {auc_delta:+.4f}")
    else:
        m_testonly = {"error": "no_match"}
        auc_delta = float("nan")

    # ── Shuffle test ────────────────────────────────────────────────────────
    real_c = df_allcell["complementarity"].values
    shuf_c = df_allcell["complementarity_shuf"].values
    diffs_c = real_c - shuf_c
    if len(diffs_c[diffs_c != 0]) >= 5:
        _, p_shuf = wilcoxon(diffs_c, alternative="greater")
        frac_pass = float((real_c > shuf_c).mean())
        print(f"\n  Within-drug shuffle test (complementarity):")
        print(f"    frac real > shuffled = {frac_pass:.3f}, Wilcoxon p = {p_shuf:.4e} "
              f"{'✓ PASS' if p_shuf < 0.05 else '✗ FAIL'}")
    else:
        p_shuf = float("nan")
        frac_pass = float("nan")

    # ── Collect table rows ─────────────────────────────────────────────────
    m = mode_metrics.get("complementarity", {})
    table1_rows.append({
        "cancer": cancer,
        "n_cells_allcell": n_allcell_cells,
        "n_cells_testonly": n_testonly_cells,
        "fold_increase": round(n_allcell_cells / n_testonly_cells, 1) if n_testonly_cells > 0 else float("nan"),
        "n_pairs_matched": n_matched,
        "synergy_rate": round(n_syn / n_matched, 3),
        "auc_allcell": round(m.get("auc", float("nan")), 4),
        "auc_ci": f"{m.get('auc_ci_lo',0):.3f}–{m.get('auc_ci_hi',0):.3f}",
        "auc_perm_p": m.get("auc_perm_p", float("nan")),
        "prauc": round(m.get("prauc", float("nan")), 4),
        "spearman_rho": round(m.get("spearman_rho", float("nan")), 4),
        "shuffle_p": round(p_shuf, 5) if not np.isnan(p_shuf) else float("nan"),
    })

    m_to = m_testonly if "error" not in m_testonly else {}
    table2_rows.append({
        "cancer": cancer,
        "auc_allcell": round(m.get("auc", float("nan")), 4),
        "auc_testonly": round(m_to.get("auc", float("nan")), 4),
        "delta_auc": round(auc_delta, 4),
        "n_cells_allcell": n_allcell_cells,
        "n_cells_testonly": n_testonly_cells,
        "perm_p_allcell": m.get("auc_perm_p", float("nan")),
        "perm_p_testonly": m_to.get("auc_perm_p", float("nan")),
    })

    # Save per-cancer matched data
    df_allcell.to_csv(RESULTS_DIR / f"nci_matched_allcell_{cancer.lower()}.csv", index=False)


# ── Pooled multi-cancer analysis ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("POOLED MULTI-CANCER ANALYSIS")
print("=" * 70)

pooled_allcell = pd.DataFrame(all_allcell_rows)
pooled_testonly = pd.DataFrame(all_testonly_rows)

if not pooled_allcell.empty:
    pooled_allcell.to_csv(RESULTS_DIR / "nci_matched_allcell_all_cancers.csv", index=False)

    print(f"Total evaluations (all-cell): {len(pooled_allcell)}")
    for c in pooled_allcell["cancer"].unique():
        sub = pooled_allcell[pooled_allcell["cancer"] == c]
        print(f"  {c:8s}: n={len(sub)}, syn={int(sub['is_syn'].sum())}/{len(sub)} "
              f"({sub['is_syn'].mean():.1%})")

    y_t_pool = pooled_allcell["is_syn"].values.astype(int)
    y_nci_pool = pooled_allcell["nci_score"].values.astype(float)

    # Pooled scoring modes
    print("\nPooled scoring modes:")
    pooled_mode_metrics = {}
    for col, label in [
        ("complementarity", "GAUGE complementarity (v5, all-cell)"),
        ("pcp", "Per-cell product (v4)"),
        ("agg_product", "Aggregate product"),
        ("inv_pearson", "1 − Pearson"),
    ]:
        if col not in pooled_allcell.columns:
            continue
        ys = pooled_allcell[col].values.astype(float)
        m = compute_metrics(y_t_pool, ys, y_nci_pool, label)
        print_metrics(m)
        pooled_mode_metrics[col] = m

    # Pooled test-only comparison
    if not pooled_testonly.empty:
        y_t_to_pool = pooled_testonly["is_syn"].values.astype(int)
        y_nci_to_pool = pooled_testonly["nci_score"].values.astype(float)
        if "complementarity" in pooled_testonly.columns:
            m_pool_to = compute_metrics(
                y_t_to_pool, pooled_testonly["complementarity"].values, y_nci_to_pool,
                "pooled test-only"
            )
        else:
            m_pool_to = {}
        m_pool_ac = pooled_mode_metrics.get("complementarity", {})
        print(f"\n  Pooled ΔAUC (all-cell − test-only): "
              f"{m_pool_ac.get('auc',0) - m_pool_to.get('auc',0):+.4f}")

    # Save pooled metrics
    pooled_m = pooled_mode_metrics.get("complementarity", {})
    table1_rows.append({
        "cancer": "POOLED",
        "n_cells_allcell": "—",
        "n_cells_testonly": "—",
        "fold_increase": "—",
        "n_pairs_matched": len(pooled_allcell),
        "synergy_rate": round(float(y_t_pool.mean()), 3),
        "auc_allcell": round(pooled_m.get("auc", float("nan")), 4),
        "auc_ci": f"{pooled_m.get('auc_ci_lo',0):.3f}–{pooled_m.get('auc_ci_hi',0):.3f}",
        "auc_perm_p": pooled_m.get("auc_perm_p", float("nan")),
        "prauc": round(pooled_m.get("prauc", float("nan")), 4),
        "spearman_rho": round(pooled_m.get("spearman_rho", float("nan")), 4),
        "shuffle_p": "—",
    })

# ── Save tables ───────────────────────────────────────────────────────────────
pd.DataFrame(table1_rows).to_csv(RESULTS_DIR / "table1_allcell_per_cancer_auc.csv", index=False)
pd.DataFrame(table2_rows).to_csv(RESULTS_DIR / "table2_allcell_vs_testonly.csv", index=False)

print("\n" + "=" * 70)
print("TABLE 1: Per-cancer AUC (all-cell analysis)")
print("=" * 70)
print(pd.DataFrame(table1_rows).to_string(index=False))

print("\n" + "=" * 70)
print("TABLE 2: All-cell vs Test-only AUC comparison")
print("=" * 70)
print(pd.DataFrame(table2_rows).to_string(index=False))

print(f"\nAll outputs saved to: {RESULTS_DIR}")
print("[DONE] Script 02 complete.")
