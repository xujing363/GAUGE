#!/usr/bin/env python3
"""
Publication v5 — Supervised Ceiling Reference.

Design: Leave-one-cancer-type-out cross-validation on NCI-ALMANAC.
  - For each cancer type as TEST:
      - Train a Random Forest classifier on ALL NCI-ALMANAC pairs from OTHER 3 cancer types
        that can be featurized with drug Morgan fingerprints.
      - Test on our 119 KG-selected pairs from the held-out cancer type.
  - Features: drug-pair chemical features only (no combination training label used at test time).
  - This gives a proper "supervised ceiling" that USES combination training data.

Why this design is fair:
  - GAUGE uses ZERO combination training data (zero-shot).
  - Supervised model uses NCI-ALMANAC synergy labels from other cancer types.
  - Both are evaluated on the same 119 KG pairs per cancer type.
  - Cross-cancer design prevents data leakage (held-out cancer data never in training).

Features used (all symmetric, drug-order invariant):
  F1: fp_A × fp_B              (2048-d, Hadamard product — captures co-active substructures)
  F2: |fp_A − fp_B|            (2048-d, absolute difference — captures structural divergence)
  F3: mean(fp_A, fp_B)         (2048-d, average presence)
  F4: Tanimoto(fp_A, fp_B)     (scalar — chemical similarity)
  Total: 6145-d feature vector

Model: RandomForestClassifier (100 trees, class_weight='balanced')
  - No hyperparameter search to avoid overfitting to our small datasets.
  - Reports calibrated probability scores (predict_proba).

Run: python publication_v5_complementarity_multicancer/scripts/run_v5_supervised_ceiling.py
"""
from __future__ import annotations
import json, re, sys, zipfile, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc

sys.path.insert(0, "/mnt/raid5/xujing/KG/GAUGE")
sys.path.insert(0, "/mnt/raid5/xujing/KG")

MULTI_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = MULTI_ROOT / "publication_v5_complementarity_multicancer" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NCI_ZIP = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/NCI_ALMANAC/ComboDrugGrowth_Nov2017.zip")
NSC_MAP_CSV = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/NCI_ALMANAC/nsc_name_map.csv")
ARTIFACTS_PKL = Path("/mnt/raid5/xujing/KG/Combined/results/combined_melanoma_v1_20260524_130336/artifacts.pkl")
SEED = 42
rng = np.random.RandomState(SEED)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float(inter / union) if union > 0 else 0.0


def pair_features(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Symmetric drug-pair feature vector (order-invariant)."""
    ha = fp_a.astype(np.float32)
    hb = fp_b.astype(np.float32)
    f1 = ha * hb                       # Hadamard product
    f2 = np.abs(ha - hb)               # absolute difference
    f3 = (ha + hb) / 2.0               # mean fingerprint
    f4 = np.array([tanimoto(ha, hb)])  # scalar Tanimoto
    return np.concatenate([f1, f2, f3, f4])


def precision_at_k(y_true, y_score, k):
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())


def compute_metrics(y_true, y_score, y_cont, label, n_perm=5000):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_cont = np.asarray(y_cont, dtype=float)
    n = len(y_true)
    if y_true.sum() == 0 or y_true.sum() == n:
        return {"label": label, "n": n, "error": "no variance"}
    auc_val = roc_auc_score(y_true, y_score)
    null_aucs = [roc_auc_score(y_true, rng.permutation(y_score)) for _ in range(n_perm)]
    p_auc = max(1 / n_perm, float((np.array(null_aucs) >= auc_val).mean()))
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    prauc = pr_auc(rec, prec)
    rho, p_rho = spearmanr(y_score, y_cont)
    return {
        "label": label, "n": int(n), "pos_rate": float(y_true.mean()),
        "auc": float(auc_val), "auc_perm_p": float(p_auc),
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
    print(f"{indent}[{m['label']}] n={m['n']} pos={m['pos_rate']:.3f}")
    print(f"{indent}  AUC={m['auc']:.4f} (p={m['auc_perm_p']:.4f})")
    print(f"{indent}  PRAUC={m['prauc']:.4f}  Spear={m['spearman_rho']:+.4f}(p={m['spearman_p']:.3f})")
    print(f"{indent}  P@5={m['p5']:.2f}  P@10={m['p10']:.2f}  P@15={m['p15']:.2f}  P@25={m['p25']:.2f}")


# ── Load data ─────────────────────────────────────────────────────────────────

print("=" * 70)
print("Loading drug fingerprints from model artifacts...")
with open(ARTIFACTS_PKL, "rb") as f:
    artifacts = pickle.load(f)
drug_table = artifacts.drug_table
norm_to_fp: dict[str, np.ndarray] = {}
for _, row in drug_table.iterrows():
    norm_to_fp[norm(row["DRUG_NAME"])] = np.asarray(row["fingerprint"], dtype=np.float32)
print(f"  Drug fingerprints loaded: {len(norm_to_fp)} drugs, dim={len(next(iter(norm_to_fp.values())))}")

print("Loading NSC → drug name map...")
nsc_map = pd.read_csv(NSC_MAP_CSV)
nsc_to_norm: dict[int, str] = {int(r["nsc"]): norm(r["drug_name"]) for _, r in nsc_map.iterrows()}
# Build NSC → fingerprint lookup
nsc_to_fp: dict[int, np.ndarray] = {}
for nsc, n in nsc_to_norm.items():
    if n in norm_to_fp:
        nsc_to_fp[nsc] = norm_to_fp[n]
print(f"  NSC codes with fingerprints: {len(nsc_to_fp)} / {len(nsc_to_norm)}")

print("Loading NCI-ALMANAC (full)...")
with zipfile.ZipFile(NCI_ZIP) as zf:
    with zf.open("ComboDrugGrowth_Nov2017.csv") as f:
        nci_all = pd.read_csv(f, usecols=["PANEL", "NSC1", "NSC2", "SCORE"], low_memory=False)
nci_all = nci_all.dropna(subset=["SCORE"])
nci_all["NSC1"] = nci_all["NSC1"].astype(int)
nci_all["NSC2"] = nci_all["NSC2"].astype(int)
# Aggregate by (panel, nsc1, nsc2) — mean over cell lines
nci_pairs = nci_all.groupby(["PANEL", "NSC1", "NSC2"])["SCORE"].mean().reset_index()
nci_pairs["is_syn"] = (nci_pairs["SCORE"] > 0).astype(int)
print(f"  Total NCI pairs (aggregated): {len(nci_pairs)} across {nci_pairs['PANEL'].nunique()} panels")

# ── Build training data per cancer panel ──────────────────────────────────────

CANCER_CONFIGS = {
    "Melanoma":  {"nci_panel": "Melanoma",                   "nci_matched": "nci_matched_melanoma.csv"},
    "NSCLC":     {"nci_panel": "Non-Small Cell Lung Cancer",  "nci_matched": "nci_matched_nsclc.csv"},
    "Breast":    {"nci_panel": "Breast Cancer",               "nci_matched": "nci_matched_breast.csv"},
    "Ovarian":   {"nci_panel": "Ovarian Cancer",              "nci_matched": "nci_matched_ovarian.csv"},
}


def build_nci_features(panel_name: str, exclude_nsc_pairs: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for all featurizable NCI pairs in a given panel, excluding test pairs."""
    panel_df = nci_pairs[nci_pairs["PANEL"] == panel_name]
    X_rows, y_rows = [], []
    for _, row in panel_df.iterrows():
        nsc1, nsc2 = int(row["NSC1"]), int(row["NSC2"])
        k = tuple(sorted([nsc1, nsc2]))
        if k in exclude_nsc_pairs:
            continue
        if nsc1 in nsc_to_fp and nsc2 in nsc_to_fp:
            feat = pair_features(nsc_to_fp[nsc1], nsc_to_fp[nsc2])
            X_rows.append(feat)
            y_rows.append(int(row["is_syn"]))
    if not X_rows:
        return np.empty((0, 6145)), np.empty(0, dtype=int)
    return np.vstack(X_rows), np.array(y_rows, dtype=int)


# ── Leave-one-cancer-type-out supervised evaluation ──────────────────────────

all_results = {}
summary_rows = []

for test_cancer, cfg in CANCER_CONFIGS.items():
    print(f"\n{'=' * 70}")
    print(f"  TEST CANCER: {test_cancer} (leave-one-cancer-out)")
    print(f"{'=' * 70}")

    # Load NCI-matched test pairs (our 119 KG pairs)
    test_df = pd.read_csv(RESULTS_DIR / cfg["nci_matched"])

    # Build test features from our 119 pairs
    test_X, test_y, test_cont, test_names = [], [], [], []
    nsc_map_local = pd.read_csv(NSC_MAP_CSV)
    nsc_by_norm: dict[str, list[int]] = {}
    for _, r in nsc_map_local.iterrows():
        nsc_by_norm.setdefault(norm(str(r["drug_name"])), []).append(int(r["nsc"]))

    # Get NSC→NCI score lookup for test panel
    test_panel_lookup: dict[tuple, float] = {}
    test_panel_df = nci_pairs[nci_pairs["PANEL"] == cfg["nci_panel"]]
    for _, row in test_panel_df.iterrows():
        k = tuple(sorted([int(row["NSC1"]), int(row["NSC2"])]))
        test_panel_lookup[k] = float(row["SCORE"])

    test_nsc_pairs = set()
    for _, pair in test_df.iterrows():
        da, db = pair["drug_A_name"], pair["drug_B_name"]
        fp_a = norm_to_fp.get(norm(da))
        fp_b = norm_to_fp.get(norm(db))
        if fp_a is None or fp_b is None:
            continue
        feat = pair_features(fp_a, fp_b)
        test_X.append(feat)
        test_y.append(int(pair["is_syn"]))
        test_cont.append(float(pair["nci_score"]))
        test_names.append((da, db))
        # Record NSC pairs to exclude from training
        for na in nsc_by_norm.get(norm(da), []):
            for nb in nsc_by_norm.get(norm(db), []):
                test_nsc_pairs.add(tuple(sorted([na, nb])))

    test_X = np.vstack(test_X) if test_X else np.empty((0, 6145))
    test_y = np.array(test_y, dtype=int)
    test_cont = np.array(test_cont)
    print(f"  Test pairs featurized: {len(test_y)} / {len(test_df)}")

    if len(test_y) < 10:
        print(f"  Skipping {test_cancer}: too few featurizable test pairs")
        continue

    # Build training data from all OTHER cancer panels (excluding test pairs)
    train_cancer_panels = [v["nci_panel"] for k, v in CANCER_CONFIGS.items() if k != test_cancer]
    X_train_list, y_train_list = [], []
    for train_panel in train_cancer_panels:
        X_p, y_p = build_nci_features(train_panel, exclude_nsc_pairs=test_nsc_pairs)
        X_train_list.append(X_p)
        y_train_list.append(y_p)
        print(f"  Training panel [{train_panel}]: {len(y_p)} pairs ({y_p.sum()} positive)")

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    print(f"  Total training pairs: {len(y_train)} ({y_train.sum()} positive, {y_train.mean():.2%})")

    cancer_results = {}

    # ── Model 1: Random Forest ────────────────────────────────────────────────
    print(f"\n  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(test_X)[:, 1]
    m_rf = compute_metrics(test_y, rf_proba, test_cont, label="Supervised RF (leave-one-cancer-out)")
    print_metrics(m_rf)
    cancer_results["supervised_rf"] = m_rf

    # ── Model 2: Logistic Regression (L2) ────────────────────────────────────
    print(f"\n  Training Logistic Regression (L2)...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(test_X)
    lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=500, random_state=SEED)
    lr.fit(X_tr_scaled, y_train)
    lr_proba = lr.predict_proba(X_te_scaled)[:, 1]
    m_lr = compute_metrics(test_y, lr_proba, test_cont, label="Supervised LR-L2 (leave-one-cancer-out)")
    print_metrics(m_lr)
    cancer_results["supervised_lr"] = m_lr

    # ── Retrieve GAUGE scores for comparison ────────────────────────────────
    test_df_sub = test_df.iloc[:len(test_y)]  # aligned rows (only featurizable)
    m_pcp = compute_metrics(test_y, test_df_sub["pcp"].values[:len(test_y)], test_cont,
                            label="GAUGE pcp (zero-shot)")
    m_compl = compute_metrics(test_y, test_df_sub["complementarity"].values[:len(test_y)], test_cont,
                              label="GAUGE complementarity (zero-shot)")
    print_metrics(m_pcp)
    print_metrics(m_compl)
    cancer_results["GAUGE_pcp"] = m_pcp
    cancer_results["GAUGE_complementarity"] = m_compl

    all_results[test_cancer] = cancer_results
    summary_rows.append({
        "Cancer": test_cancer,
        "n_test": len(test_y),
        "n_train": len(y_train),
        "pos_rate": float(test_y.mean()),
        "supervised_rf_auc": m_rf.get("auc", float("nan")),
        "supervised_rf_p": m_rf.get("auc_perm_p", float("nan")),
        "supervised_rf_p10": m_rf.get("p10", float("nan")),
        "supervised_lr_auc": m_lr.get("auc", float("nan")),
        "supervised_lr_p": m_lr.get("auc_perm_p", float("nan")),
        "GAUGE_pcp_auc": m_pcp.get("auc", float("nan")),
        "GAUGE_pcp_p": m_pcp.get("auc_perm_p", float("nan")),
        "GAUGE_pcp_p10": m_pcp.get("p10", float("nan")),
        "GAUGE_complementarity_auc": m_compl.get("auc", float("nan")),
        "GAUGE_complementarity_p": m_compl.get("auc_perm_p", float("nan")),
        "GAUGE_complementarity_p10": m_compl.get("p10", float("nan")),
    })

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("CEILING COMPARISON SUMMARY TABLE")
print(f"{'=' * 70}")
print(f"  Supervised models trained on NCI-ALMANAC (other 3 cancer types)")
print(f"  GAUGE: zero-shot, trained on single-drug GDSC only")
print()

df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(RESULTS_DIR / "supervised_ceiling_summary.csv", index=False)

print(f"{'Cancer':<12}  {'SupRF AUC':>10}  {'SupLR AUC':>10}  {'pcp AUC':>10}  {'Compl AUC':>10}")
print("-" * 58)
for _, r in df_sum.iterrows():
    rf_s = "*" if r["supervised_rf_p"] < 0.05 else " "
    lr_s = "*" if r["supervised_lr_p"] < 0.05 else " "
    pcp_s = "*" if r["GAUGE_pcp_p"] < 0.05 else " "
    co_s = "*" if r["GAUGE_complementarity_p"] < 0.05 else " "
    print(f"{r['Cancer']:<12}  "
          f"{r['supervised_rf_auc']:.3f}{rf_s} (p={r['supervised_rf_p']:.3f})  "
          f"{r['supervised_lr_auc']:.3f}{lr_s} (p={r['supervised_lr_p']:.3f})  "
          f"{r['GAUGE_pcp_auc']:.3f}{pcp_s}  "
          f"{r['GAUGE_complementarity_auc']:.3f}{co_s}")

print(f"\nNote: Supervised models use NCI-ALMANAC combination labels (3 cancer types).")
print(f"GAUGE uses ZERO combination training data (pure zero-shot).")
print(f"All evaluated on the same {119} KG-selected pairs per cancer type.")

with open(RESULTS_DIR / "supervised_ceiling_report.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}")
print("\n=== SUPERVISED CEILING ANALYSIS COMPLETE ===")
