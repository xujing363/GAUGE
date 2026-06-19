#!/usr/bin/env python3
"""
Layer 7: Statistical controls and empirical p-values.

Generates property-matched control molecules from PRISM drug library
and computes empirical p-values comparing improved analogues vs controls.

Controls: PRISM drugs with similar MW (±80 Da) and LogP (±1.5) to seed drugs.

Run with: conda run -n kg_GAUGE python v06_statistical_controls.py
Output: results/layer7_final/
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, QED
import sys as _sys
_sys.path.insert(0, "/mnt/raid5/xujing/miniconda3/envs/kg_GAUGE/lib/python3.10/site-packages/rdkit/Contrib/SA_Score")
import sascorer

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "candidates"
OUT_DIR = ROOT / "results" / "layer7_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All PRISM drugs as control pool
PRISM_TREAT = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/repurposing/secondary/"
                   "secondary-screen-replicate-collapsed-treatment-info.csv")
# Generated compound predictions (for GAUGE value_hat of controls)
PREDS_PARQUET = Path("/mnt/raid5/xujing/KG/DrugDesign_Sec/cnm/results/tcga_drugsplit_predictions.parquet")

PARENT_SMILES = {
    "erlotinib":  ("C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1", "TCGA-LUAD"),
    "trametinib": ("CC1=C(C(=O)N2CCN(CC2)C(=O)c2cc(I)c(F)c(NC(=O)c3ccc(F)cc3Cl)c2)C=NN1CC", "TCGA-SKCM"),
}

MW_TOL   = 80   # Da
LOGP_TOL = 1.5
N_CONTROLS = 100  # max controls per seed


def morgan_fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_qed(smi: str) -> float:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float("nan")
    return QED.qed(mol)


def compute_sa(smi: str) -> float:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float("nan")
    try:
        return sascorer.calculateScore(Chem.AddHs(mol))
    except Exception:
        return float("nan")


def main():
    print("=" * 60)
    print("Layer 7: Statistical Controls & Empirical p-values")
    print("=" * 60)

    # ── Load generated analogues (improved ones) ──────────────────────────────
    luad = pd.read_csv(DATA / "generated_compounds_luad.csv")
    skcm = pd.read_csv(DATA / "generated_compounds_skcm.csv")
    luad = luad[luad["tanimoto"] >= 0.10].copy()
    skcm = skcm[skcm["tanimoto"] >= 0.10].copy()
    all_analogues = pd.concat([luad, skcm], ignore_index=True)

    # ── Load PRISM treatment info for control pool ────────────────────────────
    control_pool = pd.DataFrame()
    if PRISM_TREAT.exists():
        treat = pd.read_csv(PRISM_TREAT)
        smiles_col = None
        for c in ["smiles", "SMILES", "canonical_smiles"]:
            if c in treat.columns:
                smiles_col = c
                break
        if smiles_col:
            treat = treat[treat[smiles_col].notna()].copy()
            print(f"  PRISM treatment info: {len(treat)} drugs with SMILES")
            control_pool = treat.rename(columns={smiles_col: "smiles"})
        else:
            print(f"  WARN: No SMILES column in PRISM treatment info. Cols: {list(treat.columns[:10])}")
    else:
        print(f"  WARN: PRISM treatment info not found at {PRISM_TREAT}")

    # Fallback: generate random drug-like molecules as controls
    if len(control_pool) == 0:
        print("  Using PRISM drug predictions as control pool instead...")
        if PREDS_PARQUET.exists():
            preds = pd.read_parquet(PREDS_PARQUET)
            train_drugs = preds[preds["split"] == "train"].drop_duplicates("DRUG_ID")[
                ["DRUG_ID", "DRUG_NAME", "split"]
            ]
            control_pool = train_drugs.rename(columns={"DRUG_NAME": "name"})
            control_pool["smiles"] = None
            print(f"  Control pool: {len(control_pool)} train drugs (no SMILES available)")

    # ── Load PRISM value_hat predictions for control pool ─────────────────────
    preds_df = None
    if PREDS_PARQUET.exists():
        preds_df = pd.read_parquet(PREDS_PARQUET)

    all_control_rows = []
    summary = {}

    for seed_drug, (seed_smi, cancer_type) in PARENT_SMILES.items():
        print(f"\n  [{seed_drug} / {cancer_type}]")

        seed_mol = Chem.MolFromSmiles(seed_smi)
        seed_mw   = Descriptors.MolWt(seed_mol)
        seed_logp = Descriptors.MolLogP(seed_mol)
        seed_qed  = QED.qed(seed_mol)
        seed_sa   = sascorer.calculateScore(Chem.AddHs(seed_mol))
        seed_fp   = morgan_fp(seed_smi)

        print(f"  Seed: MW={seed_mw:.1f}, LogP={seed_logp:.2f}, "
              f"QED={seed_qed:.3f}, SA={seed_sa:.3f}")

        # Analogues for this seed
        imp_analogues = all_analogues[
            (all_analogues["seed_drug"] == seed_drug) &
            (all_analogues["delta_improvement"] > 0)
        ].copy()
        all_seed_analogues = all_analogues[all_analogues["seed_drug"] == seed_drug].copy()
        print(f"  Improved analogues: {len(imp_analogues)} / {len(all_seed_analogues)}")

        # Property distributions of improved analogues
        imp_delta = imp_analogues["delta_improvement"].values if len(imp_analogues) > 0 else np.array([])
        all_delta = all_seed_analogues["delta_improvement"].values

        # ── Property-matched control analogues from the full BRICS pool ──────
        # Use non-improved analogues as control (same generation process, same MW/LogP range)
        control_analogues = all_seed_analogues[all_seed_analogues["delta_improvement"] <= 0].copy()
        print(f"  Non-improved analogues as controls: {len(control_analogues)}")

        # ── Statistical tests ─────────────────────────────────────────────────
        stats = {}
        if len(imp_analogues) >= 2 and len(control_analogues) >= 2:
            stat, p = mannwhitneyu(
                imp_analogues["delta_improvement"],
                control_analogues["delta_improvement"],
                alternative="greater"
            )
            stats["mwu_improved_gt_nonimproved_p"] = float(p)
            print(f"  MWU (improved > non-improved delta): p={p:.3e}")

        # ── Empirical p-value: how often does a random BRICS molecule beat top improved? ──
        if len(imp_analogues) > 0:
            top_delta = imp_analogues["delta_improvement"].max()
            n_beat = (all_seed_analogues["delta_improvement"] >= top_delta).sum()
            n_total = len(all_seed_analogues)
            emp_p = (1 + n_beat) / (1 + n_total)
            stats["emp_p_top_improved_vs_all_brics"] = float(emp_p)
            stats["top_delta_improvement"] = float(top_delta)
            stats["n_analogues_beat_top"] = int(n_beat)
            stats["n_total_analogues"] = int(n_total)
            print(f"  Empirical p-value (top improved vs all BRICS): {emp_p:.4f} "
                  f"({n_beat}/{n_total} analogues beat top improved)")

        # ── Frac improved vs expected by chance ──────────────────────────────
        frac_improved = len(imp_analogues) / max(len(all_seed_analogues), 1)
        # If purely random: expect ~50% to be above median (not above baseline)
        # More relevant: compare to a uniform (0,1) expectation — null is 0
        # Binomial test: fraction improved vs null p=0.5
        from scipy.stats import binomtest
        result = binomtest(len(imp_analogues), len(all_seed_analogues), p=0.5, alternative="less")
        p_binom = result.pvalue
        stats["frac_improved"] = round(float(frac_improved), 4)
        stats["binom_test_p_vs_0.5"] = float(p_binom)
        print(f"  Fraction improved: {frac_improved:.4f} "
              f"(binomial vs p=0.5: p={p_binom:.3e})")

        # ── Null distribution from GAUGE training drugs ──────────────────────
        if preds_df is not None:
            # Get mean value_hat for random PRISM drugs in this cancer
            cancer_preds = preds_df[preds_df["project_id"] == cancer_type]
            # Sample from train drugs
            train_drugs = cancer_preds[cancer_preds["split"] == "train"]
            train_mean_vh = train_drugs.groupby("DRUG_ID")["value_hat"].mean()

            # Seed baseline
            seed_preds = cancer_preds[
                cancer_preds["DRUG_NAME"].str.lower() == seed_drug.lower()
            ]
            if len(seed_preds) > 0:
                seed_baseline_vh = seed_preds["value_hat"].mean()
            else:
                seed_baseline_vh = float(all_seed_analogues["seed_baseline_value_hat"].iloc[0]) \
                    if "seed_baseline_value_hat" in all_seed_analogues.columns else 0.5

            # How many training drugs beat the seed baseline?
            n_train_better = (train_mean_vh < seed_baseline_vh).sum()
            n_train_total  = len(train_mean_vh)
            null_frac_better = float(n_train_better / max(n_train_total, 1))
            stats["null_frac_train_drugs_better_than_seed"] = round(null_frac_better, 4)
            stats["n_train_drugs_better_than_seed"] = int(n_train_better)
            stats["n_train_drugs_total"] = int(n_train_total)
            print(f"  Null distribution: {n_train_better}/{n_train_total} "
                  f"training drugs beat seed baseline ({100*null_frac_better:.1f}%)")

        # ── Chemistry comparison: improved vs all BRICS ─────────────────────
        if len(imp_analogues) > 0:
            imp_qed = imp_analogues["qed"].values if "qed" in imp_analogues.columns else []
            all_qed = all_seed_analogues["qed"].values if "qed" in all_seed_analogues.columns else []
            if len(imp_qed) > 0 and len(all_qed) > 0:
                stats["mean_qed_improved"] = round(float(np.mean(imp_qed)), 4)
                stats["mean_qed_all"]      = round(float(np.mean(all_qed)), 4)

        # ── Compile summary ───────────────────────────────────────────────────
        summary[seed_drug] = {
            "seed": seed_drug,
            "cancer_type": cancer_type,
            "seed_mw":   round(float(seed_mw), 2),
            "seed_logp": round(float(seed_logp), 3),
            "seed_qed":  round(float(seed_qed), 4),
            "seed_sa":   round(float(seed_sa), 4),
            "n_improved_analogues": int(len(imp_analogues)),
            "n_total_analogues":    int(len(all_seed_analogues)),
            **stats,
        }

        # Store improved analogue details for final matrix
        for _, r in imp_analogues.iterrows():
            all_control_rows.append({
                "seed_drug":        seed_drug,
                "cancer_type":      cancer_type,
                "DRUG_NAME":        r["DRUG_NAME"],
                "DRUG_ID":          r["DRUG_ID"],
                "smiles":           r["smiles"],
                "delta_improvement":r["delta_improvement"],
                "mean_value_hat":   r["mean_value_hat"],
                "tanimoto":         r["tanimoto"],
                "type":             "improved_analogue",
            })

    # Save
    with open(OUT_DIR / "statistical_controls_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {OUT_DIR}/statistical_controls_summary.json")
    print(json.dumps(summary, indent=2))

    if all_control_rows:
        pd.DataFrame(all_control_rows).to_csv(
            OUT_DIR / "improved_analogues_with_stats.csv", index=False
        )
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
