#!/usr/bin/env python3
"""
Final evidence matrix: integrate all validation layers.

Combines Layer 6 (chemistry), Layer 4 (transcriptome reversal),
Layer 5 (docking), and Layer 7 (statistical controls) into a
unified evidence score and Tier classification per improved analogue.

Run with: conda run -n kg_GAUGE python v07_final_evidence.py
Output: results/layer7_final/final_evidence_matrix.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
RES     = ROOT / "results"
OUT_DIR = RES / "layer7_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER6 = RES / "layer6_chemistry" / "chemistry_admet.csv"
LAYER4 = RES / "layer4_transcriptome" / "reversal_scores.csv"
LAYER5 = RES / "layer5_docking"    / "docking_scores.csv"
CTRL   = RES / "layer7_final"      / "statistical_controls_summary.json"

EVIDENCE_WEIGHTS = {
    "z_delta_improvement":    0.25,  # GAUGE efficacy gain
    "z_reversal_score":       0.15,  # transcriptome reversal
    "z_docking_improvement":  0.15,  # docking score vs parent
    "z_chem_score":           0.10,  # drug-likeness vs parent
    "penalty_hard_fail":     -0.20,  # chemistry hard fail
    "penalty_pains":         -0.10,  # PAINS alert
}


def zscore_col(series: pd.Series) -> pd.Series:
    """Z-score normalisation, handling NaN."""
    s = series.dropna()
    if len(s) < 2 or s.std() == 0:
        return series.fillna(0) - series.fillna(0).mean()
    return (series - s.mean()) / s.std()


def main():
    print("=" * 60)
    print("Final Evidence Matrix: Multi-layer Integration")
    print("=" * 60)

    # ── Load all improved analogues ───────────────────────────────────────────
    luad = pd.read_csv(ROOT / "data" / "candidates" / "generated_compounds_luad.csv")
    skcm = pd.read_csv(ROOT / "data" / "candidates" / "generated_compounds_skcm.csv")
    luad = luad[luad["tanimoto"] >= 0.10].copy()
    skcm = skcm[skcm["tanimoto"] >= 0.10].copy()
    all_df = pd.concat([luad, skcm], ignore_index=True)
    improved = all_df[all_df["delta_improvement"] > 0].copy()
    print(f"  Improved analogues: {len(improved)}")

    ev = improved[["DRUG_NAME", "DRUG_ID", "smiles", "seed_drug", "cancer_type",
                   "delta_improvement", "mean_value_hat", "tanimoto"]].copy()

    # ── Layer 6: Chemistry ───────────────────────────────────────────────────
    if LAYER6.exists():
        chem = pd.read_csv(LAYER6)
        # DRUG_ID is non-unique across seeds; match by DRUG_NAME
        chem_imp = chem[chem["DRUG_NAME"].isin(ev["DRUG_NAME"].values)][
            ["DRUG_NAME", "qed", "sa_score", "lipinski_violations",
             "n_pains_alerts", "hard_fail", "chem_score_vs_parent"]
        ].drop_duplicates("DRUG_NAME")
        ev = ev.merge(chem_imp, on="DRUG_NAME", how="left")
        print(f"  Layer 6 merged: {chem_imp['DRUG_NAME'].nunique()} drugs")
    else:
        ev[["qed", "sa_score", "lipinski_violations", "n_pains_alerts",
            "hard_fail", "chem_score_vs_parent"]] = np.nan
        print(f"  Layer 6: not found at {LAYER6}")

    # ── Layer 4: Transcriptome reversal ──────────────────────────────────────
    if LAYER4.exists():
        rev = pd.read_csv(LAYER4)
        rev_imp = rev[rev["DRUG_NAME"].isin(ev["DRUG_NAME"].values)][
            ["DRUG_NAME", "reversal_score", "delta_reversal_vs_parent"]
        ].drop_duplicates("DRUG_NAME")
        ev = ev.merge(rev_imp, on="DRUG_NAME", how="left")
        print(f"  Layer 4 merged: {rev_imp['DRUG_NAME'].nunique()} drugs")
    else:
        ev[["reversal_score", "delta_reversal_vs_parent"]] = np.nan
        print(f"  Layer 4: not found at {LAYER4}")

    # ── Layer 5: Docking ─────────────────────────────────────────────────────
    if LAYER5.exists():
        dock = pd.read_csv(LAYER5)
        dock_imp = dock[dock["DRUG_NAME"].isin(ev["DRUG_NAME"].values)][
            ["DRUG_NAME", "vina_score", "cnn_affinity", "delta_vina_vs_parent", "delta_cnn_vs_parent"]
        ].drop_duplicates("DRUG_NAME")
        ev = ev.merge(dock_imp, on="DRUG_NAME", how="left")
        print(f"  Layer 5 merged: {dock_imp['DRUG_NAME'].nunique()} drugs")
    else:
        ev[["vina_score", "cnn_affinity", "delta_vina_vs_parent", "delta_cnn_vs_parent"]] = np.nan
        print(f"  Layer 5: not found at {LAYER5}")

    # ── Load statistical controls summary ─────────────────────────────────────
    ctrl_summary = {}
    if CTRL.exists():
        with open(CTRL) as f:
            ctrl_summary = json.load(f)

    # ── Compute evidence score ───────────────────────────────────────────────
    # Z-score each metric within the improved set
    ev["z_delta_improvement"] = zscore_col(ev["delta_improvement"])
    ev["z_reversal_score"]    = zscore_col(ev["reversal_score"])
    # Docking: lower vina = better, so negate delta (delta < 0 = improvement)
    ev["z_docking_improvement"] = zscore_col(-ev["delta_vina_vs_parent"].fillna(0))
    ev["z_chem_score"]          = zscore_col(ev["chem_score_vs_parent"].fillna(0))

    ev["penalty_hard_fail"] = ev["hard_fail"].fillna(False).astype(float)
    ev["penalty_pains"]     = (ev["n_pains_alerts"].fillna(0) > 0).astype(float)

    ev["evidence_score"] = (
          EVIDENCE_WEIGHTS["z_delta_improvement"]   * ev["z_delta_improvement"]
        + EVIDENCE_WEIGHTS["z_reversal_score"]      * ev["z_reversal_score"].fillna(0)
        + EVIDENCE_WEIGHTS["z_docking_improvement"] * ev["z_docking_improvement"]
        + EVIDENCE_WEIGHTS["z_chem_score"]          * ev["z_chem_score"]
        + EVIDENCE_WEIGHTS["penalty_hard_fail"]     * ev["penalty_hard_fail"]
        + EVIDENCE_WEIGHTS["penalty_pains"]         * ev["penalty_pains"]
    )

    # ── Tier classification ──────────────────────────────────────────────────
    def classify_tier(row):
        if row["hard_fail"] if not pd.isna(row.get("hard_fail")) else False:
            return "C"
        if row["n_pains_alerts"] > 0 if not pd.isna(row.get("n_pains_alerts")) else False:
            return "C"
        # Tier A: efficacy + at least one of (reversal, docking) positive
        has_reversal = (not pd.isna(row.get("delta_reversal_vs_parent"))) and \
                       row.get("delta_reversal_vs_parent", 0) > 0
        has_docking  = (not pd.isna(row.get("delta_vina_vs_parent"))) and \
                       row.get("delta_vina_vs_parent", 0) < 0
        if has_reversal or has_docking:
            return "A"
        # Tier B: only efficacy + chemistry pass
        if row["lipinski_violations"] == 0 if not pd.isna(row.get("lipinski_violations")) else True:
            return "B"
        return "C"

    ev["tier"] = ev.apply(classify_tier, axis=1)

    # ── Print summary ─────────────────────────────────────────────────────────
    ev_sorted = ev.sort_values("evidence_score", ascending=False)

    print(f"\n  Evidence matrix ({len(ev)} improved analogues):")
    show_cols = ["DRUG_NAME", "seed_drug", "tier", "evidence_score",
                 "delta_improvement", "reversal_score", "delta_vina_vs_parent",
                 "chem_score_vs_parent", "qed", "n_pains_alerts"]
    available = [c for c in show_cols if c in ev_sorted.columns]
    print(ev_sorted[available].round(4).to_string(index=False))

    tier_counts = ev["tier"].value_counts()
    print(f"\n  Tier summary:")
    for tier in ["A", "B", "C"]:
        print(f"    Tier {tier}: {tier_counts.get(tier, 0)}")

    # ── Statistical controls summary ──────────────────────────────────────────
    print("\n  Statistical Controls Summary:")
    for seed_drug, ctrl in ctrl_summary.items():
        print(f"\n  {seed_drug}:")
        for k, v in ctrl.items():
            if k in ["seed_drug", "cancer_type", "seed_mw", "seed_logp"]:
                continue
            print(f"    {k}: {v}")

    # ── Save ──────────────────────────────────────────────────────────────────
    ev_sorted.to_csv(OUT_DIR / "final_evidence_matrix.csv", index=False)

    # Final JSON report
    final_report = {
        "n_improved_analogues": int(len(ev)),
        "tier_counts": {t: int(tier_counts.get(t, 0)) for t in ["A", "B", "C"]},
        "layers_available": {
            "layer6_chemistry": LAYER6.exists(),
            "layer4_transcriptome": LAYER4.exists(),
            "layer5_docking": LAYER5.exists(),
        },
        "top_analogues": ev_sorted.head(4)[available].round(4).to_dict(orient="records"),
        "statistical_controls": ctrl_summary,
    }
    with open(OUT_DIR / "final_evidence_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n  Saved → {OUT_DIR}/final_evidence_matrix.csv")
    print(f"  Saved → {OUT_DIR}/final_evidence_report.json")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
