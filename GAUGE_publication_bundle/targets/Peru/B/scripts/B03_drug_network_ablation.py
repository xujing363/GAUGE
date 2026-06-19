"""
B03: Drug Network Ablation — Core Scenario B Perturbation
==========================================================
Core experiment: For drug split test drugs (never seen in training),
systematically ablate the drug's KG network and measure the change in
predicted sensitivity across cell lines.

Scientific rationale:
  - Model has never seen these drugs → reliance on KG reflects genuine
    mechanism generalization, not memorization
  - For drugs with known targets (Erlotinib/EGFR, Venetoclax/BCL2, etc.),
    KG ablation should have LARGER effect in cancer types that depend on
    that specific target
  - This cancer-type-specific sensitivity to KG perturbation is the key
    evidence for the "dynamic world model" claim

Perturbation design (three levels):
  L1. Full KG knockout:     all_off → remove all three KG sources
  L2. Source-level:         ChEMBL_off / PrimeKG_off (DRKG already absent)
  L3. Drug-specific edges:  mask only edges connected to the focal drug
      using kg_mask = {"edge_ids": [list of drug-specific edge IDs]}

Key question: Does drug-specific edge ablation produce a cancer-type-
specific pattern consistent with the drug's known mechanism of action?

Outputs:
  results/drug_ablation/
    drug_network_ablation_full.csv    - per-drug ΔAUC for full KG off
    drug_source_ablation.csv          - per-drug ΔAUC per KG source
    drug_specific_edge_ablation.csv   - focal drug × drug-specific edges
    cancer_specific_delta.csv         - ΔAUC by cancer type per drug
  figures/
    B03_drug_network_ablation.pdf
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DRUG_SPLIT_RESULT_DIR, DRUG_SPLIT_PREPARED_PKL, DRUG_SPLIT_CONFIG_YAML,
    PREDICTIONS_CSV, KG_COVERAGE_CSV, CHEMBL_EDGES_CSV, PRIMEKG_EDGES_CSV,
    DRKG_EDGES_CSV, GDSC_MODEL_LIST, RESULTS, FIGURES, FOCAL_TEST_DRUGS,
    CANCER_GROUPS, DEVICE, BATCH_SIZE,
)
from utils import (
    load_experiment, run_predictions, build_session,
    load_cell_cancer_types, annotate_cancer_group, get_drug_kg_edges,
)

OUT_R = RESULTS / "drug_ablation"
OUT_F = FIGURES
OUT_R.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)


def compute_within_drug_spearman(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["AUC", "auc_hat"])
    if len(df) < 5:
        return {"spearman": np.nan, "pcc": np.nan, "n": len(df)}
    s = stats.spearmanr(df["AUC"], df["auc_hat"]).statistic
    r = stats.pearsonr(df["AUC"], df["auc_hat"])[0]
    return {"spearman": s, "pcc": r, "n": len(df)}


def main():
    print("=" * 70)
    print("B03: Drug Network Ablation — Core Scenario B Perturbation")
    print("=" * 70)

    # ── Load model and data ──────────────────────────────────────────────────
    model, prepared, config = load_experiment(
        DRUG_SPLIT_PREPARED_PKL, DRUG_SPLIT_RESULT_DIR, DRUG_SPLIT_CONFIG_YAML, DEVICE,
    )
    preds = pd.read_csv(PREDICTIONS_CSV)
    cov   = pd.read_csv(KG_COVERAGE_CSV)
    chembl = pd.read_csv(CHEMBL_EDGES_CSV)
    primekg = pd.read_csv(PRIMEKG_EDGES_CSV)
    drkg  = pd.read_csv(DRKG_EDGES_CSV)
    cell_meta = load_cell_cancer_types(GDSC_MODEL_LIST)

    test_rows = preds[preds["split"] == "test"].copy()
    # Include all splits for fusion weight calibration
    all_rows = preds.copy()

    print(f"Test rows: {len(test_rows)} | Test drugs: {test_rows['DRUG_ID'].nunique()}")

    # ── Build session (precomputes KG payload once) ──────────────────────────
    print("\nBuilding prediction session ...")
    session = build_session(model, prepared, config, DEVICE, BATCH_SIZE)

    # ── Baseline predictions ──────────────────────────────────────────────────
    print("\nRunning baseline predictions ...")
    baseline = run_predictions(model, all_rows, prepared, config, DEVICE, BATCH_SIZE, session=session)
    baseline_test = baseline[baseline["split"] == "test"].copy()

    # ── Level 1: Full KG knockout (all_off) ──────────────────────────────────
    print("\nL1: Full KG knockout (all_off) ...")
    pred_all_off = run_predictions(model, all_rows, prepared, config, DEVICE, BATCH_SIZE,
                                   kg_mask="all_off", session=session)
    pred_all_off_test = pred_all_off[pred_all_off["split"] == "test"].copy()

    delta_all_off = baseline_test[["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "AUC", "auc_hat"]].copy()
    delta_all_off["auc_hat_ko"] = pred_all_off_test["auc_hat"].values
    delta_all_off["delta_auc"] = delta_all_off["auc_hat"] - delta_all_off["auc_hat_ko"]
    delta_all_off["ablation"] = "all_off"

    # Per-drug aggregated delta
    drug_level_all_off = (
        delta_all_off.groupby(["DRUG_ID", "DRUG_NAME"])
        .agg(
            n_cells=("SANGER_MODEL_ID", "count"),
            delta_auc_mean=("delta_auc", "mean"),
            delta_auc_std=("delta_auc", "std"),
            delta_auc_abs_mean=("delta_auc", lambda x: x.abs().mean()),
            baseline_spearman=("AUC", lambda x: compute_within_drug_spearman(
                delta_all_off.loc[x.index])["spearman"]),
            ko_spearman=("AUC", lambda x: stats.spearmanr(
                x, delta_all_off.loc[x.index, "auc_hat_ko"]
            ).statistic if len(x) >= 5 else np.nan),
        )
        .reset_index()
    )
    drug_level_all_off["delta_spearman"] = (
        drug_level_all_off["baseline_spearman"] - drug_level_all_off["ko_spearman"]
    )
    drug_level_all_off = drug_level_all_off.merge(
        cov[["DRUG_ID", "has_ChEMBL", "has_PrimeKG", "graph_degree_ChEMBL", "graph_degree_PrimeKG"]],
        on="DRUG_ID", how="left",
    )
    drug_level_all_off.to_csv(OUT_R / "drug_network_ablation_full.csv", index=False)
    print(f"Saved: {OUT_R / 'drug_network_ablation_full.csv'}")

    # ── Level 2: Source-level ablation (ChEMBL_off, PrimeKG_off) ─────────────
    source_rows = []
    for source in ["ChEMBL", "PrimeKG"]:
        mask = f"{source}_off"
        print(f"\nL2: {mask} ablation ...")
        pred_s = run_predictions(model, all_rows, prepared, config, DEVICE, BATCH_SIZE,
                                 kg_mask=mask, session=session)
        pred_s_test = pred_s[pred_s["split"] == "test"].copy()

        for _, drug_row in test_rows[["DRUG_ID", "DRUG_NAME"]].drop_duplicates().iterrows():
            ddf = baseline_test[baseline_test["DRUG_ID"] == drug_row["DRUG_ID"]].copy()
            sdf = pred_s_test[pred_s_test["DRUG_ID"] == drug_row["DRUG_ID"]].copy()
            if len(ddf) < 5:
                continue
            delta = ddf["auc_hat"].values - sdf["auc_hat"].values
            baseline_sp = compute_within_drug_spearman(ddf)["spearman"]
            ko_sp = stats.spearmanr(ddf["AUC"], sdf["auc_hat"]).statistic if len(sdf) >= 5 else np.nan
            source_rows.append({
                "DRUG_ID": drug_row["DRUG_ID"],
                "DRUG_NAME": drug_row["DRUG_NAME"],
                "source_ablated": source,
                "n_cells": len(ddf),
                "delta_auc_mean": delta.mean(),
                "delta_auc_std": delta.std(),
                "delta_auc_abs_mean": np.abs(delta).mean(),
                "baseline_spearman": baseline_sp,
                "ko_spearman": ko_sp,
                "delta_spearman": baseline_sp - ko_sp if not np.isnan(ko_sp) else np.nan,
            })

    source_df = pd.DataFrame(source_rows)
    source_df = source_df.merge(
        cov[["DRUG_ID", "has_ChEMBL", "has_PrimeKG"]].drop_duplicates(),
        on="DRUG_ID", how="left",
    )
    source_df.to_csv(OUT_R / "drug_source_ablation.csv", index=False)
    print(f"Saved: {OUT_R / 'drug_source_ablation.csv'}")

    # ── Level 3: Drug-specific edge ablation for focal drugs ──────────────────
    print("\nL3: Drug-specific edge ablation for focal drugs ...")
    specific_rows = []
    cancer_rows   = []

    for drug_name, info in FOCAL_TEST_DRUGS.items():
        drug_id = info["drug_id"]
        ddf_base = baseline_test[baseline_test["DRUG_ID"] == drug_id].copy()
        if len(ddf_base) < 5:
            print(f"  Skipping {drug_name}: insufficient test rows")
            continue

        edges = get_drug_kg_edges(drug_id, chembl, drkg, primekg)
        all_edge_ids = edges["all"]
        if not all_edge_ids:
            print(f"  {drug_name}: no drug-specific edges → skip")
            continue

        print(f"  {drug_name}: {len(all_edge_ids)} drug-specific edges → ablating ...")
        pred_drug_ko = run_predictions(
            model, all_rows, prepared, config, DEVICE, BATCH_SIZE,
            kg_mask={"edge_ids": all_edge_ids}, session=session,
        )
        pred_drug_ko_test = pred_drug_ko[(pred_drug_ko["split"] == "test") &
                                         (pred_drug_ko["DRUG_ID"] == drug_id)].copy()

        delta = ddf_base["auc_hat"].values - pred_drug_ko_test["auc_hat"].values
        baseline_sp = compute_within_drug_spearman(ddf_base)["spearman"]
        ko_sp = stats.spearmanr(ddf_base["AUC"], pred_drug_ko_test["auc_hat"]).statistic

        specific_rows.append({
            "DRUG_NAME": drug_name,
            "drug_id": drug_id,
            "target": info["target"],
            "family": info["family"],
            "expected_cancer": info["cancer"],
            "n_edges_ablated": len(all_edge_ids),
            "n_chembl_edges": len(edges["ChEMBL"]),
            "n_primekg_edges": len(edges["PrimeKG"]),
            "n_cells": len(ddf_base),
            "delta_auc_mean": delta.mean(),
            "delta_auc_std": delta.std(),
            "delta_auc_abs_mean": np.abs(delta).mean(),
            "baseline_spearman": baseline_sp,
            "ko_spearman": ko_sp,
            "delta_spearman": baseline_sp - ko_sp,
        })

        # Cancer-type-specific delta
        ddf_annotated = ddf_base.copy()
        ddf_annotated["auc_hat_ko"] = pred_drug_ko_test["auc_hat"].values
        ddf_annotated["delta_auc"] = ddf_annotated["auc_hat"] - ddf_annotated["auc_hat_ko"]
        ddf_annotated = ddf_annotated.merge(cell_meta, on="SANGER_MODEL_ID", how="left")
        ddf_annotated["cancer_group"] = ddf_annotated["cancer_type"].apply(
            lambda x: annotate_cancer_group(x, CANCER_GROUPS)
        )

        for cg in ddf_annotated["cancer_group"].unique():
            cdf = ddf_annotated[ddf_annotated["cancer_group"] == cg]
            if len(cdf) < 3:
                continue
            cancer_rows.append({
                "DRUG_NAME": drug_name,
                "drug_id": drug_id,
                "target": info["target"],
                "expected_cancer": info["cancer"],
                "cancer_group": cg,
                "n_cells": len(cdf),
                "delta_auc_mean": cdf["delta_auc"].mean(),
                "delta_auc_std": cdf["delta_auc"].std(),
                "baseline_spearman": stats.spearmanr(cdf["AUC"], cdf["auc_hat"]).statistic
                                     if len(cdf) >= 5 else np.nan,
                "ko_spearman": stats.spearmanr(cdf["AUC"], cdf["auc_hat_ko"]).statistic
                               if len(cdf) >= 5 else np.nan,
                "is_expected": (cg == info["cancer"]),
            })

    specific_df  = pd.DataFrame(specific_rows)
    cancer_df    = pd.DataFrame(cancer_rows)
    specific_df.to_csv(OUT_R / "drug_specific_edge_ablation.csv", index=False)
    cancer_df.to_csv(OUT_R / "cancer_specific_delta.csv", index=False)
    print(f"Saved: {OUT_R / 'drug_specific_edge_ablation.csv'}")
    print(f"Saved: {OUT_R / 'cancer_specific_delta.csv'}")

    # ── Figures ──────────────────────────────────────────────────────────────
    _plot_ablation(drug_level_all_off, source_df, specific_df, cancer_df, OUT_F)
    print("\nB03 complete.")


def _plot_ablation(drug_level_all_off, source_df, specific_df, cancer_df, out_dir):
    """Generate publication figure B03."""
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)

    # Panel A: Full KG knockout — per drug delta spearman
    ax1 = fig.add_subplot(gs[0, :2])
    d = drug_level_all_off.dropna(subset=["delta_spearman"])
    has_kg = (d["has_ChEMBL"].fillna(0).astype(bool)) | (d["has_PrimeKG"].fillna(0).astype(bool))
    d_kg = d[has_kg].sort_values("delta_spearman", ascending=False)
    d_nokg = d[~has_kg]
    x = np.arange(len(d_kg))
    colors = ["#E74C3C" if v > 0.05 else "#3498DB" for v in d_kg["delta_spearman"]]
    ax1.bar(x, d_kg["delta_spearman"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(d_kg["DRUG_NAME"], rotation=45, ha="right", fontsize=8)
    ax1.axhline(0, color="black", linewidth=1.2)
    ax1.axhline(d_nokg["delta_spearman"].mean(), color="gray", linestyle="--",
                linewidth=1.5, label=f"No-KG mean ΔSPEARMAN={d_nokg['delta_spearman'].mean():.3f}")
    ax1.set_ylabel("ΔSpearman (baseline − KO)")
    ax1.set_title("A. Full KG Knockout for KG-Covered Test Drugs\n"
                  "(positive = KG improves prediction; model never saw these drugs)",
                  fontweight="bold")
    ax1.legend(fontsize=9)

    # Panel B: Source-level ablation
    ax2 = fig.add_subplot(gs[0, 2])
    chembl_delta = source_df[(source_df["source_ablated"] == "ChEMBL") &
                              (source_df["has_ChEMBL"])]["delta_spearman"].dropna()
    primekg_delta = source_df[(source_df["source_ablated"] == "PrimeKG") &
                               (source_df["has_PrimeKG"])]["delta_spearman"].dropna()
    bx = ax2.boxplot([chembl_delta, primekg_delta],
                     labels=["ChEMBL\nablation", "PrimeKG\nablation"],
                     patch_artist=True,
                     boxprops=dict(facecolor="lightblue"),
                     medianprops=dict(color="red", linewidth=2))
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("ΔSpearman (baseline − ablated)")
    if len(chembl_delta) >= 3 and len(primekg_delta) >= 3:
        _, pv = stats.mannwhitneyu(chembl_delta, primekg_delta, alternative="two-sided")
        ax2.set_title(f"B. Source-Level Ablation\nChEMBL vs PrimeKG (p={pv:.3f})",
                      fontweight="bold")
    else:
        ax2.set_title("B. Source-Level Ablation", fontweight="bold")

    # Panel C: Drug-specific edge ablation — focal drugs
    if len(specific_df) > 0:
        ax3 = fig.add_subplot(gs[1, :2])
        sp = specific_df.sort_values("delta_spearman", ascending=False)
        x3 = np.arange(len(sp))
        colors3 = ["#E74C3C" if v > 0 else "#2ECC71" for v in sp["delta_spearman"]]
        bars = ax3.bar(x3, sp["delta_spearman"], color=colors3,
                       edgecolor="black", linewidth=0.5, alpha=0.85)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(sp["DRUG_NAME"], rotation=30, ha="right", fontsize=9)
        ax3.set_ylabel("ΔSpearman (baseline − drug-edge KO)")
        ax3.set_title("C. Drug-Specific Edge Ablation (only drug's own KG edges removed)\n"
                      "Demonstrates mechanism-specific KG utilization",
                      fontweight="bold")
        ax3.axhline(0, color="black", linewidth=1.2)
        for bar, n in zip(bars, sp["n_edges_ablated"]):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"n={n}", ha="center", va="bottom", fontsize=7)

    # Panel D: Cancer-type specificity heatmap
    if len(cancer_df) > 0:
        ax4 = fig.add_subplot(gs[1, 2])
        pivot = cancer_df.pivot_table(
            index="DRUG_NAME", columns="cancer_group",
            values="delta_auc_mean", aggfunc="mean",
        )
        # Highlight expected cancers
        im = ax4.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=-0.02, vmax=0.02)
        ax4.set_xticks(range(len(pivot.columns)))
        ax4.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
        ax4.set_yticks(range(len(pivot.index)))
        ax4.set_yticklabels(pivot.index, fontsize=8)
        plt.colorbar(im, ax=ax4, label="Mean ΔAUC after drug-edge KO")
        # Mark expected cancer type
        for i, drug in enumerate(pivot.index):
            expected = FOCAL_TEST_DRUGS.get(drug, {}).get("cancer", "")
            for j, ct in enumerate(pivot.columns):
                if ct == expected and not np.isnan(pivot.values[i, j]):
                    ax4.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                fill=False, edgecolor="black", linewidth=2))
        ax4.set_title("D. Cancer-Type-Specific KG Effect\n(★ = expected cancer type)",
                      fontweight="bold")

    plt.suptitle(
        "Scenario B — B03: Drug Network Ablation Reveals\n"
        "Mechanism-Specific KG Utilization for Unseen Test Drugs",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(out_dir / "B03_drug_network_ablation.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "B03_drug_network_ablation.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_dir / 'B03_drug_network_ablation.pdf'}")


if __name__ == "__main__":
    main()
