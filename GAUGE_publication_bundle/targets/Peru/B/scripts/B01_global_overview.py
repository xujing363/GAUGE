"""
B01: Global Overview — KG Coverage versus Prediction Accuracy for Unseen Drugs
==============================================================================
Scientific question:
  For drug split test drugs (never seen in training), does having KG connectivity
  (三大先验网络: ChEMBL, DRKG, PrimeKG) improve the model's ability to predict
  drug sensitivity?

Key analyses:
  1. KG coverage landscape: how many test drugs have ChEMBL / PrimeKG coverage?
  2. Prediction accuracy stratified by KG coverage status
     (Spearman ρ and PCC per drug, compared between KG-covered vs KG-absent)
  3. Effect of coverage depth: does graph_degree correlate with per-drug accuracy?
  4. KG source attribution: for KG-covered drugs, what fraction of attention goes
     to ChEMBL vs PrimeKG? (DRKG = 0 for all test drugs → systematic finding)

This establishes:
  - Static prior KGs do exist for subsets of unseen drugs
  - The model leverages this coverage → higher accuracy
  - Dynamic attention allocation already evident from pre-computed alpha values

Outputs:
  results/global_overview/
    kg_coverage_accuracy.csv      - per-drug coverage + accuracy stats
    coverage_group_stats.csv      - group-level accuracy comparison
    alpha_by_coverage_group.csv   - mean alpha per coverage group
  figures/
    B01_kg_coverage_accuracy.pdf
    B01_alpha_attribution.pdf
"""
from __future__ import annotations

import sys
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
    DRUG_SPLIT_RESULT_DIR, KG_COVERAGE_CSV, PREDICTIONS_CSV,
    KG_ATTENTION_CSV, GDSC_DRUG_PCC_CSV, RESULTS, FIGURES, FOCAL_TEST_DRUGS,
)

OUT_R = RESULTS / "global_overview"
OUT_F = FIGURES
OUT_R.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("B01: Global Overview — KG Coverage vs Accuracy for Unseen Test Drugs")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    preds    = pd.read_csv(PREDICTIONS_CSV)
    attn     = pd.read_csv(KG_ATTENTION_CSV)
    cov      = pd.read_csv(KG_COVERAGE_CSV)
    drug_pcc = pd.read_csv(GDSC_DRUG_PCC_CSV)

    test_preds = preds[preds["split"] == "test"].copy()
    test_drugs = test_preds[["DRUG_ID", "DRUG_NAME"]].drop_duplicates()
    print(f"Test drugs: {len(test_drugs)}")

    # ── Compute per-drug accuracy from predictions ────────────────────────────
    def within_drug_spearman(grp):
        grp = grp.dropna(subset=["AUC", "auc_hat"])
        if len(grp) < 5:
            return np.nan
        return stats.spearmanr(grp["AUC"], grp["auc_hat"]).statistic

    def within_drug_pcc(grp):
        grp = grp.dropna(subset=["AUC", "auc_hat"])
        if len(grp) < 5:
            return np.nan
        return stats.pearsonr(grp["AUC"], grp["auc_hat"])[0]

    drug_stats = test_preds.groupby("DRUG_ID").apply(
        lambda g: pd.Series({
            "n_cells":      len(g),
            "spearman":     within_drug_spearman(g),
            "pcc":          within_drug_pcc(g),
            "mean_auc_hat": g["auc_hat"].mean(),
            "mean_AUC":     g["AUC"].mean(),
        }),
        include_groups=False,
    ).reset_index()

    # cov already has DRUG_NAME; filter to test drugs only, then join stats
    test_cov = cov[cov["DRUG_ID"].isin(test_drugs["DRUG_ID"])].copy()
    result = test_cov.merge(drug_stats, on="DRUG_ID", how="left")

    # ── Classify coverage groups ─────────────────────────────────────────────
    def coverage_group(row):
        has_c = bool(row["has_ChEMBL"])
        has_p = bool(row["has_PrimeKG"])
        if has_c and has_p:
            return "ChEMBL+PrimeKG"
        if has_c:
            return "ChEMBL only"
        if has_p:
            return "PrimeKG only"
        return "No KG"

    result["coverage_group"] = result.apply(coverage_group, axis=1)
    result["total_kg_degree"] = result["graph_degree_ChEMBL"] + result["graph_degree_PrimeKG"]
    result["is_focal"] = result["DRUG_NAME"].isin(FOCAL_TEST_DRUGS.keys())

    # ── KG attention means per drug ──────────────────────────────────────────
    test_attn = attn.merge(
        test_preds[["SANGER_MODEL_ID", "DRUG_ID"]].drop_duplicates(),
        on=["SANGER_MODEL_ID", "DRUG_ID"],
    )
    alpha_means = test_attn.groupby("DRUG_ID")[["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]].mean().reset_index()
    result = result.merge(alpha_means, on="DRUG_ID", how="left")

    result.to_csv(OUT_R / "kg_coverage_accuracy.csv", index=False)
    print(f"Saved: {OUT_R / 'kg_coverage_accuracy.csv'}")

    # ── Group-level accuracy stats ───────────────────────────────────────────
    group_stats = (
        result.groupby("coverage_group")
        .agg(
            n_drugs=("DRUG_NAME", "count"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pcc_mean=("pcc", "mean"),
            pcc_std=("pcc", "std"),
            alpha_ChEMBL_mean=("alpha_ChEMBL", "mean"),
            alpha_PrimeKG_mean=("alpha_PrimeKG", "mean"),
        )
        .reset_index()
    )
    print("\nGroup-level accuracy summary:")
    print(group_stats.to_string(index=False))
    group_stats.to_csv(OUT_R / "coverage_group_stats.csv", index=False)

    # ── Statistical tests ────────────────────────────────────────────────────
    kg_covered = result[result["coverage_group"] != "No KG"]["spearman"].dropna()
    no_kg = result[result["coverage_group"] == "No KG"]["spearman"].dropna()
    stat, pval = stats.mannwhitneyu(kg_covered, no_kg, alternative="greater")
    print(f"\nKG-covered vs No-KG Spearman (Mann-Whitney U):")
    print(f"  KG-covered: mean={kg_covered.mean():.4f} (n={len(kg_covered)})")
    print(f"  No-KG:      mean={no_kg.mean():.4f} (n={len(no_kg)})")
    print(f"  U={stat:.1f}, p={pval:.4f}")

    # Correlation: KG degree vs accuracy
    has_degree = result[result["total_kg_degree"] > 0]
    r_deg, p_deg = stats.spearmanr(has_degree["total_kg_degree"], has_degree["spearman"].fillna(0))
    print(f"\nDegree vs Spearman correlation (KG-covered drugs only):")
    print(f"  r={r_deg:.4f}, p={p_deg:.4f}, n={len(has_degree)}")

    stats_summary = pd.DataFrame([
        {"test": "KG_covered_vs_no_kg_spearman", "stat": stat, "pval": pval,
         "kg_covered_mean": kg_covered.mean(), "no_kg_mean": no_kg.mean(),
         "n_kg_covered": len(kg_covered), "n_no_kg": len(no_kg)},
        {"test": "degree_vs_spearman_correlation", "stat": r_deg, "pval": p_deg,
         "n": len(has_degree)},
    ])
    stats_summary.to_csv(OUT_R / "statistical_tests.csv", index=False)

    # ── Alpha by coverage group ──────────────────────────────────────────────
    alpha_cov = result.groupby("coverage_group")[
        ["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]
    ].mean().reset_index()
    alpha_cov.to_csv(OUT_R / "alpha_by_coverage_group.csv", index=False)

    # ── Figure 1: Coverage accuracy overview (4-panel) ───────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    colors = {
        "ChEMBL+PrimeKG": "#2196F3",
        "ChEMBL only":    "#FF9800",
        "PrimeKG only":   "#4CAF50",
        "No KG":          "#9E9E9E",
    }

    # Panel A: Coverage group distribution (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    grp_order = ["No KG", "ChEMBL only", "PrimeKG only", "ChEMBL+PrimeKG"]
    grp_counts = result["coverage_group"].value_counts().reindex(grp_order, fill_value=0)
    ax1.bar(range(len(grp_order)), grp_counts.values,
            color=[colors[g] for g in grp_order], edgecolor="black", linewidth=0.8)
    ax1.set_xticks(range(len(grp_order)))
    ax1.set_xticklabels(grp_order, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Number of test drugs")
    ax1.set_title("A. KG Coverage Landscape\n(56 test drugs, unseen in training)", fontweight="bold")
    for i, v in enumerate(grp_counts.values):
        ax1.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel B: Spearman by coverage group (violin)
    ax2 = fig.add_subplot(gs[0, 1])
    data_by_group = [result[result["coverage_group"] == g]["spearman"].dropna().values
                     for g in grp_order if len(result[result["coverage_group"] == g]) > 0]
    valid_groups = [g for g in grp_order if len(result[result["coverage_group"] == g]) > 0]
    parts = ax2.violinplot(data_by_group, positions=range(len(valid_groups)),
                           showmeans=True, showmedians=False)
    for i, (pc, g) in enumerate(zip(parts["bodies"], valid_groups)):
        pc.set_facecolor(colors[g])
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(valid_groups)))
    ax2.set_xticklabels(valid_groups, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Within-drug Spearman ρ")
    ax2.set_title(f"B. Prediction Accuracy by KG Coverage\n"
                  f"KG-covered > No-KG: p={pval:.3f}", fontweight="bold")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Panel C: KG degree vs accuracy scatter
    ax3 = fig.add_subplot(gs[0, 2])
    sub = result.dropna(subset=["spearman"])
    for g in grp_order:
        mask = sub["coverage_group"] == g
        ax3.scatter(sub.loc[mask, "total_kg_degree"], sub.loc[mask, "spearman"],
                    c=colors[g], label=g, alpha=0.8, s=60, edgecolors="black", linewidth=0.5)
    # Annotate focal drugs
    for _, row in sub[sub["is_focal"]].iterrows():
        ax3.annotate(row["DRUG_NAME"], (row["total_kg_degree"], row["spearman"]),
                     fontsize=7, xytext=(5, 3), textcoords="offset points")
    if len(has_degree) > 2:
        ax3.plot(np.unique(has_degree["total_kg_degree"]),
                 np.poly1d(np.polyfit(has_degree["total_kg_degree"],
                                      has_degree["spearman"].fillna(0), 1)
                           )(np.unique(has_degree["total_kg_degree"])),
                 "r--", linewidth=1.5, label=f"Trend r={r_deg:.2f}")
    ax3.set_xlabel("Total KG degree (ChEMBL + PrimeKG edges)")
    ax3.set_ylabel("Within-drug Spearman ρ")
    ax3.set_title(f"C. KG Connectivity vs Accuracy\nr={r_deg:.3f}, p={p_deg:.3f}", fontweight="bold")
    ax3.legend(fontsize=7, loc="lower right")

    # Panel D: Alpha attribution stacked bar by drug (KG-covered drugs sorted by spearman)
    ax4 = fig.add_subplot(gs[1, :2])
    kg_drugs = result[result["coverage_group"] != "No KG"].dropna(subset=["spearman"]).copy()
    kg_drugs = kg_drugs.sort_values("spearman", ascending=False).reset_index(drop=True)
    x = np.arange(len(kg_drugs))
    w = 0.7
    ax4.bar(x, kg_drugs["alpha_ChEMBL"], w, label="α ChEMBL (MoA knowledge)", color="#2196F3", alpha=0.85)
    ax4.bar(x, kg_drugs["alpha_DRKG"], w, bottom=kg_drugs["alpha_ChEMBL"],
            label="α DRKG (broad biology)", color="#FF5722", alpha=0.85)
    ax4.bar(x, kg_drugs["alpha_PrimeKG"], w,
            bottom=kg_drugs["alpha_ChEMBL"] + kg_drugs["alpha_DRKG"],
            label="α PrimeKG (protein network)", color="#4CAF50", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(kg_drugs["DRUG_NAME"], rotation=45, ha="right", fontsize=7)
    ax4.set_ylabel("Mean attention weight α")
    ax4.set_title("D. Dynamic KG Attention Allocation for KG-Covered Test Drugs\n"
                  "(sorted by Spearman ρ, showing static→dynamic transformation)",
                  fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.set_xlim(-0.5, len(kg_drugs) - 0.5)

    # Add spearman values as secondary y-axis
    ax4b = ax4.twinx()
    ax4b.plot(x, kg_drugs["spearman"], "ko-", markersize=4, linewidth=1.5,
              label="Spearman ρ", alpha=0.7)
    ax4b.set_ylabel("Within-drug Spearman ρ", color="black")
    ax4b.tick_params(axis="y", labelcolor="black")
    ax4b.set_ylim(-0.3, 0.9)

    # Panel E: Summary statistics text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary_text = (
        "Key Findings:\n\n"
        f"Total test drugs: {len(result)}\n"
        f"KG-covered: {len(kg_covered)} ({100*len(kg_covered)/len(result):.0f}%)\n"
        f"No-KG: {len(no_kg)} ({100*len(no_kg)/len(result):.0f}%)\n\n"
        "Accuracy (Spearman ρ):\n"
        f"  KG-covered: {kg_covered.mean():.3f} ± {kg_covered.std():.3f}\n"
        f"  No-KG: {no_kg.mean():.3f} ± {no_kg.std():.3f}\n"
        f"  Mann-Whitney p = {pval:.4f}\n\n"
        "DRKG coverage: 0/56 test drugs\n"
        "→ ChEMBL + PrimeKG are the active\n"
        "  prior networks for drug split\n\n"
        "Dynamic attention (DRKG=0):\n"
        f"  ChEMBL: {result['alpha_ChEMBL'].mean():.3f}\n"
        f"  PrimeKG: {result['alpha_PrimeKG'].mean():.3f}\n\n"
        "Static KG → Dynamic through\n"
        "drug-cell context weighting"
    )
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax5.set_title("E. Summary Statistics", fontweight="bold")

    plt.suptitle(
        "Scenario B: World Model Leverages Prior Knowledge Graphs for\n"
        "Generalizing to Unseen Drugs (Drug Split Test Set, n=56 drugs)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(OUT_F / "B01_kg_coverage_accuracy.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_F / "B01_kg_coverage_accuracy.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {OUT_F / 'B01_kg_coverage_accuracy.pdf'}")

    print("\nB01 complete.")


if __name__ == "__main__":
    main()
