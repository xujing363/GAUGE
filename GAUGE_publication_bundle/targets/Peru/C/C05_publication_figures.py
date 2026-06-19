"""
Scenario C — Step 5: Publication-Quality Figures
=================================================

Generates all figures for the synthetic lethality scenario:

Figure 1: Overview — World Model Perturbation Framework
  1a. Schematic: static KG → dynamic drug-context SL via α-gates (text/diagram)
  1b. Heatmap: delta_AUC per (drug, gene) for top 20 genes × top 15 drugs
  1c. Known target recovery: delta_AUC for known drug–target vs. random genes

Figure 2: Drug-Gene Perturbation Landscape
  2a. Volcano plot: mean_delta_AUC vs. -log10(fraction sensitised)
  2b. Scatter: top sensitising genes across EGFR inhibitors
  2c. Scatter: top sensitising genes for Venetoclax (BCL2 inhibitor)

Figure 3: SynLeth Validation
  3a. Enrichment bar chart: observed vs. expected SL overlaps
  3b. Venn-style: validated / novel / NONSL breakdown
  3c. Ranking: does SynLeth-SL pair rank higher in our predictions?

Figure 4: Cancer-Type Specificity
  4a. Heatmap: cancer type × target gene, mean delta_AUC
  4b. Lung adenocarcinoma: EGFR inhibitor SL landscape
  4c. AML: Venetoclax SL landscape

Figure 5: KG Prior Contribution (World Model Transparency)
  5a. Bar: mean α_ChEMBL / α_DRKG / α_PrimeKG per cancer type
  5b. Scatter: α_PrimeKG vs. gene perturbation effect
  5c. Static vs. dynamic KG: comparison of SL recovery rates

All figures saved to figures/ directory (PDF + PNG).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CANCER_TYPE_FOCUS,
    DOUBLE_DISJOINT_RESULT_DIR,
    FIGURES_DIR,
    GDSC_MODEL_LIST,
    KNOWN_DRUG_TARGETS,
    RESULTS_DIR,
)
from utils import load_cell_metadata

PALETTE = {
    "primary":      "#2E4057",
    "accent":       "#E84855",
    "highlight":    "#F5A623",
    "kg_chembl":    "#4A90D9",
    "kg_drkg":      "#7ED321",
    "kg_primekg":   "#BD10E0",
    "grey":         "#9B9B9B",
    "validated":    "#27AE60",
    "novel":        "#E84855",
    "non_sl":       "#BDC3C7",
}

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":   150,
    "savefig.bbox": "tight",
})


def savefig(fig, name: str) -> None:
    for ext in ["pdf", "png"]:
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {FIGURES_DIR / name}.pdf/png")
    plt.close(fig)


def load_results() -> dict[str, pd.DataFrame | None]:
    loaders = {
        "drug_gene":    RESULTS_DIR / "C01_drug_gene" / "drug_gene_delta_auc.csv",
        "top_sens":     RESULTS_DIR / "C01_drug_gene" / "top_sensitising_genes.csv",
        "known_target": RESULTS_DIR / "C01_drug_gene" / "known_target_validation.csv",
        "sl_full":      RESULTS_DIR / "C03_sl_prediction" / "sl_candidates_full.csv",
        "sl_confident": RESULTS_DIR / "C03_sl_prediction" / "sl_candidates_confident.csv",
        "sl_novel":     RESULTS_DIR / "C03_sl_prediction" / "novel_sl_predictions.csv",
        "sl_enrich":    RESULTS_DIR / "C03_sl_prediction" / "synleth_validation.csv",
        "cancer_sl":    RESULTS_DIR / "C03_sl_prediction" / "cancer_specific_sl.csv",
        "alpha_gate":   RESULTS_DIR / "C04_cancer_analysis" / "alpha_gate_by_cancer.csv",
        "wm_contrib":   RESULTS_DIR / "C04_cancer_analysis" / "world_model_kgcontrib.csv",
        "kg_attn":      DOUBLE_DISJOINT_RESULT_DIR / "kg_attention_by_prediction.csv",
    }
    data = {}
    for key, path in loaders.items():
        try:
            data[key] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  Warning: {key} not found at {path}")
            data[key] = None
    return data


# ─── Figure 1b: Drug × Gene delta_AUC heatmap ─────────────────────────────────
def fig_drug_gene_heatmap(drug_gene_df: pd.DataFrame) -> None:
    import matplotlib.colors as mcolors

    # Select top genes (highest mean |delta_AUC| across all drugs)
    gene_rank = (
        drug_gene_df.groupby("gene_name")["mean_delta_auc"]
        .mean()
        .sort_values(ascending=False)
    )
    top_genes = gene_rank.head(20).index.tolist()

    # Select top drugs (highest variance of delta_AUC)
    drug_rank = (
        drug_gene_df.groupby("DRUG_NAME")["mean_delta_auc"]
        .std()
        .sort_values(ascending=False)
    )
    top_drugs = drug_rank.head(15).index.tolist()

    pivot = drug_gene_df[
        drug_gene_df["gene_name"].isin(top_genes) &
        drug_gene_df["DRUG_NAME"].isin(top_drugs)
    ].pivot_table(index="DRUG_NAME", columns="gene_name", values="mean_delta_auc", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.RdBu_r
    vmax = np.percentile(np.abs(pivot.values[~np.isnan(pivot.values)]), 95)
    im = ax.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean ΔAUC (gene silencing → drug)", fontsize=10)
    ax.set_title(
        "Drug × Gene Perturbation Landscape\n"
        "(Δ > 0: gene silencing sensitises cell to drug; World Model, double-disjoint split)",
        fontsize=12,
    )
    savefig(fig, "C_fig1b_drug_gene_heatmap")


# ─── Figure 1c: Known target recovery ────────────────────────────────────────
def fig_known_target_recovery(drug_gene_df: pd.DataFrame) -> None:
    known_pairs = []
    for drug_name, targets in KNOWN_DRUG_TARGETS.items():
        drug_rows = drug_gene_df[drug_gene_df["DRUG_NAME"] == drug_name]
        for t in targets:
            if t == "DNA":
                continue
            row = drug_rows[drug_rows["gene_name"] == t]
            if row.empty:
                continue
            known_pairs.append({
                "drug": drug_name, "target": t,
                "delta": row.iloc[0]["mean_delta_auc"],
                "rank":  (drug_rows["mean_delta_auc"] > row.iloc[0]["mean_delta_auc"]).sum() + 1,
                "n_genes": len(drug_rows),
            })
    if not known_pairs:
        return
    kp_df = pd.DataFrame(known_pairs)
    kp_df["rank_frac"] = kp_df["rank"] / kp_df["n_genes"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: delta_AUC for known targets
    ax = axes[0]
    colors = [PALETTE["accent"] if d > 0.002 else PALETTE["grey"] for d in kp_df["delta"]]
    bars = ax.barh(
        [f"{r.drug}\n({r.target})" for _, r in kp_df.iterrows()],
        kp_df["delta"], color=colors
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean ΔAUC (gene silencing)")
    ax.set_title("Known Drug Targets: ΔAUC upon Silencing\n(positive = target supports resistance)")
    ax.axvline(0.002, color=PALETTE["accent"], linewidth=1, linestyle=":")

    # Panel B: rank of known target among all perturbed genes
    ax = axes[1]
    rank_pct = 100 * (1 - kp_df["rank_frac"])
    colors2 = [PALETTE["validated"] if r > 80 else PALETTE["highlight"] if r > 50 else PALETTE["grey"]
               for r in rank_pct]
    ax.barh(
        [f"{r.drug}\n({r.target})" for _, r in kp_df.iterrows()],
        rank_pct, color=colors2
    )
    ax.axvline(80, color=PALETTE["validated"], linewidth=1, linestyle=":")
    ax.set_xlabel("Percentile rank among all perturbed genes")
    ax.set_title("Known Target Ranking\n(>80th percentile = correct recovery)")
    ax.set_xlim(0, 100)

    plt.tight_layout()
    savefig(fig, "C_fig1c_known_target_recovery")


# ─── Figure 2a: Volcano plot ──────────────────────────────────────────────────
def fig_volcano(drug_gene_df: pd.DataFrame) -> None:
    # Aggregate across all drugs per gene
    gene_agg = (
        drug_gene_df.groupby("gene_name")
        .agg(
            mean_delta   = ("mean_delta_auc", "mean"),
            frac_sens    = ("frac_sensitising", "mean"),
            is_target    = ("is_known_target", "any"),
        )
        .reset_index()
    )
    gene_agg["neg_log_frac"] = -np.log10(1 - gene_agg["frac_sens"].clip(0, 0.9999))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Background genes
    mask_bg = (~gene_agg["is_target"]) & (gene_agg["mean_delta"] < 0.005)
    ax.scatter(
        gene_agg.loc[mask_bg, "mean_delta"],
        gene_agg.loc[mask_bg, "neg_log_frac"],
        s=8, alpha=0.3, color=PALETTE["grey"], zorder=1, label="Background"
    )

    # Sensitising genes
    mask_sens = (~gene_agg["is_target"]) & (gene_agg["mean_delta"] >= 0.005)
    ax.scatter(
        gene_agg.loc[mask_sens, "mean_delta"],
        gene_agg.loc[mask_sens, "neg_log_frac"],
        s=20, alpha=0.6, color=PALETTE["accent"], zorder=2, label="Sensitising gene"
    )

    # Known drug targets
    mask_target = gene_agg["is_target"]
    ax.scatter(
        gene_agg.loc[mask_target, "mean_delta"],
        gene_agg.loc[mask_target, "neg_log_frac"],
        s=80, alpha=0.9, color=PALETTE["primary"], marker="*", zorder=3, label="Known drug target"
    )
    for _, r in gene_agg[mask_target].iterrows():
        ax.annotate(r.gene_name, (r.mean_delta, r.neg_log_frac),
                    fontsize=8, xytext=(4, 2), textcoords="offset points")

    ax.axvline(0.005, color=PALETTE["accent"], linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Mean ΔAUC (gene silencing across all drugs)")
    ax.set_ylabel("-log₁₀(1 - fraction sensitised)")
    ax.set_title(
        "Gene Perturbation Landscape\n"
        "World Model (double-disjoint): drug-sensitising gene discovery",
        fontsize=12,
    )
    ax.legend(framealpha=0.9)
    savefig(fig, "C_fig2a_volcano")


# ─── Figure 3: SynLeth Validation ────────────────────────────────────────────
def fig_synleth_validation(sl_full: pd.DataFrame, sl_confident: pd.DataFrame,
                            enrich_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: bar chart — observed vs expected
    if enrich_df is not None and "enrichment" in enrich_df.columns:
        ax = axes[0]
        labels = enrich_df["subset"].tolist()
        enrichments = enrich_df["enrichment"].tolist()
        colors_e = [PALETTE["validated"] if e > 1 else PALETTE["grey"] for e in enrichments]
        bars = ax.bar(labels, enrichments, color=colors_e)
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_ylabel("Enrichment (observed / expected)")
        ax.set_title("SynLeth-SL Enrichment\nof Predicted Candidates")
        for bar, pval in zip(bars, enrich_df.get("pval", [None]*len(bars))):
            if pval is not None:
                label = f"p={pval:.2e}" if pval < 0.05 else "n.s."
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        label, ha="center", fontsize=9)

    # Panel B: composition of confident predictions
    ax = axes[1]
    if sl_confident is not None and len(sl_confident) > 0:
        n_validated = sl_confident["in_synleth_SL"].sum() + sl_confident["in_synleth_SDL"].sum()
        n_nonsl     = sl_confident["in_synleth_NONSL"].sum()
        n_novel     = sl_confident["novel"].sum()
        labels2 = ["Validated (SL/SDL)", "Novel (not in DB)", "NONSL (excluded)"]
        sizes   = [n_validated, n_novel, n_nonsl]
        colors2 = [PALETTE["validated"], PALETTE["novel"], PALETTE["non_sl"]]
        wedge_props = {"edgecolor": "white", "linewidth": 2}
        ax.pie(sizes, labels=labels2, colors=colors2, autopct="%1.1f%%",
               startangle=90, wedgeprops=wedge_props)
        ax.set_title(f"Confident SL Candidates (N={len(sl_confident)})\nBreakdown by SynLeth annotation")

    # Panel C: rank distribution — SL pairs vs. non-SL pairs
    ax = axes[2]
    if sl_full is not None and len(sl_full) > 0:
        sl_pairs_delta   = sl_full[sl_full["in_synleth_SL"]]["mean_delta_auc"].values
        nonsl_pairs_delta = sl_full[sl_full["in_synleth_NONSL"]]["mean_delta_auc"].values
        random_delta      = sl_full[~sl_full["in_synleth_SL"] & ~sl_full["in_synleth_NONSL"]]["mean_delta_auc"].values

        bins = np.linspace(-0.01, max(sl_full["mean_delta_auc"].max(), 0.02), 40)
        if len(sl_pairs_delta) > 0:
            ax.hist(sl_pairs_delta,   bins=bins, density=True, alpha=0.6,
                    color=PALETTE["validated"], label=f"SynLeth-SL (n={len(sl_pairs_delta)})")
        if len(nonsl_pairs_delta) > 0:
            ax.hist(nonsl_pairs_delta, bins=bins, density=True, alpha=0.6,
                    color=PALETTE["non_sl"], label=f"SynLeth-NONSL (n={len(nonsl_pairs_delta)})")
        if len(random_delta) > 5:
            ax.hist(random_delta[:2000], bins=bins, density=True, alpha=0.3,
                    color=PALETTE["grey"], label="Background pairs")
        ax.set_xlabel("Mean ΔAUC (gene silencing)")
        ax.set_ylabel("Density")
        ax.set_title("ΔAUC Distribution by SynLeth Annotation\n(SL pairs should have higher ΔAUC)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, "C_fig3_synleth_validation")


# ─── Figure 4: Cancer-type heatmap ───────────────────────────────────────────
def fig_cancer_heatmap(cancer_sl: pd.DataFrame) -> None:
    if cancer_sl is None or cancer_sl.empty:
        return
    pivot = cancer_sl.pivot_table(
        index="cancer_type", columns="target_gene",
        values="mean_delta_auc", aggfunc="mean"
    ).fillna(0)

    # Select top genes across all cancer types
    gene_mean = pivot.abs().mean(axis=0).sort_values(ascending=False)
    top_genes = gene_mean.head(15).index.tolist()
    pivot = pivot[[g for g in top_genes if g in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))
    vmax = np.percentile(np.abs(pivot.values.flatten()), 95) + 1e-6
    cmap = plt.cm.RdBu_r
    im = ax.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Mean ΔAUC", fontsize=10)
    ax.set_title(
        "Cancer-Type × Target Gene Perturbation Heatmap\n"
        "(Cell-target silencing effect on drug sensitivity)",
        fontsize=12,
    )
    plt.tight_layout()
    savefig(fig, "C_fig4_cancer_type_heatmap")


# ─── Figure 5: KG prior contribution ─────────────────────────────────────────
def fig_kg_contribution(alpha_gate: pd.DataFrame, wm_contrib: pd.DataFrame,
                         kg_attn: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: mean alpha per KG source by cancer type
    if alpha_gate is not None and not alpha_gate.empty:
        ax = axes[0]
        kg_colors = {
            "ChEMBL":  PALETTE["kg_chembl"],
            "DRKG":    PALETTE["kg_drkg"],
            "PrimeKG": PALETTE["kg_primekg"],
        }
        pivot_alpha = alpha_gate.pivot_table(
            index="cancer_type", columns="kg_prior", values="mean_alpha", aggfunc="mean"
        ).fillna(0)
        cols = [c for c in ["ChEMBL", "DRKG", "PrimeKG"] if c in pivot_alpha.columns]
        pivot_alpha = pivot_alpha[cols]
        x = np.arange(len(pivot_alpha))
        width = 0.28
        for i, col in enumerate(cols):
            ax.bar(x + i * width, pivot_alpha[col], width,
                   label=col, color=kg_colors.get(col, PALETTE["grey"]), alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(pivot_alpha.index, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Mean α attention weight")
        ax.set_title("KG Prior Attention (α) by Cancer Type\n(Three prior networks: ChEMBL, DRKG, PrimeKG)")
        ax.legend(title="KG Prior")

    # Panel B: WM contribution correlation
    if wm_contrib is not None and not wm_contrib.empty:
        ax = axes[1]
        colors_wm = [kg_colors.get(r, PALETTE["grey"]) for r in wm_contrib.get("kg_prior", [])]
        bars = ax.barh(
            wm_contrib.get("kg_prior", []),
            wm_contrib.get("spearman_r", []),
            color=colors_wm,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Spearman r (α vs ΔAUC)")
        ax.set_title("KG Prior Weight × Gene Perturbation Effect\n(World Model: static→dynamic KG)")
        for bar, pval in zip(bars, wm_contrib.get("p_value", [])):
            label = f"p={pval:.2e}" if pval < 0.05 else "n.s."
            x_pos = bar.get_width() + 0.002 if bar.get_width() >= 0 else bar.get_width() - 0.002
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                    va="center", fontsize=9)

    # Panel C: PrimeKG α distribution split by whether pair is SynLeth-SL
    ax = axes[2]
    if kg_attn is not None and not kg_attn.empty and "alpha_PrimeKG" in kg_attn.columns:
        primekg_high = kg_attn[kg_attn["alpha_PrimeKG"] > 0.7]["alpha_PrimeKG"]
        primekg_low  = kg_attn[kg_attn["alpha_PrimeKG"] <= 0.3]["alpha_PrimeKG"]
        bins = np.linspace(0, 1, 30)
        ax.hist(kg_attn["alpha_PrimeKG"], bins=bins, density=True, color=PALETTE["kg_primekg"],
                alpha=0.7, label="All pairs")
        ax.set_xlabel("α_PrimeKG attention weight")
        ax.set_ylabel("Density")
        ax.set_title("PrimeKG Prior Weight Distribution\n(per drug-cell prediction pair)")
        # Annotate: when α_PrimeKG is high, gene-gene SL priors are dominant
        ax.axvline(0.7, color=PALETTE["accent"], linestyle="--", linewidth=1.5,
                   label="α > 0.7 (PrimeKG dominant)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    savefig(fig, "C_fig5_kg_contribution")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("Scenario C — Step 5: Publication Figures")
    print("=" * 70)
    print(f"Output directory: {FIGURES_DIR}")

    data = load_results()

    # Figure 1b: Drug × Gene heatmap
    if data.get("drug_gene") is not None:
        print("\nGenerating Fig 1b: Drug-gene heatmap …")
        fig_drug_gene_heatmap(data["drug_gene"])

    # Figure 1c: Known target recovery
    if data.get("drug_gene") is not None:
        print("Generating Fig 1c: Known target recovery …")
        fig_known_target_recovery(data["drug_gene"])

    # Figure 2a: Volcano plot
    if data.get("drug_gene") is not None:
        print("Generating Fig 2a: Volcano …")
        fig_volcano(data["drug_gene"])

    # Figure 3: SynLeth validation
    if data.get("sl_full") is not None:
        print("Generating Fig 3: SynLeth validation …")
        fig_synleth_validation(data["sl_full"], data.get("sl_confident"), data.get("sl_enrich"))

    # Figure 4: Cancer-type heatmap
    if data.get("cancer_sl") is not None:
        print("Generating Fig 4: Cancer-type heatmap …")
        fig_cancer_heatmap(data["cancer_sl"])

    # Figure 5: KG contribution
    if data.get("alpha_gate") is not None or data.get("wm_contrib") is not None:
        print("Generating Fig 5: KG prior contribution …")
        fig_kg_contribution(data.get("alpha_gate"), data.get("wm_contrib"), data.get("kg_attn"))

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("Files:")
    for f in sorted(FIGURES_DIR.glob("C_fig*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
