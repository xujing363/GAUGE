"""
Script 06: Publication-Quality Figures for Scenario A
=======================================================
Generates all figures demonstrating the world model's ability to capture
causal gene-drug sensitivity relationships in unseen cell lines.

Figure panels:
  Fig A: Global perturbation heatmap (gene × drug family sensitivity landscape)
  Fig B: World model gate dynamics (static → dynamic KG, α variability)
  Fig C: Cancer-type-specific perturbation profiles (NSCLC/Melanoma/AML)
  Fig D: Drug family target specificity (rank of known targets)
  Fig E: Three-network attribution (ChEMBL/DRKG/PrimeKG contribution)
  Fig F: Known-target validation (scatter: |ΔAU| for target vs non-target)

All figures saved to figures/ with 300 DPI.

Usage:
    python scripts/06_publication_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    DRUG_FAMILIES,
    FIGURES_DIR,
    KNOWN_DRUG_TARGETS,
    RESULTS_DIR,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "figure.dpi": 150,
})

BLUE   = "#2271B2"
RED    = "#E41A1C"
ORANGE = "#FF7F00"
GREEN  = "#4DAF4A"
PURPLE = "#984EA3"
GRAY   = "#999999"

FAMILY_COLORS = {
    "EGFR_inhibitors":   BLUE,
    "BRAF_inhibitors":   RED,
    "MEK_inhibitors":    ORANGE,
    "BCL2_inhibitors":   GREEN,
    "PARP_inhibitors":   PURPLE,
    "CDK46_inhibitors":  "#A65628",
    "PI3K_inhibitors":   "#F781BF",
    "MTOR_inhibitors":   "#377EB8",
}

KG_SOURCE_COLORS = {
    "ChEMBL":  "#E41A1C",
    "DRKG":    "#FF7F00",
    "PrimeKG": "#4DAF4A",
}


# ── Helper ─────────────────────────────────────────────────────────────────

def save_fig(fig, name: str, dpi: int = 300):
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_png = FIGURES_DIR / f"{name}.png"
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


# ── Figure A: Global Perturbation Heatmap ──────────────────────────────────

def fig_global_heatmap():
    csv = RESULTS_DIR / "01_global" / "global_perturbation.csv"
    if not csv.exists():
        print("  SKIP Fig A: global_perturbation.csv not found")
        return

    df = pd.read_csv(csv)

    # Assign drug family
    drug_to_family = {}
    for fam, drugs in DRUG_FAMILIES.items():
        for d in drugs:
            drug_to_family[d] = fam

    df["drug_family"] = df["DRUG_NAME"].map(drug_to_family).fillna("other")
    family_order = [f for f in DRUG_FAMILIES if f in df["drug_family"].unique()] + ["other"]

    # Top 20 genes by mean_abs_delta
    gene_summary = (
        df.groupby("gene_name")["abs_mean_delta"].mean()
        .sort_values(ascending=False).head(20)
    )
    top_genes = gene_summary.index.tolist()

    # Top family drugs: pick 1-2 per family
    pivot_drugs = []
    for fam in family_order:
        fam_drugs = df[df["drug_family"] == fam]["DRUG_NAME"].unique()
        pivot_drugs.extend(fam_drugs[:2])

    pivot = (
        df[df["gene_name"].isin(top_genes) & df["DRUG_NAME"].isin(pivot_drugs)]
        .pivot_table(index="gene_name", columns="DRUG_NAME", values="mean_delta_auc", aggfunc="mean")
        .reindex(top_genes)
    )

    if pivot.empty:
        print("  SKIP Fig A: insufficient data for heatmap")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.6), max(5, len(top_genes) * 0.35)))
    vmax = np.nanpercentile(np.abs(pivot.values), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("rw_b", [RED, "white", BLUE])
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Mark known targets
    for ri, gene in enumerate(pivot.index):
        for ci, drug in enumerate(pivot.columns):
            if gene in KNOWN_DRUG_TARGETS.get(drug, []):
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                           fill=False, edgecolor="black", lw=1.5))

    plt.colorbar(im, ax=ax, label="Mean ΔAU (silenced − baseline)", shrink=0.6)
    ax.set_title("Global Gene Perturbation Landscape\n"
                 "(190 unseen test cell lines, ■ = known drug target)", fontweight="bold")
    ax.set_xlabel("Drug")
    ax.set_ylabel("Gene (silenced)")
    fig.tight_layout()
    save_fig(fig, "figA_global_heatmap")


# ── Figure B: World Model Gate Dynamics ────────────────────────────────────

def fig_gate_dynamics():
    alpha_csv = RESULTS_DIR / "02_gate" / "drug_kg_activation_profile.csv"
    shift_csv = RESULTS_DIR / "02_gate" / "gene_perturbation_tl_shift.csv"
    if not alpha_csv.exists():
        print("  SKIP Fig B: drug_kg_activation_profile.csv not found")
        return

    alpha_df = pd.read_csv(alpha_csv)
    shift_df = pd.read_csv(shift_csv) if shift_csv.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel B1: KG activation classification pie chart
    ax = axes[0]
    class_counts = alpha_df["kg_class"].value_counts()
    colors_pie = {"KG-silent": GRAY, "ChEMBL-only": RED,
                  "ChEMBL+PrimeKG": BLUE, "PrimeKG-dominant": GREEN}
    wedge_colors = [colors_pie.get(c, ORANGE) for c in class_counts.index]
    ax.pie(class_counts.values, labels=class_counts.index, colors=wedge_colors,
           autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title("KG Activation Classes\n(DRKG weight = 0 for all drugs)", fontweight="bold")

    # Panel B2: ChEMBL vs PrimeKG per-drug stacked bar
    ax = axes[1]
    active_df = alpha_df[alpha_df["alpha_total"] > 0.01].copy()
    active_df = active_df.sort_values("alpha_chembl", ascending=False).head(25)
    x = np.arange(len(active_df))
    ax.bar(x, active_df["alpha_chembl"].values, color=RED, alpha=0.8, label="ChEMBL")
    ax.bar(x, active_df["alpha_primekg"].values, bottom=active_df["alpha_chembl"].values,
           color=BLUE, alpha=0.8, label="PrimeKG")
    ax.set_xticks(x[::5])
    ax.set_xticklabels(active_df["DRUG_NAME"].iloc[::5], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("KG source weight α")
    ax.set_title("ChEMBL vs PrimeKG Weight\n(KG-active drugs)", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B3: Terminal latent shift (target vs control)
    ax = axes[2]
    if not shift_df.empty:
        focus_rows   = shift_df[shift_df["is_known_target"] == True].copy()
        control_rows = shift_df[shift_df["is_known_target"] == False].copy()
        all_rows_plot = pd.concat([focus_rows, control_rows])
        colors_b3 = [RED if r["is_known_target"] else GRAY for _, r in all_rows_plot.iterrows()]
        labels_b3 = [f"{r['DRUG_NAME'][:6]}←{r['gene_name']}" for _, r in all_rows_plot.iterrows()]
        ax.barh(range(len(all_rows_plot)), all_rows_plot["mean_tl_shift"].values,
                color=colors_b3, alpha=0.8)
        ax.set_yticks(range(len(labels_b3)))
        ax.set_yticklabels(labels_b3, fontsize=7)
        ax.set_xlabel("Mean terminal latent L2 shift")
        ax.set_title("Terminal Latent Shift\nafter Gene Silencing", fontweight="bold")
        known_patch   = mpatches.Patch(color=RED,  label="Known target pair")
        control_patch = mpatches.Patch(color=GRAY, label="Control pair")
        ax.legend(handles=[known_patch, control_patch], fontsize=8)

    fig.suptitle("World Model Dynamics: Selective KG Activation + Cell-Context Terminal Latent",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "figB_gate_dynamics")


# ── Figure C: Cancer-Type-Specific Perturbation ─────────────────────────────

def fig_cancer_type():
    csv = RESULTS_DIR / "03_cancer_type" / "cancer_gene_drug_perturbation.csv"
    if not csv.exists():
        print("  SKIP Fig C: cancer_gene_drug_perturbation.csv not found")
        return

    df = pd.read_csv(csv)
    if df.empty:
        print("  SKIP Fig C: empty data")
        return

    focus_cancers = [c for c in ["NSCLC", "Melanoma", "AML", "CLL_DLBCL", "Breast", "Colorectal"]
                     if c in df["cancer_type"].unique()][:4]

    fig, axes = plt.subplots(1, len(focus_cancers), figsize=(4 * len(focus_cancers), 5), sharey=False)
    if len(focus_cancers) == 1:
        axes = [axes]

    for ax, cancer in zip(axes, focus_cancers):
        cdf = df[df["cancer_type"] == cancer].copy()
        top_genes = (
            cdf.groupby("gene_name")["abs_mean_delta"]
            .mean().sort_values(ascending=False).head(10)
        )
        cdf_top = cdf[cdf["gene_name"].isin(top_genes.index)]
        gene_order = top_genes.index.tolist()
        means = [cdf_top[cdf_top["gene_name"] == g]["mean_delta_auc"].mean() for g in gene_order]
        colors = [RED if KNOWN_DRUG_TARGETS and
                  any(g in KNOWN_DRUG_TARGETS.get(d, [])
                      for d in cdf_top[cdf_top["gene_name"] == g]["DRUG_NAME"].unique())
                  else GRAY
                  for g in gene_order]
        ax.barh(range(len(gene_order)), means, color=colors, alpha=0.8)
        ax.set_yticks(range(len(gene_order)))
        ax.set_yticklabels(gene_order, fontsize=8)
        ax.set_xlabel("Mean ΔAU", fontsize=9)
        ax.set_title(cancer, fontweight="bold")
        ax.axvline(0, color=GRAY, lw=0.8)

    known_patch   = mpatches.Patch(color=RED,  label="Known drug target")
    unknown_patch = mpatches.Patch(color=GRAY, label="Non-target gene")
    fig.legend(handles=[known_patch, unknown_patch], loc="upper right", fontsize=9)
    fig.suptitle("Cancer-Type-Specific Gene Perturbation Profiles\n"
                 "(190 Unseen Test Cell Lines)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "figC_cancer_type_perturbation")


# ── Figure D: Drug Family Target Specificity ───────────────────────────────

def fig_drug_family_specificity():
    rank_csv = RESULTS_DIR / "04_drug_family" / "family_rank_heatmap.csv"
    spec_csv = RESULTS_DIR / "04_drug_family" / "drug_family_specificity.csv"
    if not rank_csv.exists():
        print("  SKIP Fig D: family_rank_heatmap.csv not found")
        return

    rank_df = pd.read_csv(rank_csv)
    spec_df = pd.read_csv(spec_csv) if spec_csv.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel D1: Target gene rank within family
    ax = axes[0]
    target_ranks = rank_df[rank_df["is_family_target"]].copy()
    if not target_ranks.empty:
        families = target_ranks["drug_family"].unique()
        for i, fam in enumerate(families):
            fam_ranks = target_ranks[target_ranks["drug_family"] == fam]["rank_in_family"].values
            ax.scatter([i] * len(fam_ranks), fam_ranks, s=60,
                       color=FAMILY_COLORS.get(fam, GRAY), zorder=3, label=fam)
        n_total = rank_df.groupby("drug_family").size()
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([f.replace("_inhibitors", "\ninh.") for f in families],
                           fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Rank of known target gene\n(lower = more important)")
        ax.set_title("Target Gene Rank\nwithin Drug Family", fontweight="bold")
        ax.invert_yaxis()

    # Panel D2: Specificity score for target genes
    ax = axes[1]
    if not spec_df.empty:
        target_spec = spec_df[spec_df["is_family_target"]].copy()
        target_spec = target_spec.sort_values("specificity_score", ascending=False).head(15)
        colors = [FAMILY_COLORS.get(f, GRAY) for f in target_spec["drug_family"]]
        ax.barh(range(len(target_spec)), target_spec["specificity_score"].values,
                color=colors, alpha=0.8)
        ax.set_yticks(range(len(target_spec)))
        ax.set_yticklabels([
            f"{row['gene_name']} ({row['drug_family'].replace('_inhibitors','').replace('_','-')})"
            for _, row in target_spec.iterrows()
        ], fontsize=8)
        ax.set_xlabel("Specificity score\n(family ΔAU / background ΔAU)")
        ax.set_title("Drug-Target Specificity\n(known target genes)", fontweight="bold")
        ax.axvline(1, color=GRAY, lw=0.8, ls="--")

    fig.suptitle("Drug Family Target Specificity — World Model Captures Drug-Gene Selectivity",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "figD_drug_family_specificity")


# ── Figure E: Three-Network Attribution ────────────────────────────────────

def fig_three_network():
    pert_csv = RESULTS_DIR / "05_network_attribution" / "network_perturbation_effect.csv"
    acc_csv  = RESULTS_DIR / "05_network_attribution" / "network_attribution_summary.csv"
    if not pert_csv.exists():
        print("  SKIP Fig E: network_perturbation_effect.csv not found")
        return

    pert_df = pd.read_csv(pert_csv)
    acc_df  = pd.read_csv(acc_csv) if acc_csv.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel E1: ΔAU by condition for each focus pair
    ax = axes[0]
    if not pert_df.empty:
        focus_pairs = pert_df.groupby(["DRUG_NAME", "gene_name"]).first().reset_index()[
            ["DRUG_NAME", "gene_name"]].head(6)
        n_pairs = len(focus_pairs)
        n_conds = len(pert_df["condition"].unique())
        x = np.arange(n_pairs)
        width = 0.15
        cond_order = ["full", "ChEMBL_off", "DRKG_off", "PrimeKG_off", "all_off"]
        cond_colors = {"full": BLUE, "ChEMBL_off": "#E41A1C", "DRKG_off": "#FF7F00",
                       "PrimeKG_off": "#4DAF4A", "all_off": GRAY}
        for ci, cond in enumerate(cond_order):
            vals = []
            for _, row in focus_pairs.iterrows():
                sub = pert_df[(pert_df["DRUG_NAME"] == row["DRUG_NAME"]) &
                              (pert_df["gene_name"] == row["gene_name"]) &
                              (pert_df["condition"] == cond)]
                vals.append(sub["abs_delta_auc"].values[0] if len(sub) > 0 else 0)
            ax.bar(x + ci * width, vals, width, label=cond, color=cond_colors.get(cond, GRAY), alpha=0.8)
        ax.set_xticks(x + width * (n_conds - 1) / 2)
        ax.set_xticklabels([
            f"{row['DRUG_NAME']}\n←{row['gene_name']}"
            for _, row in focus_pairs.iterrows()
        ], fontsize=7, rotation=15, ha="right")
        ax.set_ylabel("|ΔAU| (gene perturbation effect)")
        ax.set_title("Gene Perturbation Effect by\nKG Source Ablation", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

    # Panel E2: Accuracy by KG condition (full vs ablations)
    ax = axes[1]
    if not acc_df.empty:
        cond_order = ["full", "ChEMBL_off", "DRKG_off", "PrimeKG_off", "all_off"]
        cond_colors_e = {"full": BLUE, "ChEMBL_off": RED, "DRKG_off": ORANGE,
                         "PrimeKG_off": GREEN, "all_off": GRAY}
        x = np.arange(len(cond_order))
        active_means  = [acc_df[acc_df["condition"]==c]["mean_spearman_active"].values[0]
                         if len(acc_df[acc_df["condition"]==c]) > 0 else 0
                         for c in cond_order]
        bars = ax.bar(x, active_means,
                      color=[cond_colors_e[c] for c in cond_order], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_off", "\n−off") for c in cond_order], fontsize=9)
        ax.set_ylabel("Mean Spearman r (KG-active drugs)")
        ax.set_title("Prediction Accuracy by\nKG Source Ablation", fontweight="bold")
        ax.axhline(active_means[0], color=BLUE, lw=1, ls="--", alpha=0.5)
        for bar, val in zip(bars, active_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Three Prior Network Attribution: ChEMBL × DRKG × PrimeKG",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "figE_three_network_attribution")


# ── Figure F: Known Target Validation ──────────────────────────────────────

def fig_known_target_validation():
    csv = RESULTS_DIR / "01_global" / "global_perturbation.csv"
    if not csv.exists():
        print("  SKIP Fig F: global_perturbation.csv not found")
        return

    df = pd.read_csv(csv)
    known   = df[df["is_known_target"]]["abs_mean_delta"].values
    unknown = df[~df["is_known_target"]]["abs_mean_delta"].values

    if len(known) < 2 or len(unknown) < 2:
        print("  SKIP Fig F: insufficient data")
        return

    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(known, unknown, alternative="greater")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel F1: Distribution comparison
    ax = axes[0]
    ax.hist(unknown, bins=40, color=GRAY, alpha=0.6, label=f"Non-target (n={len(unknown):,})", density=True)
    ax.hist(known,   bins=20, color=RED,  alpha=0.8, label=f"Known target (n={len(known)})", density=True)
    ax.axvline(unknown.mean(), color=GRAY, lw=1.5, ls="--")
    ax.axvline(known.mean(),   color=RED,  lw=1.5, ls="--")
    ax.set_xlabel("|ΔAU| (absolute perturbation effect)")
    ax.set_ylabel("Density")
    ax.set_title(f"Known Targets Show Larger Effect\n"
                 f"(MWU p={p:.2e}, enrichment={known.mean()/unknown.mean():.2f}×)",
                 fontweight="bold")
    ax.legend(fontsize=9)

    # Panel F2: Per-target-pair effect size
    ax = axes[1]
    known_pairs = df[df["is_known_target"]].sort_values("abs_mean_delta", ascending=False).head(20)
    known_pairs["pair_label"] = known_pairs["DRUG_NAME"] + " ← " + known_pairs["gene_name"]
    colors = [BLUE if v > 0 else RED for v in known_pairs["mean_delta_auc"]]
    ax.barh(range(len(known_pairs)), known_pairs["abs_mean_delta"].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(known_pairs)))
    ax.set_yticklabels(known_pairs["pair_label"], fontsize=7)
    ax.set_xlabel("|ΔAU| (perturbation effect)")
    ax.set_title("Top Known Drug-Target Pairs\n(sensitivity to target gene silencing)",
                 fontweight="bold")
    ax.axvline(unknown.mean(), color=GRAY, lw=1, ls="--", label="Background mean")
    ax.legend(fontsize=8)

    fig.suptitle("Validation: Model Correctly Captures Causal Gene-Drug Relationships\n"
                 "(190 Unseen Test Cell Lines — Cell-Line Split)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "figF_known_target_validation")


# ── Figure G: Summary overview ─────────────────────────────────────────────

def fig_summary_overview():
    """Combined 2x3 summary panel."""
    global_csv  = RESULTS_DIR / "01_global" / "gene_summary.csv"
    gate_csv    = RESULTS_DIR / "02_gate" / "gate_delta_auc_correlation.csv"
    cancer_csv  = RESULTS_DIR / "03_cancer_type" / "cancer_drug_top_genes.csv"
    spec_csv    = RESULTS_DIR / "04_drug_family" / "drug_family_specificity.csv"
    net_pert    = RESULTS_DIR / "05_network_attribution" / "network_ablation_perturbation.csv"
    valid_csv   = RESULTS_DIR / "01_global" / "known_target_validation.csv"

    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # A: Top genes
    ax = fig.add_subplot(gs[0, 0])
    if global_csv.exists():
        gdf = pd.read_csv(global_csv).head(15)
        ax.barh(range(len(gdf)), gdf["mean_abs_delta"].values, color=BLUE, alpha=0.8)
        ax.set_yticks(range(len(gdf)))
        ax.set_yticklabels(gdf["gene_name"], fontsize=8)
        ax.set_xlabel("|ΔAU|")
        ax.set_title("A: Most Impactful Genes\n(Global, all drugs)")
    ax.invert_yaxis()

    # B: Gate CV
    ax = fig.add_subplot(gs[0, 1])
    if gate_csv.exists():
        gdf = pd.read_csv(gate_csv)
        ax.bar(range(len(gdf)), gdf["gate_baseline_cv"].values, color=ORANGE, alpha=0.8)
        ax.set_xticks(range(len(gdf)))
        ax.set_xticklabels(gdf["DRUG_NAME"].str.replace("inib",""), rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("KG Gate CV (σ/μ)")
        ax.set_title("B: KG Gate Variability\nacross Unseen Cell Lines")
        ax.axhline(0, color=GRAY, lw=0.8)

    # C: Cancer type known target effect
    ax = fig.add_subplot(gs[0, 2])
    if cancer_csv.exists():
        cdf = pd.read_csv(cancer_csv)
        known_c = cdf[cdf["is_known_target"] == True]
        if not known_c.empty:
            top = known_c.sort_values("abs_mean_delta", ascending=False).head(12)
            top["label"] = top["cancer_type"] + ":" + top["gene_name"] + "→" + top["DRUG_NAME"].str[:6]
            ax.barh(range(len(top)), top["abs_mean_delta"].values, color=RED, alpha=0.8)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["label"], fontsize=7)
            ax.set_xlabel("|ΔAU|")
            ax.set_title("C: Cancer-Type Known Targets")
        ax.invert_yaxis()

    # D: Specificity scores
    ax = fig.add_subplot(gs[1, 0])
    if spec_csv.exists():
        sdf = pd.read_csv(spec_csv)
        target_spec = sdf[sdf["is_family_target"]].sort_values("specificity_score", ascending=False).head(10)
        colors = [FAMILY_COLORS.get(f, GRAY) for f in target_spec["drug_family"]]
        ax.barh(range(len(target_spec)), target_spec["specificity_score"].values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(target_spec)))
        ax.set_yticklabels([f"{r['gene_name']} ({r['drug_family'][:4]})"
                            for _, r in target_spec.iterrows()], fontsize=8)
        ax.set_xlabel("Specificity score")
        ax.set_title("D: Drug-Target Specificity")
        ax.axvline(1, color=GRAY, lw=0.8, ls="--")
        ax.invert_yaxis()

    # E: Network ablation
    ax = fig.add_subplot(gs[1, 1])
    if net_pert.exists():
        ndf = pd.read_csv(net_pert)
        if not ndf.empty:
            full = ndf[ndf["condition"] == "full"].set_index(["DRUG_NAME", "gene_name"])["abs_mean_delta"]
            for cond, color in [("ChEMBL_off", RED), ("DRKG_off", ORANGE), ("PrimeKG_off", GREEN)]:
                sub = ndf[ndf["condition"] == cond].set_index(["DRUG_NAME", "gene_name"])
                if not sub.empty:
                    ratio = sub["abs_mean_delta"] / (full.reindex(sub.index) + 1e-8)
                    ax.scatter(
                        [cond.replace("_off", "")] * len(ratio), ratio.values,
                        alpha=0.7, color=color, s=40, label=cond.replace("_off", " ablated")
                    )
            ax.axhline(1.0, color=GRAY, lw=1.5, ls="--", label="Full KG baseline")
            ax.set_ylabel("ΔAU retention\n(fraction of full-KG effect)")
            ax.set_title("E: Three-Network Attribution")
            ax.legend(fontsize=7)

    # F: Validation scatter
    ax = fig.add_subplot(gs[1, 2])
    if valid_csv.exists() and (RESULTS_DIR / "01_global" / "global_perturbation.csv").exists():
        gdf = pd.read_csv(RESULTS_DIR / "01_global" / "global_perturbation.csv")
        known   = gdf[gdf["is_known_target"]]["abs_mean_delta"].values
        unknown = gdf[~gdf["is_known_target"]]["abs_mean_delta"].values
        if len(known) > 0 and len(unknown) > 0:
            ax.boxplot([unknown, known], labels=["Non-target\ngenes", "Known\ntarget genes"],
                       patch_artist=True,
                       boxprops={"facecolor": GRAY, "alpha": 0.6},
                       medianprops={"color": "black", "lw": 2})
            ax.boxplot([known], positions=[2], patch_artist=True,
                       boxprops={"facecolor": RED, "alpha": 0.8},
                       medianprops={"color": "black", "lw": 2})
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(known, unknown, alternative="greater")
            enr = known.mean() / unknown.mean()
            ax.set_ylabel("|ΔAU|")
            ax.set_title(f"F: Known Target Validation\n(p={p:.2e}, {enr:.2f}× enrichment)")

    fig.suptitle(
        "Scenario A: World Model Captures Causal Gene-Drug Sensitivity\n"
        "in 190 Unseen Cell Lines (Cell-Line Split Test Set)",
        fontweight="bold", fontsize=13
    )
    save_fig(fig, "fig_main_summary", dpi=300)


def main():
    print("=" * 70)
    print("Script 06: Publication-Quality Figures")
    print("=" * 70)
    print(f"  Output directory: {FIGURES_DIR}")

    fig_global_heatmap()
    fig_gate_dynamics()
    fig_cancer_type()
    fig_drug_family_specificity()
    fig_three_network()
    fig_known_target_validation()
    fig_summary_overview()

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("Script 06 complete.")


if __name__ == "__main__":
    main()
