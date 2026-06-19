#!/usr/bin/env python3
"""
Publication-quality figures for the virtual drug screening + generation paper.

Reads all results from cnm/results/ and produces figures in cnm/figures/.

Figure layout:
  fig01_drug_split_validation.pdf  — Drug-split holdout generalization
  fig02_tcga_indication_recovery.pdf — TCGA virtual screening validation
  fig03_moa_structural_validity.pdf  — MoA structure-activity analysis
  fig04_drug_generation.pdf          — BRICS analogue generation (if available)

Each figure is also saved as high-res PNG (600 dpi).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

ROOT    = Path(__file__).resolve().parents[3]
RES_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "train": "#2196F3",    # blue
    "val":   "#FF9800",    # orange
    "test":  "#F44336",    # red
    "null":  "#9E9E9E",    # grey
    "TCGA-LUAD":  "#4CAF50",
    "TCGA-SKCM":  "#9C27B0",
    "TCGA-BRCA":  "#E91E63",
    "TCGA-PRAD":  "#FF5722",
    "TCGA-HNSC":  "#00BCD4",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize":  8,
    "figure.dpi":       150,
})

CANCER_LABELS = {
    "TCGA-LUAD": "LUAD",
    "TCGA-SKCM": "SKCM",
    "TCGA-BRCA": "BRCA",
    "TCGA-PRAD": "PRAD",
    "TCGA-HNSC": "HNSC",
}


def save_fig(fig, name: str, dpi: int = 300):
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved {name}.pdf/.png")


# ── Figure 1: Drug-split validation ──────────────────────────────────────────

def fig01_drug_split_validation():
    drug_df = pd.read_csv(RES_DIR / "drug_split_validation.csv")
    topK    = pd.read_csv(RES_DIR / "drug_split_topK.csv")
    with open(RES_DIR / "drug_split_summary.json") as f:
        summary = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Drug-Split Model Generalization to Unseen Drugs", fontsize=12, fontweight="bold")

    # Panel A: PCC distribution by split
    ax = axes[0]
    for split in ["train", "val", "test"]:
        sub = drug_df[drug_df["split"] == split]["pcc_value_hat"].dropna()
        n   = len(sub)
        med = sub.median()
        bp  = ax.boxplot(sub, positions=[{"train": 0, "val": 1, "test": 2}[split]],
                         widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=COLORS[split], alpha=0.7),
                         medianprops=dict(color="black", linewidth=1.5),
                         whiskerprops=dict(linewidth=0.8),
                         capprops=dict(linewidth=0.8),
                         flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                         color=COLORS[split]))
        ax.text({"train": 0, "val": 1, "test": 2}[split], -0.60,
                f"n={n}\nmed={med:.2f}", ha="center", fontsize=7, color=COLORS[split])

    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Train", "Val\n(held-out)", "Test\n(held-out)"])
    ax.set_ylabel("Pearson r (value_hat vs relative_value)")
    ax.set_title("(A) PCC distribution per data split")
    ax.set_ylim(-0.7, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: top held-out drugs bar chart
    ax = axes[1]
    top20 = topK.head(20).copy()
    top20["label"] = top20["drug_name"] + "\n(" + top20["split"] + ")"
    colors_bar = [COLORS.get(s, "#607D8B") for s in top20["split"]]
    bars = ax.barh(range(len(top20)), top20["pcc_value_hat"], color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["drug_name"], fontsize=7)
    ax.set_xlabel("Pearson r (value_hat)")
    ax.set_title("(B) Top 20 held-out drugs by PCC")
    ax.axvline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend for splits
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLORS["val"], label="Val"), Patch(facecolor=COLORS["test"], label="Test")]
    ax.legend(handles=handles, fontsize=7, loc="lower right")

    # Panel C: Summary comparison bars (mean PCC per split)
    ax = axes[2]
    splits_plot = ["train", "val", "test"]
    means   = [summary[s]["mean_pcc_value_hat"] for s in splits_plot]
    medians = [summary[s]["median_pcc_value_hat"] for s in splits_plot]
    x = np.arange(len(splits_plot))
    w = 0.35
    ax.bar(x - w/2, means,   w, label="Mean",   color=[COLORS[s] for s in splits_plot], alpha=0.7)
    ax.bar(x + w/2, medians, w, label="Median", color=[COLORS[s] for s in splits_plot], alpha=0.4, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Val\n(held-out)", "Test\n(held-out)"])
    ax.set_ylabel("Pearson r (value_hat vs relative_value)")
    ax.set_title("(C) Mean/Median PCC by split")
    ax.legend(fontsize=7)
    gap_vt = summary.get("generalization_gap_train_vs_test", 0)
    ax.annotate(f"Gen. gap\n(train-test): {gap_vt:.3f}",
                xy=(0.5, 0.75), xycoords="axes fraction", fontsize=7,
                ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "fig01_drug_split_validation")
    plt.close(fig)


# ── Figure 2: TCGA Indication Recovery ──────────────────────────────────────

def fig02_tcga_indication_recovery():
    try:
        auroc_df = pd.read_csv(RES_DIR / "tcga_indication_recovery.csv")
    except FileNotFoundError:
        print("  tcga_indication_recovery.csv not found, skipping fig02")
        return
    try:
        ranks_df = pd.read_csv(RES_DIR / "tcga_indicated_drug_ranks.csv")
    except FileNotFoundError:
        ranks_df = pd.DataFrame()
    with open(RES_DIR / "tcga_indication_recovery_summary.json") as f:
        summary = json.load(f)

    null_auroc = summary["indication_recovery"].get("null_auroc", 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("TCGA Virtual Drug Screening — Indication Recovery", fontsize=12, fontweight="bold")

    # Panel A: AUROC per cancer type
    ax = axes[0]
    if len(auroc_df) > 0:
        auroc_sorted = auroc_df.sort_values("mean_auroc", ascending=True)
        colors_bar = [COLORS.get(p, "#607D8B") for p in auroc_sorted["project_id"]]
        labels = [CANCER_LABELS.get(p, p) for p in auroc_sorted["project_id"]]
        ax.barh(labels, auroc_sorted["mean_auroc"], color=colors_bar, alpha=0.8, label="Observed")
        ax.axvline(null_auroc, color=COLORS["null"], lw=1.5, ls="--", label=f"Null ({null_auroc:.3f})")
        ax.set_xlabel("Mean AUROC (per-patient)")
        ax.set_title("(A) Indication recovery AUROC\nby cancer type")
        ax.legend(fontsize=7)
        ax.set_xlim(0.4, 0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Drug rank spotlight (held-out drugs)
    ax = axes[1]
    if len(ranks_df) > 0:
        holdout_ranks = ranks_df[ranks_df["split"].isin(["val", "test"])].copy()
        if len(holdout_ranks) > 0:
            holdout_ranks = holdout_ranks.sort_values("mean_percentile", ascending=True)
            label_strs = [f"{r['DRUG_NAME']}\n({CANCER_LABELS.get(r['project_id'], r['project_id'])})"
                          for _, r in holdout_ranks.iterrows()]
            colors_r = [COLORS.get(s, "#607D8B") for s in holdout_ranks["split"]]
            ax.barh(range(len(holdout_ranks)), holdout_ranks["mean_percentile"],
                    color=colors_r, alpha=0.8)
            ax.set_yticks(range(len(holdout_ranks)))
            ax.set_yticklabels(label_strs, fontsize=7)
            ax.axvline(0.5, color=COLORS["null"], lw=1.5, ls="--", label="Random (50th pct)")
            ax.set_xlabel("Mean percentile across patients")
            ax.set_title("(B) Held-out drug rank\n(test/val split)")
            ax.legend(fontsize=7)
            from matplotlib.patches import Patch
            hs = [Patch(facecolor=COLORS["val"], label="Val split"),
                  Patch(facecolor=COLORS["test"], label="Test split")]
            ax.legend(handles=hs, fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: AUROC fraction > 0.6 per cancer
    ax = axes[2]
    if len(auroc_df) > 0:
        auroc_sorted2 = auroc_df.sort_values("frac_auroc_gt_0.6", ascending=True)
        colors_c = [COLORS.get(p, "#607D8B") for p in auroc_sorted2["project_id"]]
        labels_c = [CANCER_LABELS.get(p, p) for p in auroc_sorted2["project_id"]]
        ax.barh(labels_c, auroc_sorted2["frac_auroc_gt_0.6"], color=colors_c, alpha=0.8)
        ax.set_xlabel("Fraction of patients with AUROC > 0.6")
        ax.set_title("(C) Patient-level signal\n(AUROC > 0.6)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "fig02_tcga_indication_recovery")
    plt.close(fig)


# ── Figure 3: MoA Structure-Activity Validity ────────────────────────────────

def fig03_moa_structural_validity():
    try:
        moa_corr   = pd.read_csv(RES_DIR / "moa_within_vs_between_corr.csv")
        holdout_moa = pd.read_csv(RES_DIR / "moa_holdout_recovery.csv")
    except FileNotFoundError:
        print("  MoA result files not found, skipping fig03")
        return
    with open(RES_DIR / "chembl_moa_validation_summary.json") as f:
        summary = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("MoA-Based Structure-Activity Validation", fontsize=12, fontweight="bold")

    # Panel A: Within vs between MoA correlation (aggregated across cancers)
    ax = axes[0]
    if len(moa_corr) > 0:
        # Aggregate across cancer types
        agg = moa_corr.groupby("moa_class")[["mean_within_r", "mean_between_r"]].mean()
        agg = agg.dropna().sort_values("delta_within_vs_between" if "delta_within_vs_between" in agg else "mean_within_r", ascending=False)
        if "delta_within_vs_between" not in agg.columns:
            agg["delta_within_vs_between"] = agg["mean_within_r"] - agg["mean_between_r"]
        agg = agg.sort_values("delta_within_vs_between", ascending=False).head(15)
        x = np.arange(len(agg))
        w = 0.35
        labels_moa = [c.replace("_", "\n") for c in agg.index]
        ax.bar(x - w/2, agg["mean_within_r"],   w, label="Within-MoA",   color="#1976D2", alpha=0.8)
        ax.bar(x + w/2, agg["mean_between_r"],  w, label="Between-MoA",  color="#EF5350", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_moa, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("Mean Pearson r across cancer types")
        ax.set_title("(A) Within vs between MoA\nvalue_hat correlation")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.axhline(summary["correlation_analysis"]["overall_mean_between_moa_r"],
                   color=COLORS["null"], ls="--", lw=1, label="Avg between")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Scatter — same-class vs diff-class correlation for held-out drugs
    ax = axes[1]
    if len(holdout_moa) > 0:
        # Aggregate across cancer types for each (DRUG_ID, DRUG_NAME)
        if "DRUG_NAME" in holdout_moa.columns:
            name_col = "DRUG_NAME"
        else:
            name_col = "DRUG_ID"
        drug_agg = holdout_moa.groupby([name_col, "split", "moa_class"])[
            ["mean_r_same_class", "mean_r_diff_class"]].mean().reset_index()
        colors_s = [COLORS.get(s, "#607D8B") for s in drug_agg["split"]]
        sc = ax.scatter(drug_agg["mean_r_diff_class"], drug_agg["mean_r_same_class"],
                        c=colors_s, alpha=0.7, s=40, edgecolors="white", lw=0.5)
        # Diagonal reference
        lim = [min(drug_agg[["mean_r_same_class","mean_r_diff_class"]].min().min(), 0),
               max(drug_agg[["mean_r_same_class","mean_r_diff_class"]].max().max(), 1)]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Mean r with DIFFERENT-class train drugs")
        ax.set_ylabel("Mean r with SAME-class train drugs")
        ax.set_title("(B) Held-out drug MoA cluster recovery\n(test/val vs train drugs)")
        frac_pos = float(summary["holdout_moa_recovery"]["frac_positive_delta"])
        ax.annotate(f"{frac_pos:.1%} above diagonal\n(correct MoA cluster)",
                    xy=(0.05, 0.9), xycoords="axes fraction", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        from matplotlib.patches import Patch
        hs = [Patch(facecolor=COLORS["val"], label="Val split"),
              Patch(facecolor=COLORS["test"], label="Test split")]
        ax.legend(handles=hs, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Delta (within - between) by MoA class, aggregated across cancers
    ax = axes[2]
    if len(moa_corr) > 0:
        agg2 = moa_corr.groupby("moa_class")[["mean_within_r", "mean_between_r"]].mean()
        agg2["delta"] = agg2["mean_within_r"] - agg2["mean_between_r"]
        agg2 = agg2.sort_values("delta", ascending=True)
        colors_delta = ["#F44336" if d < 0 else "#4CAF50" for d in agg2["delta"]]
        ax.barh([c.replace("_", " ") for c in agg2.index], agg2["delta"],
                color=colors_delta, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Δ correlation (within − between MoA)")
        ax.set_title("(C) MoA-specific sensitivity\nclustering strength")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "fig03_moa_structural_validity")
    plt.close(fig)


# ── Figure 4: Drug Generation ─────────────────────────────────────────────────

def fig04_drug_generation():
    gen_files = list(RES_DIR.glob("generated_compounds_*.csv"))
    if not gen_files:
        print("  No generated_compounds_*.csv found, skipping fig04")
        return

    gen_dfs = []
    for f in gen_files:
        df = pd.read_csv(f)
        if len(df) > 0:
            gen_dfs.append(df)
    if not gen_dfs:
        return

    gen_all = pd.concat(gen_dfs, ignore_index=True)

    with open(RES_DIR / "drug_generation_summary.json") as f:
        summary = json.load(f)

    # Apply Tanimoto filter if available (keeps structurally related analogues)
    tan_thr = summary.get("tanimoto_threshold", 0.0)
    if tan_thr > 0 and "tanimoto" in gen_all.columns:
        gen_all = gen_all[gen_all["tanimoto"] >= tan_thr].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("BRICS Drug Analogue Generation and Model-Guided Prioritisation",
                 fontsize=12, fontweight="bold")

    # Panel A: Tanimoto vs value_hat scatter — lower value_hat = more cancer cell killing
    ax = axes[0]
    for seed_info in summary.get("per_seed", []):
        seed = seed_info["seed_drug"]
        cancer = seed_info["cancer_type"]
        x_col = "tanimoto" if "tanimoto" in gen_all.columns else "qed"
        sub = gen_all[gen_all["seed_drug"] == seed].dropna(subset=[x_col, "mean_value_hat"])
        if len(sub) == 0:
            continue
        baseline = seed_info["seed_baseline"]
        color = COLORS.get(cancer, "#607D8B")
        # Scatter all analogues
        ax.scatter(sub[x_col], sub["mean_value_hat"], alpha=0.20, s=6,
                   color=color)
        # Highlight improved analogues (below baseline)
        improved = sub[sub["mean_value_hat"] < baseline]
        if len(improved) > 0:
            ax.scatter(improved[x_col], improved["mean_value_hat"],
                       alpha=0.9, s=30, color=color, marker="*", zorder=5,
                       label=f"{seed} improved (n={len(improved)})")
        ax.axhline(baseline, color=color, lw=1.5, ls="--",
                   label=f"{seed} seed ({baseline:.4f})")
    if "tanimoto" in gen_all.columns:
        ax.set_xlabel(f"Tanimoto similarity to seed (≥{tan_thr})")
    else:
        ax.set_xlabel("QED drug-likeness")
    ax.set_ylabel("Mean value_hat (↓ = more cancer cell killing)")
    ax.set_title("(A) Analogue landscape\n(★ = improved over seed)")
    ax.legend(fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Top-N improved analogues ranked by value_hat (ascending)
    ax = axes[1]
    x_offset = 0
    xtick_pos, xtick_labels = [], []
    for i, seed_info in enumerate(summary.get("per_seed", [])):
        seed = seed_info["seed_drug"]
        cancer = seed_info["cancer_type"]
        # Sort ascending: lowest value_hat = most effective
        sub = gen_all[gen_all["seed_drug"] == seed].sort_values("mean_value_hat", ascending=True)
        show = sub.head(20)
        baseline = seed_info["seed_baseline"]
        color = COLORS.get(cancer, "#607D8B")
        positions = np.arange(x_offset, x_offset + len(show))
        sizes = (show["qed"].fillna(0.5) * 30 + 5) if "qed" in show.columns else 10
        ax.scatter(positions, show["mean_value_hat"], c=color, s=sizes,
                   alpha=0.85, zorder=3)
        ax.axhline(baseline, color=color, lw=2, ls="--", alpha=0.8)
        xtick_pos.append(x_offset + len(show) / 2)
        xtick_labels.append(f"{seed}\n({CANCER_LABELS.get(cancer, cancer)})")
        x_offset += len(show) + 3

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=7)
    ax.set_ylabel("Mean value_hat (↓ = better)")
    ax.set_title("(B) Top 20 candidates (ranked ↓)\n(dashed = seed drug baseline)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Hit rate (% improved) and top-1 candidate delta
    ax = axes[2]
    for i, seed_info in enumerate(summary.get("per_seed", [])):
        seed = seed_info["seed_drug"]
        cancer = seed_info["cancer_type"]
        frac   = seed_info.get("frac_improved", 0)
        top1d  = seed_info.get("top1_delta", 0)
        n_gen  = seed_info.get("n_generated", 0)
        color  = COLORS.get(cancer, "#607D8B")
        bar_h  = frac * 100
        ax.bar(i * 2, bar_h, color=color, alpha=0.75,
               label=f"{seed} ({CANCER_LABELS.get(cancer, cancer)})")
        ax.text(i * 2, bar_h + 0.3, f"{frac:.1%}",
                ha="center", fontsize=8, fontweight="bold", color=color)
        ax.text(i * 2, -0.7,  f"n={n_gen}", ha="center", fontsize=7, color="0.4")
        ax.text(i * 2, -1.3,  f"Δtop1=−{abs(top1d):.4f}", ha="center", fontsize=7, color=color)

    ax.set_xticks([i * 2 for i in range(len(summary.get("per_seed", [])))])
    ax.set_xticklabels(
        [si["seed_drug"] for si in summary.get("per_seed", [])], fontsize=8
    )
    ax.set_ylabel("% analogues improving on seed (↓ value_hat)")
    ax.set_title(f"(C) In silico hit rate\n(Tanimoto ≥ {tan_thr}, drug-like)")
    ax.set_ylim(-2, max(5, frac * 100 + 3))
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "fig04_drug_generation")
    plt.close(fig)


# ── Supplementary: MoA holdout heatmap ───────────────────────────────────────

def fig_supp_holdout_heatmap():
    try:
        holdout_moa = pd.read_csv(RES_DIR / "moa_holdout_recovery.csv")
    except FileNotFoundError:
        return
    if len(holdout_moa) == 0:
        return

    if "DRUG_NAME" in holdout_moa.columns:
        name_col = "DRUG_NAME"
    else:
        name_col = "DRUG_ID"

    # Pivot: drug × cancer_type delta
    pivot = holdout_moa.pivot_table(
        index=[name_col, "split", "moa_class"],
        columns="cancer_type",
        values="delta_same_vs_diff",
        aggfunc="mean",
    )
    if len(pivot) == 0:
        return
    # Sort by mean delta
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
    labels_y = [f"{idx[0]}\n({idx[1]},{idx[2]})" for idx in pivot.index]
    cols = [CANCER_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.5, vmax=0.8, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Δ (same-class − diff-class) correlation")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=7)
    ax.set_title("Held-Out Drug MoA Recovery Across Cancer Types\n(Δ = same-class corr − diff-class corr)", fontsize=10)

    plt.tight_layout()
    save_fig(fig, "supp_holdout_moa_heatmap")
    plt.close(fig)


# ── Supplementary: Target-expression / drug sensitivity correlation ───────────

def fig_supp_target_expression_correlation():
    """
    Supplementary: EGFR expression correlates with erlotinib sensitivity in TCGA-LUAD;
    MAP2K1 (MEK1) expression correlates with trametinib sensitivity in TCGA-SKCM.
    Uses pre-computed TCGA predictions + h5ad expression.
    """
    try:
        import scanpy as sc
        from scipy.stats import spearmanr, mannwhitneyu
    except ImportError:
        print("  scanpy not available, skipping target-expression figure")
        return

    PREDS  = RES_DIR / "tcga_drugsplit_predictions.parquet"
    H5AD   = ROOT.parent / "Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad"
    if not PREDS.exists() or not H5AD.exists():
        return

    try:
        preds = pd.read_parquet(PREDS)
        adata = sc.read_h5ad(str(H5AD))
    except Exception as e:
        print(f"  Could not load data: {e}")
        return

    pairs = [
        ("erlotinib",  "TCGA-LUAD", "EGFR"),
        ("trametinib", "TCGA-SKCM", "MAP2K1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Target Gene Expression Predicts Drug Sensitivity\n"
                 "(lower value_hat = more cancer cell killing)",
                 fontsize=11, fontweight="bold")

    for ax, (drug_name, cancer, gene_name) in zip(axes, pairs):
        # Get cancer patients
        cancer_obs = adata.obs[adata.obs["project_id"] == cancer].index
        adata_sub = adata[cancer_obs]

        # Gene expression
        gene_idx = np.where(adata.var["gene_name"] == gene_name)[0]
        if len(gene_idx) == 0:
            print(f"  {gene_name} not found in h5ad")
            ax.set_visible(False)
            continue
        gene_expr = np.array(adata_sub.X[:, gene_idx[0]]).flatten()

        # Drug predictions
        drug_preds = preds[
            (preds["project_id"] == cancer) &
            (preds["DRUG_NAME"].str.lower() == drug_name.lower())
        ]
        drug_pt = drug_preds.groupby("entity_id")["value_hat"].mean()
        obs_names = list(cancer_obs)
        vh = np.array([drug_pt.get(pid, np.nan) for pid in obs_names])

        valid = ~np.isnan(vh)
        ge = gene_expr[valid]
        vh_v = vh[valid]

        r, p = spearmanr(ge, vh_v)

        # Scatter: x=gene_expr, y=value_hat
        color = COLORS.get(cancer, "#607D8B")
        ax.scatter(ge, vh_v, alpha=0.15, s=5, color=color)

        # Quartile means
        q25, q75 = np.percentile(ge, 25), np.percentile(ge, 75)
        low_vh  = vh_v[ge <= q25]
        high_vh = vh_v[ge >= q75]
        ax.axhline(low_vh.mean(),  color="green", ls="--", lw=1.5,
                   label=f"Q1 {gene_name} mean vh={low_vh.mean():.4f}")
        ax.axhline(high_vh.mean(), color="red", ls="--", lw=1.5,
                   label=f"Q4 {gene_name} mean vh={high_vh.mean():.4f}")

        # Mann-Whitney
        _, mw_p = mannwhitneyu(high_vh, low_vh, alternative="less")

        ax.set_xlabel(f"{gene_name} expression (log TPM)")
        ax.set_ylabel(f"{drug_name} value_hat (↓ = more sensitive)")
        pstr = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.set_title(f"{drug_name} in {cancer}\nSpearman r={r:.3f} ({pstr}); MW p={mw_p:.3g}")
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "supp_target_expression_correlation")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Generating figures → {FIG_DIR}")

    print("\n[1] Fig 1: Drug-split validation...")
    try:
        fig01_drug_split_validation()
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[2] Fig 2: TCGA indication recovery...")
    try:
        fig02_tcga_indication_recovery()
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[3] Fig 3: MoA structural validity...")
    try:
        fig03_moa_structural_validity()
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[4] Fig 4: Drug generation (if results available)...")
    try:
        fig04_drug_generation()
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[5] Supplementary: MoA holdout heatmap...")
    try:
        fig_supp_holdout_heatmap()
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[6] Supplementary: Target expression vs drug sensitivity...")
    try:
        fig_supp_target_expression_correlation()
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\nDone. Figures saved to {FIG_DIR}")
    for f in sorted(FIG_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
