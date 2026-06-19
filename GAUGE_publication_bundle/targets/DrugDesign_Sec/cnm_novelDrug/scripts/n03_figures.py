#!/usr/bin/env python3
"""
n03_figures.py
==============
Generate publication-quality figures for the novel drug KG analysis.

Figures:
  figND1: KG contribution gap for MoA-relevant training drugs
    Panel A: Scatter fp-only vs full-KG mean value_hat (EGFR inh, LUAD)
    Panel B: Scatter fp-only vs full-KG mean value_hat (MEK inh, SKCM)

  figND2: BRICS analogue KG coverage via structural proxy
    Panel A: Tanimoto distribution (analogues vs nearest KG-covered training drug)
    Panel B: Ranking correlation scatter (fp-only vs KG-proxy)

  figND3: Ranking stability of improved analogues under KG-proxy
    Panel A: Erlotinib analogues — fp-only vs KG-proxy ranks (all + improved highlighted)
    Panel B: Trametinib analogues — same

  figND4: Summary evidence matrix (improved analogues with fp-only and proxy)
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR   = Path(__file__).resolve().parents[1] / "results"
FIG_DIR   = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "egfr":     "#E64B35",   # red-orange for EGFR
    "mek":      "#4DBBD5",   # blue for MEK
    "fp_only":  "#7E6FBB",   # purple for fp-only
    "proxy":    "#00A087",   # teal for KG-proxy
    "improved": "#E64B35",   # red for improved analogues
    "neutral":  "#AAAAAA",   # grey for non-improved
    "seed":     "#3C5488",   # dark blue for seed drug
    "grid":     "#E8E8E8",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


def _save(fig, name: str):
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {name}.pdf/.png")


# ── Figure ND1: KG Contribution Gap ──────────────────────────────────────────

def fig_nd1_kg_gap():
    """Scatter fp-only vs full-KG mean value_hat for MoA-class training drugs."""
    gap_df = pd.read_csv(OUT_DIR / "kg_contribution_gap.csv")
    gap_with_kg = gap_df[gap_df["in_kg"] == True].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    moa_configs = [
        ("EGFR inhibitor", "TCGA-LUAD", axes[0], PALETTE["egfr"], "EGFR inhibitors (TCGA-LUAD)"),
        ("MEK inhibitor",  "TCGA-SKCM", axes[1], PALETTE["mek"],  "MEK inhibitors (TCGA-SKCM)"),
    ]

    for moa_class, cancer_type, ax, color, title in moa_configs:
        sub = gap_with_kg[
            (gap_with_kg["moa_class"] == moa_class) &
            (gap_with_kg["cancer_type"] == cancer_type)
        ].copy()

        if len(sub) == 0:
            ax.set_title(f"{title}\n(no data)")
            continue

        x = sub["mean_vh_fp"].values
        y = sub["mean_vh_kg"].values

        # Scatter
        ax.scatter(x, y, c=color, s=80, alpha=0.85, edgecolors="white", linewidth=0.8, zorder=3)

        # Label each point
        for _, r in sub.iterrows():
            short_name = str(r["drug_name"])[:12]
            ax.annotate(short_name, (r["mean_vh_fp"], r["mean_vh_kg"]),
                        fontsize=7.5, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

        # Diagonal line (y=x)
        lo = min(x.min(), y.min()) - 0.002
        hi = max(x.max(), y.max()) + 0.002
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")

        # Pearson r annotation
        if len(x) >= 2:
            r_val, p_val = pearsonr(x, y)
            ax.text(0.05, 0.92, f"Pearson r = {r_val:.3f}\np = {p_val:.2e}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Mean value_hat (fingerprint-only)", fontsize=10)
        ax.set_ylabel("Mean value_hat (full KG)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Add subtitle with n
        n_in = int(sub["in_kg"].sum())
        ax.text(0.5, -0.13, f"n = {n_in} drugs with KG coverage",
                transform=ax.transAxes, ha="center", fontsize=9, color="gray")

    fig.suptitle(
        "KG Contribution: Full-KG vs Fingerprint-Only Predictions\n"
        "for MoA-Relevant Training Drugs",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, "figND1_kg_contribution_gap")


# ── Figure ND2: Analogue Tanimoto + Ranking Correlation ──────────────────────

def fig_nd2_proxy_coverage():
    """Tanimoto distribution and ranking correlation."""
    luad_df = pd.read_csv(OUT_DIR / "novel_drug_kg_proxy_luad.csv")
    skcm_df = pd.read_csv(OUT_DIR / "novel_drug_kg_proxy_skcm.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    datasets = [
        (luad_df, "erlotinib",  "TCGA-LUAD", PALETTE["egfr"], axes[0]),
        (skcm_df, "trametinib", "TCGA-SKCM", PALETTE["mek"],  axes[1]),
    ]

    for df, seed_name, cancer_type, color, row_axes in datasets:
        ax_hist, ax_scatter = row_axes

        tani = df["nn_tanimoto"].values
        improved = df["improved_fp"].values.astype(bool)
        improved_proxy = df["improved_proxy_fair"].values.astype(bool)
        fp_vh = df["mean_vh_fp"].values
        proxy_vh = df["mean_vh_proxy"].values

        # Panel: Tanimoto distribution
        ax_hist.hist(tani[~improved], bins=30, color=PALETTE["neutral"],
                     alpha=0.7, label=f"Not improved (n={int((~improved).sum())})", density=False)
        ax_hist.hist(tani[improved], bins=15, color=color,
                     alpha=0.9, label=f"Improved (n={int(improved.sum())})", density=False)
        ax_hist.axvline(np.median(tani), color="black", linestyle="--", lw=1.5, alpha=0.7,
                        label=f"Median = {np.median(tani):.3f}")
        ax_hist.set_xlabel("Tanimoto similarity to nearest KG-covered training drug", fontsize=10)
        ax_hist.set_ylabel("Count of analogues", fontsize=10)
        ax_hist.set_title(f"{seed_name.capitalize()} analogues ({cancer_type})\nKG proxy coverage", fontsize=11)
        ax_hist.legend(fontsize=8.5)

        # Panel: Ranking correlation scatter
        rho, p_rho = spearmanr(fp_vh, proxy_vh)
        r_p, p_p = pearsonr(fp_vh, proxy_vh)

        # Plot non-improved
        ax_scatter.scatter(fp_vh[~improved], proxy_vh[~improved],
                           c=PALETTE["neutral"], s=20, alpha=0.4, zorder=2,
                           label=f"Not improved ({int((~improved).sum())})")
        # Plot improved (highlighted)
        ax_scatter.scatter(fp_vh[improved], proxy_vh[improved],
                           c=color, s=60, alpha=0.9, edgecolors="white", linewidth=0.8, zorder=3,
                           label=f"Improved in fp-only ({int(improved.sum())})")
        # Circle both-improved
        both_imp = improved & improved_proxy
        if both_imp.sum() > 0:
            ax_scatter.scatter(fp_vh[both_imp], proxy_vh[both_imp],
                               c="none", s=120, edgecolors="black", linewidth=1.5, zorder=4,
                               label=f"Improved in BOTH ({int(both_imp.sum())})")

        lo = min(fp_vh.min(), proxy_vh.min()) - 0.001
        hi = max(fp_vh.max(), proxy_vh.max()) + 0.001
        ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)

        ax_scatter.text(0.05, 0.93,
                        f"Spearman ρ = {rho:.4f}\np = {p_rho:.2e}\n"
                        f"Pearson r = {r_p:.4f}",
                        transform=ax_scatter.transAxes, fontsize=9,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Add seed drug positions
        seed_fp_val = float(df["seed_vh_fp"].iloc[0])
        seed_proxy_val = float(df["seed_vh_proxy"].iloc[0])
        ax_scatter.axvline(seed_fp_val, color=color, lw=1.2, linestyle=":", alpha=0.7, label=f"Seed fp={seed_fp_val:.4f}")
        ax_scatter.axhline(seed_proxy_val, color=PALETTE["proxy"], lw=1.2, linestyle=":", alpha=0.7, label=f"Seed proxy={seed_proxy_val:.4f}")

        ax_scatter.set_xlabel("Mean value_hat (fingerprint-only)", fontsize=10)
        ax_scatter.set_ylabel("Mean value_hat (KG-proxy, fair baseline)", fontsize=10)
        ax_scatter.set_title(f"{seed_name.capitalize()} analogues\nFP-only vs KG-proxy ranking\n(dashed = seed drug baselines)", fontsize=10)
        ax_scatter.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "BRICS Analogue KG Coverage via Structural Proxy\n"
        "Tanimoto to Nearest KG-Covered Training Drug; Ranking Stability",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "figND2_proxy_coverage_and_ranking")


# ── Figure ND3: Rank Comparison Top Candidates ───────────────────────────────

def fig_nd3_rank_comparison():
    """Rank comparison for all analogues — highlight improved set."""
    luad_df = pd.read_csv(OUT_DIR / "novel_drug_kg_proxy_luad.csv")
    skcm_df = pd.read_csv(OUT_DIR / "novel_drug_kg_proxy_skcm.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for df, seed_name, color, ax in [
        (luad_df, "erlotinib",  PALETTE["egfr"], axes[0]),
        (skcm_df, "trametinib", PALETTE["mek"],  axes[1]),
    ]:
        # Compute ranks (1 = best = lowest vh)
        df = df.copy().reset_index(drop=True)
        df["rank_fp"]    = df["mean_vh_fp"].rank(method="min").astype(int)
        df["rank_proxy"] = df["mean_vh_proxy"].rank(method="min").astype(int)
        n = len(df)

        improved_fp    = df["improved_fp"].values.astype(bool)
        improved_proxy = df["improved_proxy_fair"].values.astype(bool)
        both = improved_fp & improved_proxy

        # Scatter: rank under fp-only vs rank under proxy
        ax.scatter(df.loc[~improved_fp, "rank_fp"], df.loc[~improved_fp, "rank_proxy"],
                   c=PALETTE["neutral"], s=18, alpha=0.35, zorder=2, label=f"Not improved ({int((~improved_fp).sum())})")
        ax.scatter(df.loc[improved_fp & ~improved_proxy, "rank_fp"],
                   df.loc[improved_fp & ~improved_proxy, "rank_proxy"],
                   c=color, s=50, alpha=0.8, marker="^", zorder=3,
                   label=f"FP-only improved, not proxy ({int((improved_fp & ~improved_proxy).sum())})")
        ax.scatter(df.loc[both, "rank_fp"], df.loc[both, "rank_proxy"],
                   c=color, s=80, alpha=1.0, edgecolors="black", linewidth=1.2, zorder=4,
                   label=f"Improved in BOTH modes ({int(both.sum())})")

        # Diagonal
        ax.plot([1, n], [1, n], "k--", lw=1, alpha=0.4, label="y = x (perfect concordance)")

        # Annotate top BOTH-improved drugs
        top_both = df[both].sort_values("rank_proxy").head(5)
        for _, r in top_both.iterrows():
            short = str(r["DRUG_NAME"])
            # Extract analogue number
            if "_analogue_" in short:
                short = short.split("_analogue_")[1][:5]
            ax.annotate(short, (r["rank_fp"], r["rank_proxy"]),
                        fontsize=6.5, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points",
                        color="black")

        rho, p_rho = spearmanr(df["rank_fp"], df["rank_proxy"])
        ax.text(0.05, 0.93, f"Spearman ρ = {rho:.4f}\np = {p_rho:.2e}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Rank by fingerprint-only (1=best)", fontsize=10)
        ax.set_ylabel("Rank by KG-proxy (1=best)", fontsize=10)
        ax.set_title(f"{seed_name.capitalize()} analogues\nRanking concordance (n={n})", fontsize=11)
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        "Rank Stability of BRICS Analogues Under KG-Proxy vs Fingerprint-Only Scoring\n"
        "Concordant top candidates confirm robustness of drug generation results",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, "figND3_rank_concordance")


# ── Figure ND4: Final Improved Analogue Table ─────────────────────────────────

def fig_nd4_improved_evidence():
    """Table-style figure showing all improved analogues under both scoring modes."""
    luad_df = pd.read_csv(OUT_DIR / "novel_drug_kg_proxy_luad.csv")
    skcm_df = pd.read_csv(OUT_DIR / "novel_drug_kg_proxy_skcm.csv")

    # Filter to fp-only improved analogues (matches original analysis)
    luad_imp = luad_df[luad_df["improved_fp"] == True].copy().sort_values("mean_vh_fp")
    skcm_imp = skcm_df[skcm_df["improved_fp"] == True].copy().sort_values("mean_vh_fp")

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, max(len(luad_imp), len(skcm_imp)) * 0.6 + 1.5)))

    for ax, imp_df, seed_name, cancer_type, color in [
        (axes[0], luad_imp, "erlotinib",  "TCGA-LUAD", PALETTE["egfr"]),
        (axes[1], skcm_imp, "trametinib", "TCGA-SKCM", PALETTE["mek"]),
    ]:
        if len(imp_df) == 0:
            ax.text(0.5, 0.5, "No improved analogues", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{seed_name.capitalize()} improved analogues", fontsize=11)
            continue

        # Build display names
        names = []
        for drug_name in imp_df["DRUG_NAME"].values:
            if "_analogue_" in str(drug_name):
                idx = str(drug_name).split("_analogue_")[1][:4]
                names.append(f"{seed_name[:5]}_{idx}")
            else:
                names.append(str(drug_name)[:15])

        x = np.arange(len(imp_df))
        width = 0.35

        # FP-only delta bars
        seed_vh = float(imp_df["seed_vh_fp"].iloc[0])
        delta_fp    = imp_df["delta_fp"].values
        delta_proxy = imp_df["delta_proxy_fair"].values
        improved_proxy_mask = imp_df["improved_proxy_fair"].values.astype(bool)

        b1 = ax.bar(x - width/2, delta_fp * 1000, width, label="FP-only Δ (×10³)",
                    color=color, alpha=0.8, edgecolor="white")
        b2 = ax.bar(x + width/2, delta_proxy * 1000, width, label="KG-proxy Δ (×10³)",
                    color=PALETTE["proxy"], alpha=0.8, edgecolor="white")

        # Mark proxy-also-improved with a star
        for j, (bp, is_imp_proxy) in enumerate(zip(b2, improved_proxy_mask)):
            if is_imp_proxy:
                ax.text(bp.get_x() + bp.get_width()/2, bp.get_height() + 0.05,
                        "★", ha="center", va="bottom", fontsize=10, color="darkgreen")

        # KG proxy NN info below
        for j, (xi, nn_name) in enumerate(zip(x, imp_df["nn_drug_name"].values)):
            tani = imp_df["nn_tanimoto"].values[j]
            ax.text(xi, -0.3, f"{str(nn_name)[:10]}\n({tani:.2f})",
                    ha="center", va="top", fontsize=6, color="gray")

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Δ value_hat (×10³) vs seed", fontsize=10)
        ax.set_title(f"{seed_name.capitalize()} improved analogues\n(★ = also improved under KG-proxy)", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.text(0.5, -0.22,
                "Below bars: nearest KG training drug (Tanimoto in parentheses)",
                transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    fig.suptitle(
        "Improved BRICS Analogues: FP-Only vs KG-Proxy Predicted Improvement\n"
        "★ = analogue confirmed improved under KG-proxy inference",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, "figND4_improved_analogues_both_modes")


# ── Figure ND0: Schematic (textual summary as supplementary) ─────────────────

def fig_nd0_architecture():
    """Conceptual schematic of fp-only vs KG-proxy inference."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # Left: fp-only for novel drugs
    ax = axes[0]
    ax.set_title("Mode A: Fingerprint-Only\n(novel drugs, drug_idx=None)", fontsize=11, fontweight="bold")

    boxes = [
        (5, 8.5, "Novel BRICS Analogue\n(not in any database)", "#F1C40F", 2.5, 0.9),
        (2, 6.0, "Morgan\nFingerprint", "#3498DB", 1.5, 0.8),
        (8, 6.0, "KG Networks\n(ChEMBL/DRKG/PrimeKG)", "#E74C3C", 1.8, 0.8),
        (5, 3.5, "z_chem\n(fingerprint embedding)", "#27AE60", 2.0, 0.8),
        (8, 3.5, "z_prior = 0\n(masked out)", "#BDC3C7", 1.8, 0.8),
        (5, 1.2, "value_hat\n(fingerprint-only prediction)", "#8E44AD", 2.5, 0.8),
    ]
    for (cx, cy, txt, clr, w, h) in boxes:
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.1", facecolor=clr,
                                        edgecolor="white", linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Arrows
    ax.annotate("", xy=(2, 6.4+0.4), xytext=(4, 7.6+0.9),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(5, 4.3), xytext=(5, 6.5-0.4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(7.1, 3.5), xytext=(8, 6.4-0.4),
                arrowprops=dict(arrowstyle="->", color="lightgray", lw=1.5, linestyle="dashed"))
    ax.annotate("", xy=(5, 2.0), xytext=(5, 2.9),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(8, 5.0, "BLOCKED\n(no entry)", ha="center", va="center", fontsize=8,
            color="#E74C3C", fontweight="bold")

    # Right: KG-proxy
    ax = axes[1]
    ax.set_title("Mode B: KG-Proxy Inference\n(novel drug + NN's KG, drug_latent=novel, drug_idx=NN)", fontsize=10, fontweight="bold")

    boxes_b = [
        (5, 8.5, "Novel BRICS Analogue\n(SMILES only)", "#F1C40F", 2.5, 0.9),
        (2, 6.0, "Morgan\nFingerprint", "#3498DB", 1.5, 0.8),
        (8, 6.0, "Nearest Training Drug\n(top Tanimoto match,\nin KG)", "#E74C3C", 2.0, 1.0),
        (2, 3.5, "z_chem\n(novel drug's embedding)", "#27AE60", 2.0, 0.8),
        (8, 3.5, "z_kg (NN's KG\nbranch embeddings)", "#E74C3C", 2.0, 0.8),
        (5, 1.2, "value_hat\n(KG-proxy prediction)", "#8E44AD", 2.5, 0.8),
    ]
    for (cx, cy, txt, clr, w, h) in boxes_b:
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.1", facecolor=clr,
                                        edgecolor="white", linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Arrows
    ax.annotate("", xy=(2, 6.4+0.4), xytext=(4, 7.6+0.9),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(8, 6.4+0.5-0.9), xytext=(8, 7.6+0.9-1.0),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5))
    ax.annotate("", xy=(2, 4.3), xytext=(2, 5.6),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(8, 4.3), xytext=(8, 5.5-0.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(4, 1.5), xytext=(2.5, 3.1),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(6, 1.5), xytext=(7.5, 3.1),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.text(5, 2.6, "alpha gate\nblending", ha="center", va="center", fontsize=8, color="gray")

    fig.suptitle(
        "GAUGE Inference for Novel BRICS Analogues\n"
        "Fingerprint-Only (Mode A) vs Knowledge-Graph Proxy (Mode B)",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    _save(fig, "figND0_inference_schematic")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating Novel Drug Analysis Figures")
    print("=" * 60)

    print("\n[0] Inference schematic...")
    fig_nd0_architecture()

    print("[1] KG contribution gap (fp-only vs full-KG)...")
    try:
        fig_nd1_kg_gap()
    except Exception as e:
        print(f"  WARNING: {e}")

    print("[2] Proxy coverage and ranking correlation...")
    try:
        fig_nd2_proxy_coverage()
    except Exception as e:
        print(f"  WARNING: {e}")

    print("[3] Rank concordance...")
    try:
        fig_nd3_rank_comparison()
    except Exception as e:
        print(f"  WARNING: {e}")

    print("[4] Improved analogues both modes...")
    try:
        fig_nd4_improved_evidence()
    except Exception as e:
        print(f"  WARNING: {e}")

    print(f"\n  All figures → {FIG_DIR}")


if __name__ == "__main__":
    main()
