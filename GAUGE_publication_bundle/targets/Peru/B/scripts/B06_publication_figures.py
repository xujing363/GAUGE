"""
B06: Publication-Quality Figures — Scenario B Summary
======================================================
Assembles the key findings from B01–B05 into high-quality multi-panel
figures suitable for a high-impact journal (Nature Methods, Nature
Communications, Briefings in Bioinformatics level).

Figure structure:
  Figure B-Main (4-panel): The core story
    Panel 1: KG coverage → accuracy (from B01)
    Panel 2: Dynamic alpha allocation (from B02)
    Panel 3: Drug network ablation (from B03)
    Panel 4: Cancer-type specificity (from B04)

  Figure B-Supp1: Static→Dynamic detail (from B02)
  Figure B-Supp2: Drug repurposing candidates (from B05)

Requires: All B01–B05 results to be present in results/ directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS, FIGURES, FOCAL_TEST_DRUGS, CANCER_GROUPS

OUT_F = FIGURES
OUT_F.mkdir(parents=True, exist_ok=True)

KNOWN_INDICATIONS = {
    "Erlotinib":    {"Lung"},
    "Gefitinib":    {"Lung"},
    "Osimertinib":  {"Lung"},
    "Venetoclax":   {"Haematological"},
    "Trametinib":   {"Skin"},
    "Crizotinib":   {"Lung"},
    "Talazoparib":  {"Breast"},
    "Ruxolitinib":  {"Haematological"},
    "Cisplatin":    {"Bladder", "Lung", "Colorectal"},
    "Cytarabine":   {"Haematological"},
}

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PANEL_COLORS = {
    "KG-covered":    "#2196F3",
    "No KG":         "#9E9E9E",
    "ChEMBL":        "#FF9800",
    "PrimeKG":       "#4CAF50",
    "DRKG":          "#FF5722",
    "baseline":      "#2C3E50",
    "ablated":       "#E74C3C",
    "expected":      "#E74C3C",
    "other":         "#95A5A6",
}


def load_results() -> dict:
    """Load all pre-computed result CSVs."""
    r = {}
    paths = {
        "coverage":        RESULTS / "global_overview" / "kg_coverage_accuracy.csv",
        "group_stats":     RESULTS / "global_overview" / "coverage_group_stats.csv",
        "stat_tests":      RESULTS / "global_overview" / "statistical_tests.csv",
        "alpha_var":       RESULTS / "dynamic_alpha" / "alpha_variance_by_drug.csv",
        "alpha_cancer":    RESULTS / "dynamic_alpha" / "alpha_by_cancer_type.csv",
        "ablation_full":   RESULTS / "drug_ablation" / "drug_network_ablation_full.csv",
        "source_abl":      RESULTS / "drug_ablation" / "drug_source_ablation.csv",
        "drug_edge_abl":   RESULTS / "drug_ablation" / "drug_specific_edge_ablation.csv",
        "cancer_delta":    RESULTS / "drug_ablation" / "cancer_specific_delta.csv",
        "expected_test":   RESULTS / "cancer_specificity" / "expected_vs_other_test.csv",
        "cancer_strat":    RESULTS / "cancer_specificity" / "cancer_stratified_delta.csv",
        "repurposing":     RESULTS / "repurposing" / "clinical_concordance.csv",
        "drug_cancer_map": RESULTS / "repurposing" / "repurposing_candidates.csv",
    }
    for key, path in paths.items():
        if path.exists():
            try:
                r[key] = pd.read_csv(path)
            except Exception as e:
                print(f"Warning: could not load {path}: {e}")
    return r


def main():
    print("=" * 70)
    print("B06: Publication-Quality Figures — Scenario B Summary")
    print("=" * 70)

    data = load_results()
    # Check minimum required results (B03 results optional - populated after GPU run)
    required = ["coverage", "group_stats", "alpha_var"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"WARNING: Missing required result files: {missing}")
        print("Please run B01 and B02 first.")
        return
    optional_missing = [k for k in ["ablation_full", "expected_test", "drug_edge_abl"] if k not in data]
    if optional_missing:
        print(f"Note: Optional model-run results not yet available: {optional_missing}")
        print("Panels C and D will be placeholders until B03/B04 complete.")

    _make_main_figure(data)
    _make_supplement_figure(data)
    _make_mechanism_schematic(data)
    print("\nB06 complete. All publication figures saved.")


def _make_main_figure(data):
    """Main 4-panel figure: the core Scenario B story."""
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)

    # ── Panel A: KG Coverage → Accuracy ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if "coverage" in data:
        cov = data["coverage"]
        kg_cov = cov[cov["coverage_group"] != "No KG"]["spearman"].dropna()
        no_kg  = cov[cov["coverage_group"] == "No KG"]["spearman"].dropna()

        parts = ax1.violinplot(
            [no_kg.values, kg_cov.values],
            positions=[0, 1], showmeans=True, showmedians=False,
        )
        parts["bodies"][0].set_facecolor(PANEL_COLORS["No KG"])
        parts["bodies"][0].set_alpha(0.7)
        parts["bodies"][1].set_facecolor(PANEL_COLORS["KG-covered"])
        parts["bodies"][1].set_alpha(0.7)

        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["No KG\ncoverage", "KG-covered\n(ChEMBL / PrimeKG)"])
        ax1.set_ylabel("Within-drug Spearman ρ\n(test drugs, n=56)")
        ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        # Stats annotation
        if "stat_tests" in data:
            pval = data["stat_tests"].iloc[0]["pval"]
            ymax = max(kg_cov.max(), no_kg.max()) + 0.05
            ax1.annotate("", xy=(1, ymax), xytext=(0, ymax),
                         arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax1.text(0.5, ymax + 0.01, f"{sig}\np={pval:.3f}", ha="center", fontsize=9)

        ax1.set_title("A. Prior KG Knowledge Improves\nGeneralization to Unseen Drugs",
                      fontweight="bold")

    # ── Panel B: Dynamic Alpha Allocation ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if "coverage" in data:
        cov = data["coverage"]
        kg_drugs = cov[(cov["coverage_group"] != "No KG")].dropna(subset=["spearman"])
        kg_drugs = kg_drugs.sort_values("spearman", ascending=False)
        x = np.arange(len(kg_drugs))
        ax2.bar(x, kg_drugs["alpha_ChEMBL"],    0.7, label="α ChEMBL (MoA)", color=PANEL_COLORS["ChEMBL"], alpha=0.85)
        ax2.bar(x, kg_drugs["alpha_DRKG"],      0.7, bottom=kg_drugs["alpha_ChEMBL"],
                label="α DRKG (biology)", color=PANEL_COLORS["DRKG"], alpha=0.85)
        ax2.bar(x, kg_drugs["alpha_PrimeKG"],   0.7,
                bottom=kg_drugs["alpha_ChEMBL"] + kg_drugs["alpha_DRKG"],
                label="α PrimeKG (network)", color=PANEL_COLORS["PrimeKG"], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(kg_drugs["DRUG_NAME"], rotation=50, ha="right", fontsize=7)
        ax2.set_ylabel("Mean KG attention weight α")
        ax2.set_title("B. Dynamic KG Attention Allocation\n"
                      "(Static prior → context-specific weight)",
                      fontweight="bold")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.set_xlim(-0.5, len(kg_drugs) - 0.5)

        # Overlay Spearman
        ax2b = ax2.twinx()
        ax2b.plot(x, kg_drugs["spearman"], "ko-", markersize=4, linewidth=1.2, alpha=0.7)
        ax2b.set_ylabel("Spearman ρ", color="black")
        ax2b.set_ylim(-0.1, 0.9)

    # ── Panel C: Drug-Specific Edge Ablation ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if "drug_edge_abl" in data:
        sp = data["drug_edge_abl"].sort_values("delta_spearman", ascending=False)
        x3 = np.arange(len(sp))
        colors3 = [PANEL_COLORS["expected"] if v > 0 else "#2ECC71"
                   for v in sp["delta_spearman"]]
        bars = ax3.bar(x3, sp["delta_spearman"], color=colors3,
                       edgecolor="black", linewidth=0.5, alpha=0.85)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(sp["DRUG_NAME"], rotation=35, ha="right", fontsize=9)
        ax3.set_ylabel("ΔSpearman (baseline − drug-edge KO)")
        ax3.set_title("C. Drug Network Ablation\n"
                      "(Removing drug-specific KG edges reduces accuracy)",
                      fontweight="bold")
        ax3.axhline(0, color="black", linewidth=1.2)
        for bar, n in zip(bars, sp["n_edges_ablated"]):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{n}e", ha="center", va="bottom", fontsize=7)
    elif "ablation_full" in data:
        d = data["ablation_full"].dropna(subset=["delta_spearman"])
        d_kg = d[d["has_ChEMBL"] | d["has_PrimeKG"]].sort_values("delta_spearman", ascending=False)
        x3 = np.arange(len(d_kg))
        colors3 = [PANEL_COLORS["expected"] if v > 0 else "#2ECC71"
                   for v in d_kg["delta_spearman"]]
        ax3.bar(x3, d_kg["delta_spearman"], color=colors3,
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(d_kg["DRUG_NAME"], rotation=45, ha="right", fontsize=7)
        ax3.set_ylabel("ΔSpearman (baseline − full KO)")
        ax3.set_title("C. Full KG Knockout Effect\n(KG-covered test drugs)",
                      fontweight="bold")
        ax3.axhline(0, color="black", linewidth=1.2)

    # ── Panel D: Cancer-Type Specificity ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if "expected_test" in data:
        exp = data["expected_test"]
        x = np.arange(len(exp))
        w = 0.35
        ax4.bar(x - w/2, exp["delta_expected_mean"], w,
                label="Expected cancer\n(primary indication)",
                color=PANEL_COLORS["expected"], alpha=0.85, edgecolor="black", linewidth=0.5)
        ax4.bar(x + w/2, exp["delta_other_mean"], w,
                label="Other cancer types",
                color=PANEL_COLORS["other"], alpha=0.85, edgecolor="black", linewidth=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(
            [f"{r['DRUG_NAME']}\n({r['target']})" for _, r in exp.iterrows()],
            fontsize=8,
        )
        ax4.set_ylabel("Mean ΔAUC (drug KG ablation effect)")
        ax4.set_title("D. Cancer-Type-Specific KG Perturbation\n"
                      "(Larger effect in mechanism-matched cancer type)",
                      fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
        for i, (_, row) in enumerate(exp.iterrows()):
            sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01
                                                          else ("*" if row["p_value"] < 0.05 else "ns"))
            ymax = max(row["delta_expected_mean"], row["delta_other_mean"]) + 0.001
            ax4.text(i, ymax, sig, ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle(
        "Figure B: World Model Leverages Prior Knowledge Graphs for\n"
        "Drug Repurposing — Drug Split Test Set (Unseen Drugs)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(OUT_F / "FigureB_main.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_F / "FigureB_main.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_F / 'FigureB_main.pdf'}")


def _make_supplement_figure(data):
    """Supplementary figure: repurposing candidates and cancer ranking."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Drug-cancer sensitivity heatmap
    ax = axes[0]
    if "drug_cancer_map" in data:
        dcs = data["drug_cancer_map"]
        focal_drugs = [d for d in FOCAL_TEST_DRUGS.keys() if d in dcs["DRUG_NAME"].values]
        if focal_drugs:
            pivot = dcs[dcs["DRUG_NAME"].isin(focal_drugs)].pivot_table(
                index="DRUG_NAME", columns="cancer_group", values="mean_auc_hat", aggfunc="mean",
            )
            # Sort by mean sensitivity
            pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                           vmin=0.6, vmax=0.88)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            plt.colorbar(im, ax=ax, label="Mean predicted AUC\n(lower = more sensitive)")
            # Mark known indications
            for i, drug in enumerate(pivot.index):
                known = KNOWN_INDICATIONS.get(drug, set())
                for j, ct in enumerate(pivot.columns):
                    if ct in known and not np.isnan(pivot.values[i, j]):
                        ax.text(j, i, "★", ha="center", va="center",
                                fontsize=11, color="black", fontweight="bold")
    ax.set_title("Supp. B1: Cancer Sensitivity Map for Unseen Test Drugs\n"
                 "(★ = known clinical indication)", fontweight="bold")

    # Novel repurposing candidates
    ax2 = axes[1]
    if "repurposing" in data:
        rep = data["repurposing"]
        novel = rep[rep["repurposing_flag"]].nsmallest(20, "mean_auc_pred")
        if len(novel) > 0:
            colors = [PANEL_COLORS["expected"] if row["mean_auc_pred"] < 0.70
                      else "#FF9800" for _, row in novel.iterrows()]
            labels = [f"{row['DRUG_NAME']} → {row['cancer_group']}" for _, row in novel.iterrows()]
            ax2.barh(range(len(novel)), 1 - novel["mean_auc_pred"],
                     color=colors, edgecolor="black", linewidth=0.4, alpha=0.85)
            ax2.set_yticks(range(len(novel)))
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel("Predicted sensitivity score (1 − mean AUC)")
            ax2.set_title("Supp. B2: Novel Repurposing Hypotheses\n"
                          "(not in known clinical indications)", fontweight="bold")
            ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "Run B05 first", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    fig.savefig(OUT_F / "FigureB_supplement.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_F / "FigureB_supplement.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_F / 'FigureB_supplement.pdf'}")


def _make_mechanism_schematic(data):
    """Conceptual schematic: Static KG → Dynamic World Model."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.axis("off")

    # Draw schematic boxes
    boxes = [
        (0.05, 0.2, 0.15, 0.6, "#E8F4FD", "ChEMBL KG\n(MoA edges)\n[static]", "#2196F3"),
        (0.22, 0.2, 0.15, 0.6, "#E8F5E9", "DRKG\n(broad biology)\n[static]", "#4CAF50"),
        (0.39, 0.2, 0.15, 0.6, "#FFF3E0", "PrimeKG\n(protein\nnetwork)\n[static]", "#FF9800"),
    ]
    for x, y, w, h, fc, label, ec in boxes:
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                              facecolor=fc, edgecolor=ec, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, fontweight="bold", color=ec)

    # Arrow to fusion
    ax.annotate("", xy=(0.60, 0.5), xytext=(0.56, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text(0.58, 0.52, "α-weighted\nfusion\n(dynamic)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="black")

    # World model box
    wm = plt.Rectangle((0.60, 0.2), 0.18, 0.6, transform=ax.transAxes,
                        facecolor="#FCE4EC", edgecolor="#C62828", linewidth=2.5, zorder=2)
    ax.add_patch(wm)
    ax.text(0.69, 0.50, "World Model\n(drug-cell context)", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, fontweight="bold", color="#C62828")

    # Arrow to output
    ax.annotate("", xy=(0.85, 0.5), xytext=(0.79, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    # Output boxes
    out_items = [
        (0.86, 0.75, "Sensitivity\nPrediction\n(AUC)", "#F3E5F5", "#7B1FA2"),
        (0.86, 0.40, "Cancer-type\nSpecificity", "#E0F2F1", "#00695C"),
        (0.86, 0.05, "Repurposing\nHypotheses", "#FFF8E1", "#E65100"),
    ]
    for x, y, lbl, fc, ec in out_items:
        rect2 = plt.Rectangle((x, y), 0.13, 0.22, transform=ax.transAxes,
                               facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.add_patch(rect2)
        ax.text(x + 0.065, y + 0.11, lbl, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, fontweight="bold", color=ec)

    ax.text(0.5, 0.95, "Scenario B: Static Prior Knowledge Graphs → Dynamic World Model",
            transform=ax.transAxes, ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.88, "Drug split test drugs (never seen in training) — KG provides generalizable mechanism knowledge",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#555555")

    # DYNAMIC label
    ax.text(0.69, 0.78, "DYNAMIC:\nα changes per (drug, cell)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#C62828",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C62828"))
    # STATIC label
    ax.text(0.28, 0.88, "STATIC prior networks", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#555"))

    fig.savefig(OUT_F / "FigureB_schematic.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_F / "FigureB_schematic.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_F / 'FigureB_schematic.pdf'}")


if __name__ == "__main__":
    main()
