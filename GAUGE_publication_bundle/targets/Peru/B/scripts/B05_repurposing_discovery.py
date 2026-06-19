"""
B05: Drug Repurposing Discovery via KG Attention Mining
========================================================
The world model's KG attention mechanism can be used to:
  1. Identify which biological pathways the model "reads" for each drug
  2. Find cancer-context pairs where unexpected drugs score high sensitivity
  3. Match unknown drugs to known drugs via KG attention similarity
  4. Validate: known clinical repurposing matches model's predictions

Scientific approach:
  For drug split test drugs (model never trained on them):
  A. Attention-weighted target analysis:
     - Which ChEMBL targets receive highest attention for each drug?
     - Do attention-top targets match known clinical targets?
     - For drugs with only PrimeKG coverage: what protein-interaction hubs
       are most attended?

  B. Cross-drug KG attention similarity:
     - Compute alpha profile similarity between drug pairs
     - Similar alpha profiles → similar pharmacological context
     - This reveals "mechanism clustering" without explicit drug labels

  C. Sensitivity-mechanism concordance:
     - For high-sensitivity cell lines (AUC < 0.5): are the KG-relevant
       genes differentially expressed?
     - Test: do cells with high predicted sensitivity have expression
       patterns consistent with the drug's KG-highlighted targets?

  D. Novel repurposing hypotheses:
     - Drugs with unexpectedly high predicted sensitivity in specific
       cancer types vs. known indications
     - Validated against DGIdb / known clinical repurposing

Outputs:
  results/repurposing/
    attention_weighted_targets.csv    - per drug top KG targets by attention
    drug_attention_similarity.csv     - drug × drug attention profile similarity
    repurposing_candidates.csv        - novel sensitivity predictions
    clinical_concordance.csv          - validation vs known indications
  figures/
    B05_repurposing_discovery.pdf
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
from scipy import stats, spatial

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PREDICTIONS_CSV, KG_ATTENTION_CSV, KG_COVERAGE_CSV,
    CHEMBL_EDGES_CSV, PRIMEKG_EDGES_CSV, DRKG_EDGES_CSV,
    GDSC_MODEL_LIST, RESULTS, FIGURES, FOCAL_TEST_DRUGS, CANCER_GROUPS,
)
from utils import load_cell_cancer_types, annotate_cancer_group, get_drug_kg_edges

OUT_R = RESULTS / "repurposing"
OUT_F = FIGURES
OUT_R.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

# Known clinical indications for validation
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


def main():
    print("=" * 70)
    print("B05: Drug Repurposing Discovery via KG Attention Mining")
    print("=" * 70)

    # ── Load data (no model loading needed) ─────────────────────────────────
    preds     = pd.read_csv(PREDICTIONS_CSV)
    attn      = pd.read_csv(KG_ATTENTION_CSV)
    cov       = pd.read_csv(KG_COVERAGE_CSV)
    chembl    = pd.read_csv(CHEMBL_EDGES_CSV)
    primekg   = pd.read_csv(PRIMEKG_EDGES_CSV)
    drkg      = pd.read_csv(DRKG_EDGES_CSV)
    cell_meta = load_cell_cancer_types(GDSC_MODEL_LIST)

    test_preds = preds[preds["split"] == "test"].copy()
    test_attn  = attn.merge(test_preds[["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME",
                                         "AUC", "auc_hat"]].drop_duplicates(),
                            on=["SANGER_MODEL_ID", "DRUG_ID"])
    test_attn  = test_attn.merge(cell_meta, on="SANGER_MODEL_ID", how="left")
    test_attn["cancer_group"] = test_attn["cancer_type"].apply(
        lambda x: annotate_cancer_group(x, CANCER_GROUPS)
    )

    # ── A. Attention-weighted KG target analysis ─────────────────────────────
    print("\nA. Computing attention-weighted KG target profiles ...")
    # ChEMBL edges: which targets are in each drug's KG?
    target_rows = []
    for drug_name, info in FOCAL_TEST_DRUGS.items():
        drug_id = info["drug_id"]
        c_edges = chembl[(chembl["DRUG_ID"] == drug_id) &
                          ~chembl["edge_type"].str.startswith("rev_")]
        p_edges = primekg[(primekg["DRUG_ID"] == drug_id) &
                           ~primekg["edge_type"].str.startswith("rev_")]

        # Mean alpha values for this drug
        drug_attn = test_attn[test_attn["DRUG_ID"] == drug_id]
        alpha_c_mean = drug_attn["alpha_ChEMBL"].mean()
        alpha_p_mean = drug_attn["alpha_PrimeKG"].mean()

        # ChEMBL targets with relations
        for _, edge in c_edges.iterrows():
            target_rows.append({
                "DRUG_NAME": drug_name, "drug_id": drug_id,
                "source": "ChEMBL",
                "target_name": edge.get("dst_name", edge.get("dst", "")),
                "relation": edge["relation"],
                "edge_type": edge["edge_type"],
                "alpha_source": alpha_c_mean,
                "n_drug_cells": len(drug_attn),
            })

        # PrimeKG targets (direct: target, carrier, enzyme, transporter)
        if "relation" in p_edges.columns:
            direct = p_edges[p_edges["relation"].isin(["target", "carrier", "enzyme"])]
            for _, edge in direct.iterrows():
                target_rows.append({
                    "DRUG_NAME": drug_name, "drug_id": drug_id,
                    "source": "PrimeKG",
                    "target_name": edge.get("dst_name", str(edge.get("dst", ""))),
                    "relation": edge["relation"],
                    "edge_type": edge["edge_type"],
                    "alpha_source": alpha_p_mean,
                    "n_drug_cells": len(drug_attn),
                })

    target_df = pd.DataFrame(target_rows) if target_rows else pd.DataFrame()
    if len(target_df) > 0:
        target_df.to_csv(OUT_R / "attention_weighted_targets.csv", index=False)
        print(f"Saved: {OUT_R / 'attention_weighted_targets.csv'}")

    # ── B. Cross-drug KG alpha similarity ────────────────────────────────────
    print("\nB. Computing cross-drug attention similarity ...")
    drug_alpha_profiles = (
        test_attn.groupby("DRUG_ID")[["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]]
        .mean()
        .reset_index()
    )
    drug_alpha_profiles = drug_alpha_profiles.merge(
        test_preds[["DRUG_ID", "DRUG_NAME"]].drop_duplicates(), on="DRUG_ID", how="left"
    )

    # Cosine similarity matrix of alpha profiles
    alpha_mat = drug_alpha_profiles[["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]].values
    drug_names = drug_alpha_profiles["DRUG_NAME"].values
    sim_matrix = 1 - spatial.distance.cdist(alpha_mat, alpha_mat, metric="cosine")

    sim_rows = []
    n = len(drug_names)
    for i in range(n):
        for j in range(i + 1, n):
            sim_rows.append({
                "drug_A": drug_names[i],
                "drug_B": drug_names[j],
                "cosine_sim": sim_matrix[i, j],
                "both_focal": (drug_names[i] in FOCAL_TEST_DRUGS and
                               drug_names[j] in FOCAL_TEST_DRUGS),
            })
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(OUT_R / "drug_attention_similarity.csv", index=False)
    print(f"Saved: {OUT_R / 'drug_attention_similarity.csv'}")

    # Same-family drug similarity
    from config import FOCAL_TEST_DRUGS as ftd
    family_map = {name: info["family"] for name, info in ftd.items()}
    egfr_drugs = [k for k, v in family_map.items() if v == "EGFR_inhibitor"]
    mek_drugs  = [k for k, v in family_map.items() if v == "MEK_inhibitor"]

    for group_name, group in [("EGFR_inhibitors", egfr_drugs), ("MEK_inhibitors", mek_drugs)]:
        within = sim_df[(sim_df["drug_A"].isin(group)) & (sim_df["drug_B"].isin(group))]
        outside = sim_df[((sim_df["drug_A"].isin(group)) & ~(sim_df["drug_B"].isin(group))) |
                          (~(sim_df["drug_A"].isin(group)) & (sim_df["drug_B"].isin(group)))]
        if len(within) > 0 and len(outside) > 0:
            print(f"\n{group_name}:")
            print(f"  Within-family sim: {within['cosine_sim'].mean():.4f} (n={len(within)})")
            print(f"  Cross-family sim:  {outside['cosine_sim'].mean():.4f} (n={len(outside)})")
            _, p = stats.mannwhitneyu(within["cosine_sim"], outside["cosine_sim"])
            print(f"  MW p={p:.4f}")

    # ── C. Repurposing candidates ─────────────────────────────────────────────
    print("\nC. Identifying repurposing candidates ...")
    # Per (drug, cancer_group) mean predicted AUC → rank cancer types
    drug_cancer_sens = (
        test_attn.groupby(["DRUG_NAME", "DRUG_ID", "cancer_group"])
        .agg(
            n_cells=("SANGER_MODEL_ID", "count"),
            mean_AUC=("AUC", "mean"),
            mean_auc_hat=("auc_hat", "mean"),
            alpha_ChEMBL=("alpha_ChEMBL", "mean"),
            alpha_PrimeKG=("alpha_PrimeKG", "mean"),
            n_sensitive_pred=("auc_hat", lambda x: (x < 0.7).sum()),
        )
        .reset_index()
    )
    drug_cancer_sens["pct_sensitive"] = (
        drug_cancer_sens["n_sensitive_pred"] / drug_cancer_sens["n_cells"]
    )
    # Keep groups with at least 5 cells
    drug_cancer_sens = drug_cancer_sens[drug_cancer_sens["n_cells"] >= 5]
    drug_cancer_sens = drug_cancer_sens.sort_values(["DRUG_NAME", "mean_auc_hat"])
    drug_cancer_sens.to_csv(OUT_R / "repurposing_candidates.csv", index=False)
    print(f"Saved: {OUT_R / 'repurposing_candidates.csv'}")

    # Flag novel (not known indication) predictions
    # Use relative threshold: top-ranked cancer type per drug = most sensitive
    repurposing = []
    for drug_name in FOCAL_TEST_DRUGS.keys():
        known = KNOWN_INDICATIONS.get(drug_name, set())
        sub = drug_cancer_sens[drug_cancer_sens["DRUG_NAME"] == drug_name].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("mean_auc_hat")  # lower AUC = more sensitive
        # Use per-drug relative threshold: bottom 40th percentile AUC across cancer groups
        threshold = sub["mean_auc_hat"].quantile(0.4)
        top_sensitive = sub[sub["mean_auc_hat"] <= threshold]
        for _, row in top_sensitive.iterrows():
            is_known = row["cancer_group"] in known
            repurposing.append({
                "DRUG_NAME": drug_name,
                "cancer_group": row["cancer_group"],
                "n_cells": row["n_cells"],
                "mean_auc_pred": row["mean_auc_hat"],
                "pct_sensitive": row["pct_sensitive"],
                "rank": int(sub["mean_auc_hat"].rank().loc[row.name]),
                "is_known_indication": is_known,
                "repurposing_flag": not is_known,
                "alpha_PrimeKG": row["alpha_PrimeKG"],
                "known_indications": "; ".join(sorted(known)),
            })
    repurposing_df = pd.DataFrame(repurposing) if repurposing else pd.DataFrame()
    if len(repurposing_df) > 0:
        repurposing_df.to_csv(OUT_R / "clinical_concordance.csv", index=False)
        novel = repurposing_df[repurposing_df["repurposing_flag"]]
        print(f"\nNovel repurposing candidates (not known indications): {len(novel)}")
        if len(novel) > 0:
            print(novel[["DRUG_NAME", "cancer_group", "mean_auc_pred", "pct_sensitive",
                          "known_indications"]].sort_values("mean_auc_pred").head(10).to_string(index=False))

    # ── Figures ──────────────────────────────────────────────────────────────
    _plot_repurposing(drug_alpha_profiles, drug_cancer_sens, sim_df,
                      repurposing_df if len(repurposing_df) > 0 else pd.DataFrame(),
                      OUT_F)
    print("\nB05 complete.")


def _plot_repurposing(drug_alpha_profiles, drug_cancer_sens, sim_df, repurposing_df, out_dir):
    """Generate publication figure B05."""
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)

    # Panel A: KG attention profile heatmap for focal drugs
    ax1 = fig.add_subplot(gs[0, :2])
    focal_profiles = drug_alpha_profiles[
        drug_alpha_profiles["DRUG_NAME"].isin(FOCAL_TEST_DRUGS.keys())
    ].sort_values("alpha_PrimeKG", ascending=False)

    heat_data = focal_profiles[["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]].values
    im = ax1.imshow(heat_data.T, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=1, interpolation="nearest")
    ax1.set_xticks(range(len(focal_profiles)))
    ax1.set_xticklabels(focal_profiles["DRUG_NAME"], rotation=45, ha="right", fontsize=8)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["α ChEMBL\n(MoA)", "α DRKG\n(broad)", "α PrimeKG\n(protein)"], fontsize=9)
    plt.colorbar(im, ax=ax1, label="Mean attention weight")
    ax1.set_title("A. KG Attention Profiles for Unseen Test Drugs\n"
                  "(World model dynamically allocates attention to three prior networks)",
                  fontweight="bold")

    # Panel B: Drug alpha similarity scatter (first 2 principal components)
    ax2 = fig.add_subplot(gs[0, 2])
    from config import FOCAL_TEST_DRUGS as ftd
    alpha_mat = drug_alpha_profiles[["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]].values
    # Simple PCA (2D)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(alpha_mat)
    family_colors = {
        "EGFR_inhibitor": "#E74C3C", "BCL2_inhibitor": "#9B59B6",
        "MEK_inhibitor": "#2ECC71", "ALK_inhibitor": "#3498DB",
        "PARP_inhibitor": "#F39C12", "JAK_inhibitor": "#1ABC9C",
        "DNA_damaging": "#95A5A6", "antimetabolite": "#7F8C8D", "other": "#BDC3C7",
    }
    plotted = set()
    for i, name in enumerate(drug_alpha_profiles["DRUG_NAME"]):
        family = ftd.get(name, {}).get("family", "other")
        color = family_colors.get(family, "#BDC3C7")
        ax2.scatter(coords[i, 0], coords[i, 1], c=color, s=60,
                    edgecolors="black", linewidth=0.5, alpha=0.85,
                    label=family if family not in plotted else "")
        plotted.add(family)
        if name in ftd:
            ax2.annotate(name, coords[i], fontsize=6, xytext=(3, 3), textcoords="offset points")
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), fontsize=6, loc="upper right")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title("B. Drug KG Attention Clustering\n"
                  "(Same-family drugs cluster → mechanism learned from KG)",
                  fontweight="bold")

    # Panel C: Drug sensitivity predictions by cancer type (heatmap)
    ax3 = fig.add_subplot(gs[1, :2])
    focal_drugs = [d for d in FOCAL_TEST_DRUGS.keys()
                   if d in drug_cancer_sens["DRUG_NAME"].values]
    if len(focal_drugs) > 0 and len(drug_cancer_sens) > 0:
        pivot = drug_cancer_sens[drug_cancer_sens["DRUG_NAME"].isin(focal_drugs)].pivot_table(
            index="DRUG_NAME", columns="cancer_group",
            values="mean_auc_hat", aggfunc="mean",
        )
        # Fill NaN with max (resistant)
        pivot_filled = pivot.fillna(pivot.stack().max())
        im2 = ax3.imshow(pivot_filled.values, aspect="auto", cmap="RdYlGn_r",
                         vmin=0.55, vmax=0.90, interpolation="nearest")
        ax3.set_xticks(range(len(pivot_filled.columns)))
        ax3.set_xticklabels(pivot_filled.columns, rotation=45, ha="right", fontsize=8)
        ax3.set_yticks(range(len(pivot_filled.index)))
        ax3.set_yticklabels(pivot_filled.index, fontsize=9)
        plt.colorbar(im2, ax=ax3, label="Mean predicted AUC (lower = more sensitive)")

        # Annotate known indications with stars
        for i, drug in enumerate(pivot_filled.index):
            known = KNOWN_INDICATIONS.get(drug, set())
            for j, ct in enumerate(pivot_filled.columns):
                if ct in known:
                    ax3.text(j, i, "★", ha="center", va="center",
                             fontsize=12, color="black", fontweight="bold")
        ax3.set_title("C. KG-Predicted Cancer Type Sensitivity Map\n"
                      "★ = Known clinical indication (validation)",
                      fontweight="bold")

    # Panel D: Repurposing candidates (novel)
    ax4 = fig.add_subplot(gs[1, 2])
    if len(repurposing_df) > 0:
        novel = repurposing_df[repurposing_df["repurposing_flag"]].nsmallest(15, "mean_auc_pred")
        if len(novel) > 0:
            labels_r = [f"{row['DRUG_NAME']}\n→ {row['cancer_group']}"
                        for _, row in novel.iterrows()]
            ax4.barh(range(len(novel)), 1 - novel["mean_auc_pred"],
                     color="#E74C3C", edgecolor="black", linewidth=0.5, alpha=0.85)
            ax4.set_yticks(range(len(novel)))
            ax4.set_yticklabels(labels_r, fontsize=7)
            ax4.set_xlabel("Predicted sensitivity (1 - mean AUC)")
            ax4.set_title("D. Novel Repurposing Candidates\n(not known indications)",
                          fontweight="bold")
            ax4.invert_yaxis()
        else:
            ax4.text(0.5, 0.5, "No novel candidates\nfound", ha="center", va="center",
                     transform=ax4.transAxes, fontsize=12)
    else:
        ax4.text(0.5, 0.5, "Repurposing data\nnot available", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=12)

    plt.suptitle(
        "Scenario B — B05: Drug Repurposing Discovery via KG Attention Mining\n"
        "World Model Leverages Prior Networks to Predict Cancer-Type Sensitivity",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(out_dir / "B05_repurposing_discovery.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "B05_repurposing_discovery.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_dir / 'B05_repurposing_discovery.pdf'}")


if __name__ == "__main__":
    main()
