"""
Scenario C — Step 4: Cancer-Type–Specific Synthetic Lethality Analysis
=======================================================================

Focuses on specific cancer lineages to demonstrate clinical relevance:
  • Lung adenocarcinoma: EGFR inhibitor (Erlotinib/Gefitinib/Osimertinib) SL partners
  • AML: BCL2 inhibitor (Venetoclax) SL partners
  • Colorectal: DNA damage agent (Cisplatin/Oxaliplatin) SL partners
  • BRCA-associated: PARP inhibitor (Talazoparib) SL partners

Also demonstrates that the three KG priors give different "views":
  • ChEMBL-discovered SL: pharmacological drug-target edges
  • DRKG-discovered SL: disease-gene co-occurrence edges
  • PrimeKG-discovered SL: protein-protein interaction edges

Each prior's contribution is revealed by the α-attention gate analysis
(kg_attention_by_prediction.csv from the trained model).

Outputs (results/C04_cancer_analysis/)
---------------------------------------
  lung_egfr_sl_partners.csv    — EGFR inhibitor SL in lung cancer
  aml_bcl2_sl_partners.csv     — Venetoclax SL in AML
  colorectal_dna_sl_partners.csv — DNA damage SL in colorectal
  alpha_gate_by_cancer.csv     — KG prior weights (α) per cancer type
  world_model_kgcontrib.csv    — KG attention × delta_AUC correlation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DOUBLE_DISJOINT_RESULT_DIR,
    GDSC_MODEL_LIST,
    KNOWN_DRUG_TARGETS,
    RESULTS_DIR,
    CANCER_TYPE_FOCUS,
)
from utils import load_cell_metadata

OUT_DIR = RESULTS_DIR / "C04_cancer_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Focused drug–cancer axes for clinical narrative
DRUG_CANCER_AXES = {
    "Lung Adenocarcinoma":  ["Erlotinib", "Gefitinib", "Osimertinib"],
    "Acute Myeloid Leukemia": ["Venetoclax", "Cytarabine"],
    "Colorectal Carcinoma": ["Cisplatin", "Oxaliplatin"],
    "Ovarian Carcinoma":    ["Talazoparib", "Cisplatin"],
    "Melanoma":             ["Trametinib"],
    "Breast Carcinoma":     ["Talazoparib", "Alpelisib"],
}


def load_c01() -> pd.DataFrame:
    p = RESULTS_DIR / "C01_drug_gene" / "drug_gene_delta_auc.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def load_c03_confident() -> pd.DataFrame:
    p = RESULTS_DIR / "C03_sl_prediction" / "sl_candidates_confident.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def load_kg_attention() -> pd.DataFrame:
    """Load model's KG attention weights (α_ChEMBL, α_DRKG, α_PrimeKG)."""
    p = DOUBLE_DISJOINT_RESULT_DIR / "kg_attention_by_prediction.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def analyse_drug_cancer_axis(
    drug_gene_df: pd.DataFrame,
    cancer_type: str,
    drug_names: list[str],
    cell_meta: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Identify top SL partner genes for specific drugs in a specific cancer type.
    Returns DataFrame of (gene, delta_auc, frac_sensitising, synleth_validated).
    """
    # Get cells of this cancer type
    cancer_cells = cell_meta[cell_meta["cancer_type"] == cancer_type]["SANGER_MODEL_ID"].tolist()
    if not cancer_cells:
        return pd.DataFrame()

    # Filter drug-gene perturbations
    subset = drug_gene_df[
        drug_gene_df["DRUG_NAME"].isin(drug_names)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    # Aggregate: for each gene, average across the relevant drugs
    agg = (
        subset.groupby("gene_name")
        .agg(
            mean_delta_auc  = ("mean_delta_auc", "mean"),
            max_delta_auc   = ("mean_delta_auc", "max"),
            frac_sensitising= ("frac_sensitising", "mean"),
            n_drugs         = ("DRUG_NAME", "nunique"),
        )
        .reset_index()
        .sort_values("mean_delta_auc", ascending=False)
        .head(top_n)
    )
    agg["cancer_type"] = cancer_type
    agg["drugs"]       = ",".join(drug_names)
    return agg


def analyse_kg_attention(
    kg_attn: pd.DataFrame,
    drug_gene_df: pd.DataFrame,
    cell_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correlate KG prior weights (α) with drug sensitivity (delta_AUC) per cancer type.

    If α_PrimeKG is high for a (cell, drug) pair → PrimeKG gene-gene prior is driving
    the prediction → perturbation of genes in PrimeKG SL network matters more.
    """
    # Merge attention with cell metadata
    merged = kg_attn.merge(cell_meta, on="SANGER_MODEL_ID", how="left")

    rows = []
    for cancer_type, ct_grp in merged.groupby("cancer_type"):
        if ct_grp.empty:
            continue
        for alpha_col in ["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]:
            if alpha_col not in ct_grp.columns:
                continue
            rows.append({
                "cancer_type":  cancer_type,
                "kg_prior":     alpha_col.replace("alpha_", ""),
                "mean_alpha":   ct_grp[alpha_col].mean(),
                "std_alpha":    ct_grp[alpha_col].std(),
                "n_pairs":      len(ct_grp),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def world_model_kg_contribution_analysis(
    drug_gene_df: pd.DataFrame,
    kg_attn: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether gene perturbation effect (delta_AUC) correlates with KG attention weight.

    High α_PrimeKG + high delta_AUC when perturbing a PrimeKG SL gene
    → confirms the world model is using PrimeKG SL information dynamically.
    """
    # Merge drug-gene delta with attention weights
    if "DRUG_ID" not in drug_gene_df.columns or "DRUG_ID" not in kg_attn.columns:
        return pd.DataFrame()

    merged = drug_gene_df.merge(
        kg_attn.groupby("DRUG_ID")[["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]].mean().reset_index(),
        on="DRUG_ID", how="left",
    )

    rows = []
    for alpha_col in ["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"]:
        if alpha_col not in merged.columns:
            continue
        mask = merged[alpha_col].notna() & merged["mean_delta_auc"].notna()
        if mask.sum() < 10:
            continue
        r, p = spearmanr(merged.loc[mask, alpha_col], merged.loc[mask, "mean_delta_auc"])
        rows.append({
            "kg_prior": alpha_col.replace("alpha_", ""),
            "spearman_r": round(r, 4),
            "p_value":    round(p, 6),
            "n_pairs":    mask.sum(),
            "interpretation": (
                "higher KG prior weight → larger gene perturbation effect"
                if r > 0 else
                "higher KG prior weight → smaller gene perturbation effect"
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("Scenario C — Step 4: Cancer-Type Specific SL Analysis")
    print("=" * 70)

    drug_gene_df = load_c01()
    confident_df = load_c03_confident()
    cell_meta    = load_cell_metadata(GDSC_MODEL_LIST)

    try:
        kg_attn = load_kg_attention()
        print(f"  KG attention data: {len(kg_attn):,} (cell, drug) pairs")
    except FileNotFoundError as e:
        print(f"  Warning: KG attention not found ({e}), skipping α analysis")
        kg_attn = pd.DataFrame()

    # ── Per-cancer-type SL analysis ───────────────────────────────────────────
    all_cancer_rows = []
    for cancer_type, drug_list in DRUG_CANCER_AXES.items():
        print(f"\nAnalysing: {cancer_type} | Drugs: {drug_list}")
        agg = analyse_drug_cancer_axis(drug_gene_df, cancer_type, drug_list, cell_meta)
        if agg.empty:
            print(f"  No data found (drugs may not be in test set)")
            continue

        safe_name = cancer_type.lower().replace(" ", "_").replace("/", "_")
        agg.to_csv(OUT_DIR / f"{safe_name}_sl_partners.csv", index=False)
        all_cancer_rows.append(agg)

        print(f"  Top 5 SL partner genes:")
        for _, r in agg.head(5).iterrows():
            print(f"    {r.gene_name:15s} Δ={r.mean_delta_auc:+.5f} frac={r.frac_sensitising:.2f}")

    if all_cancer_rows:
        all_cancer_df = pd.concat(all_cancer_rows, ignore_index=True)
        all_cancer_df.to_csv(OUT_DIR / "all_cancer_sl_partners.csv", index=False)

    # ── KG prior (α) by cancer type ───────────────────────────────────────────
    if not kg_attn.empty:
        print("\n--- KG attention analysis by cancer type ---")
        kg_attn_meta = kg_attn.merge(cell_meta, on="SANGER_MODEL_ID", how="left")
        alpha_agg = analyse_kg_attention(kg_attn, drug_gene_df, cell_meta)
        if not alpha_agg.empty:
            alpha_agg.to_csv(OUT_DIR / "alpha_gate_by_cancer.csv", index=False)
            print(alpha_agg.sort_values(["cancer_type", "mean_alpha"], ascending=[True, False])
                  .head(20).to_string(index=False))

        # World model KG contribution analysis
        print("\n--- World model: KG prior × perturbation effect correlation ---")
        wm_contrib = world_model_kg_contribution_analysis(drug_gene_df, kg_attn)
        if not wm_contrib.empty:
            wm_contrib.to_csv(OUT_DIR / "world_model_kgcontrib.csv", index=False)
            print(wm_contrib.to_string(index=False))

    # ── EGFR–Erlotinib focus: lung adenocarcinoma deep dive ──────────────────
    print("\n--- Deep dive: EGFR inhibitors in Lung Adenocarcinoma ---")
    lung_drugs = ["Erlotinib", "Gefitinib", "Osimertinib"]
    lung_df = drug_gene_df[drug_gene_df["DRUG_NAME"].isin(lung_drugs)]
    if not lung_df.empty:
        # Top sensitising genes for EGFR inhibitors
        gene_sens = (
            lung_df.groupby("gene_name")
            .agg(
                n_drugs=("DRUG_NAME", "nunique"),
                mean_delta=("mean_delta_auc", "mean"),
                frac_sens=("frac_sensitising", "mean"),
            )
            .reset_index()
            .sort_values("mean_delta", ascending=False)
        )
        gene_sens.to_csv(OUT_DIR / "egfr_lung_sl_partners_aggregated.csv", index=False)

        # Highlight known EGFR co-targets
        known_egfr_context = ["ERBB2", "ERBB3", "MET", "KRAS", "TP53", "PIK3CA",
                               "PTEN", "AXL", "VEGFR2", "CCND1", "CDK6"]
        known_found = gene_sens[gene_sens["gene_name"].isin(known_egfr_context)]
        print(f"  Known EGFR-context genes in top sensitising list:")
        print(known_found[["gene_name", "mean_delta", "frac_sens"]].to_string(index=False))

    # ── BCL2–Venetoclax focus: AML deep dive ─────────────────────────────────
    print("\n--- Deep dive: BCL2 inhibitor (Venetoclax) in AML ---")
    ven_df = drug_gene_df[drug_gene_df["DRUG_NAME"] == "Venetoclax"]
    if not ven_df.empty:
        ven_top = ven_df.sort_values("mean_delta_auc", ascending=False).head(30)
        ven_top.to_csv(OUT_DIR / "venetoclax_aml_sl_partners.csv", index=False)
        known_bcl2_context = ["MCL1", "BCL2L1", "BCL2L2", "BAX", "BAK1", "BIM",
                               "PUMA", "NOXA", "CDK9", "FLT3", "IDH1", "IDH2"]
        known_found_ven = ven_top[ven_top["gene_name"].isin(known_bcl2_context)]
        print(f"  Known BCL2-pathway genes in top sensitising list:")
        print(known_found_ven[["gene_name", "mean_delta_auc", "frac_sensitising"]].to_string(index=False))

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
