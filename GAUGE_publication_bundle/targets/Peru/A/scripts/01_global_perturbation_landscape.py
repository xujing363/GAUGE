"""
Script 01: Global In-Silico Perturbation Landscape
====================================================
Scenario A: Perturb transcriptome (gene silencing/overexpression), fix drug.
Cell-line split test set: 190 unseen cell lines the model has never seen.

For each gene in GENE_PANEL × each drug with ≥10 test cell lines:
  delta_AUC = AUC_baseline - AUC_gene_silenced
  Positive → gene promotes sensitivity; Negative → gene promotes resistance

Outputs (saved to results/01_global/):
  global_perturbation.csv           - full gene × drug × delta_AUC table
  gene_summary.csv                  - per-gene mean absolute effect across drugs
  drug_summary.csv                  - per-drug most impactful perturbation genes
  known_target_validation.csv       - validation against curated drug-target pairs
  top_gene_drug_pairs.csv           - top 50 most specific gene-drug interactions

Usage:
    python scripts/01_global_perturbation_landscape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    BATCH_SIZE,
    CONFIG_YAML,
    DEVICE,
    GDSC_MODEL_LIST,
    GENE_PANEL,
    GENE_PANEL_EXTENDED,
    KNOWN_DRUG_TARGETS,
    MIN_SAMPLES_PER_DRUG,
    PREPARED_PKL,
    RESULTS_DIR,
    RESULT_DIR,
)
from utils import (
    add_cell_metadata,
    build_inference_context,
    load_experiment,
    perturb_state_at_gene,
)

OUT_DIR = RESULTS_DIR / "01_global"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_top_genes_by_pca_loading(prepared, top_k=50):
    """
    Rank HVG genes by their aggregate influence on the PCA state representation.

    Gene influence ≈ ||W[:,g]||₁ / σ_g
    where W is the PCA loading matrix (n_pca × n_genes) and σ_g is the scaler std.
    This is the L1 norm of gene g's projection vector, normalized by expression variance.

    This deterministic approximation is equivalent to the expected |∂AUC/∂x_g| under
    uniform gradient magnitude in PCA space, and avoids backward-pass conflicts with
    the precomputed KG payload.
    """
    pca_components = prepared.artifacts.pca.components_  # (n_pca, n_genes)
    scaler_scale   = prepared.artifacts.scaler.scale_    # (n_genes,)

    # L2 norm of PCA loadings for each gene, normalized by std
    gene_pca_norm = np.abs(pca_components).sum(axis=0) / (scaler_scale + 1e-10)
    top_indices = np.argsort(gene_pca_norm)[::-1][:top_k]
    genes_list = prepared.artifacts.genes
    return [genes_list[i] for i in top_indices], gene_pca_norm


def main():
    print("=" * 70)
    print("Script 01: Global In-Silico Perturbation Landscape")
    print("Scenario A — Cell-line split test set (190 unseen cell lines)")
    print("=" * 70)

    # ── Load model ─────────────────────────────────────────────────────────
    model, prepared, config = load_experiment(PREPARED_PKL, RESULT_DIR, CONFIG_YAML, DEVICE)

    genes = prepared.artifacts.genes
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    print(f"  Total genes in model: {len(genes)}")

    # ── Get test set ───────────────────────────────────────────────────────
    test_frame = prepared.responses[prepared.responses["split"] == "test"].copy()
    test_frame = test_frame.reset_index(drop=True)
    print(f"  Test rows: {len(test_frame)}")
    print(f"  Test cell lines: {test_frame['SANGER_MODEL_ID'].nunique()}")
    print(f"  Test drugs: {test_frame['DRUG_ID'].nunique()}")

    # ── Build inference infrastructure ─────────────────────────────────────
    infra = build_inference_context(model, prepared, config, test_frame, DEVICE)

    # ── PCA/scaler artifacts ───────────────────────────────────────────────
    pca_components = prepared.artifacts.pca.components_      # (n_pca, n_genes)
    scaler_scale   = prepared.artifacts.scaler.scale_
    imputer_stats  = prepared.artifacts.imputer.statistics_

    # ── Step 1: Rank HVG genes by PCA loading importance ───────────────────
    # Approximate input sensitivity: ||W[:,g]||₁ / σ_g (analytic, no backward pass)
    print("\nStep 1: Ranking 2000 HVG genes by PCA loading importance ...")
    top_hvg_genes, gene_pca_norm = compute_top_genes_by_pca_loading(prepared, top_k=50)
    print(f"  Top 50 HVG genes by PCA loading identified")

    # ── Filter gene panel to available genes ───────────────────────────────
    # Combine curated oncology panel with PCA-loading top genes
    extended_panel = list(set(GENE_PANEL_EXTENDED + top_hvg_genes))
    candidate_genes = [(g, gene_to_idx[g]) for g in extended_panel if g in gene_to_idx]

    # Save HVG importance ranking
    all_genes = prepared.artifacts.genes
    hvg_df = pd.DataFrame({
        "gene_name": all_genes,
        "pca_loading_importance": gene_pca_norm.tolist(),
    }).sort_values("pca_loading_importance", ascending=False).reset_index(drop=True)
    hvg_df["pca_rank"] = range(1, len(hvg_df) + 1)
    hvg_df["in_oncology_panel"] = hvg_df["gene_name"].isin(set(GENE_PANEL_EXTENDED))
    hvg_df.to_csv(OUT_DIR / "hvg_pca_importance_ranking.csv", index=False)
    print(f"  Combined panel: {len(candidate_genes)} genes for perturbation")

    # ── Drug list (>= MIN_SAMPLES_PER_DRUG test cells) ─────────────────────
    drug_counts = test_frame.groupby(["DRUG_ID", "DRUG_NAME"]).size().reset_index(name="n_test_cells")
    drug_counts = drug_counts[drug_counts["n_test_cells"] >= MIN_SAMPLES_PER_DRUG]
    print(f"  Drugs with ≥{MIN_SAMPLES_PER_DRUG} test cells: {len(drug_counts)}")

    # ── Run global perturbation ────────────────────────────────────────────
    print(f"\nRunning {len(candidate_genes)} genes × {len(drug_counts)} drugs ...")

    rows = []
    for gi, (gene_name, gene_idx) in enumerate(candidate_genes):
        if gi % 10 == 0:
            print(f"  Gene {gi+1}/{len(candidate_genes)}: {gene_name}")

        for _, drug_row in drug_counts.iterrows():
            drug_id   = int(drug_row["DRUG_ID"])
            drug_name = str(drug_row["DRUG_NAME"])
            mask = test_frame["DRUG_ID"] == drug_id
            drug_test = test_frame[mask]
            local_idx = drug_test.index.to_numpy()

            tensors     = infra["tensors"]
            state_idx   = tensors.state_idx[local_idx]
            drug_idx_t  = tensors.drug_idx[local_idx]
            tensor_banks = infra["tensor_banks"]
            kg_idx      = infra["kg_drug_idx_bank"]
            kg_payload  = infra["precomputed_kg_payload"]

            baseline_auc = []
            pert_auc     = []

            model.eval()
            with torch.no_grad():
                for start in range(0, len(local_idx), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(local_idx))
                    bidx = local_idx[start:end]

                    si = tensors.state_idx[bidx]
                    di = tensors.drug_idx[bidx]
                    s  = tensor_banks.state_bank.index_select(0, si)
                    fp = tensor_banks.fp_bank.index_select(0, di)
                    ki = kg_idx.index_select(0, di) if kg_idx is not None else None

                    out_b = model(state=s, drug_fp=fp, drug_idx=ki,
                                  use_prior=True, precomputed_kg_payload=kg_payload)
                    baseline_auc.append(out_b["auc_hat"].cpu().numpy())

                    s_pert = perturb_state_at_gene(s, gene_idx, pca_components, scaler_scale, imputer_stats)
                    out_p = model(state=s_pert, drug_fp=fp, drug_idx=ki,
                                  use_prior=True, precomputed_kg_payload=kg_payload)
                    pert_auc.append(out_p["auc_hat"].cpu().numpy())

            base_arr = np.concatenate(baseline_auc)
            pert_arr = np.concatenate(pert_auc)
            delta    = base_arr - pert_arr  # positive = silencing reduces sensitivity

            is_target = gene_name in KNOWN_DRUG_TARGETS.get(drug_name, [])
            rows.append({
                "gene_name":       gene_name,
                "DRUG_ID":         drug_id,
                "DRUG_NAME":       drug_name,
                "n_test_cells":    len(drug_test),
                "mean_delta_auc":  float(delta.mean()),
                "std_delta_auc":   float(delta.std()),
                "abs_mean_delta":  float(np.abs(delta).mean()),
                "frac_positive":   float((delta > 0).mean()),
                "is_known_target": bool(is_target),
            })

    pert_df = pd.DataFrame(rows)
    pert_df.to_csv(OUT_DIR / "global_perturbation.csv", index=False)
    print(f"\nSaved: {len(pert_df)} gene-drug perturbation records")

    # ── Gene-level summary ─────────────────────────────────────────────────
    gene_summary = (
        pert_df.groupby("gene_name")
        .agg(
            n_drugs_tested=("DRUG_ID", "nunique"),
            mean_abs_delta=("abs_mean_delta", "mean"),
            mean_signed_delta=("mean_delta_auc", "mean"),
            max_abs_delta=("abs_mean_delta", "max"),
            frac_positive_across_drugs=("frac_positive", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_delta", ascending=False)
    )
    gene_summary.to_csv(OUT_DIR / "gene_summary.csv", index=False)

    # ── Drug-level summary ─────────────────────────────────────────────────
    drug_summary_rows = []
    for drug_id, grp in pert_df.groupby("DRUG_ID"):
        grp_sorted = grp.sort_values("abs_mean_delta", ascending=False)
        top_genes = grp_sorted.head(5)["gene_name"].tolist()
        drug_summary_rows.append({
            "DRUG_ID":          drug_id,
            "DRUG_NAME":        grp["DRUG_NAME"].iloc[0],
            "n_genes_tested":   len(grp),
            "max_abs_delta":    grp["abs_mean_delta"].max(),
            "mean_abs_delta":   grp["abs_mean_delta"].mean(),
            "top5_genes":       "|".join(top_genes),
            "n_known_targets":  grp["is_known_target"].sum(),
        })
    drug_summary = pd.DataFrame(drug_summary_rows).sort_values("max_abs_delta", ascending=False)
    drug_summary.to_csv(OUT_DIR / "drug_summary.csv", index=False)

    # ── Known-target validation ────────────────────────────────────────────
    known = pert_df[pert_df["is_known_target"]].copy()
    unknown = pert_df[~pert_df["is_known_target"]].copy()
    known.to_csv(OUT_DIR / "known_target_validation.csv", index=False)

    print("\n--- Known Target Validation ---")
    print(f"Known gene-drug pairs: {len(known)}")
    print(f"Mean |ΔAU| for known targets: {known['abs_mean_delta'].mean():.5f}")
    print(f"Mean |ΔAU| for non-targets:   {unknown['abs_mean_delta'].mean():.5f}")
    if len(known) > 0 and len(unknown) > 0:
        enrichment = known['abs_mean_delta'].mean() / unknown['abs_mean_delta'].mean()
        print(f"Enrichment (known/non-target): {enrichment:.2f}×")
    for _, row in known.sort_values("abs_mean_delta", ascending=False).iterrows():
        print(f"  {row['DRUG_NAME']:25s} ← {row['gene_name']:10s} | "
              f"ΔAU={row['mean_delta_auc']:+.4f} | |ΔAU|={row['abs_mean_delta']:.4f}")

    # ── Top gene-drug pairs (most specific) ───────────────────────────────
    top_pairs = pert_df.sort_values("abs_mean_delta", ascending=False).head(50)
    top_pairs.to_csv(OUT_DIR / "top_gene_drug_pairs.csv", index=False)

    # ── Summary stats ─────────────────────────────────────────────────────
    print("\n--- Top 15 Most Impactful Genes (mean |ΔAU| across drugs) ---")
    for _, row in gene_summary.head(15).iterrows():
        print(f"  {row['gene_name']:15s} | |ΔAU|={row['mean_abs_delta']:.5f} "
              f"| max={row['max_abs_delta']:.4f} | n_drugs={row['n_drugs_tested']}")

    print(f"\nAll outputs → {OUT_DIR}")
    print("Script 01 complete.")


if __name__ == "__main__":
    main()
