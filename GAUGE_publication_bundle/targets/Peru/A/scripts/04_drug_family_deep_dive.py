"""
Script 04: Drug Family Deep Dive — Target Specificity and Cross-Reactivity
===========================================================================
For each drug family (EGFR/BRAF/MEK/BCL2/PARP/CDK4-6 inhibitors):
  1. Rank genes by sensitivity effect within the family
  2. Show that target genes rank highest (within-family target specificity)
  3. Show cross-reactivity: related pathway genes also affect sensitivity
  4. Compare within-family vs. out-of-family gene-drug specificity

Key question: Can the model distinguish EGFR→Erlotinib from EGFR→Venetoclax?
(i.e., does it capture drug-target specificity, not just "important gene" effects?)

Specificity score per gene g:
    spec(g, family) = mean_{d∈family} |ΔAU(g,d)| / mean_{d∉family} |ΔAU(g,d)|

Outputs (saved to results/04_drug_family/):
  drug_family_perturbation.csv         - full perturbation table
  drug_family_specificity.csv          - target gene specificity scores
  family_rank_heatmap.csv              - gene ranks within each drug family
  within_vs_cross_family.csv           - within vs cross-family ΔAU comparison

Usage:
    python scripts/04_drug_family_deep_dive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    BATCH_SIZE,
    CONFIG_YAML,
    DEVICE,
    DRUG_FAMILIES,
    GDSC_MODEL_LIST,
    GENE_PANEL,
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

OUT_DIR = RESULTS_DIR / "04_drug_family"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Script 04: Drug Family Deep Dive — Target Specificity")
    print("=" * 70)

    model, prepared, config = load_experiment(PREPARED_PKL, RESULT_DIR, CONFIG_YAML, DEVICE)

    genes = prepared.artifacts.genes
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    test_frame = prepared.responses[prepared.responses["split"] == "test"].copy()
    test_frame = test_frame.reset_index(drop=True)
    print(f"  Test rows: {len(test_frame)}, Test cells: {test_frame['SANGER_MODEL_ID'].nunique()}")

    infra = build_inference_context(model, prepared, config, test_frame, DEVICE)
    tensors      = infra["tensors"]
    tensor_banks = infra["tensor_banks"]
    kg_idx_bank  = infra["kg_drug_idx_bank"]
    kg_payload   = infra["precomputed_kg_payload"]

    pca_components = prepared.artifacts.pca.components_
    scaler_scale   = prepared.artifacts.scaler.scale_
    imputer_stats  = prepared.artifacts.imputer.statistics_

    # ── Check which drugs are available in test set ─────────────────────────
    test_drugs = set(test_frame["DRUG_NAME"].unique())
    available_families = {}
    for family, drugs in DRUG_FAMILIES.items():
        avail = [d for d in drugs if d in test_drugs]
        if avail:
            available_families[family] = avail

    print(f"\n  Available drug families: {len(available_families)}")
    for fam, drugs in available_families.items():
        print(f"    {fam}: {drugs}")

    # ── Collect all family drugs for perturbation ──────────────────────────
    all_family_drugs = []
    for drugs in available_families.values():
        all_family_drugs.extend(drugs)
    all_family_drugs = list(set(all_family_drugs))

    # Also include background drugs (sample of non-family drugs)
    all_test_drugs = test_frame["DRUG_NAME"].unique()
    background_drugs = [d for d in all_test_drugs if d not in all_family_drugs]
    # Sample up to 30 background drugs for specificity computation
    rng = np.random.default_rng(42)
    if len(background_drugs) > 30:
        background_drugs = rng.choice(background_drugs, 30, replace=False).tolist()

    study_drugs = all_family_drugs + background_drugs
    print(f"\n  Study drugs: {len(all_family_drugs)} family + {len(background_drugs)} background")

    # ── Run perturbations ─────────────────────────────────────────────────
    candidate_genes = [(g, gene_to_idx[g]) for g in GENE_PANEL if g in gene_to_idx]
    print(f"  Candidate genes: {len(candidate_genes)}")
    print(f"\nRunning {len(candidate_genes)} genes × {len(study_drugs)} drugs ...")

    pert_rows = []
    for gi, (gene_name, gene_idx) in enumerate(candidate_genes):
        if gi % 10 == 0:
            print(f"  Gene {gi+1}/{len(candidate_genes)}: {gene_name}")

        for drug_name in study_drugs:
            drug_mask = test_frame["DRUG_NAME"] == drug_name
            drug_frame = test_frame[drug_mask]
            if len(drug_frame) < MIN_SAMPLES_PER_DRUG:
                continue
            local_idx = drug_frame.index.to_numpy()

            model.eval()
            base_auc_list = []
            pert_auc_list = []
            with torch.no_grad():
                for start in range(0, len(local_idx), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(local_idx))
                    bidx = local_idx[start:end]
                    si = tensors.state_idx[bidx]
                    di = tensors.drug_idx[bidx]
                    s  = tensor_banks.state_bank.index_select(0, si)
                    fp = tensor_banks.fp_bank.index_select(0, di)
                    ki = kg_idx_bank.index_select(0, di) if kg_idx_bank is not None else None

                    out = model(state=s, drug_fp=fp, drug_idx=ki,
                                use_prior=True, precomputed_kg_payload=kg_payload)
                    base_auc_list.append(out["auc_hat"].cpu().numpy())

                    s_p = perturb_state_at_gene(
                        s, gene_idx, pca_components, scaler_scale, imputer_stats
                    )
                    out_p = model(state=s_p, drug_fp=fp, drug_idx=ki,
                                  use_prior=True, precomputed_kg_payload=kg_payload)
                    pert_auc_list.append(out_p["auc_hat"].cpu().numpy())

            delta = np.concatenate(base_auc_list) - np.concatenate(pert_auc_list)

            # Assign family
            gene_family = "background"
            for fam, drugs in available_families.items():
                if drug_name in drugs:
                    gene_family = fam
                    break

            is_target = gene_name in KNOWN_DRUG_TARGETS.get(drug_name, [])
            pert_rows.append({
                "gene_name":       gene_name,
                "DRUG_NAME":       drug_name,
                "drug_family":     gene_family,
                "n_cells":         len(drug_frame),
                "mean_delta_auc":  float(delta.mean()),
                "abs_mean_delta":  float(np.abs(delta).mean()),
                "std_delta_auc":   float(delta.std()),
                "is_known_target": bool(is_target),
                "is_family_drug":  gene_family != "background",
            })

    pert_df = pd.DataFrame(pert_rows)
    pert_df.to_csv(OUT_DIR / "drug_family_perturbation.csv", index=False)
    print(f"\nSaved: {len(pert_df)} perturbation records")

    # ── Target specificity score ───────────────────────────────────────────
    spec_rows = []
    for gene_name, gdf in pert_df.groupby("gene_name"):
        for family, fam_drugs in available_families.items():
            fam_df  = gdf[gdf["drug_family"] == family]
            back_df = gdf[gdf["drug_family"] == "background"]
            if len(fam_df) < 1 or len(back_df) < 3:
                continue

            fam_mean  = fam_df["abs_mean_delta"].mean()
            back_mean = back_df["abs_mean_delta"].mean()
            spec_score = fam_mean / (back_mean + 1e-6)

            # Target genes for this family
            target_genes = set()
            for d in fam_drugs:
                target_genes.update(KNOWN_DRUG_TARGETS.get(d, []))
            is_target = gene_name in target_genes

            spec_rows.append({
                "gene_name":        gene_name,
                "drug_family":      family,
                "mean_delta_family": fam_mean,
                "mean_delta_background": back_mean,
                "specificity_score": spec_score,
                "is_family_target":  is_target,
                "n_family_drugs":    len(fam_df),
                "n_background_drugs": len(back_df),
            })

    spec_df = pd.DataFrame(spec_rows).sort_values("specificity_score", ascending=False)
    spec_df.to_csv(OUT_DIR / "drug_family_specificity.csv", index=False)

    # ── Rank heatmap: gene rank within drug family ─────────────────────────
    rank_rows = []
    for family, fam_drugs in available_families.items():
        fam_df = pert_df[pert_df["drug_family"] == family]
        if fam_df.empty:
            continue
        gene_family_mean = (
            fam_df.groupby("gene_name")["abs_mean_delta"]
            .mean().sort_values(ascending=False).reset_index()
        )
        gene_family_mean["rank_in_family"] = range(1, len(gene_family_mean) + 1)
        gene_family_mean["drug_family"] = family

        target_genes = set()
        for d in fam_drugs:
            target_genes.update(KNOWN_DRUG_TARGETS.get(d, []))
        gene_family_mean["is_family_target"] = gene_family_mean["gene_name"].isin(target_genes)
        rank_rows.append(gene_family_mean)

    if rank_rows:
        rank_df = pd.concat(rank_rows, ignore_index=True)
        rank_df.to_csv(OUT_DIR / "family_rank_heatmap.csv", index=False)

        # Report target gene ranks
        print("\n--- Target Gene Rank within Drug Family ---")
        for _, row in rank_df[rank_df["is_family_target"]].sort_values("rank_in_family").iterrows():
            print(f"  {row['drug_family']:20s} | {row['gene_name']:10s} "
                  f"rank={row['rank_in_family']:3d} | |ΔAU|={row['abs_mean_delta']:.5f}")

    # ── Within vs cross-family ─────────────────────────────────────────────
    cross_rows = []
    for gene_name in [g for g, _ in candidate_genes if g in gene_to_idx]:
        gdf = pert_df[pert_df["gene_name"] == gene_name]
        for family, fam_drugs in available_families.items():
            target_genes = set()
            for d in fam_drugs:
                target_genes.update(KNOWN_DRUG_TARGETS.get(d, []))
            if gene_name not in target_genes:
                continue

            family_df = gdf[gdf["drug_family"] == family]["abs_mean_delta"]
            cross_df  = gdf[gdf["drug_family"] != family]["abs_mean_delta"]
            if len(family_df) < 1 or len(cross_df) < 3:
                continue

            t_stat, p_val = stats.ttest_ind(family_df, cross_df, alternative="greater")
            cross_rows.append({
                "gene_name":       gene_name,
                "family":          family,
                "mean_within_family": family_df.mean(),
                "mean_cross_family":  cross_df.mean(),
                "fold_enrichment":    family_df.mean() / (cross_df.mean() + 1e-6),
                "t_stat":          t_stat,
                "p_value":         p_val,
            })

    cross_df_result = pd.DataFrame(cross_rows).sort_values("fold_enrichment", ascending=False)
    cross_df_result.to_csv(OUT_DIR / "within_vs_cross_family.csv", index=False)

    print("\n--- Within vs Cross-Family ΔAU (known targets) ---")
    for _, row in cross_df_result.iterrows():
        print(f"  {row['gene_name']:10s} → {row['family']:20s} | "
              f"within={row['mean_within_family']:.5f} vs cross={row['mean_cross_family']:.5f} | "
              f"fold={row['fold_enrichment']:.2f}× | p={row['p_value']:.3e}")

    print(f"\nAll outputs → {OUT_DIR}")
    print("Script 04 complete.")


if __name__ == "__main__":
    main()
