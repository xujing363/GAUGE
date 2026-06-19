"""
Script 03: Cancer-Type-Specific Perturbation Analysis
=======================================================
Hypothesis: The world model captures cancer-type-specific gene-drug relationships.

For each cancer type in the test set:
  - Which gene knockouts most sensitize cells to clinically relevant drugs?
  - Do NSCLC cells show EGFR-dependent sensitivity to EGFR inhibitors?
  - Do melanoma cells show BRAF-dependent sensitivity to BRAF inhibitors?
  - Do AML/lymphoma cells show BCL2-dependent sensitivity to Venetoclax?

The 190 test cell lines are UNSEEN during training. The model's ability to
correctly predict cancer-type-specific gene-drug relationships demonstrates
that it has learned generalizable biological rules, not memorized cell identities.

Outputs (saved to results/03_cancer_type/):
  cancer_gene_drug_perturbation.csv  - per-cancer-type perturbation matrix
  cancer_drug_top_genes.csv          - top sensitivity genes per cancer + drug
  cancer_type_heatmap_data.csv       - data for heatmap visualization
  cancer_type_summary.csv            - cancer type overview statistics

Usage:
    python scripts/03_cancer_type_perturbation.py
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
    FOCUS_CANCER_TYPES,
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
    load_cell_metadata,
    load_experiment,
    perturb_state_at_gene,
)

OUT_DIR = RESULTS_DIR / "03_cancer_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cancer type → clinically relevant drug family mapping
CANCER_DRUG_FOCUS = {
    "NSCLC":     {
        "drugs": ["Erlotinib", "Gefitinib", "Lapatinib"],
        "genes": ["EGFR", "ERBB2", "KRAS", "BRAF", "MET", "ALK"],
    },
    "Melanoma":  {
        "drugs": ["Vemurafenib", "Dabrafenib", "Trametinib", "Selumetinib"],
        "genes": ["BRAF", "NRAS", "MAP2K1", "MAP2K2", "CDKN2A"],
    },
    "AML": {
        "drugs": ["Venetoclax", "Cytarabine", "Dasatinib"],
        "genes": ["BCL2", "BCL2L1", "MCL1", "FLT3", "KIT", "TP53"],
    },
    "CLL_DLBCL": {
        "drugs": ["Venetoclax", "Navitoclax"],
        "genes": ["BCL2", "BCL2L1", "MCL1", "BAX", "BAD"],
    },
    "Breast": {
        "drugs": ["Lapatinib", "Alpelisib", "Palbociclib", "Abemaciclib"],
        "genes": ["ERBB2", "EGFR", "PIK3CA", "PTEN", "CDK4", "CDK6", "CCND1", "RB1"],
    },
    "Colorectal": {
        "drugs": ["Erlotinib", "Gefitinib", "Trametinib"],
        "genes": ["KRAS", "BRAF", "EGFR", "MAP2K1", "PIK3CA"],
    },
    "Pancreatic": {
        "drugs": ["Erlotinib", "Olaparib"],
        "genes": ["KRAS", "TP53", "SMAD4", "BRCA1", "BRCA2", "PARP1"],
    },
    "Ovarian": {
        "drugs": ["Olaparib", "Niraparib", "Rucaparib"],
        "genes": ["BRCA1", "BRCA2", "PARP1", "PARP2", "TP53"],
    },
}


def get_cancer_type_cells(cell_meta: pd.DataFrame, cancer_label: str) -> list[str]:
    """Return SANGER_MODEL_IDs matching a cancer type label (flexible matching)."""
    focus_patterns = FOCUS_CANCER_TYPES.get(cancer_label, [cancer_label])
    mask = cell_meta["cancer_type"].str.contains("|".join(focus_patterns), case=False, na=False)
    return cell_meta.loc[mask, "SANGER_MODEL_ID"].tolist()


def main():
    print("=" * 70)
    print("Script 03: Cancer-Type-Specific Perturbation Analysis")
    print("=" * 70)

    model, prepared, config = load_experiment(PREPARED_PKL, RESULT_DIR, CONFIG_YAML, DEVICE)

    genes = prepared.artifacts.genes
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    test_frame = prepared.responses[prepared.responses["split"] == "test"].copy()
    test_frame = test_frame.reset_index(drop=True)
    test_cells = test_frame["SANGER_MODEL_ID"].unique().tolist()
    print(f"  Test rows: {len(test_frame)}, Test cells: {len(test_cells)}")

    infra = build_inference_context(model, prepared, config, test_frame, DEVICE)
    tensors      = infra["tensors"]
    tensor_banks = infra["tensor_banks"]
    kg_idx_bank  = infra["kg_drug_idx_bank"]
    kg_payload   = infra["precomputed_kg_payload"]

    pca_components = prepared.artifacts.pca.components_
    scaler_scale   = prepared.artifacts.scaler.scale_
    imputer_stats  = prepared.artifacts.imputer.statistics_

    cell_meta = load_cell_metadata(GDSC_MODEL_LIST, test_cells)
    print(f"  Metadata for {len(cell_meta)} test cells")

    # Show cancer type distribution in test set
    cancer_dist = cell_meta["cancer_type"].value_counts().head(20)
    print("\n  Cancer type distribution (test set):")
    for ct, n in cancer_dist.items():
        print(f"    {ct:40s} n={n}")

    all_rows = []
    cancer_summary_rows = []

    for cancer_label, focus in CANCER_DRUG_FOCUS.items():
        cancer_cells = get_cancer_type_cells(cell_meta, cancer_label)
        cancer_cells_in_test = [c for c in cancer_cells if c in set(test_cells)]
        if len(cancer_cells_in_test) < 3:
            print(f"\n  SKIP {cancer_label}: only {len(cancer_cells_in_test)} test cells")
            continue

        print(f"\n--- {cancer_label}: {len(cancer_cells_in_test)} test cells ---")

        # Restrict test_frame to this cancer type
        cancer_frame = test_frame[test_frame["SANGER_MODEL_ID"].isin(cancer_cells_in_test)].copy()

        # Filter to focus drugs + genes
        focus_drugs = [d for d in focus["drugs"] if d in cancer_frame["DRUG_NAME"].values]
        focus_genes = [(g, gene_to_idx[g]) for g in focus["genes"] if g in gene_to_idx]
        print(f"  Focus drugs: {focus_drugs}")
        print(f"  Focus genes: {[g for g, _ in focus_genes]}")

        for drug_name in focus_drugs:
            drug_frame = cancer_frame[cancer_frame["DRUG_NAME"] == drug_name]
            if len(drug_frame) < 3:
                continue
            local_idx = drug_frame.index.to_numpy()

            model.eval()
            with torch.no_grad():
                # Baseline
                base_auc = []
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
                    base_auc.append(out["auc_hat"].cpu().numpy())
                base_auc = np.concatenate(base_auc)

                for gene_name, gene_idx in focus_genes:
                    pert_auc = []
                    for start in range(0, len(local_idx), BATCH_SIZE):
                        end = min(start + BATCH_SIZE, len(local_idx))
                        bidx = local_idx[start:end]
                        si = tensors.state_idx[bidx]
                        di = tensors.drug_idx[bidx]
                        s  = tensor_banks.state_bank.index_select(0, si)
                        fp = tensor_banks.fp_bank.index_select(0, di)
                        ki = kg_idx_bank.index_select(0, di) if kg_idx_bank is not None else None
                        s_p = perturb_state_at_gene(
                            s, gene_idx, pca_components, scaler_scale, imputer_stats
                        )
                        out_p = model(state=s_p, drug_fp=fp, drug_idx=ki,
                                      use_prior=True, precomputed_kg_payload=kg_payload)
                        pert_auc.append(out_p["auc_hat"].cpu().numpy())
                    pert_auc = np.concatenate(pert_auc)
                    delta = base_auc - pert_auc

                    is_target = gene_name in KNOWN_DRUG_TARGETS.get(drug_name, [])
                    all_rows.append({
                        "cancer_type":       cancer_label,
                        "DRUG_NAME":         drug_name,
                        "gene_name":         gene_name,
                        "n_cells":           len(drug_frame),
                        "mean_delta_auc":    float(delta.mean()),
                        "std_delta_auc":     float(delta.std()),
                        "abs_mean_delta":    float(np.abs(delta).mean()),
                        "frac_positive":     float((delta > 0).mean()),
                        "is_known_target":   bool(is_target),
                    })
                    if is_target:
                        print(f"  * {drug_name:20s} ← {gene_name:10s} "
                              f"[KNOWN TARGET] ΔAU={delta.mean():+.4f} n={len(drug_frame)}")

        # Cancer type summary
        n_known_targets = sum(
            1 for r in all_rows
            if r["cancer_type"] == cancer_label and r["is_known_target"]
        )
        cancer_summary_rows.append({
            "cancer_type":    cancer_label,
            "n_test_cells":   len(cancer_cells_in_test),
            "n_focus_drugs":  len(focus_drugs),
            "n_focus_genes":  len(focus_genes),
            "n_known_target_pairs": n_known_targets,
        })

    result_df  = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(cancer_summary_rows)

    if not result_df.empty:
        result_df.to_csv(OUT_DIR / "cancer_gene_drug_perturbation.csv", index=False)

        # Top genes per cancer type + drug
        top_rows = []
        for (ct, dn), grp in result_df.groupby(["cancer_type", "DRUG_NAME"]):
            for rank, (_, row) in enumerate(
                grp.sort_values("abs_mean_delta", ascending=False).head(5).iterrows(), 1
            ):
                top_rows.append({**row.to_dict(), "rank_within_drug": rank})
        top_df = pd.DataFrame(top_rows)
        top_df.to_csv(OUT_DIR / "cancer_drug_top_genes.csv", index=False)

        # Heatmap data: cancer type × gene → mean ΔAU (across drugs)
        heatmap_df = (
            result_df.groupby(["cancer_type", "gene_name"])["mean_delta_auc"]
            .mean().reset_index()
        )
        heatmap_df.to_csv(OUT_DIR / "cancer_type_heatmap_data.csv", index=False)

        # Validation: known target rank
        print("\n--- Known Target Rank within Cancer Type ---")
        known = result_df[result_df["is_known_target"]].copy()
        for _, row in known.sort_values("abs_mean_delta", ascending=False).iterrows():
            print(f"  {row['cancer_type']:12s} | {row['DRUG_NAME']:20s} ← {row['gene_name']:10s} "
                  f"| ΔAU={row['mean_delta_auc']:+.4f} | n={row['n_cells']}")

    summary_df.to_csv(OUT_DIR / "cancer_type_summary.csv", index=False)
    print(f"\nAll outputs → {OUT_DIR}")
    print("Script 03 complete.")


if __name__ == "__main__":
    main()
