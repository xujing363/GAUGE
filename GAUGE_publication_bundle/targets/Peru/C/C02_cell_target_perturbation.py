"""
Scenario C — Step 2: Cell-centric Drug-Target Perturbation
===========================================================

For each unseen test cell line (190 cells, double-disjoint split), silence each
known drug target in the three KG priors and measure:

    Δ(cell, target) = mean_drug AUC_baseline(cell) − mean_drug AUC_silenced(cell, target)

    Δ > 0 → target silencing broadly sensitises this cell to drugs
    Δ < 0 → target silencing desensitises

This generates a "sensitive target profile" per cell line, reflecting which
nodes in the three prior networks (ChEMBL/DRKG/PrimeKG) actively gate each
cell's drug response.

World-model interpretation
--------------------------
The terminal consequence simulator T(z_s, z_a, z_s ⊙ z_a) captures how a
cellular state z_s interacts with each drug action z_a. Silencing a target
shifts z_s away from its drug-accessible state, changing the "terminal
consequence" of treatment.

Outputs (results/C02_cell_target/)
-----------------------------------
  cell_target_delta_auc.csv   — (cell, target) × drug-mean Δ AUC
  cell_sensitive_targets.csv  — per-cell top 10 sensitive targets
  cancer_type_target_heatmap.csv — aggregated by cancer type
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, os.environ.get("KGPUB_PY_ROOT", "/mnt/raid5/xujing/KG"))

from config import (
    BATCH_SIZE,
    DELTA_AUC_THRESHOLD,
    DEVICE,
    DOUBLE_DISJOINT_CONFIG_YAML,
    DOUBLE_DISJOINT_PREPARED_PKL,
    DOUBLE_DISJOINT_RESULT_DIR,
    GDSC_MODEL_LIST,
    KNOWN_DRUG_TARGETS,
    N_HVG_PERTURB,
    RESULTS_DIR,
    SENSITISING_PERCENTILE,
)
from utils import (
    build_inference_session,
    compute_delta_z,
    load_cell_metadata,
    load_experiment,
    perturb_forward,
    select_hvg_genes,
)

import warnings as _w
with _w.catch_warnings():
    _w.filterwarnings("ignore")
    from GAUGE.train import _cached_eval_tensors

OUT_DIR = RESULTS_DIR / "C02_cell_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_SLICE = slice(0, 512)


def main() -> None:
    print("=" * 70)
    print("Scenario C — Step 2: Cell-Target Perturbation (World Model)")
    print("Model: double-disjoint split (both cell & drug unseen at training)")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────────
    model, prepared, config = load_experiment(
        DOUBLE_DISJOINT_PREPARED_PKL,
        DOUBLE_DISJOINT_RESULT_DIR,
        DOUBLE_DISJOINT_CONFIG_YAML,
        device=DEVICE,
    )

    genes         = prepared.artifacts.genes
    scaler_scale  = prepared.artifacts.scaler.scale_
    imputer_stats = prepared.artifacts.imputer.statistics_
    pca_comps     = prepared.artifacts.pca.components_

    hvg_names, hvg_idx = select_hvg_genes(list(genes), scaler_scale, n=N_HVG_PERTURB)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Drug target genes to perturb (union of all known targets + HVG genes)
    drug_target_genes: list[str] = []
    for targets in KNOWN_DRUG_TARGETS.values():
        for t in targets:
            if t in gene_to_idx and t not in drug_target_genes and t != "DNA":
                drug_target_genes.append(t)
    # Also include top HVG genes to have broader coverage
    for g in hvg_names[:500]:
        if g not in drug_target_genes:
            drug_target_genes.append(g)

    print(f"  Target genes to perturb: {len(drug_target_genes)}")
    print(f"  Known drug targets: {len([t for ts in KNOWN_DRUG_TARGETS.values() for t in ts if t in gene_to_idx])}")

    # ── Build inference session ───────────────────────────────────────────────
    context, tensor_banks, kg_bank, kg_payload = build_inference_session(
        model, prepared, config, DEVICE
    )

    test_frame = prepared.responses[prepared.responses["split"] == "test"].copy().reset_index(drop=True)
    tensors = _cached_eval_tensors(test_frame, context.eval_tensor_cache, tensor_banks, DEVICE)

    unique_cells = test_frame["SANGER_MODEL_ID"].unique()
    unique_drugs = test_frame[["DRUG_ID", "DRUG_NAME"]].drop_duplicates()
    print(f"  Test cells: {len(unique_cells)}, Test drugs: {len(unique_drugs)}")

    # ── Pre-compute Δz for each target gene ──────────────────────────────────
    delta_z_cache: dict[str, torch.Tensor] = {}
    valid_targets: list[str] = []
    for g_name in drug_target_genes:
        if g_name not in gene_to_idx:
            continue
        g_idx = gene_to_idx[g_name]
        delta_z_cache[g_name] = compute_delta_z(
            g_idx, pca_comps, scaler_scale, imputer_stats
        )
        valid_targets.append(g_name)
    print(f"  Valid target genes found in model: {len(valid_targets)}")

    # ── Identify which drugs each gene is a known target for ─────────────────
    gene_to_drugs: dict[str, list[str]] = {}
    for drug_name, targets in KNOWN_DRUG_TARGETS.items():
        for t in targets:
            gene_to_drugs.setdefault(t, []).append(drug_name)

    # ── Run perturbations: for each cell, perturb each target gene ───────────
    print(f"\nRunning cell-target perturbations …")
    rows = []
    batch_size_local = 512

    for cell_id in unique_cells:
        cell_mask = test_frame["SANGER_MODEL_ID"] == cell_id
        cell_df   = test_frame[cell_mask]
        if len(cell_df) == 0:
            continue

        # Evaluate baseline and perturbed AUC across all drugs for this cell
        cell_row_idx = torch.tensor(cell_df.index.tolist(), dtype=torch.long, device=DEVICE)

        # Get cell state (it's the same for all drugs this cell is tested with)
        # We need state per (cell, drug) pair
        for target_gene in valid_targets:
            delta_z = delta_z_cache[target_gene]

            base_aucs, pert_aucs = [], []
            for start in range(0, len(cell_df), batch_size_local):
                end = min(start + batch_size_local, len(cell_df))
                idx = cell_row_idx[start:end]

                state_idx  = tensors.state_idx[idx]
                drug_idx_t = tensors.drug_idx[idx]
                state_b    = tensor_banks.state_bank.index_select(0, state_idx)
                fp_b       = tensor_banks.fp_bank.index_select(0, drug_idx_t)
                kg_idx     = kg_bank.index_select(0, drug_idx_t) if kg_bank is not None else None

                base_aucs.append(perturb_forward(model, state_b, fp_b, kg_idx, None,    kg_payload, DEVICE, PCA_SLICE).cpu().numpy())
                pert_aucs.append(perturb_forward(model, state_b, fp_b, kg_idx, delta_z, kg_payload, DEVICE, PCA_SLICE).cpu().numpy())

            base_auc_arr = np.concatenate(base_aucs)
            pert_auc_arr = np.concatenate(pert_aucs)
            delta        = base_auc_arr - pert_auc_arr

            is_known = target_gene in gene_to_drugs
            rows.append({
                "SANGER_MODEL_ID":    cell_id,
                "target_gene":        target_gene,
                "n_drugs":            len(cell_df),
                "mean_delta_auc":     float(delta.mean()),
                "std_delta_auc":      float(delta.std()),
                "abs_mean_delta":     float(np.abs(delta).mean()),
                "frac_sensitising":   float((delta > DELTA_AUC_THRESHOLD).mean()),
                "is_known_drug_target": is_known,
                "targeted_by_drugs":  ",".join(gene_to_drugs.get(target_gene, [])),
            })

        if len(rows) % (len(valid_targets) * 10) == 0:
            n_done = len(rows) // len(valid_targets)
            print(f"  Progress: {n_done}/{len(unique_cells)} cells …", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "cell_target_delta_auc.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'cell_target_delta_auc.csv'} ({len(df):,} rows)")

    # ── Per-cell percentile rank of each target gene ──────────────────────────
    # Compute within-cell percentile rank (top 5% = "sensitive target")
    df["within_cell_percentile"] = df.groupby("SANGER_MODEL_ID")["mean_delta_auc"].rank(pct=True) * 100
    df["is_top5pct"] = df["within_cell_percentile"] >= (100 - SENSITISING_PERCENTILE)

    # ── Per-cell top sensitive targets ────────────────────────────────────────
    top_rows = []
    for cell_id, grp in df.groupby("SANGER_MODEL_ID"):
        sorted_grp = grp.sort_values("mean_delta_auc", ascending=False)
        for rank, (_, r) in enumerate(sorted_grp.head(10).iterrows(), 1):
            top_rows.append({**r.to_dict(), "rank_in_cell": rank})
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(OUT_DIR / "cell_sensitive_targets.csv", index=False)

    # ── Merge cancer-type metadata and aggregate by cancer type ──────────────
    cell_meta = load_cell_metadata(GDSC_MODEL_LIST)
    df_meta   = df.merge(cell_meta, on="SANGER_MODEL_ID", how="left")

    cancer_agg = (
        df_meta.groupby(["cancer_type", "target_gene"])
        .agg(
            n_cells           = ("SANGER_MODEL_ID", "nunique"),
            mean_delta_auc    = ("mean_delta_auc", "mean"),
            frac_sensitising  = ("frac_sensitising", "mean"),
        )
        .reset_index()
        .sort_values(["cancer_type", "mean_delta_auc"], ascending=[True, False])
    )
    cancer_agg.to_csv(OUT_DIR / "cancer_type_target_heatmap.csv", index=False)

    print(f"\nTop sensitive targets across all cells:")
    gene_agg = df.groupby("target_gene").agg(
        mean_delta=("mean_delta_auc", "mean"),
        frac_sensitising=("frac_sensitising", "mean"),
    ).sort_values("mean_delta", ascending=False)
    for g, r in gene_agg.head(15).iterrows():
        print(f"  {g:15s} | Δ={r.mean_delta:+.5f} | frac_sens={r.frac_sensitising:.2f}")

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
