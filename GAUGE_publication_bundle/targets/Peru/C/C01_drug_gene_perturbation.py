"""
Scenario C — Step 1: Drug-centric Single-Gene Perturbation
===========================================================

For each unseen test drug (56 drugs, double-disjoint split), silence each of
the top-2000 HVG genes in silico and measure the change in predicted AUC:

    Δ(drug, gene) = AUC_baseline(drug) − AUC_silenced(drug, gene)

    Δ > 0 → gene silencing sensitises cells to the drug
               (gene supports resistance; its loss = drug synergy)
    Δ < 0 → gene silencing desensitises cells

World-model interpretation
--------------------------
The model propagates: b = T(z_s, z_a, z_s ⊙ z_a)
  • z_s  = cell transcriptomic state (PCA-encoded, 512 dims)
  • z_a  = drug action embedding (ChEMBL/DRKG/PrimeKG KG-informed)
  • z_s ⊙ z_a = drug–cell interaction (captured by three prior networks)

Silencing gene g shifts z_s by Δz (analytical PCA-space update), and the
terminal consequence simulator re-evaluates the drug-action response.

Outputs (results/C01_drug_gene/)
---------------------------------
  drug_gene_delta_auc.csv  — full (drug, gene) × cell matrix summary
  top_sensitising_genes.csv — top 20 sensitising genes per drug
  known_target_validation.csv — validation: known targets should rank highly
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
    KNOWN_DRUG_TARGETS,
    MIN_SAMPLES_PER_DRUG,
    N_HVG_PERTURB,
    RESULTS_DIR,
)
from utils import (
    build_inference_session,
    compute_delta_z,
    load_experiment,
    perturb_forward,
    select_hvg_genes,
)

# Import GAUGE internal at module level (requires KG path in sys.path)
import warnings as _w
with _w.catch_warnings():
    _w.filterwarnings("ignore")
    from GAUGE.train import _cached_eval_tensors

OUT_DIR = RESULTS_DIR / "C01_drug_gene"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PCA slice: first 512 dims of the 515-dim state vector are PCA features
PCA_SLICE = slice(0, 512)


def run_baseline_for_drug(
    drug_rows: pd.DataFrame,
    tensors,
    tensor_banks,
    kg_bank,
    kg_payload,
    model,
    device: str,
    local_batch: int = 2048,
) -> tuple[np.ndarray, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
    """
    Compute baseline AUC and cache (state_b, fp_b, kg_idx) tensors for this drug's cells.
    Returns (baseline_aucs, state_list, fp_list, kg_idx_list).
    """
    n = len(drug_rows)
    row_idx = torch.tensor(drug_rows.index.tolist(), dtype=torch.long, device=device)
    baseline_aucs, state_list, fp_list, kg_list = [], [], [], []

    for start in range(0, n, local_batch):
        end = min(start + local_batch, n)
        idx = row_idx[start:end]
        state_idx  = tensors.state_idx[idx]
        drug_idx_t = tensors.drug_idx[idx]
        state_b = tensor_banks.state_bank.index_select(0, state_idx)
        fp_b    = tensor_banks.fp_bank.index_select(0, drug_idx_t)
        kg_idx  = kg_bank.index_select(0, drug_idx_t) if kg_bank is not None else None

        auc_base = perturb_forward(model, state_b, fp_b, kg_idx, None, kg_payload, device, PCA_SLICE)
        baseline_aucs.append(auc_base.cpu().numpy())
        state_list.append(state_b)
        fp_list.append(fp_b)
        kg_list.append(kg_idx)

    return np.concatenate(baseline_aucs), state_list, fp_list, kg_list


def run_perturbed_for_gene(
    delta_z: torch.Tensor,
    state_list: list[torch.Tensor],
    fp_list: list[torch.Tensor],
    kg_list: list[torch.Tensor | None],
    kg_payload,
    model,
    device: str,
) -> np.ndarray:
    """Run perturbed forward passes using cached (state, fp, kg) tensors."""
    perturbed = []
    for state_b, fp_b, kg_idx in zip(state_list, fp_list, kg_list):
        auc_pert = perturb_forward(model, state_b, fp_b, kg_idx, delta_z, kg_payload, device, PCA_SLICE)
        perturbed.append(auc_pert.cpu().numpy())
    return np.concatenate(perturbed)


def main() -> None:
    print("=" * 70)
    print("Scenario C — Step 1: Drug-Gene Perturbation (World Model)")
    print("Model: double-disjoint split (both cell & drug unseen at training)")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────────
    model, prepared, config = load_experiment(
        DOUBLE_DISJOINT_PREPARED_PKL,
        DOUBLE_DISJOINT_RESULT_DIR,
        DOUBLE_DISJOINT_CONFIG_YAML,
        device=DEVICE,
    )

    genes        = prepared.artifacts.genes            # list[str], len=41144
    scaler_scale = prepared.artifacts.scaler.scale_    # (41144,)
    imputer_stats = prepared.artifacts.imputer.statistics_  # (41144,)
    pca_comps    = prepared.artifacts.pca.components_  # (512, 41144)

    # Select HVG genes for perturbation
    hvg_names, hvg_idx = select_hvg_genes(list(genes), scaler_scale, n=N_HVG_PERTURB)
    gene_to_global_idx = {g: i for i, g in enumerate(genes)}
    print(f"  HVG genes selected: {len(hvg_names)}")
    print(f"  Sample HVGs: {hvg_names[:5]}")

    # ── Build inference session ───────────────────────────────────────────────
    context, tensor_banks, kg_bank, kg_payload = build_inference_session(
        model, prepared, config, DEVICE
    )

    test_frame = prepared.responses[prepared.responses["split"] == "test"].copy().reset_index(drop=True)
    print(f"  Test samples: {len(test_frame)}")
    print(f"  Test drugs:   {test_frame['DRUG_ID'].nunique()}")

    tensors = _cached_eval_tensors(test_frame, context.eval_tensor_cache, tensor_banks, DEVICE)

    # ── Pre-compute Δz for each HVG gene (analytical, fast) ──────────────────
    print(f"\nPre-computing Δz for {len(hvg_names)} HVG genes …")
    delta_z_cache: dict[int, torch.Tensor] = {}
    for g_idx in hvg_idx:
        delta_z_cache[int(g_idx)] = compute_delta_z(
            int(g_idx), pca_comps, scaler_scale, imputer_stats
        )

    # ── Run perturbations ─────────────────────────────────────────────────────
    unique_drugs = test_frame[["DRUG_ID", "DRUG_NAME"]].drop_duplicates()
    print(f"\nRunning perturbations: {len(unique_drugs)} drugs × {len(hvg_names)} genes …")

    rows = []
    for _, drug_row in unique_drugs.iterrows():
        drug_id   = int(drug_row["DRUG_ID"])
        drug_name = str(drug_row.get("DRUG_NAME", drug_id))
        known_targets = KNOWN_DRUG_TARGETS.get(drug_name, [])

        drug_mask  = test_frame["DRUG_ID"] == drug_id
        drug_df    = test_frame[drug_mask]
        if len(drug_df) < MIN_SAMPLES_PER_DRUG:
            continue

        print(f"  Drug: {drug_name:30s} ({len(drug_df):3d} cells) …", flush=True)

        # Compute baseline once per drug, cache tensors for re-use across genes
        base_auc, state_list, fp_list, kg_list = run_baseline_for_drug(
            drug_df, tensors, tensor_banks, kg_bank, kg_payload, model, DEVICE
        )

        for g_idx in hvg_idx:
            g_name   = genes[g_idx]
            delta_z  = delta_z_cache[int(g_idx)]

            pert_auc = run_perturbed_for_gene(
                delta_z, state_list, fp_list, kg_list, kg_payload, model, DEVICE
            )
            delta = base_auc - pert_auc   # positive → sensitising

            rows.append({
                "DRUG_ID":          drug_id,
                "DRUG_NAME":        drug_name,
                "gene_name":        g_name,
                "n_cells":          len(drug_df),
                "mean_delta_auc":   float(delta.mean()),
                "std_delta_auc":    float(delta.std()),
                "abs_mean_delta_auc": float(np.abs(delta).mean()),
                "frac_sensitising": float((delta > DELTA_AUC_THRESHOLD).mean()),
                "is_known_target":  g_name in known_targets,
            })

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "drug_gene_delta_auc.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df):,} rows)")

    # ── Top sensitising genes per drug ────────────────────────────────────────
    top_rows = []
    for (drug_id, drug_name), grp in df.groupby(["DRUG_ID", "DRUG_NAME"]):
        sorted_grp = grp.sort_values("mean_delta_auc", ascending=False)
        for rank, (_, r) in enumerate(sorted_grp.head(20).iterrows(), 1):
            top_rows.append({**r.to_dict(), "rank_sensitising": rank})
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(OUT_DIR / "top_sensitising_genes.csv", index=False)

    # ── Known target validation ───────────────────────────────────────────────
    known_df = df[df["is_known_target"]].copy()
    if len(known_df) > 0:
        known_df = known_df.sort_values("mean_delta_auc", ascending=False)
        known_df.to_csv(OUT_DIR / "known_target_validation.csv", index=False)
        print("\n--- Known target validation ---")
        for _, r in known_df.iterrows():
            print(f"  {r.DRUG_NAME:25s} ← {r.gene_name:12s} "
                  f"Δ={r.mean_delta_auc:+.4f}  frac_sens={r.frac_sensitising:.2f}")

    # ── Per-drug summary stats ─────────────────────────────────────────────────
    summary = df.groupby(["DRUG_ID", "DRUG_NAME"]).agg(
        n_genes_tested   = ("gene_name", "nunique"),
        n_sensitising    = ("mean_delta_auc", lambda x: (x > DELTA_AUC_THRESHOLD).sum()),
        max_delta_auc    = ("mean_delta_auc", "max"),
        top_gene         = ("gene_name", lambda x: x.iloc[df.loc[x.index, "mean_delta_auc"].argmax()]),
    ).reset_index()
    summary.to_csv(OUT_DIR / "drug_sensitivity_summary.csv", index=False)

    print(f"\nDone. Results in {OUT_DIR}")
    print(f"  Total (drug, gene) pairs: {len(df):,}")
    print(f"  Sensitising pairs (Δ>{DELTA_AUC_THRESHOLD}): {(df.mean_delta_auc > DELTA_AUC_THRESHOLD).sum():,}")


if __name__ == "__main__":
    main()
