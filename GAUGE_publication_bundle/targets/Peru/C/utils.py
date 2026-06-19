"""
Shared utilities for Scenario C (Synthetic Lethality).

Wraps model loading, gene-space perturbation, and tensor management
for the double-disjoint world model.
"""
from __future__ import annotations

import pickle
import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.environ.get("KGPUB_PY_ROOT", "/mnt/raid5/xujing/KG"))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from GAUGE.benchmarking import BenchmarkConfig, load_benchmark_config
    from GAUGE.train import (
        PreparedData,
        TrainExecutionContext,
        _build_prediction_session,
        _cached_eval_tensors,
        _cached_model_drug_idx_bank,
        _ensure_prepared_compat,
        _maybe_precompute_kg_payload,
        _tensor_banks,
        _use_kg_prior_path,
        load_model,
        load_prepared,
        predict_frame,
    )
    from GAUGE.drug_level import select_best_fusion_weight, with_fused_auc
    from GAUGE.model import TerminalWorldModel
    from GAUGE.repro import RUNTIME_PROFILE_STABLE


def load_experiment(prepared_pkl: Path, result_dir: Path, config_yaml: Path, device: str = "cuda:0"):
    """Load double-disjoint world model, prepared data, and benchmark config."""
    print(f"  Loading prepared data from {prepared_pkl.name} …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        prepared = load_prepared(prepared_pkl)

    # Override with result-dir artifacts (correct KG / drug IDs for this run).
    # For the double-disjoint model: kg_graph.drug_ids=240, canonical_drug_table=240 → already aligned.
    # Do NOT override canonical_drug_table with drug_table (283) — that would break the alignment.
    artifacts_path = result_dir / "artifacts.pkl"
    if artifacts_path.exists():
        print(f"  Loading result-dir artifacts …")
        with open(artifacts_path, "rb") as f:
            result_artifacts = pickle.load(f)
        prepared = replace(prepared, artifacts=result_artifacts)

    print(f"  Loading model weights …")
    model = load_model(result_dir, prepared.artifacts)
    model = model.eval().to(device)

    config = load_benchmark_config(config_yaml)
    return model, prepared, config


def build_inference_session(model, prepared, config, device):
    """Precompute KG payload and return (session, context, tensor_banks, kg_bank, kg_payload)."""
    context = TrainExecutionContext()
    context.eval_tensor_cache = {}

    tensor_banks = _tensor_banks(prepared, device)
    use_kg = _use_kg_prior_path(model, config.controls)
    kg_bank = _cached_model_drug_idx_bank(prepared, model, device, context) if use_kg else None
    kg_payload = _maybe_precompute_kg_payload(
        model, device=device, controls=config.controls,
        edge_dropout=0.0, return_branch_states=False, context=context,
    )
    return context, tensor_banks, kg_bank, kg_payload


def compute_delta_z(
    gene_idx: int,
    pca_components: np.ndarray,   # (n_pca, n_genes)
    scaler_scale: np.ndarray,     # (n_genes,)
    imputer_stats: np.ndarray,    # (n_genes,) — mean used for imputation
    perturbation_value: float = 0.0,
) -> torch.Tensor:
    """
    Compute the PCA-space perturbation vector Δz when silencing gene g.

    Under mean-field approximation:
        x_g ≈ imputer_mean[g]  (population average)
        Δz_k = −W[k,g] × (imputer_mean[g] − perturbation_value) / σ_g

    Returns a (1, n_pca) float32 tensor (broadcasted over any batch dimension).
    """
    n_pca = pca_components.shape[0]
    # W[:, g] is the g-th column of the PCA component matrix
    delta_z_np = (
        -pca_components[:, gene_idx]
        * (imputer_stats[gene_idx] - perturbation_value)
        / (scaler_scale[gene_idx] + 1e-8)
    )
    return torch.tensor(delta_z_np, dtype=torch.float32).unsqueeze(0)  # (1, n_pca)


def perturb_forward(
    model: TerminalWorldModel,
    state_b: torch.Tensor,          # (B, state_dim)
    fp_b: torch.Tensor,             # (B, fp_dim)
    kg_idx: torch.Tensor | None,    # (B,)
    delta_z: torch.Tensor | None,   # (1, n_pca) or None
    kg_payload,
    device: str,
    state_pca_slice: slice = slice(0, 512),  # where PCA features sit in state
) -> torch.Tensor:
    """
    Forward pass with optional gene perturbation applied to the PCA slice of state.
    Returns auc_hat vector (B,).
    """
    if delta_z is not None:
        state_p = state_b.clone()
        state_p[:, state_pca_slice] = state_p[:, state_pca_slice] + delta_z.to(device)
    else:
        state_p = state_b

    with torch.no_grad():
        out = model(
            state=state_p,
            drug_fp=fp_b,
            drug_idx=kg_idx,
            use_prior=True,
            precomputed_kg_payload=kg_payload,
        )
    return out["auc_hat"].float()


def select_hvg_genes(
    genes: list[str],
    scaler_scale: np.ndarray,
    n: int = 2000,
) -> tuple[list[str], np.ndarray]:
    """
    Select the top-N most variable genes (by scaler standard deviation).

    Returns (hvg_gene_names, hvg_indices_in_full_gene_array).
    """
    top_idx = np.argsort(scaler_scale)[::-1][:n]
    hvg_names = [genes[i] for i in top_idx]
    return hvg_names, top_idx


def load_synleth_database(synleth_dir: Path) -> dict[str, set[tuple[str, str]]]:
    """
    Load SynLeth database.

    Returns dict with keys: 'SL', 'SDL', 'SDR', 'NONSL'.
    Each value is a set of (gene_A, gene_B) tuples (symmetric for SL).
    """
    db: dict[str, set] = {}
    files = {
        "SL":    synleth_dir / "gene_sl_gene.tsv",
        "SDL":   synleth_dir / "gene_sdl_gene.tsv",
        "SDR":   synleth_dir / "gene_sdr_gene.tsv",
        "NONSL": synleth_dir / "gene_nonsl_gene.tsv",
    }
    for rel, path in files.items():
        if not path.exists():
            db[rel] = set()
            continue
        df = pd.read_csv(path, sep="\t")
        pairs: set = set()
        for _, row in df.iterrows():
            g1 = str(row.get("x_name", ""))
            g2 = str(row.get("y_name", ""))
            if g1 and g2:
                pairs.add((g1, g2))
                if rel == "SL":  # SL is symmetric
                    pairs.add((g2, g1))
        db[rel] = pairs
        print(f"  SynLeth [{rel}]: {len(pairs):,} pairs loaded")
    return db


def load_cell_metadata(gdsc_model_list: Path) -> pd.DataFrame:
    """Load GDSC cell-line metadata with tissue / cancer-type annotations."""
    df = pd.read_csv(gdsc_model_list, low_memory=False)
    return df[["model_id", "model_name", "tissue", "cancer_type", "cancer_type_detail"]].rename(
        columns={"model_id": "SANGER_MODEL_ID"}
    )
