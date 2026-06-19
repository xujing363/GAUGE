"""
Utilities for Scenario A: in-silico transcriptome perturbation.
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
    from GAUGE.benchmarking import AblationControls, BenchmarkConfig, load_benchmark_config
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
    from GAUGE.model import TerminalWorldModel


def load_experiment(prepared_pkl: Path, result_dir: Path, config_yaml: Path, device: str = "cuda:0"):
    """Load model, prepared data, and config for the true cell-split model."""
    print(f"Loading prepared data from {prepared_pkl} ...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        prepared = load_prepared(prepared_pkl)

    artifacts_path = result_dir / "artifacts.pkl"
    if artifacts_path.exists():
        print(f"Loading result-dir artifacts from {artifacts_path} ...")
        with open(artifacts_path, "rb") as f:
            result_artifacts = pickle.load(f)
        result_artifacts = replace(result_artifacts, canonical_drug_table=result_artifacts.drug_table)
        prepared = replace(prepared, artifacts=result_artifacts)

    print(f"Loading model from {result_dir} ...")
    model = load_model(result_dir, prepared.artifacts)
    model = model.eval().to(device)

    print(f"Loading config from {config_yaml} ...")
    config = load_benchmark_config(config_yaml)

    return model, prepared, config


def build_inference_context(model, prepared, config, test_frame, device):
    """Build all inference infrastructure needed for perturbation."""
    context = TrainExecutionContext()
    context.eval_tensor_cache = {}

    tensor_banks_obj = _tensor_banks(prepared, device)
    use_kg_prior = _use_kg_prior_path(model, config.controls)
    kg_drug_idx_bank = (
        _cached_model_drug_idx_bank(prepared, model, device, context) if use_kg_prior else None
    )
    precomputed_kg_payload = _maybe_precompute_kg_payload(
        model, device=device, controls=config.controls,
        edge_dropout=0.0, return_branch_states=False, context=context,
    )
    tensors = _cached_eval_tensors(test_frame, context.eval_tensor_cache, tensor_banks_obj, device)

    return {
        "tensor_banks": tensor_banks_obj,
        "kg_drug_idx_bank": kg_drug_idx_bank,
        "precomputed_kg_payload": precomputed_kg_payload,
        "tensors": tensors,
    }


def perturb_state_at_gene(
    state_bank_orig: torch.Tensor,
    gene_idx: int,
    pca_components: np.ndarray,
    scaler_scale: np.ndarray,
    imputer_stats: np.ndarray,
    perturbation_value: float = 0.0,
) -> torch.Tensor:
    """
    Compute perturbed state bank after setting gene g to perturbation_value.

    Linear projection: Δz_k = (Δx_g / σ_g) * W[k,g]
    where Δx_g = perturbation_value - mean_g (mean from imputer)
    """
    std_g = float(scaler_scale[gene_idx])
    if std_g < 1e-10:
        return state_bank_orig.clone()

    mean_g = float(imputer_stats[gene_idx])
    delta_x_std = (perturbation_value - mean_g) / std_g

    n_pca = pca_components.shape[0]
    gene_loadings = torch.tensor(
        pca_components[:, gene_idx],
        dtype=torch.float32,
        device=state_bank_orig.device,
    )

    delta_pca = delta_x_std * gene_loadings  # (n_pca,)

    state_perturbed = state_bank_orig.clone()
    state_perturbed[:, :n_pca] = state_perturbed[:, :n_pca] + delta_pca.unsqueeze(0)
    return state_perturbed


def run_batch_perturbation(
    model: TerminalWorldModel,
    row_idx: np.ndarray,
    infra: dict,
    pca_components: np.ndarray,
    scaler_scale: np.ndarray,
    imputer_stats: np.ndarray,
    gene_indices: list[int],
    device: str,
    batch_size: int = 2048,
    extract_gate: bool = False,
) -> dict:
    """
    Run baseline + multiple gene perturbations for a set of rows.

    Returns:
        baseline_auc: (n_rows,)
        perturbed_auc: dict[gene_idx -> (n_rows,)]
        gate_baseline: (n_rows,) if extract_gate else None
        gate_perturbed: dict[gene_idx -> (n_rows,)] if extract_gate else None
    """
    tensors = infra["tensors"]
    tensor_banks = infra["tensor_banks"]
    kg_drug_idx_bank = infra["kg_drug_idx_bank"]
    precomputed_kg_payload = infra["precomputed_kg_payload"]

    n = len(row_idx)
    baseline_auc = []
    gate_baseline = [] if extract_gate else None

    pert_auc = {gi: [] for gi in gene_indices}
    pert_gate = {gi: [] for gi in gene_indices} if extract_gate else None

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            local_idx = row_idx[start:end]

            state_idx = tensors.state_idx[local_idx]
            drug_idx_t = tensors.drug_idx[local_idx]

            state_b = tensor_banks.state_bank.index_select(0, state_idx)
            fp_b = tensor_banks.fp_bank.index_select(0, drug_idx_t)
            kg_idx = kg_drug_idx_bank.index_select(0, drug_idx_t) if kg_drug_idx_bank is not None else None

            out_base = model(
                state=state_b, drug_fp=fp_b, drug_idx=kg_idx,
                use_prior=True, precomputed_kg_payload=precomputed_kg_payload,
            )
            baseline_auc.append(out_base["auc_hat"].cpu().numpy())
            if extract_gate and "kg_gate" in out_base:
                gate_baseline.append(out_base["kg_gate"].cpu().numpy())

            for gi in gene_indices:
                state_pert = perturb_state_at_gene(
                    state_b, gi, pca_components, scaler_scale, imputer_stats
                )
                out_pert = model(
                    state=state_pert, drug_fp=fp_b, drug_idx=kg_idx,
                    use_prior=True, precomputed_kg_payload=precomputed_kg_payload,
                )
                pert_auc[gi].append(out_pert["auc_hat"].cpu().numpy())
                if extract_gate and pert_gate is not None and "kg_gate" in out_pert:
                    pert_gate[gi].append(out_pert["kg_gate"].cpu().numpy())

    result = {
        "baseline_auc": np.concatenate(baseline_auc),
        "perturbed_auc": {gi: np.concatenate(v) for gi, v in pert_auc.items()},
    }
    if extract_gate:
        result["gate_baseline"] = np.concatenate(gate_baseline) if gate_baseline else None
        result["gate_perturbed"] = (
            {gi: np.concatenate(v) for gi, v in pert_gate.items()} if pert_gate else None
        )
    return result


def load_cell_metadata(model_list_path: Path, sanger_ids: list[str]) -> pd.DataFrame:
    """Load and filter GDSC cell line metadata for the given SANGER_MODEL_IDs."""
    meta = pd.read_csv(model_list_path, low_memory=False)
    meta = meta.rename(columns={"model_id": "SANGER_MODEL_ID"})
    meta = meta[meta["SANGER_MODEL_ID"].isin(sanger_ids)]
    meta = meta[["SANGER_MODEL_ID", "model_name", "tissue", "cancer_type"]].copy()
    meta = meta.drop_duplicates("SANGER_MODEL_ID")
    return meta


def add_cell_metadata(df: pd.DataFrame, model_list_path: Path) -> pd.DataFrame:
    """Merge cancer type metadata into a DataFrame with SANGER_MODEL_ID column."""
    sids = df["SANGER_MODEL_ID"].unique().tolist()
    meta = load_cell_metadata(model_list_path, sids)
    return df.merge(meta, on="SANGER_MODEL_ID", how="left")


def run_kg_source_ablation(
    model: TerminalWorldModel,
    row_idx: np.ndarray,
    infra: dict,
    prepared,
    config,
    device: str,
    batch_size: int = 2048,
    kg_mask_name: str | None = None,
) -> np.ndarray:
    """Run prediction with a specific KG source ablated."""
    from GAUGE.benchmarking import AblationControls

    tensors = infra["tensors"]
    tensor_banks = infra["tensor_banks"]
    kg_drug_idx_bank = infra["kg_drug_idx_bank"]

    # Build kg_mask for this ablation
    kg_mask = None
    if kg_mask_name is not None:
        # Build a new precomputed_kg_payload with the given mask
        ablation_controls = AblationControls.from_name(kg_mask_name)
        context = TrainExecutionContext()
        context.eval_tensor_cache = {}
        kg_payload = _maybe_precompute_kg_payload(
            model, device=device, controls=ablation_controls,
            edge_dropout=0.0, return_branch_states=False, context=context,
        )
    else:
        kg_payload = infra["precomputed_kg_payload"]

    n = len(row_idx)
    auc_list = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            local_idx = row_idx[start:end]

            state_idx = tensors.state_idx[local_idx]
            drug_idx_t = tensors.drug_idx[local_idx]
            state_b = tensor_banks.state_bank.index_select(0, state_idx)
            fp_b = tensor_banks.fp_bank.index_select(0, drug_idx_t)
            kg_idx = kg_drug_idx_bank.index_select(0, drug_idx_t) if kg_drug_idx_bank is not None else None

            out = model(
                state=state_b, drug_fp=fp_b, drug_idx=kg_idx,
                use_prior=(kg_mask_name != "all_off"),
                precomputed_kg_payload=kg_payload,
            )
            auc_list.append(out["auc_hat"].cpu().numpy())

    return np.concatenate(auc_list)
