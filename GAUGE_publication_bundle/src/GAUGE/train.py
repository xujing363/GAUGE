from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field, replace
from numpy.lib.format import open_memmap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .cache import CacheManager
from .benchmarking import (
    CELL_TRAIN_STATISTICS_FEATURE_COLUMNS,
    AblationControls,
    BenchmarkConfig,
    apply_split,
    infer_scaffold,
    infer_target_families,
)
from .config import Paths, dataset_smiles_path, gdsc_fitted_paths, normalize_dataset_name, normalize_gdsc_source_mode
from .contracts import build_prepare_contract, export_gdsc_benchmark_contract
from .data import (
    attach_smiles,
    attach_smiles_from_table,
    load_ctrp_expression,
    load_beataml2_expression,
    load_gdsc_expression,
    load_pdx_expression,
    load_multisource_drug_prior,
    read_ctrp_v1,
    read_beataml2_fitted,
    read_gdsc_fitted,
    read_pdx_bruna,
)
from .gdsc_smiles import build_dataset_smiles_cache
from .features import (
    FeatureArtifacts,
    add_relative_targets,
    build_canonical_drug_table,
    build_cell_residual_targets,
    build_drug_table,
    build_cell_train_statistics,
    fit_relative_reward,
    fit_state_projection,
    project_expression,
    split_cell_lines,
)
from .drug_level import normalize_fusion_weight_candidates, select_best_fusion_weight, with_fused_auc
from .explainability import build_edge_rows, build_node_rows, build_source_rows, ensure_explain_dir, tensor_to_numpy
from .kg_prior import build_multikg_graph_artifacts, write_kg_reports
from .kg_path_mining import mine_paths_for_prediction
from .metrics import groupwise_auc_correlations, regression_metrics, value_metrics
from .model import TerminalWorldModel, architecture_dict
from .planner import observed_planning_metrics, rank_candidates
from .combined import run_combined_prediction
from .perturbation import plan_perturbation_mechanisms
from .repro import RUNTIME_PROFILE_STABLE, RUNTIME_PROFILE_STRICT, set_reproducible_runtime
from .utils import ensure_dir, write_json


@dataclass
class PreparedData:
    responses: pd.DataFrame
    state_matrix: pd.DataFrame
    artifacts: FeatureArtifacts
    missing_smiles: pd.DataFrame
    invalid_smiles: pd.DataFrame
    prior_audit: pd.DataFrame
    split_audit: pd.DataFrame
    leakage_audit: dict[str, Any]
    manifest: dict[str, Any]


@dataclass
class IndexedResponseTensors:
    state_idx: torch.Tensor
    drug_idx: torch.Tensor
    response: torch.Tensor
    value_train: torch.Tensor | None = None
    drug_centered_train: torch.Tensor | None = None
    cell_residual_train: torch.Tensor | None = None
    drug_group_id: torch.Tensor | None = None
    cell_group_id: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.response.shape[0])


@dataclass
class EvalIndexTensors:
    state_idx: torch.Tensor
    drug_idx: torch.Tensor

    def __len__(self) -> int:
        return int(self.state_idx.shape[0])


@dataclass
class PredictionOutputs:
    core: pd.DataFrame
    planning: pd.DataFrame | None
    explainability: dict[str, pd.DataFrame] | None = None
    runtime: dict[str, Any] | None = None


@dataclass
class PredictionSession:
    runtime_meta: dict[str, Any]
    model: TerminalWorldModel
    model_runner: Any
    context: TrainExecutionContext
    benchmark: BenchmarkConfig
    controls: AblationControls
    tensor_banks: TensorBanks
    use_kg_prior_path: bool
    kg_drug_idx_bank: torch.Tensor | None
    state_bank: torch.Tensor
    fp_bank: torch.Tensor
    prior_bank: torch.Tensor
    mask_bank: torch.Tensor
    use_terminal: bool
    model_kwargs: dict[str, Any]
    precomputed_kg_payload: dict[str, torch.Tensor] | None


@dataclass
class GroupedBatchSamplerIndex:
    n_cells: int
    n_drugs: int
    pair_position_table: np.ndarray
    pair_starts: np.ndarray
    pair_lengths: np.ndarray
    row_index_bank: np.ndarray


@dataclass
class TensorBanks:
    cell_to_idx: dict[str, int]
    drug_to_idx: dict[int, int]
    state_bank: torch.Tensor
    fp_bank: torch.Tensor
    prior_bank: torch.Tensor
    mask_bank: torch.Tensor


@dataclass
class PairwiseWorkspace:
    triangular_index_cache: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict)
    upper_triangle_mask_cache: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict)


@dataclass
class TrainExecutionContext:
    tensor_banks: TensorBanks | None = None
    grouped_sampler_index: GroupedBatchSamplerIndex | None = None
    train_tensor_cache: dict[tuple[str, str, str, str, bool], IndexedResponseTensors] | None = None
    val_tensor_cache: dict[tuple[str, str, str, str, bool], IndexedResponseTensors] | None = None
    eval_tensor_cache: dict[tuple[int, int, str], EvalIndexTensors] | None = None
    eval_kg_payload_cache: dict[tuple[Any, ...], dict[str, torch.Tensor] | None] | None = None
    kg_drug_idx_bank_cache: dict[tuple[str, tuple[int, ...]], torch.Tensor | None] | None = None
    pairwise_workspace: PairwiseWorkspace | None = None


@dataclass
class CompileController:
    requested: bool
    runtime_meta: dict[str, Any]
    prefix: str = "compile"
    active_targets: set[str] = field(default_factory=set)
    fallback_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.runtime_meta[f"{self.prefix}_requested"] = bool(self.requested)
        self.runtime_meta[f"{self.prefix}_enabled"] = False
        self.runtime_meta[f"{self.prefix}_targets"] = []
        self.runtime_meta[f"{self.prefix}_fallback_reason"] = ""

    def enable(self, name: str) -> None:
        self.active_targets.add(str(name))
        self.runtime_meta[f"{self.prefix}_targets"] = sorted(self.active_targets)
        self.runtime_meta[f"{self.prefix}_enabled"] = True

    def fallback(self, name: str, exc: BaseException) -> None:
        self.active_targets.discard(str(name))
        message = f"{name}: {type(exc).__name__}: {' '.join(str(exc).split())}".strip()
        if message not in self.fallback_reasons:
            self.fallback_reasons.append(message)
        self.runtime_meta[f"{self.prefix}_targets"] = sorted(self.active_targets)
        self.runtime_meta[f"{self.prefix}_enabled"] = bool(self.active_targets)
        self.runtime_meta[f"{self.prefix}_fallback_reason"] = "; ".join(self.fallback_reasons)


class _CompiledCallable:
    def __init__(self, target: Any, *, name: str, controller: CompileController):
        self._target = target
        self._compiled = None
        self._name = str(name)
        self._controller = controller
        if not controller.requested:
            return
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            controller.fallback(self._name, RuntimeError("torch.compile is unavailable"))
            return
        try:
            self._compiled = compile_fn(target)
            controller.enable(self._name)
        except Exception as exc:  # pragma: no cover - exact backend errors are environment-specific
            controller.fallback(self._name, exc)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._compiled is None:
            return self._target(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exact backend errors are environment-specific
            self._controller.fallback(self._name, exc)
            self._compiled = None
            return self._target(*args, **kwargs)


def _build_prediction_session(
    model: TerminalWorldModel,
    prepared: PreparedData,
    *,
    device: str,
    benchmark: BenchmarkConfig | None,
    controls: AblationControls | None,
    context: TrainExecutionContext | None,
    runtime_profile: str,
    eval_compile: bool,
    explain_dir: Path | None = None,
) -> PredictionSession:
    runtime_meta = _configure_runtime(device, runtime_profile=runtime_profile)
    model = model.to(device).eval()
    compile_controller = _compile_controller(runtime_meta, requested=bool(eval_compile), prefix="eval_compile")
    model_runner = _maybe_compile_target(model, name="TerminalWorldModel.forward", controller=compile_controller)
    context = _get_or_build_train_context(prepared, device, context)
    tensor_banks = context.tensor_banks
    assert tensor_banks is not None
    benchmark = benchmark or BenchmarkConfig(benchmark_id="adhoc", benchmark_name="adhoc")
    controls = controls or AblationControls()
    use_kg_prior_path = _use_kg_prior_path(model, controls)
    kg_drug_idx_bank = _cached_model_drug_idx_bank(prepared, model, device, context) if use_kg_prior_path else None
    state_bank, fp_bank, prior_bank, mask_bank = _apply_controls(
        tensor_banks.state_bank,
        tensor_banks.fp_bank,
        tensor_banks.prior_bank,
        tensor_banks.mask_bank,
        controls,
    )
    use_terminal = controls.use_terminal and benchmark.world_model.terminal_consequence_enabled
    model_kwargs = _model_control_kwargs(controls)
    precomputed_kg_payload = _maybe_precompute_kg_payload(
        model,
        device=device,
        controls=controls,
        edge_dropout=0.0,
        return_branch_states=bool(benchmark.explainability.enabled and explain_dir is not None),
        context=context,
    )
    return PredictionSession(
        runtime_meta=runtime_meta,
        model=model,
        model_runner=model_runner,
        context=context,
        benchmark=benchmark,
        controls=controls,
        tensor_banks=tensor_banks,
        use_kg_prior_path=use_kg_prior_path,
        kg_drug_idx_bank=kg_drug_idx_bank,
        state_bank=state_bank,
        fp_bank=fp_bank,
        prior_bank=prior_bank,
        mask_bank=mask_bank,
        use_terminal=use_terminal,
        model_kwargs=model_kwargs,
        precomputed_kg_payload=precomputed_kg_payload,
    )


def _append_csv_frame(path: Path, frame: pd.DataFrame, *, header: bool) -> bool:
    if frame.empty:
        return header
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=header, index=False)
    return False


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _state_fit_context(mapped: pd.DataFrame) -> tuple[list[str], pd.DataFrame, list[str], dict[str, str]]:
    cells = sorted(mapped["SANGER_MODEL_ID"].astype(str).unique())
    train_rows = mapped.loc[mapped["split"].eq("train")].copy()
    train_cells = sorted(train_rows["SANGER_MODEL_ID"].astype(str).unique())
    train_cell_set = set(train_cells)
    state_fit_role_by_cell = {c: ("fit" if c in train_cell_set else "transform_only") for c in cells}
    return cells, train_rows, train_cells, state_fit_role_by_cell


def _canonical_drug_table(prepared: PreparedData) -> pd.DataFrame:
    canonical = getattr(prepared.artifacts, "canonical_drug_table", None)
    if canonical is not None and not canonical.empty:
        return canonical
    return prepared.artifacts.drug_table


def _drug_id_to_canonical_idx(prepared: PreparedData) -> dict[int, int]:
    mapping = getattr(prepared.artifacts, "drug_id_to_canonical_idx", None) or {}
    if mapping:
        return {int(k): int(v) for k, v in mapping.items()}
    table = _canonical_drug_table(prepared)
    return {int(row.DRUG_ID): i for i, row in enumerate(table.itertuples(index=False))}


def _build_state_leakage_audit(
    *,
    benchmark: BenchmarkConfig,
    mapped: pd.DataFrame,
    cells: list[str],
    train_cells: list[str],
    genes: list[str],
    state: pd.DataFrame,
    prior_matrix: pd.DataFrame,
    pca,
) -> dict[str, Any]:
    train_rows = mapped.loc[mapped["split"].eq("train")]
    val_rows = mapped.loc[mapped["split"].eq("val")]
    test_rows = mapped.loc[mapped["split"].eq("test")]
    n_train_row_cell_lines = int(train_rows["SANGER_MODEL_ID"].astype(str).nunique())
    n_val_row_cell_lines = int(val_rows["SANGER_MODEL_ID"].astype(str).nunique())
    n_test_row_cell_lines = int(test_rows["SANGER_MODEL_ID"].astype(str).nunique())
    return {
        "state_projection_fit_policy": "train_rows_unique_cells",
        "scaler_fit_split": "train_rows_unique_cells",
        "imputer_fit_split": "train_rows_unique_cells",
        "pca_fit_split": "train_rows_unique_cells",
        "drug_baseline_fit_split": "train_rows_unique_cells",
        "relative_reward_fit_split": "train_rows_unique_cells",
        "relative_value_train_fit_scope": "train_rows_only",
        "relative_value_eval_fit_scope": "split_rows_for_metric_only",
        "relative_value_eval_used_for_training": False,
        "cell_train_statistics_fit_split": "train_rows_only",
        "cell_train_statistics_columns": list(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS),
        "test_used_for_model_selection": False,
        "tcga_used_for_training": False,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_name": benchmark.benchmark_name,
        "split_type": benchmark.split_type,
        "n_total_rows_after_mapping": int(len(mapped)),
        "n_state_fit_cell_lines": int(len(train_cells)),
        "n_all_projected_cell_lines": int(len(cells)),
        "n_train_row_cell_lines": n_train_row_cell_lines,
        "n_val_row_cell_lines": n_val_row_cell_lines,
        "n_test_row_cell_lines": n_test_row_cell_lines,
        "n_train_cell_lines": n_train_row_cell_lines,
        "n_val_cell_lines": n_val_row_cell_lines,
        "n_test_cell_lines": n_test_row_cell_lines,
        "requested_n_components": int(getattr(pca, "n_components", state.shape[1])),
        "actual_pca_components": int(getattr(pca, "n_components_", state.shape[1])),
        "n_genes": int(len(genes)),
        "state_dim": int(state.shape[1]),
        "prior_dim": int(prior_matrix.shape[1]),
    }


def _prepared_kg_drug_space_aligned(prepared: PreparedData) -> bool:
    artifacts = getattr(prepared, "artifacts", None)
    if artifacts is None:
        return True
    kg_graph = getattr(artifacts, "kg_graph", None)
    if kg_graph is None:
        return True
    canonical_table = getattr(artifacts, "canonical_drug_table", None)
    if canonical_table is None or canonical_table.empty:
        canonical_table = _canonical_drug_table(prepared)
    if canonical_table is None or canonical_table.empty:
        return True
    expected_ids = canonical_table["DRUG_ID"].astype(int).tolist()
    actual_ids = [int(x) for x in getattr(kg_graph, "drug_ids", [])]
    return expected_ids == actual_ids


def apply_cell_train_statistics_feature_selection(
    prepared: PreparedData,
    benchmark: BenchmarkConfig | None = None,
) -> PreparedData:
    prepared = _ensure_prepared_compat(prepared)
    requested = tuple(
        str(value)
        for value in getattr(benchmark, "cell_train_statistics_features", CELL_TRAIN_STATISTICS_FEATURE_COLUMNS)
    )
    invalid = [value for value in requested if value not in CELL_TRAIN_STATISTICS_FEATURE_COLUMNS]
    if invalid:
        valid = ", ".join(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS)
        raise ValueError(
            f"Unsupported cell_train_statistics_features requested: {', '.join(invalid)}. "
            f"Valid options: {valid}."
        )
    state = prepared.state_matrix.copy()
    existing = [column for column in CELL_TRAIN_STATISTICS_FEATURE_COLUMNS if column in state.columns]
    missing_requested = [column for column in requested if column not in state.columns]
    if missing_requested:
        raise ValueError(
            "Prepared base cache is missing requested cell_train_statistics_features: "
            + ", ".join(missing_requested)
        )
    drop_columns = [column for column in existing if column not in requested]
    if drop_columns:
        state = state.drop(columns=drop_columns)
    retained = [column for column in CELL_TRAIN_STATISTICS_FEATURE_COLUMNS if column in state.columns]
    leakage_audit = dict(getattr(prepared, "leakage_audit", {}) or {})
    leakage_audit["cell_train_statistics_columns"] = retained
    leakage_audit["state_dim"] = int(state.shape[1])
    manifest = dict(getattr(prepared, "manifest", {}) or {})
    manifest["cell_train_statistics_columns"] = retained
    manifest["cell_train_statistics_enabled"] = bool(retained)
    manifest["cell_train_statistics_features_requested"] = list(requested)
    manifest["state_dim"] = int(state.shape[1])
    artifacts = replace(prepared.artifacts, state_dim=int(state.shape[1]))
    return PreparedData(
        responses=prepared.responses,
        state_matrix=state,
        artifacts=artifacts,
        missing_smiles=prepared.missing_smiles,
        invalid_smiles=prepared.invalid_smiles,
        prior_audit=prepared.prior_audit,
        split_audit=prepared.split_audit,
        leakage_audit=leakage_audit,
        manifest=manifest,
    )


def prepare_data(
    paths: Paths,
    out_dir: Path,
    benchmark: BenchmarkConfig | None = None,
    seed: int = 7,
    n_components: int = 512,
    max_rows: int | None = None,
    gdsc_source_mode: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> PreparedData:
    ensure_dir(out_dir)
    benchmark = benchmark or BenchmarkConfig(benchmark_id="adhoc", benchmark_name="adhoc")
    dataset_name = normalize_dataset_name(getattr(benchmark, "dataset_name", "gdsc"))
    effective_gdsc_source_mode = normalize_gdsc_source_mode(gdsc_source_mode or benchmark.gdsc_source_mode)
    is_ctrp_benchmark = normalize_dataset_name(getattr(benchmark, "dataset_name", "gdsc")) == "ctrdb"
    prepare_contract = build_prepare_contract(
        paths=paths,
        benchmark=benchmark,
        seed=seed,
        n_components=n_components,
        max_rows=max_rows,
        gdsc_source_mode=effective_gdsc_source_mode,
    )
    key = prepare_contract["prepare_cache_key"]
    cache = CacheManager(cache_dir or out_dir / ".cache", use_cache=use_cache, rebuild_cache=rebuild_cache)
    cached = cache.load_pickle("prepare", key, "prepared.pkl")
    if cached is not None:
        cached = _ensure_prepared_compat(cached)
        if _prepared_kg_drug_space_aligned(cached):
            _write_prepare_outputs(cached, out_dir)
            cache.write_reports(out_dir)
            return cached
    if is_ctrp_benchmark:
        responses = read_ctrp_v1(paths.ctrp_response, max_rows=max_rows)
        expr = load_ctrp_expression(paths.ctrp_gene_expression)
        smiles_table = pd.read_csv(paths.ctrp_drug_smiles)
        mapped, missing_smiles = attach_smiles_from_table(responses, smiles_table)
    elif dataset_name == "beataml2":
        responses = read_beataml2_fitted(paths.beataml2_curve_fits, max_rows=max_rows)
        expr = load_beataml2_expression(paths.beataml2_expression)
        mapped, missing_smiles = attach_smiles(responses, paths.beataml2_smiles_cache)
    elif dataset_name == "pdx":
        responses = read_pdx_bruna(paths.pdx_response, max_rows=max_rows)
        expr = load_pdx_expression(paths.pdx_gene_expression)
        build_dataset_smiles_cache(paths, "pdx", out_path=dataset_smiles_path(paths, "pdx"))
        mapped, missing_smiles = attach_smiles(responses, dataset_smiles_path(paths, "pdx"))
    else:
        responses = read_gdsc_fitted(gdsc_fitted_paths(paths, effective_gdsc_source_mode), max_rows=max_rows)
        expr = load_gdsc_expression(paths.gdsc_expression, paths.gdsc_gene_identifiers)
        mapped, missing_smiles = attach_smiles(responses, paths.gdsc_smiles_cache)
    mapped = mapped.loc[mapped["SANGER_MODEL_ID"].isin(expr.index)].copy()
    if benchmark.static_prior.enabled:
        # Keep the legacy multisource static priors available in artifacts for
        # candidate comparisons, while the default forward path still uses the
        # graph-refined KG action encoder.
        prior_matrix, prior_audit, prior_source_stats = load_multisource_drug_prior(
            paths,
            mapped,
            dataset_name=dataset_name,
            sources=benchmark.static_prior.sources,
        )
    else:
        prior_matrix = pd.DataFrame(index=sorted(mapped["drug_key"].dropna().astype(str).unique().tolist()), dtype=np.float32)
        prior_audit = pd.DataFrame(columns=["drug_key", "drug_name", "issue"])
        prior_source_stats = {"static_prior": {"n_drugs": 0, "n_features": 0, "status": "disabled_by_config"}}
    drug_table, invalid_smiles = build_drug_table(mapped, prior_matrix)
    canonical_drug_table, drug_id_to_canonical_idx = build_canonical_drug_table(drug_table)
    if not benchmark.canonical_drug_indexing:
        canonical_drug_table = None
        drug_id_to_canonical_idx = {int(row.DRUG_ID): i for i, row in enumerate(drug_table.itertuples(index=False))}
    kg_graph = build_multikg_graph_artifacts(
        paths,
        canonical_drug_table if canonical_drug_table is not None else drug_table,
        cache_dir=cache_dir or out_dir / ".cache",
        use_cache=use_cache,
        rebuild_cache=rebuild_cache,
        prior_policy=benchmark.prior,
    )
    write_kg_reports(out_dir, kg_graph)
    mapped = mapped.loc[mapped["DRUG_ID"].isin(drug_table["DRUG_ID"])].copy()
    merged_meta = drug_table[
        [
            "DRUG_ID",
            "drug_key",
            "smiles",
            "canonical_smiles",
            "inchikey",
        ]
    ].drop_duplicates("DRUG_ID")
    mapped = mapped.merge(merged_meta, on="DRUG_ID", how="left", suffixes=("", "_drug_table"))
    if canonical_drug_table is not None and not canonical_drug_table.empty:
        canonical_meta = canonical_drug_table[
            [
                "DRUG_ID",
                "canonical_group_key",
                "canonical_group_index",
                "canonical_group_size",
                "canonical_source_drug_ids",
            ]
        ].copy()
        canonical_meta = canonical_meta.rename(columns={"DRUG_ID": "canonical_drug_id"})
        canonical_meta = canonical_meta.explode("canonical_source_drug_ids")
        canonical_meta = canonical_meta.rename(columns={"canonical_source_drug_ids": "DRUG_ID"})
        canonical_meta["DRUG_ID"] = canonical_meta["DRUG_ID"].astype(int)
        mapped = mapped.merge(canonical_meta, on="DRUG_ID", how="left")
    else:
        mapped["canonical_group_key"] = mapped["DRUG_ID"].astype(str)
        mapped["canonical_group_index"] = mapped["DRUG_ID"].map(drug_id_to_canonical_idx).astype(int)
        mapped["canonical_group_size"] = 1
        mapped["canonical_source_drug_ids"] = mapped["DRUG_ID"].map(lambda x: (int(x),))
        mapped["canonical_drug_id"] = mapped["DRUG_ID"].astype(int)
    mapped, split_audit = apply_split(mapped, benchmark, paths)
    cells, train_rows, train_cells, state_fit_role_by_cell = _state_fit_context(mapped)
    if benchmark.split_type in {"all_data", "full_data", "all"}:
        if len(train_cells) < 1:
            raise ValueError(
                f"Need at least 1 train cell line for full-data state preprocessing; got {len(train_cells)}."
            )
    else:
        if len(cells) < 3:
            raise ValueError(
                f"Need at least 3 mapped {dataset_name.upper()} cell lines for disjoint train/val/test split."
            )
        if len(train_cells) < 2:
            raise ValueError(
                f"Need at least 2 train cell lines for state preprocessing; got {len(train_cells)} for split_type={benchmark.split_type}."
            )
    genes, imputer, scaler, pca = fit_state_projection(expr.loc[cells], train_cells, n_components=n_components)
    state = pd.DataFrame(project_expression(expr.loc[cells], genes, imputer, scaler, pca), index=cells)
    cell_train_stats = build_cell_train_statistics(mapped).reindex(cells)
    if not cell_train_stats.empty:
        global_mean = float(train_rows["AUC"].mean())
        global_median = float(train_rows["AUC"].median())
        fill_values = {
            "cell_auc_train_mean": global_mean,
            "cell_auc_train_median": global_median,
            "cell_centered_sensitivity_train": global_median - global_median,
        }
        state = pd.concat([state, cell_train_stats.fillna(fill_values)], axis=1)
    if benchmark.split_type == "drug" and state.shape[1] <= 10:
        raise ValueError(
            "Drug split state_dim unexpectedly collapsed to <= 10. This usually means state preprocessing "
            "was fit from an invalid cell->split compression instead of train rows."
        )
    if benchmark.split_type == "drug" and len(train_cells) >= n_components:
        expected_state_dim = n_components + (0 if cell_train_stats.empty else int(cell_train_stats.shape[1]))
        if state.shape[1] != expected_state_dim:
            raise ValueError(
                "Drug split should preserve the requested PCA state width plus any appended train-only cell statistics; "
                f"expected state_dim={expected_state_dim}, got state_dim={state.shape[1]}."
            )
    baseline, train_values = fit_relative_reward(train_rows)
    mapped = add_relative_targets(mapped, train_values, baseline)
    leakage_audit = _build_state_leakage_audit(
        benchmark=benchmark,
        mapped=mapped,
        cells=cells,
        train_cells=train_cells,
        genes=genes,
        state=state,
        prior_matrix=prior_matrix,
        pca=pca,
    )
    manifest = {
        "dataset_name": dataset_name,
        "gdsc_source_mode": prepare_contract["gdsc_source_mode"],
        "gdsc_fitted_files": prepare_contract["gdsc_fitted_files"],
        "beataml2_curve_fits": str(paths.beataml2_curve_fits) if dataset_name == "beataml2" else None,
        "beataml2_expression": str(paths.beataml2_expression) if dataset_name == "beataml2" else None,
        "beataml2_drug_families": str(paths.beataml2_drug_families) if dataset_name == "beataml2" else None,
        "pdx_response": str(paths.pdx_response) if dataset_name == "pdx" else None,
        "pdx_expression": str(paths.pdx_gene_expression) if dataset_name == "pdx" else None,
        "pdx_drug_names": str(paths.pdx_drug_names) if dataset_name == "pdx" else None,
        "pdx_drug_smiles": str(paths.pdx_drug_smiles) if dataset_name == "pdx" else None,
        "gdsc_expression": str(paths.gdsc_expression) if dataset_name == "gdsc" else None,
        "gdsc_gene_identifiers": str(paths.gdsc_gene_identifiers) if dataset_name == "gdsc" else None,
        "primekg": str(paths.primekg),
        "tcga_h5ad_external_only": str(paths.tcga_h5ad),
        "seed": seed,
        "benchmark": benchmark.prepare_signature(),
        "max_rows": max_rows,
        "n_components_requested": n_components,
        "files_signature": prepare_contract["files_signature"],
        "gdsc_smiles_cache": prepare_contract["gdsc_smiles_cache"],
        "beataml2_smiles_cache": str(paths.beataml2_smiles_cache) if dataset_name == "beataml2" else None,
        "pdx_smiles_cache": str(paths.dataset_smiles_dir / "pdx_smiles.csv") if dataset_name == "pdx" else None,
        "blocked_prior_terms": prepare_contract["blocked_prior_terms"],
        "prepare_contract": prepare_contract,
        "n_rows_raw_response": int(len(responses)),
        "n_rows_after_smiles_expression_prior_validity": int(len(mapped)),
        "n_drugs_action_set": int(drug_table["DRUG_ID"].nunique()),
        "canonical_drug_indexing": bool(benchmark.canonical_drug_indexing),
        "n_canonical_drugs_action_set": int(canonical_drug_table["DRUG_ID"].nunique()) if canonical_drug_table is not None and not canonical_drug_table.empty else int(drug_table["DRUG_ID"].nunique()),
        "resolved_prior_policy": {
            "kg_prior": benchmark.prior.to_dict(),
            "static_prior": benchmark.static_prior.to_dict(),
        },
        "prior_sources": list(benchmark.static_prior.sources) if benchmark.static_prior.enabled else [],
        "kg_prior_sources": [
            name
            for name, cfg in (
                ("chembl", benchmark.prior.chembl),
                ("drkg", benchmark.prior.drkg),
                ("primekg", benchmark.prior.primekg),
            )
            if bool(cfg.enabled) and float(cfg.weight) > 0.0
        ],
        "kg_prior_cache_key": kg_graph.cache_key,
        "prepared_has_kg_graph": bool(kg_graph is not None),
        "kg_node_count": int(len(getattr(kg_graph, "node_table", []))) if kg_graph is not None else 0,
        "kg_edge_count": int(len(getattr(kg_graph, "edge_table", []))) if kg_graph is not None else 0,
        "prior_source_stats": prior_source_stats,
        "prepare_cache_key": key,
        "relative_target_schema_version": prepare_contract["relative_target_schema_version"],
        "cell_train_statistics_version": prepare_contract["cell_train_statistics_version"],
        "cell_train_statistics_columns": list(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS),
        "cell_train_statistics_enabled": bool(cell_train_stats.shape[1]),
        "cell_train_statistics_features_requested": list(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS),
        "state_dim": int(state.shape[1]),
    }
    if dataset_name == "ctrdb":
        manifest["ctrdb_microarray_h5ad_external_only"] = str(paths.ctrdb_microarray_h5ad)
    prepared = PreparedData(
        mapped,
        state,
        FeatureArtifacts(
            genes,
            imputer,
            scaler,
            pca,
            state_fit_role_by_cell,
            baseline,
            train_values,
            drug_table,
            list(prior_matrix.columns),
            int(state.shape[1]),
            kg_graph,
            canonical_drug_table,
            drug_id_to_canonical_idx,
        ),
        missing_smiles,
        invalid_smiles,
        prior_audit,
        split_audit,
        leakage_audit,
        manifest,
    )
    prepared = apply_cell_train_statistics_feature_selection(prepared, benchmark)
    cache.save_pickle("prepare", key, "prepared.pkl", prepared)
    _write_prepare_outputs(prepared, out_dir)
    cache.write_reports(out_dir)
    with (out_dir / "prepared.pkl").open("wb") as f:
        pickle.dump(prepared, f)
    return prepared


def _write_prepare_outputs(prepared: PreparedData, out_dir: Path) -> None:
    write_json(out_dir / "manifest.json", prepared.manifest)
    prepared.missing_smiles.to_csv(out_dir / "missing_smiles_audit.csv", index=False)
    prepared.invalid_smiles.to_csv(out_dir / "invalid_smiles_audit.csv", index=False)
    prepared.prior_audit.to_csv(out_dir / "prior_mapping_audit.csv", index=False)
    prepared.split_audit.to_csv(out_dir / "split_audit.csv", index=False)
    write_json(out_dir / "leakage_audit.json", prepared.leakage_audit)
    export_gdsc_benchmark_contract(prepared, out_dir)
    write_kg_reports(out_dir, getattr(prepared.artifacts, "kg_graph", None))


def load_prepared(path: Path) -> PreparedData:
    with path.open("rb") as f:
        return _ensure_prepared_compat(pickle.load(f))


def _ensure_prepared_compat(prepared: PreparedData) -> PreparedData:
    canonical_enabled = bool((getattr(prepared, "manifest", {}) or {}).get("canonical_drug_indexing", True))
    if not hasattr(prepared, "manifest"):
        prepared.manifest = {
            "compatibility_note": "Loaded from prepared.pkl created before manifest field existed.",
            "n_rows_after_smiles_expression_prior_validity": int(len(prepared.responses)),
            "n_drugs_action_set": int(prepared.artifacts.drug_table["DRUG_ID"].nunique()),
        }
    if not hasattr(prepared, "split_audit"):
        prepared.split_audit = pd.DataFrame()
    if not hasattr(prepared.artifacts, "state_dim"):
        prepared.artifacts.state_dim = int(prepared.state_matrix.shape[1])
    if "relative_value_train" not in prepared.responses.columns:
        prepared.responses["relative_value_train"] = prepared.responses.get("relative_value", np.nan)
    if "relative_value_eval" not in prepared.responses.columns:
        prepared.responses["relative_value_eval"] = prepared.responses.get("relative_value", np.nan)
    if "relative_value" not in prepared.responses.columns:
        prepared.responses["relative_value"] = prepared.responses["relative_value_eval"]
    if "cell_train_baseline" not in prepared.responses.columns or "cell_residual_auc_train" not in prepared.responses.columns:
        prepared.responses = build_cell_residual_targets(prepared.responses)
    if "canonical_group_key" not in prepared.responses.columns:
        canonical_table, canonical_map = build_canonical_drug_table(prepared.artifacts.drug_table)
        canonical_meta = canonical_table[
            [
                "DRUG_ID",
                "canonical_group_key",
                "canonical_group_index",
                "canonical_group_size",
                "canonical_source_drug_ids",
            ]
        ].copy()
        canonical_meta = canonical_meta.rename(columns={"DRUG_ID": "canonical_drug_id"})
        canonical_meta = canonical_meta.explode("canonical_source_drug_ids")
        canonical_meta = canonical_meta.rename(columns={"canonical_source_drug_ids": "DRUG_ID"})
        canonical_meta["DRUG_ID"] = canonical_meta["DRUG_ID"].astype(int)
        meta_cols = ["DRUG_ID", "drug_key", "smiles", "canonical_smiles", "inchikey"]
        meta_cols = [col for col in meta_cols if col in prepared.artifacts.drug_table.columns]
        merged_meta = prepared.artifacts.drug_table[meta_cols].drop_duplicates("DRUG_ID")
        prepared.responses = prepared.responses.merge(merged_meta, on="DRUG_ID", how="left")
        prepared.responses = prepared.responses.merge(canonical_meta, on="DRUG_ID", how="left")
        prepared.artifacts.canonical_drug_table = canonical_table
        prepared.artifacts.drug_id_to_canonical_idx = canonical_map
    elif getattr(prepared.artifacts, "canonical_drug_table", None) is None and canonical_enabled:
        canonical_table, canonical_map = build_canonical_drug_table(prepared.artifacts.drug_table)
        prepared.artifacts.canonical_drug_table = canonical_table
        prepared.artifacts.drug_id_to_canonical_idx = canonical_map
    if not hasattr(prepared.artifacts, "kg_graph"):
        prepared.artifacts.kg_graph = None
    kg_graph = getattr(prepared.artifacts, "kg_graph", None)
    prepared.manifest.setdefault("prepared_has_kg_graph", bool(kg_graph is not None))
    prepared.manifest.setdefault("kg_prior_cache_key", getattr(kg_graph, "cache_key", prepared.manifest.get("kg_prior_cache_key")))
    prepared.manifest.setdefault("kg_node_count", int(len(getattr(kg_graph, "node_table", []))) if kg_graph is not None else 0)
    prepared.manifest.setdefault("kg_edge_count", int(len(getattr(kg_graph, "edge_table", []))) if kg_graph is not None else 0)
    prepared.manifest.setdefault("cell_train_statistics_columns", list(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS))
    prepared.manifest.setdefault(
        "cell_train_statistics_features_requested",
        list(prepared.manifest.get("cell_train_statistics_columns", [])),
    )
    prepared.manifest.setdefault(
        "cell_train_statistics_enabled",
        bool(prepared.manifest.get("cell_train_statistics_columns")),
    )
    prepared.manifest["state_dim"] = int(prepared.state_matrix.shape[1])
    prepared.leakage_audit.setdefault(
        "cell_train_statistics_columns",
        list(prepared.manifest.get("cell_train_statistics_columns", [])),
    )
    prepared.leakage_audit["state_dim"] = int(prepared.state_matrix.shape[1])
    return prepared


def _dicts(prepared: PreparedData) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    state_by_cell = {str(idx): row.to_numpy(dtype=np.float32) for idx, row in prepared.state_matrix.iterrows()}
    drug_by_id = {}
    for row in prepared.artifacts.drug_table.itertuples(index=False):
        drug_by_id[int(row.DRUG_ID)] = {
            "fingerprint": row.fingerprint.astype(np.float32),
            "prior": row.prior.astype(np.float32),
            "prior_mask": float(row.prior_mask),
        }
    return state_by_cell, drug_by_id


def _tensor_banks(prepared: PreparedData, device: str) -> TensorBanks:
    cell_ids = prepared.state_matrix.index.astype(str).tolist()
    cell_to_idx = {cell: i for i, cell in enumerate(cell_ids)}
    state_bank = torch.as_tensor(prepared.state_matrix.to_numpy(np.float32), dtype=torch.float32, device=device)
    drug_table = _canonical_drug_table(prepared)
    drug_rows = list(drug_table.itertuples(index=False))
    drug_to_idx = _drug_id_to_canonical_idx(prepared)
    fp_bank = torch.as_tensor(np.vstack([row.fingerprint.astype(np.float32) for row in drug_rows]), dtype=torch.float32, device=device)
    if len(prepared.artifacts.prior_columns) == 0:
        prior_bank = torch.empty((len(drug_rows), 0), dtype=torch.float32, device=device)
    else:
        prior_bank = torch.as_tensor(np.vstack([row.prior.astype(np.float32) for row in drug_rows]), dtype=torch.float32, device=device)
    mask_bank = torch.as_tensor([[float(row.prior_mask)] for row in drug_rows], dtype=torch.float32, device=device)
    return TensorBanks(
        cell_to_idx=cell_to_idx,
        drug_to_idx=drug_to_idx,
        state_bank=state_bank,
        fp_bank=fp_bank,
        prior_bank=prior_bank,
        mask_bank=mask_bank,
    )


def _model_drug_idx_bank(prepared: PreparedData, model: TerminalWorldModel, device: str) -> torch.Tensor | None:
    drug_rows = list(_canonical_drug_table(prepared).itertuples(index=False))
    if model.kg_action_encoder is None:
        return None
    expected_ids = [int(row.DRUG_ID) for row in drug_rows]
    actual_ids = [int(x) for x in model.kg_action_encoder.drug_ids]
    if expected_ids != actual_ids:
        raise ValueError(
            "Canonical drug action space is misaligned with kg_graph.drug_ids. "
            f"canonical_count={len(expected_ids)} kg_count={len(actual_ids)}. "
            "Rebuild prepared artifacts so KG drug indexing matches the canonical drug table."
        )
    return model.local_drug_indices([int(row.DRUG_ID) for row in drug_rows], device=device)


def _controls_signature(controls: AblationControls) -> str:
    payload = controls.to_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _eval_frame_cache_key(frame: pd.DataFrame) -> tuple[int, int, str]:
    key_frame = frame.loc[:, ["SANGER_MODEL_ID", "DRUG_ID"]]
    hashed = pd.util.hash_pandas_object(key_frame, index=True).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.blake2b(hashed.tobytes(), digest_size=16).hexdigest()
    return (id(frame), len(frame), digest)


def _cached_model_drug_idx_bank(
    prepared: PreparedData,
    model: TerminalWorldModel,
    device: str,
    context: TrainExecutionContext | None,
) -> torch.Tensor | None:
    if context is None:
        return _model_drug_idx_bank(prepared, model, device)
    if context.kg_drug_idx_bank_cache is None:
        context.kg_drug_idx_bank_cache = {}
    drug_ids = tuple(int(row.DRUG_ID) for row in _canonical_drug_table(prepared).itertuples(index=False))
    key = (str(device), drug_ids)
    bank = context.kg_drug_idx_bank_cache.get(key)
    if bank is None:
        bank = _model_drug_idx_bank(prepared, model, device)
        context.kg_drug_idx_bank_cache[key] = bank
    return bank


def _use_kg_prior_path(model: TerminalWorldModel, controls: AblationControls) -> bool:
    if model.kg_action_encoder is None or not controls.use_prior:
        return False
    return str(controls.prior_mode) not in {"legacy_static_prior", "legacy_static", "static_prior"}


def _model_control_kwargs(controls: AblationControls) -> dict[str, Any]:
    mode_map = {
        "learned": "multikg_gat",
        "zero": "multikg_gat",
        "shuffled": "shuffled_mapping",
        "random": "random_prior",
        "legacy_static_prior": "multikg_gat",
        "legacy_static": "multikg_gat",
        "static_prior": "multikg_gat",
    }
    kg_mode = mode_map.get(str(controls.prior_mode), str(controls.prior_mode))
    return {
        "use_prior": bool(controls.use_prior),
        "kg_mode": kg_mode,
        "disable_state_attention": str(controls.prior_mode) == "no_state_attention",
    }


def _apply_controls(
    state_bank: torch.Tensor,
    fp_bank: torch.Tensor,
    prior_bank: torch.Tensor,
    mask_bank: torch.Tensor,
    controls: AblationControls,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not controls.use_state:
        state_bank = torch.zeros_like(state_bank)
    if not controls.use_drug:
        fp_bank = torch.zeros_like(fp_bank)
    if not controls.use_prior or controls.prior_mode == "zero":
        prior_bank = torch.zeros_like(prior_bank)
        mask_bank = torch.zeros_like(mask_bank)
    elif controls.prior_mode == "random" and prior_bank.numel():
        gen = torch.Generator(device=prior_bank.device if prior_bank.device.type == "cuda" else "cpu")
        gen.manual_seed(int(controls.seed))
        prior_bank = torch.randn(prior_bank.shape, generator=gen, device=prior_bank.device, dtype=prior_bank.dtype)
    elif controls.prior_mode == "shuffled" and prior_bank.shape[0] > 1:
        gen = torch.Generator(device=prior_bank.device if prior_bank.device.type == "cuda" else "cpu")
        gen.manual_seed(int(controls.seed))
        perm = torch.randperm(prior_bank.shape[0], generator=gen, device=prior_bank.device)
        prior_bank = prior_bank[perm]
        mask_bank = mask_bank[perm]
    return state_bank, fp_bank, prior_bank, mask_bank


def _configure_runtime(device: str, runtime_profile: str = RUNTIME_PROFILE_STABLE) -> dict[str, Any]:
    runtime = set_reproducible_runtime(seed=0, device=device, profile=runtime_profile)
    runtime["loader_mode"] = "tensor_slicing"
    return runtime


def _compile_controller(runtime_meta: dict[str, Any], *, requested: bool, prefix: str) -> CompileController:
    return CompileController(requested=requested, runtime_meta=runtime_meta, prefix=prefix)


def _module_parameter_versions(module: Any) -> tuple[int, ...]:
    if module is None:
        return ()
    return tuple(int(getattr(param, "_version", 0)) for param in module.parameters())


def _kg_payload_parameter_versions(model: TerminalWorldModel) -> tuple[int, ...]:
    return _module_parameter_versions(model.drug_encoder) + _module_parameter_versions(model.kg_action_encoder)


def _maybe_compile_target(target: Any, *, name: str, controller: CompileController) -> Any:
    if not controller.requested:
        return target
    return _CompiledCallable(target, name=name, controller=controller)


def _maybe_precompute_kg_payload(
    model: TerminalWorldModel,
    *,
    device: str,
    controls: AblationControls,
    edge_dropout: float = 0.0,
    return_branch_states: bool = False,
    context: TrainExecutionContext | None = None,
) -> dict[str, torch.Tensor] | None:
    if not _use_kg_prior_path(model, controls):
        return None
    if model.training or float(edge_dropout) != 0.0:
        return model.precompute_kg_payload(
            device=device,
            edge_dropout=edge_dropout,
            return_branch_states=bool(return_branch_states),
        )
    if context is None:
        return model.precompute_kg_payload(
            device=device,
            edge_dropout=edge_dropout,
            return_branch_states=bool(return_branch_states),
        )
    if context.eval_kg_payload_cache is None:
        context.eval_kg_payload_cache = {}
    key = (
        id(model),
        str(device),
        _controls_signature(controls),
        float(edge_dropout),
        bool(return_branch_states),
        _kg_payload_parameter_versions(model),
    )
    payload = context.eval_kg_payload_cache.get(key)
    if payload is None:
        payload = model.precompute_kg_payload(
            device=device,
            edge_dropout=edge_dropout,
            return_branch_states=bool(return_branch_states),
        )
        if payload is not None and return_branch_states and "branch_node_states" not in payload and model.kg_action_encoder is not None:
            payload = model.kg_action_encoder.precompute_branch_payload(
                payload["drug_latent_bank"],
                device=device,
                edge_dropout=edge_dropout,
            )
        context.eval_kg_payload_cache[key] = payload
    return payload


def _build_forward_batch_inputs(
    *,
    model: TerminalWorldModel,
    state_bank: torch.Tensor,
    fp_bank: torch.Tensor,
    prior_bank: torch.Tensor,
    mask_bank: torch.Tensor,
    state_idx: torch.Tensor,
    drug_idx: torch.Tensor,
    model_drug_idx_bank: torch.Tensor | None,
    use_kg_prior_path: bool,
    precomputed_kg_payload: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor | None]:
    out: dict[str, torch.Tensor | None] = {
        "state": state_bank.index_select(0, state_idx),
        "drug_fp": None,
        "prior": None,
        "prior_mask": None,
        "drug_idx": None,
        "drug_latent": None,
        "drug_latent_bank": None,
    }
    if use_kg_prior_path and model_drug_idx_bank is not None:
        model_drug_idx = model_drug_idx_bank.index_select(0, drug_idx)
        drug_latent_bank = None if precomputed_kg_payload is None else precomputed_kg_payload.get("drug_latent_bank")
        if drug_latent_bank is None and model.kg_action_encoder is not None:
            drug_latent_bank = model.drug_encoder(model.kg_action_encoder.drug_fingerprint_bank)
        out["drug_idx"] = model_drug_idx
        out["drug_latent_bank"] = drug_latent_bank
        if drug_latent_bank is not None:
            out["drug_latent"] = drug_latent_bank.index_select(0, model_drug_idx)
        return out
    out["drug_fp"] = fp_bank.index_select(0, drug_idx)
    out["prior"] = prior_bank.index_select(0, drug_idx)
    out["prior_mask"] = mask_bank.index_select(0, drug_idx)
    return out


def _effective_sampler(mode: str, requested_sampler: str) -> str:
    if str(mode).strip().lower() not in {"world_model", "drug_residual_world_model"}:
        return "random_pair"
    requested = str(requested_sampler or "grouped_world_model").strip().lower()
    if requested == "hybrid_drug_level":
        return "hybrid_drug_level"
    if requested == "random_pair":
        return "random_pair"
    return "grouped_world_model"


def _frame_to_index_tensors(
    frame: pd.DataFrame,
    cell_to_idx: dict[str, int],
    drug_to_idx: dict[int, int],
    device: str,
    *,
    response_field: str = "AUC",
    value_field: str = "relative_value_train",
    drug_centered_field: str = "drug_centered_auc_train",
    cell_residual_field: str = "cell_residual_auc_train",
    include_world_model_fields: bool = True,
) -> IndexedResponseTensors:
    if include_world_model_fields:
        drug_group_codes, _ = pd.factorize(frame["DRUG_ID"].astype(int), sort=False)
        cell_group_codes, _ = pd.factorize(frame["SANGER_MODEL_ID"].astype(str), sort=False)
        value_train = torch.as_tensor(frame[value_field].astype(float).to_numpy(np.float32), dtype=torch.float32, device=device)
        if drug_centered_field in frame.columns:
            drug_centered_train = torch.as_tensor(frame[drug_centered_field].astype(float).to_numpy(np.float32), dtype=torch.float32, device=device)
        else:
            drug_centered_train = torch.full((len(frame),), float("nan"), dtype=torch.float32, device=device)
        if cell_residual_field in frame.columns:
            cell_residual_train = torch.as_tensor(frame[cell_residual_field].astype(float).to_numpy(np.float32), dtype=torch.float32, device=device)
        else:
            cell_residual_train = torch.full((len(frame),), float("nan"), dtype=torch.float32, device=device)
        drug_group_id = torch.as_tensor(drug_group_codes, dtype=torch.int32, device=device)
        cell_group_id = torch.as_tensor(cell_group_codes, dtype=torch.int32, device=device)
    else:
        value_train = None
        drug_centered_train = None
        cell_residual_train = None
        drug_group_id = None
        cell_group_id = None
    return IndexedResponseTensors(
        state_idx=torch.as_tensor([cell_to_idx[str(x)] for x in frame["SANGER_MODEL_ID"]], dtype=torch.long, device=device),
        drug_idx=torch.as_tensor([drug_to_idx[int(x)] for x in frame["DRUG_ID"]], dtype=torch.long, device=device),
        response=torch.as_tensor(frame[response_field].astype(float).to_numpy(np.float32), dtype=torch.float32, device=device),
        value_train=value_train,
        drug_centered_train=drug_centered_train,
        cell_residual_train=cell_residual_train,
        drug_group_id=drug_group_id,
        cell_group_id=cell_group_id,
    )


def _frame_to_eval_tensors(
    frame: pd.DataFrame,
    cell_to_idx: dict[str, int],
    drug_to_idx: dict[int, int],
    device: str,
) -> EvalIndexTensors:
    return EvalIndexTensors(
        state_idx=torch.as_tensor([cell_to_idx[str(x)] for x in frame["SANGER_MODEL_ID"]], dtype=torch.long, device=device),
        drug_idx=torch.as_tensor([drug_to_idx[int(x)] for x in frame["DRUG_ID"]], dtype=torch.long, device=device),
    )


def _get_or_build_train_context(
    prepared: PreparedData,
    device: str,
    context: TrainExecutionContext | None,
) -> TrainExecutionContext:
    context = context or TrainExecutionContext()
    if context.tensor_banks is None:
        context.tensor_banks = _tensor_banks(prepared, device)
    if context.grouped_sampler_index is None:
        train_df = prepared.responses.loc[prepared.responses["split"].eq("train")].copy().reset_index(drop=True)
        context.grouped_sampler_index = _build_grouped_batch_sampler_index(train_df)
    if context.train_tensor_cache is None:
        context.train_tensor_cache = {}
    if context.val_tensor_cache is None:
        context.val_tensor_cache = {}
    if context.eval_tensor_cache is None:
        context.eval_tensor_cache = {}
    if context.eval_kg_payload_cache is None:
        context.eval_kg_payload_cache = {}
    if context.kg_drug_idx_bank_cache is None:
        context.kg_drug_idx_bank_cache = {}
    if context.pairwise_workspace is None:
        context.pairwise_workspace = PairwiseWorkspace()
    return context


def _cached_index_tensors(
    frame: pd.DataFrame,
    cache: dict[tuple[str, str, str, str, bool], IndexedResponseTensors],
    tensor_banks: TensorBanks,
    device: str,
    *,
    response_field: str,
    value_field: str,
    drug_centered_field: str,
    cell_residual_field: str,
    include_world_model_fields: bool,
) -> IndexedResponseTensors:
    key = (response_field, value_field, drug_centered_field, cell_residual_field, include_world_model_fields)
    tensors = cache.get(key)
    if tensors is None:
        tensors = _frame_to_index_tensors(
            frame,
            tensor_banks.cell_to_idx,
            tensor_banks.drug_to_idx,
            device,
            response_field=response_field,
            value_field=value_field,
            drug_centered_field=drug_centered_field,
            cell_residual_field=cell_residual_field,
            include_world_model_fields=include_world_model_fields,
        )
        cache[key] = tensors
    return tensors


def _cached_eval_tensors(
    frame: pd.DataFrame,
    cache: dict[tuple[int, int, str], EvalIndexTensors],
    tensor_banks: TensorBanks,
    device: str,
) -> EvalIndexTensors:
    key = _eval_frame_cache_key(frame)
    tensors = cache.get(key)
    if tensors is None:
        tensors = _frame_to_eval_tensors(frame, tensor_banks.cell_to_idx, tensor_banks.drug_to_idx, device)
        cache[key] = tensors
    return tensors


def _iter_batch_indices(n_rows: int, batch_size: int, device: str, *, shuffle: bool, seed: int) -> tuple[torch.Tensor, ...]:
    if n_rows <= 0:
        return ()
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = torch.randperm(n_rows, generator=generator)
    else:
        order = torch.arange(n_rows)
    if str(device) != "cpu":
        order = order.to(device=device)
    return tuple(order[start : start + batch_size] for start in range(0, n_rows, batch_size))


def _build_grouped_batch_sampler_index(frame: pd.DataFrame) -> GroupedBatchSamplerIndex | None:
    if frame.empty:
        return None
    cell_codes, cell_values = pd.factorize(frame["SANGER_MODEL_ID"].astype(str), sort=False)
    drug_codes, drug_values = pd.factorize(frame["DRUG_ID"].astype(int), sort=False)
    n_cells = int(cell_codes.max()) + 1 if len(cell_codes) else 0
    n_drugs = int(len(drug_values))
    row_ids = np.arange(len(frame), dtype=np.int32)
    pair_codes = cell_codes.astype(np.int64, copy=False) * max(n_drugs, 1) + drug_codes.astype(np.int64, copy=False)
    order = np.argsort(pair_codes, kind="stable")
    sorted_pair_codes = pair_codes[order]
    unique_pair_keys, pair_starts, pair_lengths = np.unique(sorted_pair_codes, return_index=True, return_counts=True)
    pair_position_table = np.full((max(n_cells, 1), max(n_drugs, 1)), -1, dtype=np.int64)
    if unique_pair_keys.size:
        pair_position_table.reshape(-1)[unique_pair_keys] = np.arange(unique_pair_keys.shape[0], dtype=np.int64)
    return GroupedBatchSamplerIndex(
        n_cells=n_cells,
        n_drugs=n_drugs,
        pair_position_table=np.ascontiguousarray(pair_position_table, dtype=np.int64),
        pair_starts=np.ascontiguousarray(pair_starts.astype(np.int64, copy=False)),
        pair_lengths=np.ascontiguousarray(pair_lengths.astype(np.int64, copy=False)),
        row_index_bank=np.ascontiguousarray(row_ids[order], dtype=np.int64),
    )


def _iter_grouped_world_model_batches(
    sampler_index: GroupedBatchSamplerIndex | None,
    *,
    n_steps: int,
    n_cell_lines: int,
    n_drugs: int,
    min_cell_drugs: int = 2,
    min_drug_cells: int = 2,
    device: str,
    seed: int,
    stats: dict[str, float] | None = None,
) -> Any:
    if sampler_index is None or sampler_index.n_cells <= 0 or sampler_index.n_drugs <= 0:
        return
    device_target = device if str(device) != "cpu" else "cpu"
    generator = np.random.default_rng(int(seed))
    for _ in range(max(n_steps, 1)):
        batch_idx: np.ndarray | None = None
        attempts = 0
        for _attempt in range(12):
            attempts += 1
            chosen_cell_codes = generator.choice(
                sampler_index.n_cells,
                size=min(sampler_index.n_cells, max(n_cell_lines, 1)),
                replace=False,
            )
            chosen_drug_codes = generator.choice(
                sampler_index.n_drugs,
                size=min(sampler_index.n_drugs, max(n_drugs, 1)),
                replace=False,
            )
            pair_positions = sampler_index.pair_position_table[np.ix_(chosen_cell_codes, chosen_drug_codes)].reshape(-1)
            matched_positions = pair_positions[pair_positions >= 0]
            if matched_positions.size == 0:
                continue
            pair_presence = sampler_index.pair_position_table[np.ix_(chosen_cell_codes, chosen_drug_codes)] >= 0
            cell_pair_counts = pair_presence.sum(axis=1)
            drug_pair_counts = pair_presence.sum(axis=0)
            if int(cell_pair_counts.max()) < int(min_cell_drugs):
                continue
            if int(drug_pair_counts.max()) < int(min_drug_cells):
                continue
            counts = sampler_index.pair_lengths[matched_positions]
            total_rows = int(counts.sum())
            if total_rows == 0:
                continue
            starts = sampler_index.pair_starts[matched_positions]
            segment_offsets = np.concatenate([np.zeros((1,), dtype=np.int64), np.cumsum(counts, dtype=np.int64)[:-1]])
            within = np.arange(total_rows, dtype=np.int64) - np.repeat(segment_offsets, counts)
            gather_positions = np.repeat(starts, counts) + within
            batch_idx = np.sort(sampler_index.row_index_bank[gather_positions])
            break
        if batch_idx is None or batch_idx.size == 0:
            continue
        if stats is not None:
            stats["batches"] = float(stats.get("batches", 0.0) + 1.0)
            stats["attempts_total"] = float(stats.get("attempts_total", 0.0) + float(attempts))
            stats["rows_total"] = float(stats.get("rows_total", 0.0) + float(batch_idx.size))
        yield torch.as_tensor(batch_idx, dtype=torch.long, device=device_target)


def _iter_hybrid_drug_level_batches(
    *,
    n_rows: int,
    batch_size: int,
    grouped_sampler_index: GroupedBatchSamplerIndex | None,
    n_grouped_steps: int,
    rank_batch_fraction: float,
    n_cell_lines: int,
    n_drugs: int,
    min_cell_drugs: int = 2,
    min_drug_cells: int = 2,
    device: str,
    seed: int,
    stats: dict[str, float] | None = None,
) -> Any:
    for batch_idx in _iter_batch_indices(n_rows, batch_size, device, shuffle=True, seed=seed):
        yield "random_pair", batch_idx
    grouped_steps = int(np.ceil(max(n_grouped_steps, 0) * max(float(rank_batch_fraction), 0.0)))
    if grouped_steps <= 0:
        return
    for batch_idx in _iter_grouped_world_model_batches(
        grouped_sampler_index,
        n_steps=grouped_steps,
        n_cell_lines=n_cell_lines,
        n_drugs=n_drugs,
        min_cell_drugs=min_cell_drugs,
        min_drug_cells=min_drug_cells,
        device=device,
        seed=seed + 104729,
        stats=stats,
    ):
        yield "grouped_rank", batch_idx


def _pairwise_triu_indices(
    workspace: PairwiseWorkspace | None,
    *,
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    if workspace is None:
        return torch.triu_indices(group_size, group_size, offset=1, device=device)
    key = (str(device), int(group_size))
    cached = workspace.triangular_index_cache.get(key)
    if cached is None:
        cached = torch.triu_indices(group_size, group_size, offset=1, device=device)
        workspace.triangular_index_cache[key] = cached
    return cached


def _pairwise_upper_triangle_mask(
    workspace: PairwiseWorkspace | None,
    *,
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    if workspace is None:
        return torch.triu(torch.ones((group_size, group_size), dtype=torch.bool, device=device), diagonal=1)
    key = (str(device), int(group_size))
    cached = workspace.upper_triangle_mask_cache.get(key)
    if cached is None:
        cached = torch.triu(torch.ones((group_size, group_size), dtype=torch.bool, device=device), diagonal=1)
        workspace.upper_triangle_mask_cache[key] = cached
    return cached


def _pairwise_margin_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    aux_response: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    min_target_gap: float,
    min_response_gap: float,
    margin: float,
    workspace: PairwiseWorkspace | None = None,
) -> tuple[torch.Tensor, int]:
    total_loss = scores.new_zeros(())
    n_pairs = 0
    n_groups = 0
    valid_mask = torch.isfinite(target)
    if not bool(valid_mask.any().item()):
        return total_loss, 0
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_idx.numel() < 2:
        return total_loss, 0
    valid_scores = scores.index_select(0, valid_idx)
    valid_target = target.index_select(0, valid_idx)
    valid_response = aux_response.index_select(0, valid_idx)
    valid_group_ids = group_ids.index_select(0, valid_idx)
    valid_group_ids, order = torch.sort(valid_group_ids, stable=True)
    valid_scores = valid_scores.index_select(0, order)
    valid_target = valid_target.index_select(0, order)
    valid_response = valid_response.index_select(0, order)
    boundaries = torch.nonzero(valid_group_ids[1:] != valid_group_ids[:-1], as_tuple=False).squeeze(1) + 1
    segment_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=scores.device), boundaries])
    segment_ends = torch.cat([boundaries, torch.tensor([valid_group_ids.shape[0]], dtype=torch.long, device=scores.device)])
    counts = segment_ends - segment_starts
    long_enough = counts >= 2
    if not bool(long_enough.any().item()):
        return scores.new_zeros(()), 0
    segment_starts = segment_starts[long_enough].detach().cpu().tolist()
    counts = counts[long_enough].detach().cpu().tolist()
    for start, count in zip(segment_starts, counts):
        tri = _pairwise_triu_indices(workspace, group_size=int(count), device=scores.device)
        left = tri[0]
        right = tri[1]
        group_scores = valid_scores.narrow(0, int(start), int(count))
        group_target = valid_target.narrow(0, int(start), int(count))
        group_response = valid_response.narrow(0, int(start), int(count))
        target_diff = group_target.index_select(0, left) - group_target.index_select(0, right)
        response_diff = group_response.index_select(0, left) - group_response.index_select(0, right)
        valid_pairs = ((target_diff.abs() > min_target_gap) | (response_diff.abs() > min_response_gap)) & (target_diff != 0)
        if not bool(valid_pairs.any().item()):
            continue
        score_diff = group_scores.index_select(0, left) - group_scores.index_select(0, right)
        desired = torch.sign(target_diff[valid_pairs])
        pair_loss = F.relu(margin - desired * score_diff[valid_pairs])
        total_loss = total_loss + pair_loss.mean()
        n_pairs += int(valid_pairs.sum().item())
        n_groups += 1
    if n_groups == 0:
        return scores.new_zeros(()), 0
    return total_loss / n_groups, n_pairs


def _terminal_drug_specificity_loss(
    terminal_latent: torch.Tensor,
    state_latent: torch.Tensor,
    cell_group_id: torch.Tensor,
    drug_group_id: torch.Tensor,
    *,
    margin: float,
    workspace: PairwiseWorkspace | None = None,
) -> tuple[torch.Tensor, int]:
    total_loss = terminal_latent.new_zeros(())
    n_pairs = 0
    n_groups = 0
    if terminal_latent.shape[0] < 2:
        return total_loss, 0
    valid_mask = torch.isfinite(terminal_latent).all(dim=1) & torch.isfinite(state_latent).all(dim=1)
    if not bool(valid_mask.any().item()):
        return total_loss, 0
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_idx.numel() < 2:
        return total_loss, 0
    valid_terminal = terminal_latent.index_select(0, valid_idx)
    valid_state = state_latent.index_select(0, valid_idx)
    valid_cell = cell_group_id.index_select(0, valid_idx)
    valid_drug = drug_group_id.index_select(0, valid_idx)
    valid_cell, order = torch.sort(valid_cell, stable=True)
    valid_terminal = valid_terminal.index_select(0, order)
    valid_state = valid_state.index_select(0, order)
    valid_drug = valid_drug.index_select(0, order)
    boundaries = torch.nonzero(valid_cell[1:] != valid_cell[:-1], as_tuple=False).squeeze(1) + 1
    segment_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=terminal_latent.device), boundaries])
    segment_ends = torch.cat([boundaries, torch.tensor([valid_cell.shape[0]], dtype=torch.long, device=terminal_latent.device)])
    counts = segment_ends - segment_starts
    long_enough = counts >= 2
    if not bool(long_enough.any().item()):
        return total_loss, 0
    segment_starts = segment_starts[long_enough].detach().cpu().tolist()
    counts = counts[long_enough].detach().cpu().tolist()
    for start, count in zip(segment_starts, counts):
        group_terminal = valid_terminal.narrow(0, int(start), int(count))
        group_state = valid_state.narrow(0, int(start), int(count))
        group_drug = valid_drug.narrow(0, int(start), int(count))
        tri = _pairwise_triu_indices(workspace, group_size=int(count), device=terminal_latent.device)
        left = tri[0]
        right = tri[1]
        different_drug = group_drug.index_select(0, left) != group_drug.index_select(0, right)
        if not bool(different_drug.any().item()):
            continue
        delta_left = group_terminal.index_select(0, left) - group_state.index_select(0, left)
        delta_right = group_terminal.index_select(0, right) - group_state.index_select(0, right)
        pair_similarity = F.cosine_similarity(delta_left, delta_right, dim=1)
        pair_loss = F.relu(pair_similarity[different_drug] - margin)
        if not bool(pair_loss.numel()):
            continue
        total_loss = total_loss + pair_loss.mean()
        n_pairs += int(different_drug.sum().item())
        n_groups += 1
    if n_groups == 0:
        return total_loss, 0
    return total_loss / n_groups, n_pairs


def _planning_metrics_disabled(top_k: int, status: str) -> dict[str, float | str]:
    return {
        "proxy_planning_gain_mean": float("nan"),
        "proxy_regret_mean": float("nan"),
        "proxy_topk_hit_rate": float("nan"),
        "proxy_planning_entities": 0.0,
        "proxy_top_k": float(top_k),
        "proxy_planning_status": status,
    }


def _kg_coverage_bucket(row: pd.Series) -> str:
    def _flag(name: str) -> bool:
        value = row.get(name, 0)
        if pd.isna(value):
            return False
        try:
            return float(value) > 0
        except Exception:
            return bool(value)

    has_chembl = _flag("has_ChEMBL")
    has_drkg = _flag("has_DRKG")
    has_primekg = _flag("has_PrimeKG")
    if not (has_chembl or has_drkg or has_primekg):
        return "no_KG"
    if has_primekg and not has_chembl and not has_drkg:
        return "PrimeKG_only"
    if has_chembl and has_primekg and not has_drkg:
        return "ChEMBL+PrimeKG"
    if has_drkg and has_primekg and not has_chembl:
        return "DRKG+PrimeKG"
    if has_chembl and has_drkg and has_primekg:
        return "all_KG"
    return "other_KG"


def _drug_bucket_label(pcc: float) -> str:
    if not np.isfinite(pcc):
        return "nan_pcc"
    if pcc < 0.0:
        return "negative_pcc_drugs"
    if pcc < 0.2:
        return "low_pcc_drugs_lt_0.2"
    if pcc > 0.6:
        return "good_pcc_drugs_gt_0.6"
    return "mid_pcc_drugs_0.2_to_0.6"


def _write_drug_level_audits(pred: pd.DataFrame, prepared: PreparedData, out_dir: Path) -> None:
    if pred.empty:
        return
    drug_metrics = groupwise_auc_correlations(pred, "DRUG_ID")
    if drug_metrics.empty:
        return
    meta = pred[
        [
            "DRUG_ID",
            "DRUG_NAME",
            "drug_key",
            "smiles",
            "canonical_smiles",
            "inchikey",
            "canonical_group_key",
            "canonical_group_index",
            "canonical_group_size",
            "canonical_drug_id",
            "split",
        ]
    ].drop_duplicates("DRUG_ID")
    coverage = pd.DataFrame()
    kg_graph = getattr(prepared.artifacts, "kg_graph", None)
    if kg_graph is not None and getattr(kg_graph, "coverage", None) is not None and not kg_graph.coverage.empty:
        coverage = kg_graph.coverage.copy()
        if "DRUG_ID" not in coverage.columns and "GDSC_DRUG_ID" in coverage.columns:
            coverage = coverage.rename(columns={"GDSC_DRUG_ID": "DRUG_ID"})
    if not coverage.empty:
        coverage = coverage.copy()
        coverage["coverage_bucket"] = coverage.apply(_kg_coverage_bucket, axis=1)
        if "DRUG_ID" in coverage.columns:
            coverage = coverage.rename(columns={"DRUG_ID": "canonical_drug_id"})
        meta = meta.merge(
            coverage[["canonical_drug_id", "coverage_bucket", "has_ChEMBL", "has_DRKG", "has_PrimeKG"]],
            on="canonical_drug_id",
            how="left",
        )
    else:
        meta["coverage_bucket"] = "unknown"
    scaffold_source = meta["smiles"].where(meta["smiles"].notna(), meta["canonical_smiles"]).fillna("")
    meta["scaffold"] = scaffold_source.map(infer_scaffold)
    primekg_path_value = str(prepared.manifest.get("primekg", "") or "").strip()
    primekg_path = Path(primekg_path_value) if primekg_path_value else None
    if primekg_path is not None and primekg_path.exists():
        family_map = infer_target_families(pred[["DRUG_ID", "DRUG_NAME"]].drop_duplicates(), primekg_path)
        meta = meta.merge(family_map, on=["DRUG_ID", "DRUG_NAME"], how="left")
    else:
        meta["target_family"] = "unknown"
        meta["family_source"] = "missing_primekg"
    detail = drug_metrics.merge(meta, on="DRUG_ID", how="left")
    detail["drug_bucket"] = detail["pcc"].map(_drug_bucket_label)
    detail.to_csv(out_dir / "gdsc_drug_error_buckets.csv", index=False)
    summary = (
        detail.groupby(["split", "drug_bucket", "coverage_bucket", "scaffold", "target_family"], observed=True)
        .agg(
            n_drugs=("DRUG_ID", "nunique"),
            mean_pcc=("pcc", "mean"),
            mean_spearman=("spearman", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "gdsc_drug_bucket_summary.csv", index=False)
    coverage_summary = (
        detail.groupby(["split", "coverage_bucket"], observed=True)
        .agg(
            n_drugs=("DRUG_ID", "nunique"),
            within_drug_pcc_mean=("pcc", "mean"),
            within_drug_spearman_mean=("spearman", "mean"),
        )
        .reset_index()
    )
    coverage_summary.to_csv(out_dir / "gdsc_coverage_metrics.csv", index=False)


def _validation_objective_score(
    objective: str,
    metrics_row: dict[str, float],
) -> tuple[float, str]:
    objective = str(objective or "val_within_drug_pcc")
    maximize_fields = {"val_within_drug_pcc"}
    candidate = metrics_row.get(objective, float("nan"))
    if np.isfinite(candidate):
        score = float(candidate) if objective in maximize_fields else -float(candidate)
        return score, objective
    fallback = float(metrics_row.get("val_fused_loss", metrics_row.get("val_raw_loss", float("inf"))))
    return -fallback, "val_fused_loss"


def train_model(
    prepared: PreparedData,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 7,
    device: str | None = None,
    benchmark: BenchmarkConfig | None = None,
    controls: AblationControls | None = None,
    context: TrainExecutionContext | None = None,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
) -> TerminalWorldModel:
    prepared = _ensure_prepared_compat(prepared)
    ensure_dir(out_dir)
    if device is None:
        raise ValueError("Explicit --device is required for train/run. Use --device cuda:N or --device cpu.")
    model: TerminalWorldModel | None = None
    context = None
    tensor_banks = None
    train_tensors = None
    val_tensors = None
    kg_drug_idx_bank = None
    state_bank = None
    fp_bank = None
    prior_bank = None
    mask_bank = None
    model_runner = None
    opt = None
    runtime_meta = set_reproducible_runtime(seed=seed, device=device, profile=runtime_profile)
    if True:
        runtime_meta["loader_mode"] = "tensor_slicing"
        benchmark = benchmark or BenchmarkConfig(benchmark_id="adhoc", benchmark_name="adhoc")
        controls = controls or AblationControls(seed=seed)
        runtime_meta["canonical_drug_indexing"] = bool(getattr(benchmark, "canonical_drug_indexing", False))
        runtime_meta["coverage_aware_kg_mask"] = bool(getattr(benchmark, "coverage_aware_kg_mask", True))
        mode = str(benchmark.model.mode).strip().lower()
        if mode not in {"regression", "world_model", "drug_residual_world_model"}:
            raise ValueError(f"Unsupported model.mode: {benchmark.model.mode!r}")
        effective_sampler = _effective_sampler(mode, benchmark.batch.sampler)
        runtime_meta["requested_batch_sampler"] = str(benchmark.batch.sampler)
        runtime_meta["effective_batch_sampler"] = effective_sampler
        if effective_sampler == "grouped_world_model":
            runtime_meta["sampler_impl"] = "numpy_grouped_world_model"
            runtime_meta["train_sampler_path"] = "GAUGE.train._iter_grouped_world_model_batches"
        elif effective_sampler == "hybrid_drug_level":
            runtime_meta["sampler_impl"] = "hybrid_drug_level"
            runtime_meta["train_sampler_path"] = "GAUGE.train._iter_hybrid_drug_level_batches"
            runtime_meta["rank_batch_fraction"] = float(benchmark.batch.rank_batch_fraction)
        else:
            runtime_meta["sampler_impl"] = "torch_random_pair"
            runtime_meta["train_sampler_path"] = "GAUGE.train._iter_batch_indices"
        uses_world_losses = mode in {"world_model", "drug_residual_world_model"}
        runtime_meta["pairwise_impl"] = "torch_segmented_pairwise" if uses_world_losses else "disabled"
        response_field = benchmark.model.target_field
        value_field = benchmark.world_model.relative_value_train_field
        val_value_field = benchmark.world_model.relative_value_eval_field
        cell_residual_train_field = "cell_residual_auc_train"
        cell_residual_eval_field = "cell_residual_auc_eval"
        context = _get_or_build_train_context(prepared, device, context)
        train_df = prepared.responses.loc[prepared.responses["split"].eq("train")].copy().reset_index(drop=True)
        val_df = prepared.responses.loc[prepared.responses["split"].eq("val")].copy()
        if benchmark.split_type in {"all_data", "full_data", "all"} or val_df.empty:
            val_df = train_df.copy()
        model_drug_table = _canonical_drug_table(prepared)
        model = TerminalWorldModel(
            state_dim=prepared.state_matrix.shape[1],
            prior_dim=len(prepared.artifacts.prior_columns),
            kg_artifacts=getattr(prepared.artifacts, "kg_graph", None),
            drug_fingerprint_bank=np.vstack([row.fingerprint.astype(np.float32) for row in model_drug_table.itertuples(index=False)]),
        ).to(device)
        train_compile_requested = str(device).startswith("cuda") and mode == "world_model" and runtime_meta.get("runtime_profile") != RUNTIME_PROFILE_STRICT
        compile_controller = _compile_controller(runtime_meta, requested=bool(train_compile_requested), prefix="train_compile")
        model_runner = _maybe_compile_target(model, name="TerminalWorldModel.forward", controller=compile_controller)
        pairwise_loss_runner = _pairwise_margin_loss
        tensor_banks = context.tensor_banks
        assert tensor_banks is not None
        use_kg_prior_path = _use_kg_prior_path(model, controls)
        runtime_meta["uses_kg_prior_path"] = bool(use_kg_prior_path)
        runtime_meta["uses_legacy_static_prior_path"] = bool(controls.use_prior and not use_kg_prior_path)
        kg_drug_idx_bank = _cached_model_drug_idx_bank(prepared, model, device, context) if use_kg_prior_path else None
        state_bank, fp_bank, prior_bank, mask_bank = _apply_controls(
            tensor_banks.state_bank,
            tensor_banks.fp_bank,
            tensor_banks.prior_bank,
            tensor_banks.mask_bank,
            controls,
        )
        include_world_model_fields = uses_world_losses
        train_tensors = _cached_index_tensors(
            train_df,
            context.train_tensor_cache,
            tensor_banks,
            device,
            response_field=response_field,
            value_field=value_field,
            drug_centered_field="drug_centered_auc_train",
            cell_residual_field=cell_residual_train_field,
            include_world_model_fields=include_world_model_fields,
        )
        val_tensors = _cached_index_tensors(
            val_df,
            context.val_tensor_cache,
            tensor_banks,
            device,
            response_field=response_field,
            value_field=val_value_field,
            drug_centered_field="drug_centered_auc_eval",
            cell_residual_field=cell_residual_eval_field,
            include_world_model_fields=include_world_model_fields,
        )
        grouped_sampler_index = context.grouped_sampler_index
        pairwise_workspace = context.pairwise_workspace
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best = {"score": float("-inf"), "epoch": 0, "selected_fusion_weight": 1.0, "state": None}
        history = []
        rank_weight = float(benchmark.training_objectives.loss_within_drug_rank_weight)
        adv_weight = float(benchmark.training_objectives.loss_policy_advantage_weight)
        value_weight = float(benchmark.training_objectives.loss_value_weight)
        centered_weight = float(benchmark.training_objectives.loss_drug_centered_weight)
        cell_residual_weight = float(getattr(benchmark.training_objectives, "loss_cell_residual_weight", 0.0))
        if cell_residual_weight <= 0.0:
            cell_residual_weight = centered_weight
        same_cell_rank_weight = float(getattr(benchmark.training_objectives, "loss_same_cell_cross_drug_rank_weight", 0.0))
        if same_cell_rank_weight <= 0.0:
            same_cell_rank_weight = adv_weight
        terminal_drug_specificity_weight = float(getattr(benchmark.training_objectives, "loss_terminal_drug_specificity_weight", 0.0))
        terminal_drug_specificity_margin = float(getattr(benchmark.training_objectives, "terminal_drug_specificity_margin", benchmark.training_objectives.advantage_margin))
        graph_weight = float(benchmark.training_objectives.loss_graph_consistency_weight)
        validation_objective = str(benchmark.training_objectives.validation_objective)
        fusion_candidates = normalize_fusion_weight_candidates(benchmark.evaluation.fusion_weight_candidates)
        fusion_metric = str(benchmark.evaluation.fusion_selection_metric)
        use_terminal = controls.use_terminal and benchmark.world_model.terminal_consequence_enabled
        model_kwargs = _model_control_kwargs(controls)
        runtime_meta["validation_objective"] = validation_objective
        runtime_meta["fusion_selection_metric"] = fusion_metric
        runtime_meta["fusion_weight_candidates"] = list(fusion_candidates)
        train_sampler_seconds = 0.0
        train_forward_seconds = 0.0
        train_pairwise_seconds = 0.0
        val_seconds_total = 0.0
        batch_rows_total = 0.0
        batch_unique_drugs_total = 0.0
        batch_unique_cells_total = 0.0
        rank_group_total = 0.0
        adv_group_total = 0.0
        rank_pairs_obs_total = 0.0
        adv_pairs_obs_total = 0.0
        batch_count_total = 0.0
        sampler_stats: dict[str, float] = {"batches": 0.0, "attempts_total": 0.0, "rows_total": 0.0}
        train_started_at = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            train_sums = {
                "loss_total": torch.zeros((), dtype=torch.float32, device=device),
                "loss_raw": torch.zeros((), dtype=torch.float32, device=device),
                "loss_absolute_auc": torch.zeros((), dtype=torch.float32, device=device),
                "loss_value": torch.zeros((), dtype=torch.float32, device=device),
                "loss_drug_centered": torch.zeros((), dtype=torch.float32, device=device),
                "loss_cell_residual": torch.zeros((), dtype=torch.float32, device=device),
                "loss_rank_drug": torch.zeros((), dtype=torch.float32, device=device),
                "loss_same_drug_cross_cell_rank": torch.zeros((), dtype=torch.float32, device=device),
                "loss_adv": torch.zeros((), dtype=torch.float32, device=device),
                "loss_same_cell_cross_drug_rank": torch.zeros((), dtype=torch.float32, device=device),
                "loss_terminal_drug_specificity": torch.zeros((), dtype=torch.float32, device=device),
                "loss_graph_consistency": torch.zeros((), dtype=torch.float32, device=device),
            }
            train_count = 0
            rank_pairs_total = 0
            adv_pairs_total = 0
            terminal_drug_specificity_pairs_total = 0
            approx_pairs = max(benchmark.batch.n_cell_lines * benchmark.batch.n_drugs, 1)
            grouped_n_steps = max(int(np.ceil(len(train_df) / approx_pairs)), 1)
            if effective_sampler == "grouped_world_model":
                batch_iter = (("grouped_rank", batch_idx) for batch_idx in _iter_grouped_world_model_batches(
                    grouped_sampler_index,
                    n_steps=grouped_n_steps,
                    n_cell_lines=benchmark.batch.n_cell_lines,
                    n_drugs=benchmark.batch.n_drugs,
                    min_cell_drugs=benchmark.batch.min_cell_drugs_per_batch,
                    min_drug_cells=benchmark.batch.min_drug_cells_per_batch,
                    device=device,
                    seed=seed + epoch,
                    stats=sampler_stats,
                ))
            elif effective_sampler == "hybrid_drug_level":
                batch_iter = iter(
                    _iter_hybrid_drug_level_batches(
                        n_rows=len(train_tensors),
                        batch_size=batch_size,
                        grouped_sampler_index=grouped_sampler_index,
                        n_grouped_steps=grouped_n_steps,
                        rank_batch_fraction=benchmark.batch.rank_batch_fraction,
                        n_cell_lines=benchmark.batch.n_cell_lines,
                        n_drugs=benchmark.batch.n_drugs,
                        min_cell_drugs=benchmark.batch.min_cell_drugs_per_batch,
                        min_drug_cells=benchmark.batch.min_drug_cells_per_batch,
                        device=device,
                        seed=seed + epoch,
                        stats=sampler_stats,
                    )
                )
            else:
                batch_iter = (("random_pair", batch_idx) for batch_idx in _iter_batch_indices(len(train_tensors), batch_size, device, shuffle=True, seed=seed + epoch))
            while True:
                sampler_started_at = time.perf_counter()
                try:
                    batch_kind, batch_idx = next(batch_iter)
                except StopIteration:
                    break
                train_sampler_seconds += time.perf_counter() - sampler_started_at
                state_idx = train_tensors.state_idx.index_select(0, batch_idx)
                drug_idx = train_tensors.drug_idx.index_select(0, batch_idx)
                response = train_tensors.response.index_select(0, batch_idx)
                batch_n = int(batch_idx.shape[0])
                batch_count_total += 1.0
                batch_rows_total += float(batch_n)
                if uses_world_losses:
                    assert train_tensors.value_train is not None
                    assert train_tensors.drug_centered_train is not None
                    assert train_tensors.cell_residual_train is not None
                    assert train_tensors.drug_group_id is not None
                    assert train_tensors.cell_group_id is not None
                    value_train = train_tensors.value_train.index_select(0, batch_idx)
                    drug_centered_train = train_tensors.drug_centered_train.index_select(0, batch_idx)
                    cell_residual_train = train_tensors.cell_residual_train.index_select(0, batch_idx)
                    drug_group_id = train_tensors.drug_group_id.index_select(0, batch_idx)
                    cell_group_id = train_tensors.cell_group_id.index_select(0, batch_idx)
                    batch_unique_drugs_total += float(torch.unique(drug_group_id).numel())
                    batch_unique_cells_total += float(torch.unique(cell_group_id).numel())
                forward_started_at = time.perf_counter()
                batch_inputs = _build_forward_batch_inputs(
                    model=model,
                    state_bank=state_bank,
                    fp_bank=fp_bank,
                    prior_bank=prior_bank,
                    mask_bank=mask_bank,
                    state_idx=state_idx,
                    drug_idx=drug_idx,
                    model_drug_idx_bank=kg_drug_idx_bank,
                    use_kg_prior_path=use_kg_prior_path,
                )
                out = model_runner(
                    batch_inputs["state"],
                    batch_inputs["drug_fp"],
                    batch_inputs["prior"],
                    batch_inputs["prior_mask"],
                    use_terminal=use_terminal,
                    drug_idx=batch_inputs["drug_idx"],
                    edge_dropout=0.1,
                    compute_kg_consistency=graph_weight > 0.0 and benchmark.coverage_aware_kg_mask,
                    drug_latent=batch_inputs["drug_latent"],
                    drug_latent_bank=batch_inputs["drug_latent_bank"],
                    fusion_weight=0.0,
                    **model_kwargs,
                    return_internal_latents=terminal_drug_specificity_weight > 0.0,
                )
                train_forward_seconds += time.perf_counter() - forward_started_at
                loss_raw = F.smooth_l1_loss(out["raw_auc_base"], response)
                loss_absolute_auc = loss_raw
                loss_value = out["raw_auc_base"].new_zeros(())
                loss_drug_centered = out["raw_auc_base"].new_zeros(())
                loss_cell_residual = out["raw_auc_base"].new_zeros(())
                loss_rank_drug = out["raw_auc_base"].new_zeros(())
                loss_same_drug_cross_cell_rank = out["raw_auc_base"].new_zeros(())
                loss_adv = out["raw_auc_base"].new_zeros(())
                loss_same_cell_cross_drug_rank = out["raw_auc_base"].new_zeros(())
                loss_terminal_drug_specificity = out["raw_auc_base"].new_zeros(())
                loss_graph_consistency = out.get("kg_consistency", out["raw_auc_base"].new_zeros(())) if graph_weight > 0.0 and benchmark.coverage_aware_kg_mask else out["raw_auc_base"].new_zeros(())
                n_rank_pairs = 0
                n_adv_pairs = 0
                n_terminal_drug_specificity_pairs = 0
                if uses_world_losses:
                    pairwise_started_at = time.perf_counter()
                    valid_value_mask = torch.isfinite(value_train)
                    if value_weight > 0.0 and bool(valid_value_mask.any().item()):
                        loss_value = F.smooth_l1_loss(out["value_hat"][valid_value_mask], value_train[valid_value_mask])
                    valid_centered_mask = torch.isfinite(drug_centered_train)
                    if centered_weight > 0.0 and bool(valid_centered_mask.any().item()):
                        loss_drug_centered = F.smooth_l1_loss(out["drug_centered_hat"][valid_centered_mask], drug_centered_train[valid_centered_mask])
                    valid_cell_residual_mask = torch.isfinite(cell_residual_train)
                    if cell_residual_weight > 0.0 and bool(valid_cell_residual_mask.any().item()):
                        loss_cell_residual = F.smooth_l1_loss(out["cell_residual_hat"][valid_cell_residual_mask], cell_residual_train[valid_cell_residual_mask])
                    rank_scores = out["value_hat"]
                    rank_target = value_train
                    rank_allowed = effective_sampler != "hybrid_drug_level" or batch_kind == "grouped_rank"
                    if rank_weight > 0.0 and rank_allowed:
                        loss_rank_drug, n_rank_pairs = pairwise_loss_runner(
                            rank_scores,
                            rank_target,
                            response,
                            drug_group_id,
                            min_target_gap=benchmark.training_objectives.min_relative_value_gap,
                            min_response_gap=benchmark.training_objectives.min_auc_gap,
                            margin=benchmark.training_objectives.rank_margin,
                            workspace=pairwise_workspace,
                        )
                        loss_same_drug_cross_cell_rank = loss_rank_drug
                    if same_cell_rank_weight > 0.0 and rank_allowed:
                        loss_same_cell_cross_drug_rank, n_adv_pairs = pairwise_loss_runner(
                            out["cell_residual_hat"],
                            cell_residual_train,
                            response,
                            cell_group_id,
                            min_target_gap=benchmark.training_objectives.min_relative_value_gap,
                            min_response_gap=benchmark.training_objectives.min_auc_gap,
                            margin=benchmark.training_objectives.advantage_margin,
                            workspace=pairwise_workspace,
                        )
                        loss_adv = loss_same_cell_cross_drug_rank
                    if terminal_drug_specificity_weight > 0.0 and "terminal_latent" in out and "state_latent" in out:
                        loss_terminal_drug_specificity, n_terminal_drug_specificity_pairs = _terminal_drug_specificity_loss(
                            out["terminal_latent"],
                            out["state_latent"],
                            cell_group_id,
                            drug_group_id,
                            margin=terminal_drug_specificity_margin,
                            workspace=pairwise_workspace,
                        )
                    rank_group_total += float(torch.unique(drug_group_id[torch.isfinite(value_train)]).numel())
                    adv_group_total += float(torch.unique(cell_group_id[torch.isfinite(value_train)]).numel())
                    rank_pairs_obs_total += float(n_rank_pairs)
                    adv_pairs_obs_total += float(n_adv_pairs)
                    train_pairwise_seconds += time.perf_counter() - pairwise_started_at
                loss = (
                    benchmark.training_objectives.loss_raw_weight * loss_absolute_auc
                    + value_weight * loss_value
                    + cell_residual_weight * loss_cell_residual
                    + rank_weight * loss_rank_drug
                    + same_cell_rank_weight * loss_same_cell_cross_drug_rank
                    + terminal_drug_specificity_weight * loss_terminal_drug_specificity
                    + graph_weight * loss_graph_consistency
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                train_sums["loss_total"] += loss.detach() * batch_n
                train_sums["loss_raw"] += loss_raw.detach() * batch_n
                train_sums["loss_absolute_auc"] += loss_absolute_auc.detach() * batch_n
                train_sums["loss_value"] += loss_value.detach() * batch_n
                train_sums["loss_drug_centered"] += loss_drug_centered.detach() * batch_n
                train_sums["loss_cell_residual"] += loss_cell_residual.detach() * batch_n
                train_sums["loss_rank_drug"] += loss_rank_drug.detach() * batch_n
                train_sums["loss_same_drug_cross_cell_rank"] += loss_same_drug_cross_cell_rank.detach() * batch_n
                train_sums["loss_adv"] += loss_adv.detach() * batch_n
                train_sums["loss_same_cell_cross_drug_rank"] += loss_same_cell_cross_drug_rank.detach() * batch_n
                train_sums["loss_terminal_drug_specificity"] += loss_terminal_drug_specificity.detach() * batch_n
                train_sums["loss_graph_consistency"] += loss_graph_consistency.detach() * batch_n
                train_count += batch_n
                rank_pairs_total += n_rank_pairs
            adv_pairs_total += n_adv_pairs
            terminal_drug_specificity_pairs_total += n_terminal_drug_specificity_pairs
        model.eval()
        val_started_at = time.perf_counter()
        val_raw_sum = torch.zeros((), dtype=torch.float32, device=device)
        val_value_sum = torch.zeros((), dtype=torch.float32, device=device)
        val_centered_sum = torch.zeros((), dtype=torch.float32, device=device)
        val_cell_residual_sum = torch.zeros((), dtype=torch.float32, device=device)
        val_value_count = 0
        val_centered_count = 0
        val_cell_residual_count = 0
        val_count = 0
        val_raw_chunks: list[np.ndarray] = []
        val_value_chunks: list[np.ndarray] = []
        val_centered_chunks: list[np.ndarray] = []
        val_cell_residual_chunks: list[np.ndarray] = []
        val_kg_payload = _maybe_precompute_kg_payload(
            model,
            device=device,
            controls=controls,
            edge_dropout=0.0,
            context=context,
        )
        with torch.no_grad():
            for batch_idx in _iter_batch_indices(len(val_tensors), batch_size, device, shuffle=False, seed=seed):
                state_idx = val_tensors.state_idx.index_select(0, batch_idx)
                drug_idx = val_tensors.drug_idx.index_select(0, batch_idx)
                response = val_tensors.response.index_select(0, batch_idx)
                batch_inputs = _build_forward_batch_inputs(
                    model=model,
                    state_bank=state_bank,
                    fp_bank=fp_bank,
                    prior_bank=prior_bank,
                    mask_bank=mask_bank,
                    state_idx=state_idx,
                    drug_idx=drug_idx,
                    model_drug_idx_bank=kg_drug_idx_bank,
                    use_kg_prior_path=use_kg_prior_path,
                    precomputed_kg_payload=val_kg_payload,
                )
                out = model_runner(
                    batch_inputs["state"],
                    batch_inputs["drug_fp"],
                    batch_inputs["prior"],
                    batch_inputs["prior_mask"],
                    use_terminal=use_terminal,
                    drug_idx=batch_inputs["drug_idx"],
                    compute_kg_consistency=False,
                    precomputed_kg_payload=val_kg_payload,
                    drug_latent=batch_inputs["drug_latent"],
                    drug_latent_bank=batch_inputs["drug_latent_bank"],
                    fusion_weight=0.0,
                    **model_kwargs,
                )
                batch_loss_raw = F.smooth_l1_loss(out["raw_auc_base"], response)
                if uses_world_losses:
                    assert val_tensors.value_train is not None
                    value_train = val_tensors.value_train.index_select(0, batch_idx)
                    valid_value_mask = torch.isfinite(value_train)
                    if value_weight > 0.0 and bool(valid_value_mask.any().item()):
                        batch_value_loss = F.smooth_l1_loss(out["value_hat"][valid_value_mask], value_train[valid_value_mask])
                        val_value_sum += batch_value_loss.detach() * int(valid_value_mask.sum().item())
                        val_value_count += int(valid_value_mask.sum().item())
                    if val_tensors.drug_centered_train is not None and centered_weight > 0.0:
                        drug_centered_train = val_tensors.drug_centered_train.index_select(0, batch_idx)
                        valid_centered_mask = torch.isfinite(drug_centered_train)
                        if bool(valid_centered_mask.any().item()):
                            batch_centered_loss = F.smooth_l1_loss(
                                out["drug_centered_hat"][valid_centered_mask],
                                drug_centered_train[valid_centered_mask],
                            )
                            val_centered_sum += batch_centered_loss.detach() * int(valid_centered_mask.sum().item())
                            val_centered_count += int(valid_centered_mask.sum().item())
                    if val_tensors.cell_residual_train is not None and cell_residual_weight > 0.0:
                        cell_residual_train = val_tensors.cell_residual_train.index_select(0, batch_idx)
                        valid_cell_residual_mask = torch.isfinite(cell_residual_train)
                        if bool(valid_cell_residual_mask.any().item()):
                            batch_cell_residual_loss = F.smooth_l1_loss(
                                out["cell_residual_hat"][valid_cell_residual_mask],
                                cell_residual_train[valid_cell_residual_mask],
                            )
                            val_cell_residual_sum += batch_cell_residual_loss.detach() * int(valid_cell_residual_mask.sum().item())
                            val_cell_residual_count += int(valid_cell_residual_mask.sum().item())
                batch_n = int(batch_idx.shape[0])
                val_raw_sum += batch_loss_raw * batch_n
                val_count += batch_n
                val_raw_chunks.append(out["raw_auc_base"].detach().cpu().numpy())
                val_value_chunks.append(out["value_hat"].detach().cpu().numpy())
                val_centered_chunks.append(out["drug_centered_hat"].detach().cpu().numpy())
                val_cell_residual_chunks.append(out["cell_residual_hat"].detach().cpu().numpy())
        val_seconds_total += time.perf_counter() - val_started_at
        train_loss = float((train_sums["loss_total"] / max(train_count, 1)).item())
        val_raw_loss = float((val_raw_sum / val_count).item()) if val_count else float("nan")
        val_value_loss = float((val_value_sum / val_value_count).item()) if val_value_count else float("nan")
        val_drug_centered_loss = float((val_centered_sum / val_centered_count).item()) if val_centered_count else float("nan")
        val_cell_residual_loss = float((val_cell_residual_sum / val_cell_residual_count).item()) if val_cell_residual_count else float("nan")
        selected_fusion_weight = 0.0
        val_within_drug_pcc = float("nan")
        val_fused_loss = val_raw_loss
        if val_count:
            val_pred = val_df.reset_index(drop=True).copy()
            val_pred["raw_auc_base"] = np.concatenate(val_raw_chunks, axis=0)
            val_pred["value_hat"] = np.concatenate(val_value_chunks, axis=0)
            val_pred["drug_centered_hat"] = np.concatenate(val_centered_chunks, axis=0)
            val_pred["cell_residual_hat"] = np.concatenate(val_cell_residual_chunks, axis=0)
            if mode == "drug_residual_world_model":
                selected = select_best_fusion_weight(
                    val_pred,
                    candidates=fusion_candidates,
                    split="val",
                    metric=fusion_metric,
                )
                selected_fusion_weight = float(selected["selected_weight"])
                val_within_drug_pcc = float(selected["selected_metric_value"])
                val_fused = with_fused_auc(val_pred, selected_fusion_weight)
                val_fused_loss = float(F.smooth_l1_loss(
                    torch.as_tensor(val_fused["auc_hat"].to_numpy(np.float32), dtype=torch.float32),
                    torch.as_tensor(val_fused[response_field].astype(float).to_numpy(np.float32), dtype=torch.float32),
                ).item())
            else:
                val_pred["auc_hat"] = val_pred["raw_auc_base"].astype(np.float32)
                val_within_drug_pcc = float("nan")
                val_fused_loss = val_raw_loss
        history_row = {
            "epoch": epoch,
            "mode": mode,
            "train_loss": train_loss,
            "loss_total": train_loss,
            "loss_raw": float((train_sums["loss_raw"] / max(train_count, 1)).item()),
            "loss_absolute_auc": float((train_sums["loss_absolute_auc"] / max(train_count, 1)).item()),
            "loss_value": float((train_sums["loss_value"] / max(train_count, 1)).item()),
            "loss_drug_centered": float((train_sums["loss_drug_centered"] / max(train_count, 1)).item()),
            "loss_cell_residual": float((train_sums["loss_cell_residual"] / max(train_count, 1)).item()),
            "loss_rank_drug": float((train_sums["loss_rank_drug"] / max(train_count, 1)).item()),
            "loss_same_drug_cross_cell_rank": float((train_sums["loss_same_drug_cross_cell_rank"] / max(train_count, 1)).item()),
            "loss_adv": float((train_sums["loss_adv"] / max(train_count, 1)).item()),
            "loss_same_cell_cross_drug_rank": float((train_sums["loss_same_cell_cross_drug_rank"] / max(train_count, 1)).item()),
            "loss_terminal_drug_specificity": float((train_sums["loss_terminal_drug_specificity"] / max(train_count, 1)).item()),
            "loss_graph_consistency": float((train_sums["loss_graph_consistency"] / max(train_count, 1)).item()),
            "n_valid_rank_pairs": int(rank_pairs_total),
            "n_valid_adv_pairs": int(adv_pairs_total),
            "n_valid_terminal_drug_specificity_pairs": int(terminal_drug_specificity_pairs_total),
            "val_loss": val_fused_loss,
            "val_raw_loss": val_raw_loss,
            "val_fused_loss": val_fused_loss,
            "val_value_loss": val_value_loss,
            "val_drug_centered_loss": val_drug_centered_loss,
            "val_cell_residual_loss": val_cell_residual_loss,
            "val_within_drug_pcc": val_within_drug_pcc,
            "val_auc_mse": val_raw_loss,
            "selected_fusion_weight": float(selected_fusion_weight),
        }
        val_score, objective_used = _validation_objective_score(validation_objective, history_row)
        history_row["validation_objective"] = validation_objective
        history_row["validation_objective_used"] = objective_used
        history_row["validation_score"] = float(val_score)
        history.append(history_row)
        if val_score > best["score"]:
            best = {
                "score": float(val_score),
                "epoch": int(epoch),
                "selected_fusion_weight": float(selected_fusion_weight),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    train_seconds_total = time.perf_counter() - train_started_at
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    model_arch = architecture_dict(model, prepared.state_matrix.shape[1], len(prepared.artifacts.prior_columns))
    model_arch.update(
        {
            "mode": mode,
            "target_field": response_field,
            "terminal_consequence_enabled": bool(controls.use_terminal and benchmark.world_model.terminal_consequence_enabled),
            "world_model": benchmark.world_model.to_dict(),
            "training_objectives": benchmark.training_objectives.to_dict(),
            "batch": benchmark.batch.to_dict(),
        }
    )
    write_json(out_dir / "model_architecture.json", model_arch)
    with (out_dir / "artifacts.pkl").open("wb") as f:
        pickle.dump(prepared.artifacts, f)
    write_json(
        out_dir / "gpu_runtime.json",
        {
            "device": str(device),
            "mode": mode,
            "controls": controls.to_dict(),
            "world_model": benchmark.world_model.to_dict(),
            "training_objectives": benchmark.training_objectives.to_dict(),
            "batch_config": benchmark.batch.to_dict(),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "batch_size": int(batch_size),
            "validation_objective": validation_objective,
            "fusion_selection_metric": fusion_metric,
            "fusion_weight_candidates": list(fusion_candidates),
            "selected_fusion_weight": float(best["selected_fusion_weight"]),
            "best_validation_epoch": int(best["epoch"]),
            "train_seconds_total": float(train_seconds_total),
            "train_seconds_sampler": float(train_sampler_seconds),
            "train_seconds_forward": float(train_forward_seconds),
            "train_seconds_pairwise": float(train_pairwise_seconds),
            "val_seconds_total": float(val_seconds_total),
            "train_batch_count": int(batch_count_total),
            "train_batch_rows_mean": float(batch_rows_total / max(batch_count_total, 1.0)),
            "train_batch_unique_drugs_mean": float(batch_unique_drugs_total / max(batch_count_total, 1.0)),
            "train_batch_unique_cells_mean": float(batch_unique_cells_total / max(batch_count_total, 1.0)),
            "train_rank_group_count_mean": float(rank_group_total / max(batch_count_total, 1.0)),
            "train_adv_group_count_mean": float(adv_group_total / max(batch_count_total, 1.0)),
            "train_rank_pairs_mean": float(rank_pairs_obs_total / max(batch_count_total, 1.0)),
            "train_adv_pairs_mean": float(adv_pairs_obs_total / max(batch_count_total, 1.0)),
            "train_sampler_attempts_mean": float(sampler_stats.get("attempts_total", 0.0) / max(sampler_stats.get("batches", 0.0), 1.0)),
            **runtime_meta,
        },
    )
    return model


def predict_frame(
    model: TerminalWorldModel,
    frame: pd.DataFrame,
    prepared: PreparedData,
    batch_size: int = 4096,
    device: str | None = None,
    benchmark: BenchmarkConfig | None = None,
    controls: AblationControls | None = None,
    context: TrainExecutionContext | None = None,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
    eval_compile: bool = False,
    *,
    include_planning_frame: bool = False,
    terminal_latent_path: Path | None = None,
    gate_audit_path: Path | None = None,
    explain_dir: Path | None = None,
    kg_mask: Any | None = None,
    session: PredictionSession | None = None,
) -> PredictionOutputs:
    prepared = _ensure_prepared_compat(prepared)
    if device is None:
        raise ValueError("Explicit --device is required for evaluate/run. Use --device cuda:N or --device cpu.")
    if session is None:
        session = _build_prediction_session(
            model,
            prepared,
            device=device,
            benchmark=benchmark,
            controls=controls,
            context=context,
            runtime_profile=runtime_profile,
            eval_compile=eval_compile,
            explain_dir=explain_dir,
        )
    runtime_meta = session.runtime_meta
    model = session.model
    model_runner = session.model_runner
    context = session.context
    tensor_banks = session.tensor_banks
    benchmark = session.benchmark
    controls = session.controls
    use_kg_prior_path = session.use_kg_prior_path
    kg_drug_idx_bank = session.kg_drug_idx_bank
    state_bank = session.state_bank
    fp_bank = session.fp_bank
    prior_bank = session.prior_bank
    mask_bank = session.mask_bank
    use_terminal = session.use_terminal
    model_kwargs = session.model_kwargs
    precomputed_kg_payload = session.precomputed_kg_payload
    tensors = _cached_eval_tensors(frame, context.eval_tensor_cache, tensor_banks, device)
    if gate_audit_path is not None:
        alpha_path = gate_audit_path.with_name("kg_attention_by_prediction.csv")
        if alpha_path.exists():
            alpha_path.unlink()
    source_path = None if explain_dir is None else Path(explain_dir) / "kg_source_contribution_by_prediction.csv"
    edge_path = None if explain_dir is None else Path(explain_dir) / "kg_top_edges_by_prediction.csv"
    node_path = None if explain_dir is None else Path(explain_dir) / "kg_top_nodes_by_prediction.csv"
    source_header = True
    edge_header = True
    node_header = True
    explain_branch_node_states = None
    if explain_enabled := bool(benchmark.explainability.enabled and explain_dir is not None):
        if precomputed_kg_payload is not None and "branch_node_states" in precomputed_kg_payload:
            explain_branch_node_states = tensor_to_numpy(precomputed_kg_payload["branch_node_states"])
    planning = frame.reset_index(drop=True).copy() if include_planning_frame else None
    pred_cols = [
        "SANGER_MODEL_ID",
        "DRUG_ID",
        "DRUG_NAME",
        "drug_key",
        "smiles",
        "canonical_smiles",
        "inchikey",
        "canonical_group_key",
        "canonical_group_index",
        "canonical_group_size",
        "canonical_drug_id",
        "split",
        "AUC",
        "relative_value",
        benchmark.world_model.relative_value_train_field,
        benchmark.world_model.relative_value_eval_field,
        "cell_train_baseline",
        "cell_residual_baseline_train",
        "cell_residual_auc_train",
        "cell_residual_baseline_eval",
        "cell_residual_auc_eval",
        "drug_centered_baseline_train",
        "drug_centered_auc_train",
        "drug_centered_baseline_eval",
        "drug_centered_auc_eval",
    ]
    pred = frame.reindex(columns=pred_cols).reset_index(drop=True).copy()
    n_rows = len(pred)
    auc_hat = np.empty((n_rows,), dtype=np.float32)
    raw_auc_base = np.empty((n_rows,), dtype=np.float32)
    value_hat = np.empty((n_rows,), dtype=np.float32)
    drug_centered_hat = np.empty((n_rows,), dtype=np.float32)
    cell_residual_hat = np.empty((n_rows,), dtype=np.float32)
    uncertainty = np.empty((n_rows,), dtype=np.float32)
    latent_writer = open_memmap(terminal_latent_path, mode="w+", dtype=np.float32, shape=(n_rows, 128)) if terminal_latent_path else None
    gate_header = True
    alpha_header = True
    eval_forward_seconds = 0.0
    explain_postprocess_seconds = 0.0
    eval_started_at = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            state_idx = tensors.state_idx[start:stop]
            drug_idx = tensors.drug_idx[start:stop]
            batch_inputs = _build_forward_batch_inputs(
                model=model,
                state_bank=state_bank,
                fp_bank=fp_bank,
                prior_bank=prior_bank,
                mask_bank=mask_bank,
                state_idx=state_idx,
                drug_idx=drug_idx,
                model_drug_idx_bank=kg_drug_idx_bank,
                use_kg_prior_path=use_kg_prior_path,
                precomputed_kg_payload=precomputed_kg_payload,
            )
            forward_started_at = time.perf_counter()
            out = model_runner(
                batch_inputs["state"],
                batch_inputs["drug_fp"],
                batch_inputs["prior"],
                batch_inputs["prior_mask"],
                use_terminal=use_terminal,
                drug_idx=batch_inputs["drug_idx"],
                compute_kg_consistency=False,
                precomputed_kg_payload=precomputed_kg_payload,
                drug_latent=batch_inputs["drug_latent"],
                drug_latent_bank=batch_inputs["drug_latent_bank"],
                fusion_weight=0.0,
                return_explanations=explain_enabled,
                explanation_level=benchmark.explainability.level,
                kg_mask=kg_mask,
                **model_kwargs,
            )
            eval_forward_seconds += time.perf_counter() - forward_started_at
            auc_hat[start:stop] = out["auc_hat"].detach().cpu().numpy()
            raw_auc_base[start:stop] = out["raw_auc_base"].detach().cpu().numpy()
            value_hat[start:stop] = out["value_hat"].detach().cpu().numpy()
            drug_centered_hat[start:stop] = out["drug_centered_hat"].detach().cpu().numpy()
            cell_residual_hat[start:stop] = out["cell_residual_hat"].detach().cpu().numpy()
            uncertainty[start:stop] = out["uncertainty"].detach().cpu().numpy()
            if latent_writer is not None:
                latent_writer[start:stop] = out["terminal_latent"].detach().cpu().numpy()
            if gate_audit_path is not None:
                gate_values = out["gate"].detach().cpu().numpy()
                gate_chunk = pd.DataFrame(gate_values)
                gate_chunk.insert(0, "DRUG_ID", pred["DRUG_ID"].iloc[start:stop].to_numpy())
                gate_chunk.insert(0, "SANGER_MODEL_ID", pred["SANGER_MODEL_ID"].iloc[start:stop].to_numpy())
                gate_chunk.to_csv(gate_audit_path, mode="a", header=gate_header, index=False)
                gate_header = False
            if gate_audit_path is not None and "kg_alpha" in out:
                alpha = out["kg_alpha"].detach().cpu().numpy()
                alpha_chunk = pd.DataFrame(alpha, columns=["alpha_ChEMBL", "alpha_DRKG", "alpha_PrimeKG"][: alpha.shape[1]])
                alpha_chunk.insert(0, "DRUG_ID", pred["DRUG_ID"].iloc[start:stop].to_numpy())
                alpha_chunk.insert(0, "SANGER_MODEL_ID", pred["SANGER_MODEL_ID"].iloc[start:stop].to_numpy())
                alpha_path = gate_audit_path.with_name("kg_attention_by_prediction.csv")
                alpha_chunk.to_csv(alpha_path, mode="a", header=alpha_header, index=False)
                alpha_header = False
            if explain_enabled and "explanations" in out:
                explain_started_at = time.perf_counter()
                batch_frame = pred.iloc[start:stop].copy()
                batch_frame.index = np.arange(start, stop, dtype=np.int64)
                bundle = out["explanations"]
                source_attention = tensor_to_numpy(bundle.kg_source_attention)
                gate_values = tensor_to_numpy(bundle.kg_gate)
                total_contribution = tensor_to_numpy(bundle.kg_total_contribution)
                source_names = list(bundle.kg_source_names or [])
                sample_queries = tensor_to_numpy(bundle.extra.get("kg_explain_query"))
                if explain_branch_node_states is None and bundle.extra.get("kg_branch_node_states") is not None:
                    explain_branch_node_states = tensor_to_numpy(bundle.extra.get("kg_branch_node_states"))
                source_rows = build_source_rows(
                    batch_frame,
                    source_attention=source_attention,
                    source_names=source_names,
                    gate_values=gate_values,
                    total_contribution=total_contribution,
                )
                edge_rows = build_edge_rows(
                    batch_frame,
                    kg_graph=getattr(prepared.artifacts, "kg_graph", None),
                    source_attention=source_attention,
                    source_names=source_names,
                    top_k_edges=benchmark.explainability.top_k_edges,
                    sample_queries=sample_queries,
                    branch_node_states=explain_branch_node_states,
                )
                node_rows = build_node_rows(edge_rows, top_k_nodes=benchmark.explainability.top_k_nodes)
                if source_path is not None:
                    source_header = _append_csv_frame(source_path, source_rows, header=source_header)
                if edge_path is not None:
                    edge_header = _append_csv_frame(edge_path, edge_rows, header=edge_header)
                if node_path is not None:
                    node_header = _append_csv_frame(node_path, node_rows, header=node_header)
                explain_branch_node_states = None
                explain_postprocess_seconds += time.perf_counter() - explain_started_at
    pred["auc_hat"] = auc_hat
    pred["raw_auc_base"] = raw_auc_base
    pred["value_hat"] = value_hat
    pred["drug_centered_hat"] = drug_centered_hat
    pred["cell_residual_hat"] = cell_residual_hat
    pred["uncertainty"] = uncertainty
    explainability_outputs = None
    if explain_enabled:
        explainability_outputs = {
            "source": _read_csv_or_empty(source_path) if source_path is not None else pd.DataFrame(),
            "edge": _read_csv_or_empty(edge_path) if edge_path is not None else pd.DataFrame(),
            "node": _read_csv_or_empty(node_path) if node_path is not None else pd.DataFrame(),
        }
    return PredictionOutputs(
        core=pred,
        planning=planning,
        explainability=explainability_outputs,
        runtime={
            "runtime_profile": str(runtime_meta.get("runtime_profile", runtime_profile)),
            "eval_compile_requested": bool(runtime_meta.get("eval_compile_requested", False)),
            "eval_compile_enabled": bool(runtime_meta.get("eval_compile_enabled", False)),
            "eval_compile_targets": list(runtime_meta.get("eval_compile_targets", [])),
            "eval_compile_fallback_reason": str(runtime_meta.get("eval_compile_fallback_reason", "")),
            "eval_seconds_total": float(time.perf_counter() - eval_started_at),
            "eval_seconds_model_forward": float(eval_forward_seconds),
            "eval_seconds_explain_postprocess": float(explain_postprocess_seconds),
        },
    )


def _run_prediction_ablation(
    model: TerminalWorldModel,
    frame: pd.DataFrame,
    prepared: PreparedData,
    *,
    batch_size: int,
    device: str,
    benchmark: BenchmarkConfig,
    controls: AblationControls | None,
    context: TrainExecutionContext | None,
    runtime_profile: str,
    eval_compile: bool,
    kg_mask: Any,
    chunk_size: int | None = None,
    session: PredictionSession | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    effective_chunk_size = None if chunk_size is None else max(1, int(chunk_size))
    prediction_index = frame.index.to_numpy(dtype=np.int64, copy=False)
    if effective_chunk_size is None or len(frame) <= effective_chunk_size:
        outputs = predict_frame(
            model,
            frame,
            prepared,
            batch_size=batch_size,
            device=device,
            benchmark=benchmark,
            controls=controls,
            context=context,
            runtime_profile=runtime_profile,
            eval_compile=eval_compile,
            kg_mask=kg_mask,
            session=session,
        )
        out = outputs.core.copy()
        out["prediction_index"] = prediction_index
        return out
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(frame), effective_chunk_size):
        chunk = frame.iloc[start : start + effective_chunk_size]
        outputs = predict_frame(
            model,
            chunk,
            prepared,
            batch_size=min(batch_size, max(len(chunk), 1)),
            device=device,
            benchmark=benchmark,
            controls=controls,
            context=context,
            runtime_profile=runtime_profile,
            eval_compile=eval_compile,
            kg_mask=kg_mask,
            session=session,
        )
        out = outputs.core.copy()
        out["prediction_index"] = prediction_index[start : start + len(chunk)]
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _build_source_ablation_rows(
    pred: pd.DataFrame,
    masked: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    base = pred[["SANGER_MODEL_ID", "DRUG_ID", "auc_hat", "raw_auc_base", "value_hat"]].copy().reset_index(drop=True)
    base.insert(0, "prediction_index", np.arange(len(base), dtype=np.int64))
    masked_aligned = (
        masked.loc[:, ["prediction_index", "auc_hat", "raw_auc_base", "value_hat"]]
        .sort_values("prediction_index")
        .set_index("prediction_index")
        .reindex(base["prediction_index"].to_numpy())
    )
    base.insert(0, "source_name", str(source_name))
    base["masked_auc_hat"] = masked_aligned["auc_hat"].to_numpy()
    base["masked_raw_auc_base"] = masked_aligned["raw_auc_base"].to_numpy()
    base["masked_value_hat"] = masked_aligned["value_hat"].to_numpy()
    base["delta_auc_hat"] = base["auc_hat"] - base["masked_auc_hat"]
    base["delta_raw_auc_base"] = base["raw_auc_base"] - base["masked_raw_auc_base"]
    base["delta_value_hat"] = base["value_hat"] - base["masked_value_hat"]
    return base


def _build_edge_ablation_rows(
    subset: pd.DataFrame,
    pred: pd.DataFrame,
    edge_group: pd.DataFrame,
    masked: pd.DataFrame,
    *,
    edge_id: int,
) -> pd.DataFrame:
    prediction_index = edge_group["prediction_index"].astype(int).to_numpy()
    base = subset[["SANGER_MODEL_ID", "DRUG_ID", "AUC"]].copy().reset_index(drop=True)
    base["prediction_index"] = prediction_index
    base["edge_id"] = int(edge_id)
    base["source_name"] = edge_group["source_name"].astype(str).to_numpy()
    base["edge_type"] = edge_group["edge_type"].astype(str).to_numpy()
    base["auc_hat"] = pred.loc[prediction_index, "auc_hat"].to_numpy()
    base["raw_auc_base"] = pred.loc[prediction_index, "raw_auc_base"].to_numpy()
    base["value_hat"] = pred.loc[prediction_index, "value_hat"].to_numpy()
    masked_aligned = masked.sort_values("prediction_index").set_index("prediction_index").reindex(prediction_index)
    base["masked_auc_hat"] = masked_aligned["auc_hat"].to_numpy()
    base["masked_raw_auc_base"] = masked_aligned["raw_auc_base"].to_numpy()
    base["masked_value_hat"] = masked_aligned["value_hat"].to_numpy()
    base["delta_auc_hat"] = base["auc_hat"] - base["masked_auc_hat"]
    base["delta_raw_auc_base"] = base["raw_auc_base"] - base["masked_raw_auc_base"]
    base["delta_value_hat"] = base["value_hat"] - base["masked_value_hat"]
    return base


def _build_node_ablation_row(
    *,
    pred: pd.DataFrame,
    prediction_index: int,
    node_row: pd.Series,
    related_edges: pd.DataFrame,
    masked: pd.DataFrame,
) -> dict[str, Any]:
    masked_row = masked.sort_values("prediction_index").iloc[0]
    base_row = pred.loc[int(prediction_index)]
    return {
        "prediction_index": int(prediction_index),
        "SANGER_MODEL_ID": base_row.get("SANGER_MODEL_ID"),
        "DRUG_ID": int(base_row.get("DRUG_ID")),
        "DRUG_NAME": base_row.get("DRUG_NAME"),
        "node_id": int(node_row["node_id"]),
        "node_name": node_row.get("node_name"),
        "node_type": node_row.get("node_type"),
        "source_name": "|".join(sorted(related_edges["source_name"].dropna().astype(str).unique().tolist())),
        "edge_ids": json.dumps(sorted(related_edges["edge_id"].astype(int).unique().tolist())),
        "n_edges_masked": int(related_edges["edge_id"].astype(int).nunique()),
        "attention_score": float(node_row.get("node_attention", float("nan"))),
        "auc_hat": float(base_row.get("auc_hat", float("nan"))),
        "raw_auc_base": float(base_row.get("raw_auc_base", float("nan"))),
        "value_hat": float(base_row.get("value_hat", float("nan"))),
        "masked_auc_hat": float(masked_row["auc_hat"]),
        "masked_raw_auc_base": float(masked_row["raw_auc_base"]),
        "masked_value_hat": float(masked_row["value_hat"]),
        "delta_auc_hat": float(base_row.get("auc_hat", float("nan"))) - float(masked_row["auc_hat"]),
        "delta_raw_auc_base": float(base_row.get("raw_auc_base", float("nan"))) - float(masked_row["raw_auc_base"]),
        "delta_value_hat": float(base_row.get("value_hat", float("nan"))) - float(masked_row["value_hat"]),
    }


def _build_node_ablation_tasks(
    *,
    top_nodes: pd.DataFrame,
    edge_rows: pd.DataFrame,
    allowed_node_types: set[str],
    node_top_k: int,
    max_edges_per_action: int,
) -> list[dict[str, Any]]:
    if top_nodes.empty or edge_rows.empty:
        return []
    selected_nodes = top_nodes.loc[
        top_nodes["node_type"].astype(str).isin(allowed_node_types)
    ].sort_values(["prediction_index", "node_rank"], ascending=[True, True])
    selected_nodes = selected_nodes.groupby("prediction_index", observed=True).head(int(node_top_k)).copy()
    tasks: list[dict[str, Any]] = []
    for node_row in selected_nodes.to_dict("records"):
        prediction_index = int(node_row["prediction_index"])
        node_id = int(node_row["node_id"])
        related_edges = edge_rows.loc[
            edge_rows["prediction_index"].eq(prediction_index)
            & (
                edge_rows["src_node_id"].eq(node_id)
                | edge_rows["dst_node_id"].eq(node_id)
            )
        ].sort_values("edge_attention", ascending=False)
        if related_edges.empty:
            continue
        related_edges = related_edges.head(int(max_edges_per_action)).copy()
        edge_ids = sorted(related_edges["edge_id"].astype(int).unique().tolist())
        tasks.append(
            {
                "prediction_index": prediction_index,
                "node_row": node_row,
                "related_edges": related_edges,
                "edge_ids": edge_ids,
            }
        )
    return tasks


def _postprocess_ablation_predictions(
    pred: pd.DataFrame,
    *,
    mode: str,
    selected_fusion_weight: float,
) -> pd.DataFrame:
    if str(mode).strip().lower() == "drug_residual_world_model":
        return with_fused_auc(pred, selected_fusion_weight)
    return pred


def _build_explanation_coverage_audits(
    pred: pd.DataFrame,
    source_rows: pd.DataFrame,
    edge_rows: pd.DataFrame,
    path_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = pred[["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "split"]].copy().reset_index(drop=True)
    coverage["prediction_index"] = np.arange(len(coverage), dtype=np.int64)
    src_idx = set(source_rows["prediction_index"].astype(int).tolist()) if not source_rows.empty else set()
    edge_idx = set(edge_rows["prediction_index"].astype(int).tolist()) if not edge_rows.empty else set()
    path_idx = set(path_rows["prediction_index"].astype(int).tolist()) if not path_rows.empty else set()
    coverage["has_source_explanation"] = coverage["prediction_index"].isin(src_idx)
    coverage["has_edge_explanation"] = coverage["prediction_index"].isin(edge_idx)
    coverage["has_path_explanation"] = coverage["prediction_index"].isin(path_idx)
    coverage["missing_edge_explanation"] = ~coverage["has_edge_explanation"]
    coverage["missing_path_explanation"] = ~coverage["has_path_explanation"]
    drug_coverage = (
        coverage.groupby(["DRUG_ID", "DRUG_NAME"], observed=True)
        .agg(
            n_predictions=("prediction_index", "size"),
            n_with_source=("has_source_explanation", "sum"),
            n_with_edge=("has_edge_explanation", "sum"),
            n_with_path=("has_path_explanation", "sum"),
        )
        .reset_index()
    )
    drug_coverage["edge_coverage_fraction"] = drug_coverage["n_with_edge"] / drug_coverage["n_predictions"].clip(lower=1)
    drug_coverage["path_coverage_fraction"] = drug_coverage["n_with_path"] / drug_coverage["n_predictions"].clip(lower=1)
    return coverage, drug_coverage


def _rank_with_controls(pred: pd.DataFrame, controls: AblationControls, top_k: int) -> pd.DataFrame:
    if controls.policy_mode == "random":
        rng = np.random.default_rng(controls.seed)
        frame = pred.copy()
        frame["planner_score"] = rng.random(len(frame))
        frame["planner_rank"] = frame.groupby("entity_id")["planner_score"].rank(
            method="first",
            ascending=False,
            na_option="bottom",
        ).astype(int)
        return frame.sort_values(["entity_id", "planner_rank"])
    ranked = rank_candidates(pred, lambda_u=0.1)
    if controls.policy_mode == "drug_only":
        drug_mean = ranked.groupby("DRUG_ID", observed=True)["value_hat"].mean()
        ranked["planner_score"] = ranked["DRUG_ID"].map(drug_mean).astype(float)
        ranked["planner_rank"] = ranked.groupby("entity_id")["planner_score"].rank(
            method="first",
            ascending=False,
            na_option="bottom",
        ).astype(int)
        ranked = ranked.sort_values(["entity_id", "planner_rank"])
    ranked.attrs["top_k"] = top_k
    return ranked


def evaluate_gdsc(
    model: TerminalWorldModel,
    prepared: PreparedData,
    out_dir: Path,
    top_k: int = 5,
    batch_size: int = 4096,
    device: str | None = None,
    benchmark: BenchmarkConfig | None = None,
    controls: AblationControls | None = None,
    context: TrainExecutionContext | None = None,
    export_planning_predictions: bool = False,
    export_terminal_latents: bool = False,
    export_gate_audit: bool = False,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
    eval_compile: bool = False,
) -> pd.DataFrame:
    benchmark = benchmark or BenchmarkConfig(benchmark_id="adhoc", benchmark_name="adhoc")
    mode = str(benchmark.model.mode).strip().lower()
    planner_metric_field = benchmark.evaluation.planner_metric_field
    explain_dirs = ensure_explain_dir(out_dir) if benchmark.explainability.enabled else None
    perturbation_forward_chunk_size = max(1, min(int(batch_size), 256))
    performance_runtime: dict[str, Any] = {
        "prediction": {},
        "explainability": {"postprocess_seconds": 0.0},
        "ablation": {"source_seconds": 0.0, "edge_seconds": 0.0, "node_seconds": 0.0},
        "perturbation": {"total_seconds": 0.0},
    }
    if export_gate_audit and (out_dir / "gate_audit.csv").exists():
        (out_dir / "gate_audit.csv").unlink()
    if export_gate_audit:
        alpha_path = out_dir / "kg_attention_by_prediction.csv"
        if alpha_path.exists():
            alpha_path.unlink()
    if explain_dirs is not None:
        for filename in [
            "kg_source_contribution_by_prediction.csv",
            "kg_top_edges_by_prediction.csv",
            "kg_top_nodes_by_prediction.csv",
            "kg_top_paths_by_prediction.csv",
            "kg_source_ablation_by_prediction.csv",
            "kg_edge_ablation_by_prediction.csv",
        ]:
            path = explain_dirs["root"] / filename
            if path.exists():
                path.unlink()
    outputs = predict_frame(
        model,
        prepared.responses,
        prepared,
        batch_size=batch_size,
        device=device,
        benchmark=benchmark,
        controls=controls,
        context=context,
        runtime_profile=runtime_profile,
        eval_compile=eval_compile,
        include_planning_frame=export_planning_predictions,
        terminal_latent_path=out_dir / "terminal_latents.npy" if export_terminal_latents else None,
        gate_audit_path=out_dir / "gate_audit.csv" if export_gate_audit else None,
        explain_dir=explain_dirs["root"] if explain_dirs is not None else None,
    )
    ablation_session = _build_prediction_session(
        model,
        prepared,
        device=str(device),
        benchmark=benchmark,
        controls=controls,
        context=context,
        runtime_profile=runtime_profile,
        eval_compile=eval_compile,
        explain_dir=None,
    )
    performance_runtime["prediction"] = dict(outputs.runtime or {})
    performance_runtime["explainability"]["postprocess_seconds"] = float((outputs.runtime or {}).get("eval_seconds_explain_postprocess", 0.0))
    pred = outputs.core
    pred["entity_id"] = pred["SANGER_MODEL_ID"].astype(str)
    selected_fusion_weight = 0.0
    fusion_metric_value = float("nan")
    if mode == "drug_residual_world_model":
        selected = select_best_fusion_weight(
            pred,
            candidates=benchmark.evaluation.fusion_weight_candidates,
            split="val",
            metric=benchmark.evaluation.fusion_selection_metric,
        )
        selected_fusion_weight = float(selected["selected_weight"])
        fusion_metric_value = float(selected["selected_metric_value"])
        pred = with_fused_auc(pred, selected_fusion_weight)
    pred = _rank_with_controls(pred, controls or AblationControls(), top_k=top_k)
    main_metrics = []
    proxy_metrics = []
    drug_pcc_frames = []
    cell_pcc_frames = []
    for split, group in pred.groupby("split", observed=True):
        main_row = {"split": split, "n": len(group)}
        main_row.update(regression_metrics(group))
        if mode in {"world_model", "drug_residual_world_model"}:
            main_row.update(value_metrics(group, target_col=planner_metric_field))
        main_metrics.append(main_row)

        proxy_row = {"split": split, "n": len(group)}
        if mode in {"world_model", "drug_residual_world_model"} and benchmark.world_model.planner_enabled:
            proxy_row.update(observed_planning_metrics(group, top_k=top_k, metric_field=planner_metric_field))
        else:
            proxy_row.update(_planning_metrics_disabled(top_k, f"disabled_for_mode:{mode}"))
        proxy_metrics.append(proxy_row)

        drug_pcc = groupwise_auc_correlations(group, "DRUG_ID")
        if not drug_pcc.empty:
            drug_names = group[["DRUG_ID", "DRUG_NAME"]].drop_duplicates()
            drug_pcc = drug_pcc.merge(drug_names, on="DRUG_ID", how="left")
            drug_pcc.insert(0, "split", split)
            drug_pcc = drug_pcc[["split", "DRUG_ID", "DRUG_NAME", "n", "pcc", "spearman"]]
            drug_pcc_frames.append(drug_pcc)

        cell_pcc = groupwise_auc_correlations(group, "entity_id")
        if not cell_pcc.empty:
            cell_pcc.insert(0, "split", split)
            cell_pcc = cell_pcc[["split", "entity_id", "n", "pcc"]]
            cell_pcc_frames.append(cell_pcc)

    pd.DataFrame(main_metrics).to_csv(out_dir / "gdsc_metrics.csv", index=False)
    pd.DataFrame(proxy_metrics).to_csv(out_dir / "gdsc_proxy_metrics.csv", index=False)
    drug_pcc_frame = (
        pd.concat(drug_pcc_frames, ignore_index=True)
        if drug_pcc_frames
        else pd.DataFrame(columns=["split", "DRUG_ID", "DRUG_NAME", "n", "pcc", "spearman"])
    )
    drug_pcc_frame.to_csv(out_dir / "gdsc_drug_pcc.csv", index=False)
    cell_pcc_frame = (
        pd.concat(cell_pcc_frames, ignore_index=True)
        if cell_pcc_frames
        else pd.DataFrame(columns=["split", "entity_id", "n", "pcc"])
    )
    cell_pcc_frame.to_csv(out_dir / "gdsc_cell_pcc.csv", index=False)
    pred[
        [
            "SANGER_MODEL_ID",
            "DRUG_ID",
            "DRUG_NAME",
            "drug_key",
            "smiles",
            "canonical_smiles",
            "inchikey",
            "canonical_group_key",
            "canonical_group_index",
            "canonical_group_size",
            "canonical_drug_id",
            "split",
            "AUC",
            "relative_value",
            benchmark.world_model.relative_value_train_field,
            benchmark.world_model.relative_value_eval_field,
            "cell_train_baseline",
            "cell_residual_baseline_train",
            "cell_residual_auc_train",
            "cell_residual_baseline_eval",
            "cell_residual_auc_eval",
            "drug_centered_baseline_train",
            "drug_centered_auc_train",
            "drug_centered_baseline_eval",
            "drug_centered_auc_eval",
            "auc_hat",
            "raw_auc_base",
            "value_hat",
            "cell_residual_hat",
            "drug_centered_hat",
            "uncertainty",
        ]
    ].to_csv(out_dir / "predictions.csv", index=False)
    _write_drug_level_audits(pred, prepared, out_dir)
    if outputs.explainability is not None and explain_dirs is not None:
        source_rows = outputs.explainability.get("source", pd.DataFrame())
        edge_rows = outputs.explainability.get("edge", pd.DataFrame())
        node_rows = outputs.explainability.get("node", pd.DataFrame())
        path_rows = (
            mine_paths_for_prediction(
                edge_rows=edge_rows,
                max_hops=benchmark.explainability.path_max_hops,
                top_k_paths=benchmark.explainability.top_k_paths,
            )
            if benchmark.explainability.export_top_paths and not edge_rows.empty
            else pd.DataFrame()
        )
        path_rows.to_csv(explain_dirs["root"] / "kg_top_paths_by_prediction.csv", index=False)
        source_ablation_rows = pd.DataFrame()
        edge_ablation_rows = pd.DataFrame()
        node_ablation_rows = pd.DataFrame()
        if benchmark.explainability.run_ablation:
            source_frames: list[pd.DataFrame] = []
            source_names = sorted(source_rows["source_name"].dropna().astype(str).unique().tolist()) if not source_rows.empty else []
            source_started_at = time.perf_counter()
            for source_name in source_names:
                masked = _run_prediction_ablation(
                    model,
                    prepared.responses,
                    prepared,
                    batch_size=batch_size,
                    device=str(device),
                    benchmark=benchmark,
                    controls=controls,
                    context=context,
                    runtime_profile=runtime_profile,
                    eval_compile=eval_compile,
                    kg_mask={"source_off": [source_name]},
                    chunk_size=perturbation_forward_chunk_size,
                    session=ablation_session,
                )
                masked = _postprocess_ablation_predictions(
                    masked,
                    mode=mode,
                    selected_fusion_weight=selected_fusion_weight,
                )
                source_frames.append(_build_source_ablation_rows(pred, masked, source_name=source_name))
            if source_frames:
                source_ablation_rows = pd.concat(source_frames, ignore_index=True)
            performance_runtime["ablation"]["source_seconds"] = float(time.perf_counter() - source_started_at)

            edge_k = int(benchmark.perturbation.action_top_k) if benchmark.perturbation.enabled else 1
            top_edge_rows = (
                edge_rows.sort_values(["prediction_index", "edge_attention"], ascending=[True, False])
                .groupby("prediction_index", observed=True)
                .head(edge_k)
                .copy()
            ) if not edge_rows.empty else pd.DataFrame()
            edge_frames: list[pd.DataFrame] = []
            pred_reset = pred.reset_index(drop=True).copy()
            if not top_edge_rows.empty:
                edge_started_at = time.perf_counter()
                for edge_id, edge_group in top_edge_rows.groupby("edge_id", observed=True):
                    subset = pred_reset.iloc[edge_group["prediction_index"].astype(int).tolist()].copy()
                    masked = _run_prediction_ablation(
                        model,
                        subset,
                        prepared,
                        batch_size=min(batch_size, max(len(subset), 1)),
                        device=str(device),
                        benchmark=benchmark,
                        controls=controls,
                        context=context,
                        runtime_profile=runtime_profile,
                        eval_compile=eval_compile,
                        kg_mask={"edge_ids": [int(edge_id)]},
                        chunk_size=perturbation_forward_chunk_size,
                        session=ablation_session,
                    )
                    masked = _postprocess_ablation_predictions(
                        masked,
                        mode=mode,
                        selected_fusion_weight=selected_fusion_weight,
                    )
                    edge_frames.append(
                        _build_edge_ablation_rows(
                            subset,
                            pred_reset,
                            edge_group,
                            masked,
                            edge_id=int(edge_id),
                        )
                    )
                performance_runtime["ablation"]["edge_seconds"] = float(time.perf_counter() - edge_started_at)
            if edge_frames:
                edge_ablation_rows = pd.concat(edge_frames, ignore_index=True)
            if benchmark.perturbation.enabled and "node" in set(benchmark.perturbation.action_levels):
                node_tasks = _build_node_ablation_tasks(
                    top_nodes=node_rows,
                    edge_rows=edge_rows,
                    allowed_node_types=set(benchmark.perturbation.allowed_node_types),
                    node_top_k=int(benchmark.perturbation.node_top_k),
                    max_edges_per_action=int(benchmark.perturbation.max_edges_per_action),
                )
                node_records: list[dict[str, Any]] = []
                node_started_at = time.perf_counter()
                masked_by_key: dict[tuple[int, ...], pd.DataFrame] = {}
                tasks_by_key: dict[tuple[int, ...], list[dict[str, Any]]] = {}
                for task in node_tasks:
                    tasks_by_key.setdefault(tuple(task["edge_ids"]), []).append(task)
                for edge_key, group_tasks in tasks_by_key.items():
                    prediction_indices = [int(task["prediction_index"]) for task in group_tasks]
                    subset = pred_reset.iloc[prediction_indices].copy()
                    masked = _run_prediction_ablation(
                        model,
                        subset,
                        prepared,
                        batch_size=min(batch_size, max(len(subset), 1)),
                        device=str(device),
                        benchmark=benchmark,
                        controls=controls,
                        context=context,
                        runtime_profile=runtime_profile,
                        eval_compile=eval_compile,
                        kg_mask={"edge_ids": list(edge_key)},
                        chunk_size=perturbation_forward_chunk_size,
                        session=ablation_session,
                    )
                    masked = _postprocess_ablation_predictions(
                        masked,
                        mode=mode,
                        selected_fusion_weight=selected_fusion_weight,
                    )
                    masked_by_key[edge_key] = masked.sort_values("prediction_index").set_index("prediction_index", drop=False)
                for task in node_tasks:
                    prediction_index = int(task["prediction_index"])
                    edge_key = tuple(task["edge_ids"])
                    masked_row = masked_by_key[edge_key].loc[[prediction_index]].reset_index(drop=True)
                    node_records.append(
                        _build_node_ablation_row(
                            pred=pred_reset,
                            prediction_index=prediction_index,
                            node_row=pd.Series(task["node_row"]),
                            related_edges=task["related_edges"],
                            masked=masked_row,
                        )
                    )
                performance_runtime["ablation"]["node_seconds"] = float(time.perf_counter() - node_started_at)
                if node_records:
                    node_ablation_rows = pd.DataFrame(node_records)
        source_ablation_rows.to_csv(explain_dirs["root"] / "kg_source_ablation_by_prediction.csv", index=False)
        edge_ablation_rows.to_csv(explain_dirs["root"] / "kg_edge_ablation_by_prediction.csv", index=False)
        prediction_coverage, drug_coverage = _build_explanation_coverage_audits(pred, source_rows, edge_rows, path_rows)
        pd.DataFrame(
            [
                {
                    "n_predictions": int(len(pred)),
                    "n_source_rows": int(len(source_rows)),
                    "n_edge_rows": int(len(edge_rows)),
                    "n_node_rows": int(len(node_rows)),
                    "n_path_rows": int(len(path_rows)),
                }
            ]
        ).to_csv(explain_dirs["summary"] / "kg_source_contribution_summary.csv", index=False)
        prediction_coverage.to_csv(explain_dirs["summary"] / "prediction_explanation_coverage_audit.csv", index=False)
        drug_coverage.to_csv(explain_dirs["summary"] / "drug_explanation_coverage_audit.csv", index=False)
        if benchmark.perturbation.enabled:
            perturbation_started_at = time.perf_counter()
            plan_perturbation_mechanisms(
                prediction_frame=pred,
                edge_rows=edge_rows,
                edge_ablation_rows=edge_ablation_rows,
                node_ablation_rows=node_ablation_rows,
                out_dir=out_dir,
                config=benchmark.perturbation,
                kg_graph=getattr(prepared.artifacts, "kg_graph", None),
            )
            performance_runtime["perturbation"]["total_seconds"] = float(time.perf_counter() - perturbation_started_at)
    if benchmark.combined.enabled:
        run_combined_prediction(
            model=model,
            prepared=prepared,
            paths=Paths(),
            benchmark=benchmark,
            out_dir=out_dir,
            device=device,
        )
    if outputs.planning is not None:
        planning = outputs.planning.loc[pred.index].copy()
        for column in [
            "auc_hat",
            "raw_auc_base",
            "value_hat",
            "cell_residual_hat",
            "drug_centered_hat",
            "uncertainty",
            "entity_id",
            "planner_score",
            "planner_rank",
        ]:
            planning[column] = pred[column].to_numpy()
        if "ood_score" not in planning.columns:
            planning["ood_score"] = 0.0
        planning.to_csv(out_dir / "planning_predictions.csv", index=False)
        runtime_path = out_dir / "gpu_runtime.json"
        runtime = {}
        if runtime_path.exists():
            with runtime_path.open("r", encoding="utf-8") as f:
                runtime = json.load(f)
        runtime.update(
            {
                "eval_batch_size": int(batch_size),
                "export_planning_predictions": bool(export_planning_predictions),
                "export_terminal_latents": bool(export_terminal_latents),
                "export_gate_audit": bool(export_gate_audit),
                "canonical_drug_indexing": bool(getattr(benchmark, "canonical_drug_indexing", False)),
                "coverage_aware_kg_mask": bool(getattr(benchmark, "coverage_aware_kg_mask", True)),
                "eval_rows": int(len(pred)),
                "strict_drug_level_metric": "within_drug_pcc_mean",
                "strict_drug_level_split": "test",
                "strict_drug_level_test_rows": int(prepared.responses["split"].eq("test").sum()),
                "strict_drug_level_test_drugs": int(prepared.responses.loc[prepared.responses["split"].eq("test"), "DRUG_ID"].astype(int).nunique()),
                "strict_drug_level_target": benchmark.evaluation.target_within_drug_pcc,
                "strict_expected_test_rows": benchmark.evaluation.strict_test_rows,
                "strict_expected_test_drugs": benchmark.evaluation.strict_test_drugs,
                "sample_guardrail_delta": benchmark.evaluation.sample_guardrail_delta,
                "fusion_selection_metric": benchmark.evaluation.fusion_selection_metric,
                "fusion_weight_candidates": list(normalize_fusion_weight_candidates(benchmark.evaluation.fusion_weight_candidates)),
                "selected_fusion_weight": float(selected_fusion_weight),
                "selected_fusion_metric_value": float(fusion_metric_value),
                "runtime_profile": str(runtime_profile),
                **(outputs.runtime or {}),
            }
        )
        write_json(runtime_path, runtime)
        write_json(out_dir / "performance_runtime_breakdown.json", performance_runtime)
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["fusion_selection_metric"] = benchmark.evaluation.fusion_selection_metric
            manifest["fusion_weight_candidates"] = list(normalize_fusion_weight_candidates(benchmark.evaluation.fusion_weight_candidates))
            manifest["selected_fusion_weight"] = float(selected_fusion_weight)
            manifest["strict_expected_test_rows"] = benchmark.evaluation.strict_test_rows
            manifest["strict_expected_test_drugs"] = benchmark.evaluation.strict_test_drugs
            write_json(manifest_path, manifest)
    return pred


def load_model(run_dir: Path, artifacts: FeatureArtifacts, strict: bool = True) -> TerminalWorldModel:
    try:
        state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "model.pt", map_location="cpu")
    drug_table = artifacts.canonical_drug_table if getattr(artifacts, "canonical_drug_table", None) is not None else artifacts.drug_table
    fp_bank = np.vstack([row.fingerprint.astype(np.float32) for row in drug_table.itertuples(index=False)])
    state_dim = int(getattr(artifacts, "state_dim", 0) or getattr(artifacts.pca, "n_components_", 0))
    state_weight = state.get("state_encoder.net.0.weight") if isinstance(state, dict) else None
    if state_weight is not None and getattr(state_weight, "shape", None):
        state_dim = int(state_weight.shape[1])
    model = TerminalWorldModel(
        state_dim=state_dim,
        prior_dim=len(artifacts.prior_columns),
        kg_artifacts=getattr(artifacts, "kg_graph", None),
        drug_fingerprint_bank=fp_bank,
    )
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model
