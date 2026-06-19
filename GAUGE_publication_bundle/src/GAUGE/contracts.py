from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .cache import cache_key, files_signature
from .data import BLOCKED_PRIOR_TERMS
from .config import DATASET_DEFAULT, Paths, dataset_smiles_path, gdsc_fitted_paths, normalize_dataset_name, normalize_gdsc_source_mode
from .kg_prior import KG_PRIOR_SCHEMA_VERSION, resolve_chembl_source
from .utils import write_json

if TYPE_CHECKING:
    from .benchmarking import BenchmarkConfig
    from .train import PreparedData


PREPARE_CONTRACT_KEYS = (
    "benchmark",
    "dataset_name",
    "seed",
    "n_components",
    "max_rows",
    "gdsc_source_mode",
    "gdsc_fitted_files",
    "state_projection_policy_version",
    "relative_target_schema_version",
    "cell_residual_target_schema_version",
    "drug_canonicalization_version",
    "prior_sources",
    "files_signature",
    "gdsc_smiles_cache",
    "blocked_prior_terms",
    "kg_prior_schema_version",
    "kg_drug_index_space_version",
    "resolved_prior_policy",
    "kg_prior_sources",
    "prior_sources",
    "smiles_attachment_schema_version",
    "cell_train_statistics_version",
    "chembl_release",
    "chembl_sqlite_tar",
    "prepare_cache_key",
)


def _is_ctrp_benchmark(benchmark: BenchmarkConfig) -> bool:
    return normalize_dataset_name(getattr(benchmark, "dataset_name", DATASET_DEFAULT)) == "ctrdb"


def _prepare_cache_files(paths: Paths, benchmark: BenchmarkConfig, gdsc_source_mode: str) -> list[Path]:
    _, chembl_sqlite_tar, _ = resolve_chembl_source(paths, prior_policy=benchmark.prior)
    if _is_ctrp_benchmark(benchmark):
        return [
            paths.ctrp_response,
            paths.ctrp_gene_expression,
            paths.ctrp_drug_smiles,
            paths.primekg,
        ]
    dataset_name = normalize_dataset_name(getattr(benchmark, "dataset_name", DATASET_DEFAULT))
    if dataset_name == "beataml2":
        return [
            paths.beataml2_curve_fits,
            paths.beataml2_raw_inhibitor,
            paths.beataml2_expression,
            paths.beataml2_counts,
            paths.beataml2_drug_families,
            paths.beataml2_sample_mapping,
            paths.beataml2_clinical,
            paths.beataml2_smiles_cache,
            paths.primekg,
            chembl_sqlite_tar,
            paths.root / "KG_GAUGE_PublicData" / "drug" / "drkg" / "drkg.tar.gz",
        ]
    if dataset_name == "pdx":
        return [
            paths.pdx_response,
            paths.pdx_gene_expression,
            paths.pdx_drug_smiles,
            paths.dataset_smiles_dir / "pdx_smiles.csv",
            paths.primekg,
            chembl_sqlite_tar,
            paths.root / "KG_GAUGE_PublicData" / "drug" / "drkg" / "drkg.tar.gz",
        ]
    cache_files = [
        *gdsc_fitted_paths(paths, gdsc_source_mode),
        paths.gdsc_expression,
        paths.gdsc_gene_identifiers,
        paths.primekg,
        paths.gdsc_smiles_cache,
        chembl_sqlite_tar,
        paths.root / "KG_GAUGE_PublicData" / "drug" / "drkg" / "drkg.tar.gz",
    ]
    return cache_files


def build_prepare_contract(
    paths: Paths,
    benchmark: BenchmarkConfig,
    seed: int,
    n_components: int,
    max_rows: int | None,
    gdsc_source_mode: str | None = None,
) -> dict[str, Any]:
    dataset_name = normalize_dataset_name(getattr(benchmark, "dataset_name", DATASET_DEFAULT))
    chembl_release, chembl_sqlite_tar, _ = resolve_chembl_source(paths, prior_policy=benchmark.prior)
    effective_gdsc_source_mode = (
        normalize_gdsc_source_mode(gdsc_source_mode or benchmark.gdsc_source_mode)
        if dataset_name == "gdsc"
        else None
    )
    if _is_ctrp_benchmark(benchmark):
        response_files = [paths.ctrp_response]
        response_cache_path = str(paths.ctrp_drug_smiles)
    elif dataset_name == "beataml2":
        response_files = [paths.beataml2_curve_fits]
        response_cache_path = str(paths.beataml2_smiles_cache)
    elif dataset_name == "pdx":
        response_files = [paths.pdx_response]
        response_cache_path = str(dataset_smiles_path(paths, "pdx"))
    else:
        response_files = gdsc_fitted_paths(paths, effective_gdsc_source_mode)
        response_cache_path = str(paths.gdsc_smiles_cache)
    gdsc_files = [str(path) for path in response_files]
    resolved_prior_policy = {
        "kg_prior": benchmark.prior.to_dict(),
        "static_prior": benchmark.static_prior.to_dict(),
    }
    kg_prior_sources = [
        name
        for name, cfg in (
            ("chembl", benchmark.prior.chembl),
            ("drkg", benchmark.prior.drkg),
            ("primekg", benchmark.prior.primekg),
        )
        if bool(cfg.enabled) and float(cfg.weight) > 0.0
    ]
    static_prior_sources = list(benchmark.static_prior.sources) if benchmark.static_prior.enabled else []
    contract = {
        "benchmark": benchmark.prepare_signature(),
        "dataset_name": dataset_name,
        "seed": int(seed),
        "n_components": int(n_components),
        "max_rows": max_rows,
        "gdsc_source_mode": effective_gdsc_source_mode,
        "gdsc_fitted_files": gdsc_files,
        "state_projection_policy_version": "v2_train_rows_unique_cells",
        "relative_target_schema_version": "v2_train_eval_split_fields",
        "cell_residual_target_schema_version": "v1_train_cell_median",
        "drug_canonicalization_version": "v1_canonical_smiles_inchikey",
        "files_signature": files_signature(_prepare_cache_files(paths, benchmark, effective_gdsc_source_mode)),
        "gdsc_smiles_cache": response_cache_path,
        "blocked_prior_terms": list(BLOCKED_PRIOR_TERMS),
        "resolved_prior_policy": resolved_prior_policy,
        "prior_sources": static_prior_sources,
        "kg_prior_sources": kg_prior_sources,
        "kg_prior_schema_version": int(KG_PRIOR_SCHEMA_VERSION),
        "kg_drug_index_space_version": "v3_canonical_aligned_strict",
        "smiles_attachment_schema_version": 2,
        "cell_train_statistics_version": "v1_train_only_nonleak",
        "chembl_release": chembl_release,
        "chembl_sqlite_tar": str(chembl_sqlite_tar),
    }
    contract["prepare_cache_key"] = cache_key({"kind": "prepared_data", **contract})
    return contract


def _compat_prepare_contract(prepared: PreparedData) -> dict[str, Any]:
    manifest = getattr(prepared, "manifest", {}) or {}
    contract = manifest.get("prepare_contract")
    if isinstance(contract, dict):
        return contract
    compat = {
        "benchmark": manifest.get("benchmark"),
        "dataset_name": manifest.get("dataset_name"),
        "seed": manifest.get("seed"),
        "n_components": manifest.get("n_components_requested"),
        "max_rows": manifest.get("max_rows"),
        "gdsc_source_mode": manifest.get("gdsc_source_mode"),
        "gdsc_fitted_files": manifest.get("gdsc_fitted_files"),
        "state_projection_policy_version": manifest.get("state_projection_policy_version"),
        "relative_target_schema_version": manifest.get("relative_target_schema_version"),
        "cell_residual_target_schema_version": manifest.get("cell_residual_target_schema_version"),
        "drug_canonicalization_version": manifest.get("drug_canonicalization_version"),
        "prior_sources": manifest.get("prior_sources"),
        "files_signature": manifest.get("files_signature"),
        "gdsc_smiles_cache": manifest.get("gdsc_smiles_cache"),
        "blocked_prior_terms": manifest.get("blocked_prior_terms"),
        "prepare_cache_key": manifest.get("prepare_cache_key"),
        "kg_prior_schema_version": manifest.get("kg_prior_schema_version"),
        "kg_drug_index_space_version": manifest.get("kg_drug_index_space_version"),
        "resolved_prior_policy": manifest.get("resolved_prior_policy"),
        "kg_prior_sources": manifest.get("kg_prior_sources"),
        "smiles_attachment_schema_version": manifest.get("smiles_attachment_schema_version"),
        "cell_train_statistics_version": manifest.get("cell_train_statistics_version"),
        "chembl_release": manifest.get("chembl_release"),
        "chembl_sqlite_tar": manifest.get("chembl_sqlite_tar"),
    }
    return compat


def prepare_contract_mismatch_reasons(prepared: PreparedData, expected: dict[str, Any]) -> list[str]:
    actual = _compat_prepare_contract(prepared)
    reasons: list[str] = []
    for key in PREPARE_CONTRACT_KEYS:
        if actual.get(key) != expected.get(key):
            reasons.append(f"{key} mismatch")
    responses = getattr(prepared, "responses", pd.DataFrame())
    for field in ("relative_value_train", "relative_value_eval", "relative_value"):
        if field not in responses.columns:
            reasons.append(f"{field} missing")
    resolved_policy = actual.get("resolved_prior_policy") or expected.get("resolved_prior_policy") or {}
    kg_prior = resolved_policy.get("kg_prior", {}) if isinstance(resolved_policy, dict) else {}
    source_configs = [
        kg_prior.get("chembl", {}) if isinstance(kg_prior, dict) else {},
        kg_prior.get("drkg", {}) if isinstance(kg_prior, dict) else {},
        kg_prior.get("primekg", {}) if isinstance(kg_prior, dict) else {},
    ]
    expects_kg_graph = any(bool(cfg.get("enabled", False)) and float(cfg.get("weight", 0.0)) > 0.0 for cfg in source_configs)
    artifacts = getattr(prepared, "artifacts", None)
    kg_graph = getattr(artifacts, "kg_graph", None) if artifacts is not None else None
    if expects_kg_graph and kg_graph is None:
        reasons.append("kg_graph missing")
    manifest = getattr(prepared, "manifest", {}) or {}
    manifest_kg_key = manifest.get("kg_prior_cache_key")
    actual_kg_key = getattr(kg_graph, "cache_key", None) if kg_graph is not None else None
    if manifest_kg_key is not None and actual_kg_key is not None and manifest_kg_key != actual_kg_key:
        reasons.append("kg_prior_cache_key mismatch")
    canonical_enabled = bool((manifest.get("benchmark") or {}).get("canonical_drug_indexing", True))
    canonical_table = getattr(artifacts, "canonical_drug_table", None) if artifacts is not None else None
    if canonical_enabled and kg_graph is not None and canonical_table is not None and not canonical_table.empty:
        expected_drug_ids = canonical_table["DRUG_ID"].astype(int).tolist()
        actual_drug_ids = [int(x) for x in getattr(kg_graph, "drug_ids", [])]
        if expected_drug_ids != actual_drug_ids:
            reasons.append("kg_graph drug space mismatch")
    return reasons


def export_gdsc_benchmark_contract(prepared: PreparedData, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses = prepared.responses.copy()
    if "relative_value_train" not in responses.columns:
        responses["relative_value_train"] = responses.get("relative_value", pd.Series(np.nan, index=responses.index))
    if "relative_value_eval" not in responses.columns:
        responses["relative_value_eval"] = responses.get("relative_value", pd.Series(np.nan, index=responses.index))
    if "relative_value" not in responses.columns:
        responses["relative_value"] = responses["relative_value_eval"]
    pair_cols = [
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
        "relative_value_train",
        "relative_value_eval",
        "cell_train_baseline",
        "cell_residual_baseline_train",
        "cell_residual_auc_train",
        "cell_residual_baseline_eval",
        "cell_residual_auc_eval",
        "auc_delta_vs_train_median",
    ]
    for extra in ["drug_key", "smiles"]:
        if extra in responses.columns:
            pair_cols.append(extra)
    pair_cols = [col for col in pair_cols if col in responses.columns]
    pair_frame = responses[pair_cols].copy()
    pair_frame.to_csv(out_dir / "gdsc_pairs.csv", index=False)

    train_ref = responses.loc[responses["split"].eq("train"), ["DRUG_ID", "AUC"]].copy()
    train_ref.to_csv(out_dir / "train_auc_reference.csv", index=False)

    prepare_contract = _compat_prepare_contract(prepared)
    benchmark = prepare_contract.get("benchmark") or prepared.manifest.get("benchmark") or {}
    payload = {
        "benchmark_id": benchmark.get("benchmark_id", "unknown"),
        "benchmark_name": benchmark.get("benchmark_name", "unknown"),
        "dataset_name": benchmark.get("dataset_name", prepared.manifest.get("dataset_name")),
        "split_type": benchmark.get("split_type"),
        "split_seed": benchmark.get("split_seed"),
        "gdsc_source_mode": prepare_contract.get("gdsc_source_mode", prepared.manifest.get("gdsc_source_mode")),
        "gdsc_fitted_files": prepare_contract.get("gdsc_fitted_files", prepared.manifest.get("gdsc_fitted_files")),
        "dataset_smiles_cache": prepare_contract.get("gdsc_smiles_cache", prepared.manifest.get("gdsc_smiles_cache")),
        "train_fraction": benchmark.get("train_fraction"),
        "val_fraction": benchmark.get("val_fraction"),
        "test_fraction": benchmark.get("test_fraction"),
        "max_rows": prepare_contract.get("max_rows", prepared.manifest.get("max_rows")),
        "n_components": prepare_contract.get("n_components", prepared.manifest.get("n_components_requested")),
        "seed": prepare_contract.get("seed", prepared.manifest.get("seed")),
        "prepare_cache_key": prepare_contract.get("prepare_cache_key", prepared.manifest.get("prepare_cache_key")),
        "resolved_prior_policy": prepare_contract.get("resolved_prior_policy", prepared.manifest.get("resolved_prior_policy")),
        "n_pairs": int(len(pair_frame)),
        "n_train_pairs": int(len(train_ref)),
        "files": {
            "gdsc_pairs": "gdsc_pairs.csv",
            "train_auc_reference": "train_auc_reference.csv",
        },
    }
    write_json(out_dir / "benchmark_contract.json", payload)


def load_benchmark_contract(contract_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    contract_dir = Path(contract_dir)
    with (contract_dir / "benchmark_contract.json").open("r", encoding="utf-8") as f:
        contract = json.load(f)
    pairs = pd.read_csv(contract_dir / "gdsc_pairs.csv")
    train_ref = pd.read_csv(contract_dir / "train_auc_reference.csv")
    return contract, pairs, train_ref
