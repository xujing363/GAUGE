from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PRISM_PATCH_ROOT = Path(
    os.environ.get(
        "KGPUB_PRISM_PATCH_ROOT",
        "/mnt/raid5/xujing/KG/PRISM/Secondary",
    )
)
if str(PRISM_PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PRISM_PATCH_ROOT))

from GAUGE import benchmark_cli as shared_cli
from GAUGE import benchmarking as shared_benchmarking
from GAUGE import features as shared_features
from GAUGE import train as shared_train

from hvg2000_projection import IdentityProjection


@dataclass(frozen=True)
class ProjectionSpec:
    mode: str = "hvg"
    hvg_n_genes: int = 2000
    pca_n_components: int = 512
    hvg_selection_method: str = "variance"


_CURRENT_SPEC = ProjectionSpec()
_PATCHED = False
_ORIGINALS: dict[str, Any] = {}


def _read_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark config must be a mapping, got {type(payload).__name__}.")
    return payload


def _projection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    section = payload.get("input_projection", {}) or {}
    if not isinstance(section, dict):
        raise ValueError("input_projection must be a mapping if provided.")
    return section


def _set_current_spec(spec: ProjectionSpec) -> None:
    global _CURRENT_SPEC
    _CURRENT_SPEC = spec


def _current_spec() -> ProjectionSpec:
    return _CURRENT_SPEC


def _selection_key(series: pd.Series, gene: str) -> tuple[float, str]:
    value = float(series.get(gene, float("-inf")))
    if np.isnan(value):
        value = float("-inf")
    return (-value, str(gene))


def _select_hvg_genes(expr: pd.DataFrame, train_cell_lines: list[str], n_hvg: int) -> list[str]:
    genes = expr.columns.astype(str).tolist()
    train_expr = expr.loc[train_cell_lines, genes].astype(np.float32)
    variances = train_expr.var(axis=0, ddof=0).fillna(float("-inf"))
    ordered = sorted(genes, key=lambda gene: _selection_key(variances, gene))
    return ordered[: min(int(n_hvg), len(ordered))]


def _fit_identity_projection(n_features: int) -> IdentityProjection:
    return IdentityProjection(int(n_features))


def _fit_state_projection(
    expr: pd.DataFrame,
    train_cell_lines: list[str],
    n_components: int = 512,
) -> tuple[list[str], SimpleImputer, StandardScaler, Any]:
    spec = _current_spec()
    mode = str(spec.mode).strip().lower()
    if mode not in {"hvg", "pca", "hvg_then_pca"}:
        raise ValueError(f"Unsupported input_projection.mode: {spec.mode!r}")

    if mode == "pca":
        genes = expr.columns.astype(str).tolist()
    else:
        genes = _select_hvg_genes(expr, train_cell_lines, spec.hvg_n_genes)

    train_expr = expr.loc[train_cell_lines, genes].astype(np.float32)
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_expr))

    if mode == "hvg":
        pca = _fit_identity_projection(len(genes))
        return genes, imputer, scaler, pca

    effective_components = min(int(spec.pca_n_components), x_train.shape[0], x_train.shape[1])
    if effective_components < 1:
        raise ValueError("Cannot fit PCA with no train samples or genes.")
    pca = PCA(n_components=effective_components, random_state=0)
    pca.fit(x_train)
    return genes, imputer, scaler, pca


def _project_expression(
    expr: pd.DataFrame,
    genes: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    pca: Any,
) -> np.ndarray:
    aligned = expr.reindex(columns=genes, fill_value=0.0).astype(np.float32)
    transformed = scaler.transform(imputer.transform(aligned))
    if isinstance(pca, IdentityProjection):
        return np.asarray(transformed, dtype=np.float32)
    return pca.transform(transformed)


def _wrap_load_benchmark_config(path: Path):
    config = _ORIGINALS["load_benchmark_config"](path)
    payload = _read_payload(path)
    section = _projection_payload(payload)
    mode = str(section.get("mode", "hvg")).strip().lower()
    if mode not in {"hvg", "pca", "hvg_then_pca"}:
        raise ValueError("input_projection.mode must be one of: hvg, pca, hvg_then_pca")
    hvg_n_genes = int(section.get("hvg_n_genes", payload.get("n_hvg", getattr(config, "n_hvg", 2000))))
    pca_n_components = int(section.get("pca_n_components", payload.get("n_components", getattr(config, "n_components", 512))))
    hvg_selection_method = str(section.get("hvg_selection_method", "variance"))
    object.__setattr__(config, "input_projection_mode", mode)
    object.__setattr__(config, "input_projection_hvg_n_genes", hvg_n_genes)
    object.__setattr__(config, "input_projection_pca_n_components", pca_n_components)
    object.__setattr__(config, "input_projection_hvg_selection_method", hvg_selection_method)
    object.__setattr__(config, "n_hvg", hvg_n_genes)
    object.__setattr__(config, "n_components", hvg_n_genes if mode == "hvg" else pca_n_components)
    _set_current_spec(
        ProjectionSpec(
            mode=mode,
            hvg_n_genes=hvg_n_genes,
            pca_n_components=pca_n_components,
            hvg_selection_method=hvg_selection_method,
        )
    )
    return config


def _wrap_prepare_data(*args, **kwargs):
    prepared = _ORIGINALS["prepare_data"](*args, **kwargs)
    benchmark = kwargs.get("benchmark")
    out_dir = Path(args[1]) if len(args) > 1 else Path(kwargs["out_dir"])
    cache_dir = kwargs.get("cache_dir") or (out_dir / ".cache")
    paths = kwargs.get("paths") or args[0]
    seed = kwargs.get("seed", 7)
    n_components = kwargs.get("n_components", getattr(benchmark, "n_components", 512))
    max_rows = kwargs.get("max_rows", getattr(benchmark, "max_rows", None))
    gdsc_source_mode = kwargs.get("gdsc_source_mode", None)

    spec = _current_spec()
    prepared.manifest["input_projection_mode"] = spec.mode
    prepared.manifest["input_projection_hvg_n_genes"] = int(spec.hvg_n_genes)
    prepared.manifest["input_projection_pca_n_components"] = int(spec.pca_n_components)
    prepared.manifest["input_projection_hvg_selection_method"] = spec.hvg_selection_method
    prepared.leakage_audit["input_projection_mode"] = spec.mode
    prepared.leakage_audit["input_projection_hvg_n_genes"] = int(spec.hvg_n_genes)
    prepared.leakage_audit["input_projection_pca_n_components"] = int(spec.pca_n_components)
    prepared.leakage_audit["input_projection_hvg_selection_method"] = spec.hvg_selection_method

    prepare_contract = shared_features.build_relative_targets  # sentinel for linting; no-op reference
    _ = prepare_contract
    from GAUGE.contracts import build_prepare_contract
    from GAUGE.cache import CacheManager

    expected = build_prepare_contract(
        paths=paths,
        benchmark=benchmark,
        seed=seed,
        n_components=int(n_components),
        max_rows=max_rows,
        gdsc_source_mode=gdsc_source_mode,
    )
    cache = CacheManager(cache_dir, use_cache=True, rebuild_cache=False)
    key = expected["prepare_cache_key"]
    cache.save_pickle("prepare", key, "prepared.pkl", prepared)
    shared_train._write_prepare_outputs(prepared, out_dir)
    with (out_dir / "prepared.pkl").open("wb") as f:
        import pickle

        pickle.dump(prepared, f)
    return prepared


def _wrap_write_raw_manifest(raw_links_dir: Path, paths, config) -> None:
    _ORIGINALS["write_raw_manifest"](raw_links_dir, paths, config)
    dataset_name = str(getattr(config, "dataset_name", "gdsc")).strip().lower()
    if dataset_name != "gdsc":
        return
    mode = str(getattr(config, "gdsc_source_mode", "v2")).strip().lower()
    manifest_path = Path(raw_links_dir) / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    raw_sources = dict(payload.get("raw_sources", {}) or {})
    if mode == "v1":
        raw_sources["gdsc_fitted_1"] = str(paths.gdsc_fitted_1)
        raw_sources.pop("gdsc_fitted_2", None)
    elif mode == "v2":
        raw_sources["gdsc_fitted_1"] = str(paths.gdsc_fitted_2)
        raw_sources.pop("gdsc_fitted_2", None)
    else:
        raw_sources["gdsc_fitted_1"] = str(paths.gdsc_fitted_1)
        raw_sources["gdsc_fitted_2"] = str(paths.gdsc_fitted_2)
    payload["gdsc_source_mode"] = mode
    payload["raw_sources"] = raw_sources
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _ORIGINALS["load_benchmark_config"] = shared_benchmarking.load_benchmark_config
    _ORIGINALS["prepare_data"] = shared_train.prepare_data
    _ORIGINALS["fit_state_projection"] = shared_train.fit_state_projection
    _ORIGINALS["project_expression"] = shared_train.project_expression
    _ORIGINALS["write_raw_manifest"] = shared_benchmarking.write_raw_manifest
    _ORIGINALS["signature"] = shared_benchmarking.BenchmarkConfig.signature

    def _signature(self):
        payload = dict(_ORIGINALS["signature"](self))
        payload["input_projection"] = {
            "mode": getattr(self, "input_projection_mode", "hvg"),
            "hvg_n_genes": int(getattr(self, "input_projection_hvg_n_genes", getattr(self, "n_hvg", 2000))),
            "pca_n_components": int(getattr(self, "input_projection_pca_n_components", getattr(self, "n_components", 512))),
            "hvg_selection_method": str(getattr(self, "input_projection_hvg_selection_method", "variance")),
        }
        return payload

    shared_benchmarking.BenchmarkConfig.signature = _signature
    shared_benchmarking.load_benchmark_config = _wrap_load_benchmark_config
    shared_benchmarking.write_raw_manifest = _wrap_write_raw_manifest
    shared_train.prepare_data = _wrap_prepare_data
    shared_train.fit_state_projection = _fit_state_projection
    shared_train.project_expression = _project_expression
    shared_features.fit_state_projection = _fit_state_projection
    shared_features.project_expression = _project_expression
    shared_cli.load_benchmark_config = _wrap_load_benchmark_config
    shared_cli.write_raw_manifest = _wrap_write_raw_manifest
    shared_cli.prepare_data = _wrap_prepare_data


@contextmanager
def projection_context(spec: ProjectionSpec) -> Iterator[None]:
    previous = _current_spec()
    _set_current_spec(spec)
    try:
        yield
    finally:
        _set_current_spec(previous)
