from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmarking import BenchmarkConfig
from .config import Paths, normalize_dataset_name
from .data import load_beataml2_expression, load_ctrp_expression, load_gdsc_expression, load_pdx_expression
from .metrics import regression_metrics
from .train import PreparedData, load_model, predict_frame
from .utils import ensure_dir, write_json


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_raw_expression(paths: Paths, benchmark: BenchmarkConfig) -> pd.DataFrame | None:
    dataset_name = normalize_dataset_name(getattr(benchmark, "dataset_name", "gdsc"))
    if dataset_name == "ctrdb":
        return load_ctrp_expression(paths.ctrp_gene_expression)
    if dataset_name == "beataml2":
        return load_beataml2_expression(paths.beataml2_expression)
    if dataset_name == "pdx":
        return load_pdx_expression(paths.pdx_gene_expression)
    if dataset_name == "gdsc":
        return load_gdsc_expression(paths.gdsc_expression, paths.gdsc_gene_identifiers)
    return None


def _distance_table(feature_matrix: pd.DataFrame, train_cells: list[str], query_cells: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mat = feature_matrix.loc[train_cells].to_numpy(dtype=np.float32)
    query_mat = feature_matrix.loc[query_cells].to_numpy(dtype=np.float32)
    query_sq = np.sum(query_mat * query_mat, axis=1, dtype=np.float32)[:, None]
    train_sq = np.sum(train_mat * train_mat, axis=1, dtype=np.float32)[None, :]
    distances_sq = np.maximum(query_sq + train_sq - (2.0 * np.matmul(query_mat, train_mat.T)), 0.0)
    distances = np.sqrt(distances_sq, dtype=np.float32)
    order = np.argsort(distances, axis=1)
    ordered_distances = np.take_along_axis(distances, order, axis=1)
    ordered_cells = np.asarray(train_cells, dtype=object)[order]
    order_frame = pd.DataFrame(ordered_cells, index=query_cells)
    distance_frame = pd.DataFrame(ordered_distances, index=query_cells)
    return order_frame, distance_frame


def _train_drug_matrix(responses: pd.DataFrame) -> pd.DataFrame:
    train_rows = responses.loc[responses["split"].eq("train"), ["SANGER_MODEL_ID", "DRUG_ID", "AUC"]].copy()
    return train_rows.pivot_table(index="SANGER_MODEL_ID", columns="DRUG_ID", values="AUC", aggfunc="mean")


def _knn_transfer_predictions(
    responses: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    *,
    k: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    heldout = responses.loc[responses["split"].ne("train")].copy().reset_index(drop=True)
    train_cells = sorted(responses.loc[responses["split"].eq("train"), "SANGER_MODEL_ID"].astype(str).unique())
    query_cells = sorted(heldout["SANGER_MODEL_ID"].astype(str).unique())
    order_frame, distance_frame = _distance_table(feature_matrix, train_cells, query_cells)
    train_drug = _train_drug_matrix(responses)
    global_by_drug = responses.loc[responses["split"].eq("train")].groupby("DRUG_ID")["AUC"].median().astype(float).to_dict()
    global_default = float(responses.loc[responses["split"].eq("train"), "AUC"].median())
    preds: list[float] = []
    neighbor_counts: list[int] = []
    for row in heldout.itertuples(index=False):
        ordered_cells = order_frame.loc[str(row.SANGER_MODEL_ID)].tolist()
        ordered_distances = distance_frame.loc[str(row.SANGER_MODEL_ID)].to_numpy(dtype=float)
        values: list[float] = []
        weights: list[float] = []
        for cell, dist in zip(ordered_cells, ordered_distances):
            if len(values) >= k:
                break
            value = train_drug.get(int(row.DRUG_ID))
            if value is None or cell not in value.index:
                continue
            auc = value.loc[cell]
            if pd.isna(auc):
                continue
            values.append(float(auc))
            weights.append(1.0 / max(float(dist), 1e-6))
        if values:
            pred = float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))
        else:
            pred = float(global_by_drug.get(int(row.DRUG_ID), global_default))
        preds.append(pred)
        neighbor_counts.append(len(values))
    heldout["auc_hat"] = np.asarray(preds, dtype=np.float32)
    heldout["entity_id"] = heldout["SANGER_MODEL_ID"].astype(str)
    heldout["neighbor_count"] = np.asarray(neighbor_counts, dtype=int)
    metrics = {}
    for split, group in heldout.groupby("split", observed=True):
        metric_frame = group[["entity_id", "DRUG_ID", "AUC", "auc_hat"]].replace([np.inf, -np.inf], np.nan).dropna()
        metrics[str(split)] = regression_metrics(metric_frame.copy()) if not metric_frame.empty else {}
    heldout_metric_frame = heldout[["entity_id", "DRUG_ID", "AUC", "auc_hat"]].replace([np.inf, -np.inf], np.nan).dropna()
    metrics["heldout"] = regression_metrics(heldout_metric_frame.copy()) if not heldout_metric_frame.empty else {}
    return heldout, metrics


def _neighbor_profile_audit(
    responses: pd.DataFrame,
    processed_state: pd.DataFrame,
    raw_expr: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    train_cells = sorted(responses.loc[responses["split"].eq("train"), "SANGER_MODEL_ID"].astype(str).unique())
    heldout_cells = sorted(responses.loc[responses["split"].ne("train"), "SANGER_MODEL_ID"].astype(str).unique())
    proc_order, proc_dist = _distance_table(processed_state, train_cells, heldout_cells)
    raw_order = raw_dist = None
    if raw_expr is not None:
        raw_cells = [cell for cell in train_cells + heldout_cells if cell in raw_expr.index]
        raw_expr = raw_expr.loc[raw_cells]
        if set(train_cells).issubset(set(raw_expr.index)) and set(heldout_cells).issubset(set(raw_expr.index)):
            raw_order, raw_dist = _distance_table(raw_expr, train_cells, heldout_cells)
    train_auc = _train_drug_matrix(responses)
    all_auc = responses.pivot_table(index="SANGER_MODEL_ID", columns="DRUG_ID", values="AUC", aggfunc="mean")
    rows: list[dict[str, Any]] = []
    raw_status = "available" if raw_order is not None else "unavailable"
    for cell in heldout_cells:
        proc_neighbor = str(proc_order.loc[cell, 0])
        proc_shared = all_auc.loc[[cell, proc_neighbor]].dropna(axis=1, how="any")
        raw_neighbor = ""
        raw_shared = pd.DataFrame()
        if raw_order is not None and raw_dist is not None:
            raw_neighbor = str(raw_order.loc[cell, 0])
            raw_shared = all_auc.loc[[cell, raw_neighbor]].dropna(axis=1, how="any")
        split_values = responses.loc[responses["SANGER_MODEL_ID"].astype(str).eq(cell), "split"].astype(str).unique().tolist()
        rows.append(
            {
                "SANGER_MODEL_ID": cell,
                "heldout_split": ",".join(sorted(split_values)),
                "processed_neighbor_cell": proc_neighbor,
                "processed_neighbor_distance": float(proc_dist.loc[cell, 0]),
                "processed_shared_drugs": int(proc_shared.shape[1]),
                "processed_true_profile_corr": _safe_corr(
                    proc_shared.loc[cell].to_numpy(dtype=float),
                    proc_shared.loc[proc_neighbor].to_numpy(dtype=float),
                ),
                "raw_expression_status": raw_status,
                "raw_neighbor_cell": raw_neighbor,
                "raw_neighbor_distance": float(raw_dist.loc[cell, 0]) if raw_dist is not None else float("nan"),
                "raw_shared_drugs": int(raw_shared.shape[1]) if not raw_shared.empty else 0,
                "raw_true_profile_corr": _safe_corr(
                    raw_shared.loc[cell].to_numpy(dtype=float),
                    raw_shared.loc[raw_neighbor].to_numpy(dtype=float),
                )
                if raw_neighbor and not raw_shared.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows), {"raw_expression_status": raw_status}


def _select_probe_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    heldout = predictions.loc[predictions["split"].ne("train")].copy()
    selected: list[pd.Series] = []
    for _, group in heldout.groupby("SANGER_MODEL_ID", observed=True):
        group = group.sort_values("AUC").reset_index(drop=True)
        picks = {0, len(group) // 2, len(group) - 1}
        for idx in sorted(picks):
            selected.append(group.iloc[idx])
    if not selected:
        return heldout.iloc[:0].copy()
    return pd.DataFrame(selected).drop_duplicates(subset=["SANGER_MODEL_ID", "DRUG_ID"]).reset_index(drop=True)


def _empty_local_perturbation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "SANGER_MODEL_ID",
            "DRUG_ID",
            "DRUG_NAME",
            "split",
            "AUC",
            "auc_hat",
            "audit_source_cell",
            "audit_neighbor_cell",
            "audit_path_type",
            "audit_alpha",
            "probe_row_id",
        ]
    )


def _local_perturbation_audit(
    *,
    result_dir: Path,
    benchmark: BenchmarkConfig,
    prepared: PreparedData,
    predictions: pd.DataFrame,
    device: str,
    nearest_neighbors: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    probe_rows = _select_probe_rows(predictions)
    if not probe_rows.empty:
        probe_rows = probe_rows.loc[probe_rows["SANGER_MODEL_ID"].astype(str).isin(set(nearest_neighbors))].reset_index(drop=True)
    if probe_rows.empty:
        return _empty_local_perturbation_frame(), {"non_monotone_fraction": float("nan"), "max_abs_step_p95": float("nan")}
    artifacts = pd.read_pickle(result_dir / "artifacts.pkl")
    model = load_model(result_dir, artifacts)
    state_frames = []
    frame_rows = []
    eps_scale = 0.05
    interpolation_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for row_idx, row in probe_rows.iterrows():
        cell = str(row["SANGER_MODEL_ID"])
        neighbor = nearest_neighbors[cell]
        source = prepared.state_matrix.loc[cell].to_numpy(dtype=np.float32)
        target = prepared.state_matrix.loc[neighbor].to_numpy(dtype=np.float32)
        direction = target - source
        norm = float(np.linalg.norm(direction))
        unit = direction / norm if norm > 0 else np.zeros_like(direction)
        local_delta = unit * max(norm * eps_scale, 1e-3)
        path_specs = [
            ("local_minus", -0.05, source - local_delta),
            ("original", 0.0, source),
            ("local_plus", 0.05, source + local_delta),
        ]
        path_specs.extend((f"interp_{alpha:.2f}", alpha, source + (direction * alpha)) for alpha in interpolation_alphas[1:])
        for path_type, alpha, state_vec in path_specs:
            synthetic_cell = f"audit::{row_idx}::{path_type}"
            state_frames.append(pd.Series(state_vec, index=prepared.state_matrix.columns, name=synthetic_cell))
            synthetic = row.copy()
            synthetic["SANGER_MODEL_ID"] = synthetic_cell
            synthetic["audit_source_cell"] = cell
            synthetic["audit_neighbor_cell"] = neighbor
            synthetic["audit_path_type"] = path_type
            synthetic["audit_alpha"] = alpha
            frame_rows.append(synthetic)
    state_matrix = pd.concat([prepared.state_matrix, pd.DataFrame(state_frames)], axis=0)
    temp_prepared = PreparedData(
        responses=prepared.responses,
        state_matrix=state_matrix,
        artifacts=replace(prepared.artifacts, state_dim=int(state_matrix.shape[1])),
        missing_smiles=prepared.missing_smiles,
        invalid_smiles=prepared.invalid_smiles,
        prior_audit=prepared.prior_audit,
        split_audit=prepared.split_audit,
        leakage_audit=prepared.leakage_audit,
        manifest=prepared.manifest,
    )
    temp_frame = pd.DataFrame(frame_rows).reset_index(drop=True)
    predicted = predict_frame(
        model,
        temp_frame,
        temp_prepared,
        batch_size=2048,
        device=device,
        benchmark=benchmark,
        controls=benchmark.controls,
    ).core
    predicted["audit_source_cell"] = temp_frame["audit_source_cell"].to_numpy()
    predicted["audit_neighbor_cell"] = temp_frame["audit_neighbor_cell"].to_numpy()
    predicted["audit_path_type"] = temp_frame["audit_path_type"].to_numpy()
    predicted["audit_alpha"] = temp_frame["audit_alpha"].to_numpy(dtype=float)
    predicted["probe_row_id"] = np.repeat(np.arange(len(probe_rows), dtype=int), len(predicted) // len(probe_rows))
    predicted = predicted.sort_values(["probe_row_id", "audit_alpha", "audit_path_type"]).reset_index(drop=True)
    summary_rows = []
    for probe_id, group in predicted.groupby("probe_row_id", observed=True):
        interp = group.loc[group["audit_path_type"].astype(str).str.startswith("interp_") | group["audit_path_type"].eq("original")].sort_values("audit_alpha")
        series = interp["auc_hat"].to_numpy(dtype=float)
        deltas = np.diff(series)
        signs = np.sign(deltas[np.abs(deltas) > 1e-6])
        direction_changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
        summary_rows.append(
            {
                "probe_row_id": int(probe_id),
                "max_abs_step": float(np.max(np.abs(deltas))) if deltas.size else 0.0,
                "direction_changes": direction_changes,
                "endpoint_delta": float(series[-1] - series[0]) if series.size else 0.0,
                "non_monotone": bool(direction_changes > 0),
            }
        )
    summary = pd.DataFrame(summary_rows)
    metrics = {
        "non_monotone_fraction": float(summary["non_monotone"].mean()) if not summary.empty else float("nan"),
        "max_abs_step_p95": float(summary["max_abs_step"].quantile(0.95)) if not summary.empty else float("nan"),
    }
    return predicted, metrics


def _metric_value(metrics_by_scope: dict[str, dict[str, float]], scope: str, metric: str) -> float:
    return float((metrics_by_scope.get(scope) or {}).get(metric, float("nan")))


def _build_summary_frame(
    *,
    model_metrics: dict[str, dict[str, float]],
    processed_metrics: dict[str, dict[str, float]],
    raw_metrics: dict[str, dict[str, float]] | None,
    neighbor_audit: pd.DataFrame,
    perturb_metrics: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name, metrics_by_scope in [
        ("model", model_metrics),
        ("processed_state_knn_transfer", processed_metrics),
        ("raw_expression_knn_transfer", raw_metrics or {}),
    ]:
        for scope, metrics in metrics_by_scope.items():
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "category": "predictive_metrics",
                        "source": source_name,
                        "scope": scope,
                        "metric": metric_name,
                        "value": float(value) if pd.notna(value) else float("nan"),
                    }
                )
    rows.extend(
        [
            {
                "category": "neighbor_smoothness",
                "source": "processed_state",
                "scope": "heldout_cells",
                "metric": "true_profile_corr_mean",
                "value": float(neighbor_audit["processed_true_profile_corr"].mean()),
            },
            {
                "category": "neighbor_smoothness",
                "source": "raw_expression",
                "scope": "heldout_cells",
                "metric": "true_profile_corr_mean",
                "value": float(neighbor_audit["raw_true_profile_corr"].mean()),
            },
            {
                "category": "local_perturbation",
                "source": "processed_state",
                "scope": "probe_rows",
                "metric": "non_monotone_fraction",
                "value": float(perturb_metrics.get("non_monotone_fraction", float("nan"))),
            },
            {
                "category": "local_perturbation",
                "source": "processed_state",
                "scope": "probe_rows",
                "metric": "max_abs_step_p95",
                "value": float(perturb_metrics.get("max_abs_step_p95", float("nan"))),
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_verdict(
    *,
    model_metrics: dict[str, dict[str, float]],
    processed_metrics: dict[str, dict[str, float]],
    raw_metrics: dict[str, dict[str, float]] | None,
    neighbor_audit: pd.DataFrame,
    perturb_metrics: dict[str, float],
) -> dict[str, Any]:
    scope = "test" if "test" in processed_metrics else "heldout"
    model_within_cell = _metric_value(model_metrics, scope, "within_cell_pcc_mean")
    processed_within_cell = _metric_value(processed_metrics, scope, "within_cell_pcc_mean")
    raw_within_cell = _metric_value(raw_metrics or {}, scope, "within_cell_pcc_mean")
    model_overall = _metric_value(model_metrics, scope, "overall_pcc")
    processed_overall = _metric_value(processed_metrics, scope, "overall_pcc")
    proc_neighbor_corr = float(neighbor_audit["processed_true_profile_corr"].mean())
    raw_neighbor_corr = float(neighbor_audit["raw_true_profile_corr"].mean())
    non_monotone_fraction = float(perturb_metrics.get("non_monotone_fraction", float("nan")))
    max_abs_step_p95 = float(perturb_metrics.get("max_abs_step_p95", float("nan")))
    thresholds = {
        "processed_beats_model_within_cell_delta": 0.05,
        "processed_beats_model_overall_delta": 0.05,
        "neighbor_corr_advantage_delta": 0.02,
        "non_monotone_fraction_high": 0.25,
        "max_abs_step_p95_high": 0.15,
    }
    evidence: list[dict[str, Any]] = []
    if np.isfinite(processed_within_cell) and np.isfinite(model_within_cell) and processed_within_cell - model_within_cell > thresholds["processed_beats_model_within_cell_delta"]:
        evidence.append(
            {
                "type": "processed_transfer_beats_model_within_cell",
                "processed_within_cell_pcc_mean": processed_within_cell,
                "model_within_cell_pcc_mean": model_within_cell,
            }
        )
    if np.isfinite(processed_overall) and np.isfinite(model_overall) and processed_overall - model_overall > thresholds["processed_beats_model_overall_delta"]:
        evidence.append(
            {
                "type": "processed_transfer_beats_model_overall",
                "processed_overall_pcc": processed_overall,
                "model_overall_pcc": model_overall,
            }
        )
    if np.isfinite(proc_neighbor_corr) and np.isfinite(raw_neighbor_corr) and proc_neighbor_corr - raw_neighbor_corr > thresholds["neighbor_corr_advantage_delta"]:
        evidence.append(
            {
                "type": "processed_neighbors_more_profile_consistent_than_raw",
                "processed_true_profile_corr_mean": proc_neighbor_corr,
                "raw_true_profile_corr_mean": raw_neighbor_corr,
            }
        )
    if np.isfinite(non_monotone_fraction) and non_monotone_fraction > thresholds["non_monotone_fraction_high"]:
        evidence.append({"type": "local_path_non_monotone_fraction_high", "value": non_monotone_fraction})
    if np.isfinite(max_abs_step_p95) and max_abs_step_p95 > thresholds["max_abs_step_p95_high"]:
        evidence.append({"type": "local_path_step_jump_high", "value": max_abs_step_p95})
    smooth_ready = (
        np.isfinite(model_within_cell)
        and np.isfinite(processed_within_cell)
        and model_within_cell >= processed_within_cell - 0.02
        and (not np.isfinite(non_monotone_fraction) or non_monotone_fraction <= thresholds["non_monotone_fraction_high"])
        and (not np.isfinite(max_abs_step_p95) or max_abs_step_p95 <= thresholds["max_abs_step_p95_high"])
    )
    if smooth_ready and not any(item["type"].startswith("processed_transfer_beats_model") for item in evidence):
        label = "smooth_neighbor_use"
    elif any(item["type"].startswith("processed_transfer_beats_model") for item in evidence):
        label = "non_smooth_identity_like"
    else:
        label = "mixed_or_uncertain"
    return {
        "label": label,
        "scope": scope,
        "thresholds": thresholds,
        "evidence": evidence,
        "model_within_cell_pcc_mean": model_within_cell,
        "processed_transfer_within_cell_pcc_mean": processed_within_cell,
        "raw_transfer_within_cell_pcc_mean": raw_within_cell,
        "processed_neighbor_true_profile_corr_mean": proc_neighbor_corr,
        "raw_neighbor_true_profile_corr_mean": raw_neighbor_corr,
        "non_monotone_fraction": non_monotone_fraction,
        "max_abs_step_p95": max_abs_step_p95,
    }


def _plot_neighbor_smoothness(path: Path, neighbor_audit: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        neighbor_audit["processed_neighbor_distance"],
        neighbor_audit["processed_true_profile_corr"],
        alpha=0.7,
        label="processed-state",
    )
    if neighbor_audit["raw_true_profile_corr"].notna().any():
        ax.scatter(
            neighbor_audit["raw_neighbor_distance"],
            neighbor_audit["raw_true_profile_corr"],
            alpha=0.7,
            label="raw-expression",
        )
    ax.set_xlabel("Nearest Train Distance")
    ax.set_ylabel("True Response Profile Corr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_transfer_comparison(path: Path, summary: pd.DataFrame, scope: str) -> None:
    subset = summary.loc[
        summary["category"].eq("predictive_metrics")
        & summary["scope"].eq(scope)
        & summary["metric"].isin(["overall_pcc", "within_cell_pcc_mean"])
    ].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(index="source", columns="metric", values="value", aggfunc="first").fillna(np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Transfer Comparison ({scope})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_sensitivity_hist(path: Path, local_perturbation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if not local_perturbation.empty:
        max_jump = (
            local_perturbation.sort_values(["probe_row_id", "audit_alpha"])
            .groupby("probe_row_id", observed=True)["auc_hat"]
            .apply(lambda s: np.max(np.abs(np.diff(s.to_numpy(dtype=float)))) if len(s) > 1 else 0.0)
        )
        ax.hist(max_jump.to_numpy(dtype=float), bins=20, alpha=0.85)
    ax.set_xlabel("Max Absolute Step Change")
    ax.set_ylabel("Probe Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_cell_state(
    *,
    benchmark: BenchmarkConfig,
    prepared: PreparedData,
    paths: Paths,
    result_dir: Path,
    device: str,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    visual_dir = ensure_dir(result_dir / "visualizations")
    predictions = pd.read_csv(result_dir / "predictions.csv")
    predictions["SANGER_MODEL_ID"] = predictions["SANGER_MODEL_ID"].astype(str)
    prepared.responses["SANGER_MODEL_ID"] = prepared.responses["SANGER_MODEL_ID"].astype(str)
    model_metrics = {}
    heldout_pred = predictions.loc[predictions["split"].ne("train")].copy()
    heldout_pred["entity_id"] = heldout_pred["SANGER_MODEL_ID"].astype(str)
    for split, group in heldout_pred.groupby("split", observed=True):
        model_metrics[str(split)] = regression_metrics(group[["entity_id", "DRUG_ID", "AUC", "auc_hat"]].copy())
    model_metrics["heldout"] = regression_metrics(heldout_pred[["entity_id", "DRUG_ID", "AUC", "auc_hat"]].copy())
    raw_expr = None
    try:
        raw_expr = _load_raw_expression(paths, benchmark)
    except Exception:
        raw_expr = None
    neighbor_audit, neighbor_meta = _neighbor_profile_audit(prepared.responses.copy(), prepared.state_matrix.copy(), raw_expr)
    processed_transfer, processed_metrics = _knn_transfer_predictions(prepared.responses.copy(), prepared.state_matrix.copy())
    raw_transfer_metrics = None
    if raw_expr is not None and set(prepared.state_matrix.index).issubset(set(raw_expr.index)):
        _, raw_transfer_metrics = _knn_transfer_predictions(
            prepared.responses.copy(),
            raw_expr.loc[prepared.state_matrix.index].copy(),
        )
    nearest_neighbors = dict(
        zip(
            neighbor_audit["SANGER_MODEL_ID"].astype(str),
            neighbor_audit["processed_neighbor_cell"].astype(str),
        )
    )
    local_perturbation, perturb_metrics = _local_perturbation_audit(
        result_dir=result_dir,
        benchmark=benchmark,
        prepared=prepared,
        predictions=predictions,
        device=device,
        nearest_neighbors=nearest_neighbors,
    )
    summary = _build_summary_frame(
        model_metrics=model_metrics,
        processed_metrics=processed_metrics,
        raw_metrics=raw_transfer_metrics,
        neighbor_audit=neighbor_audit,
        perturb_metrics=perturb_metrics,
    )
    verdict = _build_verdict(
        model_metrics=model_metrics,
        processed_metrics=processed_metrics,
        raw_metrics=raw_transfer_metrics,
        neighbor_audit=neighbor_audit,
        perturb_metrics=perturb_metrics,
    )
    verdict["raw_expression_status"] = neighbor_meta["raw_expression_status"]
    neighbor_audit.to_csv(result_dir / "cell_state_neighbor_audit.csv", index=False)
    summary.to_csv(result_dir / "cell_state_sensitivity_summary.csv", index=False)
    local_perturbation.to_csv(result_dir / "cell_state_local_perturbation.csv", index=False)
    write_json(result_dir / "cell_state_verdict.json", verdict)
    _plot_neighbor_smoothness(visual_dir / "cell_state_neighbor_smoothness.png", neighbor_audit)
    _plot_transfer_comparison(
        visual_dir / "cell_state_transfer_comparison.png",
        summary,
        "test" if "test" in processed_metrics else "heldout",
    )
    _plot_sensitivity_hist(visual_dir / "cell_state_sensitivity_hist.png", local_perturbation)
    return {
        "neighbor_audit_rows": int(len(neighbor_audit)),
        "local_perturbation_rows": int(len(local_perturbation)),
        "verdict": verdict,
        "summary_rows": int(len(summary)),
    }
