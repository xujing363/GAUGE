from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import regression_metrics


DEFAULT_FUSION_WEIGHT_CANDIDATES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)


def normalize_fusion_weight_candidates(candidates: Iterable[float] | None) -> tuple[float, ...]:
    if candidates is None:
        return DEFAULT_FUSION_WEIGHT_CANDIDATES
    normalized = tuple(float(x) for x in candidates)
    return normalized or DEFAULT_FUSION_WEIGHT_CANDIDATES


def fuse_auc_predictions(
    raw_auc_base: np.ndarray | pd.Series,
    residual_hat: np.ndarray | pd.Series,
    fusion_weight: float,
) -> np.ndarray:
    return np.asarray(raw_auc_base, dtype=np.float32) + float(fusion_weight) * np.asarray(residual_hat, dtype=np.float32)


def with_fused_auc(
    frame: pd.DataFrame,
    fusion_weight: float,
    *,
    raw_col: str = "raw_auc_base",
    centered_col: str = "cell_residual_hat",
    out_col: str = "auc_hat",
) -> pd.DataFrame:
    out = frame.copy()
    raw = out[raw_col] if raw_col in out.columns else out.get("raw_auc_hat", out["auc_hat"])
    residual_col = centered_col if centered_col in out.columns else "drug_centered_hat"
    out[out_col] = fuse_auc_predictions(raw, out[residual_col], fusion_weight)
    return out


def _metrics_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "entity_id" not in out.columns and "SANGER_MODEL_ID" in out.columns:
        out["entity_id"] = out["SANGER_MODEL_ID"].astype(str)
    if "entity_id" not in out.columns:
        out["entity_id"] = "entity_0"
    return out


def select_best_fusion_weight(
    frame: pd.DataFrame,
    *,
    candidates: Iterable[float] | None = None,
    split: str = "val",
    metric: str = "within_drug_pcc_mean",
) -> dict[str, Any]:
    split_frame = frame.loc[frame["split"].astype(str).eq(str(split))].copy()
    normalized_candidates = normalize_fusion_weight_candidates(candidates)
    candidate_metrics: dict[float, float] = {}
    best_weight = 1.0 if 1.0 in normalized_candidates else normalized_candidates[0]
    best_metric = float("-inf")
    best_rmse = float("inf")
    has_finite = False
    for candidate in normalized_candidates:
        scored = _metrics_ready_frame(with_fused_auc(split_frame, candidate))
        metrics = regression_metrics(scored) if not scored.empty else {}
        metric_value = float(metrics.get(metric, float("nan")))
        rmse_value = float(metrics.get("raw_auc_rmse", float("inf")))
        candidate_metrics[float(candidate)] = metric_value if np.isfinite(metric_value) else float("-inf")
        if np.isfinite(metric_value) and (
            not has_finite
            or metric_value > best_metric
            or (np.isclose(metric_value, best_metric) and rmse_value < best_rmse)
        ):
            best_metric = metric_value
            best_rmse = rmse_value
            best_weight = float(candidate)
            has_finite = True
    if not has_finite:
        best_metric = float("nan")
    return {
        "selected_weight": float(best_weight),
        "selected_metric": str(metric),
        "selected_split": str(split),
        "selected_metric_value": float(best_metric),
        "candidate_metrics": candidate_metrics,
    }


def build_cell_train_median_baseline(
    frame: pd.DataFrame,
    *,
    cell_col: str = "SANGER_MODEL_ID",
    split_col: str = "split",
    response_col: str = "AUC",
) -> pd.Series:
    train_rows = frame.loc[frame[split_col].astype(str).eq("train"), [cell_col, response_col]].copy()
    if train_rows.empty:
        return pd.Series(np.full((len(frame),), np.nan, dtype=np.float32), index=frame.index, dtype=np.float32)
    baseline = train_rows.groupby(cell_col, observed=True)[response_col].median().astype(np.float32)
    global_median = float(train_rows[response_col].median())
    return frame[cell_col].astype(str).map(baseline).fillna(global_median).astype(np.float32)


def audit_gdsc2_drug_level_predictions(
    predictions: pd.DataFrame,
    *,
    runtime: dict[str, Any] | None = None,
    fusion_weight: float | None = None,
    expected_total_rows: int = 227396,
    expected_test_rows: int = 45583,
    expected_test_drugs: int = 57,
) -> pd.DataFrame:
    frame = predictions.copy().reset_index(drop=True)
    actual_total_rows = int(len(frame))
    actual_test_rows = int(frame["split"].astype(str).eq("test").sum())
    actual_test_drugs = int(frame.loc[frame["split"].astype(str).eq("test"), "DRUG_ID"].astype(int).nunique())
    if actual_total_rows != int(expected_total_rows):
        raise ValueError(f"expected_total_rows mismatch: actual={actual_total_rows} expected_total_rows={expected_total_rows}")
    if actual_test_rows != int(expected_test_rows):
        raise ValueError(f"expected_test_rows mismatch: actual={actual_test_rows} expected_test_rows={expected_test_rows}")
    if actual_test_drugs != int(expected_test_drugs):
        raise ValueError(f"expected_test_drugs mismatch: actual={actual_test_drugs} expected_test_drugs={expected_test_drugs}")

    selected_fusion_weight = (
        float(fusion_weight)
        if fusion_weight is not None
        else float((runtime or {}).get("selected_fusion_weight", 1.0))
    )
    views: dict[str, pd.Series] = {
        "artifact_auc_hat": frame["auc_hat"].astype(np.float32),
        "raw_head": frame.get("raw_auc_base", frame.get("raw_auc_hat", frame["auc_hat"])).astype(np.float32),
        "fused_selected": pd.Series(
            fuse_auc_predictions(
                frame.get("raw_auc_base", frame.get("raw_auc_hat", frame["auc_hat"])),
                frame.get("cell_residual_hat", frame["drug_centered_hat"]),
                selected_fusion_weight,
            ),
            index=frame.index,
            dtype=np.float32,
        ),
        "cell_train_median_baseline": build_cell_train_median_baseline(frame),
    }
    if {"drug_centered_baseline_eval", "drug_centered_hat"} <= set(frame.columns):
        views["centered_head_eval"] = (
            frame["drug_centered_baseline_eval"].astype(np.float32) + frame["drug_centered_hat"].astype(np.float32)
        )
    if {"cell_residual_baseline_eval", "cell_residual_hat"} <= set(frame.columns):
        views["cell_residual_head_eval"] = (
            frame["cell_residual_baseline_eval"].astype(np.float32) + frame["cell_residual_hat"].astype(np.float32)
        )

    rows: list[dict[str, Any]] = []
    for prediction_view, auc_hat in views.items():
        scored = _metrics_ready_frame(frame.assign(auc_hat=np.asarray(auc_hat, dtype=np.float32)))
        for split_name, group in scored.groupby("split", observed=True):
            metrics = regression_metrics(group)
            rows.append(
                {
                    "prediction_view": prediction_view,
                    "split": split_name,
                    "n": int(len(group)),
                    "selected_fusion_weight": float(selected_fusion_weight),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def audit_gdsc1_drug_level_predictions(
    predictions: pd.DataFrame,
    *,
    runtime: dict[str, Any] | None = None,
    fusion_weight: float | None = None,
    expected_total_rows: int = 227396,
    expected_test_rows: int = 45583,
    expected_test_drugs: int = 57,
) -> pd.DataFrame:
    return audit_gdsc2_drug_level_predictions(
        predictions,
        runtime=runtime,
        fusion_weight=fusion_weight,
        expected_total_rows=expected_total_rows,
        expected_test_rows=expected_test_rows,
        expected_test_drugs=expected_test_drugs,
    )
