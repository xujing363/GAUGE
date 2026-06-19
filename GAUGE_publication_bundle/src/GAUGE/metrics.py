from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from .utils import safe_corr


def _group_corr_stats(
    frame: pd.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    *,
    method: str,
    min_points: int,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for group_key, group in frame.groupby(group_col, observed=True):
        valid = group[[x_col, y_col]].dropna()
        n = int(len(valid))
        if n < min_points:
            continue
        corr = safe_corr(valid[x_col], valid[y_col], method=method)
        if np.isfinite(corr):
            rows.append({"group_key": group_key, "n": n, "corr": float(corr)})
    detail = pd.DataFrame(rows)
    mean = float(detail["corr"].mean()) if not detail.empty else float("nan")
    count = float(len(detail))
    return mean, count, detail


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["AUC"].astype(float).to_numpy()
    yhat = frame["auc_hat"].astype(float).to_numpy()
    overall_pcc = safe_corr(frame["AUC"], frame["auc_hat"], method="pearson")
    within_drug_spearman_mean, within_drug_spearman_n_drugs, _ = _group_corr_stats(
        frame,
        "DRUG_ID",
        "AUC",
        "auc_hat",
        method="spearman",
        min_points=3,
    )
    within_drug_pcc_mean, within_drug_pcc_n_drugs, _ = _group_corr_stats(
        frame,
        "DRUG_ID",
        "AUC",
        "auc_hat",
        method="pearson",
        min_points=2,
    )
    within_cell_pcc_mean, within_cell_pcc_n_cells, _ = _group_corr_stats(
        frame,
        "entity_id",
        "AUC",
        "auc_hat",
        method="pearson",
        min_points=2,
    )
    out = {
        "raw_auc_rmse": float(np.sqrt(skm.mean_squared_error(y, yhat))),
        "raw_auc_mae": float(skm.mean_absolute_error(y, yhat)),
        "overall_pcc": overall_pcc,
        "within_drug_spearman_mean": within_drug_spearman_mean,
        "within_drug_spearman_n_drugs": within_drug_spearman_n_drugs,
        "within_drug_pcc_mean": within_drug_pcc_mean,
        "within_drug_pcc_n_drugs": within_drug_pcc_n_drugs,
        "within_cell_pcc_mean": within_cell_pcc_mean,
        "within_cell_pcc_n_cells": within_cell_pcc_n_cells,
    }
    return out


def value_metrics(frame: pd.DataFrame, target_col: str = "relative_value_eval") -> dict[str, float]:
    valid = frame[[target_col, "value_hat"]].dropna()
    out = {
        "value_rmse": float("nan"),
        "value_spearman": float("nan"),
        "within_drug_value_spearman": float("nan"),
        "within_drug_value_n_drugs": 0.0,
    }
    if not valid.empty:
        y = valid[target_col].astype(float).to_numpy()
        yhat = valid["value_hat"].astype(float).to_numpy()
        out["value_rmse"] = float(np.sqrt(skm.mean_squared_error(y, yhat)))
        out["value_spearman"] = safe_corr(valid[target_col], valid["value_hat"], method="spearman")
    within_drug_value_spearman, within_drug_value_n_drugs, _ = _group_corr_stats(
        frame.dropna(subset=[target_col, "value_hat"]),
        "DRUG_ID",
        target_col,
        "value_hat",
        method="spearman",
        min_points=3,
    )
    out["within_drug_value_spearman"] = within_drug_value_spearman
    out["within_drug_value_n_drugs"] = within_drug_value_n_drugs
    return out


def groupwise_auc_correlations(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_key, group in frame.groupby(group_col, observed=True):
        valid = group[["AUC", "auc_hat"]].dropna()
        n = int(len(valid))
        if n < 2:
            continue
        pcc = safe_corr(valid["AUC"], valid["auc_hat"], method="pearson")
        if not np.isfinite(pcc):
            continue
        row = {"n": n, "pcc": float(pcc)}
        if group_col == "DRUG_ID":
            rho = safe_corr(valid["AUC"], valid["auc_hat"], method="spearman")
            row["spearman"] = float(rho) if np.isfinite(rho) else float("nan")
        row[group_col] = group_key
        rows.append(row)
    columns = [group_col, "n", "pcc"]
    if group_col == "DRUG_ID":
        columns.append("spearman")
    return pd.DataFrame(rows, columns=columns)


def binary_response_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    out: dict[str, float] = {"n": float(len(y_true)), "threshold": float(threshold)}
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(skm.roc_auc_score(y_true, score))
        out["auprc"] = float(skm.average_precision_score(y_true, score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    out["balanced_accuracy"] = float(skm.balanced_accuracy_score(y_true, score >= threshold))
    return out


def calibration_by_quantile(y_true: np.ndarray, score: np.ndarray, q: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"response": y_true, "score": score}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["quantile", "n", "score_mean", "response_rate"])
    bins = min(q, frame["score"].nunique())
    if bins < 2:
        frame["quantile"] = 0
    else:
        frame["quantile"] = pd.qcut(frame["score"], q=bins, labels=False, duplicates="drop")
    return (
        frame.groupby("quantile", observed=True)
        .agg(n=("response", "size"), score_mean=("score", "mean"), response_rate=("response", "mean"))
        .reset_index()
    )
