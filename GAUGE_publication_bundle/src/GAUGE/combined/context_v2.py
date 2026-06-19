from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import safe_corr


@dataclass(frozen=True)
class SelectedContext:
    cancer_type: str
    split: str
    selected_cells: list[str]
    context_report: pd.DataFrame
    context_row: dict[str, Any]


def _canonical_cancer(value: Any) -> str:
    return str(value or "").strip()


def _score_reliable_row(row: pd.Series, *, min_drug_rows: int, min_drug_pcc: float, min_drug_spearman: float, drug_rule: str) -> tuple[bool, str]:
    reasons: list[str] = []
    n = float(row.get("n", np.nan))
    pcc = float(row.get("pcc", np.nan))
    spearman = float(row.get("spearman", np.nan))
    if not np.isfinite(n) or n < min_drug_rows:
        reasons.append(f"n<{min_drug_rows}")
    pcc_ok = np.isfinite(pcc) and pcc >= min_drug_pcc
    spearman_ok = np.isfinite(spearman) and spearman >= min_drug_spearman
    if drug_rule == "pcc_or_spearman":
        metric_ok = pcc_ok or spearman_ok
        if not metric_ok:
            reasons.append(f"pcc<{min_drug_pcc} and spearman<{min_drug_spearman}")
    elif drug_rule == "pcc_and_spearman":
        metric_ok = pcc_ok and spearman_ok
        if not pcc_ok:
            reasons.append(f"pcc<{min_drug_pcc}")
        if not spearman_ok:
            reasons.append(f"spearman<{min_drug_spearman}")
    elif drug_rule == "pcc_only":
        metric_ok = pcc_ok
        if not pcc_ok:
            reasons.append(f"pcc<{min_drug_pcc}")
    elif drug_rule == "spearman_only":
        metric_ok = spearman_ok
        if not spearman_ok:
            reasons.append(f"spearman<{min_drug_spearman}")
    else:
        raise ValueError(f"Unsupported drug_rule: {drug_rule}")
    passed = not reasons and metric_ok
    return bool(passed), ";".join(reasons)


def select_reliable_drugs(
    drug_pcc_path: Path,
    *,
    split: str = "test",
    min_drug_rows: int = 50,
    min_drug_pcc: float = 0.50,
    min_drug_spearman: float = 0.50,
    drug_rule: str = "pcc_or_spearman",
) -> pd.DataFrame:
    frame = pd.read_csv(drug_pcc_path)
    frame = frame[frame["split"].astype(str).eq(split)].copy()
    if frame["DRUG_ID"].duplicated().any():
        dupes = sorted(frame.loc[frame["DRUG_ID"].duplicated(keep=False), "DRUG_ID"].astype(str).unique())
        raise ValueError(f"Duplicate DRUG_ID rows in reliable drug evidence for split={split}: {', '.join(dupes[:10])}")
    for col in ("DRUG_ID", "n", "pcc", "spearman"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    rows: list[tuple[bool, str]] = [
        _score_reliable_row(
            row,
            min_drug_rows=min_drug_rows,
            min_drug_pcc=min_drug_pcc,
            min_drug_spearman=min_drug_spearman,
            drug_rule=drug_rule,
        )
        for _, row in frame.iterrows()
    ]
    frame["passed_reliable_rule"] = [x[0] for x in rows]
    frame["fail_reason"] = [x[1] for x in rows]
    frame["drug_rule"] = drug_rule
    frame["min_drug_rows"] = int(min_drug_rows)
    frame["min_drug_pcc"] = float(min_drug_pcc)
    frame["min_drug_spearman"] = float(min_drug_spearman)
    return frame.sort_values(["passed_reliable_rule", "DRUG_ID"], ascending=[False, True]).reset_index(drop=True)


def _context_metrics(predictions: pd.DataFrame, cells: set[str], cancer_type: str, split: str) -> dict[str, Any]:
    sub = predictions[predictions["SANGER_MODEL_ID"].astype(str).isin(cells)].copy()
    if "AUC" in sub.columns and "value_hat" in sub.columns:
        internal_pcc = safe_corr(sub["AUC"], sub["value_hat"], method="pearson")
    elif "AUC" in sub.columns and "auc_hat" in sub.columns:
        internal_pcc = safe_corr(sub["AUC"], sub["auc_hat"], method="pearson")
    else:
        internal_pcc = float("nan")
    within_cell = [
        safe_corr(group["AUC"], group["value_hat"], method="pearson")
        for _, group in sub.groupby("SANGER_MODEL_ID")
        if "AUC" in group.columns and "value_hat" in group.columns
    ]
    within_drug = [
        safe_corr(group["AUC"], group["value_hat"], method="pearson")
        for _, group in sub.groupby("DRUG_ID")
        if "AUC" in group.columns and "value_hat" in group.columns
    ]
    return {
        "cancer_type": cancer_type,
        "split": split,
        "n_rows": int(len(sub)),
        "n_cells": int(sub["SANGER_MODEL_ID"].nunique()) if not sub.empty else 0,
        "n_drugs": int(sub["DRUG_ID"].nunique()) if not sub.empty else 0,
        "internal_pcc": internal_pcc,
        "within_cell_pcc_median": float(np.nanmedian(within_cell)) if within_cell else float("nan"),
        "within_drug_pcc_median": float(np.nanmedian(within_drug)) if within_drug else float("nan"),
    }


def select_context_from_gdsc_internal(
    *,
    predictions_path: Path,
    drug_pcc_path: Path,
    cell_pcc_path: Path,
    model_list_path: Path,
    cancer_type: str = "Melanoma",
    split: str = "test",
    min_context_cells: int = 5,
    min_context_rows: int = 500,
) -> SelectedContext:
    predictions = pd.read_csv(predictions_path)
    drug_pcc = pd.read_csv(drug_pcc_path)
    cell_pcc = pd.read_csv(cell_pcc_path)
    model_list = pd.read_csv(model_list_path)
    required = {"SANGER_MODEL_ID", "DRUG_ID", "split"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions.csv missing required columns for context selection: {missing}")
    predictions = predictions[predictions["split"].astype(str).eq(split)].copy()
    model_col = "model_id" if "model_id" in model_list.columns else "SANGER_MODEL_ID"
    if model_col not in model_list.columns or "cancer_type" not in model_list.columns:
        raise ValueError("model_list must contain model_id/SANGER_MODEL_ID and cancer_type")
    model_list = model_list[[model_col, "cancer_type"]].dropna(subset=[model_col, "cancer_type"]).copy()
    model_list[model_col] = model_list[model_col].astype(str)
    predictions["SANGER_MODEL_ID"] = predictions["SANGER_MODEL_ID"].astype(str)
    cancer_to_cells = {
        _canonical_cancer(name): set(group[model_col].astype(str))
        for name, group in model_list.groupby("cancer_type", dropna=True)
    }
    rows = [_context_metrics(predictions, cells, name, split) for name, cells in sorted(cancer_to_cells.items())]
    report = pd.DataFrame(rows).sort_values(["n_rows", "n_cells"], ascending=[False, False]).reset_index(drop=True)
    fixed_matches = [name for name in cancer_to_cells if name.lower() == cancer_type.lower()]
    if not fixed_matches:
        raise ValueError(f"Fixed cancer_type not found in model_list: {cancer_type}")
    selected_name = fixed_matches[0]
    selected_cells_all = cancer_to_cells[selected_name]
    selected_predictions = predictions[predictions["SANGER_MODEL_ID"].isin(selected_cells_all)]
    selected_cells = sorted(selected_predictions["SANGER_MODEL_ID"].unique().astype(str).tolist())
    context_row = report[report["cancer_type"].astype(str).str.lower().eq(selected_name.lower())]
    if context_row.empty:
        raise ValueError(f"No context report row for {selected_name}")
    payload = context_row.iloc[0].to_dict()
    if int(payload["n_cells"]) < min_context_cells or int(payload["n_rows"]) < min_context_rows:
        raise ValueError(
            f"Selected context {selected_name} fails thresholds: "
            f"n_cells={payload['n_cells']} min={min_context_cells}, n_rows={payload['n_rows']} min={min_context_rows}"
        )
    if drug_pcc.empty or cell_pcc.empty:
        raise ValueError("GDSC internal reliability files are required for context selection")
    return SelectedContext(
        cancer_type=selected_name,
        split=split,
        selected_cells=selected_cells,
        context_report=report,
        context_row=payload,
    )
