"""Evaluation metrics for the GAUGE benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd

FUSION_CANDIDATES = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3)


def safe_corr(a, b, method: str = "pearson") -> float:
    s = pd.DataFrame({"a": np.asarray(a, dtype=float), "b": np.asarray(b, dtype=float)}).dropna()
    if len(s) < 2 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return float("nan")
    return float(s["a"].corr(s["b"], method=method))


def _group_corr(frame, group_col, x, y, *, method, min_points):
    """Mean within-group correlation (pairwise dropna, constant groups dropped)."""
    d = frame[[group_col, x, y]].dropna()
    if d.empty:
        return float("nan"), 0.0
    g = d[group_col]
    n = g.groupby(g, observed=True).transform("size")
    d = d.loc[n >= min_points]
    if d.empty:
        return float("nan"), 0.0
    g = d[group_col]
    xv, yv = d[x].astype(float), d[y].astype(float)
    if method == "spearman":
        xv = xv.groupby(g, observed=True).rank()
        yv = yv.groupby(g, observed=True).rank()
    xc = xv - xv.groupby(g, observed=True).transform("mean")
    yc = yv - yv.groupby(g, observed=True).transform("mean")
    num = (xc * yc).groupby(g, observed=True).sum()
    den = np.sqrt((xc * xc).groupby(g, observed=True).sum() * (yc * yc).groupby(g, observed=True).sum())
    r = (num / den).to_numpy()
    r = r[np.isfinite(r)]
    return (float(r.mean()) if r.size else float("nan")), float(r.size)


def decomposition_axes(frame: pd.DataFrame) -> dict[str, float]:
    """Between-drug / between-cell / interaction Pearson r of the prediction."""
    f = frame.dropna(subset=["AUC", "auc_hat"])
    dmy = f.groupby("DRUG_ID")["AUC"].mean()
    dmp = f.groupby("DRUG_ID")["auc_hat"].mean().reindex(dmy.index)
    cmy = f.groupby("SANGER_MODEL_ID")["AUC"].mean()
    cmp_ = f.groupby("SANGER_MODEL_ID")["auc_hat"].mean().reindex(cmy.index)

    def p(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    d, c = f["DRUG_ID"].to_numpy(), f["SANGER_MODEL_ID"].to_numpy()
    y, ph = f["AUC"].to_numpy(float), f["auc_hat"].to_numpy(float)
    ry = y - dmy.reindex(d).to_numpy() - cmy.reindex(c).to_numpy() + y.mean()
    rp = ph - dmp.reindex(d).to_numpy() - cmp_.reindex(c).to_numpy() + ph.mean()
    return {"between_drug_pcc": p(dmy, dmp), "between_cell_pcc": p(cmy, cmp_),
            "interaction_pcc": p(ry, rp)}


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["AUC"].astype(float).to_numpy()
    yh = frame["auc_hat"].astype(float).to_numpy()
    wds, wdsn = _group_corr(frame, "DRUG_ID", "AUC", "auc_hat", method="spearman", min_points=3)
    wdp, wdpn = _group_corr(frame, "DRUG_ID", "AUC", "auc_hat", method="pearson", min_points=2)
    wcp, wcpn = _group_corr(frame, "entity_id", "AUC", "auc_hat", method="pearson", min_points=2)
    return {
        "raw_auc_rmse": float(np.sqrt(np.mean((y - yh) ** 2))),
        "raw_auc_mae": float(np.mean(np.abs(y - yh))),
        "overall_pcc": safe_corr(frame["AUC"], frame["auc_hat"], method="pearson"),
        "within_drug_spearman_mean": wds, "within_drug_spearman_n_drugs": wdsn,
        "within_drug_pcc_mean": wdp, "within_drug_pcc_n_drugs": wdpn,
        "within_cell_pcc_mean": wcp, "within_cell_pcc_n_cells": wcpn,
    }


def value_metrics(frame: pd.DataFrame, target_col: str = "relative_value_eval") -> dict[str, float]:
    v = frame[[target_col, "value_hat"]].dropna()
    out = {"value_rmse": float("nan"), "value_spearman": float("nan")}
    if not v.empty:
        out["value_rmse"] = float(np.sqrt(np.mean(
            (v[target_col].to_numpy(float) - v["value_hat"].to_numpy(float)) ** 2)))
        out["value_spearman"] = safe_corr(v[target_col], v["value_hat"], method="spearman")
    m, n = _group_corr(frame.dropna(subset=[target_col, "value_hat"]), "DRUG_ID",
                       target_col, "value_hat", method="spearman", min_points=3)
    out["within_drug_value_spearman"] = m
    out["within_drug_value_n_drugs"] = n
    return out


def select_fusion_weight(pred: pd.DataFrame) -> tuple[float, float]:
    """Pick the cell-residual fusion weight on the validation split (overall PCC)."""
    y = pred["AUC"].astype(float).to_numpy()
    raw = pred["raw_auc_hat"].astype(float).to_numpy()
    res = pred["cell_residual_hat"].astype(float).to_numpy()
    yc = y - y.mean()
    yss = float(np.dot(yc, yc))
    best_w, best_m, best_rmse, found = FUSION_CANDIDATES[0], -np.inf, np.inf, False
    for w in FUSION_CANDIDATES:
        yh = raw + float(w) * res
        hc = yh - yh.mean()
        den = np.sqrt(yss * float(np.dot(hc, hc)))
        mv = float(np.dot(yc, hc) / den) if den > 0 else float("nan")
        rv = float(np.sqrt(np.mean((y - yh) ** 2)))
        if np.isfinite(mv) and (not found or mv > best_m or (np.isclose(mv, best_m) and rv < best_rmse)):
            best_w, best_m, best_rmse, found = float(w), mv, rv, True
    return best_w, (best_m if found else float("nan"))
