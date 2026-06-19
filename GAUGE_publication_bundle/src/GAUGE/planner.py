from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_planner_scores(frame: pd.DataFrame) -> pd.Series:
    # Treat missing scores as lowest priority instead of crashing on astype(int).
    return frame.groupby("entity_id")["planner_score"].rank(
        method="first",
        ascending=False,
        na_option="bottom",
    ).astype(int)


def rank_candidates(predictions: pd.DataFrame, lambda_u: float = 0.1, lambda_ood: float = 0.0) -> pd.DataFrame:
    frame = predictions.copy()
    if "ood_score" not in frame.columns:
        frame["ood_score"] = 0.0
    frame["planner_score"] = frame["value_hat"] - lambda_u * frame["uncertainty"] - lambda_ood * frame["ood_score"]
    frame["planner_rank"] = _rank_planner_scores(frame)
    return frame.sort_values(["entity_id", "planner_rank"])


def unique_drug_ranked(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(["entity_id", "planner_score"], ascending=[True, False]).copy()
    ranked = ranked.drop_duplicates(subset=["entity_id", "DRUG_ID"], keep="first")
    ranked["planner_rank"] = _rank_planner_scores(ranked)
    return ranked.sort_values(["entity_id", "planner_rank"])


def observed_planning_metrics(
    pred: pd.DataFrame,
    top_k: int = 5,
    metric_field: str = "relative_value_eval",
) -> dict[str, float | str]:
    if metric_field not in pred.columns:
        return {
            "proxy_planning_gain_mean": float("nan"),
            "proxy_regret_mean": float("nan"),
            "proxy_topk_hit_rate": float("nan"),
            "proxy_planning_entities": 0.0,
            "proxy_top_k": float(top_k),
            "proxy_planning_status": f"missing_metric_field:{metric_field}",
        }
    gains = []
    regrets = []
    hits = []
    for _, group in pred.groupby("entity_id", observed=True):
        if group[metric_field].notna().sum() < 2:
            continue
        ranked = unique_drug_ranked(group)
        chosen = ranked.iloc[0]
        best = ranked.loc[ranked[metric_field].idxmax()]
        gains.append(float(chosen[metric_field] - ranked[metric_field].mean()))
        regrets.append(float(best[metric_field] - chosen[metric_field]))
        hits.append(bool(best["DRUG_ID"] in ranked.head(top_k)["DRUG_ID"].tolist()))
    return {
        "proxy_planning_gain_mean": float(np.mean(gains)) if gains else float("nan"),
        "proxy_regret_mean": float(np.mean(regrets)) if regrets else float("nan"),
        "proxy_topk_hit_rate": float(np.mean(hits)) if hits else float("nan"),
        "proxy_planning_entities": float(len(gains)),
        "proxy_top_k": float(top_k),
        "proxy_planning_status": "ok" if gains else f"insufficient_metric_values:{metric_field}",
    }
