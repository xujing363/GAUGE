from __future__ import annotations

import numpy as np
import pandas as pd


def absolute_activity(auc_hat) -> np.ndarray:
    """Cross-drug-comparable activity, derived from the **absolute** predicted AUC.

        absolute_activity = 1 - AUC_hat

    AUC is a dose-response summary on a common scale for every compound, so it
    is comparable across drugs; lower AUC = more sensitive, hence the sign
    flip. The transform is deliberately **not** clipped: some screens (PRISM in
    particular) report AUC > 1 for compounds that leave a line growing faster
    than control, and clipping would collapse every such drug to a single tied
    value and destroy the ranking. Ordering by `absolute_activity` descending
    is therefore exactly ordering by absolute predicted AUC ascending; the
    number is on the AUC scale (typically ~[-1.5, 1]), not a probability.

    This exists because the relative sensitive value (RTV) **must not** be used
    to compare different drugs: RTV is defined as a within-drug percentile,

        RTV_{i,d} = 1 - rank_d(AUC_i) / N_d,

    i.e. it is rank-normalised *inside each drug's own reference distribution*.
    That normalisation deletes exactly the between-drug potency information a
    cross-drug ranking needs: a weak drug's best-responding cell line and a
    potent drug's best-responding cell line both score RTV ~ 1. RTV remains the
    right read-out for the question it was defined for -- "how does this sample
    respond to this drug relative to other samples given the same drug".
    """
    return 1.0 - np.asarray(auc_hat, dtype=float)


def _rank_planner_scores(frame: pd.DataFrame) -> pd.Series:
    # Treat missing scores as lowest priority instead of crashing on astype(int).
    return frame.groupby("entity_id")["planner_score"].rank(
        method="first",
        ascending=False,
        na_option="bottom",
    ).astype(int)


def rank_candidates(predictions: pd.DataFrame, lambda_u: float = 0.1, lambda_ood: float = 0.0) -> pd.DataFrame:
    """Rank candidate drugs against each other for each entity.

    The activity term is `absolute_activity(auc_hat)`, **not** the relative
    sensitive value (RTV): ranking different drugs against one another is a
    cross-drug comparison, and RTV is a within-drug percentile that is not
    comparable across drugs (see `absolute_activity`). With the default
    `lambda_u`/`lambda_ood` the ordering is that of absolute predicted AUC,
    ascending.
    """
    frame = predictions.copy()
    if "auc_hat" not in frame.columns:
        raise KeyError(
            "rank_candidates needs an 'auc_hat' column: cross-drug ranking is scored "
            "from the absolute predicted AUC, not from the within-drug 'value_hat' (RTV)."
        )
    if "ood_score" not in frame.columns:
        frame["ood_score"] = 0.0
    frame["absolute_activity"] = absolute_activity(frame["auc_hat"])
    frame["planner_score"] = frame["absolute_activity"] - lambda_u * frame["uncertainty"] - lambda_ood * frame["ood_score"]
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
