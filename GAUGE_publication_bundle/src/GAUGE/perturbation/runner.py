from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..benchmarking import PerturbationConfig


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part for part in text.split("|") if part]
    if isinstance(parsed, list):
        return [int(x) for x in parsed]
    return [int(parsed)]


def _degree_lookup(kg_graph: Any | None) -> dict[int, int]:
    if kg_graph is None:
        return {}
    edge_table = getattr(kg_graph, "edge_table", pd.DataFrame())
    if edge_table.empty:
        return {}
    src_counts = edge_table["src"].astype(int).value_counts()
    dst_counts = edge_table["dst"].astype(int).value_counts()
    return src_counts.add(dst_counts, fill_value=0).astype(int).to_dict()


def _degree_bin(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 3:
        return "2_3"
    if value <= 7:
        return "4_7"
    return "8_plus"


def _calibrate_predictions(prediction_frame: pd.DataFrame, config: PerturbationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pred = prediction_frame.reset_index(drop=True).copy()
    val_dist_by_drug = {
        int(drug_id): group["auc_hat"].dropna().to_numpy(np.float32)
        for drug_id, group in pred.loc[pred["split"].eq("val")].groupby("DRUG_ID", observed=True)
    }
    train_val_dist_by_drug = {
        int(drug_id): group["auc_hat"].dropna().to_numpy(np.float32)
        for drug_id, group in pred.loc[pred["split"].isin(["train", "val"])].groupby("DRUG_ID", observed=True)
    }
    # Cross-drug fallback: population-level val distribution (for drug-split held-out drugs)
    global_val_dist = pred.loc[pred["split"].eq("val"), "auc_hat"].dropna().to_numpy(np.float32)
    global_train_val_dist = pred.loc[pred["split"].isin(["train", "val"]), "auc_hat"].dropna().to_numpy(np.float32)

    for pred_idx, row in pred.iterrows():
        drug_id = int(row["DRUG_ID"])
        auc_hat = float(row["auc_hat"])
        val_dist = val_dist_by_drug.get(drug_id, np.asarray([], dtype=np.float32))
        train_val_dist = train_val_dist_by_drug.get(drug_id, np.asarray([], dtype=np.float32))
        min_rows = int(config.min_calibration_rows)
        if len(val_dist) >= min_rows:
            dist = val_dist
            source = "val"
        elif len(train_val_dist) >= min_rows:
            dist = train_val_dist
            source = "train+val"
        elif len(global_val_dist) >= min_rows:
            # Drug-split fallback: compare against population-level val distribution
            dist = global_val_dist
            source = "global_val"
        elif len(global_train_val_dist) >= min_rows:
            dist = global_train_val_dist
            source = "global_train+val"
        else:
            dist = np.asarray([], dtype=np.float32)
            source = "uncalibrated"
        percentile = float(np.mean(dist <= auc_hat)) if dist.size else float("nan")
        if not np.isfinite(percentile):
            status = "uncalibrated"
        elif percentile >= float(config.resistant_percentile):
            status = "resistant"
        elif percentile >= float(config.partial_resistant_percentile):
            status = "partial_resistance"
        elif percentile <= float(config.sensitive_percentile):
            status = "sensitive"
        else:
            status = "intermediate"
        rows.append(
            {
                "prediction_index": int(pred_idx),
                "SANGER_MODEL_ID": row.get("SANGER_MODEL_ID"),
                "DRUG_ID": drug_id,
                "DRUG_NAME": row.get("DRUG_NAME"),
                "baseline_auc": auc_hat,
                "baseline_value": float(row.get("value_hat", float("nan"))),
                "uncertainty": float(row.get("uncertainty", float("nan"))),
                "resistance_percentile": percentile,
                "baseline_status": status,
                "calibration_source": source,
            }
        )
    return pd.DataFrame(rows)


def _build_edge_actions(edge_rows: pd.DataFrame, edge_ablation_rows: pd.DataFrame) -> pd.DataFrame:
    if edge_rows.empty or edge_ablation_rows.empty:
        return pd.DataFrame()
    merged = edge_ablation_rows.merge(
        edge_rows[
            [
                "prediction_index",
                "edge_id",
                "source_name",
                "edge_type",
                "relation",
                "src_node_id",
                "src_node_name",
                "src_node_type",
                "dst_node_id",
                "dst_node_name",
                "dst_node_type",
                "edge_attention",
            ]
        ],
        on=["prediction_index", "edge_id", "source_name", "edge_type"],
        how="left",
    ).copy()
    merged["action_level"] = "edge"
    merged["action_rank"] = (
        merged.groupby("prediction_index", observed=True)["edge_attention"].rank(method="first", ascending=False).astype(int)
    )
    merged["mechanism_id"] = "edge:" + merged["edge_id"].astype(int).astype(str)
    merged["mechanism_name"] = merged["dst_node_name"].where(
        merged["dst_node_name"].notna() & merged["dst_node_name"].astype(str).ne(""),
        merged["src_node_name"],
    )
    merged["mechanism_type"] = merged["edge_type"].astype(str)
    merged["attention_score"] = merged["edge_attention"].astype(float)
    merged["edge_ids"] = "[" + merged["edge_id"].astype(int).astype(str) + "]"
    merged["n_edges_masked"] = 1
    merged["action_source_name"] = merged["source_name"].astype(str)
    merged["match_key"] = merged["source_name"].astype(str)
    return merged


def _build_node_actions(node_ablation_rows: pd.DataFrame) -> pd.DataFrame:
    if node_ablation_rows.empty:
        return pd.DataFrame()
    out = node_ablation_rows.copy()
    out["action_level"] = "node"
    out["action_rank"] = out.groupby("prediction_index", observed=True)["attention_score"].rank(method="first", ascending=False).astype(int)
    out["mechanism_id"] = "node:" + out["node_id"].astype(int).astype(str)
    out["mechanism_name"] = out["node_name"].astype(str)
    out["mechanism_type"] = out["node_type"].astype(str)
    out["action_source_name"] = out["source_name"].astype(str)
    out["match_key"] = out["node_type"].astype(str)
    return out


def _edge_set_rows(actions: pd.DataFrame, kg_graph: Any | None) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    edge_table = getattr(kg_graph, "edge_table", pd.DataFrame()) if kg_graph is not None else pd.DataFrame()
    node_table = getattr(kg_graph, "node_table", pd.DataFrame()) if kg_graph is not None else pd.DataFrame()
    edge_lookup = edge_table.set_index("edge_id").to_dict("index") if not edge_table.empty and "edge_id" in edge_table.columns else {}
    node_lookup = node_table.set_index("node_id").to_dict("index") if not node_table.empty and "node_id" in node_table.columns else {}
    rows: list[dict[str, Any]] = []
    for action in actions.itertuples(index=False):
        for edge_id in _as_int_list(getattr(action, "edge_ids", None)):
            edge = edge_lookup.get(int(edge_id), {})
            src_id = int(edge.get("src", -1)) if edge else -1
            dst_id = int(edge.get("dst", -1)) if edge else -1
            src_info = node_lookup.get(src_id, {})
            dst_info = node_lookup.get(dst_id, {})
            rows.append(
                {
                    "action_id": getattr(action, "mechanism_id"),
                    "prediction_index": int(getattr(action, "prediction_index")),
                    "edge_id": int(edge_id),
                    "source_name": edge.get("source", getattr(action, "action_source_name", "")),
                    "edge_type": edge.get("edge_type", getattr(action, "mechanism_type", "")),
                    "relation": edge.get("relation", ""),
                    "src_node_id": src_id,
                    "src_node_name": src_info.get("name", ""),
                    "src_node_type": src_info.get("node_type", ""),
                    "dst_node_id": dst_id,
                    "dst_node_name": dst_info.get("name", ""),
                    "dst_node_type": dst_info.get("node_type", ""),
                    "edge_attention": getattr(action, "attention_score", float("nan")),
                    "edge_delta_auc_hat": getattr(action, "delta_auc_hat", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def _attach_random_controls(actions: pd.DataFrame, config: PerturbationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actions.empty:
        return actions.copy(), pd.DataFrame()
    rng = np.random.default_rng(0)
    action_records = list(actions.itertuples(index=False))
    grouped_by_prediction_level: dict[tuple[int, str], list[Any]] = {}
    grouped_by_level: dict[str, list[Any]] = {}
    for action in action_records:
        level = str(action.action_level)
        grouped_by_prediction_level.setdefault((int(action.prediction_index), level), []).append(action)
        grouped_by_level.setdefault(level, []).append(action)
    rows: list[dict[str, Any]] = []
    percentiles: list[float] = []
    random_controls = int(config.random_controls)
    for action in action_records:
        action_level = str(action.action_level)
        action_mechanism_id = str(action.mechanism_id)
        pool = [
            control
            for control in grouped_by_prediction_level.get((int(action.prediction_index), action_level), [])
            if str(control.mechanism_id) != action_mechanism_id
        ]
        if action_level == "edge" and bool(config.random_match_by_source):
            action_source = str(action.action_source_name)
            pool = [control for control in pool if str(control.action_source_name) == action_source]
        if action_level == "node" and bool(config.random_match_by_node_type):
            action_type = str(action.mechanism_type)
            pool = [control for control in pool if str(control.mechanism_type) == action_type]
        if bool(config.random_match_by_degree_bin):
            degree_bin = str(action.degree_bin)
            pool = [control for control in pool if str(control.degree_bin) == degree_bin]
        if not pool:
            pool = [
                control
                for control in grouped_by_level.get(action_level, [])
                if str(control.mechanism_id) != action_mechanism_id
            ]
        if len(pool) <= random_controls:
            sampled = pool
        else:
            pool_index = pd.Index(range(len(pool)))
            sampled_positions = pool_index.to_series().sample(
                n=random_controls,
                random_state=int(rng.integers(0, 2**31 - 1)),
                replace=False,
            ).to_numpy(dtype=np.int64)
            sampled = [pool[int(pos)] for pos in sampled_positions]
        sampled_delta = np.asarray(
            [float(control.delta_auc_hat) for control in sampled if pd.notna(control.delta_auc_hat)],
            dtype=np.float32,
        )
        percentile = float(np.mean(sampled_delta <= float(action.delta_auc_hat))) if sampled_delta.size else float("nan")
        percentiles.append(percentile)
        for control in sampled:
            rows.append(
                {
                    "prediction_index": int(action.prediction_index),
                    "mechanism_id": action_mechanism_id,
                    "action_level": action_level,
                    "control_mechanism_id": str(control.mechanism_id),
                    "control_match_key": str(getattr(control, "match_key", "")),
                    "control_delta_auc_hat": float(control.delta_auc_hat),
                    "selected_delta_auc_hat": float(action.delta_auc_hat),
                }
            )
    out = actions.copy()
    out["random_control_percentile"] = percentiles
    return out, pd.DataFrame(rows)


def _classify_actions(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions
    delta = actions["delta_auc_hat"].to_numpy(np.float64)
    rp = actions["random_control_percentile"].to_numpy(np.float64)
    finite_delta = np.isfinite(delta)
    finite_rp = np.isfinite(rp)
    directions = np.where(
        finite_delta & (delta > 1e-8),
        "supports_resistance",
        np.where(finite_delta & (np.abs(delta) <= 1e-8), "neutral", "supports_sensitivity"),
    )
    classes = np.where(
        finite_delta & (delta > 1e-8) & finite_rp & (rp >= 0.95),
        "faithful_resistance_mechanism",
        np.where(
            finite_delta & (delta > 1e-8),
            "hidden_influential",
            np.where(finite_delta & (np.abs(delta) <= 1e-8), "attention_only", "irrelevant"),
        ),
    )
    out = actions.copy()
    out["direction"] = directions
    out["faithfulness_class"] = classes
    return out


def _write_report(path: Path, frame: pd.DataFrame) -> None:
    lines: list[str] = []
    if frame.empty:
        lines.append("# Perturbation Mechanism Report")
        lines.append("")
        lines.append("No perturbation mechanisms were available for this run.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append("# Perturbation Mechanism Report")
    for (sample_id, drug_name), group in frame.groupby(["SANGER_MODEL_ID", "DRUG_NAME"], observed=True):
        top = group.sort_values(["final_score", "action_rank"], ascending=[False, True]).head(3)
        first = top.iloc[0]
        lines.append("")
        lines.append(f"## Case: {sample_id} - {drug_name}")
        lines.append(f"Baseline predicted AUC: {first['baseline_auc']:.6f}")
        lines.append(
            f"Drug-specific resistance percentile: {first['resistance_percentile']:.6f} ({first['baseline_status']})"
        )
        lines.append("Top faithful KG mechanisms:")
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            lines.append(
                f"{rank}. {row.mechanism_name}: attention={row.attention_score:.6f}, "
                f"ablation_gain={row.delta_auc_hat:.6f}, random_percentile={row.random_control_percentile:.6f}"
            )
        lines.append(
            "Boundary statement: This is a model reliance analysis, not experimental proof of therapeutic intervention."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan_perturbation_mechanisms(
    *,
    prediction_frame: pd.DataFrame,
    edge_rows: pd.DataFrame,
    edge_ablation_rows: pd.DataFrame,
    node_ablation_rows: pd.DataFrame,
    out_dir: Path,
    config: PerturbationConfig,
    kg_graph: Any | None = None,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    reference_frame: optional full benchmark predictions (with train/val/test splits).
    Used to provide cross-drug calibration for drug-split scenarios where the target
    drug is held out and has no within-drug val distribution.
    """
    perturbation_dir = Path(out_dir) / str(config.output_dir)
    perturbation_dir.mkdir(parents=True, exist_ok=True)
    # Merge reference frame to supply val distribution for novel drugs
    if reference_frame is not None and not reference_frame.empty:
        merged_frame = pd.concat([reference_frame, prediction_frame], ignore_index=True)
    else:
        merged_frame = prediction_frame
    calibration = _calibrate_predictions(merged_frame, config)
    # Filter calibration back to only the prediction_frame rows
    pred_ids = set(zip(prediction_frame["SANGER_MODEL_ID"], prediction_frame["DRUG_ID"].astype(int)))
    calibration = calibration[
        calibration.apply(lambda r: (r["SANGER_MODEL_ID"], int(r["DRUG_ID"])) in pred_ids, axis=1)
    ].reset_index(drop=True)
    calibration["prediction_index"] = range(len(calibration))
    degree_lookup = _degree_lookup(kg_graph)
    edge_actions = _build_edge_actions(edge_rows, edge_ablation_rows)
    node_actions = _build_node_actions(node_ablation_rows)
    actions = pd.concat([edge_actions, node_actions], ignore_index=True) if not edge_actions.empty or not node_actions.empty else pd.DataFrame()
    if actions.empty:
        calibration.to_csv(perturbation_dir / "drug_specific_calibration.csv", index=False)
        pd.DataFrame().to_csv(perturbation_dir / "kg_mechanism_ablation_by_prediction.csv", index=False)
        pd.DataFrame().to_csv(perturbation_dir / "kg_ablation_edge_sets.csv", index=False)
        pd.DataFrame().to_csv(perturbation_dir / "random_control_ablation.csv", index=False)
        _write_report(perturbation_dir / "mechanism_report.md", pd.DataFrame())
        return pd.DataFrame()
    actions = actions.merge(calibration, on=["prediction_index", "SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME"], how="left")
    actions = actions.reset_index(drop=True)
    actions["degree"] = 1
    if "node_id" in actions.columns:
        mask = actions["action_level"].astype(str).eq("node") & actions["node_id"].notna()
        actions.loc[mask, "degree"] = actions.loc[mask, "node_id"].astype(int).map(degree_lookup).fillna(0).astype(int)
    deg = actions["degree"].astype(int)
    actions["degree_bin"] = np.where(deg <= 1, "1", np.where(deg <= 3, "2_3", np.where(deg <= 7, "4_7", "8_plus")))
    actions, random_controls = _attach_random_controls(actions, config)
    actions = _classify_actions(actions)
    actions["final_score"] = (
        actions["delta_auc_hat"].clip(lower=0.0).fillna(0.0) * 0.6
        + actions["attention_score"].fillna(0.0) * 0.2
        + actions["random_control_percentile"].fillna(0.0) * 0.2
    )
    actions["rationale"] = (
        actions["action_level"].astype(str)
        + " mechanism "
        + actions["mechanism_name"].astype(str)
        + " changes predicted AUC by "
        + actions["delta_auc_hat"].map(lambda value: f"{value:.6f}")
        + " after KG masking."
    )
    edge_sets = _edge_set_rows(actions, kg_graph)
    calibration.to_csv(perturbation_dir / "drug_specific_calibration.csv", index=False)
    actions.to_csv(perturbation_dir / "kg_mechanism_ablation_by_prediction.csv", index=False)
    edge_sets.to_csv(perturbation_dir / "kg_ablation_edge_sets.csv", index=False)
    random_controls.to_csv(perturbation_dir / "random_control_ablation.csv", index=False)
    _write_report(perturbation_dir / "mechanism_report.md", actions)
    return actions
