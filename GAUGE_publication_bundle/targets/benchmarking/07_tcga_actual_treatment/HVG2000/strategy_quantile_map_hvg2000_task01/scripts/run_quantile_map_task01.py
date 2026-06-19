from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import sparse
from statsmodels.stats.multitest import fdrcorrection

from GAUGE.data import load_gdsc_expression, tcga_actual_drugs, tcga_binary_episode_frame, tcga_os_frame
from GAUGE.external import (
    _align_projected_states,
    _drug_lookup,
    _fit_response_auc_fixed_effect,
    _one_sided_mann_whitney,
    _predict_many_pairs,
    _response_auc_metric_rows,
)
from GAUGE.features import FeatureArtifacts
from GAUGE.planner import rank_candidates
from GAUGE.repro import RUNTIME_PROFILE_STABLE, set_reproducible_runtime
from GAUGE.train import load_model
from GAUGE.utils import normalize_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quantile-map aligned TCGA task01 evaluation with a frozen HVG2000 source model.")
    parser.add_argument("--config", type=Path, default=Path("strategy_quantile_map_hvg2000_task01/config.yaml"))
    return parser


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_tcga_hvg_matrix(h5ad_path: Path, genes: list[str], batch_size: int) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    data = ad.read_h5ad(h5ad_path, backed="r")
    if "gene_name" in data.var.columns:
        names = data.var["gene_name"].astype(str).tolist()
    else:
        names = [str(x).split(".")[0] for x in data.var_names]
    gene_to_idx: dict[str, int] = {}
    for idx, gene in enumerate(names):
        if gene and gene not in gene_to_idx:
            gene_to_idx[gene] = idx
    present_pos = [(pos, gene_to_idx[gene]) for pos, gene in enumerate(genes) if gene in gene_to_idx]
    missing = [gene for gene in genes if gene not in gene_to_idx]
    target_pos = [x[0] for x in present_pos]
    source_idx = [x[1] for x in present_pos]
    matrix = np.zeros((data.n_obs, len(genes)), dtype=np.float32)
    for start in range(0, data.n_obs, batch_size):
        stop = min(start + batch_size, data.n_obs)
        if source_idx:
            x = data[start:stop, source_idx].X
            if sparse.issparse(x):
                x = x.toarray()
            matrix[start:stop, target_pos] = np.asarray(x, dtype=np.float32)
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    return matrix, obs, missing


def _quantile_map_gene(source_col: np.ndarray, target_col: np.ndarray) -> np.ndarray:
    src = np.asarray(source_col, dtype=np.float32)
    tgt = np.asarray(target_col, dtype=np.float32)
    src_sorted = np.sort(src)
    ranks = np.argsort(np.argsort(tgt, kind="mergesort"), kind="mergesort").astype(np.float32)
    quantiles = (ranks + 0.5) / max(len(tgt), 1)
    grid = np.linspace(0.0, 1.0, num=len(src_sorted), endpoint=False, dtype=np.float32) + (0.5 / max(len(src_sorted), 1))
    mapped = np.interp(quantiles, grid, src_sorted, left=src_sorted[0], right=src_sorted[-1])
    return mapped.astype(np.float32, copy=False)


def _quantile_map_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.empty_like(target, dtype=np.float32)
    for idx in range(target.shape[1]):
        out[:, idx] = _quantile_map_gene(source[:, idx], target[:, idx])
    return out


def _append_zero_state_padding(states_2d: np.ndarray, target_dim: int) -> np.ndarray:
    current_dim = int(states_2d.shape[1])
    if current_dim == int(target_dim):
        return states_2d.astype(np.float32, copy=False)
    if current_dim > int(target_dim):
        return states_2d[:, :target_dim].astype(np.float32, copy=False)
    pad = np.zeros((states_2d.shape[0], int(target_dim) - current_dim), dtype=np.float32)
    return np.concatenate([states_2d.astype(np.float32, copy=False), pad], axis=1)


def _fast_response(model, artifacts: FeatureArtifacts, obs: pd.DataFrame, states: np.ndarray, out_dir: Path, *, device: str, min_group_size: int) -> pd.DataFrame:
    lookup = _drug_lookup(artifacts)
    state_by_sample = {str(idx): states[i] for i, idx in enumerate(obs.index.to_list())}
    episodes, label_audit = tcga_binary_episode_frame(obs)
    label_audit.to_csv(out_dir / "tcga_response_label_audit.csv", index=False)
    entity_ids, state_rows, drugs_by_entity, kept_rows, unmapped_rows = [], [], [], [], []
    for row in episodes.itertuples(index=False):
        sample_id = str(row.sample_id)
        if sample_id not in state_by_sample:
            unmapped_rows.append({"episode_id": str(row.episode_id), "patient_id": str(row.patient_id), "sample_id": sample_id, "drug_name": str(row.drug_name), "drug_key": str(row.drug_key), "response": str(row.response), "issue": "missing_projected_state"})
            continue
        payload = lookup.get(str(row.drug_key))
        if payload is None:
            unmapped_rows.append({"episode_id": str(row.episode_id), "patient_id": str(row.patient_id), "sample_id": sample_id, "drug_name": str(row.drug_name), "drug_key": str(row.drug_key), "response": str(row.response), "issue": "unmapped_tcga_response_drug"})
            continue
        entity_ids.append(str(row.episode_id))
        state_rows.append(state_by_sample[sample_id])
        drugs_by_entity.append([payload])
        kept_rows.append({"episode_id": str(row.episode_id), "patient_id": str(row.patient_id), "sample_id": sample_id, "project_id": str(row.project_id), "response": str(row.response), "y": int(row.y), "DRUG_ID": int(payload["DRUG_ID"]), "DRUG_NAME": payload["DRUG_NAME"]})
    pd.DataFrame(unmapped_rows).to_csv(out_dir / "tcga_response_unmapped_episodes.csv", index=False)
    score_frame = _predict_many_pairs(model, np.vstack(state_rows).astype(np.float32, copy=False), entity_ids, drugs_by_entity, device=device, batch_size=4096).rename(columns={"entity_id": "episode_id"})
    meta = pd.DataFrame(kept_rows)
    episode_scores = meta.merge(score_frame, on=["episode_id", "DRUG_ID", "DRUG_NAME"], how="left")
    episode_scores.to_csv(out_dir / "tcga_response_auc_episode_scores.csv", index=False)
    per_drug_rows = []
    for drug_id, group in episode_scores.groupby("DRUG_ID", observed=True):
        pos = group.loc[group["y"].eq(1), "auc_hat"]
        neg = group.loc[group["y"].eq(0), "auc_hat"]
        n_pos = int(pos.notna().sum())
        n_neg = int(neg.notna().sum())
        eligible = n_pos >= min_group_size and n_neg >= min_group_size
        stat, p_val, cles = _one_sided_mann_whitney(pos, neg) if eligible else (float("nan"), float("nan"), float("nan"))
        pos_median = float(pd.to_numeric(pos, errors="coerce").dropna().median()) if n_pos else float("nan")
        neg_median = float(pd.to_numeric(neg, errors="coerce").dropna().median()) if n_neg else float("nan")
        per_drug_rows.append({"DRUG_ID": int(drug_id), "DRUG_NAME": str(group["DRUG_NAME"].iloc[0]), "n_total": int(len(group)), "n_pos": n_pos, "n_neg": n_neg, "eligible": bool(eligible), "responder_auc_hat_median": pos_median, "nonresponder_auc_hat_median": neg_median, "median_delta_auc_hat": float(pos_median - neg_median) if np.isfinite(pos_median) and np.isfinite(neg_median) else float("nan"), "u_statistic": stat, "p_one_sided_mwu_lt": p_val, "auc_response_discrimination": cles, "direction_match": bool(np.isfinite(pos_median) and np.isfinite(neg_median) and pos_median < neg_median)})
    by_drug = pd.DataFrame(per_drug_rows)
    eligible_mask = by_drug["eligible"].eq(True) & by_drug["p_one_sided_mwu_lt"].notna()
    by_drug["q_bh_lt"] = np.nan
    if bool(eligible_mask.any()):
        _, qvals = fdrcorrection(by_drug.loc[eligible_mask, "p_one_sided_mwu_lt"].astype(float).to_numpy())
        by_drug.loc[eligible_mask, "q_bh_lt"] = qvals
    by_drug.to_csv(out_dir / "tcga_response_auc_by_drug.csv", index=False)
    eligible_drugs = set(by_drug.loc[by_drug["eligible"].eq(True), "DRUG_ID"].astype(int).tolist()) if not by_drug.empty else set()
    pooled = _fit_response_auc_fixed_effect(episode_scores, eligible_drugs)
    pd.DataFrame(_response_auc_metric_rows(episode_scores, by_drug, pooled, min_group_size=min_group_size)).to_csv(out_dir / "tcga_response_auc_validation_metrics.csv", index=False)
    return episode_scores


def _evaluate_os(model, artifacts: FeatureArtifacts, obs: pd.DataFrame, states: np.ndarray, out_dir: Path, *, device: str, top_k: int) -> None:
    lookup = _drug_lookup(artifacts)
    os_df = tcga_os_frame(obs)
    actual = tcga_actual_drugs(obs)
    state_by_id = {str(obs.index[i]): states[i] for i in range(len(obs))}
    unmapped, entity_ids, state_rows, drugs_by_entity = [], [], [], []
    for patient_id in os_df.index:
        drugs = []
        for raw in actual.loc[patient_id]:
            key = normalize_name(raw)
            if key in lookup:
                drugs.append(lookup[key])
            elif key:
                unmapped.append({"entity_id": patient_id, "raw_drug": raw, "drug_key": key, "issue": "unmapped_tcga_actual_drug"})
        if not drugs:
            continue
        entity_ids.append(patient_id)
        state_rows.append(state_by_id[patient_id])
        drugs_by_entity.append(drugs)
    state_rows_arr = np.vstack(state_rows) if state_rows else np.empty((0, states.shape[1]), dtype=np.float32)
    scores = _predict_many_pairs(model, state_rows_arr, entity_ids, drugs_by_entity, device=device, batch_size=4096)
    if not scores.empty:
        scores = scores.merge(os_df.reset_index(names="entity_id"), on="entity_id", how="left")
    scores.to_csv(out_dir / "tcga_actual_treatment_scores.csv", index=False)
    pd.DataFrame(unmapped).to_csv(out_dir / "tcga_unmapped_actual_drugs.csv", index=False)
    metric_rows = []
    if not scores.empty:
        patient = scores.groupby("entity_id", observed=True).agg(v_hat_actual_mean=("value_hat", "mean"), time=("time", "first"), event=("event", "first"), age_at_diagnosis=("age_at_diagnosis", "first"), project_id=("project_id", "first"))
        candidate_drugs = list(lookup.values())
        candidate_scores = _predict_many_pairs(model, state_rows_arr, entity_ids, [candidate_drugs for _ in entity_ids], device=device)
        candidate_ranked = rank_candidates(candidate_scores, lambda_u=0.1) if not candidate_scores.empty else pd.DataFrame()
        top_by_patient = candidate_ranked.loc[candidate_ranked["planner_rank"].le(top_k)].groupby("entity_id", observed=True)["DRUG_ID"].apply(lambda col: set(col.astype(int))).to_dict()
        actual_by_patient = scores.groupby("entity_id", observed=True)["DRUG_ID"].apply(lambda col: set(col.astype(int))).to_dict()
        agreements = [bool(actual_by_patient.get(pid, set()) & top_by_patient.get(pid, set())) for pid in entity_ids]
        metric_rows.append({"analysis": f"policy_agreement_actual_drug_in_planner_top_{top_k}", "n": len(agreements), "agreement_rate": float(np.mean(agreements)) if agreements else float("nan")})
        cox_data = patient[["time", "event", "v_hat_actual_mean", "age_at_diagnosis", "project_id"]].dropna()
        if len(cox_data) >= 10 and cox_data["event"].sum() >= 3:
            try:
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col="time", event_col="event", strata=["project_id"], formula="v_hat_actual_mean + age_at_diagnosis")
                row = cph.summary.loc["v_hat_actual_mean"]
                metric_rows.append({"analysis": "cox_os_v_hat_actual_mean_age_project_strata", "n": len(cox_data), "events": int(cox_data["event"].sum()), "coef": float(row["coef"]), "hazard_ratio": float(row["exp(coef)"]), "p": float(row["p"])})
            except Exception as exc:
                metric_rows.append({"analysis": "cox_os_v_hat_actual_mean_age_project_strata", "status": f"failed: {exc}"})
            median = float(patient["v_hat_actual_mean"].median())
            high = patient["v_hat_actual_mean"].ge(median)
            try:
                lr = logrank_test(patient.loc[high, "time"], patient.loc[~high, "time"], patient.loc[high, "event"], patient.loc[~high, "event"])
                metric_rows.append({"analysis": "km_logrank_preregistered_median_v_hat_actual_mean", "n": len(patient), "events": int(patient["event"].sum()), "median_threshold": median, "p": float(lr.p_value), "test_statistic": float(lr.test_statistic)})
            except Exception as exc:
                metric_rows.append({"analysis": "km_logrank_preregistered_median_v_hat_actual_mean", "status": f"failed: {exc}"})
    pd.DataFrame(metric_rows).to_csv(out_dir / "tcga_os_survival_metrics.csv", index=False)


def _build_prediction_frame(model, artifacts: FeatureArtifacts, obs: pd.DataFrame, states: np.ndarray, *, device: str) -> pd.DataFrame:
    lookup = _drug_lookup(artifacts)
    entity_ids = obs.index.astype(str).tolist()
    candidate_drugs = list(lookup.values())
    rows = []
    step = 128
    for start in range(0, len(candidate_drugs), step):
        chunk = candidate_drugs[start : start + step]
        rows.append(_predict_many_pairs(model, states, entity_ids, [chunk for _ in range(len(entity_ids))], device=device, batch_size=4096))
    pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["entity_id", "DRUG_ID", "DRUG_NAME", "auc_hat", "value_hat", "uncertainty"])
    return pred


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    root = config_path.parents[1]
    cfg = _read_config(config_path)
    out_dir = (root / cfg["output_dirs"]["result_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_runtime(seed=0, device=cfg["device"], profile=RUNTIME_PROFILE_STABLE)

    with (Path(cfg["source_run_dir"]) / "artifacts.pkl").open("rb") as handle:
        artifacts: FeatureArtifacts = pickle.load(handle)
    model = load_model(Path(cfg["source_run_dir"]), artifacts, strict=True)

    gdsc_expr = load_gdsc_expression(Path(cfg["gdsc_expression"]), Path(cfg["gdsc_gene_identifiers"]))
    fit_cells = [cell for cell, role in artifacts.split_by_cell.items() if role == "fit" and cell in gdsc_expr.index]
    source_hvg = gdsc_expr.loc[fit_cells, artifacts.genes].reindex(columns=artifacts.genes, fill_value=0.0).astype(np.float32).to_numpy()
    tcga_hvg_raw, obs, missing_genes = _load_tcga_hvg_matrix(Path(cfg["tcga_h5ad"]), artifacts.genes, int(cfg["batch_size"]))
    tcga_hvg_aligned = _quantile_map_matrix(source_hvg, tcga_hvg_raw)

    imputed = artifacts.imputer.transform(tcga_hvg_aligned)
    scaled = artifacts.scaler.transform(imputed)
    states_core = artifacts.pca.transform(scaled).astype(np.float32, copy=False)
    states = _append_zero_state_padding(states_core, int(artifacts.state_dim or states_core.shape[1]))
    states = _align_projected_states(states, artifacts)

    response_scores = _fast_response(model, artifacts, obs, states, out_dir, device=cfg["device"], min_group_size=int(cfg["min_group_size"]))
    _evaluate_os(model, artifacts, obs, states, out_dir, device=cfg["device"], top_k=int(cfg["top_k"]))
    pred = _build_prediction_frame(model, artifacts, obs, states, device=cfg["device"])
    pred = pred.merge(obs.reset_index(names="entity_id")[["entity_id", "project_id"]], on="entity_id", how="left")
    pred.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame([{"analysis": "quantile_map_alignment", "source_run_dir": str(Path(cfg["source_run_dir"]).resolve()), "n_source_fit_cells": int(len(fit_cells)), "n_tcga_samples": int(len(obs)), "n_hvg_genes": int(len(artifacts.genes)), "n_missing_tcga_hvg_genes": int(len(missing_genes)), "response_episode_rows": int(len(response_scores))}]).to_csv(out_dir / "mapping_audit.csv", index=False)
    (out_dir / "strategy_summary.json").write_text(json.dumps({"strategy_name": cfg["strategy_name"], "source_run_dir": str(Path(cfg["source_run_dir"]).resolve()), "result_dir": str(out_dir), "device": cfg["device"], "n_source_fit_cells": int(len(fit_cells)), "n_tcga_samples": int(len(obs)), "n_hvg_genes": int(len(artifacts.genes)), "n_missing_tcga_hvg_genes": int(len(missing_genes))}, indent=2))
    print(f"[quantile-map task01] wrote strategy outputs to {out_dir}")


if __name__ == "__main__":
    main()
