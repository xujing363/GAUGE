from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import hashlib
import numpy as np
import pandas as pd
import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import pairwise_distances
from scipy import sparse
from statsmodels.stats.multitest import fdrcorrection

from .cache import CacheManager, cache_key, file_signature
from .config import Paths
from .data import split_drug_list, tcga_actual_drugs, tcga_binary_episode_frame, tcga_os_frame
from .features import FeatureArtifacts, project_expression
from .metrics import binary_response_metrics, calibration_by_quantile
from .model import TerminalWorldModel
from .planner import rank_candidates
from .repro import RUNTIME_PROFILE_STABLE, set_reproducible_runtime
from .utils import normalize_name


def _drug_lookup(artifacts: FeatureArtifacts) -> dict[str, dict[str, Any]]:
    out = {}
    for row in artifacts.drug_table.itertuples(index=False):
        payload = {
            "DRUG_ID": int(row.DRUG_ID),
            "DRUG_NAME": row.DRUG_NAME,
            "fingerprint": row.fingerprint.astype(np.float32),
            "prior": row.prior.astype(np.float32),
            "prior_mask": float(row.prior_mask),
        }
        out[normalize_name(row.DRUG_NAME)] = payload
    return out


def _hash_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()[:24]


def _artifacts_signature(artifacts: FeatureArtifacts) -> dict[str, Any]:
    gene_hash = hashlib.sha256("\n".join(artifacts.genes).encode("utf-8")).hexdigest()[:24]
    return {
        "genes_hash": gene_hash,
        "n_genes": len(artifacts.genes),
        "imputer_statistics": _hash_array(np.asarray(artifacts.imputer.statistics_, dtype=np.float32)),
        "scaler_mean": _hash_array(np.asarray(artifacts.scaler.mean_, dtype=np.float32)),
        "scaler_scale": _hash_array(np.asarray(artifacts.scaler.scale_, dtype=np.float32)),
        "pca_components": _hash_array(np.asarray(artifacts.pca.components_, dtype=np.float32)),
        "pca_mean": _hash_array(np.asarray(artifacts.pca.mean_, dtype=np.float32)),
        "state_dim": int(getattr(artifacts, "state_dim", getattr(artifacts.pca, "n_components_", 0)) or 0),
    }


def _project_h5ad_states(
    h5ad_path: Path,
    artifacts: FeatureArtifacts,
    var_gene_name_col: str = "gene_name",
    batch_size: int = 512,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    data = ad.read_h5ad(h5ad_path, backed="r")
    if var_gene_name_col in data.var.columns:
        names = data.var[var_gene_name_col].astype(str).tolist()
    else:
        names = [str(x).split(".")[0] for x in data.var_names]
    gene_to_idx: dict[str, int] = {}
    for i, gene in enumerate(names):
        if gene and gene not in gene_to_idx:
            gene_to_idx[gene] = i
    present_pos = [(pos, gene_to_idx[gene]) for pos, gene in enumerate(artifacts.genes) if gene in gene_to_idx]
    missing = [gene for gene in artifacts.genes if gene not in gene_to_idx]
    target_pos = [x[0] for x in present_pos]
    source_idx = [x[1] for x in present_pos]
    projected = np.empty((data.n_obs, artifacts.pca.n_components_), dtype=np.float32)
    for start in range(0, data.n_obs, batch_size):
        stop = min(start + batch_size, data.n_obs)
        aligned = np.zeros((stop - start, len(artifacts.genes)), dtype=np.float32)
        if source_idx:
            x = data[start:stop, source_idx].X
            if sparse.issparse(x):
                x = x.toarray()
            aligned[:, target_pos] = np.asarray(x, dtype=np.float32)
        imputed = artifacts.imputer.transform(aligned)
        scaled = artifacts.scaler.transform(imputed)
        projected[start:stop] = artifacts.pca.transform(scaled).astype(np.float32, copy=False)
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    return projected if data.n_obs else np.empty((0, artifacts.pca.n_components_), dtype=np.float32), obs, missing


def _project_h5ad_cached(
    namespace: str,
    h5ad_path: Path,
    artifacts: FeatureArtifacts,
    cache: CacheManager,
    var_gene_name_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    key = cache_key(
        {
            "kind": "h5ad_projected_state",
            "namespace": namespace,
            "h5ad": file_signature(h5ad_path),
            "var_gene_name_col": var_gene_name_col,
            "artifacts": _artifacts_signature(artifacts),
        }
    )
    cached = cache.load_pickle(namespace, key, "states_obs.pkl")
    if cached is not None:
        return _align_projected_states(cached["states"], artifacts), cached["obs"]
    states, obs, missing = _project_h5ad_states(h5ad_path, artifacts, var_gene_name_col=var_gene_name_col)
    cache.save_pickle(namespace, key, "states_obs.pkl", {"states": states, "obs": obs, "missing_genes": missing})
    return _align_projected_states(states, artifacts), obs


def _align_projected_states(states: np.ndarray, artifacts: FeatureArtifacts) -> np.ndarray:
    target_dim = int(getattr(artifacts, "state_dim", 0) or getattr(artifacts.pca, "n_components_", 0) or states.shape[1])
    if states.shape[1] == target_dim:
        return states.astype(np.float32, copy=False)
    if states.shape[1] > target_dim:
        return np.asarray(states[:, :target_dim], dtype=np.float32)
    aligned = np.zeros((states.shape[0], target_dim), dtype=np.float32)
    if states.size:
        aligned[:, : states.shape[1]] = np.asarray(states, dtype=np.float32)
    return aligned


def _predict_pairs(
    model: TerminalWorldModel,
    states: np.ndarray,
    entity_ids: list[str],
    drugs: list[dict[str, Any]] | list[list[dict[str, Any]]],
    device: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    rows = []
    latents = []
    with torch.no_grad():
        for i, entity_id in enumerate(entity_ids):
            entity_drugs: list[dict[str, Any]]
            if drugs and isinstance(drugs[0], list):
                entity_drugs = drugs[i]  # type: ignore[index]
            else:
                entity_drugs = drugs  # type: ignore[assignment]
            if not entity_drugs:
                continue
            state = torch.tensor(np.repeat(states[i : i + 1], len(entity_drugs), axis=0), dtype=torch.float32, device=device)
            fp = torch.tensor(np.vstack([d["fingerprint"] for d in entity_drugs]), dtype=torch.float32, device=device)
            prior = torch.tensor(np.vstack([d["prior"] for d in entity_drugs]), dtype=torch.float32, device=device)
            mask = torch.tensor([[d["prior_mask"]] for d in entity_drugs], dtype=torch.float32, device=device)
            kg_idx = model.local_drug_indices([int(d["DRUG_ID"]) for d in entity_drugs], device=device)
            out = model(state, fp, prior, mask, drug_idx=kg_idx)
            for j, drug in enumerate(entity_drugs):
                rows.append(
                    {
                        "entity_id": entity_id,
                        "DRUG_ID": drug["DRUG_ID"],
                        "DRUG_NAME": drug["DRUG_NAME"],
                        "auc_hat": float(out["auc_hat"][j].detach().cpu()),
                        "value_hat": float(out["value_hat"][j].detach().cpu()),
                        "uncertainty": float(out["uncertainty"][j].detach().cpu()),
                    }
                )
            latents.append(out["terminal_latent"].detach().cpu().numpy())
    return pd.DataFrame(rows), np.vstack(latents) if latents else np.empty((0, 128), dtype=np.float32)


def _one_sided_mann_whitney(lower_group: pd.Series, higher_group: pd.Series) -> tuple[float, float, float]:
    left = pd.to_numeric(lower_group, errors="coerce").dropna().to_numpy(np.float64)
    right = pd.to_numeric(higher_group, errors="coerce").dropna().to_numpy(np.float64)
    if left.size == 0 or right.size == 0:
        return float("nan"), float("nan"), float("nan")
    stat, p = sps.mannwhitneyu(left, right, alternative="less")
    auc_like = float(stat / (left.size * right.size))
    common_language = float(1.0 - auc_like)
    return float(stat), float(p), common_language


def _response_auc_metric_rows(
    episode_scores: pd.DataFrame,
    by_drug: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    min_group_size: int,
) -> list[dict[str, Any]]:
    mapped_drugs = int(episode_scores["DRUG_ID"].nunique()) if not episode_scores.empty else 0
    eligible = by_drug.loc[by_drug["eligible"].eq(True)].copy() if not by_drug.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = [
        {
            "analysis": "tcga_response_auc_validation_support",
            "n_episodes": int(len(episode_scores)),
            "n_drugs_mapped": mapped_drugs,
            "eligible_drugs": int(len(eligible)),
            "min_group_size": int(min_group_size),
        }
    ]
    if not eligible.empty:
        rows.append(
            {
                "analysis": "tcga_response_auc_validation_direction",
                "eligible_drugs": int(len(eligible)),
                "direction_match_rate": float(eligible["direction_match"].mean()),
                "nominal_p_lt_0_05_count": int((eligible["p_one_sided_mwu_lt"].astype(float) < 0.05).sum()),
                "fdr_lt_0_05_count": int((eligible["q_bh_lt"].astype(float) < 0.05).sum()),
            }
        )
    else:
        rows.append(
            {
                "analysis": "tcga_response_auc_validation_direction",
                "status": "no eligible drugs for per-drug validation",
                "min_group_size": int(min_group_size),
            }
        )
    if not pooled.empty:
        pooled_row = pooled.iloc[0].to_dict()
        pooled_row["analysis"] = "tcga_response_auc_validation_pooled_fixed_effect"
        rows.append(pooled_row)
    else:
        rows.append(
            {
                "analysis": "tcga_response_auc_validation_pooled_fixed_effect",
                "status": "pooled fixed-effect model unavailable",
            }
        )
    return rows


def _fit_response_auc_fixed_effect(episode_scores: pd.DataFrame, eligible_drugs: set[int]) -> pd.DataFrame:
    frame = episode_scores.loc[episode_scores["DRUG_ID"].isin(eligible_drugs)].copy()
    if frame.empty or frame["y"].nunique() < 2 or frame["DRUG_ID"].nunique() < 1:
        return pd.DataFrame([{"status": "insufficient data for pooled fixed-effect model"}])
    try:
        fit = smf.glm("y ~ auc_hat + C(DRUG_ID)", data=frame, family=sm.families.Binomial()).fit()
    except Exception as exc:
        return pd.DataFrame([{"status": f"failed: {exc}"}])
    if "auc_hat" not in fit.params.index:
        return pd.DataFrame([{"status": "auc_hat coefficient missing from pooled fixed-effect model"}])
    coef = float(fit.params["auc_hat"])
    z_stat = float(fit.tvalues["auc_hat"])
    p_two_sided = float(fit.pvalues["auc_hat"])
    p_one_sided_lt = float(p_two_sided / 2.0) if coef < 0 else float(1.0 - (p_two_sided / 2.0))
    return pd.DataFrame(
        [
            {
                "status": "ok",
                "coef_auc_hat": coef,
                "odds_ratio_auc_hat": float(np.exp(coef)),
                "z_auc_hat": z_stat,
                "p_two_sided_auc_hat": p_two_sided,
                "p_one_sided_auc_hat_lt": p_one_sided_lt,
                "n_episodes": int(len(frame)),
                "n_drugs": int(frame["DRUG_ID"].nunique()),
            }
        ]
    )


def _evaluate_tcga_response_auc_validation(
    model: TerminalWorldModel,
    artifacts: FeatureArtifacts,
    obs: pd.DataFrame,
    states: np.ndarray,
    out_dir: Path,
    *,
    device: str,
    min_group_size: int = 3,
) -> pd.DataFrame:
    lookup = _drug_lookup(artifacts)
    state_by_sample = {str(idx): states[i] for i, idx in enumerate(obs.index.to_list())}
    episodes, label_audit = tcga_binary_episode_frame(obs)
    label_audit.to_csv(out_dir / "tcga_response_label_audit.csv", index=False)
    if episodes.empty:
        empty_scores = pd.DataFrame(
            columns=["episode_id", "patient_id", "sample_id", "project_id", "DRUG_ID", "DRUG_NAME", "response", "y", "auc_hat", "value_hat", "uncertainty"]
        )
        empty_scores.to_csv(out_dir / "tcga_response_auc_episode_scores.csv", index=False)
        pd.DataFrame(
            [{"analysis": "tcga_response_auc_validation", "status": "no labeled single-agent therapy episodes"}]
        ).to_csv(out_dir / "tcga_response_auc_validation_metrics.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "tcga_response_auc_by_drug.csv", index=False)
        return empty_scores
    entity_ids: list[str] = []
    state_rows: list[np.ndarray] = []
    drugs_by_entity: list[list[dict[str, Any]]] = []
    kept_rows: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    for row in episodes.itertuples(index=False):
        sample_id = str(row.sample_id)
        if sample_id not in state_by_sample:
            unmapped_rows.append(
                {
                    "episode_id": str(row.episode_id),
                    "patient_id": str(row.patient_id),
                    "sample_id": sample_id,
                    "drug_name": str(row.drug_name),
                    "drug_key": str(row.drug_key),
                    "response": str(row.response),
                    "issue": "missing_projected_state",
                }
            )
            continue
        payload = lookup.get(str(row.drug_key))
        if payload is None:
            unmapped_rows.append(
                {
                    "episode_id": str(row.episode_id),
                    "patient_id": str(row.patient_id),
                    "sample_id": sample_id,
                    "drug_name": str(row.drug_name),
                    "drug_key": str(row.drug_key),
                    "response": str(row.response),
                    "issue": "unmapped_tcga_response_drug",
                }
            )
            continue
        entity_ids.append(str(row.episode_id))
        state_rows.append(state_by_sample[sample_id])
        drugs_by_entity.append([payload])
        kept_rows.append(
            {
                "episode_id": str(row.episode_id),
                "patient_id": str(row.patient_id),
                "sample_id": sample_id,
                "project_id": str(row.project_id),
                "response": str(row.response),
                "y": int(row.y),
                "DRUG_ID": int(payload["DRUG_ID"]),
                "DRUG_NAME": payload["DRUG_NAME"],
            }
        )
    pd.DataFrame(
        unmapped_rows,
        columns=["episode_id", "patient_id", "sample_id", "drug_name", "drug_key", "response", "issue"],
    ).to_csv(out_dir / "tcga_response_unmapped_episodes.csv", index=False)
    if not kept_rows:
        empty_scores = pd.DataFrame(
            columns=["episode_id", "patient_id", "sample_id", "project_id", "DRUG_ID", "DRUG_NAME", "response", "y", "auc_hat", "value_hat", "uncertainty"]
        )
        empty_scores.to_csv(out_dir / "tcga_response_auc_episode_scores.csv", index=False)
        pd.DataFrame(
            [{"analysis": "tcga_response_auc_validation", "status": "no labeled mapped single-agent therapy episodes"}]
        ).to_csv(out_dir / "tcga_response_auc_validation_metrics.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "tcga_response_auc_by_drug.csv", index=False)
        return empty_scores
    score_frame, _ = _predict_pairs(
        model,
        np.vstack(state_rows).astype(np.float32, copy=False),
        entity_ids,
        drugs_by_entity,
        device=device,
    )
    meta = pd.DataFrame(kept_rows)
    score_frame = score_frame.rename(columns={"entity_id": "episode_id"})
    episode_scores = meta.merge(score_frame, on=["episode_id", "DRUG_ID", "DRUG_NAME"], how="left")
    episode_scores.to_csv(out_dir / "tcga_response_auc_episode_scores.csv", index=False)
    per_drug_rows: list[dict[str, Any]] = []
    for drug_id, group in episode_scores.groupby("DRUG_ID", observed=True):
        pos = group.loc[group["y"].eq(1), "auc_hat"]
        neg = group.loc[group["y"].eq(0), "auc_hat"]
        n_pos = int(pos.notna().sum())
        n_neg = int(neg.notna().sum())
        eligible = n_pos >= min_group_size and n_neg >= min_group_size
        stat, p_val, cles = _one_sided_mann_whitney(pos, neg) if eligible else (float("nan"), float("nan"), float("nan"))
        pos_median = float(pd.to_numeric(pos, errors="coerce").dropna().median()) if n_pos else float("nan")
        neg_median = float(pd.to_numeric(neg, errors="coerce").dropna().median()) if n_neg else float("nan")
        per_drug_rows.append(
            {
                "DRUG_ID": int(drug_id),
                "DRUG_NAME": str(group["DRUG_NAME"].iloc[0]),
                "n_total": int(len(group)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "eligible": bool(eligible),
                "responder_auc_hat_median": pos_median,
                "nonresponder_auc_hat_median": neg_median,
                "median_delta_auc_hat": float(pos_median - neg_median) if np.isfinite(pos_median) and np.isfinite(neg_median) else float("nan"),
                "u_statistic": stat,
                "p_one_sided_mwu_lt": p_val,
                "auc_response_discrimination": cles,
                "direction_match": bool(np.isfinite(pos_median) and np.isfinite(neg_median) and pos_median < neg_median),
            }
        )
    by_drug = pd.DataFrame(per_drug_rows)
    if not by_drug.empty:
        eligible_mask = by_drug["eligible"].eq(True) & by_drug["p_one_sided_mwu_lt"].notna()
        by_drug["q_bh_lt"] = np.nan
        if bool(eligible_mask.any()):
            _, qvals = fdrcorrection(by_drug.loc[eligible_mask, "p_one_sided_mwu_lt"].astype(float).to_numpy())
            by_drug.loc[eligible_mask, "q_bh_lt"] = qvals
    by_drug.to_csv(out_dir / "tcga_response_auc_by_drug.csv", index=False)
    eligible_drugs = set(by_drug.loc[by_drug["eligible"].eq(True), "DRUG_ID"].astype(int).tolist()) if not by_drug.empty else set()
    pooled = _fit_response_auc_fixed_effect(episode_scores, eligible_drugs)
    metric_rows = _response_auc_metric_rows(episode_scores, by_drug, pooled, min_group_size=min_group_size)
    pd.DataFrame(metric_rows).to_csv(out_dir / "tcga_response_auc_validation_metrics.csv", index=False)
    return episode_scores


def _predict_many_pairs(
    model: TerminalWorldModel,
    states: np.ndarray,
    entity_ids: list[str],
    drugs_by_entity: list[list[dict[str, Any]]],
    device: str,
    batch_size: int = 16384,
) -> pd.DataFrame:
    model = model.to(device).eval()
    total_pairs = sum(len(drugs) for drugs in drugs_by_entity)
    if total_pairs == 0:
        return pd.DataFrame()
    precomputed_kg_payload = model.precompute_kg_payload(device=device)
    entity_chunks: list[np.ndarray] = []
    drug_id_chunks: list[np.ndarray] = []
    drug_name_chunks: list[np.ndarray] = []
    auc_hat_chunks: list[np.ndarray] = []
    value_hat_chunks: list[np.ndarray] = []
    uncertainty_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in _iter_many_pair_chunks(states, entity_ids, drugs_by_entity, batch_size=batch_size):
            out = model(
                torch.as_tensor(chunk["state"], dtype=torch.float32, device=device),
                torch.as_tensor(chunk["fp"], dtype=torch.float32, device=device),
                torch.as_tensor(chunk["prior"], dtype=torch.float32, device=device),
                torch.as_tensor(chunk["mask"], dtype=torch.float32, device=device),
                drug_idx=model.local_drug_indices(chunk["drug_id"].astype(int).tolist(), device=device),
                precomputed_kg_payload=precomputed_kg_payload,
            )
            entity_chunks.append(chunk["entity_id"])
            drug_id_chunks.append(chunk["drug_id"])
            drug_name_chunks.append(chunk["drug_name"])
            auc_hat_chunks.append(out["auc_hat"].detach().cpu().numpy())
            value_hat_chunks.append(out["value_hat"].detach().cpu().numpy())
            uncertainty_chunks.append(out["uncertainty"].detach().cpu().numpy())
    return pd.DataFrame(
        {
            "entity_id": np.concatenate(entity_chunks) if entity_chunks else np.empty((0,), dtype=object),
            "DRUG_ID": np.concatenate(drug_id_chunks) if drug_id_chunks else np.empty((0,), dtype=np.int64),
            "DRUG_NAME": np.concatenate(drug_name_chunks) if drug_name_chunks else np.empty((0,), dtype=object),
            "auc_hat": np.concatenate(auc_hat_chunks) if auc_hat_chunks else np.empty((0,), dtype=np.float32),
            "value_hat": np.concatenate(value_hat_chunks) if value_hat_chunks else np.empty((0,), dtype=np.float32),
            "uncertainty": np.concatenate(uncertainty_chunks) if uncertainty_chunks else np.empty((0,), dtype=np.float32),
        }
    )


def _iter_many_pair_chunks(
    states: np.ndarray,
    entity_ids: list[str],
    drugs_by_entity: list[list[dict[str, Any]]],
    *,
    batch_size: int,
):
    state_parts: list[np.ndarray] = []
    fp_parts: list[np.ndarray] = []
    prior_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    entity_parts: list[np.ndarray] = []
    drug_id_parts: list[np.ndarray] = []
    drug_name_parts: list[np.ndarray] = []
    rows = 0

    def _flush():
        nonlocal state_parts, fp_parts, prior_parts, mask_parts, entity_parts, drug_id_parts, drug_name_parts, rows
        chunk = {
            "state": np.concatenate(state_parts, axis=0) if state_parts else np.empty((0, states.shape[1]), dtype=np.float32),
            "fp": np.concatenate(fp_parts, axis=0) if fp_parts else np.empty((0, 2048), dtype=np.float32),
            "prior": np.concatenate(prior_parts, axis=0) if prior_parts else np.empty((0, 0), dtype=np.float32),
            "mask": np.concatenate(mask_parts, axis=0) if mask_parts else np.empty((0, 1), dtype=np.float32),
            "entity_id": np.concatenate(entity_parts, axis=0) if entity_parts else np.empty((0,), dtype=object),
            "drug_id": np.concatenate(drug_id_parts, axis=0) if drug_id_parts else np.empty((0,), dtype=np.int64),
            "drug_name": np.concatenate(drug_name_parts, axis=0) if drug_name_parts else np.empty((0,), dtype=object),
        }
        state_parts, fp_parts, prior_parts, mask_parts = [], [], [], []
        entity_parts, drug_id_parts, drug_name_parts = [], [], []
        rows = 0
        return chunk

    for i, drugs in enumerate(drugs_by_entity):
        if not drugs:
            continue
        start = 0
        while start < len(drugs):
            remaining = max(batch_size - rows, 1)
            sub = drugs[start : start + remaining]
            take = len(sub)
            state_parts.append(np.repeat(states[i : i + 1], take, axis=0).astype(np.float32, copy=False))
            fp_parts.append(np.vstack([drug["fingerprint"] for drug in sub]).astype(np.float32, copy=False))
            prior_dim = len(sub[0]["prior"]) if sub else 0
            prior_parts.append(
                np.vstack([drug["prior"] for drug in sub]).astype(np.float32, copy=False)
                if prior_dim
                else np.empty((take, 0), dtype=np.float32)
            )
            mask_parts.append(np.asarray([[drug["prior_mask"]] for drug in sub], dtype=np.float32))
            entity_parts.append(np.asarray([entity_ids[i]] * take, dtype=object))
            drug_id_parts.append(np.asarray([drug["DRUG_ID"] for drug in sub], dtype=np.int64))
            drug_name_parts.append(np.asarray([drug["DRUG_NAME"] for drug in sub], dtype=object))
            rows += take
            start += take
            if rows >= batch_size:
                yield _flush()
    if rows > 0:
        yield _flush()


def _ood_score(train_states: np.ndarray, query_states: np.ndarray) -> np.ndarray:
    if len(train_states) == 0 or len(query_states) == 0:
        return np.zeros((len(query_states),), dtype=np.float32)
    if torch.cuda.is_available():
        train = torch.as_tensor(train_states, dtype=torch.float32, device="cuda")
        mins = []
        step = 4096
        with torch.no_grad():
            for start in range(0, len(query_states), step):
                query = torch.as_tensor(query_states[start : start + step], dtype=torch.float32, device="cuda")
                mins.append(torch.cdist(query, train).amin(dim=1).detach().cpu().numpy())
        return np.concatenate(mins).astype(np.float32, copy=False)
    d = pairwise_distances(query_states, train_states, metric="euclidean")
    return d.min(axis=1).astype(np.float32)


def evaluate_tcga_actual_treatments(
    model: TerminalWorldModel,
    artifacts: FeatureArtifacts,
    paths: Paths,
    out_dir: Path,
    top_k: int = 5,
    device: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
    export_terminal_latents: bool = False,
    terminal_latent_path: Path | None = None,
) -> pd.DataFrame:
    if device is None:
        raise ValueError("Explicit --device is required for evaluate/run. Use --device cuda:N or --device cpu.")
    set_reproducible_runtime(seed=0, device=device, profile=runtime_profile)
    cache = CacheManager(cache_dir or out_dir / ".cache", use_cache=use_cache, rebuild_cache=rebuild_cache)
    states, obs = _project_h5ad_cached("tcga_projection", paths.tcga_h5ad, artifacts, cache, var_gene_name_col="gene_name")
    response_episode_scores = _evaluate_tcga_response_auc_validation(
        model,
        artifacts,
        obs,
        states,
        out_dir,
        device=device,
    )
    os_df = tcga_os_frame(obs)
    actual = tcga_actual_drugs(obs)
    lookup = _drug_lookup(artifacts)
    obs_ids = obs.index.to_list()
    state_by_id = {idx: states[i] for i, idx in enumerate(obs_ids)}
    unmapped = []
    entity_ids = []
    state_rows = []
    drugs_by_entity = []
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
    scores, latents = _predict_pairs(model, state_rows_arr, entity_ids, drugs_by_entity, device=device)
    if export_terminal_latents and terminal_latent_path is not None and latents.size:
        np.save(terminal_latent_path, latents.astype(np.float32, copy=False))
    if not scores.empty:
        scores = scores.merge(os_df.reset_index(names="entity_id"), on="entity_id", how="left")
        scores.to_csv(out_dir / "tcga_actual_treatment_scores.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "tcga_actual_treatment_scores.csv", index=False)
        metric_rows = [{"analysis": "tcga_os", "status": "insufficient mapped actual-treatment OS samples"}]
        response_metrics_path = out_dir / "tcga_response_auc_validation_metrics.csv"
        if response_metrics_path.exists():
            response_metrics = pd.read_csv(response_metrics_path)
            if not response_metrics.empty:
                metric_rows.extend(response_metrics.to_dict("records"))
        pd.DataFrame(metric_rows).to_csv(out_dir / "tcga_os_survival_metrics.csv", index=False)
        pd.DataFrame(unmapped, columns=["entity_id", "raw_drug", "drug_key", "issue"]).to_csv(out_dir / "tcga_unmapped_actual_drugs.csv", index=False)
        cache.write_reports(out_dir)
        return scores
    pd.DataFrame(unmapped, columns=["entity_id", "raw_drug", "drug_key", "issue"]).to_csv(out_dir / "tcga_unmapped_actual_drugs.csv", index=False)
    patient = scores.groupby("entity_id", observed=True).agg(
        v_hat_actual_mean=("value_hat", "mean"),
        v_hat_actual_max=("value_hat", "max"),
        time=("time", "first"),
        event=("event", "first"),
        age_at_diagnosis=("age_at_diagnosis", "first"),
        project_id=("project_id", "first"),
    )
    metric_rows = []
    if not scores.empty:
        candidate_drugs = list(lookup.values())
        candidate_scores = _predict_many_pairs(
            model,
            state_rows_arr,
            entity_ids,
            [candidate_drugs for _ in entity_ids],
            device=device,
        )
        candidate_ranked = rank_candidates(candidate_scores, lambda_u=0.1) if not candidate_scores.empty else pd.DataFrame()
        top_by_patient = (
            candidate_ranked.loc[candidate_ranked["planner_rank"].le(top_k)]
            .groupby("entity_id", observed=True)["DRUG_ID"]
            .apply(lambda col: set(col.astype(int)))
            .to_dict()
        )
        actual_by_patient = (
            scores.groupby("entity_id", observed=True)["DRUG_ID"]
            .apply(lambda col: set(col.astype(int)))
            .to_dict()
        )
        agreements = [
            bool(actual_by_patient.get(patient_id, set()) & top_by_patient.get(patient_id, set()))
            for patient_id in entity_ids
        ]
        metric_rows.append(
            {
                "analysis": f"policy_agreement_actual_drug_in_planner_top_{top_k}",
                "n": len(agreements),
                "agreement_rate": float(np.mean(agreements)) if agreements else float("nan"),
            }
        )
    if len(patient) >= 10 and patient["event"].sum() >= 3:
        cox_data = patient[["time", "event", "v_hat_actual_mean", "age_at_diagnosis", "project_id"]].dropna()
        try:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col="time", event_col="event", strata=["project_id"], formula="v_hat_actual_mean + age_at_diagnosis")
            row = cph.summary.loc["v_hat_actual_mean"]
            metric_rows.append(
                {
                    "analysis": "cox_os_v_hat_actual_mean_age_project_strata",
                    "n": len(cox_data),
                    "events": int(cox_data["event"].sum()),
                    "coef": float(row["coef"]),
                    "hazard_ratio": float(row["exp(coef)"]),
                    "p": float(row["p"]),
                }
            )
        except Exception as exc:
            metric_rows.append({"analysis": "cox_os_v_hat_actual_mean_age_project_strata", "status": f"failed: {exc}"})
        median = float(patient["v_hat_actual_mean"].median())
        high = patient["v_hat_actual_mean"].ge(median)
        try:
            lr = logrank_test(
                patient.loc[high, "time"],
                patient.loc[~high, "time"],
                patient.loc[high, "event"],
                patient.loc[~high, "event"],
            )
            metric_rows.append(
                {
                    "analysis": "km_logrank_preregistered_median_v_hat_actual_mean",
                    "n": len(patient),
                    "events": int(patient["event"].sum()),
                    "median_threshold": median,
                    "p": float(lr.p_value),
                    "test_statistic": float(lr.test_statistic),
                }
            )
        except Exception as exc:
            metric_rows.append({"analysis": "km_logrank_preregistered_median_v_hat_actual_mean", "status": f"failed: {exc}"})
    else:
        metric_rows.append({"analysis": "tcga_os", "status": "insufficient mapped actual-treatment OS samples"})
    if not response_episode_scores.empty:
        metric_rows.append(
            {
                "analysis": "tcga_response_auc_validation_support_n_episodes",
                "n": int(len(response_episode_scores)),
                "n_drugs": int(response_episode_scores["DRUG_ID"].nunique()),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "tcga_os_survival_metrics.csv", index=False)
    cache.write_reports(out_dir)
    return scores


def evaluate_ctrdb_response(
    model: TerminalWorldModel,
    artifacts: FeatureArtifacts,
    paths: Paths,
    out_dir: Path,
    device: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
) -> pd.DataFrame:
    if device is None:
        raise ValueError("Explicit --device is required for evaluate/run. Use --device cuda:N or --device cpu.")
    set_reproducible_runtime(seed=0, device=device, profile=runtime_profile)
    cache = CacheManager(cache_dir or out_dir / ".cache", use_cache=use_cache, rebuild_cache=rebuild_cache)
    states, obs = _project_h5ad_cached("ctrdb_projection", paths.ctrdb_microarray_h5ad, artifacts, cache, var_gene_name_col="__none__")
    lookup = _drug_lookup(artifacts)
    unmapped = []
    entity_ids = []
    state_rows = []
    drugs_by_entity = []
    responses = []
    for i, sample_id in enumerate(obs.index):
        raw_list = obs.loc[sample_id, "Drug_list"] if "Drug_list" in obs.columns else None
        drugs = []
        for raw in split_drug_list(raw_list):
            key = normalize_name(raw)
            if key in lookup:
                drugs.append(lookup[key])
            elif key:
                unmapped.append({"entity_id": sample_id, "raw_drug": raw, "drug_key": key, "issue": "unmapped_ctrdb_drug"})
        if not drugs:
            continue
        entity_ids.append(sample_id)
        state_rows.append(states[i])
        drugs_by_entity.append(drugs)
        responses.append(str(obs.loc[sample_id, "Response"]))
    state_rows_arr = np.vstack(state_rows) if state_rows else np.empty((0, states.shape[1]), dtype=np.float32)
    scores = _predict_many_pairs(model, state_rows_arr, entity_ids, drugs_by_entity, device=device)
    if not scores.empty:
        response_map = dict(zip(entity_ids, responses))
        scores["Response"] = scores["entity_id"].map(response_map)
    scores.to_csv(out_dir / "ctrdb_response_scores.csv", index=False)
    pd.DataFrame(unmapped, columns=["entity_id", "raw_drug", "drug_key", "issue"]).to_csv(out_dir / "ctrdb_unmapped_drugs.csv", index=False)
    metric_rows = []
    if not scores.empty:
        sample = scores.groupby("entity_id", observed=True).agg(
            value_max=("value_hat", "max"), value_mean=("value_hat", "mean"), response=("Response", "first")
        )
        sample["y"] = sample["response"].eq("Response").astype(int)
        for agg in ["max", "mean"]:
            vals = binary_response_metrics(sample["y"].to_numpy(), sample[f"value_{agg}"].to_numpy(), threshold=0.5)
            vals["analysis"] = f"ctrdb_response_value_{agg}"
            metric_rows.append(vals)
            calibration_by_quantile(sample["y"].to_numpy(), sample[f"value_{agg}"].to_numpy()).to_csv(
                out_dir / f"ctrdb_calibration_value_{agg}.csv", index=False
            )
    else:
        metric_rows.append({"analysis": "ctrdb_response", "status": "no samples with GDSC action-set mapped drugs"})
    pd.DataFrame(metric_rows).to_csv(out_dir / "ctrdb_response_metrics.csv", index=False)
    cache.write_reports(out_dir)
    return scores
