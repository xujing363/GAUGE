from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import weakref

import numpy as np
import pandas as pd
import torch


EXPLANATION_LEVELS = ("source", "node", "edge", "path")
_KG_EXPLAIN_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[Any],
        dict[int, pd.DataFrame],
        dict[int, dict[str, pd.DataFrame]],
        dict[int, dict[str, Any]],
        dict[int, dict[str, dict[str, np.ndarray]]],
    ],
] = {}


@dataclass
class KGExplanationBundle:
    kg_gate: torch.Tensor | None = None
    kg_total_contribution: torch.Tensor | None = None
    kg_source_attention: torch.Tensor | None = None
    kg_node_attention: torch.Tensor | None = None
    kg_edge_attention: torch.Tensor | None = None
    kg_message_norm: torch.Tensor | None = None
    kg_edge_ids: torch.Tensor | None = None
    kg_node_ids: torch.Tensor | None = None
    kg_source_names: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_explanation_level(value: str | None) -> str:
    level = str(value or "source").strip().lower()
    if level not in EXPLANATION_LEVELS:
        raise ValueError(f"explanation_level must be one of {', '.join(EXPLANATION_LEVELS)}; got {value!r}")
    return level


def tensor_to_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return value.detach().cpu().numpy()


def ensure_explain_dir(out_dir: Path) -> dict[str, Path]:
    explain_dir = Path(out_dir) / "explain"
    summary_dir = explain_dir / "summary"
    cards_dir = explain_dir / "cards"
    figures_dir = explain_dir / "figures"
    for path in (explain_dir, summary_dir, cards_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": explain_dir, "summary": summary_dir, "cards": cards_dir, "figures": figures_dir}


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    arr = np.asarray(values, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / total


def _get_kg_explain_cache(
    kg_graph: Any | None,
) -> tuple[
    dict[int, pd.DataFrame],
    dict[int, dict[str, pd.DataFrame]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, dict[str, np.ndarray]]],
]:
    if kg_graph is None:
        return {}, {}, {}, {}
    cache_key = id(kg_graph)
    cached = _KG_EXPLAIN_CACHE.get(cache_key)
    if cached is not None:
        kg_ref, edges_by_drug, edges_by_source_cache, node_lookup, source_array_cache = cached
        if kg_ref() is kg_graph:
            return edges_by_drug, edges_by_source_cache, node_lookup, source_array_cache
        _KG_EXPLAIN_CACHE.pop(cache_key, None)

    edge_table = getattr(kg_graph, "edge_table", pd.DataFrame())
    node_table = getattr(kg_graph, "node_table", pd.DataFrame())
    if edge_table.empty or node_table.empty:
        edges_by_drug: dict[int, pd.DataFrame] = {}
        edges_by_source_cache: dict[int, dict[str, pd.DataFrame]] = {}
        node_lookup: dict[int, dict[str, Any]] = {}
        source_array_cache: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    else:
        edge_frame = edge_table.copy()
        if "edge_id" not in edge_frame.columns:
            edge_frame.insert(0, "edge_id", np.arange(len(edge_frame), dtype=np.int64))
        edge_frame["_drug_id"] = edge_frame["DRUG_ID"].astype(int)
        edge_frame["_source_name"] = edge_frame["source"].astype(str)
        edges_by_drug = {
            int(drug_id): drug_group.drop(columns=["_drug_id"])
            for drug_id, drug_group in edge_frame.groupby("_drug_id", sort=False)
        }
        edges_by_source_cache = {}
        node_lookup = node_table.set_index("node_id").to_dict("index") if "node_id" in node_table.columns else {}
        source_array_cache = {}
    try:
        _KG_EXPLAIN_CACHE[cache_key] = (weakref.ref(kg_graph), edges_by_drug, edges_by_source_cache, node_lookup, source_array_cache)
    except TypeError:
        pass
    return edges_by_drug, edges_by_source_cache, node_lookup, source_array_cache


def build_source_rows(
    batch_frame: pd.DataFrame,
    *,
    source_attention: np.ndarray | None,
    source_names: list[str],
    gate_values: np.ndarray | None,
    total_contribution: np.ndarray | None,
) -> pd.DataFrame:
    if source_attention is None:
        return pd.DataFrame()
    gate_mean = None if gate_values is None else gate_values.mean(axis=1)
    total = None if total_contribution is None else total_contribution.reshape(-1)
    n_samples = len(batch_frame)
    n_sources = min(len(source_names), int(source_attention.shape[1]))
    if n_samples == 0 or n_sources == 0:
        return pd.DataFrame()
    source_attention = np.asarray(source_attention[:, :n_sources])
    data = {
        "prediction_index": np.repeat(batch_frame.index.to_numpy(), n_sources),
        "SANGER_MODEL_ID": np.repeat(batch_frame["SANGER_MODEL_ID"].to_numpy(), n_sources),
        "DRUG_ID": np.repeat(batch_frame["DRUG_ID"].astype(int).to_numpy(), n_sources),
        "source_name": np.tile(np.asarray(source_names[:n_sources], dtype=object), n_samples),
        "source_attention": source_attention.reshape(-1),
        "kg_total_contribution": None if total is None else np.repeat(total, n_sources),
        "kg_gate_mean": None if gate_mean is None else np.repeat(gate_mean, n_sources),
    }
    return pd.DataFrame(data)


def build_edge_rows(
    batch_frame: pd.DataFrame,
    *,
    kg_graph: Any | None,
    source_attention: np.ndarray | None,
    source_names: list[str],
    top_k_edges: int,
    sample_queries: np.ndarray | None = None,
    branch_node_states: np.ndarray | None = None,
) -> pd.DataFrame:
    if kg_graph is None or source_attention is None:
        return pd.DataFrame()
    edges_by_drug, edges_by_source_cache, node_lookup, source_array_cache = _get_kg_explain_cache(kg_graph)
    if not edges_by_drug or not node_lookup:
        return pd.DataFrame()
    out_rows: list[dict[str, Any]] = []
    for local_idx, row in enumerate(batch_frame.itertuples(index=False)):
        drug_id = int(getattr(row, "DRUG_ID"))
        per_drug = edges_by_drug.get(drug_id)
        if per_drug is None or per_drug.empty:
            continue
        edges_by_source = edges_by_source_cache.get(drug_id)
        if edges_by_source is None:
            edges_by_source = {
                source_name: source_group.drop(columns=["_source_name"])
                for source_name, source_group in per_drug.groupby("_source_name", sort=False)
            }
            edges_by_source_cache[drug_id] = edges_by_source
        source_arrays = source_array_cache.get(drug_id)
        if source_arrays is None:
            source_arrays = {}
            for source_name, sub in edges_by_source.items():
                weights = sub["weight"].astype(float).to_numpy(copy=False) if "weight" in sub.columns else np.ones((len(sub),), dtype=np.float32)
                source_arrays[str(source_name)] = {
                    "edge_id": sub["edge_id"].astype(int).to_numpy(copy=False),
                    "edge_type": sub["edge_type"].astype(str).to_numpy(copy=False),
                    "relation": sub["relation"].astype(str).to_numpy(copy=False),
                    "source": sub["source"].astype(str).to_numpy(copy=False),
                    "src": sub["src"].astype(int).to_numpy(copy=False),
                    "dst": sub["dst"].astype(int).to_numpy(copy=False),
                    "weight": weights.astype(np.float32, copy=False),
                    "structural_score": _normalize_weights(weights),
                }
            source_array_cache[drug_id] = source_arrays
        edge_frames: list[pd.DataFrame] = []
        for source_idx, source_name in enumerate(source_names[: source_attention.shape[1]]):
            arrays = source_arrays.get(str(source_name))
            if arrays is None or arrays["edge_id"].size == 0:
                continue
            edge_attention: np.ndarray
            sample_context_score: np.ndarray
            if (
                sample_queries is not None
                and branch_node_states is not None
                and source_idx < branch_node_states.shape[0]
                and local_idx < sample_queries.shape[0]
            ):
                query = np.asarray(sample_queries[local_idx], dtype=np.float32)
                query_norm = float(np.linalg.norm(query))
                if query_norm > 0:
                    query = query / query_norm
                node_bank = np.asarray(branch_node_states[source_idx], dtype=np.float32)
                src_ids = arrays["src"]
                dst_ids = arrays["dst"]
                src_valid = (src_ids >= 0) & (src_ids < node_bank.shape[0])
                dst_valid = (dst_ids >= 0) & (dst_ids < node_bank.shape[0])
                src_score = np.zeros((src_ids.shape[0],), dtype=np.float32)
                dst_score = np.zeros((dst_ids.shape[0],), dtype=np.float32)
                if src_valid.any():
                    src_embed = node_bank[src_ids[src_valid]]
                    src_norm = np.linalg.norm(src_embed, axis=1, keepdims=True).clip(min=1e-6)
                    src_score[src_valid] = np.maximum((src_embed / src_norm) @ query, 0.0)
                if dst_valid.any():
                    dst_embed = node_bank[dst_ids[dst_valid]]
                    dst_norm = np.linalg.norm(dst_embed, axis=1, keepdims=True).clip(min=1e-6)
                    dst_score[dst_valid] = np.maximum((dst_embed / dst_norm) @ query, 0.0)
                context_raw = (0.35 * src_score) + (0.65 * dst_score)
                context_score = _normalize_weights(context_raw)
                combined = (arrays["structural_score"] + 1e-6) * (context_score + 1e-6)
                edge_attention = _normalize_weights(combined) * float(source_attention[local_idx, source_idx])
                sample_context_score = context_raw
            else:
                edge_attention = arrays["structural_score"] * float(source_attention[local_idx, source_idx])
                sample_context_score = np.full(arrays["edge_id"].shape, np.nan, dtype=np.float32)
            source_attention_value = float(source_attention[local_idx, source_idx])
            edge_frames.append(
                pd.DataFrame(
                    {
                        "edge_id": arrays["edge_id"],
                        "source": arrays["source"],
                        "edge_type": arrays["edge_type"],
                        "relation": arrays["relation"],
                        "src": arrays["src"],
                        "dst": arrays["dst"],
                        "weight": arrays["weight"],
                        "source_attention": source_attention_value,
                        "edge_attention": edge_attention,
                        "sample_context_score": sample_context_score,
                    }
                )
            )
        if not edge_frames:
            continue
        scored = pd.concat(edge_frames, ignore_index=True).sort_values("edge_attention", ascending=False).head(int(top_k_edges))
        if scored.empty:
            continue
        src_ids = scored["src"].astype(int).to_numpy(copy=False)
        dst_ids = scored["dst"].astype(int).to_numpy(copy=False)
        src_names = np.asarray([None] * len(src_ids), dtype=object)
        src_types = np.asarray([None] * len(src_ids), dtype=object)
        dst_names = np.asarray([None] * len(dst_ids), dtype=object)
        dst_types = np.asarray([None] * len(dst_ids), dtype=object)
        if node_lookup:
            src_infos = [node_lookup.get(int(node_id)) for node_id in src_ids]
            dst_infos = [node_lookup.get(int(node_id)) for node_id in dst_ids]
            src_names = np.asarray([None if info is None else str(info.get("name", "")) for info in src_infos], dtype=object)
            src_types = np.asarray([None if info is None else str(info.get("node_type", "")) for info in src_infos], dtype=object)
            dst_names = np.asarray([None if info is None else str(info.get("name", "")) for info in dst_infos], dtype=object)
            dst_types = np.asarray([None if info is None else str(info.get("node_type", "")) for info in dst_infos], dtype=object)
        scored_frame = pd.DataFrame(
            {
                "prediction_index": int(batch_frame.index[local_idx]),
                "SANGER_MODEL_ID": getattr(row, "SANGER_MODEL_ID"),
                "DRUG_ID": drug_id,
                "edge_id": scored["edge_id"].astype(int).to_numpy(copy=False),
                "source_name": scored["source"].astype(str).to_numpy(copy=False),
                "edge_type": scored["edge_type"].astype(str).to_numpy(copy=False),
                "relation": scored["relation"].astype(str).to_numpy(copy=False),
                "src_node_id": src_ids,
                "src_node_name": src_names,
                "src_node_type": src_types,
                "dst_node_id": dst_ids,
                "dst_node_name": dst_names,
                "dst_node_type": dst_types,
                "edge_weight": scored["weight"].astype(float).to_numpy(copy=False) if "weight" in scored.columns else np.ones(len(scored), dtype=np.float32),
                "source_attention": scored["source_attention"].astype(float).to_numpy(copy=False),
                "edge_attention": scored["edge_attention"].astype(float).to_numpy(copy=False),
                "sample_context_score": scored["sample_context_score"].to_numpy(copy=False)
                if "sample_context_score" in scored.columns
                else np.full(len(scored), np.nan),
            }
        )
        out_rows.append(scored_frame)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def build_node_rows(edge_rows: pd.DataFrame, *, top_k_nodes: int) -> pd.DataFrame:
    if edge_rows.empty:
        return pd.DataFrame()
    src_frame = edge_rows[
        ["prediction_index", "SANGER_MODEL_ID", "DRUG_ID", "src_node_id", "src_node_name", "src_node_type", "edge_attention"]
    ].rename(columns={"src_node_id": "node_id", "src_node_name": "node_name", "src_node_type": "node_type"})
    dst_frame = edge_rows[
        ["prediction_index", "SANGER_MODEL_ID", "DRUG_ID", "dst_node_id", "dst_node_name", "dst_node_type", "edge_attention"]
    ].rename(columns={"dst_node_id": "node_id", "dst_node_name": "node_name", "dst_node_type": "node_type"})
    node_frame = pd.concat([src_frame, dst_frame], ignore_index=True)
    grouped = (
        node_frame.groupby(["prediction_index", "SANGER_MODEL_ID", "DRUG_ID", "node_id", "node_name", "node_type"], observed=True)["edge_attention"]
        .sum()
        .reset_index()
        .rename(columns={"edge_attention": "node_attention"})
    )
    grouped["node_rank"] = grouped.groupby("prediction_index")["node_attention"].rank(method="first", ascending=False).astype(int)
    return grouped.loc[grouped["node_rank"].le(int(top_k_nodes))].sort_values(["prediction_index", "node_rank"]).reset_index(drop=True)
