from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd


def mine_paths_for_prediction(
    *,
    edge_rows: pd.DataFrame,
    max_hops: int = 4,
    top_k_paths: int = 10,
    allowed_templates: set[tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    if edge_rows.empty:
        return pd.DataFrame()
    results: list[dict[str, Any]] = []
    for pred_idx, group in edge_rows.groupby("prediction_index", observed=True):
        adjacency: dict[int, list[dict[str, Any]]] = defaultdict(list)
        node_meta: dict[int, tuple[str, str]] = {}
        drug_id = int(group["DRUG_ID"].iloc[0])
        drug_candidates = set(group.loc[group["src_node_type"].eq("drug"), "src_node_id"].astype(int).tolist())
        drug_candidates.update(group.loc[group["dst_node_type"].eq("drug"), "dst_node_id"].astype(int).tolist())
        for row in group.itertuples(index=False):
            node_meta[int(row.src_node_id)] = (str(row.src_node_name), str(row.src_node_type))
            node_meta[int(row.dst_node_id)] = (str(row.dst_node_name), str(row.dst_node_type))
            adjacency[int(row.src_node_id)].append(
                {
                    "edge_id": int(row.edge_id),
                    "edge_type": str(row.edge_type),
                    "score": float(row.edge_attention),
                    "dst": int(row.dst_node_id),
                    "dst_name": str(row.dst_node_name),
                    "dst_type": str(row.dst_node_type),
                }
            )
        path_rows: list[dict[str, Any]] = []
        for start_node in sorted(drug_candidates):
            start_name, start_type = node_meta.get(int(start_node), ("", "drug"))
            queue: deque[tuple[int, list[int], list[int], list[str], list[str], list[str], list[float]]] = deque()
            queue.append((start_node, [start_node], [], [], [start_name], [start_type], []))
            while queue:
                node_id, node_path, edge_ids, edge_types, node_names, node_types, scores = queue.popleft()
                if len(edge_ids) >= int(max_hops):
                    continue
                for edge in adjacency.get(node_id, []):
                    next_node = int(edge["dst"])
                    if next_node in node_path:
                        continue
                    next_node_path = node_path + [next_node]
                    next_edge_ids = edge_ids + [int(edge["edge_id"])]
                    next_edge_types = edge_types + [str(edge["edge_type"])]
                    next_node_names = node_names + [str(edge["dst_name"])]
                    next_node_types = node_types + [str(edge["dst_type"])]
                    next_scores = scores + [float(edge["score"])]
                    template = tuple(next_edge_types)
                    if allowed_templates and template not in allowed_templates:
                        queue.append((next_node, next_node_path, next_edge_ids, next_edge_types, next_node_names, next_node_types, next_scores))
                        continue
                    if len(next_edge_ids) >= 2:
                        path_rows.append(
                            {
                                "prediction_index": int(pred_idx),
                                "SANGER_MODEL_ID": str(group["SANGER_MODEL_ID"].iloc[0]),
                                "DRUG_ID": drug_id,
                                "path_node_ids": " > ".join(str(x) for x in next_node_path),
                                "path_node_names": " > ".join(next_node_names),
                                "path_node_types": " > ".join(next_node_types),
                                "path_edge_ids": " > ".join(str(x) for x in next_edge_ids),
                                "path_edge_types": " > ".join(next_edge_types),
                                "path_template_type": " / ".join(next_edge_types),
                                "attention_score": float(np.mean(next_scores)),
                            }
                        )
                    queue.append((next_node, next_node_path, next_edge_ids, next_edge_types, next_node_names, next_node_types, next_scores))
        if not path_rows:
            continue
        frame = pd.DataFrame(path_rows).sort_values("attention_score", ascending=False).head(int(top_k_paths)).reset_index(drop=True)
        frame["path_rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)
        results.append(frame)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)
