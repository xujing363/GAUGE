"""Local knowledge-graph and explainability tools for the GAUGE Assistant.

These surface capability the app already has on dedicated pages -- the KG
Network Viewer and the KG Explainability page -- but which the chat agent
previously could not reach. They are fully offline (no network): everything
comes out of the bundled ``artifacts.kg_graph`` and the GAUGE model itself.

``build_drug_subgraph`` is the single source of truth for the drug
neighbourhood walk; ``app/pages/11_*_KG_Network_Viewer.py`` imports it so the
chat and the visual page can never drift apart.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from .bundle import ModelBundle
from .predict import DrugNotFoundError, SampleResolutionError, predict_one, resolve_drug


def build_drug_subgraph(bundle: ModelBundle, drug_id: int, hops: int = 1, max_nodes: int = 60) -> nx.Graph:
    """ChEMBL/DRKG/PrimeKG neighbourhood around one drug as a networkx graph.

    Lifted verbatim (logic-wise) from the KG Network Viewer page so the chat
    agent and the visual page share one implementation.
    """
    kg = bundle.artifacts.kg_graph
    coverage = kg.coverage
    cov_row = coverage.loc[coverage["DRUG_ID"] == int(drug_id)]
    G = nx.Graph()
    if cov_row.empty:
        return G
    cov_row = cov_row.iloc[0]
    node_table = kg.node_table.set_index("node_id", drop=False)
    edge_table = kg.edge_table

    seed_nodes: dict[str, int] = {}
    for src in bundle.meta.get("kg_sources", []):
        node_id = cov_row.get(f"{src}_node_id")
        if pd.isna(node_id) or not cov_row.get(f"has_{src}", False):
            continue
        node_id = int(node_id)
        seed_nodes[src] = node_id
        G.add_node(node_id, name="drug", node_type="drug", source=src)

    frontier = set(seed_nodes.values())
    for _ in range(hops):
        if not frontier or G.number_of_nodes() >= max_nodes:
            break
        next_frontier: set[int] = set()
        relevant = edge_table.loc[edge_table["src"].isin(frontier) | edge_table["dst"].isin(frontier)]
        relevant = relevant.head(max_nodes * 4)
        for row in relevant.itertuples(index=False):
            if G.number_of_nodes() >= max_nodes:
                break
            for node_id in (row.src, row.dst):
                if node_id not in G:
                    info = node_table.loc[node_id] if node_id in node_table.index else None
                    name = str(info["name"]) if info is not None else str(node_id)
                    ntype = str(info["node_type"]) if info is not None else "unknown"
                    G.add_node(node_id, name=name, node_type=ntype, source=row.source)
                    next_frontier.add(node_id)
            G.add_edge(row.src, row.dst, relation=str(row.relation), source=row.source)
        frontier = next_frontier
    return G


def kg_neighborhood(bundle: ModelBundle, uploaded: dict, drug: str, hops: int = 1, max_nodes: int = 40) -> dict[str, Any]:
    """Tool: summarise the bundled KG neighbourhood around a drug.

    Returns coverage per knowledge graph plus the most informative edges, so
    the LLM can describe *which mechanisms/targets/diseases* the drug is linked
    to in GAUGE's prior knowledge -- the structure its attention reasons over.
    """
    try:
        resolved = resolve_drug(bundle, drug)
    except DrugNotFoundError as exc:
        return {"error": str(exc)}
    if not resolved.known or resolved.drug_id is None:
        return {"error": f"{drug!r} is not a library drug with knowledge-graph coverage."}

    kg = bundle.artifacts.kg_graph
    cov_row = kg.coverage.loc[kg.coverage["DRUG_ID"] == int(resolved.drug_id)]
    coverage = {}
    if not cov_row.empty:
        r = cov_row.iloc[0]
        for src in bundle.meta.get("kg_sources", []):
            coverage[src] = {
                "covered": bool(r.get(f"has_{src}", False)),
                "graph_degree": int(r.get(f"graph_degree_{src}", 0) or 0),
            }

    G = build_drug_subgraph(bundle, int(resolved.drug_id), hops=hops, max_nodes=max_nodes)
    if G.number_of_nodes() <= 1:
        return {
            "drug": resolved.name,
            "kg_coverage": coverage,
            "edges": [],
            "note": "No edges in the bundled (filtered) graph at this hop depth.",
        }
    edges = [
        {"from": G.nodes[u].get("name"), "to": G.nodes[v].get("name"), "relation": d.get("relation"), "source": d.get("source")}
        for u, v, d in G.edges(data=True)
    ][: max_nodes]
    return {
        "drug": resolved.name,
        "kg_coverage": coverage,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "edges": edges,
        "kg_sources_legend": {
            "ChEMBL": "mechanism of action",
            "DRKG": "gene/protein drug-target edges",
            "PrimeKG": "protein-disease edges",
        },
    }


def explain_prediction(bundle: ModelBundle, uploaded: dict, cell_line: str, drug: str) -> dict[str, Any]:
    """Tool: explain a single GAUGE prediction via KG source-attention + percentile.

    A focused view of ``predict_one`` that foregrounds *why* GAUGE produced the
    score (which knowledge graph drove it, where the AUC sits in the drug's own
    distribution) rather than the raw numbers alone.
    """
    sample: str | dict = uploaded.get(cell_line, cell_line)
    try:
        result = predict_one(bundle, sample, drug)
    except (DrugNotFoundError, SampleResolutionError) as exc:
        return {"error": str(exc)}

    kg_alpha = {k: round(v, 3) for k, v in result.kg_alpha.items()} if result.kg_alpha else None
    dominant = max(kg_alpha, key=kg_alpha.get) if kg_alpha else None
    return {
        "cell_line": cell_line,
        "drug": result.drug.name,
        "relative_sensitive_value": round(result.value_hat, 3),
        "predicted_absolute_auc": round(result.auc_hat, 3),
        "kg_source_attention": kg_alpha,
        "dominant_knowledge_graph": dominant,
        "gate_strength": round(result.gate_strength, 3) if result.gate_strength is not None else None,
        "percentile_note": result.percentile_text,
        "interpretation_hint": (
            "Higher relative_sensitive_value = better predicted response. Lower predicted_absolute_auc = "
            "more sensitive. dominant_knowledge_graph indicates which prior (ChEMBL=mechanism, "
            "DRKG=drug-target, PrimeKG=protein-disease) most shaped this prediction."
        ),
    }


LOCAL_TOOL_IMPLS = {
    "kg_neighborhood": kg_neighborhood,
    "explain_prediction": explain_prediction,
}

LOCAL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "kg_neighborhood",
            "description": (
                "Summarise GAUGE's bundled knowledge-graph neighbourhood around a library drug: which of "
                "the three knowledge graphs cover it, and the linked targets/mechanisms/diseases. Offline; "
                "use to explain what prior knowledge GAUGE has about a drug."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {"type": "string", "description": "Library drug name or ID"},
                    "hops": {"type": "integer", "description": "Neighbourhood hops, 1 or 2 (default 1)"},
                    "max_nodes": {"type": "integer", "description": "Max nodes to expand (default 40)"},
                },
                "required": ["drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_prediction",
            "description": (
                "Explain a single GAUGE prediction: returns the score plus the knowledge-graph "
                "source-attention (which prior drove it) and where the AUC sits in the drug's distribution. "
                "Use when the user asks WHY GAUGE predicted something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_line": {"type": "string", "description": "Known sample ID or an uploaded sample name"},
                    "drug": {"type": "string", "description": "Drug name or SMILES"},
                },
                "required": ["cell_line", "drug"],
            },
        },
    },
]
