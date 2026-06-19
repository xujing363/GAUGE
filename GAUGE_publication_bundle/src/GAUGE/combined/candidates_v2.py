from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import DEFAULT_CACHE_DIR, Paths
from ..kg_prior import KG_SOURCES, build_multikg_graph_artifacts
from ..utils import normalize_name


TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "T1_MAPK_vertical": (["braf", "raf"], ["mek", "mapk", "erk"]),
    "T2_MAPK_PI3K_escape": (["mek", "mapk", "erk"], ["pi3k", "mtor", "akt", "pik3"]),
    "T3_RTK_MAPK_bypass": (["egfr", "erbb", "met", "alk", "ret", "ros1"], ["mek", "mapk", "raf", "erk"]),
    "T4_MAPK_cell_cycle": (["mek", "mapk", "braf", "erk"], ["cdk", "cyclin", "cell cycle", "aurora"]),
    "T5_RTK_DNA_damage": (["egfr", "erbb", "met", "alk", "ret", "ros1", "pdgfr", "vegfr", "kit", "src", "abl"], ["dna replication", "top1", "top2", "antimetabolite"]),
    "T6_RTK_Mitosis": (["egfr", "erbb", "met", "alk", "ret", "ros1", "pdgfr", "vegfr", "kit", "src", "abl"], ["mitosis", "microtubule", "taxane", "aurora"]),
    "T7_Src_Cell_cycle": (["src", "abl", "ephrins"], ["cdk", "cell cycle", "cyclin", "aurora"]),
}


def _contains_any(value: Any, terms: list[str]) -> bool:
    text = str(value or "").lower()
    return any(term in text for term in terms)


def _strategy_alias(value: str | None) -> str:
    text = str(value or "contextual_kg_reliable_drugs_v2").strip().lower()
    aliases = {
        "contextual_kg_reliable_drugs_v2": "template",
        "kg_target_pathway": "template",
        "template": "template",
        "template_kg": "template",
        "contextual_multikg_auto_v1": "auto_multikg",
        "auto_multikg": "auto_multikg",
        "auto_kg": "auto_multikg",
        "automatic_kg": "auto_multikg",
        "contextual_multikg_template_intersection_v1": "auto_multikg_template_intersection",
        "auto_multikg_template_intersection": "auto_multikg_template_intersection",
        "auto_kg_template_intersection": "auto_multikg_template_intersection",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported candidate strategy: {value!r}")
    return aliases[text]


def _split_terms(value: Any) -> set[str]:
    text = str(value or "")
    parts = [
        normalize_name(part)
        for raw in text.replace(";", ",").replace("|", ",").split(",")
        for part in [raw.strip()]
    ]
    return {part for part in parts if part}


def _prepare_reliable_drug_annotations(
    *,
    drug_table: pd.DataFrame,
    reliable_drug_evidence: pd.DataFrame,
    screened_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    screened = pd.read_csv(screened_path, usecols=["DRUG_ID", "DRUG_NAME", "TARGET", "TARGET_PATHWAY"])
    screened["DRUG_ID"] = pd.to_numeric(screened["DRUG_ID"], errors="coerce")
    screened = screened.dropna(subset=["DRUG_ID"]).copy()
    screened["DRUG_ID"] = screened["DRUG_ID"].astype(int)
    screened["drug_key"] = screened["DRUG_NAME"].map(normalize_name)

    drug_map = drug_table.copy()
    drug_map["DRUG_ID"] = pd.to_numeric(drug_map["DRUG_ID"], errors="coerce")
    drug_map = drug_map.dropna(subset=["DRUG_ID"]).copy()
    drug_map["DRUG_ID"] = drug_map["DRUG_ID"].astype(int)
    drug_map["drug_key"] = drug_map["DRUG_NAME"].map(normalize_name)

    reliable = reliable_drug_evidence[reliable_drug_evidence["passed_reliable_rule"]].copy()
    reliable["DRUG_ID"] = pd.to_numeric(reliable["DRUG_ID"], errors="coerce")
    reliable = reliable.dropna(subset=["DRUG_ID"]).copy()
    reliable["DRUG_ID"] = reliable["DRUG_ID"].astype(int)

    merged = drug_map.merge(screened, on=["DRUG_ID", "DRUG_NAME", "drug_key"], how="left")
    merged = merged[merged["DRUG_ID"].isin(set(reliable["DRUG_ID"]))].copy()
    merged["TARGET"] = merged["TARGET"].fillna("")
    merged["TARGET_PATHWAY"] = merged["TARGET_PATHWAY"].fillna("")
    merged["target_terms"] = merged["TARGET"].map(_split_terms)
    merged["pathway_terms"] = merged["TARGET_PATHWAY"].map(_split_terms)
    return merged.reset_index(drop=True), reliable.reset_index(drop=True)


def _build_template_candidate_pairs(
    *,
    merged: pd.DataFrame,
    reliable: pd.DataFrame,
    template_ids: tuple[str, ...],
) -> pd.DataFrame:
    reliable_by_id = reliable.set_index("DRUG_ID", drop=False)
    rows: list[dict[str, Any]] = []
    for template_id in template_ids:
        if template_id not in TEMPLATES:
            continue
        left_terms, right_terms = TEMPLATES[template_id]
        left = merged[
            merged["TARGET"].apply(lambda x: _contains_any(x, left_terms))
            | merged["TARGET_PATHWAY"].apply(lambda x: _contains_any(x, left_terms))
        ]
        right = merged[
            merged["TARGET"].apply(lambda x: _contains_any(x, right_terms))
            | merged["TARGET_PATHWAY"].apply(lambda x: _contains_any(x, right_terms))
        ]
        for a in left.itertuples(index=False):
            for b in right.itertuples(index=False):
                if int(a.DRUG_ID) == int(b.DRUG_ID):
                    continue
                a_ev = reliable_by_id.loc[int(a.DRUG_ID)]
                b_ev = reliable_by_id.loc[int(b.DRUG_ID)]
                first, second = (a, b) if int(a.DRUG_ID) <= int(b.DRUG_ID) else (b, a)
                first_ev = reliable_by_id.loc[int(first.DRUG_ID)]
                second_ev = reliable_by_id.loc[int(second.DRUG_ID)]
                rows.append(
                    {
                        "unordered_pair_key": f"{int(first.DRUG_ID)}||{int(second.DRUG_ID)}",
                        "drug_A_id": int(first.DRUG_ID),
                        "drug_A_name": str(first.DRUG_NAME),
                        "drug_B_id": int(second.DRUG_ID),
                        "drug_B_name": str(second.DRUG_NAME),
                        "kg_template_id": template_id,
                        "drug_A_test_n": int(first_ev["n"]),
                        "drug_A_test_pcc": float(first_ev["pcc"]),
                        "drug_A_test_spearman": float(first_ev["spearman"]),
                        "drug_B_test_n": int(second_ev["n"]),
                        "drug_B_test_pcc": float(second_ev["pcc"]),
                        "drug_B_test_spearman": float(second_ev["spearman"]),
                        "passed_context_filter": bool(a_ev["passed_reliable_rule"] and b_ev["passed_reliable_rule"]),
                        "target_A": str(getattr(first, "TARGET", "") or ""),
                        "pathway_A": str(getattr(first, "TARGET_PATHWAY", "") or ""),
                        "target_B": str(getattr(second, "TARGET", "") or ""),
                        "pathway_B": str(getattr(second, "TARGET_PATHWAY", "") or ""),
                        "candidate_strategy": "template",
                        "kg_support_source_count": 0,
                        "kg_support_sources": "",
                        "kg_support_score": 0.0,
                        "kg_evidence_types": "",
                        "kg_path_signature": "",
                    }
                )
    columns = [
        "unordered_pair_key",
        "drug_A_id",
        "drug_A_name",
        "drug_B_id",
        "drug_B_name",
        "kg_template_id",
        "drug_A_test_n",
        "drug_A_test_pcc",
        "drug_A_test_spearman",
        "drug_B_test_n",
        "drug_B_test_pcc",
        "drug_B_test_spearman",
        "passed_context_filter",
        "target_A",
        "pathway_A",
        "target_B",
        "pathway_B",
        "candidate_strategy",
        "kg_support_source_count",
        "kg_support_sources",
        "kg_support_score",
        "kg_evidence_types",
        "kg_path_signature",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["unordered_pair_key", "kg_template_id"])
        .sort_values(["kg_template_id", "unordered_pair_key"])[columns]
        .reset_index(drop=True)
    )


def _collect_multikg_profiles(meta: pd.DataFrame, artifacts: Any) -> dict[int, dict[str, Any]]:
    node_table = artifacts.node_table.copy()
    node_table = node_table.set_index("node_id", drop=False)
    edge_table = artifacts.edge_table.copy()
    edge_table = edge_table[~edge_table["edge_type"].astype(str).str.startswith("rev_")].copy()
    profiles: dict[int, dict[str, Any]] = {}
    for row in meta.itertuples(index=False):
        drug_id = int(row.DRUG_ID)
        profile = {
            "direct_by_source": {source: set() for source in KG_SOURCES},
            "expanded_by_source": {source: set() for source in KG_SOURCES},
            "target_terms": set(getattr(row, "target_terms", set()) or set()),
            "pathway_terms": set(getattr(row, "pathway_terms", set()) or set()),
            "target_text": str(getattr(row, "TARGET", "") or ""),
            "pathway_text": str(getattr(row, "TARGET_PATHWAY", "") or ""),
            "drug_name": str(row.DRUG_NAME),
        }
        coverage_rows = artifacts.coverage.loc[artifacts.coverage["DRUG_ID"].astype(int).eq(drug_id)]
        if coverage_rows.empty:
            profiles[drug_id] = profile
            continue
        coverage_row = coverage_rows.iloc[0]
        for source in KG_SOURCES:
            node_col = f"{source}_node_id"
            if node_col not in coverage_row.index:
                continue
            direct_edges = edge_table.loc[
                edge_table["source"].eq(source) & edge_table["src"].eq(int(coverage_row[node_col]))
            ].copy()
            for edge in direct_edges.itertuples(index=False):
                if int(edge.dst) not in node_table.index:
                    continue
                node = node_table.loc[int(edge.dst)]
                feature = (str(node["node_type"]), str(node["normalized_name"]), str(node["name"]))
                profile["direct_by_source"][source].add(feature)
            all_edges = edge_table.loc[
                edge_table["source"].eq(source) & edge_table["DRUG_ID"].astype(int).eq(drug_id)
            ].copy()
            node_ids = set(all_edges["src"].astype(int).tolist()) | set(all_edges["dst"].astype(int).tolist())
            for node_id in node_ids:
                if node_id not in node_table.index:
                    continue
                node = node_table.loc[int(node_id)]
                if str(node["node_type"]) == "drug":
                    continue
                feature = (str(node["node_type"]), str(node["normalized_name"]), str(node["name"]))
                profile["expanded_by_source"][source].add(feature)
        profiles[drug_id] = profile
    return profiles


def _shared_node_summary(
    left: set[tuple[str, str, str]],
    right: set[tuple[str, str, str]],
    *,
    max_items: int = 3,
) -> tuple[int, list[str]]:
    shared = sorted(left & right, key=lambda item: (item[0], item[1], item[2]))
    preview = [f"{node_type}:{name}" for node_type, _, name in shared[:max_items]]
    return len(shared), preview


def _build_multikg_auto_candidate_pairs(
    *,
    merged: pd.DataFrame,
    reliable: pd.DataFrame,
    paths: Paths,
    prior_policy: Any | None,
    cache_dir: Path,
    max_auto_pairs: int | None,
    prepared_artifacts: Any | None,
    prepared_manifest: dict[str, Any] | None,
) -> pd.DataFrame:
    reliable_by_id = reliable.set_index("DRUG_ID", drop=False)
    if merged.empty:
        return pd.DataFrame()
    kg_input = merged.drop_duplicates("DRUG_ID").copy()
    # Automatic candidate generation for held-out validation must stay on local KG assets only.
    # Disable PubChem CID synonym expansion here to avoid opportunistic external alias fetches.
    if "pubchem_cid" in kg_input.columns:
        kg_input["pubchem_cid"] = pd.NA
    if "pubchem_title" in kg_input.columns:
        kg_input["pubchem_title"] = ""
    artifacts = None
    if prepared_artifacts is not None and getattr(prepared_artifacts, "kg_graph", None) is not None:
        artifacts = prepared_artifacts.kg_graph
    if artifacts is None and prepared_manifest:
        cache_key = str(prepared_manifest.get("kg_prior_cache_key", "") or "").strip()
        if cache_key:
            cached_pickle = Path(cache_dir) / "kg_prior" / cache_key / "kg_artifacts.pkl"
            if cached_pickle.exists():
                try:
                    artifacts = pd.read_pickle(cached_pickle)
                except Exception:
                    artifacts = None
    if artifacts is None:
        artifacts = build_multikg_graph_artifacts(
            paths,
            kg_input,
            cache_dir,
            use_cache=True,
            rebuild_cache=False,
            prior_policy=prior_policy,
        )
    profiles = _collect_multikg_profiles(merged, artifacts)
    max_pairs = (
        int(max_auto_pairs)
        if max_auto_pairs is not None
        else min(3000, max(500, int(len(merged.drop_duplicates("DRUG_ID")) * 20)))
    )
    rows: list[dict[str, Any]] = []
    for left_row, right_row in combinations(merged.drop_duplicates("DRUG_ID").itertuples(index=False), 2):
        left_id = int(left_row.DRUG_ID)
        right_id = int(right_row.DRUG_ID)
        left_profile = profiles.get(left_id, {})
        right_profile = profiles.get(right_id, {})
        source_hits: list[str] = []
        evidence_types: list[str] = []
        path_fragments: list[str] = []
        total_direct = 0
        total_expanded = 0
        for source in KG_SOURCES:
            direct_count, direct_preview = _shared_node_summary(
                left_profile.get("direct_by_source", {}).get(source, set()),
                right_profile.get("direct_by_source", {}).get(source, set()),
            )
            expanded_count, expanded_preview = _shared_node_summary(
                left_profile.get("expanded_by_source", {}).get(source, set()),
                right_profile.get("expanded_by_source", {}).get(source, set()),
            )
            if direct_count or expanded_count:
                source_hits.append(source)
            if direct_count:
                evidence_types.append(f"{source}:shared_direct")
                total_direct += direct_count
            if expanded_count:
                evidence_types.append(f"{source}:shared_multihop")
                total_expanded += expanded_count
            preview = direct_preview or expanded_preview
            if preview:
                path_fragments.append(f"{source}[{'; '.join(preview)}]")
        target_overlap = left_profile.get("target_terms", set()) & right_profile.get("target_terms", set())
        pathway_overlap = left_profile.get("pathway_terms", set()) & right_profile.get("pathway_terms", set())
        target_disjoint = bool(left_profile.get("target_terms") and right_profile.get("target_terms") and not target_overlap)
        if pathway_overlap:
            evidence_types.append("screened_pathway_overlap")
            path_fragments.append("GDSC_pathway[" + "; ".join(sorted(pathway_overlap)[:3]) + "]")
        if not source_hits:
            continue
        support_score = (
            4.0 * len(set(source_hits))
            + 3.0 * min(total_direct, 5)
            + 2.0 * min(total_expanded, 10)
            + 1.0 * min(len(pathway_overlap), 3)
            + (1.0 if target_disjoint else 0.0)
        )
        if len(set(source_hits)) < 2 and (total_direct + total_expanded) < 3:
            continue
        left_ev = reliable_by_id.loc[left_id]
        right_ev = reliable_by_id.loc[right_id]
        rows.append(
            {
                "unordered_pair_key": f"{left_id}||{right_id}",
                "drug_A_id": left_id,
                "drug_A_name": str(left_row.DRUG_NAME),
                "drug_B_id": right_id,
                "drug_B_name": str(right_row.DRUG_NAME),
                "kg_template_id": "AUTO_multikg_shared_neighborhood",
                "drug_A_test_n": int(left_ev["n"]),
                "drug_A_test_pcc": float(left_ev["pcc"]),
                "drug_A_test_spearman": float(left_ev["spearman"]),
                "drug_B_test_n": int(right_ev["n"]),
                "drug_B_test_pcc": float(right_ev["pcc"]),
                "drug_B_test_spearman": float(right_ev["spearman"]),
                "passed_context_filter": bool(left_ev["passed_reliable_rule"] and right_ev["passed_reliable_rule"]),
                "target_A": left_profile.get("target_text", ""),
                "pathway_A": left_profile.get("pathway_text", ""),
                "target_B": right_profile.get("target_text", ""),
                "pathway_B": right_profile.get("pathway_text", ""),
                "candidate_strategy": "auto_multikg",
                "kg_support_source_count": int(len(set(source_hits))),
                "kg_support_sources": "|".join(sorted(set(source_hits))),
                "kg_support_score": float(support_score),
                "kg_evidence_types": "|".join(evidence_types),
                "kg_path_signature": " || ".join(path_fragments[:6]),
            }
        )
    columns = [
        "unordered_pair_key",
        "drug_A_id",
        "drug_A_name",
        "drug_B_id",
        "drug_B_name",
        "kg_template_id",
        "drug_A_test_n",
        "drug_A_test_pcc",
        "drug_A_test_spearman",
        "drug_B_test_n",
        "drug_B_test_pcc",
        "drug_B_test_spearman",
        "passed_context_filter",
        "target_A",
        "pathway_A",
        "target_B",
        "pathway_B",
        "candidate_strategy",
        "kg_support_source_count",
        "kg_support_sources",
        "kg_support_score",
        "kg_evidence_types",
        "kg_path_signature",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["kg_support_source_count", "kg_support_score", "unordered_pair_key"],
            ascending=[False, False, True],
        )
        .head(max_pairs)[columns]
        .reset_index(drop=True)
    )


def _build_auto_multikg_template_intersection_pairs(
    *,
    merged: pd.DataFrame,
    reliable: pd.DataFrame,
    paths: Paths,
    template_ids: tuple[str, ...],
    prior_policy: Any | None,
    cache_dir: Path,
    max_auto_pairs: int | None,
    prepared_artifacts: Any | None,
    prepared_manifest: dict[str, Any] | None,
) -> pd.DataFrame:
    template_pairs = _build_template_candidate_pairs(
        merged=merged,
        reliable=reliable,
        template_ids=template_ids,
    )
    auto_pairs = _build_multikg_auto_candidate_pairs(
        merged=merged,
        reliable=reliable,
        paths=paths,
        prior_policy=prior_policy,
        cache_dir=cache_dir,
        max_auto_pairs=max_auto_pairs,
        prepared_artifacts=prepared_artifacts,
        prepared_manifest=prepared_manifest,
    )
    if template_pairs.empty or auto_pairs.empty:
        return pd.DataFrame(columns=auto_pairs.columns if not auto_pairs.empty else template_pairs.columns)

    template_meta = (
        template_pairs.groupby("unordered_pair_key", as_index=False)
        .agg(
            kg_template_id=("kg_template_id", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
            template_target_A=("target_A", "first"),
            template_pathway_A=("pathway_A", "first"),
            template_target_B=("target_B", "first"),
            template_pathway_B=("pathway_B", "first"),
        )
    )
    intersected = auto_pairs.merge(template_meta, on="unordered_pair_key", how="inner")
    if intersected.empty:
        return pd.DataFrame(columns=auto_pairs.columns)

    intersected["kg_template_id"] = intersected["kg_template_id_y"].where(
        intersected["kg_template_id_y"].astype(str).str.len() > 0,
        intersected["kg_template_id_x"],
    )
    intersected["target_A"] = intersected["template_target_A"].where(
        intersected["template_target_A"].astype(str).str.len() > 0,
        intersected["target_A"],
    )
    intersected["pathway_A"] = intersected["template_pathway_A"].where(
        intersected["template_pathway_A"].astype(str).str.len() > 0,
        intersected["pathway_A"],
    )
    intersected["target_B"] = intersected["template_target_B"].where(
        intersected["template_target_B"].astype(str).str.len() > 0,
        intersected["target_B"],
    )
    intersected["pathway_B"] = intersected["template_pathway_B"].where(
        intersected["template_pathway_B"].astype(str).str.len() > 0,
        intersected["pathway_B"],
    )
    intersected["candidate_strategy"] = "auto_multikg_template_intersection"
    intersected["kg_evidence_types"] = intersected["kg_evidence_types"].fillna("").map(
        lambda text: text if "template_intersection" in str(text) else (f"{text}|template_intersection".strip("|"))
    )

    columns = [
        "unordered_pair_key",
        "drug_A_id",
        "drug_A_name",
        "drug_B_id",
        "drug_B_name",
        "kg_template_id",
        "drug_A_test_n",
        "drug_A_test_pcc",
        "drug_A_test_spearman",
        "drug_B_test_n",
        "drug_B_test_pcc",
        "drug_B_test_spearman",
        "passed_context_filter",
        "target_A",
        "pathway_A",
        "target_B",
        "pathway_B",
        "candidate_strategy",
        "kg_support_source_count",
        "kg_support_sources",
        "kg_support_score",
        "kg_evidence_types",
        "kg_path_signature",
    ]
    return (
        intersected.sort_values(
            ["kg_support_source_count", "kg_support_score", "unordered_pair_key"],
            ascending=[False, False, True],
        )[columns]
        .reset_index(drop=True)
    )


def build_contextual_candidate_pairs(
    *,
    drug_table: pd.DataFrame,
    reliable_drug_evidence: pd.DataFrame,
    paths: Paths | None = None,
    screened_compounds_path: Path | None = None,
    template_ids: tuple[str, ...] = tuple(TEMPLATES),
    candidate_strategy: str = "contextual_kg_reliable_drugs_v2",
    prior_policy: Any | None = None,
    cache_dir: Path | None = None,
    max_auto_pairs: int | None = None,
    prepared_artifacts: Any | None = None,
    prepared_manifest: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if drug_table.empty:
        return pd.DataFrame()
    paths = paths or Paths()
    screened_path = screened_compounds_path or paths.gdsc_screened_compounds
    merged, reliable = _prepare_reliable_drug_annotations(
        drug_table=drug_table,
        reliable_drug_evidence=reliable_drug_evidence,
        screened_path=screened_path,
    )
    strategy = _strategy_alias(candidate_strategy)
    if strategy == "template":
        return _build_template_candidate_pairs(
            merged=merged,
            reliable=reliable,
            template_ids=template_ids,
        )
    if strategy == "auto_multikg_template_intersection":
        return _build_auto_multikg_template_intersection_pairs(
            merged=merged,
            reliable=reliable,
            paths=paths,
            template_ids=template_ids,
            prior_policy=prior_policy,
            cache_dir=cache_dir or DEFAULT_CACHE_DIR,
            max_auto_pairs=max_auto_pairs,
            prepared_artifacts=prepared_artifacts,
            prepared_manifest=prepared_manifest,
        )
    return _build_multikg_auto_candidate_pairs(
        merged=merged,
        reliable=reliable,
        paths=paths,
        prior_policy=prior_policy,
        cache_dir=cache_dir or DEFAULT_CACHE_DIR,
        max_auto_pairs=max_auto_pairs,
        prepared_artifacts=prepared_artifacts,
        prepared_manifest=prepared_manifest,
    )
