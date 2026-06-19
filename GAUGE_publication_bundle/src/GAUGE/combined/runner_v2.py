from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from ..benchmarking import BenchmarkConfig
from ..config import DEFAULT_CACHE_DIR, Paths
from ..utils import ensure_dir, write_json
from .audit_v2 import freeze_recommendations, validate_recommendation_hashes, write_audit_summary, write_leakage_guard_report
from .candidates_v2 import build_contextual_candidate_pairs
from .context_v2 import select_context_from_gdsc_internal, select_reliable_drugs
from .external_eval_v2 import write_external_metrics
from .scorer_v2 import score_candidate_pairs_per_cell


def _score_and_write_recommendations(
    *,
    model: Any,
    prepared: Any,
    result_dir: Path,
    selected_cells: list[str],
    context_label: str,
    candidate_pairs: pd.DataFrame,
    lambda_u: float,
    combination_score_mode: str,
    device: str | None,
) -> pd.DataFrame:
    per_cell, agg = score_candidate_pairs_per_cell(
        model=model,
        prepared=prepared,
        candidate_pairs=candidate_pairs,
        selected_cells=selected_cells,
        context_label=context_label,
        lambda_u=lambda_u,
        combination_score_mode=combination_score_mode,
        device=device,
    )
    per_cell.to_csv(result_dir / "contextual_combination_predictions_per_cell.csv", index=False)
    agg.to_csv(result_dir / "contextual_combination_predictions.csv", index=False)
    return agg


def _write_selected_context(result_dir: Path, *, selected: Any, reliable: pd.DataFrame, thresholds: dict[str, Any]) -> None:
    payload = {
        "selection_mode": "fixed_cancer_type",
        "cancer_type": selected.cancer_type,
        "split": selected.split,
        "thresholds": thresholds,
        "selected_context_cells": selected.selected_cells,
        "selected_context_label": selected.cancer_type,
        "reliable_drugs": reliable.loc[reliable["passed_reliable_rule"], ["DRUG_ID", "DRUG_NAME", "n", "pcc", "spearman"]].to_dict("records"),
        "allowed_inputs_before_recommendation": [
            "GDSC predictions.csv",
            "GDSC gdsc_drug_pcc.csv",
            "GDSC gdsc_cell_pcc.csv",
            "GDSC model_list_20260420.csv",
            "KG target/pathway templates",
            "frozen GAUGE source model/artifacts",
        ],
        "forbidden_inputs_before_recommendation": ["DrugComb", "NCI_ALMANAC"],
        "external_validation_inputs_used_before_recommendation": [],
    }
    write_json(result_dir / "selected_context.json", payload)


def _resolve_kg_cache_metadata(*, prepared: Any, cache_dir: Path) -> dict[str, Any]:
    manifest = getattr(prepared, "manifest", None) or {}
    prepared_artifacts = getattr(prepared, "artifacts", None)
    kg_graph = getattr(prepared_artifacts, "kg_graph", None) if prepared_artifacts is not None else None
    cache_key = str(manifest.get("kg_prior_cache_key", "") or "").strip()
    cached_pickle = Path(cache_dir) / "kg_prior" / cache_key / "kg_artifacts.pkl" if cache_key else None
    cached_pickle_exists = bool(cached_pickle and cached_pickle.exists())
    source = "rebuilt_runtime"
    if kg_graph is not None:
        source = "prepared_artifacts.kg_graph"
    elif cached_pickle_exists:
        source = "local_cached_pickle"
    return {
        "kg_cache_source": source,
        "kg_prior_cache_key": cache_key,
        "kg_cached_pickle": str(cached_pickle) if cached_pickle else "",
        "kg_cached_pickle_exists": cached_pickle_exists,
        "kg_graph_present_in_prepared_artifacts": bool(kg_graph is not None),
        "kg_graph_node_count": int(len(getattr(kg_graph, "node_table", []))) if kg_graph is not None else None,
        "kg_graph_edge_count": int(len(getattr(kg_graph, "edge_table", []))) if kg_graph is not None else None,
    }


def _write_rationales_and_mechanism_summary(
    *,
    result_dir: Path,
    cancer_type: str,
    selected_cells: list[str],
    candidate_pairs: pd.DataFrame,
    scored: pd.DataFrame,
    per_cell: pd.DataFrame,
) -> None:
    if scored.empty or "combo_rank" not in scored.columns:
        (result_dir / "combination_rationales.jsonl").write_text("", encoding="utf-8")
        lines = [
            "# Mechanism Summary",
            "",
            f"- Cancer type: `{cancer_type}`",
            f"- Selected context cells: {len(selected_cells)}",
            "- Scored candidate pairs: 0",
        ]
        (result_dir / "mechanism_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    candidate_lookup = {str(row["unordered_pair_key"]): row for row in candidate_pairs.to_dict("records")}
    per_cell_lookup = {str(key): frame.copy() for key, frame in per_cell.groupby("unordered_pair_key")} if not per_cell.empty else {}
    rationale_path = result_dir / "combination_rationales.jsonl"
    top_scored = scored.sort_values("combo_rank", ascending=True).copy()
    lines: list[str] = [
        "# Mechanism Summary",
        "",
        f"- Cancer type: `{cancer_type}`",
        f"- Selected context cells: {len(selected_cells)}",
        f"- Scored candidate pairs: {len(scored)}",
    ]
    with rationale_path.open("w", encoding="utf-8") as fh:
        for row in top_scored.itertuples(index=False):
            key = str(row.unordered_pair_key)
            meta = candidate_lookup.get(key, {})
            cell_frame = per_cell_lookup.get(key, pd.DataFrame())
            top_cells: list[dict[str, Any]] = []
            if not cell_frame.empty:
                ranked_cells = cell_frame.sort_values("combo_score", ascending=False).head(5)
                for item in ranked_cells.to_dict("records"):
                    top_cells.append(
                        {
                            "SANGER_MODEL_ID": str(item.get("SANGER_MODEL_ID", "")),
                            "combo_score": float(item.get("combo_score", np.nan)),
                            "uplift_A_to_B": float(item.get("uplift_A_to_B", np.nan)),
                            "uplift_B_to_A": float(item.get("uplift_B_to_A", np.nan)),
                            "dominant_probe": str(item.get("dominant_probe", "")),
                        }
                    )
            payload = {
                "unordered_pair_key": key,
                "combo_rank": int(getattr(row, "combo_rank")),
                "cancer_type": cancer_type,
                "drug_A_id": int(getattr(row, "drug_A_id")),
                "drug_A_name": str(getattr(row, "drug_A_name")),
                "drug_B_id": int(getattr(row, "drug_B_id")),
                "drug_B_name": str(getattr(row, "drug_B_name")),
                "context_combo_score_median": float(getattr(row, "context_combo_score_median", np.nan)),
                "median_uplift_A_to_B": float(getattr(row, "median_uplift_A_to_B", np.nan)),
                "median_uplift_B_to_A": float(getattr(row, "median_uplift_B_to_A", np.nan)),
                "dominant_probe": str(getattr(row, "dominant_probe", "")),
                "n_selected_cells": int(getattr(row, "n_selected_cells", 0)),
                "n_cells_positive_uplift": int(getattr(row, "n_cells_positive_uplift", 0)),
                "kg_template_id": str(meta.get("kg_template_id", "")),
                "candidate_strategy": str(meta.get("candidate_strategy", "")),
                "kg_support_source_count": int(meta.get("kg_support_source_count", 0) or 0),
                "kg_support_sources": str(meta.get("kg_support_sources", "")),
                "kg_support_score": float(meta.get("kg_support_score", 0.0) or 0.0),
                "kg_evidence_types": str(meta.get("kg_evidence_types", "")),
                "kg_path_signature": str(meta.get("kg_path_signature", "")),
                "target_A": str(meta.get("target_A", "")),
                "pathway_A": str(meta.get("pathway_A", "")),
                "target_B": str(meta.get("target_B", "")),
                "pathway_B": str(meta.get("pathway_B", "")),
                "top_supporting_cells": top_cells,
                "rationale_text": str(getattr(row, "rationale_text", "")),
            }
            fh.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        if not top_scored.empty:
            top10 = top_scored.head(10)
            lines.extend(
                [
                    "",
                    "## Top 10",
                    *[
                        (
                            f"- rank={int(item.combo_rank)} pair={item.drug_A_name}+{item.drug_B_name} "
                            f"score={float(item.context_combo_score_median):.4f} "
                            f"dominant={item.dominant_probe} positive_cells={int(item.n_cells_positive_uplift)}"
                        )
                        for item in top10.itertuples(index=False)
                    ],
                ]
            )
    (result_dir / "mechanism_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contextual_combined_v2_from_existing_run(
    *,
    source_run_dir: Path,
    result_dir: Path,
    model_list_path: Path,
    paths: Paths | None = None,
    benchmark: BenchmarkConfig | None = None,
    load_model_and_prepared: Callable[[], tuple[Any, Any]] | None = None,
    device: str | None = None,
    cancer_type: str = "Melanoma",
    split: str = "test",
    min_context_cells: int = 5,
    min_context_rows: int = 500,
    min_drug_rows: int = 50,
    min_drug_pcc: float = 0.50,
    min_drug_spearman: float = 0.50,
    drug_rule: str = "pcc_or_spearman",
    aggregate_context_score: str = "median",
    lambda_u: float = 0.1,
    candidate_strategy: str = "contextual_kg_reliable_drugs_v2",
    combination_score_mode: str = "uplift_max",
    template_ids: tuple[str, ...] = (
        "T1_MAPK_vertical",
        "T2_MAPK_PI3K_escape",
        "T3_RTK_MAPK_bypass",
        "T4_MAPK_cell_cycle",
    ),
    run_external: bool = True,
    drugcomb_path: Path | None = None,
    nci_almanac_path: Path | None = None,
    top_k_report: int = 25,
) -> pd.DataFrame:
    source_run_dir = Path(source_run_dir)
    result_dir = ensure_dir(Path(result_dir))
    paths = paths or Paths()
    thresholds = {
        "min_context_cells": int(min_context_cells),
        "min_context_rows": int(min_context_rows),
        "min_drug_rows": int(min_drug_rows),
        "min_drug_pcc": float(min_drug_pcc),
        "min_drug_spearman": float(min_drug_spearman),
        "drug_rule": drug_rule,
        "aggregate_context_score": aggregate_context_score,
        "lambda_u": float(lambda_u),
    }
    selected = select_context_from_gdsc_internal(
        predictions_path=source_run_dir / "predictions.csv",
        drug_pcc_path=source_run_dir / "gdsc_drug_pcc.csv",
        cell_pcc_path=source_run_dir / "gdsc_cell_pcc.csv",
        model_list_path=model_list_path,
        cancer_type=cancer_type,
        split=split,
        min_context_cells=min_context_cells,
        min_context_rows=min_context_rows,
    )
    selected.context_report.to_csv(result_dir / "context_selection_report.csv", index=False)
    reliable = select_reliable_drugs(
        source_run_dir / "gdsc_drug_pcc.csv",
        split=split,
        min_drug_rows=min_drug_rows,
        min_drug_pcc=min_drug_pcc,
        min_drug_spearman=min_drug_spearman,
        drug_rule=drug_rule,
    )
    reliable.to_csv(result_dir / "reliable_drug_evidence.csv", index=False)
    _write_selected_context(result_dir, selected=selected, reliable=reliable, thresholds=thresholds)
    if load_model_and_prepared is None:
        raise ValueError("load_model_and_prepared must be provided by the existing-run script")
    model, prepared = load_model_and_prepared()
    kg_cache_meta = _resolve_kg_cache_metadata(prepared=prepared, cache_dir=DEFAULT_CACHE_DIR)
    candidate_pairs = build_contextual_candidate_pairs(
        drug_table=prepared.artifacts.drug_table,
        reliable_drug_evidence=reliable,
        paths=paths,
        candidate_strategy=candidate_strategy,
        template_ids=template_ids,
        prior_policy=getattr(benchmark, "prior", None) if benchmark is not None else None,
        cache_dir=DEFAULT_CACHE_DIR,
        prepared_artifacts=prepared.artifacts,
        prepared_manifest=getattr(prepared, "manifest", None),
    )
    candidate_pairs.to_csv(result_dir / "contextual_candidate_pairs.csv", index=False)
    scored = _score_and_write_recommendations(
        model=model,
        prepared=prepared,
        result_dir=result_dir,
        selected_cells=selected.selected_cells,
        context_label=selected.cancer_type,
        candidate_pairs=candidate_pairs,
        lambda_u=lambda_u,
        combination_score_mode=combination_score_mode,
        device=device,
    )
    try:
        per_cell = pd.read_csv(result_dir / "contextual_combination_predictions_per_cell.csv")
    except EmptyDataError:
        per_cell = pd.DataFrame()
    _write_rationales_and_mechanism_summary(
        result_dir=result_dir,
        cancer_type=selected.cancer_type,
        selected_cells=selected.selected_cells,
        candidate_pairs=candidate_pairs,
        scored=scored,
        per_cell=per_cell,
    )
    frozen = freeze_recommendations(result_dir)
    external_results: dict[str, Any] = {}
    if run_external:
        external_results = write_external_metrics(result_dir, drugcomb_path=drugcomb_path, nci_almanac_path=nci_almanac_path)
    else:
        pd.DataFrame([{"analysis": "drugcomb", "status": "not_run"}]).to_csv(result_dir / "drugcomb_external_metrics.csv", index=False)
        pd.DataFrame([{"analysis": "nci", "status": "not_run"}]).to_csv(result_dir / "nci_external_metrics.csv", index=False)
        external_results = {"DrugComb": {"status": "not_run"}, "NCI_ALMANAC": {"status": "not_run"}}
    hash_check = validate_recommendation_hashes(result_dir, frozen)
    write_leakage_guard_report(result_dir, hash_check=hash_check, external_results=external_results)
    write_audit_summary(result_dir, top_k=top_k_report)
    write_json(
        result_dir / "combined_manifest.json",
        {
            "workflow": "contextual_combined_v2",
            "source_run_dir": str(source_run_dir),
            "benchmark_id": getattr(benchmark, "benchmark_id", None),
            "selection": json.loads((result_dir / "selected_context.json").read_text(encoding="utf-8")),
            "candidate_strategy": candidate_strategy,
            "combination_score_mode": combination_score_mode,
            "template_ids": list(template_ids),
            "n_candidate_pairs": int(len(candidate_pairs)),
            "n_scored_pairs": int(len(scored)),
            "ranking_field": "context_combo_score_median",
            "kg_cache": kg_cache_meta,
        },
    )
    return scored
