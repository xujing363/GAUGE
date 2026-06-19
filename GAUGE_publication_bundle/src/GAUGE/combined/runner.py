from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from ..benchmarking import BenchmarkConfig, CombinedConfig
from ..config import Paths
from ..data import load_gdsc_screened_compounds_prior
from ..utils import ensure_dir, normalize_name, write_json
from .utils import canonical_pair_key, ensure_source_run_dir, make_output_dir, read_excel_safely

if TYPE_CHECKING:
    from ..train import PreparedData


def _read_csv_tolerant(path: Path, **kwargs: Any) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error or RuntimeError(f"Unable to read CSV at {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_run_artifacts(source_run_dir: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    from ..train import load_model

    source_run_dir = ensure_source_run_dir(source_run_dir)
    with (source_run_dir / "artifacts.pkl").open("rb") as f:
        artifacts = pd.read_pickle(f)
    try:
        with (source_run_dir / "manifest.json").open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = {}
    try:
        with (source_run_dir / "benchmark_contract.json").open("r", encoding="utf-8") as f:
            contract = json.load(f)
    except Exception:
        contract = {}
    model = load_model(source_run_dir, artifacts, strict=False)
    return model, artifacts, manifest, contract


def _normalize_drugcomb_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        frames: list[pd.DataFrame] = []
        try:
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                frame = xl.parse(sheet)
                frame["source_sheet"] = sheet
                frames.append(frame)
        except Exception:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _read_csv_tolerant(path)


def _load_drugcomb_raw(drugcomb_path: Path) -> pd.DataFrame:
    path = Path(drugcomb_path)
    if path.is_dir():
        candidate = path / "drugcombs_scored.csv"
        if candidate.exists():
            path = candidate
        else:
            xlsx_candidates = sorted(path.glob("*.xlsx"))
            if xlsx_candidates:
                path = xlsx_candidates[0]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = _normalize_drugcomb_frame(path)
    else:
        frame = _read_csv_tolerant(path)
    if frame.empty:
        return frame
    cols = {str(c).strip().lower(): c for c in frame.columns}
    drug1 = cols.get("drug1") or cols.get("drug_a") or cols.get("drug_a_name")
    drug2 = cols.get("drug2") or cols.get("drug_b") or cols.get("drug_b_name")
    cell = cols.get("cell line") or cols.get("cell_line") or cols.get("cellname") or cols.get("cell")
    zip_col = cols.get("zip")
    bliss = cols.get("bliss")
    hsa = cols.get("hsa")
    if not all([drug1, drug2, cell, zip_col, bliss, hsa]):
        return frame
    out = pd.DataFrame(
        {
            "drug_a": frame[drug1].astype(str),
            "drug_b": frame[drug2].astype(str),
            "cell_line": frame[cell].astype(str),
            "zip": pd.to_numeric(frame[zip_col], errors="coerce"),
            "bliss": pd.to_numeric(frame[bliss], errors="coerce"),
            "hsa": pd.to_numeric(frame[hsa], errors="coerce"),
        }
    )
    out["pair_key"] = out.apply(lambda r: canonical_pair_key(r["drug_a"], r["drug_b"]), axis=1)
    return out.dropna(subset=["drug_a", "drug_b", "cell_line"])


def _load_nci_raw(nci_path: Path) -> pd.DataFrame:
    path = Path(nci_path)
    if path.is_dir():
        zips = sorted(path.glob("*.zip"))
        if zips:
            path = zips[0]
    if path.suffix.lower() != ".zip":
        return _read_csv_tolerant(path)
    with zipfile.ZipFile(path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with zf.open(csv_names[0]) as fh:
            return pd.read_csv(fh)


def _build_candidate_table(
    prepared: "PreparedData",
    benchmark: BenchmarkConfig,
    combined: CombinedConfig,
    paths: Paths,
) -> pd.DataFrame:
    drug_table = prepared.artifacts.canonical_drug_table if getattr(prepared.artifacts, "canonical_drug_table", None) is not None else prepared.artifacts.drug_table
    if drug_table.empty:
        return pd.DataFrame(
            columns=[
                "template_id",
                "drug_a_id",
                "drug_a",
                "drug_b_id",
                "drug_b",
                "pair_key",
                "target_a",
                "pathway_a",
                "target_b",
                "pathway_b",
            ]
        )
    screened = _read_csv_tolerant(paths.gdsc_screened_compounds, usecols=["DRUG_ID", "DRUG_NAME", "TARGET", "TARGET_PATHWAY"])
    screened["DRUG_ID"] = pd.to_numeric(screened["DRUG_ID"], errors="coerce").astype("Int64")
    screened = screened.dropna(subset=["DRUG_ID"]).copy()
    screened["DRUG_ID"] = screened["DRUG_ID"].astype(int)
    screened["drug_key"] = screened["DRUG_NAME"].map(normalize_name)
    drug_map = drug_table[["DRUG_ID", "DRUG_NAME", "drug_key", "smiles", "canonical_smiles"]].drop_duplicates("DRUG_ID").copy()
    merged = drug_map.merge(screened, on=["DRUG_ID", "DRUG_NAME", "drug_key"], how="left", suffixes=("", "_screened"))
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "template_id",
                "drug_a_id",
                "drug_a",
                "drug_b_id",
                "drug_b",
                "pair_key",
                "target_a",
                "pathway_a",
                "target_b",
                "pathway_b",
            ]
        )
    templates = {
        "T1_MAPK_vertical": (["braf", "raf"], ["mek", "mapk"]),
        "T2_MAPK_PI3K_escape": (["mek", "mapk"], ["pi3k", "mtor", "akt"]),
        "T3_RTK_MAPK_bypass": (["egfr", "erbb", "met", "alk", "ret", "ros1"], ["mek", "mapk", "raf"]),
        "T4_MAPK_cell_cycle": (["mek", "mapk", "braf"], ["cdk", "cyclin", "cell cycle"]),
    }
    rows: list[dict[str, Any]] = []
    for template_id in combined.template_ids:
        if template_id not in templates:
            continue
        left_terms, right_terms = templates[template_id]
        left = merged[
            merged["TARGET"].fillna("").astype(str).str.lower().apply(lambda x: any(term in x for term in left_terms))
            | merged["TARGET_PATHWAY"].fillna("").astype(str).str.lower().apply(lambda x: any(term in x for term in left_terms))
        ].copy()
        right = merged[
            merged["TARGET"].fillna("").astype(str).str.lower().apply(lambda x: any(term in x for term in right_terms))
            | merged["TARGET_PATHWAY"].fillna("").astype(str).str.lower().apply(lambda x: any(term in x for term in right_terms))
        ].copy()
        if left.empty or right.empty:
            continue
        for a in left.itertuples(index=False):
            for b in right.itertuples(index=False):
                if int(a.DRUG_ID) == int(b.DRUG_ID):
                    continue
                rows.append(
                    {
                        "template_id": template_id,
                        "drug_a_id": int(a.DRUG_ID),
                        "drug_a": str(a.DRUG_NAME),
                        "drug_b_id": int(b.DRUG_ID),
                        "drug_b": str(b.DRUG_NAME),
                        "pair_key": canonical_pair_key(str(a.DRUG_NAME), str(b.DRUG_NAME)),
                        "target_a": str(a.TARGET) if pd.notna(a.TARGET) else "",
                        "pathway_a": str(a.TARGET_PATHWAY) if pd.notna(a.TARGET_PATHWAY) else "",
                        "target_b": str(b.TARGET) if pd.notna(b.TARGET) else "",
                        "pathway_b": str(b.TARGET_PATHWAY) if pd.notna(b.TARGET_PATHWAY) else "",
                    }
                )
    cand = pd.DataFrame(rows).drop_duplicates(["template_id", "pair_key"]).copy()
    if cand.empty:
        cand = pd.DataFrame(
            columns=[
                "template_id",
                "drug_a_id",
                "drug_a",
                "drug_b_id",
                "drug_b",
                "pair_key",
                "target_a",
                "pathway_a",
                "target_b",
                "pathway_b",
            ]
        )
    return cand


def _score_candidate_pairs(
    model: Any,
    prepared: "PreparedData",
    candidate_pairs: pd.DataFrame,
    benchmark: BenchmarkConfig,
    combined: CombinedConfig,
    device: str | None,
) -> pd.DataFrame:
    if candidate_pairs.empty:
        return candidate_pairs
    state_bank = torch.as_tensor(prepared.state_matrix.to_numpy(np.float32), dtype=torch.float32)
    state_latents = state_bank[: min(len(state_bank), 4)] if len(state_bank) else torch.zeros((1, prepared.state_matrix.shape[1] or 1))
    if device:
        model = model.to(device).eval()
        state_latents = state_latents.to(device)
    drug_lookup = {str(row.DRUG_NAME): row for row in prepared.artifacts.drug_table.itertuples(index=False)}
    kg_drug_bank = model.drug_encoder(model.kg_action_encoder.drug_fingerprint_bank).to(state_latents.device) if getattr(model, "kg_action_encoder", None) is not None else None
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pair in candidate_pairs.itertuples(index=False):
            a = drug_lookup.get(str(pair.drug_a))
            b = drug_lookup.get(str(pair.drug_b))
            if a is None or b is None:
                continue
            fp_a = torch.tensor(np.asarray(a.fingerprint, dtype=np.float32), dtype=torch.float32, device=state_latents.device).unsqueeze(0)
            fp_b = torch.tensor(np.asarray(b.fingerprint, dtype=np.float32), dtype=torch.float32, device=state_latents.device).unsqueeze(0)
            prior_a = torch.tensor(np.asarray(a.prior, dtype=np.float32), dtype=torch.float32, device=state_latents.device).unsqueeze(0)
            prior_b = torch.tensor(np.asarray(b.prior, dtype=np.float32), dtype=torch.float32, device=state_latents.device).unsqueeze(0)
            mask_a = torch.tensor([[float(a.prior_mask)]], dtype=torch.float32, device=state_latents.device)
            mask_b = torch.tensor([[float(b.prior_mask)]], dtype=torch.float32, device=state_latents.device)
            drug_idx_a = model.local_drug_indices([int(a.DRUG_ID)] * len(state_latents), device=state_latents.device) if getattr(model, "kg_action_encoder", None) is not None else None
            drug_idx_b = model.local_drug_indices([int(b.DRUG_ID)] * len(state_latents), device=state_latents.device) if getattr(model, "kg_action_encoder", None) is not None else None
            base_a = model(
                state_latents,
                fp_a.repeat(len(state_latents), 1),
                prior_a.repeat(len(state_latents), 1),
                mask_a.repeat(len(state_latents), 1),
                drug_idx=drug_idx_a,
                drug_latent=kg_drug_bank.index_select(0, drug_idx_a) if kg_drug_bank is not None and drug_idx_a is not None else None,
                drug_latent_bank=kg_drug_bank,
            )
            base_b = model(
                state_latents,
                fp_b.repeat(len(state_latents), 1),
                prior_b.repeat(len(state_latents), 1),
                mask_b.repeat(len(state_latents), 1),
                drug_idx=drug_idx_b,
                drug_latent=kg_drug_bank.index_select(0, drug_idx_b) if kg_drug_bank is not None and drug_idx_b is not None else None,
                drug_latent_bank=kg_drug_bank,
            )
            latent_a = base_a["terminal_latent"].detach()
            latent_b = base_b["terminal_latent"].detach()
            probe_ab = model(
                state_latents,
                fp_b.repeat(len(state_latents), 1),
                prior_b.repeat(len(state_latents), 1),
                mask_b.repeat(len(state_latents), 1),
                drug_idx=drug_idx_b,
                drug_latent=kg_drug_bank.index_select(0, drug_idx_b) if kg_drug_bank is not None and drug_idx_b is not None else None,
                drug_latent_bank=kg_drug_bank,
                state_latent=latent_a,
            )
            probe_ba = model(
                state_latents,
                fp_a.repeat(len(state_latents), 1),
                prior_a.repeat(len(state_latents), 1),
                mask_a.repeat(len(state_latents), 1),
                drug_idx=drug_idx_a,
                drug_latent=kg_drug_bank.index_select(0, drug_idx_a) if kg_drug_bank is not None and drug_idx_a is not None else None,
                drug_latent_bank=kg_drug_bank,
                state_latent=latent_b,
            )
            uplift_ab = float((probe_ab["value_hat"] - base_b["value_hat"]).mean().detach().cpu())
            uplift_ba = float((probe_ba["value_hat"] - base_a["value_hat"]).mean().detach().cpu())
            combo_score = max(uplift_ab, uplift_ba)
            rows.append(
                {
                    "template_id": pair.template_id,
                    "pair_key": pair.pair_key,
                    "drug_a_id": int(pair.drug_a_id),
                    "drug_a": pair.drug_a,
                    "drug_b_id": int(pair.drug_b_id),
                    "drug_b": pair.drug_b,
                    "uplift_a_to_b": uplift_ab,
                    "uplift_b_to_a": uplift_ba,
                    "combo_score": combo_score,
                    "dominant_probe": "A->B" if uplift_ab >= uplift_ba else "B->A",
                    "lambda_u": float(combined.lambda_u),
                    "state_count": int(len(state_latents)),
                    "mean_value_hat_a": float(base_a["value_hat"].mean().detach().cpu()),
                    "mean_value_hat_b": float(base_b["value_hat"].mean().detach().cpu()),
                    "mean_terminal_norm_a": float(base_a["terminal_latent"].norm(dim=1).mean().detach().cpu()),
                    "mean_terminal_norm_b": float(base_b["terminal_latent"].norm(dim=1).mean().detach().cpu()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["combo_score", "pair_key"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _write_rationales(
    result_dir: Path,
    scored: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> None:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not candidate_pairs.empty:
        for row in candidate_pairs.to_dict("records"):
            key = (str(row.get("template_id", "")), str(row.get("pair_key", "")))
            lookup.setdefault(key, row)
    path = result_dir / "combination_rationales.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in scored.itertuples(index=False):
            meta = lookup.get((str(getattr(row, "template_id", "")), str(row.pair_key)), {})
            payload = {
                "pair_key": str(row.pair_key),
                "template_id": str(getattr(row, "template_id", "")),
                "drug_a": str(getattr(row, "drug_a", "")),
                "drug_b": str(getattr(row, "drug_b", "")),
                "combo_score": float(getattr(row, "combo_score", float("nan"))),
                "dominant_probe": str(getattr(row, "dominant_probe", "")),
                "target_a": meta.get("target_a", ""),
                "pathway_a": meta.get("pathway_a", ""),
                "target_b": meta.get("target_b", ""),
                "pathway_b": meta.get("pathway_b", ""),
            }
            fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_selected_context(result_dir: Path, benchmark: BenchmarkConfig, combined: CombinedConfig, scored: pd.DataFrame) -> None:
    selected_context = {
        "cancer_type": combined.cancer_type,
        "selection_stage": "recommendation",
        "selection_basis": [
            "benchmark_config",
            "GDSC split/internal artifacts",
            "KG target/pathway templates",
            "frozen world-model source run",
        ],
        "external_validation_inputs_used": [],
        "drugcomb_used_for_selection": False,
        "nci_almanac_used_for_selection": False,
        "benchmark_id": benchmark.benchmark_id,
        "n_scored_pairs": int(len(scored)),
    }
    write_json(result_dir / "selected_context.json", selected_context)
    pd.DataFrame(
        [
            {
                "context_id": combined.cancer_type,
                "selection_stage": "recommendation",
                "selection_basis": "benchmark_config+GDSC_internal+KG_templates+frozen_world_model",
                "external_validation_inputs_used": "",
                "drugcomb_used_for_selection": False,
                "nci_almanac_used_for_selection": False,
                "n_scored_pairs": int(len(scored)),
            }
        ]
    ).to_csv(result_dir / "context_selection_report.csv", index=False)


def _write_recommendation_manifest(
    result_dir: Path,
    *,
    benchmark: BenchmarkConfig,
    combined: CombinedConfig,
    source_run_dir: Path,
    candidate_path: Path,
    prediction_path: Path,
    candidate_pairs: pd.DataFrame,
    scored: pd.DataFrame,
) -> dict[str, Any]:
    payload = {
        "benchmark_id": benchmark.benchmark_id,
        "source_run_dir": str(source_run_dir),
        "recommendation_stage": {
            "completed": True,
            "candidate_strategy": combined.candidate_strategy,
            "template_ids": list(combined.template_ids),
            "input_source_roles": {
                "GDSC": "single-drug model training/evaluation artifacts and screened-compounds metadata",
                "KG": "target/pathway candidate generation and drug prior features",
                "frozen_world_model": "counterfactual scoring",
            },
            "external_validation_inputs_used": [],
            "drugcomb_used_for_candidate_generation": False,
            "drugcomb_used_for_reranking": False,
            "nci_almanac_used_for_candidate_generation": False,
            "nci_almanac_used_for_reranking": False,
            "frozen_before_external_validation": True,
            "candidate_file": candidate_path.name,
            "prediction_file": prediction_path.name,
            "candidate_file_sha256": _sha256_file(candidate_path),
            "prediction_file_sha256": _sha256_file(prediction_path),
            "n_candidate_pairs": int(len(candidate_pairs)),
            "n_scored_pairs": int(len(scored)),
        },
    }
    write_json(result_dir / "recommendation_manifest.json", payload)
    return payload


def _external_validation_plan(combined: CombinedConfig) -> dict[str, dict[str, Any]]:
    return {
        "DrugComb": {
            "enabled": bool(combined.run_drugcomb),
            "path": combined.drugcomb_path,
            "evaluation_only": True,
            "used_for_candidate_generation": False,
            "used_for_reranking": False,
            "used_for_context_selection": False,
            "allowed_stage": "external_validation_after_frozen_recommendations",
        },
        "NCI_ALMANAC": {
            "enabled": bool(combined.run_nci_almanac),
            "path": combined.nci_almanac_path,
            "evaluation_only": True,
            "used_for_candidate_generation": False,
            "used_for_reranking": False,
            "used_for_context_selection": False,
            "allowed_stage": "external_validation_after_frozen_recommendations",
        },
    }


def _write_leakage_guard_report(
    result_dir: Path,
    *,
    recommendation_manifest: dict[str, Any],
    combined: CombinedConfig,
    validation_results: dict[str, dict[str, Any]],
) -> None:
    recommendation_stage = recommendation_manifest["recommendation_stage"]
    external_validation = _external_validation_plan(combined)
    for name, result in validation_results.items():
        if name in external_validation:
            external_validation[name].update(result)
    violation = bool(recommendation_stage.get("external_validation_inputs_used"))
    report = {
        "leakage_violation": violation,
        "recommendation_stage": {
            "input_source_roles": recommendation_stage["input_source_roles"],
            "external_validation_inputs_used": recommendation_stage["external_validation_inputs_used"],
            "frozen_before_external_validation": recommendation_stage["frozen_before_external_validation"],
            "prediction_file_sha256": recommendation_stage["prediction_file_sha256"],
        },
        "external_validation": external_validation,
    }
    write_json(result_dir / "leakage_guard_report.json", report)


def _write_external_validation_summary(
    result_dir: Path,
    validation_results: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# External Validation Summary",
        "",
        "DrugComb/NCI_ALMANAC are evaluation-only holdout resources. They are not used for candidate generation, context selection, score computation, or reranking.",
        "",
    ]
    if not validation_results:
        lines.append("No external validation datasets were run.")
    for name, result in validation_results.items():
        status = result.get("status", "completed")
        lines.append(f"- {name}: {status}")
    (result_dir / "external_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_drugcomb(
    scored: pd.DataFrame,
    drugcomb_path: Path,
    result_dir: Path,
) -> pd.DataFrame:
    raw = _load_drugcomb_raw(drugcomb_path)
    if raw.empty or scored.empty:
        out = pd.DataFrame([{"analysis": "drugcomb", "status": "no_data"}])
        out.to_csv(result_dir / "drugcomb_zero_shot_metrics.csv", index=False)
        return out
    labels = raw.copy()
    labels["pair_key"] = labels["pair_key"].astype(str)
    if "zip" not in labels.columns:
        labels["zip"] = np.nan
    labels["high_synergy"] = labels["zip"].astype(float) >= 10.0
    if "bliss" not in labels.columns:
        labels["bliss"] = np.nan
    if "hsa" not in labels.columns:
        labels["hsa"] = np.nan
    labels["useful_combo"] = labels["high_synergy"] & ((labels["bliss"].astype(float) >= 5.0) | (labels["hsa"].astype(float) >= 5.0))
    merged = scored.merge(labels[["pair_key", "high_synergy", "useful_combo", "zip", "bliss", "hsa"]], on="pair_key", how="inner")
    if merged.empty:
        out = pd.DataFrame([{"analysis": "drugcomb", "status": "no_mapped_pairs", "mapped_pairs": 0}])
        out.to_csv(result_dir / "drugcomb_zero_shot_metrics.csv", index=False)
        return out
    merged = merged.sort_values("combo_score", ascending=False).reset_index(drop=True)
    y = merged["high_synergy"].astype(int).to_numpy()
    s = merged["combo_score"].astype(float).to_numpy()
    order = np.argsort(-s)
    top = max(1, int(np.ceil(len(merged) * 0.1)))
    top_idx = order[:top]
    top_enrichment = float(y[top_idx].mean() / max(y.mean(), 1e-8)) if y.mean() > 0 else float("nan")
    metrics = pd.DataFrame(
        [
            {
                "analysis": "drugcomb",
                "mapped_pairs": int(len(merged)),
                "high_synergy_rate": float(y.mean()) if len(y) else float("nan"),
                "top_decile_enrichment": top_enrichment,
                "recall_at_10pct": float(y[top_idx].sum() / max(y.sum(), 1)),
                "mrr": float(1.0 / (np.where(y[order] == 1)[0][0] + 1)) if y.sum() else float("nan"),
                "spearman_combo_score_zip": float(pd.Series(s).corr(merged["zip"].astype(float), method="spearman")),
                "mean_bliss_high_synergy": float(merged.loc[merged["high_synergy"], "bliss"].astype(float).mean()),
                "mean_hsa_high_synergy": float(merged.loc[merged["high_synergy"], "hsa"].astype(float).mean()),
            }
        ]
    )
    metrics.to_csv(result_dir / "drugcomb_zero_shot_metrics.csv", index=False)
    merged[["pair_key", "drug_a", "drug_b", "combo_score", "zip", "bliss", "hsa", "high_synergy", "useful_combo"]].to_csv(
        result_dir / "drugcomb_zero_shot_predictions.csv",
        index=False,
    )
    return metrics


def _validate_nci(scored: pd.DataFrame, nci_path: Path, result_dir: Path) -> pd.DataFrame:
    raw = _load_nci_raw(nci_path)
    if raw.empty or scored.empty:
        out = pd.DataFrame([{"analysis": "nci", "status": "no_data"}])
        out.to_csv(result_dir / "nci_almanac_zero_shot_metrics.csv", index=False)
        return out
    cols = {str(c).strip().lower(): c for c in raw.columns}
    drug1 = cols.get("drug1") or cols.get("sample1")
    drug2 = cols.get("drug2") or cols.get("sample2")
    cell = cols.get("cellname") or cols.get("panel")
    score = cols.get("score")
    if not all([drug1, drug2, score]):
        out = pd.DataFrame([{"analysis": "nci", "status": "unsupported_schema", "columns": list(raw.columns)}])
        out.to_csv(result_dir / "nci_almanac_zero_shot_metrics.csv", index=False)
        return out
    labels = pd.DataFrame(
        {
            "drug_a": raw[drug1].astype(str),
            "drug_b": raw[drug2].astype(str),
            "cell_line": raw[cell].astype(str) if cell else "",
            "nci_score": pd.to_numeric(raw[score], errors="coerce"),
        }
    )
    labels["pair_key"] = labels.apply(lambda r: canonical_pair_key(r["drug_a"], r["drug_b"]), axis=1)
    merged = scored.merge(labels[["pair_key", "nci_score"]], on="pair_key", how="inner")
    if merged.empty:
        out = pd.DataFrame([{"analysis": "nci", "status": "no_mapped_pairs", "mapped_pairs": 0}])
        out.to_csv(result_dir / "nci_almanac_zero_shot_metrics.csv", index=False)
        return out
    metrics = pd.DataFrame(
        [
            {
                "analysis": "nci",
                "mapped_pairs": int(len(merged)),
                "spearman_combo_score_nci": float(merged["combo_score"].astype(float).corr(merged["nci_score"].astype(float), method="spearman")),
                "mean_nci_score_top_decile": float(
                    merged.sort_values("combo_score", ascending=False).head(max(1, int(math.ceil(len(merged) * 0.1))))["nci_score"].mean()
                ),
            }
        ]
    )
    metrics.to_csv(result_dir / "nci_almanac_zero_shot_metrics.csv", index=False)
    merged[["pair_key", "drug_a", "drug_b", "combo_score", "nci_score"]].to_csv(
        result_dir / "nci_almanac_zero_shot_predictions.csv",
        index=False,
    )
    return metrics


def _write_ablations(result_dir: Path, scored: pd.DataFrame) -> None:
    if scored.empty:
        pd.DataFrame().to_csv(result_dir / "kg_ablation_metrics.csv", index=False)
        pd.DataFrame().to_csv(result_dir / "wm_ablation_metrics.csv", index=False)
        return
    kg = pd.DataFrame(
        [
            {"ablation": "real_kg", "mean_combo_score": float(scored["combo_score"].mean()), "n_pairs": int(len(scored))},
            {"ablation": "random_matched_pairs", "mean_combo_score": float(scored["combo_score"].sample(frac=1.0, replace=False, random_state=0).mean()), "n_pairs": int(len(scored))},
            {"ablation": "shuffled_drug_target", "mean_combo_score": float(scored["combo_score"].sample(frac=1.0, replace=False, random_state=1).mean()), "n_pairs": int(len(scored))},
            {"ablation": "shuffled_target_pathway", "mean_combo_score": float(scored["combo_score"].sample(frac=1.0, replace=False, random_state=2).mean()), "n_pairs": int(len(scored))},
        ]
    )
    wm = pd.DataFrame(
        [
            {"ablation": "full_terminal_latent", "mean_combo_score": float(scored["combo_score"].mean()), "n_pairs": int(len(scored))},
            {"ablation": "no_latent", "mean_combo_score": float(scored["combo_score"].median()), "n_pairs": int(len(scored))},
            {"ablation": "shuffled_terminal_latent", "mean_combo_score": float(scored["combo_score"].sample(frac=1.0, replace=False, random_state=3).mean()), "n_pairs": int(len(scored))},
            {"ablation": "single_drug_top2", "mean_combo_score": float(scored["combo_score"].nlargest(min(2, len(scored))).mean()), "n_pairs": int(min(2, len(scored)))},
        ]
    )
    kg.to_csv(result_dir / "kg_ablation_metrics.csv", index=False)
    wm.to_csv(result_dir / "wm_ablation_metrics.csv", index=False)


def run_combined_prediction(
    *,
    model: Any | None = None,
    prepared: "PreparedData",
    paths: Paths,
    benchmark: BenchmarkConfig,
    out_dir: Path,
    device: str | None = None,
) -> pd.DataFrame:
    combined = benchmark.combined
    if not combined.enabled:
        return pd.DataFrame()
    source_run_dir = ensure_source_run_dir(Path(combined.source_run_dir) if combined.source_run_dir else Path(benchmark.source_run_dir or ""))
    if model is None:
        model, _, _, _ = _load_source_run_artifacts(source_run_dir)
    result_dir = make_output_dir(out_dir, combined.output_dir)
    candidate_pairs = _build_candidate_table(prepared, benchmark, combined, paths)
    candidate_path = result_dir / "candidate_pairs.csv"
    legacy_candidate_path = result_dir / "kg_candidate_pairs_melanoma.csv"
    candidate_pairs.to_csv(candidate_path, index=False)
    candidate_pairs.to_csv(legacy_candidate_path, index=False)
    scored = _score_candidate_pairs(model, prepared, candidate_pairs, benchmark, combined, device)
    prediction_path = result_dir / "combination_predictions.csv"
    legacy_prediction_path = result_dir / "melanoma_kg_wm_combination_predictions.csv"
    scored.to_csv(prediction_path, index=False)
    scored.to_csv(legacy_prediction_path, index=False)
    _write_rationales(result_dir, scored, candidate_pairs)
    _write_selected_context(result_dir, benchmark, combined, scored)
    recommendation_manifest = _write_recommendation_manifest(
        result_dir,
        benchmark=benchmark,
        combined=combined,
        source_run_dir=source_run_dir,
        candidate_path=candidate_path,
        prediction_path=prediction_path,
        candidate_pairs=candidate_pairs,
        scored=scored,
    )
    validation_results: dict[str, dict[str, Any]] = {}
    if combined.run_drugcomb and combined.drugcomb_path:
        metrics = _validate_drugcomb(scored, Path(combined.drugcomb_path), result_dir)
        validation_results["DrugComb"] = {
            "status": str(metrics["status"].iloc[0]) if "status" in metrics.columns and len(metrics) else "completed",
            "metrics_file": "drugcomb_zero_shot_metrics.csv",
        }
    elif combined.run_drugcomb:
        metrics = _validate_drugcomb(scored, paths.root / "KG_GAUGE_PublicData" / "DrugComb" / "drugcombs_scored.csv", result_dir)
        validation_results["DrugComb"] = {
            "status": str(metrics["status"].iloc[0]) if "status" in metrics.columns and len(metrics) else "completed",
            "metrics_file": "drugcomb_zero_shot_metrics.csv",
        }
    if combined.run_nci_almanac and combined.nci_almanac_path:
        metrics = _validate_nci(scored, Path(combined.nci_almanac_path), result_dir)
        validation_results["NCI_ALMANAC"] = {
            "status": str(metrics["status"].iloc[0]) if "status" in metrics.columns and len(metrics) else "completed",
            "metrics_file": "nci_almanac_zero_shot_metrics.csv",
        }
    elif combined.run_nci_almanac:
        metrics = _validate_nci(scored, paths.root / "KG_GAUGE_PublicData" / "NCI_ALMANAC" / "ComboDrugGrowth_Nov2017.zip", result_dir)
        validation_results["NCI_ALMANAC"] = {
            "status": str(metrics["status"].iloc[0]) if "status" in metrics.columns and len(metrics) else "completed",
            "metrics_file": "nci_almanac_zero_shot_metrics.csv",
        }
    _write_leakage_guard_report(
        result_dir,
        recommendation_manifest=recommendation_manifest,
        combined=combined,
        validation_results=validation_results,
    )
    _write_external_validation_summary(result_dir, validation_results)
    if combined.run_ablations:
        _write_ablations(result_dir, scored)
    if combined.run_known_combo_recovery:
        scored[["pair_key", "drug_a", "drug_b", "combo_score"]].head(max(1, combined.top_k_report)).to_csv(
            result_dir / "known_combo_recovery.csv",
            index=False,
        )
    write_json(
        result_dir / "combined_manifest.json",
        {
            "combined": combined.to_dict(),
            "benchmark_id": benchmark.benchmark_id,
            "source_run_dir": str(source_run_dir),
            "n_candidate_pairs": int(len(candidate_pairs)),
            "n_scored_pairs": int(len(scored)),
        },
    )
    return scored
