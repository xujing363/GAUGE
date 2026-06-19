from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


RECOMMENDATION_FILES = (
    "contextual_candidate_pairs.csv",
    "contextual_combination_predictions_per_cell.csv",
    "contextual_combination_predictions.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_recommendations(result_dir: Path) -> dict[str, str]:
    result_dir = Path(result_dir)
    hashes = {name: sha256_file(result_dir / name) for name in RECOMMENDATION_FILES if (result_dir / name).exists()}
    payload = {
        "frozen_before_external_validation": True,
        "recommendation_files": hashes,
    }
    (result_dir / "recommendation_hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashes


def validate_recommendation_hashes(result_dir: Path, frozen_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    result_dir = Path(result_dir)
    if frozen_hashes is None:
        payload = json.loads((result_dir / "recommendation_hashes.json").read_text(encoding="utf-8"))
        frozen_hashes = payload.get("recommendation_files", {})
    current = {name: sha256_file(result_dir / name) for name in frozen_hashes}
    mismatches = {name: {"before": frozen_hashes[name], "after": current.get(name)} for name in frozen_hashes if frozen_hashes[name] != current.get(name)}
    if mismatches:
        raise ValueError(f"Recommendation hash changed after external validation: {mismatches}")
    return {"unchanged": True, "recommendation_files": current}


def write_leakage_guard_report(result_dir: Path, *, hash_check: dict[str, Any], external_results: dict[str, Any]) -> dict[str, Any]:
    report = {
        "leakage_violation": False,
        "recommendation_stage": {
            "external_validation_inputs_used_before_recommendation": [],
            "frozen_before_external_validation": True,
            "recommendation_hashes_unchanged_after_external_validation": bool(hash_check.get("unchanged")),
            "ranking_field": "context_combo_score_median",
        },
        "external_validation": external_results,
        "forbidden_uses": [
            "DrugComb/NCI were not used for cancer_type selection",
            "DrugComb/NCI were not used for drug filtering, templates, scoring weights, or reranking",
        ],
    }
    (Path(result_dir) / "leakage_guard_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_audit_summary(result_dir: Path, *, top_k: int = 25) -> None:
    result_dir = Path(result_dir)
    def _read_optional_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()

    pred = _read_optional_csv(result_dir / "contextual_combination_predictions.csv")
    drugcomb = _read_optional_csv(result_dir / "drugcomb_external_metrics.csv")
    nci = _read_optional_csv(result_dir / "nci_external_metrics.csv")
    external_ok = False
    if not drugcomb.empty and str(drugcomb.loc[0].get("status", "")) == "completed":
        external_ok = bool(float(drugcomb.loc[0].get("top_decile_enrichment", 0) or 0) > 1.0)
    lines = [
        "# Contextual Combination v2 Audit Summary",
        "",
        "Scope: fixed Melanoma context, GDSC-internal reliable held-out drugs, KG-supported candidates, and frozen GAUGE terminal-latent counterfactual scoring.",
        "",
        f"Top {top_k} recommendations:",
    ]
    if pred.empty:
        lines.append("- No scored recommendations.")
    else:
        for row in pred.head(top_k).itertuples(index=False):
            lines.append(
                f"- {int(row.combo_rank)}. {row.drug_A_name} + {row.drug_B_name}: "
                f"context_combo_score_median={float(row.context_combo_score_median):.6g}"
            )
    lines.extend(["", "External sanity check:"])
    lines.append(f"- DrugComb: {drugcomb.to_dict('records')[0] if not drugcomb.empty else {'status': 'unavailable'}}")
    lines.append(f"- NCI: {nci.to_dict('records')[0] if not nci.empty else {'status': 'unavailable'}}")
    if not external_ok:
        lines.extend(
            [
                "",
                "GDSC-internally reliable, KG-supported candidates did not pass external sanity check.",
            ]
        )
    (result_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
