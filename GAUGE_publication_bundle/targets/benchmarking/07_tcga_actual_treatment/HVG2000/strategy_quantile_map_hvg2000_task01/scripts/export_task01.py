from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_RESULT_FILES = [
    "predictions.csv",
    "tcga_actual_treatment_scores.csv",
    "tcga_response_auc_episode_scores.csv",
    "tcga_response_auc_validation_metrics.csv",
    "tcga_os_survival_metrics.csv",
    "mapping_audit.csv",
    "tcga_unmapped_actual_drugs.csv",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export task01-style files from a local strategy result directory.")
    parser.add_argument("--task-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    task_dir = args.task_dir.resolve()
    config = yaml.safe_load((task_dir / "config.yaml").read_text())
    root = task_dir.parents[1]
    result_dir = (root / str(config["strategy_result_dir"])).resolve()
    baseline_dir = (root / str(config["baseline_result_dir"])).resolve()
    outputs_dir = task_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_RESULT_FILES if not (result_dir / name).exists()]
    if missing:
        raise SystemExit(f"Strategy result dir missing files: {', '.join(missing)}")

    predictions = pd.read_csv(result_dir / "predictions.csv").rename(columns={"entity_id": "sample_id"})
    predictions.insert(0, "patient_id", predictions["sample_id"].astype(str).str.replace(r"-01[A-Z]$", "", regex=True))
    actual_scores = pd.read_csv(result_dir / "tcga_actual_treatment_scores.csv").rename(columns={"entity_id": "sample_id"})
    actual_scores.insert(0, "patient_id", actual_scores["sample_id"].astype(str).str.replace(r"-01[A-Z]$", "", regex=True))
    response_scores = pd.read_csv(result_dir / "tcga_response_auc_episode_scores.csv")
    mapping_audit = pd.read_csv(result_dir / "tcga_unmapped_actual_drugs.csv")
    response_metrics = pd.read_csv(result_dir / "tcga_response_auc_validation_metrics.csv")
    survival_metrics = pd.read_csv(result_dir / "tcga_os_survival_metrics.csv")
    survival_meta = (
        actual_scores[["patient_id", "sample_id", "event", "time", "age_at_diagnosis", "project_id"]]
        .drop_duplicates(subset=["patient_id", "sample_id"], keep="first")
    )
    predictions = predictions.merge(
        survival_meta,
        on=["patient_id", "sample_id", "project_id"],
        how="left",
        validate="many_to_one",
    )

    patient_key_map = pd.concat(
        [
            predictions[["patient_id", "sample_id", "project_id"]].drop_duplicates(),
            response_scores[["patient_id", "sample_id", "project_id"]].drop_duplicates(),
        ],
        ignore_index=True,
    ).drop_duplicates().sort_values(["project_id", "patient_id", "sample_id"])
    drug_key_map = pd.concat(
        [
            predictions[["DRUG_ID", "DRUG_NAME"]].drop_duplicates(),
            response_scores[["DRUG_ID", "DRUG_NAME"]].drop_duplicates(),
        ],
        ignore_index=True,
    ).drop_duplicates().sort_values(["DRUG_NAME", "DRUG_ID"])
    drug_key_map["drug_key"] = drug_key_map["DRUG_NAME"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")

    predictions.to_csv(outputs_dir / "patient_drug_predictions.csv", index=False)
    actual_scores.to_csv(outputs_dir / "actual_treatment_scores.csv", index=False)
    response_scores.to_csv(outputs_dir / "response_episode_scores.csv", index=False)
    patient_key_map.to_csv(outputs_dir / "patient_key_map.csv", index=False)
    drug_key_map.to_csv(outputs_dir / "drug_key_map.csv", index=False)
    mapping_audit.to_csv(outputs_dir / "mapping_audit.csv", index=False)
    response_metrics.to_csv(outputs_dir / "response_validation_metrics.csv", index=False)
    survival_metrics.to_csv(outputs_dir / "os_survival_metrics.csv", index=False)

    pd.concat(
        [pd.read_csv(baseline_dir / "tcga_response_auc_validation_metrics.csv").assign(source="baseline"), response_metrics.assign(source="strategy")],
        ignore_index=True,
    ).to_csv(outputs_dir / "baseline_vs_strategy_response_validation_metrics.csv", index=False)
    pd.concat(
        [pd.read_csv(baseline_dir / "tcga_os_survival_metrics.csv").assign(source="baseline"), survival_metrics.assign(source="strategy")],
        ignore_index=True,
    ).to_csv(outputs_dir / "baseline_vs_strategy_os_survival_metrics.csv", index=False)

    summary = {
        "result_dir": str(result_dir),
        "strategy_result_dir": str(result_dir),
        "baseline_result_dir": str(baseline_dir),
        "n_prediction_rows": int(len(predictions)),
        "n_actual_treatment_rows": int(len(actual_scores)),
        "n_response_episode_rows": int(len(response_scores)),
        "n_patients": int(patient_key_map["patient_id"].nunique()),
        "n_drugs": int(drug_key_map["DRUG_ID"].nunique()),
    }
    pd.Series(summary).to_json(outputs_dir / "dataset_profile.json", indent=2)
    print(f"[strategy task01 export] wrote outputs to {outputs_dir}")


if __name__ == "__main__":
    main()
