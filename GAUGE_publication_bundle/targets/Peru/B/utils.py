"""
Shared utilities for Scenario B: Drug Repurposing Analysis.
"""
from __future__ import annotations

import pickle
import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("KGPUB_PY_ROOT", "/mnt/raid5/xujing/KG"))


def load_experiment(prepared_pkl: Path, result_dir: Path, config_yaml: Path, device: str = "cuda:0"):
    """Load model, prepared data, and config."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from GAUGE.benchmarking import load_benchmark_config
        from GAUGE.train import load_model, load_prepared

    print(f"Loading prepared data from {prepared_pkl} ...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        prepared = load_prepared(prepared_pkl)

    artifacts_path = result_dir / "artifacts.pkl"
    if artifacts_path.exists():
        print(f"Loading result-dir artifacts from {artifacts_path} ...")
        with open(artifacts_path, "rb") as f:
            result_artifacts = pickle.load(f)
        # If canonical_drug_table is None, leave it as-is (falls back to drug_table).
        # If canonical_drug_table is already set AND matches kg_graph.drug_ids, keep it.
        # Only override if None (older artifacts) to avoid drug-index mismatch.
        cdt = getattr(result_artifacts, "canonical_drug_table", None)
        if cdt is None:
            result_artifacts = replace(result_artifacts, canonical_drug_table=result_artifacts.drug_table)
        prepared = replace(prepared, artifacts=result_artifacts)

    print(f"Loading model from {result_dir} ...")
    model = load_model(result_dir, prepared.artifacts)
    model = model.eval().to(device)

    config = load_benchmark_config(config_yaml)
    return model, prepared, config


def run_predictions(model, frame, prepared, config, device, batch_size=8192, kg_mask=None, session=None):
    """Run predictions with optional KG mask."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from GAUGE.train import TrainExecutionContext, predict_frame
        from GAUGE.drug_level import select_best_fusion_weight, with_fused_auc
        from GAUGE.repro import RUNTIME_PROFILE_STABLE

    context = TrainExecutionContext()
    outputs = predict_frame(
        model, frame, prepared,
        batch_size=batch_size, device=device,
        benchmark=config, controls=config.controls,
        context=context, runtime_profile=RUNTIME_PROFILE_STABLE,
        eval_compile=False, kg_mask=kg_mask, session=session,
    )
    pred = outputs.core.copy()

    mode = str(config.model.mode).strip().lower()
    if mode == "drug_residual_world_model":
        selected = select_best_fusion_weight(
            pred, candidates=list(config.evaluation.fusion_weight_candidates),
            split="val", metric=config.evaluation.fusion_selection_metric,
        )
        fusion_weight = float(selected["selected_weight"])
        pred = with_fused_auc(pred, fusion_weight)
        pred["_fusion_weight"] = fusion_weight
    return pred


def build_session(model, prepared, config, device, batch_size=8192):
    """Build reusable prediction session."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from GAUGE.train import TrainExecutionContext, _build_prediction_session
        from GAUGE.repro import RUNTIME_PROFILE_STABLE

    context = TrainExecutionContext()
    session = _build_prediction_session(
        model, prepared, device=device, benchmark=config,
        controls=config.controls, context=context,
        runtime_profile=RUNTIME_PROFILE_STABLE, eval_compile=False, explain_dir=None,
    )
    return session


def load_precomputed(result_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all pre-computed result CSVs without needing the model."""
    rd = Path(result_dir)
    return {
        "predictions":     pd.read_csv(rd / "predictions.csv"),
        "kg_attention":    pd.read_csv(rd / "kg_attention_by_prediction.csv"),
        "kg_coverage":     pd.read_csv(rd / "kg_coverage_by_drug.csv"),
        "chembl_edges":    pd.read_csv(rd / "chembl_moa_edges.csv"),
        "drkg_edges":      pd.read_csv(rd / "drkg_filtered_edges.csv"),
        "primekg_edges":   pd.read_csv(rd / "primekg_filtered_edges.csv"),
        "drug_pcc":        pd.read_csv(rd / "gdsc_drug_pcc.csv"),
    }


def get_drug_kg_edges(drug_id: int, chembl: pd.DataFrame, drkg: pd.DataFrame, primekg: pd.DataFrame):
    """Return forward (non-reverse) edge IDs for a drug across all three KGs."""
    c = chembl[(chembl["DRUG_ID"] == drug_id) & ~chembl["edge_type"].str.startswith("rev_")]
    d = drkg[(drkg["DRUG_ID"] == drug_id) & ~drkg["edge_type"].str.startswith("rev_")]
    p = primekg[(primekg["DRUG_ID"] == drug_id) & ~primekg["edge_type"].str.startswith("rev_")]
    return {
        "ChEMBL":  c["edge_id"].tolist(),
        "DRKG":    d["edge_id"].tolist(),
        "PrimeKG": p["edge_id"].tolist(),
        "all":     c["edge_id"].tolist() + d["edge_id"].tolist() + p["edge_id"].tolist(),
    }


def load_cell_cancer_types(gdsc_model_list: Path) -> pd.DataFrame:
    """Load GDSC cell line metadata with cancer types."""
    meta = pd.read_csv(gdsc_model_list, usecols=["model_id", "tissue", "cancer_type", "cancer_type_detail"])
    meta = meta.rename(columns={"model_id": "SANGER_MODEL_ID"})
    return meta


def annotate_cancer_group(cancer_type: str, cancer_groups: dict) -> str:
    """Map fine-grained cancer_type to broad cancer group."""
    if not isinstance(cancer_type, str):
        return "Other"
    ct = cancer_type.strip()
    for group, types in cancer_groups.items():
        if ct in types or any(t.lower() in ct.lower() for t in types):
            return group
    return "Other"
