from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .cache import cache_key
from .config import CHEMBL_RELEASE_DEFAULT, DATASET_DEFAULT, GDSC_SOURCE_MODE_DEFAULT, Paths, gdsc_fitted_paths, normalize_chembl_release, normalize_dataset_name, normalize_gdsc_source_mode
from .kg_prior import KG_SOURCE_EDGE_TYPES, KG_SOURCE_NODE_TYPES, _validate_policy_values
from .utils import ensure_dir, normalize_name, write_json


CTRP_BENCHMARK_NAME_BY_SPLIT = {
    "random_cell": "CTRP Random Cell Split",
    "drug": "CTRP Drug Split",
    "scaffold": "CTRP Scaffold Split",
    "target_family": "CTRP Target Family Split",
    "double_disjoint": "CTRP Double Disjoint Split",
    "cross_dataset": "CTRP Cross Dataset",
}

CTRP_BENCHMARK_SUFFIX_BY_SPLIT = {
    "random_cell": "random_cell_split",
    "drug": "drug_split",
    "scaffold": "scaffold_split",
    "target_family": "target_family_split",
    "double_disjoint": "double_disjoint_split",
    "cross_dataset": "cross_dataset",
}


TARGET_FAMILY_ORDER = [
    "EGFR/ERBB",
    "MAPK/RAF/MEK",
    "PI3K/AKT/mTOR",
    "CDK/cell-cycle",
    "PARP/DNA-repair",
    "HDAC/epigenetic",
    "proteasome",
    "tubulin",
    "topoisomerase",
    "antimetabolite",
    "platinum/DNA-crosslinking",
    "other",
]

CELL_TRAIN_STATISTICS_FEATURE_COLUMNS = (
    "cell_auc_train_mean",
    "cell_auc_train_median",
    "cell_centered_sensitivity_train",
)

TARGET_FAMILY_KEYWORDS = {
    "EGFR/ERBB": ["egfr", "erbb", "her2", "her 2"],
    "MAPK/RAF/MEK": ["mapk", "raf", "mek", "erk", "braf"],
    "PI3K/AKT/mTOR": ["pi3k", "akt", "mtor", "pik3"],
    "CDK/cell-cycle": ["cdk", "cyclin", "cell cycle", "aurora"],
    "PARP/DNA-repair": ["parp", "dna repair", "brca", "atm", "atr"],
    "HDAC/epigenetic": ["hdac", "epigen", "dnmt", "bromodomain", "brd"],
    "proteasome": ["proteasome", "psmb"],
    "tubulin": ["tubulin", "microtubule"],
    "topoisomerase": ["topoisomerase", "top1", "top2"],
    "antimetabolite": ["antimetabolite", "folate", "thymidylate", "pyrimidine", "purine"],
    "platinum/DNA-crosslinking": ["platinum", "crosslink", "cisplatin", "carboplatin", "oxaliplatin"],
}


@dataclass(frozen=True)
class AblationControls:
    use_state: bool = True
    use_drug: bool = True
    use_prior: bool = True
    prior_mode: str = "learned"
    use_terminal: bool = True
    policy_mode: str = "model"
    seed: int = 7

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_state": self.use_state,
            "use_drug": self.use_drug,
            "use_prior": self.use_prior,
            "prior_mode": self.prior_mode,
            "use_terminal": self.use_terminal,
            "policy_mode": self.policy_mode,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ModelConfig:
    mode: str = "regression"
    target_field: str = "AUC"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_field": self.target_field,
        }


@dataclass(frozen=True)
class WorldModelConfig:
    relative_value_enabled: bool = True
    relative_value_train_field: str = "relative_value_train"
    relative_value_eval_field: str = "relative_value_eval"
    relative_value_direction: str = "auc_lower_is_better"
    terminal_consequence_enabled: bool = True
    planner_enabled: bool = True
    planner_top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_value": {
                "enabled": self.relative_value_enabled,
                "train_field": self.relative_value_train_field,
                "eval_field": self.relative_value_eval_field,
                "direction": self.relative_value_direction,
            },
            "terminal_consequence": {
                "enabled": self.terminal_consequence_enabled,
            },
            "planner": {
                "enabled": self.planner_enabled,
                "top_k": self.planner_top_k,
            },
        }


@dataclass(frozen=True)
class TrainingObjectiveConfig:
    loss_raw_weight: float = 1.0
    loss_value_weight: float = 1.0
    loss_drug_centered_weight: float = 0.0
    loss_cell_residual_weight: float = 0.0
    loss_within_drug_rank_weight: float = 0.05
    loss_policy_advantage_weight: float = 0.05
    loss_same_cell_cross_drug_rank_weight: float = 0.0
    loss_terminal_drug_specificity_weight: float = 0.0
    loss_graph_consistency_weight: float = 0.02
    validation_objective: str = "val_within_drug_pcc"
    rank_margin: float = 0.05
    advantage_margin: float = 0.05
    terminal_drug_specificity_margin: float = 0.05
    min_relative_value_gap: float = 0.15
    min_auc_gap: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_raw_weight": self.loss_raw_weight,
            "loss_value_weight": self.loss_value_weight,
            "loss_drug_centered_weight": self.loss_drug_centered_weight,
            "loss_cell_residual_weight": self.loss_cell_residual_weight,
            "loss_within_drug_rank_weight": self.loss_within_drug_rank_weight,
            "loss_policy_advantage_weight": self.loss_policy_advantage_weight,
            "loss_same_cell_cross_drug_rank_weight": self.loss_same_cell_cross_drug_rank_weight,
            "loss_terminal_drug_specificity_weight": self.loss_terminal_drug_specificity_weight,
            "loss_graph_consistency_weight": self.loss_graph_consistency_weight,
            "validation_objective": self.validation_objective,
            "rank_margin": self.rank_margin,
            "advantage_margin": self.advantage_margin,
            "terminal_drug_specificity_margin": self.terminal_drug_specificity_margin,
            "min_relative_value_gap": self.min_relative_value_gap,
            "min_auc_gap": self.min_auc_gap,
        }


@dataclass(frozen=True)
class BatchSamplerConfig:
    sampler: str = "random_pair"
    n_cell_lines: int = 16
    n_drugs: int = 8
    rank_batch_fraction: float = 0.25
    min_cell_drugs_per_batch: int = 2
    min_drug_cells_per_batch: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler": self.sampler,
            "n_cell_lines": self.n_cell_lines,
            "n_drugs": self.n_drugs,
            "rank_batch_fraction": self.rank_batch_fraction,
            "min_cell_drugs_per_batch": self.min_cell_drugs_per_batch,
            "min_drug_cells_per_batch": self.min_drug_cells_per_batch,
        }


@dataclass(frozen=True)
class KGPriorSourceConfig:
    enabled: bool = True
    weight: float = 1.0
    node_types: tuple[str, ...] = ()
    edge_types: tuple[str, ...] = ()
    release: str = CHEMBL_RELEASE_DEFAULT
    sqlite_tar: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "weight": float(self.weight),
            "node_types": list(self.node_types),
            "edge_types": list(self.edge_types),
            "release": str(self.release),
            "sqlite_tar": self.sqlite_tar,
        }


@dataclass(frozen=True)
class KGPriorConfig:
    chembl: KGPriorSourceConfig = field(
        default_factory=lambda: KGPriorSourceConfig(
            node_types=KG_SOURCE_NODE_TYPES["chembl"],
            edge_types=KG_SOURCE_EDGE_TYPES["chembl"],
        )
    )
    drkg: KGPriorSourceConfig = field(
        default_factory=lambda: KGPriorSourceConfig(
            node_types=KG_SOURCE_NODE_TYPES["drkg"],
            edge_types=KG_SOURCE_EDGE_TYPES["drkg"],
        )
    )
    primekg: KGPriorSourceConfig = field(
        default_factory=lambda: KGPriorSourceConfig(
            node_types=KG_SOURCE_NODE_TYPES["primekg"],
            edge_types=KG_SOURCE_EDGE_TYPES["primekg"],
        )
    )
    include_side_effects: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "chembl": self.chembl.to_dict(),
            "drkg": self.drkg.to_dict(),
            "primekg": self.primekg.to_dict(),
            "include_side_effects": bool(self.include_side_effects),
        }


@dataclass(frozen=True)
class StaticPriorConfig:
    enabled: bool = True
    sources: tuple[str, ...] = ("primekg", "gdsc_screened_compounds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class EvaluationConfig:
    planner_metric_field: str = "relative_value_eval"
    fusion_selection_metric: str = "within_drug_pcc_mean"
    fusion_weight_candidates: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    target_within_drug_pcc: float | None = None
    sample_guardrail_delta: float | None = None
    strict_test_rows: int | None = None
    strict_test_drugs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_metric_field": self.planner_metric_field,
            "fusion_selection_metric": self.fusion_selection_metric,
            "fusion_weight_candidates": list(self.fusion_weight_candidates),
            "target_within_drug_pcc": self.target_within_drug_pcc,
            "sample_guardrail_delta": self.sample_guardrail_delta,
            "strict_test_rows": self.strict_test_rows,
            "strict_test_drugs": self.strict_test_drugs,
        }


@dataclass(frozen=True)
class ExplainabilityConfig:
    enabled: bool = False
    level: str = "source"
    export_top_edges: bool = False
    export_top_nodes: bool = False
    export_top_paths: bool = False
    run_ablation: bool = False
    top_k_edges: int = 20
    top_k_nodes: int = 20
    top_k_paths: int = 10
    path_max_hops: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "level": self.level,
            "export_top_edges": self.export_top_edges,
            "export_top_nodes": self.export_top_nodes,
            "export_top_paths": self.export_top_paths,
            "run_ablation": self.run_ablation,
            "top_k_edges": self.top_k_edges,
            "top_k_nodes": self.top_k_nodes,
            "top_k_paths": self.top_k_paths,
            "path_max_hops": self.path_max_hops,
        }


@dataclass(frozen=True)
class PerturbationConfig:
    enabled: bool = False
    stub_mode: bool = False
    output_dir: str = "perturbation"
    action_top_k: int = 20
    action_levels: tuple[str, ...] = ("edge", "node")
    node_top_k: int = 10
    allowed_node_types: tuple[str, ...] = ("gene", "protein", "pathway", "biological_process", "molecular_function")
    max_edges_per_action: int = 100
    calibration_method: str = "drug_specific_percentile"
    resistant_percentile: float = 0.80
    partial_resistant_percentile: float = 0.60
    sensitive_percentile: float = 0.25
    min_calibration_rows: int = 10
    random_controls: int = 100
    random_match_by_source: bool = True
    random_match_by_node_type: bool = True
    random_match_by_degree_bin: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stub_mode": self.stub_mode,
            "output_dir": self.output_dir,
            "action_top_k": self.action_top_k,
            "action_levels": list(self.action_levels),
            "node_top_k": self.node_top_k,
            "allowed_node_types": list(self.allowed_node_types),
            "max_edges_per_action": self.max_edges_per_action,
            "calibration_method": self.calibration_method,
            "resistant_percentile": self.resistant_percentile,
            "partial_resistant_percentile": self.partial_resistant_percentile,
            "sensitive_percentile": self.sensitive_percentile,
            "min_calibration_rows": self.min_calibration_rows,
            "random_controls": self.random_controls,
            "random_match_by_source": self.random_match_by_source,
            "random_match_by_node_type": self.random_match_by_node_type,
            "random_match_by_degree_bin": self.random_match_by_degree_bin,
        }


Task5Config = PerturbationConfig


@dataclass(frozen=True)
class CombinedConfig:
    enabled: bool = False
    output_dir: str = "Combined"
    source_run_dir: str | None = None
    cancer_type: str = "melanoma"
    candidate_strategy: str = "kg_target_pathway"
    combination_score_mode: str = "uplift_max"
    template_ids: tuple[str, ...] = (
        "T1_MAPK_vertical",
        "T2_MAPK_PI3K_escape",
        "T3_RTK_MAPK_bypass",
        "T4_MAPK_cell_cycle",
    )
    lambda_u: float = 0.1
    top_k_report: int = 25
    drugcomb_path: str | None = None
    nci_almanac_path: str | None = None
    run_drugcomb: bool = True
    run_nci_almanac: bool = True
    run_ablations: bool = True
    run_known_combo_recovery: bool = True
    fallback_scope: str = "cell_line_match"
    label_rule: str = "zip10_bliss5_or_hsa5"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "output_dir": self.output_dir,
            "source_run_dir": self.source_run_dir,
            "cancer_type": self.cancer_type,
            "candidate_strategy": self.candidate_strategy,
            "combination_score_mode": self.combination_score_mode,
            "template_ids": list(self.template_ids),
            "lambda_u": float(self.lambda_u),
            "top_k_report": int(self.top_k_report),
            "drugcomb_path": self.drugcomb_path,
            "nci_almanac_path": self.nci_almanac_path,
            "run_drugcomb": bool(self.run_drugcomb),
            "run_nci_almanac": bool(self.run_nci_almanac),
            "run_ablations": bool(self.run_ablations),
            "run_known_combo_recovery": bool(self.run_known_combo_recovery),
            "fallback_scope": self.fallback_scope,
            "label_rule": self.label_rule,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    benchmark_name: str
    dataset_name: str = DATASET_DEFAULT
    task_type: str = "gdsc_regression"
    split_type: str = "random_cell"
    split_seed: int = 7
    train_fraction: float = 0.70
    val_fraction: float = 0.10
    test_fraction: float = 0.20
    n_components: int = 512
    n_hvg: int = 3000
    epochs: int = 20
    batch_size: int = 512
    eval_batch_size: int | None = None
    lr: float = 1e-3
    top_k: int = 5
    evaluation_profiles: tuple[str, ...] = ("gdsc",)
    export_planning_predictions: bool = False
    export_terminal_latents: bool = True
    export_gate_audit: bool = False
    label_policy: str | None = None
    instance_unit: str | None = None
    max_rows: int | None = None
    external_validation_only: bool = False
    canonical_drug_indexing: bool = False
    coverage_aware_kg_mask: bool = True
    source_benchmark: str | None = None
    source_run_dir: str | None = None
    gdsc_source_mode: str = GDSC_SOURCE_MODE_DEFAULT
    cell_train_statistics_features: tuple[str, ...] = CELL_TRAIN_STATISTICS_FEATURE_COLUMNS
    prism_enabled: bool = False
    notes: tuple[str, ...] = ()
    family_order: tuple[str, ...] = tuple(TARGET_FAMILY_ORDER)
    ablation_variants: tuple[str, ...] = ()
    model: ModelConfig = field(default_factory=ModelConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    training_objectives: TrainingObjectiveConfig = field(default_factory=TrainingObjectiveConfig)
    batch: BatchSamplerConfig = field(default_factory=BatchSamplerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    combined: CombinedConfig = field(default_factory=CombinedConfig)
    controls: AblationControls = field(default_factory=AblationControls)
    prior: KGPriorConfig = field(default_factory=KGPriorConfig)
    static_prior: StaticPriorConfig = field(default_factory=StaticPriorConfig)

    def signature(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "split_type": self.split_type,
            "split_seed": self.split_seed,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "n_components": self.n_components,
            "n_hvg": self.n_hvg,
            "evaluation_profiles": list(self.evaluation_profiles),
            "eval_batch_size": self.eval_batch_size,
            "export_planning_predictions": self.export_planning_predictions,
            "export_terminal_latents": self.export_terminal_latents,
            "export_gate_audit": self.export_gate_audit,
            "label_policy": self.label_policy,
            "instance_unit": self.instance_unit,
            "external_validation_only": self.external_validation_only,
            "canonical_drug_indexing": self.canonical_drug_indexing,
            "coverage_aware_kg_mask": self.coverage_aware_kg_mask,
            "source_benchmark": self.source_benchmark,
            "prism_enabled": self.prism_enabled,
            "gdsc_source_mode": self.gdsc_source_mode if self.dataset_name == "gdsc" else None,
            "cell_train_statistics_features": list(self.cell_train_statistics_features),
            "family_order": list(self.family_order),
            "model": self.model.to_dict(),
            "world_model": self.world_model.to_dict(),
            "training_objectives": self.training_objectives.to_dict(),
            "batch": self.batch.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "explainability": self.explainability.to_dict(),
            "perturbation": self.perturbation.to_dict(),
            "combined": self.combined.to_dict(),
            "controls": self.controls.to_dict(),
            "prior": self.prior.to_dict(),
            "static_prior": self.static_prior.to_dict(),
            "ablation_variants": list(self.ablation_variants),
        }

    def prepare_signature(self) -> dict[str, Any]:
        payload = self.signature()
        for key in [
            "model",
            "world_model",
            "training_objectives",
            "batch",
            "evaluation",
            "explainability",
            "perturbation",
            "combined",
            "controls",
            "ablation_variants",
            "cell_train_statistics_features",
        ]:
            payload.pop(key, None)
        return payload

    @property
    def task5(self) -> PerturbationConfig:
        return self.perturbation

    def cache_key(self) -> str:
        return cache_key({"kind": "benchmark_config", **self.signature()})


def _path_is_within(root: Path, candidate: str | None) -> bool:
    if not candidate:
        return False
    try:
        resolved = Path(candidate).expanduser().resolve()
    except Exception:
        return False
    root = Path(root).expanduser().resolve()
    return resolved == root or root in resolved.parents


def _infer_dataset_name(payload: dict[str, Any]) -> str:
    explicit = payload.get("dataset_name")
    paths = Paths()
    ctrp_fields = [field for field in ("expr_path", "auc_path", "metadata_path") if payload.get(field)]
    ctrp_like = bool(ctrp_fields) and all(_path_is_within(paths.ctrp_dir, payload.get(field)) for field in ctrp_fields)
    if explicit is None or str(explicit).strip() == "":
        return "ctrdb" if ctrp_like else DATASET_DEFAULT
    dataset_name = normalize_dataset_name(explicit)
    if ctrp_like and dataset_name != "ctrdb":
        raise ValueError(
            "Benchmark config points to CTRP inputs but dataset_name is not ctrdb. "
            "Set dataset_name to ctrdb or leave it unset so it can be inferred."
        )
    return dataset_name


def _resolve_ctrp_benchmark_identity(payload: dict[str, Any], dataset_name: str) -> tuple[str, str]:
    benchmark_id = str(payload["benchmark_id"])
    benchmark_name = str(payload.get("benchmark_name", benchmark_id))
    if dataset_name != "ctrdb":
        return benchmark_id, benchmark_name
    split_type = str(payload.get("split_type", "random_cell")).strip()
    split_name = CTRP_BENCHMARK_NAME_BY_SPLIT.get(split_type, f"CTRP {split_type.replace('_', ' ').title()}")
    split_suffix = CTRP_BENCHMARK_SUFFIX_BY_SPLIT.get(split_type, split_type)
    if not benchmark_name.lower().startswith("ctrp "):
        benchmark_name = split_name
    if benchmark_id.endswith("_CTRP_v1") or benchmark_id.endswith("_CTRP_v2"):
        return benchmark_id, benchmark_name
    if "_CTRP_" in benchmark_id:
        return benchmark_id, benchmark_name
    if not benchmark_id.endswith("_CTRP_v2"):
        benchmark_id = f"{benchmark_id}_{split_suffix}_CTRP_v2"
    return benchmark_id, benchmark_name


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    controls = AblationControls(**payload.get("controls", {}))
    model_payload = payload.get("model", {}) or {}
    world_model_payload = payload.get("world_model", {}) or {}
    relative_payload = world_model_payload.get("relative_value", {}) or {}
    terminal_payload = world_model_payload.get("terminal_consequence", {}) or {}
    planner_payload = world_model_payload.get("planner", {}) or {}
    training_payload = payload.get("training", {}) or {}
    batch_payload = payload.get("batch", {}) or {}
    evaluation_payload = payload.get("evaluation", {}) or {}
    explainability_payload = payload.get("explainability", {}) or {}
    legacy_task5_payload = payload.get("task5", {}) or {}
    perturbation_payload = payload.get("perturbation", legacy_task5_payload) or legacy_task5_payload
    combined_payload = payload.get("combined", {}) or {}
    prior_payload = payload.get("prior", payload.get("kg_prior", {})) or {}
    static_prior_payload = payload.get("static_prior", {}) or {}
    top_k = int(payload.get("top_k", planner_payload.get("top_k", 5)))
    gdsc_source_mode = normalize_gdsc_source_mode(payload.get("gdsc_source_mode", GDSC_SOURCE_MODE_DEFAULT))
    dataset_name = _infer_dataset_name(payload)
    benchmark_id, benchmark_name = _resolve_ctrp_benchmark_identity(payload, dataset_name)
    raw_cell_train_statistics_features = payload.get("cell_train_statistics_features", list(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS))
    if raw_cell_train_statistics_features is None:
        raw_cell_train_statistics_features = list(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS)
    if not isinstance(raw_cell_train_statistics_features, list):
        raise ValueError("cell_train_statistics_features must be a YAML list of strings.")
    invalid_cell_train_statistics_features = [
        str(value)
        for value in raw_cell_train_statistics_features
        if str(value) not in CELL_TRAIN_STATISTICS_FEATURE_COLUMNS
    ]
    if invalid_cell_train_statistics_features:
        valid = ", ".join(CELL_TRAIN_STATISTICS_FEATURE_COLUMNS)
        invalid = ", ".join(invalid_cell_train_statistics_features)
        raise ValueError(
            f"cell_train_statistics_features contains invalid columns: {invalid}. "
            f"Valid options: {valid}."
        )
    cell_train_statistics_features = tuple(str(value) for value in raw_cell_train_statistics_features)

    def _source_config(name: str, default_node_types: tuple[str, ...], default_edge_types: tuple[str, ...]) -> KGPriorSourceConfig:
        source_payload = prior_payload.get(name, {}) or {}
        raw_node_types = source_payload.get("node_types")
        raw_edge_types = source_payload.get("edge_types")
        node_types = default_node_types if raw_node_types is None else tuple(raw_node_types)
        edge_types = default_edge_types if raw_edge_types is None else tuple(raw_edge_types)
        node_types = _validate_policy_values(name, "node_types", node_types, KG_SOURCE_NODE_TYPES[name])
        edge_types = _validate_policy_values(name, "edge_types", edge_types, KG_SOURCE_EDGE_TYPES[name])
        return KGPriorSourceConfig(
            enabled=bool(source_payload.get("enabled", True)),
            weight=float(source_payload.get("weight", 1.0)),
            node_types=node_types,
            edge_types=edge_types,
            release=normalize_chembl_release(source_payload.get("release", CHEMBL_RELEASE_DEFAULT)) if name == "chembl" else CHEMBL_RELEASE_DEFAULT,
            sqlite_tar=source_payload.get("sqlite_tar") if name == "chembl" else None,
        )

    return BenchmarkConfig(
        benchmark_id=benchmark_id,
        benchmark_name=benchmark_name,
        dataset_name=dataset_name,
        task_type=str(payload.get("task_type", "gdsc_regression")),
        split_type=str(payload.get("split_type", "random_cell")),
        split_seed=int(payload.get("split_seed", 7)),
        train_fraction=float(payload.get("train_fraction", 0.70)),
        val_fraction=float(payload.get("val_fraction", 0.10)),
        test_fraction=float(payload.get("test_fraction", 0.20)),
        n_components=int(payload.get("n_components", 512)),
        n_hvg=int(payload.get("n_hvg", 3000)),
        epochs=int(payload.get("epochs", 20)),
        batch_size=int(payload.get("batch_size", 512)),
        eval_batch_size=payload.get("eval_batch_size"),
        lr=float(payload.get("lr", 1e-3)),
        top_k=top_k,
        evaluation_profiles=tuple(payload.get("evaluation_profiles", ["gdsc"])),
        export_planning_predictions=bool(payload.get("export_planning_predictions", False)),
        export_terminal_latents=bool(payload.get("export_terminal_latents", True)),
        export_gate_audit=bool(payload.get("export_gate_audit", False)),
        label_policy=payload.get("label_policy"),
        instance_unit=payload.get("instance_unit"),
        max_rows=payload.get("max_rows"),
        external_validation_only=bool(payload.get("external_validation_only", False)),
        canonical_drug_indexing=bool(payload.get("canonical_drug_indexing", False)),
        coverage_aware_kg_mask=bool(payload.get("coverage_aware_kg_mask", True)),
        source_benchmark=payload.get("source_benchmark"),
        source_run_dir=payload.get("source_run_dir"),
        gdsc_source_mode=gdsc_source_mode,
        cell_train_statistics_features=cell_train_statistics_features,
        prism_enabled=bool(payload.get("prism_enabled", False)),
        notes=tuple(payload.get("notes", [])),
        family_order=tuple(payload.get("family_order", TARGET_FAMILY_ORDER)),
        ablation_variants=tuple(payload.get("ablation_variants", [])),
        model=ModelConfig(
            mode=str(model_payload.get("mode", "regression")),
            target_field=str(model_payload.get("target_field", "AUC")),
        ),
        world_model=WorldModelConfig(
            relative_value_enabled=bool(relative_payload.get("enabled", True)),
            relative_value_train_field=str(relative_payload.get("train_field", "relative_value_train")),
            relative_value_eval_field=str(relative_payload.get("eval_field", "relative_value_eval")),
            relative_value_direction=str(relative_payload.get("direction", "auc_lower_is_better")),
            terminal_consequence_enabled=bool(terminal_payload.get("enabled", True)),
            planner_enabled=bool(planner_payload.get("enabled", True)),
            planner_top_k=int(planner_payload.get("top_k", top_k)),
        ),
        training_objectives=TrainingObjectiveConfig(
            loss_raw_weight=float(training_payload.get("loss_raw_weight", 1.0)),
            loss_value_weight=float(training_payload.get("loss_value_weight", 1.0)),
            loss_drug_centered_weight=float(training_payload.get("loss_drug_centered_weight", 0.0)),
            loss_cell_residual_weight=float(training_payload.get("loss_cell_residual_weight", training_payload.get("loss_drug_centered_weight", 0.0))),
            loss_within_drug_rank_weight=float(training_payload.get("loss_within_drug_rank_weight", 0.05)),
            loss_policy_advantage_weight=float(training_payload.get("loss_policy_advantage_weight", 0.05)),
            loss_same_cell_cross_drug_rank_weight=float(
                training_payload.get("loss_same_cell_cross_drug_rank_weight", training_payload.get("loss_policy_advantage_weight", 0.05))
            ),
            loss_terminal_drug_specificity_weight=float(training_payload.get("loss_terminal_drug_specificity_weight", 0.0)),
            loss_graph_consistency_weight=float(training_payload.get("loss_graph_consistency_weight", 0.02)),
            validation_objective=str(training_payload.get("validation_objective", "val_within_drug_pcc")),
            rank_margin=float(training_payload.get("rank_margin", 0.05)),
            advantage_margin=float(training_payload.get("advantage_margin", 0.05)),
            terminal_drug_specificity_margin=float(training_payload.get("terminal_drug_specificity_margin", training_payload.get("advantage_margin", 0.05))),
            min_relative_value_gap=float(training_payload.get("min_relative_value_gap", 0.15)),
            min_auc_gap=float(training_payload.get("min_auc_gap", 0.05)),
        ),
        batch=BatchSamplerConfig(
            sampler=str(batch_payload.get("sampler", "random_pair")),
            n_cell_lines=int(batch_payload.get("n_cell_lines", 16)),
            n_drugs=int(batch_payload.get("n_drugs", 8)),
            rank_batch_fraction=float(batch_payload.get("rank_batch_fraction", 0.25)),
            min_cell_drugs_per_batch=int(batch_payload.get("min_cell_drugs_per_batch", 2)),
            min_drug_cells_per_batch=int(batch_payload.get("min_drug_cells_per_batch", 2)),
        ),
        evaluation=EvaluationConfig(
            planner_metric_field=str(evaluation_payload.get("planner_metric_field", "relative_value_eval")),
            fusion_selection_metric=str(evaluation_payload.get("fusion_selection_metric", "within_drug_pcc_mean")),
            fusion_weight_candidates=tuple(float(x) for x in evaluation_payload.get("fusion_weight_candidates", [0.0, 0.5, 1.0, 1.5, 2.0])),
            target_within_drug_pcc=(
                None
                if evaluation_payload.get("target_within_drug_pcc") is None
                else float(evaluation_payload.get("target_within_drug_pcc"))
            ),
            sample_guardrail_delta=(
                None
                if evaluation_payload.get("sample_guardrail_delta") is None
                else float(evaluation_payload.get("sample_guardrail_delta"))
            ),
            strict_test_rows=(
                None
                if evaluation_payload.get("strict_test_rows") is None
                else int(evaluation_payload.get("strict_test_rows"))
            ),
            strict_test_drugs=(
                None
                if evaluation_payload.get("strict_test_drugs") is None
                else int(evaluation_payload.get("strict_test_drugs"))
            ),
        ),
        explainability=ExplainabilityConfig(
            enabled=bool(explainability_payload.get("enabled", False)),
            level=str(explainability_payload.get("level", "source")),
            export_top_edges=bool(explainability_payload.get("export_top_edges", False)),
            export_top_nodes=bool(explainability_payload.get("export_top_nodes", False)),
            export_top_paths=bool(explainability_payload.get("export_top_paths", False)),
            run_ablation=bool(explainability_payload.get("run_ablation", False)),
            top_k_edges=int(explainability_payload.get("top_k_edges", 20)),
            top_k_nodes=int(explainability_payload.get("top_k_nodes", 20)),
            top_k_paths=int(explainability_payload.get("top_k_paths", 10)),
            path_max_hops=int(explainability_payload.get("path_max_hops", 4)),
        ),
        perturbation=PerturbationConfig(
            enabled=bool(perturbation_payload.get("enabled", False)),
            stub_mode=bool(perturbation_payload.get("stub_mode", False)),
            output_dir=str(perturbation_payload.get("output_dir", "perturbation")),
            action_top_k=int(perturbation_payload.get("action_top_k", 20)),
            action_levels=tuple(perturbation_payload.get("action_levels", ["edge", "node"])),
            node_top_k=int(perturbation_payload.get("node_top_k", 10)),
            allowed_node_types=tuple(
                perturbation_payload.get(
                    "allowed_node_types",
                    ["gene", "protein", "pathway", "biological_process", "molecular_function"],
                )
            ),
            max_edges_per_action=int(perturbation_payload.get("max_edges_per_action", 100)),
            calibration_method=str(perturbation_payload.get("calibration_method", "drug_specific_percentile")),
            resistant_percentile=float(perturbation_payload.get("resistant_percentile", 0.80)),
            partial_resistant_percentile=float(perturbation_payload.get("partial_resistant_percentile", 0.60)),
            sensitive_percentile=float(perturbation_payload.get("sensitive_percentile", 0.25)),
            min_calibration_rows=int(perturbation_payload.get("min_calibration_rows", 10)),
            random_controls=int(perturbation_payload.get("random_controls", 100)),
            random_match_by_source=bool(perturbation_payload.get("random_match_by_source", True)),
            random_match_by_node_type=bool(perturbation_payload.get("random_match_by_node_type", True)),
            random_match_by_degree_bin=bool(perturbation_payload.get("random_match_by_degree_bin", True)),
        ),
        combined=CombinedConfig(
            enabled=bool(combined_payload.get("enabled", False)),
            output_dir=str(combined_payload.get("output_dir", "Combined")),
            source_run_dir=combined_payload.get("source_run_dir"),
            cancer_type=str(combined_payload.get("cancer_type", "melanoma")),
            candidate_strategy=str(combined_payload.get("candidate_strategy", "kg_target_pathway")),
            combination_score_mode=str(combined_payload.get("combination_score_mode", "uplift_max")),
            template_ids=tuple(
                combined_payload.get(
                    "template_ids",
                    [
                        "T1_MAPK_vertical",
                        "T2_MAPK_PI3K_escape",
                        "T3_RTK_MAPK_bypass",
                        "T4_MAPK_cell_cycle",
                    ],
                )
            ),
            lambda_u=float(combined_payload.get("lambda_u", 0.1)),
            top_k_report=int(combined_payload.get("top_k_report", 25)),
            drugcomb_path=combined_payload.get("drugcomb_path"),
            nci_almanac_path=combined_payload.get("nci_almanac_path"),
            run_drugcomb=bool(combined_payload.get("run_drugcomb", True)),
            run_nci_almanac=bool(combined_payload.get("run_nci_almanac", True)),
            run_ablations=bool(combined_payload.get("run_ablations", True)),
            run_known_combo_recovery=bool(combined_payload.get("run_known_combo_recovery", True)),
            fallback_scope=str(combined_payload.get("fallback_scope", "cell_line_match")),
            label_rule=str(combined_payload.get("label_rule", "zip10_bliss5_or_hsa5")),
        ),
        controls=controls,
        prior=KGPriorConfig(
            chembl=_source_config("chembl", KG_SOURCE_NODE_TYPES["chembl"], KG_SOURCE_EDGE_TYPES["chembl"]),
            drkg=_source_config("drkg", KG_SOURCE_NODE_TYPES["drkg"], KG_SOURCE_EDGE_TYPES["drkg"]),
            primekg=_source_config("primekg", KG_SOURCE_NODE_TYPES["primekg"], KG_SOURCE_EDGE_TYPES["primekg"]),
            include_side_effects=bool(prior_payload.get("include_side_effects", False)),
        ),
        static_prior=StaticPriorConfig(
            enabled=bool(static_prior_payload.get("enabled", True)),
            sources=tuple(static_prior_payload.get("sources", ("primekg", "gdsc_screened_compounds"))),
        ),
    )


def benchmark_paths(benchmark_dir: Path) -> dict[str, Path]:
    benchmark_dir = Path(benchmark_dir).resolve()
    return {
        "benchmark_dir": benchmark_dir,
        "config": benchmark_dir / "configs" / "default.yaml",
        "data": benchmark_dir / "data",
        "raw_links": benchmark_dir / "data" / "raw_links",
        "processed": benchmark_dir / "data" / "processed",
        "results": benchmark_dir / "results",
    }


def split_entities(items: list[Any], seed: int, train_fraction: float, val_fraction: float, test_fraction: float) -> dict[Any, str]:
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")
    unique = list(dict.fromkeys(items))
    if len(unique) < 3:
        raise ValueError("Need at least 3 unique entities to build train/val/test splits.")
    train_items, rest = train_test_split(unique, train_size=train_fraction, random_state=seed)
    rest_fraction = val_fraction + test_fraction
    if len(rest) == 0:
        val_items = []
        test_items = []
    elif len(rest) == 1:
        val_items = rest
        test_items = []
    elif len(rest) == 2:
        val_items = [rest[0]]
        test_items = [rest[1]]
    else:
        val_share = val_fraction / rest_fraction if rest_fraction else 0.5
        val_items, test_items = train_test_split(rest, train_size=val_share, random_state=seed)
    out = {item: "train" for item in train_items}
    out.update({item: "val" for item in val_items})
    out.update({item: "test" for item in test_items})
    return out


def infer_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return f"fallback_smiles::{normalize_name(smiles)}"

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid_smiles"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold or "acyclic"


def infer_target_families(responses: pd.DataFrame, primekg_path: Path) -> pd.DataFrame:
    drug_meta = responses[["DRUG_ID", "DRUG_NAME"]].drop_duplicates("DRUG_ID").copy()
    lookup = {
        normalize_name(name): {"DRUG_ID": int(drug_id), "DRUG_NAME": str(name)}
        for drug_id, name in drug_meta[["DRUG_ID", "DRUG_NAME"]].itertuples(index=False)
    }
    counts = {key: {family: 0 for family in TARGET_FAMILY_ORDER} for key in lookup}
    chunks = pd.read_csv(
        primekg_path,
        usecols=["x_name", "y_name", "relation", "display_relation"],
        chunksize=250_000,
    )
    for chunk in chunks:
        x_key = chunk["x_name"].map(normalize_name)
        y_key = chunk["y_name"].map(normalize_name)
        rel_text = (
            chunk["x_name"].astype(str)
            + " "
            + chunk["y_name"].astype(str)
            + " "
            + chunk["relation"].astype(str)
            + " "
            + chunk["display_relation"].astype(str)
        ).str.lower()
        for family, keywords in TARGET_FAMILY_KEYWORDS.items():
            hit = rel_text.apply(lambda text: any(keyword in text for keyword in keywords))
            for key_series in [x_key, y_key]:
                keys = key_series.loc[hit & key_series.isin(lookup)]
                for key in keys:
                    counts[key][family] += 1
    rows = []
    for key, meta in lookup.items():
        family_scores = counts[key]
        family = max(family_scores, key=family_scores.get)
        if family_scores[family] == 0:
            family = "other"
        rows.append(
            {
                "DRUG_ID": meta["DRUG_ID"],
                "DRUG_NAME": meta["DRUG_NAME"],
                "target_family": family,
                "family_source": "primekg_keyword_scan" if family != "other" else "fallback_other",
            }
        )
    return pd.DataFrame(rows)


def apply_split(
    mapped: pd.DataFrame,
    config: BenchmarkConfig,
    paths: Paths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = mapped.copy()
    split_audit_rows: list[dict[str, Any]] = []
    if config.split_type in {"all_data", "full_data", "all"}:
        frame["split"] = "train"
        split_audit_rows.append(
            {
                "audit_type": "full_data_summary",
                "split": "train",
                "n_rows": int(len(frame)),
                "n_cells": int(frame["SANGER_MODEL_ID"].astype(str).nunique()),
                "n_drugs": int(frame["DRUG_ID"].astype(int).nunique()),
                "detail": config.split_type,
            }
        )
        return frame, pd.DataFrame(split_audit_rows)
    if config.split_type == "random_cell":
        split_map = split_entities(
            sorted(frame["SANGER_MODEL_ID"].astype(str).unique()),
            seed=config.split_seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
        )
        frame["split"] = frame["SANGER_MODEL_ID"].map(split_map)
    elif config.split_type == "drug":
        split_map = split_entities(
            sorted(frame["DRUG_ID"].astype(int).unique().tolist()),
            seed=config.split_seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
        )
        frame["split"] = frame["DRUG_ID"].map(split_map)
    elif config.split_type == "scaffold":
        drug_scaffold = frame[["DRUG_ID", "DRUG_NAME", "smiles"]].drop_duplicates("DRUG_ID").copy()
        drug_scaffold["scaffold"] = drug_scaffold["smiles"].map(infer_scaffold)
        scaffold_map = split_entities(
            sorted(drug_scaffold["scaffold"].astype(str).unique().tolist()),
            seed=config.split_seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
        )
        frame = frame.merge(drug_scaffold[["DRUG_ID", "scaffold"]], on="DRUG_ID", how="left")
        frame["split"] = frame["scaffold"].map(scaffold_map)
        invalid = drug_scaffold.loc[drug_scaffold["scaffold"].eq("invalid_smiles")]
        for row in invalid.itertuples(index=False):
            split_audit_rows.append(
                {
                    "audit_type": "invalid_scaffold_smiles",
                    "DRUG_ID": int(row.DRUG_ID),
                    "DRUG_NAME": row.DRUG_NAME,
                    "group_key": row.scaffold,
                    "detail": "RDKit failed to parse SMILES; grouped as invalid_smiles.",
                }
            )
    elif config.split_type == "target_family":
        family_map = infer_target_families(frame, paths.primekg)
        present_families = family_map["target_family"].astype(str).unique().tolist()
        family_groups = [family for family in config.family_order if family in present_families]
        family_split = split_entities(
            family_groups,
            seed=config.split_seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
        )
        frame = frame.merge(family_map, on=["DRUG_ID", "DRUG_NAME"], how="left")
        frame["target_family"] = frame["target_family"].fillna("other")
        frame["split"] = frame["target_family"].map(family_split)
    elif config.split_type == "double_disjoint":
        adjacency: dict[str, set[str]] = defaultdict(set)
        for cell, drug in frame[["SANGER_MODEL_ID", "DRUG_ID"]].drop_duplicates().itertuples(index=False):
            c_node = f"cell::{cell}"
            d_node = f"drug::{int(drug)}"
            adjacency[c_node].add(d_node)
            adjacency[d_node].add(c_node)
        components: list[list[str]] = []
        seen: set[str] = set()
        for node in adjacency:
            if node in seen:
                continue
            stack = [node]
            comp: list[str] = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                comp.append(current)
                stack.extend(adjacency[current] - seen)
            components.append(sorted(comp))
        if len(components) >= 3:
            component_keys = ["|".join(comp) for comp in components]
            component_split = split_entities(
                component_keys,
                seed=config.split_seed,
                train_fraction=config.train_fraction,
                val_fraction=config.val_fraction,
                test_fraction=config.test_fraction,
            )
            cell_split: dict[str, str] = {}
            drug_split: dict[int, str] = {}
            for comp, key in zip(components, component_keys):
                split_name = component_split[key]
                for node in comp:
                    prefix, raw = node.split("::", 1)
                    if prefix == "cell":
                        cell_split[raw] = split_name
                    else:
                        drug_split[int(raw)] = split_name
            frame["cell_split"] = frame["SANGER_MODEL_ID"].astype(str).map(cell_split)
            frame["drug_split"] = frame["DRUG_ID"].astype(int).map(drug_split)
            frame["split"] = frame["cell_split"]
        else:
            best_frame = None
            best_score = (-1, -1)
            for offset in range(64):
                cell_split = split_entities(
                    sorted(frame["SANGER_MODEL_ID"].astype(str).unique()),
                    seed=config.split_seed + offset,
                    train_fraction=config.train_fraction,
                    val_fraction=config.val_fraction,
                    test_fraction=config.test_fraction,
                )
                drug_split = split_entities(
                    sorted(frame["DRUG_ID"].astype(int).unique().tolist()),
                    seed=config.split_seed + offset,
                    train_fraction=config.train_fraction,
                    val_fraction=config.val_fraction,
                    test_fraction=config.test_fraction,
                )
                candidate = frame.copy()
                candidate["cell_split"] = candidate["SANGER_MODEL_ID"].astype(str).map(cell_split)
                candidate["drug_split"] = candidate["DRUG_ID"].astype(int).map(drug_split)
                candidate = candidate.loc[candidate["cell_split"].eq(candidate["drug_split"])].copy()
                candidate["split"] = candidate["cell_split"]
                score = (candidate["split"].nunique(), len(candidate))
                if score > best_score:
                    best_frame = candidate
                    best_score = score
                if score[0] == 3:
                    break
            frame = best_frame if best_frame is not None else frame.iloc[0:0].copy()
    else:
        raise ValueError(f"Unsupported split_type: {config.split_type}")
    frame = frame.loc[frame["split"].isin(["train", "val", "test"])].copy()
    if frame["split"].nunique() < 3:
        raise ValueError(f"Split {config.split_type} did not produce train/val/test rows after filtering.")
    for split_name, group in frame.groupby("split", observed=True):
        split_audit_rows.append(
            {
                "audit_type": "split_summary",
                "split": split_name,
                "n_rows": int(len(group)),
                "n_cells": int(group["SANGER_MODEL_ID"].astype(str).nunique()),
                "n_drugs": int(group["DRUG_ID"].astype(int).nunique()),
                "detail": config.split_type,
            }
        )
    return frame, pd.DataFrame(split_audit_rows)


def latest_result_dir(results_dir: Path) -> Path | None:
    candidates = [p for p in Path(results_dir).iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.name) if candidates else None


def write_raw_manifest(raw_links_dir: Path, paths: Paths, config: BenchmarkConfig) -> None:
    ensure_dir(raw_links_dir)
    dataset_name = normalize_dataset_name(getattr(config, "dataset_name", DATASET_DEFAULT))
    if dataset_name == "beataml2":
        raw_sources = {
            "beataml2_curve_fits": str(paths.beataml2_curve_fits),
            "beataml2_raw_inhibitor": str(paths.beataml2_raw_inhibitor),
            "beataml2_expression": str(paths.beataml2_expression),
            "beataml2_counts": str(paths.beataml2_counts),
            "beataml2_drug_families": str(paths.beataml2_drug_families),
            "beataml2_sample_mapping": str(paths.beataml2_sample_mapping),
            "beataml2_clinical": str(paths.beataml2_clinical),
            "beataml2_smiles_cache": str(paths.beataml2_smiles_cache),
            "ctrp_response": str(paths.ctrp_response),
            "ctrp_gene_expression": str(paths.ctrp_gene_expression),
            "ctrp_drug_names": str(paths.ctrp_drug_names),
            "ctrp_drug_smiles": str(paths.ctrp_drug_smiles),
            "ctrdb_smiles_cache": str(paths.ctrdb_smiles_cache),
            "tcga_smiles_cache": str(paths.tcga_smiles_cache),
            "primekg": str(paths.primekg),
            "tcga_h5ad": str(paths.tcga_h5ad),
            "ctrdb_microarray_h5ad": str(paths.ctrdb_microarray_h5ad),
        }
    elif dataset_name == "ctrdb":
        raw_sources = {
            "ctrp_dir": str(paths.ctrp_dir),
            "ctrp_response": str(paths.ctrp_response),
            "ctrp_gene_expression": str(paths.ctrp_gene_expression),
            "ctrp_drug_names": str(paths.ctrp_drug_names),
            "ctrp_drug_smiles": str(paths.ctrp_drug_smiles),
            "primekg": str(paths.primekg),
        }
    elif dataset_name == "pdx":
        raw_sources = {
            "pdx_response": str(paths.pdx_response),
            "pdx_gene_expression": str(paths.pdx_gene_expression),
            "pdx_drug_names": str(paths.pdx_drug_names),
            "pdx_drug_smiles": str(paths.pdx_drug_smiles),
            "pdx_smiles_cache": str(paths.dataset_smiles_dir / "pdx_smiles.csv"),
            "ctrp_response": str(paths.ctrp_response),
            "ctrp_gene_expression": str(paths.ctrp_gene_expression),
            "ctrp_drug_names": str(paths.ctrp_drug_names),
            "ctrp_drug_smiles": str(paths.ctrp_drug_smiles),
            "ctrdb_smiles_cache": str(paths.ctrdb_smiles_cache),
            "tcga_smiles_cache": str(paths.tcga_smiles_cache),
            "primekg": str(paths.primekg),
            "tcga_h5ad": str(paths.tcga_h5ad),
            "ctrdb_microarray_h5ad": str(paths.ctrdb_microarray_h5ad),
        }
    else:
        gdsc_files = gdsc_fitted_paths(paths, config.gdsc_source_mode)
        raw_sources = {
            "gdsc_fitted_1": str(paths.gdsc_fitted_1),
            "gdsc_screened_compounds": str(paths.gdsc_screened_compounds),
            "gdsc_expression": str(paths.gdsc_expression),
            "gdsc_gene_identifiers": str(paths.gdsc_gene_identifiers),
            "gdsc_smiles_cache": str(paths.gdsc_smiles_cache),
            "primekg": str(paths.primekg),
        }
        if len(gdsc_files) > 1:
            raw_sources["gdsc_fitted_2"] = str(paths.gdsc_fitted_2)
    payload = {
        "benchmark_id": config.benchmark_id,
        "benchmark_name": config.benchmark_name,
        "dataset_name": dataset_name,
        "gdsc_source_mode": config.gdsc_source_mode if dataset_name == "gdsc" else None,
        "resolved_prior_policy": {
            "kg_prior": config.prior.to_dict(),
            "static_prior": config.static_prior.to_dict(),
        },
        "raw_sources": raw_sources,
    }
    write_json(raw_links_dir / "manifest.json", payload)
