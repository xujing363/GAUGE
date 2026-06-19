from __future__ import annotations

import argparse
import pickle
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from .cell_state_audit import analyze_cell_state
from .benchmarking import (
    AblationControls,
    benchmark_paths,
    latest_result_dir,
    load_benchmark_config,
    write_raw_manifest,
)
from .ctrp_v2 import ensure_ctrp_v2_runtime
from .config import Paths
from .contracts import build_prepare_contract, prepare_contract_mismatch_reasons
from .external import evaluate_ctrdb_response, evaluate_tcga_actual_treatments
from .tcga_binary import run_tcga_binary_benchmark
from .repro import RUNTIME_PROFILE_STRICT, set_reproducible_runtime
from .runtime import release_gpu_memory
from .visualization import generate_run_visualizations
from .train import (
    TrainExecutionContext,
    _write_prepare_outputs,
    apply_cell_train_statistics_feature_selection,
    evaluate_gdsc,
    load_model,
    load_prepared,
    prepare_data,
    train_model,
)
from .utils import copy_config_snapshot, ensure_dir

CANDIDATE_DRUG_LEVEL_TARGET = 0.60
CANDIDATE_SAMPLE_GUARD_DELTA = -0.01


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark wrapper around GAUGE.")
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument(
        "--config",
        default=None,
        help="Benchmark config path relative to --benchmark-dir, or an absolute path.",
    )
    p.add_argument("--processed-name", default="default")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--runtime-profile", default=RUNTIME_PROFILE_STRICT)
    p.add_argument("--eval-compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-components", type=int, default=None)
    p.add_argument("--n-hvg", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--source-run-dir", type=Path, default=None)
    p.add_argument("--gdsc-source-mode", choices=("v1", "v2", "both"), default=None)
    p.add_argument("--tcga-use-hvg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--export-planning-predictions", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--export-terminal-latents", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--export-gate-audit", action=argparse.BooleanOptionalAction, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("train")
    sub.add_parser("evaluate")
    sub.add_parser("run")
    analyze = sub.add_parser("analyze-cell-state")
    analyze.add_argument("--result-dir", type=Path, required=True)
    return p


def validate_device(value: str | None) -> str:
    if not value:
        raise SystemExit("Explicit device is required. Use cuda:N or cpu.")
    if value != "cpu" and not value.startswith("cuda:"):
        raise SystemExit("Invalid device. Use cuda:N or cpu.")
    return value


def _resolve_config_path(bench_paths: dict[str, Path], config_arg: str | Path | None) -> Path:
    if config_arg is None:
        return bench_paths["config"]
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = bench_paths["benchmark_dir"] / config_path
    return config_path.resolve()


def _resolve_eval_options(args: argparse.Namespace, config, device: str) -> dict[str, object]:
    default_eval_batch_size = _default_eval_batch_size(config, device)
    return {
        "eval_batch_size": int(args.eval_batch_size or config.eval_batch_size or default_eval_batch_size),
        "export_planning_predictions": (
            config.export_planning_predictions
            if args.export_planning_predictions is None
            else bool(args.export_planning_predictions)
        ),
        "export_terminal_latents": (
            config.export_terminal_latents
            if args.export_terminal_latents is None
            else bool(args.export_terminal_latents)
        ),
        "export_gate_audit": (
            config.export_gate_audit
            if args.export_gate_audit is None
            else bool(args.export_gate_audit)
        ),
    }


def _default_eval_batch_size(config, device: str) -> int:
    if device == "cpu":
        return 4096
    if config.benchmark_id == "07_tcga_actual_treatment":
        return 32768
    if config.benchmark_id == "06_cross_dataset":
        return 24576
    return 16384


def _resolve_train_options(args: argparse.Namespace, config) -> dict[str, int | float]:
    return {
        "epochs": int(args.epochs if args.epochs is not None else config.epochs),
        "batch_size": int(args.batch_size if args.batch_size is not None else config.batch_size),
        "lr": float(args.lr if args.lr is not None else config.lr),
    }


def _maybe_materialize_combined_source_run(config, result_dir: Path) -> None:
    combined = getattr(config, "combined", None)
    source_run_dir = None
    if combined is not None and bool(getattr(combined, "enabled", False)):
        source_run_dir = getattr(combined, "source_run_dir", None) or getattr(config, "source_run_dir", None)
    if not source_run_dir:
        return
    source_run = Path(source_run_dir).expanduser().resolve()
    for name in ("model.pt", "artifacts.pkl"):
        source_path = source_run / name
        target_path = result_dir / name
        if target_path.exists():
            continue
        if not source_path.exists():
            raise SystemExit(f"Combined source_run_dir is missing required file: {source_path}")
        shutil.copy2(source_path, target_path)


def _log_batch_resolution(
    *,
    config,
    effective_epochs: int | None,
    effective_batch_size: int | None,
    effective_eval_batch_size: int | None,
) -> None:
    def _fmt(value: object) -> str:
        return "null" if value is None else str(value)

    print(
        "[CONFIG] epochs={} batch_size={} eval_batch_size={}".format(
            _fmt(getattr(config, "epochs", None)),
            _fmt(getattr(config, "batch_size", None)),
            _fmt(getattr(config, "eval_batch_size", None)),
        )
    )
    print(
        "[EFFECTIVE] epochs={} batch_size={} eval_batch_size={}".format(
            _fmt(effective_epochs),
            _fmt(effective_batch_size),
            _fmt(effective_eval_batch_size),
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ensure_ctrp_v2_runtime()
    set_reproducible_runtime(seed=args.seed, device=args.device or "cpu", profile=args.runtime_profile)
    paths = Paths()
    bench_paths = benchmark_paths(args.benchmark_dir)
    config_path = _resolve_config_path(bench_paths, args.config)
    config = load_benchmark_config(config_path)
    if getattr(args, "split_seed", None) is not None:
        config = replace(config, split_seed=int(args.split_seed))
    write_raw_manifest(bench_paths["raw_links"], paths, config)
    processed_dir = ensure_dir(bench_paths["processed"] / args.processed_name)
    model = None
    prepared = None
    artifacts = None
    context = None
    try:
        if config.task_type == "tcga_binary_response":
            run_tcga_binary_benchmark(args, config, bench_paths, paths)
            return
        if args.cmd == "prepare":
            prepare_data(
                paths,
                processed_dir,
                benchmark=config,
                seed=args.seed,
                n_components=args.n_components or config.n_components,
                max_rows=args.max_rows if args.max_rows is not None else config.max_rows,
                gdsc_source_mode=getattr(args, "gdsc_source_mode", None) or config.gdsc_source_mode,
            )
            return
        if args.cmd == "analyze-cell-state":
            validate_device(args.device)
            prepared = _load_or_prepare(processed_dir, paths, config, args)
            analyze_cell_state(
                benchmark=config,
                prepared=prepared,
                paths=paths,
                result_dir=args.result_dir,
                device=args.device,
            )
            return
        if config.benchmark_id == "07_tcga_actual_treatment":
            _run_tcga_external_only(args, config, bench_paths, paths)
            return
        validate_device(args.device)
        if args.cmd == "evaluate" and args.run_name is None:
            latest = latest_result_dir(bench_paths["results"])
            if latest is None:
                raise SystemExit("No existing result directory found for evaluate. Pass --run-name after a train/run step.")
            result_dir = latest
        else:
            run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = ensure_dir(bench_paths["results"] / run_name)
        copy_config_snapshot(config_path, [result_dir])
        if args.cmd == "train":
            prepared = _load_or_prepare(processed_dir, paths, config, args)
            context = TrainExecutionContext()
            train_options = _resolve_train_options(args, config)
            _log_batch_resolution(
                config=config,
                effective_epochs=int(train_options["epochs"]),
                effective_batch_size=int(train_options["batch_size"]),
                effective_eval_batch_size=None,
            )
            train_model(
                prepared,
                result_dir,
                epochs=int(train_options["epochs"]),
                batch_size=int(train_options["batch_size"]),
                lr=float(train_options["lr"]),
                seed=args.seed,
                device=args.device,
                benchmark=config,
                controls=config.controls,
                context=context,
                runtime_profile=args.runtime_profile,
            )
            _copy_prepare_contract(processed_dir, result_dir)
            _write_prepare_outputs(prepared, result_dir)
            generate_run_visualizations(result_dir, device=args.device)
            return
        if args.cmd == "evaluate":
            prepared = _load_or_prepare(processed_dir, paths, config, args)
            context = TrainExecutionContext()
            _maybe_materialize_combined_source_run(config, result_dir)
            model = load_model(
                result_dir,
                prepared.artifacts,
                strict=not bool(getattr(getattr(config, "combined", None), "enabled", False)),
            )
            eval_options = _resolve_eval_options(args, config, args.device)
            _log_batch_resolution(
                config=config,
                effective_epochs=getattr(config, "epochs", None),
                effective_batch_size=getattr(config, "batch_size", None),
                effective_eval_batch_size=int(eval_options["eval_batch_size"]),
            )
            _evaluate_profiles(
                config,
                model,
                prepared,
                result_dir,
                paths,
                args.device,
                args.top_k or config.top_k,
                benchmark=config,
                context=context,
                runtime_profile=args.runtime_profile,
                eval_compile=bool(args.eval_compile),
                **eval_options,
            )
            _copy_prepare_contract(processed_dir, result_dir)
            _write_prepare_outputs(prepared, result_dir)
            _materialize_contracts(config, processed_dir, result_dir)
            generate_run_visualizations(result_dir, device=args.device)
            return
        if args.cmd == "run":
            if config.ablation_variants:
                _run_ablations(args, config, bench_paths, paths, processed_dir, result_dir, config_path)
                return
            prepared = _load_or_prepare(processed_dir, paths, config, args)
            context = TrainExecutionContext()
            train_options = _resolve_train_options(args, config)
            eval_options = _resolve_eval_options(args, config, args.device)
            _log_batch_resolution(
                config=config,
                effective_epochs=int(train_options["epochs"]),
                effective_batch_size=int(train_options["batch_size"]),
                effective_eval_batch_size=int(eval_options["eval_batch_size"]),
            )
            model = train_model(
                prepared,
                result_dir,
                epochs=int(train_options["epochs"]),
                batch_size=int(train_options["batch_size"]),
                lr=float(train_options["lr"]),
                seed=args.seed,
                device=args.device,
                benchmark=config,
                controls=config.controls,
                context=context,
                runtime_profile=args.runtime_profile,
            )
            _copy_prepare_contract(processed_dir, result_dir)
            _write_prepare_outputs(prepared, result_dir)
            _evaluate_profiles(
                config,
                model,
                prepared,
                result_dir,
                paths,
                args.device,
                args.top_k or config.top_k,
                benchmark=config,
                context=context,
                runtime_profile=args.runtime_profile,
                eval_compile=bool(args.eval_compile),
                **eval_options,
            )
            _materialize_contracts(config, processed_dir, result_dir)
            generate_run_visualizations(result_dir, device=args.device)
    finally:
        release_gpu_memory(model, prepared, artifacts, context)


def _load_or_prepare(processed_dir: Path, paths: Paths, config, args) -> object:
    prepared_path = processed_dir / "prepared.pkl"
    gdsc_source_mode = getattr(args, "gdsc_source_mode", None) or config.gdsc_source_mode
    expected = build_prepare_contract(
        paths=paths,
        benchmark=config,
        seed=args.seed,
        n_components=args.n_components or config.n_components,
        max_rows=args.max_rows if args.max_rows is not None else config.max_rows,
        gdsc_source_mode=gdsc_source_mode,
    )
    if prepared_path.exists():
        prepared = load_prepared(prepared_path)
        mismatch_reasons = prepare_contract_mismatch_reasons(prepared, expected)
        if not mismatch_reasons:
            return apply_cell_train_statistics_feature_selection(prepared, config)
        if _can_upgrade_prepared_contract(prepared, expected):
            _upgrade_prepared_contract(prepared, expected)
            with prepared_path.open("wb") as f:
                pickle.dump(prepared, f)
            _write_prepare_outputs(prepared, processed_dir)
            return apply_cell_train_statistics_feature_selection(prepared, config)
    prepared = prepare_data(
        paths,
        processed_dir,
        benchmark=config,
        seed=args.seed,
        n_components=args.n_components or config.n_components,
        max_rows=args.max_rows if args.max_rows is not None else config.max_rows,
        gdsc_source_mode=gdsc_source_mode,
    )
    return apply_cell_train_statistics_feature_selection(prepared, config)


def _can_upgrade_prepared_contract(prepared, expected: dict[str, object]) -> bool:
    manifest = getattr(prepared, "manifest", {}) or {}
    benchmark = manifest.get("benchmark") or {}
    expected_benchmark = expected.get("benchmark") or {}
    actual_kg_prior_sources = _normalize_kg_prior_sources(manifest.get("kg_prior_sources"))
    expected_kg_prior_sources = _normalize_kg_prior_sources(expected.get("kg_prior_sources"))
    required_manifest_equalities = [
        ("seed", manifest.get("seed"), expected.get("seed")),
        ("n_components_requested", manifest.get("n_components_requested"), expected.get("n_components")),
        ("max_rows", manifest.get("max_rows"), expected.get("max_rows")),
        ("gdsc_source_mode", manifest.get("gdsc_source_mode"), expected.get("gdsc_source_mode")),
        ("gdsc_fitted_files", manifest.get("gdsc_fitted_files"), expected.get("gdsc_fitted_files")),
        ("files_signature", manifest.get("files_signature"), expected.get("files_signature")),
        ("gdsc_smiles_cache", manifest.get("gdsc_smiles_cache"), expected.get("gdsc_smiles_cache")),
        ("prior_sources", manifest.get("prior_sources"), expected.get("prior_sources")),
        ("kg_prior_sources", actual_kg_prior_sources, expected_kg_prior_sources),
    ]
    for _, actual, wanted in required_manifest_equalities:
        if actual != wanted:
            return False
    if _prepare_benchmark_identity(benchmark) != _prepare_benchmark_identity(expected_benchmark):
        return False
    responses = getattr(prepared, "responses", pd.DataFrame())
    required_response_cols = {
        "relative_value",
        "relative_value_train",
        "relative_value_eval",
        "canonical_group_key",
        "canonical_group_index",
        "canonical_group_size",
        "canonical_drug_id",
    }
    if not required_response_cols.issubset(set(responses.columns)):
        return False
    artifacts = getattr(prepared, "artifacts", None)
    if artifacts is None:
        return False
    if getattr(artifacts, "kg_graph", None) is None:
        return False
    canonical_drug_table = getattr(artifacts, "canonical_drug_table", None)
    if canonical_drug_table is None or canonical_drug_table.empty:
        return False
    expected_drug_ids = canonical_drug_table["DRUG_ID"].astype(int).tolist()
    actual_drug_ids = [int(x) for x in getattr(artifacts.kg_graph, "drug_ids", [])]
    if expected_drug_ids != actual_drug_ids:
        return False
    return True


def _prepare_benchmark_identity(benchmark: dict[str, object] | None) -> dict[str, object]:
    benchmark = dict(benchmark or {})
    ignored_keys = {
        "evaluation_profiles",
        "eval_batch_size",
        "export_planning_predictions",
        "export_terminal_latents",
        "export_gate_audit",
        "cell_train_statistics_features",
        "family_order",
        "ablation_variants",
        "model",
        "world_model",
        "training_objectives",
        "batch",
        "evaluation",
        "controls",
        "prior",
        "static_prior",
        "notes",
    }
    for key in ignored_keys:
        benchmark.pop(key, None)
    return benchmark


def _normalize_kg_prior_sources(sources: object) -> list[str] | None:
    if sources is None:
        return None
    aliases = {
        "chembl_moa": "chembl",
        "drkg_filtered": "drkg",
        "primekg_filtered": "primekg",
    }
    normalized: list[str] = []
    for item in sources:
        normalized.append(aliases.get(str(item), str(item)))
    return normalized


def _upgrade_prepared_contract(prepared, expected: dict[str, object]) -> None:
    manifest = getattr(prepared, "manifest", {}) or {}
    manifest["benchmark"] = expected["benchmark"]
    manifest["seed"] = expected["seed"]
    manifest["n_components_requested"] = expected["n_components"]
    manifest["max_rows"] = expected["max_rows"]
    manifest["gdsc_source_mode"] = expected["gdsc_source_mode"]
    manifest["gdsc_fitted_files"] = expected["gdsc_fitted_files"]
    manifest["state_projection_policy_version"] = expected["state_projection_policy_version"]
    manifest["relative_target_schema_version"] = expected["relative_target_schema_version"]
    manifest["cell_residual_target_schema_version"] = expected["cell_residual_target_schema_version"]
    manifest["drug_canonicalization_version"] = expected["drug_canonicalization_version"]
    manifest["files_signature"] = expected["files_signature"]
    manifest["gdsc_smiles_cache"] = expected["gdsc_smiles_cache"]
    manifest["blocked_prior_terms"] = expected["blocked_prior_terms"]
    manifest["resolved_prior_policy"] = expected["resolved_prior_policy"]
    manifest["prior_sources"] = expected["prior_sources"]
    manifest["kg_prior_sources"] = expected["kg_prior_sources"]
    manifest["kg_prior_schema_version"] = expected["kg_prior_schema_version"]
    manifest["kg_drug_index_space_version"] = expected["kg_drug_index_space_version"]
    manifest["smiles_attachment_schema_version"] = expected["smiles_attachment_schema_version"]
    manifest["cell_train_statistics_version"] = expected["cell_train_statistics_version"]
    manifest["canonical_drug_indexing"] = bool(expected["benchmark"].get("canonical_drug_indexing", False))
    manifest["prepare_cache_key"] = expected["prepare_cache_key"]
    manifest["prepare_contract"] = expected
    prepared.manifest = manifest


def _copy_prepare_contract(processed_dir: Path, result_dir: Path) -> None:
    for name in [
        "manifest.json",
        "leakage_audit.json",
        "cache_hits.csv",
        "cache_manifest.json",
        "split_audit.csv",
        "missing_smiles_audit.csv",
        "invalid_smiles_audit.csv",
        "prior_mapping_audit.csv",
        "benchmark_contract.json",
        "gdsc_pairs.csv",
        "train_auc_reference.csv",
        "drug_mapping_master.csv",
        "chembl_moa_edges.csv",
        "drkg_filtered_edges.csv",
        "primekg_filtered_edges.csv",
        "kg_node_index.csv",
        "kg_edge_audit.csv",
        "kg_coverage_by_drug.csv",
    ]:
        src = processed_dir / name
        if src.exists():
            shutil.copy2(src, result_dir / name)


def _evaluate_profiles(
    config,
    model,
    prepared,
    result_dir: Path,
    paths: Paths,
    device: str,
    top_k: int,
    benchmark,
    context: TrainExecutionContext | None,
    runtime_profile: str,
    eval_compile: bool,
    eval_batch_size: int,
    export_planning_predictions: bool,
    export_terminal_latents: bool,
    export_gate_audit: bool,
) -> None:
    for profile in config.evaluation_profiles:
        if profile == "gdsc":
            evaluate_gdsc(
                model,
                prepared,
                result_dir,
                top_k=top_k,
                batch_size=eval_batch_size,
                device=device,
                benchmark=benchmark,
                controls=config.controls,
                context=context,
                runtime_profile=runtime_profile,
                eval_compile=bool(eval_compile),
                export_planning_predictions=export_planning_predictions,
                export_terminal_latents=export_terminal_latents,
                export_gate_audit=export_gate_audit,
            )
        elif profile == "ctrdb":
            evaluate_ctrdb_response(model, prepared.artifacts, paths, result_dir, device=device, runtime_profile=runtime_profile)
        elif profile == "tcga":
            evaluate_tcga_actual_treatments(
                model,
                prepared.artifacts,
                paths,
                result_dir,
                top_k=top_k,
                device=device,
                runtime_profile=runtime_profile,
            )
        elif profile in {"prism", "prism_secondary"}:
            pd.DataFrame(
                [{"analysis": "prism_mapping", "status": "not_implemented_in_public_release"}]
            ).to_csv(result_dir / "prism_mapping_audit.csv", index=False)
        else:
            raise ValueError(f"Unsupported evaluation profile: {profile}")


def _materialize_contracts(config, processed_dir: Path, result_dir: Path) -> None:
    metric_frames = []
    existing_metrics = result_dir / "metrics.csv"
    if existing_metrics.exists():
        frame = _safe_read_csv(existing_metrics)
        if not frame.empty:
            metric_frames.append(frame)
    for name, source in [
        ("gdsc_metrics.csv", "gdsc"),
        ("ctrdb_response_metrics.csv", "ctrdb"),
        ("tcga_os_survival_metrics.csv", "tcga"),
        ("tcga_response_auc_validation_metrics.csv", "tcga"),
    ]:
        path = result_dir / name
        if path.exists():
            frame = _safe_read_csv(path)
            if "metric_source" not in frame.columns:
                frame.insert(0, "metric_source", source)
            metric_frames.append(frame)
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(result_dir / "metrics.csv", index=False)
    prediction_candidates = [
        result_dir / "predictions.csv",
        result_dir / "planning_predictions.csv",
        result_dir / "ctrdb_response_scores.csv",
        result_dir / "tcga_actual_treatment_scores.csv",
    ]
    for idx, candidate in enumerate(prediction_candidates):
        if candidate.exists():
            if idx == 0:
                break
            shutil.copy2(candidate, result_dir / "predictions.csv")
            break
    mapping_frames = []
    for name, source in [
        ("missing_smiles_audit.csv", "missing_smiles"),
        ("invalid_smiles_audit.csv", "invalid_smiles"),
        ("prior_mapping_audit.csv", "prior"),
        ("split_audit.csv", "split"),
        ("ctrdb_unmapped_drugs.csv", "ctrdb"),
        ("tcga_unmapped_actual_drugs.csv", "tcga"),
        ("prism_mapping_audit.csv", "prism"),
    ]:
        path = result_dir / name if (result_dir / name).exists() else processed_dir / name
        if path.exists():
            frame = _safe_read_csv(path)
            if "mapping_source" not in frame.columns:
                frame.insert(0, "mapping_source", source)
            mapping_frames.append(frame)
    if mapping_frames:
        pd.concat(mapping_frames, ignore_index=True).to_csv(result_dir / "mapping_audit.csv", index=False)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _resolve_source_run(config, bench_paths: dict[str, Path]) -> Path:
    if config.source_run_dir:
        source_run = Path(config.source_run_dir)
        _validate_source_run_for_tcga(source_run)
        return source_run
    if config.source_benchmark:
        source_root = bench_paths["benchmark_dir"].parent / config.source_benchmark / "results"
        latest = latest_result_dir(source_root)
        if latest is not None:
            _validate_source_run_for_tcga(latest)
            return latest
    raise SystemExit("TCGA actual-treatment benchmark requires an existing source run directory with model.pt and artifacts.pkl.")


def _validate_source_run_for_tcga(source_run: Path) -> None:
    manifest_path = Path(source_run) / "manifest.json"
    contract_path = Path(source_run) / "benchmark_contract.json"
    if not manifest_path.exists() or not contract_path.exists():
        raise SystemExit("TCGA source run must come from a validated full 01 run with manifest.json and benchmark_contract.json.")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with contract_path.open("r", encoding="utf-8") as f:
        contract = json.load(f)
    benchmark_id = contract.get("benchmark_id") or manifest.get("benchmark", {}).get("benchmark_id")
    if not str(benchmark_id).startswith("01_random_cell_split"):
        raise SystemExit(
            "TCGA source run must come from benchmark 01_random_cell_split "
            f"or its CTRP v1/v2 variant, got {benchmark_id!r}."
        )
    if manifest.get("max_rows") is not None:
        raise SystemExit("TCGA source run must come from a full 01 run; stale reduced max_rows manifest detected.")


def _run_tcga_external_only(args, config, bench_paths: dict[str, Path], paths: Paths) -> None:
    if args.cmd == "prepare":
        ensure_dir(bench_paths["processed"] / args.processed_name)
        return
    validate_device(args.device)
    if args.source_run_dir is not None:
        config = replace(config, source_run_dir=str(Path(args.source_run_dir)))
    source_run = _resolve_source_run(config, bench_paths)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = ensure_dir(bench_paths["results"] / run_name)
    shutil.copy2(source_run / "artifacts.pkl", result_dir / "artifacts.pkl")
    shutil.copy2(source_run / "model.pt", result_dir / "model.pt")
    artifacts = pd.read_pickle(result_dir / "artifacts.pkl")
    model = load_model(result_dir, artifacts, strict=False)
    evaluate_tcga_actual_treatments(
        model,
        artifacts,
        paths,
        result_dir,
        top_k=args.top_k or config.top_k,
        device=args.device,
        runtime_profile=args.runtime_profile,
        export_terminal_latents=bool(getattr(config, "export_terminal_latents", False)),
        terminal_latent_path=result_dir / "terminal_latents.npy",
    )
    pd.DataFrame([{"metric_source": "tcga", "analysis": "source_run_dir", "status": str(source_run)}]).to_csv(
        result_dir / "metrics.csv",
        index=False,
    )
    _materialize_contracts(config, bench_paths["processed"] / args.processed_name, result_dir)
    generate_run_visualizations(result_dir, device=args.device)


def _variant_config_and_controls(config, name: str, seed: int):
    controls = AblationControls(seed=seed)
    world_model_batch = replace(config.batch, sampler="grouped_world_model")
    regression_batch = replace(config.batch, sampler="random_pair")
    regression_model = replace(config.model, mode="regression")
    world_model = replace(config.model, mode="world_model")
    chembl_off = replace(config.prior.chembl, enabled=False, weight=0.0)
    drkg_off = replace(config.prior.drkg, enabled=False, weight=0.0)
    primekg_off = replace(config.prior.primekg, enabled=False, weight=0.0)
    chembl_only = replace(
        config.prior,
        chembl=replace(config.prior.chembl, enabled=True, weight=1.0, edge_types=("action_type_target", "has_action_type", "acts_on_target")),
        drkg=drkg_off,
        primekg=primekg_off,
        include_side_effects=False,
    )
    drkg_only = replace(
        config.prior,
        chembl=chembl_off,
        drkg=replace(config.prior.drkg, enabled=True, weight=1.0, edge_types=("drug_protein", "protein_protein")),
        primekg=primekg_off,
        include_side_effects=False,
    )
    primekg_only = replace(
        config.prior,
        chembl=chembl_off,
        drkg=drkg_off,
        primekg=replace(config.prior.primekg, enabled=True, weight=1.0, edge_types=("drug_protein", "protein_protein")),
        include_side_effects=False,
    )
    regression_objectives = replace(
        config.training_objectives,
        loss_value_weight=0.0,
        loss_within_drug_rank_weight=0.0,
        loss_policy_advantage_weight=0.0,
    )
    value_only_objectives = replace(
        config.training_objectives,
        loss_within_drug_rank_weight=0.0,
        loss_policy_advantage_weight=0.0,
    )
    if name in {"regression_raw"}:
        return replace(config, model=regression_model, training_objectives=regression_objectives, batch=regression_batch), controls
    if name in {"world_model_value_only"}:
        return replace(config, model=world_model, training_objectives=value_only_objectives, batch=world_model_batch), controls
    if name in {"world_model_full", "full"}:
        return replace(config, model=world_model, batch=world_model_batch), controls
    if name in {"all_sources"}:
        return replace(config, model=world_model, batch=world_model_batch), controls
    if name in {"no_chembl"}:
        return replace(config, model=world_model, batch=world_model_batch, prior=replace(config.prior, chembl=chembl_off)), controls
    if name in {"no_drkg"}:
        return replace(config, model=world_model, batch=world_model_batch, prior=replace(config.prior, drkg=drkg_off)), controls
    if name in {"no_primekg"}:
        return replace(config, model=world_model, batch=world_model_batch, prior=replace(config.prior, primekg=primekg_off)), controls
    if name in {"chembl_only"}:
        return replace(config, model=world_model, batch=world_model_batch, prior=chembl_only), controls
    if name in {"drkg_only"}:
        return replace(config, model=world_model, batch=world_model_batch, prior=drkg_only), controls
    if name in {"primekg_only"}:
        return replace(config, model=world_model, batch=world_model_batch, prior=primekg_only), controls
    if name in {"edge_type_filtered_chembl_action_only"}:
        return replace(
            config,
            model=world_model,
            batch=world_model_batch,
            prior=replace(
                chembl_only,
                chembl=replace(config.prior.chembl, enabled=True, weight=1.0, edge_types=("action_type_target",)),
            ),
        ), controls
    if name in {"edge_type_filtered_drkg_drug_protein_only"}:
        return replace(
            config,
            model=world_model,
            batch=world_model_batch,
            prior=replace(
                drkg_only,
                drkg=replace(config.prior.drkg, enabled=True, weight=1.0, edge_types=("drug_protein",)),
            ),
        ), controls
    if name in {"edge_type_filtered_primekg_protein_only"}:
        return replace(
            config,
            model=world_model,
            batch=world_model_batch,
            prior=replace(
                primekg_only,
                primekg=replace(config.prior.primekg, enabled=True, weight=1.0, edge_types=("drug_protein",)),
            ),
        ), controls
    if name in {"edge_type_filtered_primekg_side_effects"}:
        return replace(
            config,
            model=world_model,
            batch=world_model_batch,
            prior=replace(
                config.prior,
                chembl=chembl_off,
                drkg=drkg_off,
                primekg=replace(config.prior.primekg, enabled=True, weight=1.0, edge_types=("drug_protein", "drug_side_effect")),
                include_side_effects=True,
            ),
        ), controls
    if name in {"world_model_no_terminal", "no_terminal"}:
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, use_terminal=False)
    if name in {"world_model_no_prior", "no_prior"}:
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, use_prior=False, prior_mode="zero")
    if name in {"world_model_shuffled_prior", "shuffled_prior"}:
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="shuffled")
    if name in {"smiles_only", "world_model_no_prior"}:
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, use_prior=False, prior_mode="zero")
    if name in {"legacy_static_prior", "legacy_static"}:
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="legacy_static_prior")
    if name == "concat_only":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="concat_only")
    if name == "chembl_gat_only":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="chembl_only")
    if name == "drkg_gat_only":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="drkg_only")
    if name == "primekg_gat_only":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="primekg_only")
    if name == "multikg_gat":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="learned")
    if name == "multikg_no_state_attention":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="no_state_attention")
    if name == "multikg_no_ranking_loss":
        no_rank = replace(config.training_objectives, loss_within_drug_rank_weight=0.0)
        return replace(config, model=world_model, batch=world_model_batch, training_objectives=no_rank), controls
    if name == "multikg_no_graph_consistency":
        no_cons = replace(config.training_objectives, loss_graph_consistency_weight=0.0)
        return replace(config, model=world_model, batch=world_model_batch, training_objectives=no_cons), controls
    if name in {"shuffled_drug_node_mapping", "degree_matched_edge_rewiring"}:
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="shuffled_mapping")
    if name in {"random_graph_same_degree", "random_graph"}:
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="random_graph")
    if name == "chem_only":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, use_prior=False)
    if name == "random_prior":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, prior_mode="random")
    if name == "drug_only":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, use_state=False)
    if name == "no_action":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, use_drug=False, use_prior=False, prior_mode="zero")
    if name == "random_policy":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, policy_mode="random")
    if name == "drug_only_policy":
        return replace(config, model=world_model, batch=world_model_batch), replace(controls, policy_mode="drug_only")
    raise ValueError(f"Unsupported ablation variant: {name}")


def _variant_summary_row(
    *,
    variant: str,
    variant_config,
    controls: AblationControls,
    variant_dir: Path,
) -> pd.DataFrame:
    metrics = pd.read_csv(variant_dir / "gdsc_metrics.csv")
    runtime = _safe_read_json(variant_dir / "gpu_runtime.json")
    requested_epochs = int(getattr(variant_config, "epochs", 0))
    train_seconds_total = runtime.get("train_seconds_total")
    per_epoch = None
    if train_seconds_total is not None and requested_epochs > 0:
        per_epoch = float(train_seconds_total) / float(requested_epochs)
    enriched = metrics.copy()
    enriched.insert(0, "variant", variant)
    enriched["benchmark_id"] = variant_config.benchmark_id
    enriched["split_type"] = variant_config.split_type
    enriched["model_mode"] = variant_config.model.mode
    enriched["requested_batch_sampler"] = variant_config.batch.sampler
    enriched["effective_batch_sampler"] = runtime.get("effective_batch_sampler", runtime.get("requested_batch_sampler", variant_config.batch.sampler))
    enriched["pairwise_impl"] = runtime.get("pairwise_impl", "")
    enriched["prior_mode"] = controls.prior_mode
    enriched["use_prior"] = bool(controls.use_prior)
    enriched["runtime_profile"] = runtime.get("runtime_profile", "")
    enriched["train_seconds_total"] = train_seconds_total
    enriched["train_seconds_per_epoch"] = per_epoch
    enriched["drug_level_target"] = float(CANDIDATE_DRUG_LEVEL_TARGET)
    enriched["drug_level_target_pass"] = enriched["within_drug_pcc_mean"].astype(float) >= float(CANDIDATE_DRUG_LEVEL_TARGET)
    return enriched


def _apply_candidate_gates(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    summary = summary.copy()
    if "drug_level_target" not in summary.columns:
        summary["drug_level_target"] = float(CANDIDATE_DRUG_LEVEL_TARGET)
    if "drug_level_target_pass" not in summary.columns:
        summary["drug_level_target_pass"] = summary["within_drug_pcc_mean"].astype(float) >= float(CANDIDATE_DRUG_LEVEL_TARGET)
    baseline_variant = "multikg_gat" if "multikg_gat" in set(summary["variant"].astype(str)) else str(summary.iloc[0]["variant"])
    baseline = (
        summary.loc[summary["variant"].astype(str).eq(baseline_variant), ["split", "within_cell_pcc_mean"]]
        .rename(columns={"within_cell_pcc_mean": "baseline_within_cell_pcc_mean"})
        .copy()
    )
    summary = summary.merge(baseline, on="split", how="left")
    summary["sample_guard_variant"] = baseline_variant
    summary["sample_guard_delta_threshold"] = float(CANDIDATE_SAMPLE_GUARD_DELTA)
    summary["within_cell_pcc_delta_vs_baseline"] = (
        summary["within_cell_pcc_mean"].astype(float) - summary["baseline_within_cell_pcc_mean"].astype(float)
    )
    summary["sample_guard_pass"] = summary["within_cell_pcc_delta_vs_baseline"] >= float(CANDIDATE_SAMPLE_GUARD_DELTA)
    summary["promotion_pass"] = summary["drug_level_target_pass"] & summary["sample_guard_pass"]
    return summary


def _safe_read_json(path: Path) -> dict[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _run_ablations(
    args,
    config,
    bench_paths,
    paths: Paths,
    processed_dir: Path,
    result_dir: Path,
    config_path: Path,
) -> None:
    prepared = _load_or_prepare(processed_dir, paths, config, args)
    context = TrainExecutionContext()
    summary_rows = []
    variant_root = ensure_dir(result_dir / "variants")
    runtime_profile = getattr(args, "runtime_profile", RUNTIME_PROFILE_STRICT)
    eval_options = _resolve_eval_options(args, config, args.device)
    train_options = _resolve_train_options(args, config)
    _log_batch_resolution(
        config=config,
        effective_epochs=int(train_options["epochs"]),
        effective_batch_size=int(train_options["batch_size"]),
        effective_eval_batch_size=int(eval_options["eval_batch_size"]),
    )
    for variant in config.ablation_variants:
        variant_config, controls = _variant_config_and_controls(config, variant, args.seed)
        variant_dir = ensure_dir(variant_root / variant)
        copy_config_snapshot(config_path, [variant_dir])
        model = train_model(
            prepared,
            variant_dir,
            epochs=int(args.epochs if args.epochs is not None else variant_config.epochs),
            batch_size=int(args.batch_size if args.batch_size is not None else variant_config.batch_size),
            lr=float(args.lr if args.lr is not None else variant_config.lr),
            seed=args.seed,
            device=args.device,
            benchmark=variant_config,
            controls=controls,
            context=context,
            runtime_profile=runtime_profile,
        )
        _copy_prepare_contract(processed_dir, variant_dir)
        _write_prepare_outputs(prepared, variant_dir)
        pred = evaluate_gdsc(
            model,
            prepared,
            variant_dir,
            top_k=args.top_k or variant_config.top_k,
            batch_size=eval_options["eval_batch_size"],
            device=args.device,
            benchmark=variant_config,
            controls=controls,
            context=context,
            runtime_profile=runtime_profile,
            eval_compile=bool(getattr(args, "eval_compile", False)),
            export_planning_predictions=eval_options["export_planning_predictions"],
            export_terminal_latents=eval_options["export_terminal_latents"],
            export_gate_audit=eval_options["export_gate_audit"],
        )
        _materialize_contracts(config, processed_dir, variant_dir)
        generate_run_visualizations(variant_dir, device=args.device)
        summary_rows.append(
            _variant_summary_row(
                variant=variant,
                variant_config=variant_config,
                controls=controls,
                variant_dir=variant_dir,
            )
        )
        if pred.empty:
            continue
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    summary = _apply_candidate_gates(summary)
    summary.to_csv(result_dir / "ablation_summary.csv", index=False)
    md_lines = [
        f"# {config.benchmark_name}",
        "",
        f"Ablation variants: {', '.join(config.ablation_variants)}",
        "",
        f"Drug-level promotion target: `within_drug_pcc_mean >= {CANDIDATE_DRUG_LEVEL_TARGET:.2f}`.",
        f"Sample guard: `within_cell_pcc_mean` must not drop by more than {abs(CANDIDATE_SAMPLE_GUARD_DELTA):.2f} vs suite baseline.",
        "",
        "Results are aggregated from `variants/<name>/gdsc_metrics.csv` plus runtime metadata.",
    ]
    (result_dir / "final_benchmark_report.md").write_text("\n".join(md_lines))
    html = "<html><body><h1>{}</h1><p>{}</p></body></html>".format(
        config.benchmark_name,
        "Results are stored in variants/ and summarized in ablation_summary.csv.",
    )
    (result_dir / "final_benchmark_report.html").write_text(html)
    generate_run_visualizations(result_dir, device=args.device)


if __name__ == "__main__":
    main()
