from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .config import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, Paths
from .external import evaluate_ctrdb_response, evaluate_tcga_actual_treatments
from .gdsc_smiles import build_all_dataset_smiles_caches, build_dataset_smiles_cache
from .repro import set_reproducible_runtime
from .runtime import release_gpu_memory
from .train import evaluate_gdsc, load_model, load_prepared, prepare_data, train_model
from .utils import ensure_dir


def _resolve_eval_batch_size(device: str, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    return 4096 if device == "cpu" else 16384


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Terminal consequence drug world model.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-rows", type=int, default=None, help="Optional bounded response rows for sanity runs.")
    p.add_argument("--device", default=None)
    p.add_argument("--runtime-profile", default=None)
    p.add_argument("--eval-compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    build_smiles_all = sub.add_parser("build-dataset-smiles-cache")
    build_smiles_all.add_argument("--dataset", choices=["all", "gdsc", "beataml2", "ctrdb", "tcga"], default="all")
    build_smiles_all.add_argument("--overwrite", action="store_true")
    build_smiles = sub.add_parser("build-gdsc-smiles-cache")
    build_smiles.add_argument("--out-path", type=Path, default=None)
    build_smiles.add_argument("--overwrite", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("--n-components", type=int, default=512)
    train = sub.add_parser("train")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--n-components", type=int, default=512)
    evalp = sub.add_parser("evaluate")
    evalp.add_argument("--top-k", type=int, default=5)
    evalp.add_argument("--eval-batch-size", type=int, default=None)
    evalp.add_argument("--export-planning-predictions", action=argparse.BooleanOptionalAction, default=False)
    evalp.add_argument("--export-terminal-latents", action=argparse.BooleanOptionalAction, default=False)
    evalp.add_argument("--export-gate-audit", action=argparse.BooleanOptionalAction, default=False)
    run = sub.add_parser("run")
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--lr", type=float, default=1e-3)
    run.add_argument("--n-components", type=int, default=512)
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--eval-batch-size", type=int, default=None)
    run.add_argument("--export-planning-predictions", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--export-terminal-latents", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--export-gate-audit", action=argparse.BooleanOptionalAction, default=False)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_reproducible_runtime(seed=args.seed, device=args.device or "cpu", profile=args.runtime_profile)
    paths = Paths()
    out_dir = ensure_dir(args.out_dir)
    use_cache = not args.no_cache
    model = None
    prepared = None
    artifacts = None
    try:
        if args.cmd == "build-dataset-smiles-cache":
            if args.dataset == "all":
                build_all_dataset_smiles_caches(paths, overwrite=args.overwrite)
            else:
                build_dataset_smiles_cache(paths, dataset=args.dataset, overwrite=args.overwrite)
            return
        if args.cmd == "build-gdsc-smiles-cache":
            build_dataset_smiles_cache(paths, dataset="gdsc", out_path=args.out_path, overwrite=args.overwrite)
            return
        if args.cmd == "prepare":
            prepare_data(
                paths,
                out_dir,
                seed=args.seed,
                n_components=args.n_components,
                max_rows=args.max_rows,
                cache_dir=args.cache_dir,
                use_cache=use_cache,
                rebuild_cache=args.rebuild_cache,
            )
            return
        if args.cmd == "train":
            _require_device(args)
            prepared_path = out_dir / "prepared.pkl"
            prepared = load_prepared(prepared_path) if prepared_path.exists() else prepare_data(
                paths,
                out_dir,
                seed=args.seed,
                n_components=args.n_components,
                max_rows=args.max_rows,
                cache_dir=args.cache_dir,
                use_cache=use_cache,
                rebuild_cache=args.rebuild_cache,
            )
            train_model(
                prepared,
                out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                runtime_profile=args.runtime_profile,
            )
            return
        if args.cmd == "evaluate":
            _require_device(args)
            prepared = load_prepared(out_dir / "prepared.pkl")
            with (out_dir / "artifacts.pkl").open("rb") as f:
                artifacts = pickle.load(f)
            model = load_model(out_dir, artifacts)
            evaluate_gdsc(
                model,
                prepared,
                out_dir,
                top_k=args.top_k,
                batch_size=_resolve_eval_batch_size(args.device, args.eval_batch_size),
                device=args.device,
                runtime_profile=args.runtime_profile,
                eval_compile=bool(args.eval_compile),
                export_planning_predictions=bool(args.export_planning_predictions),
                export_terminal_latents=bool(args.export_terminal_latents),
                export_gate_audit=bool(args.export_gate_audit),
            )
            evaluate_tcga_actual_treatments(
                model,
                artifacts,
                paths,
                out_dir,
                top_k=args.top_k,
                device=args.device,
                cache_dir=args.cache_dir,
                use_cache=use_cache,
                rebuild_cache=args.rebuild_cache,
                runtime_profile=args.runtime_profile,
            )
            evaluate_ctrdb_response(
                model,
                artifacts,
                paths,
                out_dir,
                device=args.device,
                cache_dir=args.cache_dir,
                use_cache=use_cache,
                rebuild_cache=args.rebuild_cache,
                runtime_profile=args.runtime_profile,
            )
            return
        if args.cmd == "run":
            _require_device(args)
            prepared = prepare_data(
                paths,
                out_dir,
                seed=args.seed,
                n_components=args.n_components,
                max_rows=args.max_rows,
                cache_dir=args.cache_dir,
                use_cache=use_cache,
                rebuild_cache=args.rebuild_cache,
            )
            model = train_model(
                prepared,
                out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                runtime_profile=args.runtime_profile,
            )
            evaluate_gdsc(
                model,
                prepared,
                out_dir,
                top_k=args.top_k,
                batch_size=_resolve_eval_batch_size(args.device, args.eval_batch_size),
                device=args.device,
                runtime_profile=args.runtime_profile,
                eval_compile=bool(args.eval_compile),
                export_planning_predictions=bool(args.export_planning_predictions),
                export_terminal_latents=bool(args.export_terminal_latents),
                export_gate_audit=bool(args.export_gate_audit),
            )
            evaluate_tcga_actual_treatments(
                model,
                prepared.artifacts,
                paths,
                out_dir,
                top_k=args.top_k,
                device=args.device,
                cache_dir=args.cache_dir,
                use_cache=use_cache,
                rebuild_cache=args.rebuild_cache,
                runtime_profile=args.runtime_profile,
            )
            evaluate_ctrdb_response(
                model,
                prepared.artifacts,
                paths,
                out_dir,
                device=args.device,
                cache_dir=args.cache_dir,
                use_cache=use_cache,
                rebuild_cache=args.rebuild_cache,
                runtime_profile=args.runtime_profile,
            )
            return
    finally:
        release_gpu_memory(model, prepared, artifacts)


def _require_device(args: argparse.Namespace) -> None:
    if not args.device:
        raise SystemExit("Explicit --device is required for train/evaluate/run. Use --device cuda:N or --device cpu.")
    if args.device != "cpu" and not args.device.startswith("cuda:"):
        raise SystemExit("Invalid --device. Use --device cuda:N or --device cpu.")


if __name__ == "__main__":
    main()
