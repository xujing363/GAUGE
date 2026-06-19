from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss

from .benchmarking import BenchmarkConfig, split_entities
from .cache import CacheManager, cache_key, files_signature
from .config import Paths
from .data import (
    expression_from_h5ad,
    h5ad_gene_list,
    load_multisource_drug_prior,
    select_hvg_from_h5ad,
    tcga_binary_episode_frame,
    tcga_binary_label_summary,
    attach_smiles,
)
from .features import FeatureArtifacts, build_drug_table, fit_state_projection, project_expression
from .kg_prior import build_multikg_graph_artifacts, write_kg_reports
from .metrics import binary_response_metrics, calibration_by_quantile
from .model import TerminalWorldModel, architecture_dict
from .repro import RUNTIME_PROFILE_STABLE, set_reproducible_runtime
from .train import load_model
from .visualization import generate_run_visualizations
from .utils import copy_config_snapshot, ensure_dir, write_json


@dataclass
class TcgaBinaryPrepared:
    episodes: pd.DataFrame
    state_matrix: pd.DataFrame
    artifacts: FeatureArtifacts
    label_audit: pd.DataFrame
    split_audit: pd.DataFrame
    mapping_audit: pd.DataFrame
    label_summary: pd.DataFrame
    manifest: dict[str, Any]


@dataclass
class BinaryEpisodeTensors:
    state_idx: torch.Tensor
    drug_idx: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return int(self.y.shape[0])


@dataclass
class BinaryTensorBanks:
    state_bank: torch.Tensor
    drug_bank: torch.Tensor
    prior_bank: torch.Tensor
    mask_bank: torch.Tensor
    patient_to_idx: dict[str, int]
    drug_to_idx: dict[int, int]


def _cache_contract_key(benchmark: BenchmarkConfig, paths: Paths, n_hvg: int, use_hvg: bool) -> str:
    payload = {
        "kind": "tcga_binary_prepared",
        "benchmark": benchmark.signature(),
        "gene_mode": "hvg" if use_hvg else "all_genes",
        "n_hvg": int(n_hvg) if use_hvg else None,
        "files_signature": files_signature(
            [
                paths.tcga_h5ad,
                paths.tcga_smiles_cache,
                paths.primekg,
                paths.gdsc_screened_compounds,
            ]
        ),
    }
    return cache_key(payload)


def prepare_tcga_binary_data(
    paths: Paths,
    out_dir: Path,
    benchmark: BenchmarkConfig,
    seed: int,
    n_components: int,
    n_hvg: int,
    use_hvg: bool = True,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> TcgaBinaryPrepared:
    ensure_dir(out_dir)
    cache = CacheManager(cache_dir or out_dir / ".cache", use_cache=use_cache, rebuild_cache=rebuild_cache)
    key = _cache_contract_key(benchmark, paths, n_hvg, use_hvg)
    cached = cache.load_pickle("prepare", key, "prepared.pkl")
    if cached is not None:
        cached = _ensure_prepared_compat(cached)
        _write_prepare_outputs(cached, out_dir, benchmark, seed, n_components, n_hvg, key, use_hvg, from_cache=True)
        cache.write_reports(out_dir)
        return cached

    import anndata as ad

    data = ad.read_h5ad(paths.tcga_h5ad, backed="r")
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    episodes, label_audit = tcga_binary_episode_frame(obs)
    if episodes.empty:
        raise ValueError("TCGA binary benchmark found no labeled single-agent therapy episodes.")

    episodes["DRUG_ID"] = pd.factorize(episodes["drug_key"], sort=True)[0].astype(int)
    episodes["sample_id"] = episodes["sample_id"].astype(str)
    episodes["patient_id"] = episodes["patient_id"].astype(str)
    episodes["project_id"] = episodes["project_id"].astype(str)

    drug_universe = episodes[["DRUG_ID", "drug_name"]].drop_duplicates("DRUG_ID").copy()
    drug_universe.rename(columns={"drug_name": "DRUG_NAME"}, inplace=True)
    drug_with_smiles, missing_smiles = attach_smiles(drug_universe, paths.tcga_smiles_cache)
    drug_keys = sorted(drug_with_smiles["drug_key"].astype(str).unique().tolist())
    prior_matrix = pd.DataFrame(index=drug_keys, dtype=np.float32)
    prior_audit = pd.DataFrame(columns=["drug_key", "drug_name", "issue"])
    prior_source_stats = {"static_prior": {"n_drugs": int(len(drug_keys)), "n_features": 0, "status": "disabled_for_multikg_graph_prior"}}
    drug_table, invalid_smiles = build_drug_table(drug_with_smiles, prior_matrix)
    kg_graph = build_multikg_graph_artifacts(
        paths,
        drug_table,
        cache_dir=cache_dir or out_dir / ".cache",
        use_cache=use_cache,
        rebuild_cache=rebuild_cache,
        prior_policy=benchmark.prior,
    )
    valid_drug_ids = set(drug_table["DRUG_ID"].astype(int).tolist())
    episodes = episodes.loc[episodes["DRUG_ID"].isin(valid_drug_ids)].copy()
    if episodes.empty:
        raise ValueError("TCGA binary benchmark lost all episodes after SMILES filtering.")

    if benchmark.split_type == "tcga_patient":
        split_map = split_entities(
            sorted(episodes["patient_id"].astype(str).unique().tolist()),
            seed=benchmark.split_seed,
            train_fraction=benchmark.train_fraction,
            val_fraction=benchmark.val_fraction,
            test_fraction=benchmark.test_fraction,
        )
        episodes["split"] = episodes["patient_id"].map(split_map)
    elif benchmark.split_type == "tcga_drug":
        split_map = split_entities(
            sorted(episodes["DRUG_ID"].astype(int).unique().tolist()),
            seed=benchmark.split_seed,
            train_fraction=benchmark.train_fraction,
            val_fraction=benchmark.val_fraction,
            test_fraction=benchmark.test_fraction,
        )
        episodes["split"] = episodes["DRUG_ID"].map(split_map)
    else:
        raise ValueError(f"Unsupported TCGA binary split_type: {benchmark.split_type}")
    episodes = episodes.loc[episodes["split"].isin(["train", "val", "test"])].copy()
    if episodes["split"].nunique() < 3:
        raise ValueError(f"Split {benchmark.split_type} did not produce train/val/test rows after filtering.")

    split_audit = _split_audit(episodes, benchmark.split_type)
    label_summary = tcga_binary_label_summary(episodes)

    patient_to_sample = (
        episodes[["patient_id", "sample_id"]]
        .drop_duplicates("patient_id")
        .set_index("patient_id")["sample_id"]
        .to_dict()
    )
    train_patients = sorted(episodes.loc[episodes["split"].eq("train"), "patient_id"].astype(str).unique().tolist())
    train_samples = [patient_to_sample[p] for p in train_patients if p in patient_to_sample]
    if use_hvg:
        genes = select_hvg_from_h5ad(paths.tcga_h5ad, train_samples, n_hvg=n_hvg, var_gene_name_col="gene_name")
        gene_mode = "hvg"
    else:
        genes = h5ad_gene_list(paths.tcga_h5ad, var_gene_name_col="gene_name")
        gene_mode = "all_genes"
    expr, obs_loaded = expression_from_h5ad(paths.tcga_h5ad, genes, var_gene_name_col="gene_name")
    obs_loaded = obs_loaded.loc[expr.index].copy()
    expr = expr.loc[obs_loaded.index]
    expr["patient_id"] = obs_loaded["case_submitter_id"].astype(str)
    patient_expr = expr.groupby("patient_id", observed=True)[genes].mean()
    train_patients_expr = [p for p in train_patients if p in patient_expr.index]
    if len(train_patients_expr) < 3:
        raise ValueError("Not enough TCGA training patients to fit the projection.")
    genes, imputer, scaler, pca = fit_state_projection(patient_expr, train_patients_expr, n_components=n_components)
    state = pd.DataFrame(
        project_expression(patient_expr, genes, imputer, scaler, pca),
        index=patient_expr.index.astype(str),
    )

    episodes = episodes.loc[episodes["patient_id"].isin(state.index) & episodes["DRUG_ID"].isin(drug_table["DRUG_ID"])].copy()
    if episodes.empty:
        raise ValueError("No TCGA binary episodes remain after aligning patient state and drug table.")
    episodes["episode_id"] = episodes["episode_id"].astype(str)
    episodes = episodes.sort_values(["split", "patient_id", "DRUG_ID", "episode_id"]).reset_index(drop=True)

    artifacts = FeatureArtifacts(
        genes=genes,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        split_by_cell=split_map if benchmark.split_type == "tcga_patient" else {},
        drug_baseline_auc={},
        drug_auc_train_values={},
        drug_table=drug_table,
        prior_columns=list(prior_matrix.columns),
        kg_graph=kg_graph,
    )
    manifest = {
        "benchmark": benchmark.prepare_signature(),
        "seed": int(seed),
        "gene_mode": gene_mode,
        "n_components_requested": int(n_components),
        "n_hvg_requested": int(n_hvg) if use_hvg else None,
        "n_genes_selected": int(len(genes)),
        "prepare_cache_key": key,
        "n_episodes": int(len(episodes)),
        "n_patients": int(episodes["patient_id"].nunique()),
        "n_drugs": int(episodes["DRUG_ID"].nunique()),
        "n_positive": int(episodes["y"].sum()),
        "n_negative": int(len(episodes) - int(episodes["y"].sum())),
        "task_type": benchmark.task_type,
        "split_type": benchmark.split_type,
        "label_policy": benchmark.label_policy,
        "instance_unit": benchmark.instance_unit,
        "tcga_h5ad": str(paths.tcga_h5ad),
        "tcga_smiles_cache": str(paths.tcga_smiles_cache),
        "primekg": str(paths.primekg),
        "resolved_prior_policy": {
            "kg_prior": benchmark.prior.to_dict(),
            "static_prior": benchmark.static_prior.to_dict(),
        },
        "prior_source_stats": prior_source_stats,
    }
    prepared = TcgaBinaryPrepared(
        episodes=episodes,
        state_matrix=state,
        artifacts=artifacts,
        label_audit=pd.concat([label_audit, missing_smiles, invalid_smiles, prior_audit], ignore_index=True, sort=False),
        split_audit=split_audit,
        mapping_audit=pd.concat([missing_smiles, invalid_smiles, prior_audit], ignore_index=True, sort=False),
        label_summary=label_summary,
        manifest=manifest,
    )
    cache.save_pickle("prepare", key, "prepared.pkl", prepared)
    _write_prepare_outputs(prepared, out_dir, benchmark, seed, n_components, n_hvg, key, use_hvg, from_cache=False)
    write_kg_reports(out_dir, kg_graph)
    cache.write_reports(out_dir)
    with (out_dir / "prepared.pkl").open("wb") as f:
        pickle.dump(prepared, f)
    return prepared


def load_prepared(path: Path) -> TcgaBinaryPrepared:
    with Path(path).open("rb") as f:
        return _ensure_prepared_compat(pickle.load(f))


def train_tcga_binary_model(
    prepared: TcgaBinaryPrepared,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 7,
    device: str | None = None,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
) -> TerminalWorldModel:
    ensure_dir(out_dir)
    if device is None:
        raise ValueError("Explicit --device is required for train/run. Use --device cuda:N or --device cpu.")
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("Invalid --device. Use --device cuda:N or --device cpu.")
    runtime = set_reproducible_runtime(seed=seed, device=device, profile=runtime_profile)
    model = TerminalWorldModel(
        state_dim=prepared.state_matrix.shape[1],
        prior_dim=len(prepared.artifacts.prior_columns),
        kg_artifacts=getattr(prepared.artifacts, "kg_graph", None),
        drug_fingerprint_bank=np.vstack([row.fingerprint.astype(np.float32) for row in prepared.artifacts.drug_table.itertuples(index=False)]),
    ).to(device)
    banks = _tensor_banks(prepared, device)
    kg_drug_idx_bank = model.local_drug_indices(
        [int(row.DRUG_ID) for row in prepared.artifacts.drug_table.itertuples(index=False)],
        device=device,
    )
    train_tensors = _frame_to_index_tensors(
        prepared.episodes.loc[prepared.episodes["split"].eq("train")],
        banks.patient_to_idx,
        banks.drug_to_idx,
        device,
    )
    val_tensors = _frame_to_index_tensors(
        prepared.episodes.loc[prepared.episodes["split"].eq("val")],
        banks.patient_to_idx,
        banks.drug_to_idx,
        device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    best = {"loss": float("inf"), "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        train_count = 0
        for batch_idx in _iter_batch_indices(len(train_tensors), batch_size, device, shuffle=True, seed=seed + epoch):
            state_row_idx = train_tensors.state_idx.index_select(0, batch_idx)
            drug_row_idx = train_tensors.drug_idx.index_select(0, batch_idx)
            target = train_tensors.y.index_select(0, batch_idx)
            out = model(
                banks.state_bank.index_select(0, state_row_idx),
                banks.drug_bank.index_select(0, drug_row_idx),
                banks.prior_bank.index_select(0, drug_row_idx),
                banks.mask_bank.index_select(0, drug_row_idx),
                use_terminal=True,
                drug_idx=kg_drug_idx_bank.index_select(0, drug_row_idx) if kg_drug_idx_bank is not None else None,
                compute_kg_consistency=False,
            )
            loss = criterion(out["binary_logit"], target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch_n = int(batch_idx.shape[0])
            train_loss_sum += loss.detach() * batch_n
            train_count += batch_n
        model.eval()
        val_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        val_count = 0
        with torch.no_grad():
            for batch_idx in _iter_batch_indices(len(val_tensors), batch_size, device, shuffle=False, seed=seed):
                state_row_idx = val_tensors.state_idx.index_select(0, batch_idx)
                drug_row_idx = val_tensors.drug_idx.index_select(0, batch_idx)
                target = val_tensors.y.index_select(0, batch_idx)
                out = model(
                    banks.state_bank.index_select(0, state_row_idx),
                    banks.drug_bank.index_select(0, drug_row_idx),
                    banks.prior_bank.index_select(0, drug_row_idx),
                    banks.mask_bank.index_select(0, drug_row_idx),
                    use_terminal=True,
                    drug_idx=kg_drug_idx_bank.index_select(0, drug_row_idx) if kg_drug_idx_bank is not None else None,
                    compute_kg_consistency=False,
                )
                loss = criterion(out["binary_logit"], target)
                batch_n = int(batch_idx.shape[0])
                val_loss_sum += loss * batch_n
                val_count += batch_n
        train_loss = float((train_loss_sum / max(train_count, 1)).item())
        val_loss = float((val_loss_sum / max(val_count, 1)).item()) if val_count else train_loss
        history.append({"epoch": epoch, "train_bce": train_loss, "val_bce": val_loss})
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    write_json(
        out_dir / "model_architecture.json",
        architecture_dict(model, prepared.state_matrix.shape[1], len(prepared.artifacts.prior_columns)),
    )
    with (out_dir / "artifacts.pkl").open("wb") as f:
        pickle.dump(prepared.artifacts, f)
    runtime["device"] = str(device)
    runtime["batch_size"] = int(batch_size)
    write_json(out_dir / "gpu_runtime.json", runtime)
    return model


def evaluate_tcga_binary(
    model: TerminalWorldModel,
    prepared: TcgaBinaryPrepared,
    out_dir: Path,
    device: str | None = None,
    batch_size: int | None = None,
    banks: BinaryTensorBanks | None = None,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
    export_terminal_latents: bool = False,
    terminal_latent_path: Path | None = None,
) -> pd.DataFrame:
    if device is None:
        raise ValueError("Explicit --device is required for evaluate/run. Use --device cuda:N or --device cpu.")
    pred = predict_tcga_binary(
        model,
        prepared,
        device=device,
        batch_size=batch_size,
        banks=banks,
        runtime_profile=runtime_profile,
        terminal_latent_path=terminal_latent_path if export_terminal_latents else None,
    )
    pred.to_csv(out_dir / "tcga_binary_scores.csv", index=False)
    pred[["episode_id", "patient_id", "sample_id", "DRUG_ID", "DRUG_NAME", "split", "y", "prob", "logit"]].to_csv(
        out_dir / "predictions.csv",
        index=False,
    )
    metrics_frames = []
    calibration_frames = []
    for split, group in pred.groupby("split", observed=True):
        if group.empty:
            continue
        metrics = binary_response_metrics(group["y"].to_numpy(), group["prob"].to_numpy(), threshold=0.5)
        metrics["split"] = split
        metrics["brier"] = float(brier_score_loss(group["y"].to_numpy().astype(int), group["prob"].to_numpy()))
        metrics["n_pos"] = int(group["y"].sum())
        metrics["n_neg"] = int(len(group) - int(group["y"].sum()))
        metrics_frames.append(metrics)
        cal = calibration_by_quantile(group["y"].to_numpy(), group["prob"].to_numpy())
        if not cal.empty:
            cal.insert(0, "split", split)
            calibration_frames.append(cal)
    metrics_df = pd.DataFrame(metrics_frames)
    metrics_df.to_csv(out_dir / "tcga_binary_metrics.csv", index=False)
    if calibration_frames:
        pd.concat(calibration_frames, ignore_index=True).to_csv(out_dir / "tcga_binary_calibration.csv", index=False)
    else:
        pd.DataFrame(columns=["split", "quantile", "n", "score_mean", "response_rate"]).to_csv(
            out_dir / "tcga_binary_calibration.csv",
            index=False,
        )
    _write_binary_summary(out_dir, metrics_df)
    return pred


def predict_tcga_binary(
    model: TerminalWorldModel,
    prepared: TcgaBinaryPrepared,
    device: str | None = None,
    batch_size: int | None = None,
    banks: BinaryTensorBanks | None = None,
    runtime_profile: str = RUNTIME_PROFILE_STABLE,
    terminal_latent_path: Path | None = None,
) -> pd.DataFrame:
    if device is None:
        raise ValueError("Explicit --device is required for evaluate/run. Use --device cuda:N or --device cpu.")
    set_reproducible_runtime(seed=0, device=device, profile=runtime_profile)
    model = model.to(device).eval()
    batch_size = int(batch_size or _default_eval_batch_size(device))
    banks = banks or _tensor_banks(prepared, device)
    kg_drug_idx_bank = model.local_drug_indices(
        [int(row.DRUG_ID) for row in prepared.artifacts.drug_table.itertuples(index=False)],
        device=device,
    )
    precomputed_kg_payload = model.precompute_kg_payload(device=device)
    frame = prepared.episodes.reset_index(drop=True).copy()
    tensors = _frame_to_index_tensors(frame, banks.patient_to_idx, banks.drug_to_idx, device)
    prob = np.empty((len(frame),), dtype=np.float32)
    logit = np.empty((len(frame),), dtype=np.float32)
    latent_writer = None
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            state_idx = tensors.state_idx[start:stop]
            drug_idx = tensors.drug_idx[start:stop]
            out = model(
                banks.state_bank.index_select(0, state_idx),
                banks.drug_bank.index_select(0, drug_idx),
                banks.prior_bank.index_select(0, drug_idx),
                banks.mask_bank.index_select(0, drug_idx),
                use_terminal=True,
                drug_idx=kg_drug_idx_bank.index_select(0, drug_idx) if kg_drug_idx_bank is not None else None,
                compute_kg_consistency=False,
                precomputed_kg_payload=precomputed_kg_payload,
            )
            logit[start:stop] = out["binary_logit"].detach().cpu().numpy()
            prob[start:stop] = torch.sigmoid(out["binary_logit"]).detach().cpu().numpy()
            if terminal_latent_path is not None:
                latent = out["terminal_latent"].detach().cpu().numpy()
                if latent_writer is None:
                    latent_writer = open_memmap(
                        terminal_latent_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(frame), int(latent.shape[-1])),
                    )
                latent_writer[start:stop] = latent
    if latent_writer is not None:
        latent_writer.flush()
    frame["prob"] = prob
    frame["logit"] = logit
    if "drug_name" in frame.columns and "DRUG_NAME" not in frame.columns:
        frame["DRUG_NAME"] = frame["drug_name"]
    return frame


def run_tcga_binary_benchmark(
    args,
    config: BenchmarkConfig,
    bench_paths: dict[str, Path],
    paths: Paths,
) -> None:
    processed_dir = ensure_dir(bench_paths["processed"] / args.processed_name)
    use_hvg = bool(getattr(args, "tcga_use_hvg", True))
    runtime_profile = getattr(args, "runtime_profile", RUNTIME_PROFILE_STABLE)
    export_terminal_latents = (
        bool(args.export_terminal_latents)
        if getattr(args, "export_terminal_latents", None) is not None
        else bool(getattr(config, "export_terminal_latents", False))
    )
    if args.cmd == "prepare":
        prepare_tcga_binary_data(
            paths,
            processed_dir,
            benchmark=config,
            seed=args.seed,
            n_components=args.n_components or config.n_components,
            n_hvg=args.n_hvg or config.n_hvg,
            use_hvg=use_hvg,
            use_cache=not getattr(args, "no_cache", False),
            rebuild_cache=getattr(args, "rebuild_cache", False),
        )
        return
    if args.device is None:
        raise SystemExit("Explicit --device is required for train/evaluate/run. Use --device cuda:N or --device cpu.")
    if args.cmd == "evaluate" and args.run_name is None:
        latest = _latest_result_dir(bench_paths["results"])
        if latest is None:
            raise SystemExit("No existing result directory found for evaluate. Pass --run-name after a train/run step.")
        result_dir = latest
    else:
        from datetime import datetime

        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = ensure_dir(bench_paths["results"] / run_name)
    copy_config_snapshot(bench_paths["config"], [result_dir])
    prepared = _load_or_prepare_binary(processed_dir, paths, config, args, bench_paths)
    eval_batch_size = int(getattr(args, "eval_batch_size", None) or config.eval_batch_size or _default_eval_batch_size(args.device))
    effective_epochs = int(args.epochs if args.epochs is not None else config.epochs)
    effective_batch_size = int(args.batch_size if args.batch_size is not None else config.batch_size)
    if args.cmd == "train":
        _log_batch_resolution(
            config=config,
            effective_epochs=effective_epochs,
            effective_batch_size=effective_batch_size,
            effective_eval_batch_size=None,
        )
        train_tcga_binary_model(
            prepared,
            result_dir,
            epochs=effective_epochs,
            batch_size=effective_batch_size,
            lr=float(args.lr if args.lr is not None else config.lr),
            seed=args.seed,
            device=args.device,
            runtime_profile=runtime_profile,
        )
        _copy_binary_prepare_outputs(processed_dir, result_dir)
        generate_run_visualizations(result_dir, device=args.device)
        return
    if args.cmd == "evaluate":
        banks = _tensor_banks(prepared, args.device)
        model = load_model(result_dir, prepared.artifacts, strict=False)
        _log_batch_resolution(
            config=config,
            effective_epochs=getattr(config, "epochs", None),
            effective_batch_size=getattr(config, "batch_size", None),
            effective_eval_batch_size=eval_batch_size,
        )
        evaluate_tcga_binary(
            model,
            prepared,
            result_dir,
            device=args.device,
            batch_size=eval_batch_size,
            banks=banks,
            runtime_profile=runtime_profile,
            export_terminal_latents=export_terminal_latents,
            terminal_latent_path=result_dir / "terminal_latents.npy",
        )
        _copy_binary_prepare_outputs(processed_dir, result_dir)
        generate_run_visualizations(result_dir, device=args.device)
        return
    if args.cmd == "run":
        banks = _tensor_banks(prepared, args.device)
        _log_batch_resolution(
            config=config,
            effective_epochs=effective_epochs,
            effective_batch_size=effective_batch_size,
            effective_eval_batch_size=eval_batch_size,
        )
        model = train_tcga_binary_model(
            prepared,
            result_dir,
            epochs=effective_epochs,
            batch_size=effective_batch_size,
            lr=float(args.lr if args.lr is not None else config.lr),
            seed=args.seed,
            device=args.device,
            runtime_profile=runtime_profile,
        )
        evaluate_tcga_binary(
            model,
            prepared,
            result_dir,
            device=args.device,
            batch_size=eval_batch_size,
            banks=banks,
            runtime_profile=runtime_profile,
            export_terminal_latents=export_terminal_latents,
            terminal_latent_path=result_dir / "terminal_latents.npy",
        )
        _copy_binary_prepare_outputs(processed_dir, result_dir)
        generate_run_visualizations(result_dir, device=args.device)


def _load_or_prepare_binary(processed_dir: Path, paths: Paths, config: BenchmarkConfig, args, bench_paths: dict[str, Path]) -> TcgaBinaryPrepared:
    prepared_path = processed_dir / "prepared.pkl"
    use_hvg = bool(getattr(args, "tcga_use_hvg", True))
    if prepared_path.exists():
        prepared = load_prepared(prepared_path)
        expected_key = _cache_contract_key(config, paths, args.n_hvg or config.n_hvg, use_hvg)
        if prepared.manifest.get("prepare_cache_key") == expected_key:
            return prepared
    return prepare_tcga_binary_data(
        paths,
        processed_dir,
        benchmark=config,
        seed=args.seed,
        n_components=args.n_components or config.n_components,
        n_hvg=args.n_hvg or config.n_hvg,
        use_hvg=use_hvg,
        cache_dir=processed_dir / ".cache",
        use_cache=not getattr(args, "no_cache", False),
        rebuild_cache=getattr(args, "rebuild_cache", False),
    )


def _tensor_banks(prepared: TcgaBinaryPrepared, device: str) -> BinaryTensorBanks:
    patient_ids = prepared.state_matrix.index.astype(str).tolist()
    patient_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    state_bank = torch.as_tensor(prepared.state_matrix.to_numpy(np.float32), dtype=torch.float32, device=device)
    drug_rows = list(prepared.artifacts.drug_table.itertuples(index=False))
    drug_to_idx = {int(row.DRUG_ID): i for i, row in enumerate(drug_rows)}
    drug_bank = torch.as_tensor(np.vstack([row.fingerprint.astype(np.float32) for row in drug_rows]), dtype=torch.float32, device=device)
    if len(prepared.artifacts.prior_columns) == 0:
        prior_bank = torch.empty((len(drug_rows), 0), dtype=torch.float32, device=device)
    else:
        prior_bank = torch.as_tensor(np.vstack([row.prior.astype(np.float32) for row in drug_rows]), dtype=torch.float32, device=device)
    mask_bank = torch.as_tensor([[float(row.prior_mask)] for row in drug_rows], dtype=torch.float32, device=device)
    return BinaryTensorBanks(
        state_bank=state_bank,
        drug_bank=drug_bank,
        prior_bank=prior_bank,
        mask_bank=mask_bank,
        patient_to_idx=patient_to_idx,
        drug_to_idx=drug_to_idx,
    )


def _frame_to_index_tensors(frame: pd.DataFrame, patient_to_idx: dict[str, int], drug_to_idx: dict[int, int], device: str) -> BinaryEpisodeTensors:
    return BinaryEpisodeTensors(
        state_idx=torch.as_tensor([patient_to_idx[str(x)] for x in frame["patient_id"]], dtype=torch.long, device=device),
        drug_idx=torch.as_tensor([drug_to_idx[int(x)] for x in frame["DRUG_ID"]], dtype=torch.long, device=device),
        y=torch.as_tensor(frame["y"].astype(np.float32).to_numpy(), dtype=torch.float32, device=device),
    )


def _default_eval_batch_size(device: str) -> int:
    return 4096 if device == "cpu" else 32768


def _log_batch_resolution(
    *,
    config: BenchmarkConfig,
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


def _iter_batch_indices(n_rows: int, batch_size: int, device: str, *, shuffle: bool, seed: int) -> tuple[torch.Tensor, ...]:
    if n_rows <= 0:
        return ()
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = torch.randperm(n_rows, generator=generator)
    else:
        order = torch.arange(n_rows)
    if str(device) != "cpu":
        order = order.to(device=device)
    return tuple(order[start : start + batch_size] for start in range(0, n_rows, batch_size))


def _split_audit(frame: pd.DataFrame, split_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in frame.groupby("split", observed=True):
        rows.append(
            {
                "audit_type": "split_summary",
                "split": split,
                "n_rows": int(len(group)),
                "n_patients": int(group["patient_id"].nunique()),
                "n_drugs": int(group["DRUG_ID"].nunique()),
                "detail": split_type,
            }
        )
    return pd.DataFrame(rows)


def _write_prepare_outputs(
    prepared: TcgaBinaryPrepared,
    out_dir: Path,
    benchmark: BenchmarkConfig,
    seed: int,
    n_components: int,
    n_hvg: int,
    key: str,
    *,
    use_hvg: bool,
    from_cache: bool,
) -> None:
    out_dir = ensure_dir(out_dir)
    write_json(out_dir / "manifest.json", prepared.manifest)
    prepared.episodes.to_csv(out_dir / "tcga_binary_episodes.csv", index=False)
    prepared.label_audit.to_csv(out_dir / "label_audit.csv", index=False)
    prepared.split_audit.to_csv(out_dir / "split_audit.csv", index=False)
    prepared.mapping_audit.to_csv(out_dir / "mapping_audit.csv", index=False)
    prepared.label_summary.to_csv(out_dir / "label_summary.csv", index=False)
    write_kg_reports(out_dir, getattr(prepared.artifacts, "kg_graph", None))
    write_json(
        out_dir / "benchmark_contract.json",
        {
            "benchmark_id": benchmark.benchmark_id,
            "benchmark_name": benchmark.benchmark_name,
            "task_type": benchmark.task_type,
            "split_type": benchmark.split_type,
            "label_policy": benchmark.label_policy,
            "instance_unit": benchmark.instance_unit,
            "seed": int(seed),
            "n_components": int(n_components),
            "n_hvg": int(n_hvg) if use_hvg else None,
            "gene_mode": "hvg" if use_hvg else "all_genes",
            "n_genes_selected": int(len(prepared.artifacts.genes)),
            "prepare_cache_key": key,
            "n_episodes": int(len(prepared.episodes)),
            "n_patients": int(prepared.episodes["patient_id"].nunique()),
            "n_drugs": int(prepared.episodes["DRUG_ID"].nunique()),
            "n_positive": int(prepared.episodes["y"].sum()),
            "n_negative": int(len(prepared.episodes) - int(prepared.episodes["y"].sum())),
            "from_cache": bool(from_cache),
            "files": {
                "tcga_binary_episodes": "tcga_binary_episodes.csv",
                "label_audit": "label_audit.csv",
                "split_audit": "split_audit.csv",
                "mapping_audit": "mapping_audit.csv",
                "label_summary": "label_summary.csv",
            },
        },
    )


def _write_binary_summary(out_dir: Path, metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=["split", "n", "threshold", "auroc", "auprc", "balanced_accuracy", "brier", "n_pos", "n_neg"])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)


def _copy_binary_prepare_outputs(processed_dir: Path, result_dir: Path) -> None:
    for name in [
        "manifest.json",
        "benchmark_contract.json",
        "tcga_binary_episodes.csv",
        "label_audit.csv",
        "split_audit.csv",
        "mapping_audit.csv",
        "label_summary.csv",
        "cache_hits.csv",
        "cache_manifest.json",
    ]:
        src = processed_dir / name
        if src.exists():
            import shutil

            shutil.copy2(src, result_dir / name)


def _ensure_prepared_compat(prepared: Any) -> TcgaBinaryPrepared:
    if hasattr(prepared, "manifest") and hasattr(prepared, "episodes"):
        if not hasattr(prepared.artifacts, "kg_graph"):
            prepared.artifacts.kg_graph = None
        return prepared
    raise TypeError("Loaded TCGA binary prepared artifact is incompatible.")


def _latest_result_dir(results_dir: Path) -> Path | None:
    candidates = [p for p in Path(results_dir).iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.name) if candidates else None
