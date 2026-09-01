"""Train and evaluate GAUGE on one benchmark split / seed.

    python -m gauge_bench.train --split drug_split --seed 5 --out-dir runs/drug_split_seed5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_dataset
from .metrics import (FUSION_CANDIDATES, decomposition_axes, regression_metrics,
                      select_fusion_weight, value_metrics)
from .model import GAUGE, TrainingObjectives, gauge_loss

DEFAULTS = dict(epochs=50, batch_size=512, eval_batch_size=16384, lr=6e-5,
                weight_decay=1e-4, edge_dropout=0.1, chem_dropout=0.6,
                n_cell_lines=32, n_drugs=16, rank_batch_fraction=0.25)


class GroupedIndex:
    """Cell x drug pair -> row positions, for the grouped (ranking) batches."""

    def __init__(self, frame: pd.DataFrame):
        cell_codes, _ = pd.factorize(frame["SANGER_MODEL_ID"].astype(str), sort=False)
        drug_codes, drug_vals = pd.factorize(frame["DRUG_ID"].astype(int), sort=False)
        self.n_cells = int(cell_codes.max()) + 1 if len(cell_codes) else 0
        self.n_drugs = int(len(drug_vals))
        pair = cell_codes.astype(np.int64) * max(self.n_drugs, 1) + drug_codes.astype(np.int64)
        order = np.argsort(pair, kind="stable")
        keys, starts, lengths = np.unique(pair[order], return_index=True, return_counts=True)
        self.table = np.full((max(self.n_cells, 1), max(self.n_drugs, 1)), -1, dtype=np.int64)
        self.table.reshape(-1)[keys] = np.arange(keys.shape[0], dtype=np.int64)
        self.starts, self.lengths = starts.astype(np.int64), lengths.astype(np.int64)
        self.rows = np.arange(len(frame), dtype=np.int64)[order]


def grouped_batches(idx: GroupedIndex, *, n_steps, n_cell_lines, n_drugs,
                    min_cell_drugs=2, min_drug_cells=2, seed=0):
    if idx.n_cells <= 0 or idx.n_drugs <= 0:
        return
    rng = np.random.default_rng(int(seed))
    for _ in range(max(n_steps, 1)):
        for _attempt in range(12):
            cc = rng.choice(idx.n_cells, size=min(idx.n_cells, max(n_cell_lines, 1)), replace=False)
            dc = rng.choice(idx.n_drugs, size=min(idx.n_drugs, max(n_drugs, 1)), replace=False)
            block = idx.table[np.ix_(cc, dc)]
            pos = block.reshape(-1)
            matched = pos[pos >= 0]
            if matched.size == 0:
                continue
            presence = block >= 0
            if int(presence.sum(axis=1).max()) < min_cell_drugs:
                continue
            if int(presence.sum(axis=0).max()) < min_drug_cells:
                continue
            counts = idx.lengths[matched]
            total = int(counts.sum())
            if total == 0:
                continue
            starts = idx.starts[matched]
            offs = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]])
            within = np.arange(total, dtype=np.int64) - np.repeat(offs, counts)
            yield np.sort(idx.rows[np.repeat(starts, counts) + within])
            break


def hybrid_batches(n_rows, batch_size, gidx, n_grouped_steps, rank_batch_fraction,
                   n_cell_lines, n_drugs, seed):
    g = torch.Generator(); g.manual_seed(int(seed))
    order = torch.randperm(n_rows, generator=g).numpy()
    for s in range(0, n_rows, batch_size):
        yield "random_pair", order[s:s + batch_size]
    steps = int(np.ceil(max(n_grouped_steps, 0) * max(float(rank_batch_fraction), 0.0)))
    if steps <= 0:
        return
    for b in grouped_batches(gidx, n_steps=steps, n_cell_lines=n_cell_lines,
                             n_drugs=n_drugs, seed=seed + 104729):
        yield "grouped_rank", b


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="drug_split",
                    choices=["drug_split", "scaffold_split", "chemcluster_split"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kg-mode", default="multikg_gat")
    for name, value in DEFAULTS.items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    seed = args.seed
    device = torch.device(args.device)
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ds = load_dataset(args.data_root, args.split, seed)
    resp, kg = ds.responses, ds.kg
    print(f"[LOAD] {args.split} seed {seed}: rows={len(resp)} state_dim={ds.state_dim} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    state_bank = torch.as_tensor(ds.state_matrix, dtype=torch.float32, device=device)
    cell_to_row = ds.cell_to_row
    drug_to_local = kg.drug_to_local
    fp_bank = ds.fingerprints

    frames, tensors = {}, {}
    for split in ("train", "val", "test"):
        df = resp.loc[resp["split"].eq(split)].reset_index(drop=True).copy()
        frames[split] = df
        tensors[split] = {
            "state_idx": torch.as_tensor(df["SANGER_MODEL_ID"].astype(str).map(cell_to_row).to_numpy(np.int64), device=device),
            "drug_idx": torch.as_tensor(df["canonical_drug_id"].astype(int).map(drug_to_local).to_numpy(np.int64), device=device),
            "auc": torch.as_tensor(df["AUC"].to_numpy(np.float32), device=device),
            "value_train": torch.as_tensor(df["relative_value_train"].to_numpy(np.float32), device=device),
            "cell_residual_train": torch.as_tensor(df["cell_residual_auc_train"].to_numpy(np.float32), device=device),
            "drug_group_id": torch.as_tensor(df["canonical_drug_id"].astype(int).to_numpy(np.int64), device=device),
            "cell_group_id": torch.as_tensor(df["SANGER_MODEL_ID"].astype(str).map(cell_to_row).to_numpy(np.int64), device=device),
        }
        print(f"[SPLIT] {split}: n={len(df)} drugs={df['canonical_drug_id'].nunique()} "
              f"cells={df['SANGER_MODEL_ID'].nunique()}", flush=True)

    model = GAUGE(state_dim=ds.state_dim, kg=kg, drug_fingerprint_bank=fp_bank,
                  chem_dropout=args.chem_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cfg = TrainingObjectives()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params={n_params} nodes={kg.node_table.shape[0]} edges={kg.edge_table.shape[0]} "
          f"chem_dropout={args.chem_dropout} kg_mode={args.kg_mode}", flush=True)

    gidx = GroupedIndex(frames["train"])
    n_train = len(frames["train"])
    grouped_n_steps = max(int(np.ceil(n_train / max(args.n_cell_lines * args.n_drugs, 1))), 1)

    @torch.no_grad()
    def evaluate(split: str) -> pd.DataFrame:
        model.eval()
        branch_all = model.precompute_branch_all(edge_dropout=0.0)
        t = tensors[split]
        raws, vals, resids = [], [], []
        for s in range(0, len(frames[split]), args.eval_batch_size):
            sl = slice(s, s + args.eval_batch_size)
            out = model(state_bank.index_select(0, t["state_idx"][sl]),
                        drug_idx=t["drug_idx"][sl], kg_mode=args.kg_mode, edge_dropout=0.0,
                        branch_all=branch_all, fusion_weight=0.0)
            raws.append(out["raw_auc_hat"].float().cpu().numpy())
            vals.append(out["value_hat"].float().cpu().numpy())
            resids.append(out["cell_residual_hat"].float().cpu().numpy())
        pred = frames[split].copy()
        pred["raw_auc_hat"] = np.concatenate(raws)
        pred["value_hat"] = np.concatenate(vals)
        pred["cell_residual_hat"] = np.concatenate(resids)
        pred["entity_id"] = pred["SANGER_MODEL_ID"].astype(str)
        return pred

    history, best = [], {"score": -np.inf, "epoch": -1, "w": 0.0, "state": None}
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.perf_counter()
        sums: dict[str, torch.Tensor] = {}
        count = 0
        t = tensors["train"]
        plan = list(hybrid_batches(n_train, args.batch_size, gidx, grouped_n_steps,
                                   args.rank_batch_fraction, args.n_cell_lines, args.n_drugs,
                                   seed=seed + epoch))
        flat = torch.as_tensor(np.concatenate([r for _, r in plan]), dtype=torch.long, device=device)
        bounds = np.cumsum([0] + [len(r) for _, r in plan])
        for step, (kind, rows) in enumerate(plan):
            bi = flat[bounds[step]:bounds[step + 1]]
            out = model(state_bank.index_select(0, t["state_idx"].index_select(0, bi)),
                        drug_idx=t["drug_idx"].index_select(0, bi), kg_mode=args.kg_mode,
                        edge_dropout=args.edge_dropout, fusion_weight=0.0)
            loss, parts = gauge_loss(
                out,
                auc=t["auc"].index_select(0, bi),
                value_train=t["value_train"].index_select(0, bi),
                cell_residual_train=t["cell_residual_train"].index_select(0, bi),
                drug_group_id=t["drug_group_id"].index_select(0, bi),
                cell_group_id=t["cell_group_id"].index_select(0, bi),
                cfg=cfg,
                ranking_enabled=(kind == "grouped_rank"),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = len(rows); count += n
            for k, v in parts.items():
                sums[k] = sums[k] + v * n if k in sums else v * n

        val_pred = evaluate("val")
        w, _ = select_fusion_weight(val_pred)
        val_fused = val_pred.copy()
        val_fused["auc_hat"] = (val_fused["raw_auc_hat"].to_numpy(np.float32)
                                + w * val_fused["cell_residual_hat"].to_numpy(np.float32))
        vm = regression_metrics(val_fused)
        row = {"epoch": epoch, **{k: float(v) / max(count, 1) for k, v in sums.items()},
               "selected_fusion_weight": w,
               "val_overall_pcc": vm["overall_pcc"],
               "val_within_drug_pcc": vm["within_drug_pcc_mean"],
               "val_within_drug_spearman": vm["within_drug_spearman_mean"],
               "val_raw_auc_rmse": vm["raw_auc_rmse"],
               "epoch_seconds": time.perf_counter() - ep_t0}
        score = float(vm["overall_pcc"]) if np.isfinite(vm["overall_pcc"]) else -np.inf
        row["validation_score"] = score
        history.append(row)
        print(f"[EPOCH {epoch:02d}] loss={row['loss_total']:.5f} val_pcc={score:.4f} "
              f"val_wd_pcc={vm['within_drug_pcc_mean']:.4f} w={w} ({row['epoch_seconds']:.0f}s)", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "w": w,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    train_seconds = time.perf_counter() - train_started
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    w = float(best["w"])
    rows, preds = [], []
    for split in ("train", "val", "test"):
        pred = evaluate(split)
        pred["auc_hat"] = (pred["raw_auc_hat"].to_numpy(np.float32)
                           + w * pred["cell_residual_hat"].to_numpy(np.float32))
        m = {"split": split, "n": len(pred), **regression_metrics(pred),
             **value_metrics(pred), **decomposition_axes(pred)}
        rows.append(m)
        preds.append(pred[["SANGER_MODEL_ID", "DRUG_ID", "canonical_drug_id", "split", "AUC",
                           "relative_value_eval", "raw_auc_hat", "cell_residual_hat",
                           "value_hat", "auc_hat"]])
        print(f"[FINAL {split}] n={len(pred)} overall_pcc={m['overall_pcc']:.4f} "
              f"within_drug_pcc={m['within_drug_pcc_mean']:.4f} "
              f"between_drug_pcc={m['between_drug_pcc']:.4f} rmse={m['raw_auc_rmse']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "gdsc_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)

    json.dump({
        "split": args.split, "seed": seed, "device": str(device), "kg_mode": args.kg_mode,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "edge_dropout": args.edge_dropout,
        "chem_dropout": args.chem_dropout, "sampler": "hybrid_drug_level",
        "n_cell_lines": args.n_cell_lines, "n_drugs": args.n_drugs,
        "rank_batch_fraction": args.rank_batch_fraction,
        "validation_objective": "val_overall_pcc",
        "fusion_selection_metric": "overall_pcc",
        "fusion_weight_candidates": list(FUSION_CANDIDATES),
        "selected_fusion_weight": w, "best_epoch": best["epoch"],
        "best_val_overall_pcc": best["score"],
        "parameter_count": n_params, "state_dim": ds.state_dim,
        "kg_nodes": int(kg.node_table.shape[0]), "kg_edges": int(kg.edge_table.shape[0]),
        "train_seconds_total": train_seconds,
        "training_objectives": cfg.__dict__,
        "peak_gpu_mem_GB": float(torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0,
    }, (out_dir / "run_meta.json").open("w"), indent=2)
    print(f"[DONE] {args.split} seed={seed} best_epoch={best['epoch']} w={w} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
