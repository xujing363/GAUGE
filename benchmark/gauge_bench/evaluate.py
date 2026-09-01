"""Score a trained GAUGE checkpoint on one benchmark split / seed.

    python -m gauge_bench.evaluate --split drug_split --seed 5 \
        --checkpoint checkpoints/drug_split/seed5 --out-dir runs/eval_seed5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_dataset
from .metrics import decomposition_axes, regression_metrics, value_metrics
from .model import GAUGE


@torch.no_grad()
def predict(model: GAUGE, ds, frame: pd.DataFrame, device, *, kg_mode="multikg_gat",
            batch_size: int = 16384) -> pd.DataFrame:
    model.eval()
    branch_all = model.precompute_branch_all(edge_dropout=0.0)
    state_bank = torch.as_tensor(ds.state_matrix, dtype=torch.float32, device=device)
    state_idx = torch.as_tensor(
        frame["SANGER_MODEL_ID"].astype(str).map(ds.cell_to_row).to_numpy(np.int64), device=device)
    drug_idx = torch.as_tensor(
        frame["canonical_drug_id"].astype(int).map(ds.kg.drug_to_local).to_numpy(np.int64), device=device)
    raws, vals, resids = [], [], []
    for s in range(0, len(frame), batch_size):
        sl = slice(s, s + batch_size)
        out = model(state_bank.index_select(0, state_idx[sl]), drug_idx=drug_idx[sl],
                    kg_mode=kg_mode, edge_dropout=0.0, branch_all=branch_all, fusion_weight=0.0)
        raws.append(out["raw_auc_hat"].float().cpu().numpy())
        vals.append(out["value_hat"].float().cpu().numpy())
        resids.append(out["cell_residual_hat"].float().cpu().numpy())
    pred = frame.copy()
    pred["raw_auc_hat"] = np.concatenate(raws)
    pred["value_hat"] = np.concatenate(vals)
    pred["cell_residual_hat"] = np.concatenate(resids)
    pred["entity_id"] = pred["SANGER_MODEL_ID"].astype(str)
    return pred


def load_checkpoint(ckpt_dir: Path, ds, device) -> tuple[GAUGE, dict]:
    meta_path = ckpt_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    model = GAUGE(state_dim=ds.state_dim, kg=ds.kg, drug_fingerprint_bank=ds.fingerprints).to(device)
    state = torch.load(ckpt_dir / "model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # fixed_source_logits parameterises the static-fusion variant only; it is
    # untouched by the default multikg_gat path and may be absent.
    optional = {"kg_action_encoder.fixed_source_logits"}
    if set(missing) - optional or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model, meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="drug_split",
                    choices=["drug_split", "scaffold_split", "chemcluster_split"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fusion-weight", type=float, default=None,
                    help="default: the weight recorded in the checkpoint's run_meta.json")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    ds = load_dataset(args.data_root, args.split, args.seed)
    model, meta = load_checkpoint(Path(args.checkpoint), ds, device)
    w = args.fusion_weight if args.fusion_weight is not None else float(meta.get("selected_fusion_weight", 0.0))

    rows, preds = [], []
    for split in ("train", "val", "test"):
        frame = ds.responses.loc[ds.responses["split"].eq(split)].reset_index(drop=True)
        pred = predict(model, ds, frame, device)
        pred["auc_hat"] = (pred["raw_auc_hat"].to_numpy(np.float32)
                           + w * pred["cell_residual_hat"].to_numpy(np.float32))
        rows.append({"split": split, "n": len(pred), **regression_metrics(pred),
                     **value_metrics(pred), **decomposition_axes(pred)})
        preds.append(pred[["SANGER_MODEL_ID", "DRUG_ID", "canonical_drug_id", "split", "AUC",
                           "relative_value_eval", "raw_auc_hat", "cell_residual_hat",
                           "value_hat", "auc_hat"]])
        m = rows[-1]
        print(f"[{split}] n={m['n']} overall_pcc={m['overall_pcc']:.4f} "
              f"within_drug_pcc={m['within_drug_pcc_mean']:.4f} "
              f"between_drug_pcc={m['between_drug_pcc']:.4f}", flush=True)

    metrics = pd.DataFrame(rows)
    if args.out_dir:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "gdsc_metrics.csv", index=False)
        pd.concat(preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)
        print(f"[DONE] -> {out}", flush=True)


if __name__ == "__main__":
    main()
