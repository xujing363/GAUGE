#!/usr/bin/env python3
"""Export a self-contained GAUGE benchmark dataset from an internal prepared.pkl.

The internal objects are pickles of `drugwm.*` classes and cannot be loaded
outside the lab environment.  This writes the same content as plain parquet /
npz / csv, which `gauge.data.load_dataset` reads directly.

Run inside the lab environment; the released repository ships the output.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESP_COLS = [
    "SANGER_MODEL_ID", "DRUG_ID", "canonical_drug_id", "split", "AUC",
    "relative_value_train", "relative_value_eval", "cell_residual_auc_train",
]


def export(prepared_path: Path, out_dir: Path, shared_dir: Path, seed: int,
           write_shared: bool) -> None:
    for p in ("/mnt/raid5/xujing/KG",
              "/mnt/raid5/xujing/KG_publication/manu/bib_major/M02/"
              "canonical_drug_split_5seeds/HVG2000"):
        if p not in sys.path:
            sys.path.insert(0, p)

    with prepared_path.open("rb") as fh:
        prepared = pickle.load(fh)
    art, kg = prepared.artifacts, prepared.artifacts.kg_graph

    (out_dir / "responses").mkdir(parents=True, exist_ok=True)

    resp = prepared.responses[RESP_COLS].copy()
    resp["SANGER_MODEL_ID"] = resp["SANGER_MODEL_ID"].astype(str)
    resp["DRUG_ID"] = resp["DRUG_ID"].astype(int)
    resp["canonical_drug_id"] = resp["canonical_drug_id"].astype(int)
    resp["split"] = resp["split"].astype(str)
    resp.to_parquet(out_dir / "responses" / f"seed{seed}.parquet", index=False)

    if not write_shared:
        return
    shared_dir.mkdir(parents=True, exist_ok=True)

    state = prepared.state_matrix
    np.savez_compressed(
        shared_dir / "state_matrix.npz",
        values=state.to_numpy(np.float32),
        cell_ids=np.asarray(state.index.astype(str), dtype=np.str_),
    )

    kg.node_table[["node_id", "node_type"]].to_parquet(shared_dir / "kg_nodes.parquet", index=False)
    kg.edge_table[["src", "dst", "edge_type", "source"]].to_parquet(
        shared_dir / "kg_edges.parquet", index=False)
    kg.coverage.to_parquet(shared_dir / "kg_coverage.parquet", index=False)

    drug_ids = [int(x) for x in kg.drug_ids]
    ct = art.canonical_drug_table.set_index("DRUG_ID")
    fp = np.stack([np.asarray(ct.loc[d, "fingerprint"], dtype=np.uint8) for d in drug_ids])
    np.savez_compressed(
        shared_dir / "drug_bank.npz",
        drug_ids=np.asarray(drug_ids, dtype=np.int64),
        fingerprints=np.packbits(fp, axis=1),
        n_bits=np.asarray(fp.shape[1]),
    )
    ct.reset_index()[["DRUG_ID", "DRUG_NAME", "canonical_smiles", "inchikey",
                      "canonical_drug_id"]].to_csv(shared_dir / "drugs.csv", index=False)

    (shared_dir / "meta.json").write_text(json.dumps({
        "state_dim": int(state.shape[1]),
        "n_cells": int(state.shape[0]),
        "n_drug_bank": len(drug_ids),
        "branch_names": [str(b) for b in kg.branch_names],
        "n_kg_nodes": int(kg.node_table.shape[0]),
        "n_kg_edges": int(kg.edge_table.shape[0]),
        "n_pairs": int(len(resp)),
        "fingerprint_bits": int(fp.shape[1]),
    }, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True,
                    help="directory holding prepared_seed<S>.pkl")
    ap.add_argument("--out-dir", required=True, help="per-split output directory")
    ap.add_argument("--shared-dir", required=True,
                    help="output directory for the split-independent tensors")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 7, 43, 91, 98])
    args = ap.parse_args()

    src, out, shared = Path(args.prepared_dir), Path(args.out_dir), Path(args.shared_dir)
    write_shared = not (shared / "meta.json").is_file()
    for i, seed in enumerate(args.seeds):
        export(src / f"prepared_seed{seed}.pkl", out, shared, seed,
               write_shared=(write_shared and i == 0))
        print(f"[export] seed {seed} -> {out}", flush=True)


if __name__ == "__main__":
    main()
