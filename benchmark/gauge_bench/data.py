"""Loader for the self-contained benchmark dataset under benchmark/data/."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import KGArtifacts

SPLITS = ("drug_split", "scaffold_split", "chemcluster_split")
SEEDS = (5, 7, 43, 91, 98)


@dataclass
class BenchmarkData:
    responses: pd.DataFrame          # one row per cell-line x drug measurement
    state_matrix: np.ndarray         # (n_cells, state_dim) float32
    cell_ids: list[str]              # row order of state_matrix
    kg: KGArtifacts
    fingerprints: np.ndarray         # (n_drug_bank, n_bits) float32, KG bank order
    meta: dict

    @property
    def cell_to_row(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.cell_ids)}

    @property
    def state_dim(self) -> int:
        return int(self.state_matrix.shape[1])


def load_dataset(data_root: str | Path, split: str, seed: int) -> BenchmarkData:
    root = Path(data_root)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    shared = root / "shared"
    resp_path = root / split / "responses" / f"seed{seed}.parquet"
    if not resp_path.is_file():
        raise FileNotFoundError(resp_path)

    responses = pd.read_parquet(resp_path)
    responses["SANGER_MODEL_ID"] = responses["SANGER_MODEL_ID"].astype(str)

    state = np.load(shared / "state_matrix.npz", allow_pickle=False)
    bank = np.load(shared / "drug_bank.npz", allow_pickle=False)
    n_bits = int(bank["n_bits"])
    fingerprints = np.unpackbits(bank["fingerprints"], axis=1)[:, :n_bits].astype(np.float32)

    kg = KGArtifacts(
        node_table=pd.read_parquet(shared / "kg_nodes.parquet"),
        edge_table=pd.read_parquet(shared / "kg_edges.parquet"),
        coverage=pd.read_parquet(shared / "kg_coverage.parquet"),
        drug_ids=[int(x) for x in bank["drug_ids"]],
        branch_names=tuple(json.loads((shared / "meta.json").read_text())["branch_names"]),
    )

    missing = set(responses["canonical_drug_id"].astype(int)) - set(kg.drug_ids)
    if missing:
        raise ValueError(f"{len(missing)} canonical drugs are absent from the KG bank")

    return BenchmarkData(
        responses=responses,
        state_matrix=state["values"].astype(np.float32),
        cell_ids=[str(c) for c in state["cell_ids"]],
        kg=kg,
        fingerprints=fingerprints,
        meta=json.loads((shared / "meta.json").read_text()),
    )
