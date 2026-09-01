"""Smoke tests for the benchmark package (CPU, no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

pytest.importorskip("torch")
import torch  # noqa: E402

from gauge_bench.data import SEEDS, SPLITS, load_dataset  # noqa: E402
from gauge_bench.evaluate import load_checkpoint, predict  # noqa: E402
from gauge_bench.metrics import decomposition_axes, regression_metrics  # noqa: E402

DATA = BENCH / "data"
pytestmark = pytest.mark.skipif(not (DATA / "shared" / "meta.json").is_file(),
                                reason="benchmark data not present")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA, "drug_split", 5)


def test_dataset_shapes(dataset):
    assert dataset.state_matrix.shape == (944, 2000)
    assert len(dataset.cell_ids) == 944
    assert dataset.fingerprints.shape[1] == 2048
    assert set(dataset.fingerprints.ravel()) <= {0.0, 1.0}
    assert len(dataset.responses) == 194058
    assert set(dataset.responses["split"]) == {"train", "val", "test"}


def test_splits_are_drug_disjoint():
    for split in SPLITS:
        for seed in SEEDS:
            ds = load_dataset(DATA, split, seed)
            groups = {s: set(g["canonical_drug_id"]) for s, g in
                      ds.responses.groupby("split", observed=True)}
            assert not groups["train"] & groups["test"]
            assert not groups["train"] & groups["val"]
            assert not groups["val"] & groups["test"]


def test_checkpoint_matches_recorded_metrics(dataset):
    ckpt = BENCH / "checkpoints" / "drug_split" / "seed5"
    if not (ckpt / "model.pt").is_file():
        pytest.skip("checkpoint not present")
    import json
    import pandas as pd

    device = torch.device("cpu")
    model, meta = load_checkpoint(ckpt, dataset, device)
    w = float(meta["selected_fusion_weight"])
    frame = dataset.responses.loc[dataset.responses["split"].eq("test")].reset_index(drop=True)
    pred = predict(model, dataset, frame, device)
    pred["auc_hat"] = pred["raw_auc_hat"].to_numpy(np.float32) + w * pred["cell_residual_hat"].to_numpy(np.float32)

    got = regression_metrics(pred)
    ref = pd.read_csv(ckpt / "gdsc_metrics.csv")
    ref = ref.loc[ref["split"].eq("test")].iloc[0]
    assert got["overall_pcc"] == pytest.approx(ref["overall_pcc"], abs=1e-4)
    assert got["within_drug_pcc_mean"] == pytest.approx(ref["within_drug_pcc_mean"], abs=1e-4)
    assert set(decomposition_axes(pred)) == {"between_drug_pcc", "between_cell_pcc", "interaction_pcc"}
    assert json.loads((ckpt / "run_meta.json").read_text())["chem_dropout"] == 0.6
