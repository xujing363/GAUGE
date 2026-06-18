from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _drugwm_path  # noqa: F401  must precede numpy/pandas/torch (see module docstring)

import numpy as np
import pandas as pd
import torch

from drugwm.features import FeatureArtifacts  # noqa: E402
from drugwm.model import TerminalWorldModel  # noqa: E402

SOFTWARE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = SOFTWARE_ROOT / "models"

STAT_COLS = ["cell_auc_train_mean", "cell_auc_train_median", "cell_centered_sensitivity_train"]


@dataclass
class ModelBundle:
    mode: str
    model: TerminalWorldModel
    artifacts: FeatureArtifacts
    cell_state_matrix: pd.DataFrame
    cell_stats_lookup: pd.DataFrame
    cell_metadata: pd.DataFrame
    drug_library: pd.DataFrame
    response_table: pd.DataFrame
    global_auc_stats: dict[str, float]
    meta: dict[str, Any]
    precomputed_kg_payload: dict[str, torch.Tensor] | None
    device: str = "cpu"

    @property
    def state_dim(self) -> int:
        return int(self.model.state_encoder.net[0].in_features)

    @property
    def fusion_weight(self) -> float:
        return float(self.meta.get("selected_fusion_weight", 1.0))

    @property
    def drug_table(self) -> pd.DataFrame:
        table = self.artifacts.canonical_drug_table
        return table if table is not None else self.artifacts.drug_table

    def has_kg_routing(self, drug_id: int) -> bool:
        encoder = self.model.kg_action_encoder
        return encoder is not None and int(drug_id) in encoder.drug_to_local

    @property
    def gene_level_state(self) -> bool:
        """True if state-vector column i is literally (standardized) expression
        of artifacts.genes[i] (HVG-identity projection), as opposed to a PCA
        rotation that mixes many genes per component."""
        return type(self.artifacts.pca).__name__ == "IdentityProjection"

    def gene_proxy_series(self, gene: str) -> pd.Series | None:
        """Standardized expression proxy for one gene across all known cell
        lines, read directly out of the bundled (already gene-aligned) state
        matrix. Only available when `gene_level_state` is True."""
        if not self.gene_level_state or gene not in self.artifacts.genes:
            return None
        col = str(self.artifacts.genes.index(gene))
        if col not in self.cell_state_matrix.columns:
            return None
        return self.cell_state_matrix[col]


def _build_model(artifacts: FeatureArtifacts, state_dict: dict[str, torch.Tensor]) -> TerminalWorldModel:
    drug_table = artifacts.canonical_drug_table if artifacts.canonical_drug_table is not None else artifacts.drug_table
    fp_bank = np.vstack([row.fingerprint.astype(np.float32) for row in drug_table.itertuples(index=False)])
    state_dim = int(artifacts.state_dim or 0)
    state_weight = state_dict.get("state_encoder.net.0.weight")
    if state_weight is not None and getattr(state_weight, "shape", None):
        state_dim = int(state_weight.shape[1])
    model = TerminalWorldModel(
        state_dim=state_dim,
        prior_dim=len(artifacts.prior_columns),
        kg_artifacts=getattr(artifacts, "kg_graph", None),
        drug_fingerprint_bank=fp_bank,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


_CACHE: dict[tuple[str, str], ModelBundle] = {}


def load_bundle(mode: str = "gdsc_cell_split", models_dir: Path | None = None, device: str = "cpu") -> ModelBundle:
    """Load a self-contained GAUGE model bundle (cached per mode+device)."""
    root = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    bundle_dir = root / mode
    cache_key = (str(bundle_dir), device)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    with (bundle_dir / "artifacts.pkl").open("rb") as f:
        artifacts: FeatureArtifacts = pickle.load(f)
    try:
        state_dict = torch.load(bundle_dir / "model.pt", map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(bundle_dir / "model.pt", map_location=device)
    model = _build_model(artifacts, state_dict).to(device)

    cell_state_matrix = pd.read_csv(bundle_dir / "cell_state_matrix.csv.gz", index_col=0)
    cell_state_matrix.columns = [str(c) for c in cell_state_matrix.columns]
    cell_stats_lookup = pd.read_csv(bundle_dir / "cell_stats_lookup.csv", index_col=0)
    cell_metadata_path = bundle_dir / "cell_metadata.csv"
    cell_metadata = pd.read_csv(cell_metadata_path) if cell_metadata_path.exists() else pd.DataFrame()
    drug_library = pd.read_csv(bundle_dir / "drug_library.csv")
    response_path = bundle_dir / "response_table.csv.gz"
    response_table = pd.read_csv(response_path) if response_path.exists() else pd.DataFrame()
    with (bundle_dir / "global_auc_stats.json").open() as f:
        global_auc_stats = json.load(f)
    with (bundle_dir / "bundle_meta.json").open() as f:
        meta = json.load(f)

    precomputed_kg_payload = None
    if model.kg_action_encoder is not None:
        with torch.no_grad():
            precomputed_kg_payload = model.precompute_kg_payload(device=device, return_branch_states=True)

    bundle = ModelBundle(
        mode=mode,
        model=model,
        artifacts=artifacts,
        cell_state_matrix=cell_state_matrix,
        cell_stats_lookup=cell_stats_lookup,
        cell_metadata=cell_metadata,
        drug_library=drug_library,
        response_table=response_table,
        global_auc_stats=global_auc_stats,
        meta=meta,
        precomputed_kg_payload=precomputed_kg_payload,
        device=device,
    )
    _CACHE[cache_key] = bundle
    return bundle
