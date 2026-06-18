"""One-time export: turn a raw GAUGE (drugwm) training-run directory into a
slim, self-contained model bundle that the GAUGE software can ship and load
without depending on the original training repository's absolute paths or
sys.path monkey-patches.

The run directories used here did not persist a `prepared.pkl` (only
`artifacts.pkl` + `model.pt` + the exported `gdsc_pairs.csv` benchmark
contract -- the field is always named that for historical reasons even for
non-GDSC datasets). We reconstruct the equivalent of `PreparedData.state_matrix`
by re-running the *already-fitted* projection in `artifacts` (genes/imputer/
scaler/pca) over each dataset's raw expression matrix, and recomputing the
train-only cell statistics from the pairs file. This is an exact replay,
not an approximation: the imputer/scaler/pca objects are already fit (frozen)
in artifacts.pkl, so applying them again is deterministic.

Usage:
    python export_model_bundle.py --mode gdsc_cell_split
    python export_model_bundle.py --mode all
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

REPO_ROOT = Path("/mnt/raid5/xujing/KG")
SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = SOFTWARE_ROOT / "gauge_core" / "vendor"

# Order matters: VENDOR_DIR also provides a (minimal) `drugwm` package, used
# by the deployed app. This script needs the *full* original `drugwm`
# (train.py, kg_prior.py, ...), so REPO_ROOT must win the `import drugwm`
# lookup -> insert it last so it ends up first in sys.path.
for p in (str(SOFTWARE_ROOT), str(VENDOR_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from drugwm.config import Paths  # noqa: E402
from drugwm.data import load_gdsc_expression  # noqa: E402
from drugwm.features import FeatureArtifacts, build_cell_train_statistics, project_expression  # noqa: E402
from drugwm.train import load_model  # noqa: E402

# Imported as a bare top-level module (not `gauge_core.kg_types`) deliberately:
# importing the `gauge_core` package here would vendor a *minimal* `drugwm`
# package onto sys.path that shadows this script's need for the full
# original `drugwm` (train.py, kg_prior.py, etc.) under the same module name.
from kg_types import KGGraphArtifacts  # noqa: E402

STAT_COLS = ["cell_auc_train_mean", "cell_auc_train_median", "cell_centered_sensitivity_train"]
GDSC_MODEL_LIST_CSV = REPO_ROOT / "KG_DrugWM_PublicData/GDSC/model_list_20260420.csv"
DEPMAP_MODEL_CSV = REPO_ROOT / "KG_DrugWM_PublicData/depmap/Model.csv"


def _load_gdsc_expression() -> pd.DataFrame:
    paths = Paths()
    return load_gdsc_expression(paths.gdsc_expression, paths.gdsc_gene_identifiers)


def _load_depmap_expression() -> pd.DataFrame:
    """DepMap `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`: rows are
    profiling entries (index by ModelID = ACH-xxxxxx), columns are
    "SYMBOL (entrez_id)".  Keep the full "SYMBOL (entrez_id)" column names --
    the PRISM artifacts.genes list is in this same format so the HVG gene
    selection in project_expression matches on the full name."""
    path = REPO_ROOT / "KG_DrugWM_PublicData/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    raw = pd.read_csv(path)
    raw = raw.drop_duplicates(subset="ModelID", keep="first").set_index("ModelID")
    gene_cols = [c for c in raw.columns if c not in {"SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"}]
    expr = raw[gene_cols].apply(pd.to_numeric, errors="coerce")
    expr.index.name = "SANGER_MODEL_ID"
    return expr.astype("float32")


def _gdsc_cell_metadata(cells: list[str]) -> pd.DataFrame | None:
    if not GDSC_MODEL_LIST_CSV.exists():
        return None
    meta = pd.read_csv(GDSC_MODEL_LIST_CSV, usecols=["model_id", "model_name", "tissue", "cancer_type", "COSMIC_ID"])
    meta = meta.rename(columns={"model_id": "SANGER_MODEL_ID"})
    return meta.loc[meta["SANGER_MODEL_ID"].isin(cells)]


def _depmap_cell_metadata(cells: list[str]) -> pd.DataFrame | None:
    if not DEPMAP_MODEL_CSV.exists():
        return None
    meta = pd.read_csv(
        DEPMAP_MODEL_CSV,
        usecols=["ModelID", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease"],
    )
    meta = meta.rename(
        columns={
            "ModelID": "SANGER_MODEL_ID",
            "CellLineName": "model_name",
            "OncotreeLineage": "tissue",
            "OncotreePrimaryDisease": "cancer_type",
        }
    )
    return meta.loc[meta["SANGER_MODEL_ID"].isin(cells)]


@dataclass
class DatasetSpec:
    run_dir: Path
    load_expression: Callable[[], pd.DataFrame]
    load_cell_metadata: Callable[[list[str]], pd.DataFrame | None]
    split_type: str
    display_name: str
    notes: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "gdsc_cell_split": DatasetSpec(
        run_dir=REPO_ROOT / "benchmarking/01_random_cell_split/HVG2000/results/GDSCv2_seed/true_seed7",
        load_expression=_load_gdsc_expression,
        load_cell_metadata=_gdsc_cell_metadata,
        split_type="random_cell_line_held_out",
        display_name="GDSC — known compound library (cell-line split, recommended default)",
        notes=(
            "GDSC1/2. input_projection_mode=hvg (top-2000-variance genes selected on the "
            "training cell lines; identity projection, no PCA dimensionality reduction) "
            "+ 3 train-only cell-level AUC statistics."
        ),
    ),
    "gdsc_drug_split": DatasetSpec(
        run_dir=REPO_ROOT / "benchmarking/02_drug_split/HVG2000/results/GDSC_v2_seed/true_seed7",
        load_expression=_load_gdsc_expression,
        load_cell_metadata=_gdsc_cell_metadata,
        split_type="drug_held_out",
        display_name="GDSC — novel-compound mode (drug split)",
        notes="Same as gdsc_cell_split but evaluated under a held-out-drug split.",
    ),
    "prism_cell_split": DatasetSpec(
        run_dir=REPO_ROOT
        / "benchmarking/01_random_cell_split/HVG2000/results/PRISM_secondary_seed/full_terminal_specificity_20260525_040830/seed7",
        load_expression=_load_depmap_expression,
        load_cell_metadata=_depmap_cell_metadata,
        split_type="random_cell_line_held_out",
        display_name="PRISM Repurposing Secondary Screen — large compound library (cell-line split, recommended for PRISM)",
        notes=(
            "Broad Institute Drug Repurposing Hub PRISM secondary screen "
            "(~1,400+ compounds incl. approved/investigational drugs), DepMap "
            "(ACH-id) cell lines and expression "
            "(OmicsExpressionTPMLogp1HumanProteinCodingGenes), real-valued "
            "AUC (not bounded to [0,1] the way GDSC's is). Held-out-cell-line split."
        ),
    ),
    "prism_drug_split": DatasetSpec(
        run_dir=REPO_ROOT / "benchmarking/02_drug_split/HVG2000/results/seed/full_20260526_014336_pid2160191_seed5",
        load_expression=_load_depmap_expression,
        load_cell_metadata=_depmap_cell_metadata,
        split_type="drug_held_out",
        display_name="PRISM Repurposing Secondary Screen — large compound library (novel-compound mode)",
        notes=(
            "Same dataset as prism_cell_split, evaluated under a held-out-drug split "
            "instead (seed5, not seed7 -- the seed7 drug-split run for PRISM was not "
            "available with a saved checkpoint)."
        ),
    ),
}


def export(mode: str, out_dir: Path) -> None:
    spec = DATASET_SPECS[mode]
    run_dir = spec.run_dir
    print(f"[{mode}] loading artifacts.pkl from {run_dir} ...")
    with (run_dir / "artifacts.pkl").open("rb") as f:
        artifacts: FeatureArtifacts = pickle.load(f)

    print(f"[{mode}] loading + validating model.pt ...")
    model = load_model(run_dir, artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    print(f"[{mode}] writing slim artifacts.pkl ...")
    slim = FeatureArtifacts(
        genes=artifacts.genes,
        imputer=artifacts.imputer,
        scaler=artifacts.scaler,
        pca=artifacts.pca,
        split_by_cell={},
        drug_baseline_auc=artifacts.drug_baseline_auc,
        drug_auc_train_values=artifacts.drug_auc_train_values,
        drug_table=artifacts.drug_table,
        prior_columns=artifacts.prior_columns,
        state_dim=artifacts.state_dim,
        kg_graph=KGGraphArtifacts.from_original(artifacts.kg_graph),
        canonical_drug_table=artifacts.canonical_drug_table,
        drug_id_to_canonical_idx=artifacts.drug_id_to_canonical_idx,
    )
    with (out_dir / "artifacts.pkl").open("wb") as f:
        pickle.dump(slim, f)

    print(f"[{mode}] loading gdsc_pairs.csv benchmark contract ...")
    pairs = pd.read_csv(run_dir / "gdsc_pairs.csv", usecols=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "split", "AUC"])
    pairs["SANGER_MODEL_ID"] = pairs["SANGER_MODEL_ID"].astype(str)
    cells = sorted(pairs["SANGER_MODEL_ID"].unique().tolist())
    pairs.to_csv(out_dir / "response_table.csv.gz", index=False, compression="gzip")

    print(f"[{mode}] loading raw expression matrix ...")
    raw_expr = spec.load_expression()

    print(f"[{mode}] re-projecting {len(cells)} known cell lines through fitted artifacts ...")
    cells_in_expr = [c for c in cells if c in raw_expr.index]
    missing = sorted(set(cells) - set(cells_in_expr))
    if missing:
        print(f"[{mode}] WARNING: {len(missing)} cell lines absent from raw expression file, skipped (e.g. {missing[:5]})")
    projected = project_expression(raw_expr.loc[cells_in_expr], artifacts.genes, artifacts.imputer, artifacts.scaler, artifacts.pca)
    state = pd.DataFrame(projected, index=cells_in_expr)

    cell_stats = build_cell_train_statistics(pairs).reindex(cells_in_expr)
    train_rows = pairs.loc[pairs["split"].eq("train")]
    global_mean = float(train_rows["AUC"].mean())
    global_median = float(train_rows["AUC"].median())
    fill_values = {
        "cell_auc_train_mean": global_mean,
        "cell_auc_train_median": global_median,
        "cell_centered_sensitivity_train": 0.0,
    }
    state = pd.concat([state, cell_stats.fillna(fill_values)], axis=1)
    state.index.name = "SANGER_MODEL_ID"
    state.to_csv(out_dir / "cell_state_matrix.csv.gz", compression="gzip")
    state[STAT_COLS].to_csv(out_dir / "cell_stats_lookup.csv")

    with (out_dir / "global_auc_stats.json").open("w") as f:
        json.dump({"global_auc_train_mean": global_mean, "global_auc_train_median": global_median}, f, indent=2)

    print(f"[{mode}] exporting drug library (chemistry + KG coverage) ...")
    drug_table = artifacts.canonical_drug_table if artifacts.canonical_drug_table is not None else artifacts.drug_table
    keep_cols = [c for c in ["DRUG_ID", "DRUG_NAME", "drug_key", "canonical_smiles", "smiles", "inchikey", "pubchem_cid"] if c in drug_table.columns]
    drug_lib = drug_table[keep_cols].copy()
    kg_graph = getattr(artifacts, "kg_graph", None)
    coverage = getattr(kg_graph, "coverage", None)
    if coverage is not None and not coverage.empty:
        cov_cols = [c for c in coverage.columns if c.startswith("has_")]
        drug_lib = drug_lib.merge(coverage[["DRUG_ID", *cov_cols]], on="DRUG_ID", how="left")
    drug_lib.to_csv(out_dir / "drug_library.csv", index=False)

    print(f"[{mode}] exporting cell-line metadata ...")
    meta = spec.load_cell_metadata(state.index.tolist())
    if meta is not None and not meta.empty:
        meta.to_csv(out_dir / "cell_metadata.csv", index=False)

    metrics_path = run_dir / "metrics.csv"
    reported_metrics = pd.read_csv(metrics_path).to_dict(orient="records") if metrics_path.exists() else []
    manifest_path = run_dir / "manifest.json"
    selected_fusion_weight = 1.0
    if manifest_path.exists():
        with manifest_path.open() as f:
            selected_fusion_weight = float(json.load(f).get("selected_fusion_weight", 1.0))
    bundle_meta = {
        "mode": mode,
        "display_name": spec.display_name,
        "source_run_dir": str(run_dir),
        "split_type": spec.split_type,
        "state_dim": int(model.state_encoder.net[0].in_features),
        "n_known_cell_lines": int(state.shape[0]),
        "n_known_drugs": int(drug_lib.shape[0]),
        "kg_sources": list(getattr(kg_graph, "branch_names", [])),
        "selected_fusion_weight": selected_fusion_weight,
        "reported_metrics_this_seed": reported_metrics,
        "notes": spec.notes,
    }
    with (out_dir / "bundle_meta.json").open("w") as f:
        json.dump(bundle_meta, f, indent=2, default=str)

    print(f"[{mode}] done -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*DATASET_SPECS.keys(), "all"], default="all")
    args = parser.parse_args()
    modes = list(DATASET_SPECS.keys()) if args.mode == "all" else [args.mode]
    for m in modes:
        export(m, SOFTWARE_ROOT / "models" / m)
