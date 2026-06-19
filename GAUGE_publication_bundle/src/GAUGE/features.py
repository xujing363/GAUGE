from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureArtifacts:
    genes: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    split_by_cell: dict[str, str]
    drug_baseline_auc: dict[int, float]
    drug_auc_train_values: dict[int, Sequence[float]]
    drug_table: pd.DataFrame
    prior_columns: list[str]
    state_dim: int | None = None
    kg_graph: Any | None = None
    canonical_drug_table: pd.DataFrame | None = None
    drug_id_to_canonical_idx: dict[int, int] = field(default_factory=dict)


_RDKIT_IMPORT_ERROR: Exception | None = None
_RDKIT_CHEM: Any | None = None
_RDKIT_DATASTRUCTS: Any | None = None
_RDKIT_FP_GENERATORS: dict[int, Any] = {}


def split_cell_lines(cell_lines: list[str], seed: int = 7) -> dict[str, str]:
    train, tmp = train_test_split(cell_lines, train_size=0.70, random_state=seed)
    val_fraction = 1 / 3 if len(tmp) >= 3 else 0.5
    val, test = train_test_split(tmp, train_size=val_fraction, random_state=seed)
    out = {x: "train" for x in train}
    out.update({x: "val" for x in val})
    out.update({x: "test" for x in test})
    return out


# NOTE: NOT used by the HVG2000 strategy. HVG2000 replaces `pca` with
# IdentityProjection (hvg2000_projection.py), so features remain 2000-dim
# with no PCA reduction. This function is only the generic/legacy path.
def fit_state_projection(expr: pd.DataFrame, train_cell_lines: list[str], n_components: int = 512) -> tuple[list[str], SimpleImputer, StandardScaler, PCA]:
    genes = expr.columns.astype(str).tolist()
    train_expr = expr.loc[train_cell_lines, genes].astype(np.float32)
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_expr))
    n = min(n_components, x_train.shape[0], x_train.shape[1])
    if n < 1:
        raise ValueError("Cannot fit PCA with no train samples or genes.")
    pca = PCA(n_components=n, random_state=0)
    pca.fit(x_train)
    return genes, imputer, scaler, pca


def project_expression(expr: pd.DataFrame, genes: list[str], imputer: SimpleImputer, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    aligned = expr.reindex(columns=genes, fill_value=0.0).astype(np.float32)
    return pca.transform(scaler.transform(imputer.transform(aligned)))


def fit_relative_reward(responses: pd.DataFrame) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    baseline = responses.groupby("DRUG_ID")["AUC"].median().astype(float).to_dict()
    values = {
        int(drug_id): np.sort(group["AUC"].astype(np.float32).to_numpy(copy=True))
        for drug_id, group in responses.groupby("DRUG_ID", observed=True)
    }
    return {int(k): float(v) for k, v in baseline.items()}, values


def fit_drug_percentile_reference(
    rows: pd.DataFrame,
    response_col: str = "AUC",
    drug_col: str = "DRUG_ID",
) -> dict[int, np.ndarray]:
    return {
        int(drug_id): np.sort(group[response_col].astype(np.float32).to_numpy(copy=True))
        for drug_id, group in rows.groupby(drug_col, observed=True)
    }


def relative_auc_percentile(auc: float, train_values: Sequence[float]) -> float:
    if len(train_values) == 0:
        return float("nan")
    arr = np.asarray(train_values, dtype=float)
    return float(1.0 - (np.searchsorted(arr, auc, side="right") / len(arr)))


def apply_relative_value(
    rows: pd.DataFrame,
    reference_by_drug: dict[int, Sequence[float]],
    *,
    response_col: str = "AUC",
    drug_col: str = "DRUG_ID",
) -> np.ndarray:
    response = rows[response_col].astype(np.float32).to_numpy(copy=False)
    drug_ids = rows[drug_col].astype(int).to_numpy(copy=False)
    relative_value = np.full((len(rows),), np.nan, dtype=np.float32)
    for drug_id, sorted_response in reference_by_drug.items():
        mask = drug_ids == int(drug_id)
        if not np.any(mask):
            continue
        arr = np.asarray(sorted_response, dtype=np.float32)
        relative_value[mask] = 1.0 - (np.searchsorted(arr, response[mask], side="right") / len(arr))
    return relative_value


def build_relative_targets(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    response_col: str = "AUC",
    drug_col: str = "DRUG_ID",
) -> pd.DataFrame:
    out = frame.copy()
    out["relative_value_train"] = np.nan
    out["relative_value_eval"] = np.nan
    train_rows = out.loc[out[split_col].eq("train")].copy()
    train_reference = fit_drug_percentile_reference(train_rows, response_col=response_col, drug_col=drug_col)
    if not train_rows.empty:
        train_mask = out[split_col].eq("train").to_numpy(copy=False)
        out.loc[train_mask, "relative_value_train"] = apply_relative_value(
            out.loc[train_mask],
            train_reference,
            response_col=response_col,
            drug_col=drug_col,
        )
    for split_name, split_rows in out.groupby(split_col, observed=True):
        split_reference = fit_drug_percentile_reference(split_rows, response_col=response_col, drug_col=drug_col)
        split_mask = out[split_col].eq(split_name).to_numpy(copy=False)
        out.loc[split_mask, "relative_value_eval"] = apply_relative_value(
            out.loc[split_mask],
            split_reference,
            response_col=response_col,
            drug_col=drug_col,
        )
    # Legacy compatibility alias. New code should use the explicit train/eval fields.
    out["relative_value"] = out["relative_value_eval"].astype(np.float32)
    return out


def build_drug_centered_targets(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    response_col: str = "AUC",
    drug_col: str = "DRUG_ID",
) -> pd.DataFrame:
    out = frame.copy()
    train_rows = out.loc[out[split_col].eq("train")]
    train_baseline = train_rows.groupby(drug_col, observed=True)[response_col].median().astype(float).to_dict()
    out["drug_centered_baseline_train"] = out[drug_col].map(train_baseline).astype(float)
    out["drug_centered_auc_train"] = np.nan
    train_mask = out[split_col].eq("train")
    out.loc[train_mask, "drug_centered_auc_train"] = (
        out.loc[train_mask, response_col].astype(float) - out.loc[train_mask, "drug_centered_baseline_train"].astype(float)
    )
    out["drug_centered_baseline_eval"] = np.nan
    out["drug_centered_auc_eval"] = np.nan
    for split_name, split_rows in out.groupby(split_col, observed=True):
        split_baseline = split_rows.groupby(drug_col, observed=True)[response_col].median().astype(float).to_dict()
        split_mask = out[split_col].eq(split_name)
        out.loc[split_mask, "drug_centered_baseline_eval"] = out.loc[split_mask, drug_col].map(split_baseline).astype(float)
        out.loc[split_mask, "drug_centered_auc_eval"] = (
            out.loc[split_mask, response_col].astype(float) - out.loc[split_mask, "drug_centered_baseline_eval"].astype(float)
        )
    return out


def build_cell_residual_targets(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    response_col: str = "AUC",
    cell_col: str = "SANGER_MODEL_ID",
) -> pd.DataFrame:
    out = frame.copy()
    out["cell_train_baseline"] = np.nan
    out["cell_residual_baseline_train"] = np.nan
    out["cell_residual_auc_train"] = np.nan
    out["cell_residual_baseline_eval"] = np.nan
    out["cell_residual_auc_eval"] = np.nan
    if split_col not in out.columns or response_col not in out.columns or cell_col not in out.columns:
        return out
    train_rows = out.loc[out[split_col].eq("train"), [cell_col, response_col]].copy()
    if train_rows.empty:
        return out
    train_baseline = train_rows.groupby(cell_col, observed=True)[response_col].median().astype(float).to_dict()
    baseline = out[cell_col].map(train_baseline).astype(float)
    out["cell_train_baseline"] = baseline
    out["cell_residual_baseline_train"] = baseline
    out["cell_residual_baseline_eval"] = baseline
    residual = out[response_col].astype(float) - baseline.astype(float)
    train_mask = out[split_col].eq("train")
    out.loc[train_mask, "cell_residual_auc_train"] = residual.loc[train_mask]
    out["cell_residual_auc_eval"] = residual
    return out


def build_cell_train_statistics(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    cell_col: str = "SANGER_MODEL_ID",
    response_col: str = "AUC",
) -> pd.DataFrame:
    train_rows = frame.loc[frame[split_col].eq("train"), [cell_col, response_col]].copy()
    if train_rows.empty:
        return pd.DataFrame(
            columns=[
                "cell_auc_train_mean",
                "cell_auc_train_median",
                "cell_centered_sensitivity_train",
            ]
        )
    group = train_rows.groupby(cell_col, observed=True)[response_col]
    stats = pd.DataFrame(
        {
            "cell_auc_train_mean": group.mean().astype(np.float32),
            "cell_auc_train_median": group.median().astype(np.float32),
        }
    )
    global_train_median = float(train_rows[response_col].median())
    stats["cell_centered_sensitivity_train"] = (global_train_median - stats["cell_auc_train_median"]).astype(np.float32)
    return stats


def add_relative_targets(responses: pd.DataFrame, train_values: dict[int, Sequence[float]], baseline: dict[int, float]) -> pd.DataFrame:
    frame = build_relative_targets(responses)
    frame["auc_delta_vs_train_median"] = frame["DRUG_ID"].map(baseline).astype(float) - frame["AUC"].astype(float)
    frame = build_drug_centered_targets(frame)
    frame = build_cell_residual_targets(frame)
    # Keep held-out rows even when their drug never appeared in train. Those rows
    # cannot support train-relative planning metrics, but they are still valid
    # for raw AUC regression/correlation evaluation on val/test.
    return frame


def _canonical_group_key(row: pd.Series | Any) -> str:
    for field in ("inchikey", "canonical_smiles", "smiles", "drug_key", "DRUG_NAME"):
        value = str(getattr(row, field, "")).strip()
        if value and value.lower() != "nan" and value != "<NA>":
            return value
    return f"drug_id::{int(getattr(row, 'DRUG_ID'))}"


def build_canonical_drug_table(drug_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    if drug_table.empty:
        empty = drug_table.copy()
        if "canonical_group_key" not in empty.columns:
            empty["canonical_group_key"] = pd.Series(dtype=str)
        if "canonical_group_size" not in empty.columns:
            empty["canonical_group_size"] = pd.Series(dtype=np.int32)
        if "canonical_source_drug_ids" not in empty.columns:
            empty["canonical_source_drug_ids"] = pd.Series(dtype=object)
        return empty, {}

    frame = drug_table.copy()
    frame["canonical_group_key"] = frame.apply(_canonical_group_key, axis=1)
    ordered_groups = [
        (group_key, group.sort_values("DRUG_ID", kind="stable"))
        for group_key, group in frame.groupby("canonical_group_key", sort=False, observed=True)
    ]
    ordered_groups.sort(key=lambda item: int(item[1]["DRUG_ID"].min()))
    rows: list[dict[str, Any]] = []
    mapping: dict[int, int] = {}
    for canonical_idx, (group_key, group) in enumerate(ordered_groups):
        rep = group.iloc[0].copy()
        source_ids = tuple(int(x) for x in group["DRUG_ID"].astype(int).tolist())
        rep["DRUG_ID"] = int(rep["DRUG_ID"])
        rep["canonical_drug_id"] = int(rep["DRUG_ID"])
        rep["canonical_group_key"] = str(group_key)
        rep["canonical_group_index"] = int(canonical_idx)
        rep["canonical_group_size"] = int(len(group))
        rep["canonical_source_drug_ids"] = source_ids
        if "fingerprint" in group.columns:
            rep["fingerprint"] = group.iloc[0]["fingerprint"].astype(np.float32, copy=True)
        if "prior" in group.columns and len(group):
            prior_stack = np.stack([np.asarray(x, dtype=np.float32) for x in group["prior"]], axis=0)
            rep["prior"] = np.max(prior_stack, axis=0).astype(np.float32, copy=False)
        if "prior_mask" in group.columns:
            rep["prior_mask"] = float(np.max(group["prior_mask"].astype(float).to_numpy()))
        rows.append(rep.to_dict())
        for drug_id in source_ids:
            mapping[int(drug_id)] = int(canonical_idx)
    canonical = pd.DataFrame(rows)
    if not canonical.empty and "canonical_group_index" in canonical.columns:
        canonical = canonical.sort_values("canonical_group_index", kind="stable").reset_index(drop=True)
    return canonical, mapping


def _load_rdkit() -> tuple[Any, Any, Any]:
    global _RDKIT_IMPORT_ERROR, _RDKIT_CHEM, _RDKIT_DATASTRUCTS
    if _RDKIT_CHEM is not None and _RDKIT_DATASTRUCTS is not None:
        from rdkit.Chem import rdFingerprintGenerator

        return _RDKIT_CHEM, _RDKIT_DATASTRUCTS, rdFingerprintGenerator
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except Exception as exc:
        _RDKIT_IMPORT_ERROR = exc
        raise RuntimeError(
            "RDKit is required for real Morgan fingerprints. The active environment "
            "could not import RDKit; fix the RDKit/NumPy environment instead of using "
            "hashed or random drug features."
        ) from exc
    _RDKIT_CHEM = Chem
    _RDKIT_DATASTRUCTS = DataStructs
    return Chem, DataStructs, rdFingerprintGenerator


def morgan_fp(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    Chem, DataStructs, rdFingerprintGenerator = _load_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    gen = _RDKIT_FP_GENERATORS.get(n_bits)
    if gen is None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        _RDKIT_FP_GENERATORS[n_bits] = gen
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    try:
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        # Some RDKit/Numpy combinations intermittently reject the direct conversion.
        # Fall back to a bitstring decode so fingerprint generation remains stable.
        bitstring = fp.ToBitString()
        arr[:] = np.fromiter((1.0 if ch == "1" else 0.0 for ch in bitstring), dtype=np.float32, count=n_bits)
    return arr


def canonicalize_smiles(smiles: str) -> tuple[str, str]:
    try:
        Chem, _, _ = _load_rdkit()
    except RuntimeError:
        return str(smiles), ""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "", ""
    return Chem.MolToSmiles(mol, canonical=True), Chem.MolToInchiKey(mol)


def build_drug_table(responses: pd.DataFrame, prior_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    cols = ["DRUG_ID", "DRUG_NAME", "drug_key", "smiles"]
    if "pubchem_cid" in responses.columns:
        cols.append("pubchem_cid")
    if "pubchem_title" in responses.columns:
        cols.append("pubchem_title")
    unique_drugs = responses[cols].drop_duplicates("DRUG_ID")
    prior_lookup = {
        str(drug_key): prior_matrix.loc[drug_key].to_numpy(dtype=np.float32, copy=True) for drug_key in prior_matrix.index
    }
    empty_prior = np.zeros((prior_matrix.shape[1],), dtype=np.float32)
    fingerprint_cache: dict[str, np.ndarray | None] = {}
    for row in unique_drugs.itertuples(index=False):
        smiles = str(row.smiles)
        fp = fingerprint_cache.get(smiles)
        if smiles not in fingerprint_cache:
            fp = morgan_fp(smiles)
            fingerprint_cache[smiles] = None if fp is None else np.ascontiguousarray(fp, dtype=np.float32)
        fp = fingerprint_cache[smiles]
        if fp is None:
            invalid.append({"DRUG_ID": row.DRUG_ID, "DRUG_NAME": row.DRUG_NAME, "smiles": row.smiles, "issue": "invalid_smiles"})
            continue
        canonical_smiles, inchikey = canonicalize_smiles(smiles)
        prior = prior_lookup.get(str(row.drug_key), empty_prior)
        records.append(
            {
                "DRUG_ID": int(row.DRUG_ID),
                "DRUG_NAME": row.DRUG_NAME,
                "drug_key": row.drug_key,
                "smiles": row.smiles,
                "pubchem_cid": getattr(row, "pubchem_cid", pd.NA),
                "pubchem_title": getattr(row, "pubchem_title", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "fingerprint": fp.copy(),
                "prior": prior.copy(),
                "prior_mask": float(str(row.drug_key) in prior_lookup),
            }
        )
    table = pd.DataFrame(records)
    return table, pd.DataFrame(invalid, columns=["DRUG_ID", "DRUG_NAME", "smiles", "issue"])
