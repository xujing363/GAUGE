from __future__ import annotations

import json
import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .gdsc_smiles import load_dataset_smiles_cache
from .utils import normalize_name
from .xlsx import read_xlsx_first_sheet


GDSC_RESPONSE_COLUMNS = ["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50", "AUC"]
PDX_RESPONSE_COLUMNS = ["cell_line_name", "pubchem_id", "drug_name", "AUC"]
BEATAML2_RESPONSE_COLUMNS = [
    "dbgap_subject_id",
    "dbgap_rnaseq_sample",
    "inhibitor",
    "type",
    "status",
    "paper_inclusion",
    "ic50",
    "auc",
]

# TCGA treatment annotations mix actual agents with procedure-level labels and
# placeholder categories. Those non-drug labels should not be treated as
# unmapped candidate drugs.
TCGA_NON_DRUG_TREATMENT_KEYS = {
    "3d conformal",
    "ablation",
    "ablation or embolization",
    "chemotherapy",
    "external beam",
    "gamma knife",
    "high dose",
    "implant",
    "implants",
    "internal",
    "minimally invasive",
    "nos",
    "not reported",
    "open",
    "pharmaceutical therapy",
    "pleurodesis",
    "radiation",
    "radiation therapy",
    "radiofrequency",
    "radioisotope",
    "resection",
    "srs",
    "stereotactic",
    "sterotactic",
    "surgery",
    "systemic",
    "therapy",
    "total androgen blockade",
    "transplant",
    "unknown",
    "whipple",
}

TCGA_NON_DRUG_TREATMENT_PATTERNS = (
    "beam",
    "embolization",
    "knife",
    "radiation",
    "resection",
    "surgical",
    "therapy",
    "transplant",
)

TCGA_ACTUAL_DRUG_ALIASES = {
    "nab paclitaxel": "Paclitaxel",
    "pegylated liposomal doxorubicin": "Doxorubicin",
    "pemetrexed disodium": "Pemetrexed",
    "vinorelbine tartrate": "Vinorelbine",
}


def _is_tcga_non_drug_treatment(key: str) -> bool:
    if not key:
        return False
    if key in TCGA_NON_DRUG_TREATMENT_KEYS:
        return True
    return any(pattern in key for pattern in TCGA_NON_DRUG_TREATMENT_PATTERNS)


def _stable_int_id(value: str) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _truthy_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def _prism_smiles_is_valid(smiles: str) -> bool:
    try:
        from .features import morgan_fp
    except Exception:
        return False
    return morgan_fp(smiles) is not None


def _clean_prism_smiles(raw_smiles: object) -> dict[str, object]:
    if pd.isna(raw_smiles):
        return {
            "smiles_raw": pd.NA,
            "smiles_clean": pd.NA,
            "smiles_clean_rule": "missing_smiles",
            "smiles_clean_parts": (),
        }
    raw_text = str(raw_smiles).strip()
    if not raw_text:
        return {
            "smiles_raw": raw_text,
            "smiles_clean": pd.NA,
            "smiles_clean_rule": "missing_smiles",
            "smiles_clean_parts": (),
        }
    parts = [part.strip() for part in raw_text.split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return {
            "smiles_raw": raw_text,
            "smiles_clean": pd.NA,
            "smiles_clean_rule": "missing_smiles",
            "smiles_clean_parts": (),
        }
    if len(set(parts)) == 1:
        clean_smiles = parts[0]
        return {
            "smiles_raw": raw_text,
            "smiles_clean": clean_smiles,
            "smiles_clean_rule": "single_fragment" if len(parts) == 1 else "dedup_repeated_fragments",
            "smiles_clean_parts": (clean_smiles,),
        }
    clean_smiles = "".join(parts)
    rule = "concatenated_fragments" if _prism_smiles_is_valid(clean_smiles) else "concatenated_fragments_invalid"
    return {
        "smiles_raw": raw_text,
        "smiles_clean": clean_smiles,
        "smiles_clean_rule": rule,
        "smiles_clean_parts": tuple(parts),
    }


def _load_prism_responses(path: str, auc_encoding: str = "latin1") -> pd.DataFrame:
    frame = pd.read_csv(path, encoding=auc_encoding)
    required = {"depmap_id", "name", "broad_id", "auc", "smiles", "passed_str_profiling"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PRISM response file missing required columns: {', '.join(missing)}")
    frame = frame.loc[frame["passed_str_profiling"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    frame = frame.loc[frame["depmap_id"].notna()].copy()
    frame["SANGER_MODEL_ID"] = frame["depmap_id"].astype(str)
    frame["DRUG_NAME"] = frame["name"].astype(str)
    frame["drug_key"] = frame["broad_id"].astype(str)
    frame["DRUG_ID"] = pd.factorize(frame["drug_key"], sort=True)[0].astype(int)
    frame["AUC"] = pd.to_numeric(frame["auc"], errors="coerce")
    frame["LN_IC50"] = pd.to_numeric(frame["ic50"], errors="coerce") if "ic50" in frame.columns else pd.NA
    smiles_clean = pd.DataFrame([_clean_prism_smiles(raw_smiles) for raw_smiles in frame["smiles"]], index=frame.index)
    frame = pd.concat([frame.drop(columns=["smiles"]), smiles_clean], axis=1)
    frame["smiles"] = frame["smiles_clean"]
    frame = frame.dropna(subset=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "AUC"]).copy()
    return frame


def read_gdsc_fitted(paths: list[Path], max_rows: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    per_file = None
    if max_rows is not None:
        # Over-read each source file to reduce ordering bias, then sample.
        per_file = max(5000, int(np.ceil((max_rows * 20) / max(1, len(paths)))))
    for path in paths:
        try:
            frame = pd.read_excel(path, usecols=GDSC_RESPONSE_COLUMNS, nrows=per_file)
        except ImportError:
            frame = read_xlsx_first_sheet(path, usecols=GDSC_RESPONSE_COLUMNS, nrows=per_file)
        frame = frame[GDSC_RESPONSE_COLUMNS].copy()
        frame["source_file"] = path.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["SANGER_MODEL_ID"] = out["SANGER_MODEL_ID"].astype(str)
    out["DRUG_ID"] = pd.to_numeric(out["DRUG_ID"], errors="coerce").astype("Int64")
    out["LN_IC50"] = pd.to_numeric(out["LN_IC50"], errors="coerce")
    out["AUC"] = pd.to_numeric(out["AUC"], errors="coerce")
    out = out.dropna(subset=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50", "AUC"])
    if max_rows is not None:
        n = min(max_rows, len(out))
        out = out.sample(n=n, random_state=0, replace=False)
    return out


def read_beataml2_fitted(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", usecols=BEATAML2_RESPONSE_COLUMNS)
    out = frame.copy()
    out = out.loc[_truthy_series(out["paper_inclusion"])].copy()
    out = out.loc[out["type"].astype(str).str.strip().str.lower().eq("single-agent")].copy()
    out["dbgap_subject_id"] = pd.to_numeric(out["dbgap_subject_id"], errors="coerce").astype("Int64")
    out["dbgap_rnaseq_sample"] = out["dbgap_rnaseq_sample"].astype(str).str.strip()
    out["inhibitor"] = out["inhibitor"].astype(str).str.strip()
    out["status"] = out["status"].astype(str).str.strip()
    out["type"] = out["type"].astype(str).str.strip()
    valid_sample = out["dbgap_rnaseq_sample"].notna() & ~out["dbgap_rnaseq_sample"].str.lower().isin({"", "nan", "none", "null"})
    valid_inhibitor = out["inhibitor"].notna() & ~out["inhibitor"].str.lower().isin({"", "nan", "none", "null"})
    out = out.loc[valid_sample & valid_inhibitor].copy()
    ic50 = pd.to_numeric(out["ic50"], errors="coerce")
    out["LN_IC50"] = ic50.where(ic50 > 0).map(lambda x: np.log(float(x)) if pd.notna(x) else pd.NA)
    out["AUC"] = pd.to_numeric(out["auc"], errors="coerce")
    out["SANGER_MODEL_ID"] = out["dbgap_rnaseq_sample"]
    out["DRUG_NAME"] = out["inhibitor"]
    out["drug_key"] = out["DRUG_NAME"].map(normalize_name)
    out["DRUG_ID"] = out["drug_key"].map(_stable_int_id).astype("Int64")
    out = out.dropna(subset=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50", "AUC"]).copy()
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=0, replace=False)
    out["source_file"] = Path(path).name
    return out


def read_ctrp_v1(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"cellosaurus_id", "drug_name", "pubchem_id", "AUC"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CTRP response file missing required columns: {', '.join(missing)}")
    out = frame.copy()
    out["cellosaurus_id"] = out["cellosaurus_id"].astype(str)
    out["pubchem_id"] = pd.to_numeric(out["pubchem_id"], errors="coerce").astype("Int64")
    out["drug_name"] = out["drug_name"].astype(str)
    out["AUC"] = pd.to_numeric(out["AUC"], errors="coerce")
    if "LN_IC50_curvecurator" in out.columns:
        out["LN_IC50"] = pd.to_numeric(out["LN_IC50_curvecurator"], errors="coerce")
    elif "IC50_curvecurator" in out.columns:
        out["LN_IC50"] = pd.to_numeric(out["IC50_curvecurator"], errors="coerce")
    else:
        out["LN_IC50"] = pd.NA
    out["SANGER_MODEL_ID"] = out["cellosaurus_id"]
    out["DRUG_ID"] = out["pubchem_id"]
    out["DRUG_NAME"] = out["drug_name"]
    out = out.dropna(subset=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "AUC"]).copy()
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=0, replace=False)
    return out


def read_pdx_bruna(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"cell_line_name", "pubchem_id", "drug_name", "AUC"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PDX response file missing required columns: {', '.join(missing)}")
    out = frame.copy()
    out["cell_line_name"] = out["cell_line_name"].astype(str)
    out["pubchem_id"] = pd.to_numeric(out["pubchem_id"], errors="coerce").astype("Int64")
    out["drug_name"] = out["drug_name"].astype(str)
    out["AUC"] = pd.to_numeric(out["AUC"], errors="coerce")
    out["LN_IC50"] = pd.NA
    out["SANGER_MODEL_ID"] = out["cell_line_name"]
    out["DRUG_ID"] = out["pubchem_id"]
    out["DRUG_NAME"] = out["drug_name"]
    out = out.dropna(subset=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "AUC"]).copy()
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=0, replace=False)
    return out


def load_gene_identifier_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, usecols=["gene_id", "hgnc_symbol"])
    df = df.dropna(subset=["gene_id", "hgnc_symbol"])
    return dict(zip(df["gene_id"].astype(str), df["hgnc_symbol"].astype(str)))


def load_gdsc_expression(path: Path, gene_identifier_path: Path) -> pd.DataFrame:
    gene_map = load_gene_identifier_map(gene_identifier_path)
    with path.open("r", newline="") as f:
        first_row = next(csv.reader(f))
    model_ids = first_row[3:]
    raw = pd.read_csv(path, skiprows=3)
    raw.columns = ["gene_symbol", "ensembl_gene_id", "gene_id"] + model_ids[: len(raw.columns) - 3]
    if not {"gene_symbol", "ensembl_gene_id", "gene_id"}.issubset(raw.columns[:3]):
        raise ValueError("Unexpected GDSC expression header; expected metadata rows then gene_symbol/ensembl_gene_id/gene_id.")
    raw["hgnc_symbol"] = raw["gene_id"].astype(str).map(gene_map).fillna(raw["gene_symbol"].astype(str))
    value_cols = [c for c in raw.columns if c not in {"gene_symbol", "ensembl_gene_id", "gene_id", "hgnc_symbol"}]
    expr = raw.set_index("hgnc_symbol")[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(expr.index).mean().T
    expr.index.name = "SANGER_MODEL_ID"
    return expr.astype(np.float32)


def load_beataml2_expression(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    required = {"stable_id", "display_label", "description", "biotype"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Unexpected BeatAML2 expression header; missing columns: {', '.join(missing)}")
    raw = raw.copy()
    raw["gene_name"] = raw["display_label"].fillna(raw["stable_id"]).astype(str).str.strip()
    raw["gene_name"] = raw["gene_name"].where(raw["gene_name"].astype(bool), raw["stable_id"].astype(str))
    value_cols = [c for c in raw.columns if c not in {"stable_id", "display_label", "description", "biotype", "gene_name"}]
    expr = raw.set_index("gene_name")[value_cols].apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(expr.index).mean().T
    expr.index.name = "SANGER_MODEL_ID"
    return expr.astype(np.float32)


def load_ctrp_expression(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"cellosaurus_id", "cell_line_name"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"CTRP gene expression file missing required columns: {', '.join(missing)}")
    expr = raw.copy()
    expr["cellosaurus_id"] = expr["cellosaurus_id"].astype(str)
    expr = expr.drop(columns=["cell_line_name"])
    expr = expr.set_index("cellosaurus_id")
    expr.index.name = "SANGER_MODEL_ID"
    expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr.astype(np.float32)


def load_pdx_expression(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"PDX expression file is empty: {path}")
    if "cell_line_name" not in raw.columns:
        raise ValueError("PDX expression file missing required column 'cell_line_name'.")
    expr = raw.copy()
    expr["cell_line_name"] = expr["cell_line_name"].astype(str)
    expr = expr.set_index("cell_line_name")
    expr.index.name = "SANGER_MODEL_ID"
    expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr.astype(np.float32)


def attach_smiles(responses: pd.DataFrame, cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = load_dataset_smiles_cache(cache_path)
    resolved = cache.loc[cache["status"].astype(str).eq("resolved")].copy()
    resolved_by_id = resolved.loc[resolved["DRUG_ID"].notna()].copy()
    smiles_by_id = dict(zip(resolved_by_id["DRUG_ID"].astype(int), resolved_by_id["smiles"].astype(str)))
    smiles_by_name = dict(zip(resolved["drug_key"].astype(str), resolved["smiles"].astype(str)))
    meta_cols = [c for c in ["pubchem_cid", "pubchem_title", "pubchem_query", "query_rank", "status"] if c in cache.columns]
    meta_by_id = {}
    meta_by_name = {}
    if meta_cols:
        cache_by_id = cache.loc[cache["DRUG_ID"].notna()].copy()
        cache_by_name = cache.copy()
        for row in cache_by_id[["DRUG_ID", *meta_cols]].itertuples(index=False):
            meta_by_id[int(row.DRUG_ID)] = {c: getattr(row, c) for c in meta_cols}
        for row in cache_by_name[["drug_key", *meta_cols]].itertuples(index=False):
            meta_by_name[str(row.drug_key)] = {c: getattr(row, c) for c in meta_cols}
    cache_by_id = cache.loc[cache["DRUG_ID"].notna()].copy()
    status_by_id = dict(zip(cache_by_id["DRUG_ID"].astype(int), cache_by_id["status"].astype(str)))
    status_by_name = dict(zip(cache["drug_key"].astype(str), cache["status"].astype(str)))
    frame = responses.copy()
    frame["drug_key"] = frame["DRUG_NAME"].map(normalize_name)
    frame["smiles"] = frame["DRUG_ID"].map(smiles_by_id).astype("object")
    frame.loc[frame["smiles"].isna(), "smiles"] = frame.loc[frame["smiles"].isna(), "drug_key"].map(smiles_by_name)
    for col in meta_cols:
        frame[col] = pd.NA
        if meta_by_id:
            frame[col] = frame["DRUG_ID"].map(lambda x: meta_by_id.get(int(x), {}).get(col) if pd.notna(x) else pd.NA)
        if meta_by_name:
            mask = frame[col].isna()
            if mask.any():
                frame.loc[mask, col] = frame.loc[mask, "drug_key"].map(lambda x: meta_by_name.get(str(x), {}).get(col) if pd.notna(x) else pd.NA)
    missing = frame.loc[frame["smiles"].isna(), ["DRUG_ID", "DRUG_NAME", "drug_key"]].drop_duplicates().copy()
    missing["status"] = missing["DRUG_ID"].map(status_by_id)
    missing.loc[missing["status"].isna(), "status"] = missing.loc[missing["status"].isna(), "drug_key"].map(status_by_name)
    missing["issue"] = missing["status"].fillna("missing_smiles").map(lambda x: f"status:{x}" if x != "missing_smiles" else x)
    audit = missing.drop(columns=["status"]).sort_values(["DRUG_NAME", "DRUG_ID"])
    return frame.dropna(subset=["smiles"]).copy(), audit


def attach_smiles_from_table(
    responses: pd.DataFrame,
    smiles_table: pd.DataFrame,
    *,
    id_col: str = "pubchem_id",
    name_col: str = "drug_name",
    smiles_col: str = "canonical_smiles",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = smiles_table.copy()
    if id_col not in table.columns or name_col not in table.columns or smiles_col not in table.columns:
        raise ValueError(f"Smiles table must contain {id_col!r}, {name_col!r}, and {smiles_col!r}")
    table = table.dropna(subset=[id_col, smiles_col]).copy()
    table[id_col] = pd.to_numeric(table[id_col], errors="coerce").astype("Int64")
    table = table.dropna(subset=[id_col]).copy()
    table[id_col] = table[id_col].astype(int)
    table[name_col] = table[name_col].astype(str)
    table["drug_key"] = table[name_col].map(normalize_name)
    table = table.drop_duplicates(subset=[id_col], keep="first")
    smiles_by_id = dict(zip(table[id_col].astype(int), table[smiles_col].astype(str)))
    smiles_by_name = dict(zip(table["drug_key"].astype(str), table[smiles_col].astype(str)))
    frame = responses.copy()
    frame["drug_key"] = frame["DRUG_NAME"].map(normalize_name)
    frame["smiles"] = frame["DRUG_ID"].map(smiles_by_id).astype("object")
    frame.loc[frame["smiles"].isna(), "smiles"] = frame.loc[frame["smiles"].isna(), "drug_key"].map(smiles_by_name)
    missing = frame.loc[frame["smiles"].isna(), ["DRUG_ID", "DRUG_NAME", "drug_key"]].drop_duplicates().copy()
    missing["issue"] = "missing_smiles"
    audit = missing.sort_values(["DRUG_NAME", "DRUG_ID"])
    return frame.dropna(subset=["smiles"]).copy(), audit


BLOCKED_PRIOR_TERMS = ("indication", "off-label", "therapeutic", "contraindication")


def _normalize_name_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    text = text.str.replace(r"\b(hydrochloride|mesylate|malate|sodium|potassium|phosphate|acetate|sulfate)\b", "", regex=True)
    text = text.str.replace(r"[^a-z0-9]+", " ", regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def load_primekg_prior(primekg_path: Path, drug_names: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = {normalize_name(x): str(x) for x in drug_names.dropna().unique()}
    rows: list[dict[str, Any]] = []
    chunks = pd.read_csv(
        primekg_path,
        usecols=["relation", "display_relation", "x_type", "x_name", "y_type", "y_name"],
        chunksize=250_000,
    )
    for chunk in chunks:
        rel_text = (chunk["relation"].astype(str) + " " + chunk["display_relation"].astype(str)).str.lower()
        allowed = ~rel_text.str.contains("|".join(BLOCKED_PRIOR_TERMS), regex=True, na=False)
        chunk = chunk.loc[allowed].copy()
        x_key = _normalize_name_series(chunk["x_name"])
        y_key = _normalize_name_series(chunk["y_name"])
        for side, keys, type_col, other_type_col in [
            ("x", x_key, "x_type", "y_type"),
            ("y", y_key, "y_type", "x_type"),
        ]:
            sub = chunk.loc[keys.isin(wanted)]
            if sub.empty:
                continue
            side_keys = keys.loc[sub.index]
            for key, relation, neighbor_type in zip(side_keys, sub["display_relation"].astype(str), sub[other_type_col].astype(str)):
                rows.append(
                    {
                        "drug_key": key,
                        "drug_name": wanted[key],
                        "feature": f"source=primekg|rel={relation}|neighbor={neighbor_type}",
                        "count": 1.0,
                    }
                )
    if not rows:
        audit = pd.DataFrame({"drug_key": list(wanted), "drug_name": list(wanted.values()), "issue": "missing_prior"})
        return pd.DataFrame(), audit
    long = pd.DataFrame(rows)
    matrix = long.pivot_table(index="drug_key", columns="feature", values="count", aggfunc="sum", fill_value=0.0)
    matrix = np.log1p(matrix).astype(np.float32)
    matrix = pd.DataFrame(matrix, index=long.pivot_table(index="drug_key", values="count", aggfunc="sum").index, columns=matrix.columns)
    missing = sorted(set(wanted) - set(matrix.index))
    audit = pd.DataFrame({"drug_key": missing, "drug_name": [wanted[k] for k in missing], "issue": "missing_prior"})
    return matrix, audit


def load_gdsc_screened_compounds_prior(
    screened_compounds_path: Path,
    responses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = responses[["DRUG_ID", "DRUG_NAME", "drug_key"]].drop_duplicates("DRUG_ID").copy()
    wanted["DRUG_ID"] = pd.to_numeric(wanted["DRUG_ID"], errors="coerce").astype("Int64")
    wanted = wanted.dropna(subset=["DRUG_ID", "drug_key", "DRUG_NAME"])
    wanted["DRUG_ID"] = wanted["DRUG_ID"].astype(int)
    screened = pd.read_csv(
        screened_compounds_path,
        usecols=["DRUG_ID", "DRUG_NAME", "SYNONYMS", "TARGET", "TARGET_PATHWAY"],
    )
    screened["DRUG_ID"] = pd.to_numeric(screened["DRUG_ID"], errors="coerce").astype("Int64")
    screened = screened.dropna(subset=["DRUG_ID"])
    screened["DRUG_ID"] = screened["DRUG_ID"].astype(int)
    merged = wanted.merge(screened, on="DRUG_ID", how="left", suffixes=("", "_screened"))
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        feature_terms: list[tuple[str, str]] = []
        for raw in _split_prior_terms(getattr(row, "TARGET", None)):
            feature_terms.append(("target", raw))
        for raw in _split_prior_terms(getattr(row, "TARGET_PATHWAY", None)):
            feature_terms.append(("pathway", raw))
        seen: set[str] = set()
        for kind, term in feature_terms:
            feature = f"source=gdsc_screened_compounds|{kind}={term}"
            if feature in seen:
                continue
            seen.add(feature)
            rows.append(
                {
                    "drug_key": str(row.drug_key),
                    "drug_name": str(row.DRUG_NAME),
                    "feature": feature,
                    "count": 1.0,
                }
            )
    if not rows:
        audit = pd.DataFrame(
            {
                "drug_key": wanted["drug_key"].astype(str).tolist(),
                "drug_name": wanted["DRUG_NAME"].astype(str).tolist(),
                "issue": "missing_screened_compounds_prior",
            }
        )
        return pd.DataFrame(), audit
    long = pd.DataFrame(rows)
    matrix = long.pivot_table(index="drug_key", columns="feature", values="count", aggfunc="sum", fill_value=0.0)
    matrix = np.log1p(matrix).astype(np.float32)
    matrix = pd.DataFrame(matrix, index=matrix.index.astype(str), columns=matrix.columns)
    missing = sorted(set(wanted["drug_key"].astype(str)) - set(matrix.index))
    audit = pd.DataFrame(
        {
            "drug_key": missing,
            "drug_name": [wanted.loc[wanted["drug_key"].astype(str).eq(k), "DRUG_NAME"].iloc[0] for k in missing],
            "issue": "missing_screened_compounds_prior",
        }
    )
    return matrix, audit


def load_beataml2_drug_family_prior(
    drug_families_path: Path,
    responses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = responses[["DRUG_ID", "DRUG_NAME", "drug_key"]].drop_duplicates("DRUG_ID").copy()
    wanted["DRUG_ID"] = pd.to_numeric(wanted["DRUG_ID"], errors="coerce").astype("Int64")
    wanted = wanted.dropna(subset=["DRUG_ID", "drug_key", "DRUG_NAME"])
    wanted["DRUG_ID"] = wanted["DRUG_ID"].astype(int)
    families = pd.read_excel(drug_families_path, sheet_name="drug_family", usecols=["inhibitor", "family", "manual_addition"])
    families["inhibitor"] = families["inhibitor"].astype(str).str.strip()
    families["family"] = families["family"].astype(str).str.strip()
    families = families.loc[families["inhibitor"].ne("") & families["family"].ne("")].copy()
    families["drug_key"] = families["inhibitor"].map(normalize_name)
    merged = wanted.merge(families[["drug_key", "inhibitor", "family"]], on="drug_key", how="left")
    rows: list[dict[str, Any]] = []
    for row in merged.dropna(subset=["family"]).itertuples(index=False):
        rows.append(
            {
                "drug_key": str(row.drug_key),
                "drug_name": str(row.DRUG_NAME),
                "feature": f"source=beataml2_drug_families|family={normalize_name(row.family)}",
                "count": 1.0,
            }
        )
    if not rows:
        audit = pd.DataFrame(
            {
                "drug_key": wanted["drug_key"].astype(str).tolist(),
                "drug_name": wanted["DRUG_NAME"].astype(str).tolist(),
                "issue": "missing_drug_family_prior",
            }
        )
        return pd.DataFrame(), audit
    long = pd.DataFrame(rows)
    matrix = long.pivot_table(index="drug_key", columns="feature", values="count", aggfunc="sum", fill_value=0.0)
    matrix = np.log1p(matrix).astype(np.float32)
    matrix = pd.DataFrame(matrix, index=matrix.index.astype(str), columns=matrix.columns)
    missing = sorted(set(wanted["drug_key"].astype(str)) - set(matrix.index))
    audit = pd.DataFrame(
        {
            "drug_key": missing,
            "drug_name": [wanted.loc[wanted["drug_key"].astype(str).eq(k), "DRUG_NAME"].iloc[0] for k in missing],
            "issue": "missing_drug_family_prior",
        }
    )
    return matrix, audit


def load_multisource_drug_prior(
    paths: Any,
    responses: pd.DataFrame,
    dataset_name: str = "gdsc",
    sources: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_name = str(dataset_name).strip().lower()
    second_source_name = "beataml2_drug_families" if dataset_name == "beataml2" else "gdsc_screened_compounds"
    requested_sources = {str(source) for source in sources} if sources is not None else {"primekg", second_source_name}
    primekg_matrix = pd.DataFrame()
    primekg_audit = pd.DataFrame(columns=["drug_key", "drug_name", "issue"])
    screened_matrix = pd.DataFrame()
    screened_audit = pd.DataFrame(columns=["drug_key", "drug_name", "issue"])
    if "primekg" in requested_sources:
        primekg_matrix, primekg_audit = load_primekg_prior(paths.primekg, responses["DRUG_NAME"])
    if second_source_name in requested_sources:
        if dataset_name == "beataml2":
            screened_matrix, screened_audit = load_beataml2_drug_family_prior(paths.beataml2_drug_families, responses)
        else:
            screened_matrix, screened_audit = load_gdsc_screened_compounds_prior(paths.gdsc_screened_compounds, responses)

    matrices: list[pd.DataFrame] = []
    source_stats = {
        "primekg": {
            "n_drugs": int(primekg_matrix.shape[0]),
            "n_features": int(primekg_matrix.shape[1]),
        },
        second_source_name: {
            "n_drugs": int(screened_matrix.shape[0]),
            "n_features": int(screened_matrix.shape[1]),
        },
    }
    if not primekg_matrix.empty:
        matrices.append(primekg_matrix)
    if not screened_matrix.empty:
        matrices.append(screened_matrix)

    if matrices:
        combined = pd.concat(matrices, axis=1).fillna(0.0)
        combined = combined.groupby(level=0).max()
        combined = combined.sort_index(axis=0).sort_index(axis=1).astype(np.float32)
    else:
        combined = pd.DataFrame(dtype=np.float32)

    wanted = {normalize_name(x): str(x) for x in responses["DRUG_NAME"].dropna().unique()}
    missing = sorted(set(wanted) - set(combined.index.astype(str)))
    audit = pd.DataFrame({"drug_key": missing, "drug_name": [wanted[k] for k in missing], "issue": "missing_prior"})
    source_stats["union"] = {
        "n_drugs": int(combined.shape[0]),
        "n_features": int(combined.shape[1]),
        "n_missing": int(len(audit)),
    }
    source_stats["source_audits"] = {
        "primekg_missing": int(len(primekg_audit)),
        f"{second_source_name}_missing": int(len(screened_audit)),
    }
    return combined, audit, source_stats


def _split_prior_terms(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r",|;|\+|\band\b", text, flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = normalize_name(part)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def expression_from_h5ad(
    h5ad_path: Path,
    target_genes: list[str],
    batch_size: int = 512,
    var_gene_name_col: str = "gene_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = ad.read_h5ad(h5ad_path, backed="r")
    gene_to_idx: dict[str, int] = {}
    if var_gene_name_col in data.var.columns:
        names = data.var[var_gene_name_col].astype(str).tolist()
    else:
        names = [str(x).split(".")[0] for x in data.var_names]
    for i, gene in enumerate(names):
        key = str(gene)
        if key and key not in gene_to_idx:
            gene_to_idx[key] = i
    present = [g for g in target_genes if g in gene_to_idx]
    missing = [g for g in target_genes if g not in gene_to_idx]
    idx = [gene_to_idx[g] for g in present]
    blocks = []
    for start in range(0, data.n_obs, batch_size):
        x = data[start : start + batch_size, idx].X
        if sparse.issparse(x):
            x = x.toarray()
        blocks.append(np.asarray(x, dtype=np.float32))
    arr = np.vstack(blocks) if blocks else np.empty((0, len(present)), dtype=np.float32)
    expr = pd.DataFrame(arr, index=data.obs_names.to_list(), columns=present)
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    for gene in missing:
        expr[gene] = 0.0
    expr = expr[target_genes]
    return expr, obs


def _h5ad_gene_names(data: ad.AnnData, var_gene_name_col: str = "gene_name") -> list[str]:
    if var_gene_name_col in data.var.columns:
        names = data.var[var_gene_name_col].astype(str).tolist()
    else:
        names = [str(x).split(".")[0] for x in data.var_names]
    gene_to_idx: dict[str, int] = {}
    genes: list[str] = []
    for gene in names:
        key = str(gene)
        if key and key not in gene_to_idx:
            gene_to_idx[key] = len(genes)
            genes.append(key)
    return genes


def h5ad_gene_list(h5ad_path: Path, var_gene_name_col: str = "gene_name") -> list[str]:
    data = ad.read_h5ad(h5ad_path, backed="r")
    return _h5ad_gene_names(data, var_gene_name_col=var_gene_name_col)


def parse_therapy_agents(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    agents: list[str] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("therapy", "")).lower() != "yes":
                continue
            raw_agents = item.get("agents") or item.get("agent") or item.get("drug") or item.get("type")
            if isinstance(raw_agents, list):
                agents.extend(str(a) for a in raw_agents)
            elif raw_agents:
                agents.extend(split_drug_list(raw_agents))
    return agents


def split_drug_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    text = re.sub(r"\b[A-Z0-9]{2,}\s*\(", "(", text)
    parts = re.split(r"[;+/,]|\band\b", text)
    out: list[str] = []
    for part in parts:
        cleaned = re.sub(r"[()]", " ", part).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned and len(cleaned) > 1:
            out.append(cleaned)
    return out


def tcga_actual_drugs(obs: pd.DataFrame) -> pd.Series:
    result = []
    for _, row in obs.iterrows():
        drugs: list[str] = []
        for col in ["drug", "drug2"]:
            if col in obs.columns:
                drugs.extend(split_drug_list(row.get(col)))
        if "therapies" in obs.columns:
            drugs.extend(parse_therapy_agents(row.get("therapies")))
        dedup = []
        seen = set()
        for d in drugs:
            key = normalize_name(d)
            if not key or key in seen or _is_tcga_non_drug_treatment(key):
                continue
            canonical = TCGA_ACTUAL_DRUG_ALIASES.get(key, str(d).strip())
            canonical_key = normalize_name(canonical)
            if not canonical_key or canonical_key in seen:
                continue
            seen.add(canonical_key)
            dedup.append(canonical)
        result.append(dedup)
    return pd.Series(result, index=obs.index)


TCGA_BINARY_POSITIVE_RESPONSES = {"complete response", "partial response"}
TCGA_BINARY_NEGATIVE_RESPONSES = {
    "progressive disease",
    "stable disease",
    "persistent disease",
    "no response",
}
TCGA_BINARY_ALLOWED_RESPONSES = TCGA_BINARY_POSITIVE_RESPONSES | TCGA_BINARY_NEGATIVE_RESPONSES
TCGA_BINARY_EXCLUDED_RESPONSES = {
    "unknown",
    "not reported",
    "treatment ongoing",
    "no measurable disease",
    "treatment stopped due to toxicity",
}


def parse_tcga_episode_agents(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        text = str(value)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            raw_items = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                agents = item.get("agents") or item.get("agent") or item.get("drug") or ""
                if isinstance(agents, list):
                    raw_items.extend(str(a) for a in agents)
                elif agents:
                    raw_items.extend(split_drug_list(agents))
            return [x for x in raw_items if normalize_name(x)]
        return split_drug_list(text)
    out: list[str] = []
    for item in raw_items:
        out.extend(split_drug_list(item))
    return [x for x in out if normalize_name(x)]


def tcga_binary_episode_frame(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for sample_id, row in obs.iterrows():
        patient_id = str(row.get("case_submitter_id") or sample_id)
        project_id = str(row.get("project_id", "unknown"))
        therapy_payload = row.get("therapies")
        try:
            therapies = json.loads(therapy_payload) if isinstance(therapy_payload, str) and therapy_payload not in ("", "nan") else []
        except json.JSONDecodeError:
            therapies = []
            audit_rows.append(
                {
                    "entity_id": patient_id,
                    "sample_id": sample_id,
                    "issue": "invalid_therapy_json",
                }
            )
        episode_idx = 0
        for item in therapies:
            if not isinstance(item, dict):
                continue
            if str(item.get("therapy", "")).lower() != "yes":
                continue
            response = str(item.get("response", "")).strip().lower()
            if response in TCGA_BINARY_POSITIVE_RESPONSES:
                y = 1
            elif response in TCGA_BINARY_NEGATIVE_RESPONSES:
                y = 0
            else:
                audit_rows.append(
                    {
                        "entity_id": patient_id,
                        "sample_id": sample_id,
                        "response": response or "",
                        "issue": "excluded_response",
                    }
                )
                continue
            agents = parse_tcga_episode_agents(item.get("agents") or item.get("agent") or item.get("drug") or "")
            if len(agents) != 1:
                audit_rows.append(
                    {
                        "entity_id": patient_id,
                        "sample_id": sample_id,
                        "response": response,
                        "issue": "excluded_non_single_agent_episode",
                    }
                )
                continue
            drug_name = str(agents[0]).strip()
            drug_key = normalize_name(drug_name)
            if not drug_key:
                audit_rows.append(
                    {
                        "entity_id": patient_id,
                        "sample_id": sample_id,
                        "response": response,
                        "issue": "excluded_empty_drug_name",
                    }
                )
                continue
            rows.append(
                {
                    "episode_id": f"{sample_id}::{episode_idx}",
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "project_id": project_id,
                    "drug_name": drug_name,
                    "drug_key": drug_key,
                    "response": response,
                    "y": int(y),
                }
            )
            episode_idx += 1
    frame = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows, columns=["entity_id", "sample_id", "response", "issue"])
    return frame, audit


def tcga_binary_label_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["split", "n", "n_pos", "n_neg", "pos_rate"])
    return (
        frame.groupby("split", observed=True)
        .agg(n=("y", "size"), n_pos=("y", "sum"))
        .reset_index()
        .assign(n_neg=lambda x: x["n"] - x["n_pos"], pos_rate=lambda x: x["n_pos"] / x["n"])
    )


def select_hvg_from_h5ad(
    h5ad_path: Path,
    obs_names: list[str],
    n_hvg: int = 3000,
    var_gene_name_col: str = "gene_name",
    batch_size: int = 256,
) -> list[str]:
    data = ad.read_h5ad(h5ad_path, backed="r")
    genes = _h5ad_gene_names(data, var_gene_name_col=var_gene_name_col)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    idx = [gene_to_idx[gene] for gene in genes]
    obs_index = {str(name): i for i, name in enumerate(data.obs_names.to_list())}
    rows = [obs_index[name] for name in obs_names if name in obs_index]
    if not rows:
        raise ValueError("No overlapping observation names found while selecting TCGA HVGs.")
    n_genes = len(genes)
    sum_x = np.zeros((n_genes,), dtype=np.float64)
    sum_x2 = np.zeros((n_genes,), dtype=np.float64)
    n_seen = 0
    from scipy import sparse as _sparse

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        x = data[batch_rows, :].X
        if _sparse.issparse(x):
            x = x.toarray()
        arr = np.asarray(x, dtype=np.float32).astype(np.float64, copy=False)[:, idx]
        sum_x += arr.sum(axis=0)
        sum_x2 += (arr * arr).sum(axis=0)
        n_seen += arr.shape[0]
    if n_seen == 0:
        raise ValueError("No rows available while selecting TCGA HVGs.")
    mean = sum_x / n_seen
    var = sum_x2 / n_seen - mean * mean
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    top_n = min(int(n_hvg), len(genes))
    order = np.argsort(var)[::-1][:top_n]
    return [genes[i] for i in order]


def tcga_os_frame(obs: pd.DataFrame) -> pd.DataFrame:
    vital = obs["vital_status"].astype(str)
    death = pd.to_numeric(obs.get("days_to_death"), errors="coerce")
    follow = pd.to_numeric(obs.get("days_to_last_follow_up"), errors="coerce")
    time = death.where(death.notna(), follow)
    frame = pd.DataFrame(index=obs.index)
    frame["event"] = vital.eq("Dead").astype(int)
    frame["time"] = time
    age_col = "age_at_diagnosis" if "age_at_diagnosis" in obs.columns else "age_at_diagnosis_days"
    if age_col in obs.columns:
        frame["age_at_diagnosis"] = pd.to_numeric(obs[age_col], errors="coerce") / (365.25 if age_col.endswith("_days") else 1.0)
    else:
        frame["age_at_diagnosis"] = np.nan
    frame["project_id"] = obs.get("project_id", "unknown").astype(str)
    frame = frame.loc[frame["time"].notna() & frame["time"].ge(0) & vital.isin(["Alive", "Dead"])]
    return frame
