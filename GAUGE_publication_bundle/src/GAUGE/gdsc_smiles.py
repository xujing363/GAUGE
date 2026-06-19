from __future__ import annotations

import csv
import http.client
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import anndata as ad
import pandas as pd

from .config import Paths
from .utils import normalize_name

SUPPORTED_DATASETS = ("gdsc", "beataml2", "ctrdb", "tcga", "pdx")
CACHE_COLUMNS = [
    "dataset",
    "DRUG_ID",
    "DRUG_NAME",
    "drug_key",
    "smiles",
    "pubchem_cid",
    "pubchem_title",
    "pubchem_query",
    "query_rank",
    "status",
]

PUBCHEM_PROPERTY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/ConnectivitySMILES,Title/JSON"
PUBCHEM_CID_SYNONYMS_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
CHEMBL_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={query}&limit=20"
NON_SMALL_MOLECULE_PATTERNS = {
    "tcga": [
        "ablation",
        "embolization",
        "biopsy",
        "brachytherapy",
        "clinical trial",
        "chemotherapy",
        "radiation",
        "cyberknife",
        "cryo",
        "pancreatectomy",
        "resection",
        "implant",
        "solution",
        "injection",
        "ancillary treatment",
        "vaccine",
        "immunotoxin",
        "anticalin",
        "conformal",
        "autologous",
        "antibody",
        "monoclonal",
        "peptide",
        "vaccine",
        "trial",
        "combination",
        "inhibitor mk",
        "surgery",
        "therapy",
        "external beam",
        "gamma knife",
        "excisional",
        "high dose",
        "hormone analog",
        "antiseizure",
        "not reported",
        "low dose",
        "internal",
        "microwave",
        "minimally invasive",
        "organ transplantation",
        "isolated limb perfusion",
        "oncolytic",
        "virus",
        "immunostimulator",
        "pan vegfr",
        "aurora kinase",
        "extract",
    ],
    "ctrdb": [
        "anthracycline",
        "glucocorticoids",
        "chop-like",
        "asparaginase",
        "chop like",
        "escherichia coli",
        "vaccine",
        "as02b",
    ],
}
BIOLOGIC_PATTERNS = (
    "mab",
    "zumab",
    "ximab",
    "momab",
    "cept",
    "antibody",
    "pegfilgrastim",
    "aldesleukin",
    "asparaginase",
    "interferon",
    "interleukin",
    "filgrastim",
    "trebananib",
)
MANUAL_QUERY_ALIASES = {
    "gdsc": {
        "ascorbate vitamin c": ["vitamin C", "ascorbic acid", "ascorbate"],
    },
    "ctrdb": {
        "recmage a3": ["MAGE-A3"],
        "trastuzumab emtansine": ["trastuzumab emtansine", "ado-trastuzumab emtansine"],
    },
    "tcga": {
        "recmage a3": ["MAGE-A3"],
        "mage a3 peptide vaccine": ["MAGE-A3"],
    },
}


class GdscSmilesCacheError(RuntimeError):
    pass


def dataset_smiles_path(paths: Paths, dataset: str) -> Path:
    dataset = str(dataset).strip().lower()
    if dataset == "gdsc":
        return paths.gdsc_smiles_cache
    if dataset == "beataml2":
        return paths.beataml2_smiles_cache
    if dataset == "ctrdb":
        return paths.ctrdb_smiles_cache
    if dataset == "tcga":
        return paths.tcga_smiles_cache
    if dataset == "pdx":
        return paths.dataset_smiles_dir / "pdx_smiles.csv"
    raise GdscSmilesCacheError(f"Unsupported dataset {dataset!r}. Expected one of {SUPPORTED_DATASETS}.")


def build_all_dataset_smiles_caches(
    paths: Paths,
    overwrite: bool = False,
    request_delay_seconds: float = 0.0,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    query_cache: dict[str, dict[str, object] | None] = {}
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for dataset in SUPPORTED_DATASETS:
        results[dataset] = build_dataset_smiles_cache(
            paths,
            dataset=dataset,
            overwrite=overwrite,
            request_delay_seconds=request_delay_seconds,
            query_cache=query_cache,
        )
    return results


def build_gdsc_smiles_cache(
    paths: Paths,
    dataset: str = "gdsc",
    overwrite: bool = False,
    request_delay_seconds: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible GDSC cache builder kept for older tests/scripts."""
    dataset = str(dataset).strip().lower()
    if dataset != "gdsc":
        return build_dataset_smiles_cache(paths, dataset=dataset, overwrite=overwrite, request_delay_seconds=request_delay_seconds)
    out_path = Path(dataset_smiles_path(paths, "gdsc"))
    universe = read_gdsc_drug_universe(paths)
    existing = _load_existing_dataset_cache(out_path, "gdsc") if out_path.exists() and not overwrite else pd.DataFrame(columns=CACHE_COLUMNS)
    existing_by_key = {_cache_row_key(row.DRUG_ID, row.drug_key): row._asdict() for row in existing.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for row in universe.itertuples(index=False):
        row_key = _cache_row_key(row.DRUG_ID, row.drug_key)
        if row_key in existing_by_key:
            rows.append(existing_by_key[row_key])
            continue
        result = _resolve_pubchem_smiles(str(row.DRUG_NAME))
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        if result is None:
            unresolved.append(
                {
                    "dataset": "gdsc",
                    "DRUG_ID": int(row.DRUG_ID) if pd.notna(row.DRUG_ID) else pd.NA,
                    "DRUG_NAME": str(row.DRUG_NAME),
                    "drug_key": str(row.drug_key),
                    "issue": "pubchem_not_found",
                }
            )
            continue
        rows.append(
            {
                "dataset": "gdsc",
                "DRUG_ID": int(row.DRUG_ID) if pd.notna(row.DRUG_ID) else pd.NA,
                "DRUG_NAME": str(row.DRUG_NAME),
                "drug_key": str(row.drug_key),
                "smiles": result["smiles"],
                "pubchem_cid": int(result["cid"]),
                "pubchem_title": result["title"],
                "pubchem_query": result["query"],
                "query_rank": int(result["query_rank"]),
                "status": "resolved",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    resolved.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    unresolved_df = pd.DataFrame(unresolved, columns=["dataset", "DRUG_ID", "DRUG_NAME", "drug_key", "issue"])
    report_path = out_path.with_name(f"{out_path.stem}_unresolved.csv")
    if not unresolved_df.empty:
        unresolved_df.to_csv(report_path, index=False)
        raise GdscSmilesCacheError(f"PubChem did not resolve {len(unresolved_df)} gdsc drugs. See {report_path}.")
    if report_path.exists():
        report_path.unlink()
    return resolved, unresolved_df


def build_dataset_smiles_cache(
    paths: Paths,
    dataset: str,
    out_path: Path | None = None,
    overwrite: bool = False,
    request_delay_seconds: float = 0.0,
    query_cache: dict[str, dict[str, object] | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = str(dataset).strip().lower()
    out_path = Path(out_path or dataset_smiles_path(paths, dataset))
    if dataset == "pdx":
        return _build_pdx_smiles_cache(paths, out_path)
    query_cache = query_cache if query_cache is not None else {}
    universe = read_dataset_drug_universe(paths, dataset)
    existing = _load_existing_dataset_cache(out_path, dataset) if out_path.exists() and not overwrite else pd.DataFrame(columns=CACHE_COLUMNS)
    existing_by_key = {_cache_row_key(row.DRUG_ID, row.drug_key): row._asdict() for row in existing.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for row in universe.itertuples(index=False):
        cache_key = _cache_row_key(row.DRUG_ID, row.drug_key)
        existing_row = existing_by_key.get(cache_key)
        if existing_row is not None:
            rows.append(existing_row)
            continue
        classification = _classify_dataset_entry(dataset, str(row.DRUG_NAME))
        if classification is not None:
            rows.append(
                {
                    "dataset": dataset,
                    "DRUG_ID": int(row.DRUG_ID) if pd.notna(row.DRUG_ID) else pd.NA,
                    "DRUG_NAME": str(row.DRUG_NAME),
                    "drug_key": str(row.drug_key),
                    "smiles": "",
                    "pubchem_cid": pd.NA,
                    "pubchem_title": "",
                    "pubchem_query": "",
                    "query_rank": pd.NA,
                    "status": classification,
                }
            )
            continue
        result = None
        queries = list(row.query_names) if hasattr(row, "query_names") else [str(row.DRUG_NAME)]
        for query in queries:
            cache_key = str(query)
            if cache_key not in query_cache:
                query_cache[cache_key] = _resolve_smiles(cache_key)
                if request_delay_seconds > 0:
                    time.sleep(request_delay_seconds)
            result = query_cache[cache_key]
            if result is not None:
                break
        if result is None:
            unresolved.append(
                {
                    "dataset": dataset,
                    "DRUG_ID": int(row.DRUG_ID) if pd.notna(row.DRUG_ID) else pd.NA,
                    "DRUG_NAME": str(row.DRUG_NAME),
                    "drug_key": str(row.drug_key),
                    "issue": "pubchem_chembl_not_found",
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "DRUG_ID": int(row.DRUG_ID) if pd.notna(row.DRUG_ID) else pd.NA,
                    "DRUG_NAME": str(row.DRUG_NAME),
                    "drug_key": str(row.drug_key),
                    "smiles": result["smiles"],
                    "pubchem_cid": int(result["cid"]),
                    "pubchem_title": result["title"],
                    "pubchem_query": result["query"],
                    "query_rank": int(result["query_rank"]),
                    "status": "resolved",
                }
            )
    resolved = pd.DataFrame(rows, columns=CACHE_COLUMNS).drop_duplicates(subset=["dataset", "DRUG_ID", "drug_key"], keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    unresolved_df = pd.DataFrame(unresolved, columns=["dataset", "DRUG_ID", "DRUG_NAME", "drug_key", "issue"])
    if not unresolved_df.empty:
        report_path = out_path.with_name(f"{out_path.stem}_unresolved.csv")
        unresolved_df.to_csv(report_path, index=False)
        if dataset != "beataml2":
            raise GdscSmilesCacheError(
                f"PubChem/ChEMBL did not resolve {len(unresolved_df)} {dataset} drugs. See {report_path} and fix before prepare/run."
            )
    report_path = out_path.with_name(f"{out_path.stem}_unresolved.csv")
    if unresolved_df.empty and report_path.exists():
        report_path.unlink()
    return resolved, unresolved_df


def _build_pdx_smiles_cache(paths: Paths, out_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(paths.pdx_drug_smiles, usecols=["pubchem_id", "drug_name", "canonical_smiles"])
    required = {"pubchem_id", "drug_name", "canonical_smiles"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise GdscSmilesCacheError(f"PDX smiles table missing required columns: {missing}")
    table = table.dropna(subset=["pubchem_id", "drug_name", "canonical_smiles"]).copy()
    table["pubchem_id"] = pd.to_numeric(table["pubchem_id"], errors="coerce").astype("Int64")
    table = table.dropna(subset=["pubchem_id"]).copy()
    table["pubchem_id"] = table["pubchem_id"].astype(int)
    table["drug_name"] = table["drug_name"].astype(str).str.strip()
    table["drug_key"] = table["drug_name"].map(normalize_name)
    resolved = pd.DataFrame(
        {
            "dataset": "pdx",
            "DRUG_ID": table["pubchem_id"],
            "DRUG_NAME": table["drug_name"],
            "drug_key": table["drug_key"],
            "smiles": table["canonical_smiles"].astype(str),
            "pubchem_cid": pd.NA,
            "pubchem_title": "",
            "pubchem_query": "",
            "query_rank": pd.NA,
            "status": "resolved",
        }
    ).drop_duplicates(subset=["dataset", "DRUG_ID", "drug_key"], keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    unresolved_df = pd.DataFrame(columns=["dataset", "DRUG_ID", "DRUG_NAME", "drug_key", "issue"])
    report_path = out_path.with_name(f"{out_path.stem}_unresolved.csv")
    if report_path.exists():
        report_path.unlink()
    return resolved, unresolved_df


def read_dataset_drug_universe(paths: Paths, dataset: str) -> pd.DataFrame:
    dataset = str(dataset).strip().lower()
    if dataset == "gdsc":
        return read_gdsc_drug_universe(paths)
    if dataset == "beataml2":
        return read_beataml2_drug_universe(paths)
    if dataset == "ctrdb":
        return read_ctrdb_drug_universe(paths)
    if dataset == "tcga":
        return read_tcga_drug_universe(paths)
    if dataset == "pdx":
        return read_pdx_drug_universe(paths)
    raise GdscSmilesCacheError(f"Unsupported dataset {dataset!r}. Expected one of {SUPPORTED_DATASETS}.")


def read_gdsc_drug_universe(paths: Paths) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [paths.gdsc_fitted_1, paths.gdsc_fitted_2]:
        frame = pd.read_excel(path, usecols=["DRUG_ID", "DRUG_NAME"])
        frames.append(frame[["DRUG_ID", "DRUG_NAME"]].dropna())
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out["DRUG_ID"] = pd.to_numeric(out["DRUG_ID"], errors="raise").astype(int)
    out["DRUG_NAME"] = out["DRUG_NAME"].astype(str).str.strip()
    out["drug_key"] = out["DRUG_NAME"].map(normalize_name)
    alias_map = _gdsc_query_aliases(paths)
    out["query_names"] = out.apply(
        lambda row: _merge_query_names(
            "gdsc",
            str(row["DRUG_NAME"]),
            alias_map.get(int(row["DRUG_ID"]), [str(row["DRUG_NAME"])]),
        ),
        axis=1,
    )
    out = out.sort_values(["DRUG_ID", "DRUG_NAME"]).reset_index(drop=True)
    return out


def read_beataml2_drug_universe(paths: Paths) -> pd.DataFrame:
    from .data import read_beataml2_fitted

    frame = read_beataml2_fitted(paths.beataml2_curve_fits)
    out = frame[["DRUG_ID", "DRUG_NAME", "drug_key"]].drop_duplicates().copy()
    out["DRUG_ID"] = pd.to_numeric(out["DRUG_ID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["DRUG_ID", "DRUG_NAME", "drug_key"])
    out["DRUG_ID"] = out["DRUG_ID"].astype(int)
    out["query_names"] = out["DRUG_NAME"].map(lambda x: _merge_query_names("beataml2", str(x), [str(x)]))
    out = out.sort_values(["DRUG_ID", "DRUG_NAME"]).reset_index(drop=True)
    return out


def read_ctrdb_drug_universe(paths: Paths) -> pd.DataFrame:
    from .data import split_drug_list

    data = ad.read_h5ad(paths.ctrdb_microarray_h5ad, backed="r")
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    if hasattr(data, "file") and data.file is not None:
        data.file.close()
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in obs.get("Drug_list", pd.Series(index=obs.index, dtype=object)).tolist():
        for drug_name in split_drug_list(raw):
            key = normalize_name(drug_name)
            if key and key not in seen:
                seen.add(key)
                records.append(
                    {
                        "DRUG_ID": pd.NA,
                        "DRUG_NAME": drug_name,
                        "drug_key": key,
                        "query_names": _merge_query_names("ctrdb", drug_name, [drug_name]),
                    }
                )
    return pd.DataFrame(records, columns=["DRUG_ID", "DRUG_NAME", "drug_key", "query_names"]).sort_values("DRUG_NAME").reset_index(drop=True)


def read_tcga_drug_universe(paths: Paths) -> pd.DataFrame:
    from .data import tcga_actual_drugs

    data = ad.read_h5ad(paths.tcga_h5ad, backed="r")
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    if hasattr(data, "file") and data.file is not None:
        data.file.close()
    series = tcga_actual_drugs(obs)
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for drugs in series.tolist():
        for drug_name in drugs:
            key = normalize_name(drug_name)
            if key and key not in seen:
                seen.add(key)
                records.append(
                    {
                        "DRUG_ID": pd.NA,
                        "DRUG_NAME": drug_name,
                        "drug_key": key,
                        "query_names": _merge_query_names("tcga", drug_name, [drug_name]),
                    }
                )
    return pd.DataFrame(records, columns=["DRUG_ID", "DRUG_NAME", "drug_key", "query_names"]).sort_values("DRUG_NAME").reset_index(drop=True)


def read_pdx_drug_universe(paths: Paths) -> pd.DataFrame:
    table = pd.read_csv(paths.pdx_drug_smiles, usecols=["pubchem_id", "drug_name"])
    table["pubchem_id"] = pd.to_numeric(table["pubchem_id"], errors="coerce").astype("Int64")
    table = table.dropna(subset=["pubchem_id", "drug_name"]).copy()
    table["pubchem_id"] = table["pubchem_id"].astype(int)
    table["drug_name"] = table["drug_name"].astype(str).str.strip()
    table["drug_key"] = table["drug_name"].map(normalize_name)
    table["query_names"] = table["drug_name"].map(lambda x: _merge_query_names("pdx", x, [x]))
    table = table.rename(columns={"pubchem_id": "DRUG_ID", "drug_name": "DRUG_NAME"})
    return table[["DRUG_ID", "DRUG_NAME", "drug_key", "query_names"]].sort_values("DRUG_NAME").reset_index(drop=True)


def load_dataset_smiles_cache(path: Path, expected_dataset: str | None = None) -> pd.DataFrame:
    cache_path = Path(path)
    if not cache_path.exists():
        raise GdscSmilesCacheError(
            f"Missing dataset SMILES cache: {cache_path}. Build it first with `python -m GAUGE build-dataset-smiles-cache --dataset all`."
        )
    frame = pd.read_csv(cache_path)
    missing_cols = [col for col in CACHE_COLUMNS if col not in frame.columns]
    if missing_cols:
        raise GdscSmilesCacheError(f"Dataset SMILES cache is missing required columns: {missing_cols}")
    frame = frame[CACHE_COLUMNS].copy()
    datasets = frame["dataset"].astype(str).unique().tolist()
    if expected_dataset is not None:
        if frame["dataset"].astype(str).nunique() != 1 or frame["dataset"].astype(str).iloc[0] != expected_dataset:
            raise GdscSmilesCacheError(f"Expected a {expected_dataset} dataset file at {cache_path}, found {datasets}")
    elif any(dataset not in SUPPORTED_DATASETS for dataset in datasets):
        raise GdscSmilesCacheError(f"Expected a supported dataset file at {cache_path}, found {datasets}")
    non_null_drug_ids = frame["DRUG_ID"].notna()
    if frame.loc[non_null_drug_ids, "DRUG_ID"].duplicated().any():
        dupes = frame.loc[non_null_drug_ids & frame["DRUG_ID"].duplicated(keep=False), ["DRUG_ID", "DRUG_NAME"]]
        raise GdscSmilesCacheError(f"Dataset SMILES cache has duplicate DRUG_ID rows:\n{dupes.head(10).to_string(index=False)}")
    return frame


def load_gdsc_smiles_cache(path: Path) -> pd.DataFrame:
    return load_dataset_smiles_cache(path, expected_dataset="gdsc")


def _load_existing_dataset_cache(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing_cols = [col for col in CACHE_COLUMNS if col not in frame.columns]
    if missing_cols:
        raise GdscSmilesCacheError(f"Existing dataset SMILES cache is missing required columns: {missing_cols}")
    frame = frame[CACHE_COLUMNS].copy()
    datasets = frame["dataset"].astype(str).unique().tolist()
    if datasets != [dataset]:
        raise GdscSmilesCacheError(f"Expected dataset cache {path} to contain only {dataset}, found {datasets}")
    return frame


def _cache_row_key(drug_id: object, drug_key: object) -> tuple[str, str]:
    if pd.notna(drug_id):
        return ("drug_id", str(int(drug_id)))
    return ("drug_key", str(drug_key))


def _resolve_smiles(drug_name: str) -> dict[str, object] | None:
    for rank, query in enumerate(_query_variants(drug_name), start=1):
        pubchem = _pubchem_name_lookup(query)
        if pubchem is not None and _result_matches_query(query, pubchem):
            pubchem["query_rank"] = rank
            return pubchem
        chembl = _chembl_name_lookup(query)
        if chembl is not None and _chembl_result_matches_query(query, chembl):
            chembl["query_rank"] = rank
            return chembl
    return None


def _resolve_pubchem_smiles(drug_name: str) -> dict[str, object] | None:
    # Backward-compatible helper used by ad-hoc diagnostics.
    for rank, query in enumerate(_query_variants(drug_name), start=1):
        result = _pubchem_name_lookup(query)
        if result is not None and _result_matches_query(query, result):
            result["query_rank"] = rank
            return result
    return None


def _gdsc_query_aliases(paths: Paths) -> dict[int, list[str]]:
    df = pd.read_csv(paths.gdsc_screened_compounds, usecols=["DRUG_ID", "DRUG_NAME", "SYNONYMS"])
    aliases: dict[int, list[str]] = {}
    for row in df.dropna(subset=["DRUG_ID", "DRUG_NAME"]).itertuples(index=False):
        query_names: list[str] = []
        seen: set[str] = set()
        synonym_chunks = _split_alias_chunks(str(getattr(row, "SYNONYMS", "") or ""))
        for raw in [row.DRUG_NAME, *synonym_chunks]:
            for name in _expand_alias_candidates(str(raw)):
                key = normalize_name(name)
                if name and key and key not in seen:
                    seen.add(key)
                    query_names.append(name)
        aliases[int(row.DRUG_ID)] = query_names
    return aliases


def _merge_query_names(dataset: str, drug_name: str, base_names: list[str]) -> list[str]:
    dataset = str(dataset).strip().lower()
    merged: list[str] = []
    seen: set[str] = set()
    manual = MANUAL_QUERY_ALIASES.get(dataset, {}).get(normalize_name(drug_name), [])
    for raw in [drug_name, *base_names, *manual]:
        for name in _expand_alias_candidates(str(raw)):
            key = normalize_name(name)
            if name and key and key not in seen:
                seen.add(key)
                merged.append(name)
    return merged


def _classify_dataset_entry(dataset: str, drug_name: str) -> str | None:
    key = normalize_name(drug_name)
    if dataset == "gdsc":
        if re.fullmatch(r"[a-z0-9]+_\d{4,}", key):
            return "excluded_non_public_identifier"
        if re.fullmatch(r"(fy|zl)\d{2,}", key):
            return "excluded_non_public_identifier"
        if re.fullmatch(r"bd[a-z0-9]+a", key):
            return "excluded_non_public_identifier"
        if re.fullmatch(r"thr \d+", key):
            return "excluded_non_public_identifier"
    for pattern in NON_SMALL_MOLECULE_PATTERNS.get(dataset, []):
        if pattern in key:
            return "excluded_non_small_molecule_or_non_drug"
    if any(token in key for token in BIOLOGIC_PATTERNS):
        return "excluded_biologic_or_large_molecule"
    return None


def _result_matches_query(query: str, result: dict[str, object]) -> bool:
    q = _match_key(query)
    title = _match_key(result.get("title", ""))
    if not title:
        return True
    if q == title:
        return True
    q_tokens = {tok for tok in q.split() if tok}
    title_tokens = {tok for tok in title.split() if tok}
    if not q_tokens:
        return True
    overlap = q_tokens & title_tokens
    if q_tokens <= title_tokens:
        return True
    if len(overlap) >= max(1, min(2, len(q_tokens))):
        return True
    cid = result.get("cid")
    if cid is None:
        return False
    return _pubchem_cid_has_query_name(int(cid), query)


def _match_key(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _query_variants(drug_name: str) -> list[str]:
    raw = str(drug_name).strip()
    variants: list[str] = []
    for candidate in _expand_alias_candidates(raw):
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _split_alias_chunks(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    for piece in re.split(r",|;|\|", text):
        piece = piece.strip()
        if piece:
            chunks.append(piece)
    return chunks


def _expand_alias_candidates(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    seeds: list[str] = [raw]
    normalized_dash = raw.replace("–", "-").replace("—", "-")
    if normalized_dash not in seeds:
        seeds.append(normalized_dash)
    if "," in normalized_dash:
        for part in [p.strip() for p in normalized_dash.split(",") if p.strip()]:
            if part not in seeds:
                seeds.append(part)
    if "/" in normalized_dash:
        slash_spaced = normalized_dash.replace("/", " ")
        if slash_spaced not in seeds:
            seeds.append(slash_spaced)
    if "_" in normalized_dash:
        underscore_spaced = normalized_dash.replace("_", " ")
        if underscore_spaced not in seeds:
            seeds.append(underscore_spaced)
        base_prefix = re.sub(r"_\d{3,}$", "", normalized_dash).strip()
        if base_prefix and base_prefix not in seeds:
            seeds.append(base_prefix)
    paren_stripped = re.sub(r"\([^)]*\)", "", normalized_dash).strip(" ,")
    if paren_stripped and paren_stripped not in seeds:
        seeds.append(paren_stripped)
    # Pull out likely compound codes from longer labels, e.g. "Gamma-Secretase Inhibitor RO4929097".
    for code in re.findall(r"\b[A-Za-z]{1,8}-?\d{3,}[A-Za-z0-9-]*\b", normalized_dash):
        if code not in seeds:
            seeds.append(code)

    variants: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        candidates = [
            seed,
            " ".join(seed.split()),
            seed.replace("-", " "),
            seed.replace("-", ""),
            seed.replace("/", " "),
        ]
        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", candidate).strip(" ,")
            key = normalize_name(candidate)
            if candidate and key and key not in seen:
                seen.add(key)
                variants.append(candidate)
    return variants


def _pubchem_cid_has_query_name(cid: int, query: str) -> bool:
    synonyms = _pubchem_cid_synonyms(cid)
    query_key = _match_key(query)
    return any(_match_key(name) == query_key for name in synonyms)


def _pubchem_cid_synonyms(cid: int, retries: int = 2, timeout_seconds: int = 8) -> list[str]:
    req = Request(PUBCHEM_CID_SYNONYMS_URL.format(cid=int(cid)), method="GET")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                body = json.load(resp)
            info = body.get("InformationList", {}).get("Information", [])
            if not info:
                return []
            synonyms = info[0].get("Synonym", [])
            return [str(name).strip() for name in synonyms if str(name).strip()]
        except HTTPError as exc:
            if exc.code == 404:
                return []
            last_error = exc
        except URLError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc
        except http.client.RemoteDisconnected as exc:
            last_error = exc
        time.sleep(min(2**attempt, 3))
    raise GdscSmilesCacheError(f"PubChem synonym lookup failed for CID {cid!r}: {last_error}") from last_error


def _pubchem_name_lookup(query: str, retries: int = 2, timeout_seconds: int = 8) -> dict[str, object] | None:
    payload = urlencode({"name": query}).encode("utf-8")
    req = Request(PUBCHEM_PROPERTY_URL, data=payload, method="POST")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                body = json.load(resp)
            props = body.get("PropertyTable", {}).get("Properties", [])
            if not props:
                return None
            row = props[0]
            smiles = row.get("ConnectivitySMILES") or row.get("SMILES")
            cid = row.get("CID")
            if not smiles or cid is None:
                return None
            return {
                "cid": int(cid),
                "smiles": str(smiles).strip(),
                "title": str(row.get("Title", "")).strip(),
                "query": query,
            }
        except HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = exc
        except URLError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc
        except http.client.RemoteDisconnected as exc:
            last_error = exc
        time.sleep(min(2**attempt, 3))
    raise GdscSmilesCacheError(f"PubChem lookup failed for {query!r}: {last_error}") from last_error


def _chembl_result_matches_query(query: str, result: dict[str, object]) -> bool:
    query_key = _match_key(query)
    if not query_key:
        return False
    names = [str(result.get("title", ""))]
    names.extend([str(x) for x in result.get("aliases", [])])
    for name in names:
        nk = _match_key(name)
        if not nk:
            continue
        if nk == query_key:
            return True
        q_tokens = {tok for tok in query_key.split() if tok}
        n_tokens = {tok for tok in nk.split() if tok}
        if q_tokens and (q_tokens <= n_tokens or len(q_tokens & n_tokens) >= max(1, min(2, len(q_tokens)))):
            return True
    return False


def _chembl_name_lookup(query: str, retries: int = 2, timeout_seconds: int = 8) -> dict[str, object] | None:
    req = Request(CHEMBL_SEARCH_URL.format(query=urlencode({"x": query})[2:]), method="GET")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                body = json.load(resp)
            molecules = body.get("molecules", []) or []
            for row in molecules:
                mol_struct = row.get("molecule_structures") or {}
                smiles = mol_struct.get("canonical_smiles")
                chembl_id = row.get("molecule_chembl_id")
                if not smiles or not chembl_id:
                    continue
                aliases: list[str] = []
                for syn in row.get("molecule_synonyms") or []:
                    synonym = syn.get("molecule_synonym")
                    if synonym:
                        aliases.append(str(synonym))
                pref_name = str(row.get("pref_name") or chembl_id)
                return {
                    "cid": int(re.sub(r"\D+", "", str(chembl_id)) or 0),
                    "smiles": str(smiles).strip(),
                    "title": pref_name.strip(),
                    "query": query,
                    "aliases": aliases,
                }
            return None
        except HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = exc
        except URLError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc
        except http.client.RemoteDisconnected as exc:
            last_error = exc
        time.sleep(min(2**attempt, 3))
    # Network jitter on the public ChEMBL API should not abort a full dataset build.
    return None
