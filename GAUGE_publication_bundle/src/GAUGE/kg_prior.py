from __future__ import annotations

import json
import sqlite3
import shutil
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .cache import cache_key, file_signature, files_signature
from .config import CHEMBL_RELEASE_DEFAULT, default_chembl_uniprot_mapping, resolve_chembl_sqlite_tar
from .utils import normalize_name


KG_PRIOR_SCHEMA_VERSION = 5
KG_SOURCES = ("ChEMBL", "DRKG", "PrimeKG")
KG_SOURCE_KEYS = ("chembl", "drkg", "primekg")


def _canonical_pair_edge_type(left: str, right: str) -> str:
    left = str(left)
    right = str(right)
    if left == "drug" and right != "drug":
        return f"drug_{right}"
    if right == "drug" and left != "drug":
        return f"drug_{left}"
    a, b = sorted((left, right))
    return f"{a}_{b}"


KG_SOURCE_NODE_TYPES = {
    "chembl": ("drug", "target", "action_type", "target_type", "mechanism"),
    "drkg": (
        "drug",
        "gene",
        "protein",
        "disease",
        "pathway",
        "side_effect",
        "symptom",
        "anatomy",
        "biological_process",
        "molecular_function",
        "cellular_component",
        "taxonomy",
        "other",
    ),
    "primekg": (
        "drug",
        "protein",
        "side_effect",
        "disease",
        "pathway",
        "phenotype",
        "anatomy",
        "biological_process",
        "molecular_function",
        "cellular_component",
        "exposure",
        "other",
    ),
}


def _undirected_pair_edge_types(node_types: tuple[str, ...]) -> tuple[str, ...]:
    non_drug = [node_type for node_type in node_types if node_type != "drug"]
    edge_types: list[str] = [f"drug_{node_type}" for node_type in non_drug]
    for idx, left in enumerate(non_drug):
        for right in non_drug[idx:]:
            edge_types.append(_canonical_pair_edge_type(left, right))
    return tuple(edge_types)


KG_SOURCE_EDGE_TYPES = {
    "chembl": (
        "action_type_target",
        "has_action_type",
        "acts_on_target",
        "target_has_target_type",
        "drug_has_mechanism",
        "mechanism_targets_target",
    ),
    "drkg": _undirected_pair_edge_types(KG_SOURCE_NODE_TYPES["drkg"]),
    "primekg": _undirected_pair_edge_types(KG_SOURCE_NODE_TYPES["primekg"]),
}
KG_SOURCE_NAME_BY_KEY = {
    "chembl": "ChEMBL",
    "drkg": "DRKG",
    "primekg": "PrimeKG",
}
PUBCHEM_CID_SYNONYMS_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"


@dataclass
class MultiKGGraphArtifacts:
    drug_ids: list[int]
    drug_to_local: dict[int, int]
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    coverage: pd.DataFrame
    edge_audit: pd.DataFrame
    cache_key: str
    source_files: list[str]
    source_dir: str
    resolved_prior_policy: dict[str, Any] = field(default_factory=dict)
    branch_names: list[str] = field(default_factory=lambda: list(KG_SOURCES))

    @property
    def n_drugs(self) -> int:
        return len(self.drug_ids)

    @property
    def n_nodes(self) -> int:
        return int(len(self.node_table))

    def branch_mask_for_drugs(self, drug_ids: list[int]) -> np.ndarray:
        frame = self.coverage.set_index("DRUG_ID")
        cols = [f"has_{name}" for name in self.branch_names]
        out = []
        for drug_id in drug_ids:
            if int(drug_id) in frame.index:
                out.append(frame.loc[int(drug_id), cols].astype(np.float32).to_numpy(copy=True))
            else:
                out.append(np.zeros((len(cols),), dtype=np.float32))
        return np.vstack(out).astype(np.float32, copy=False) if out else np.empty((0, len(cols)), dtype=np.float32)

    def branch_degree_for_drugs(self, drug_ids: list[int]) -> np.ndarray:
        frame = self.coverage.set_index("DRUG_ID")
        cols = [f"graph_degree_{name}" for name in self.branch_names]
        out = []
        for drug_id in drug_ids:
            if int(drug_id) in frame.index:
                out.append(frame.loc[int(drug_id), cols].astype(np.float32).to_numpy(copy=True))
            else:
                out.append(np.zeros((len(cols),), dtype=np.float32))
        return np.vstack(out).astype(np.float32, copy=False) if out else np.empty((0, len(cols)), dtype=np.float32)


def _policy_dict(policy: Any | None) -> dict[str, Any]:
    if policy is None:
        return {}
    if hasattr(policy, "to_dict"):
        try:
            data = policy.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if isinstance(policy, dict):
        return policy
    return {}


def _validate_policy_values(source_key: str, field_name: str, values: tuple[str, ...], valid_values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            normalized.append(text)
    invalid = [value for value in normalized if value not in valid_values]
    if invalid:
        raise ValueError(
            f"Invalid {source_key}.{field_name}: {invalid}. "
            f"Valid options: {list(valid_values)}"
        )
    return tuple(normalized)


def _resolve_source_policy(
    policy: Any | None,
    source_key: str,
    default_node_types: tuple[str, ...],
    default_edge_types: tuple[str, ...],
) -> dict[str, Any]:
    data = _policy_dict(policy)
    source = data.get(source_key, {}) if isinstance(data, dict) else {}
    if hasattr(source, "to_dict"):
        source = source.to_dict()
    if not isinstance(source, dict):
        source = {}
    raw_node_types = source.get("node_types")
    raw_edge_types = source.get("edge_types")
    node_types = default_node_types if raw_node_types is None else tuple(raw_node_types)
    edge_types = default_edge_types if raw_edge_types is None else tuple(raw_edge_types)
    node_types = _validate_policy_values(source_key, "node_types", node_types, KG_SOURCE_NODE_TYPES[source_key])
    edge_types = _validate_policy_values(source_key, "edge_types", edge_types, KG_SOURCE_EDGE_TYPES[source_key])
    resolved = {
        "enabled": bool(source.get("enabled", True)),
        "weight": float(source.get("weight", 1.0)),
        "node_types": list(node_types),
        "edge_types": list(edge_types),
    }
    if source_key == "chembl":
        resolved["release"] = str(source.get("release", CHEMBL_RELEASE_DEFAULT))
        resolved["sqlite_tar"] = source.get("sqlite_tar")
    return resolved


def _resolve_prior_policy(prior_policy: Any | None, include_side_effects: bool | None) -> dict[str, Any]:
    data = _policy_dict(prior_policy)
    if data and "chembl" in data and "drkg" in data and "primekg" in data:
        resolved = {
            "chembl": _resolve_source_policy(data, "chembl", KG_SOURCE_NODE_TYPES["chembl"], KG_SOURCE_EDGE_TYPES["chembl"]),
            "drkg": _resolve_source_policy(data, "drkg", KG_SOURCE_NODE_TYPES["drkg"], KG_SOURCE_EDGE_TYPES["drkg"]),
            "primekg": _resolve_source_policy(data, "primekg", KG_SOURCE_NODE_TYPES["primekg"], KG_SOURCE_EDGE_TYPES["primekg"]),
            "include_side_effects": bool(data.get("include_side_effects", False) if include_side_effects is None else include_side_effects),
        }
    else:
        resolved = {
            "chembl": {
                "enabled": True,
                "weight": 1.0,
                "node_types": list(KG_SOURCE_NODE_TYPES["chembl"]),
                "edge_types": list(KG_SOURCE_EDGE_TYPES["chembl"]),
                "release": CHEMBL_RELEASE_DEFAULT,
                "sqlite_tar": None,
            },
            "drkg": {
                "enabled": True,
                "weight": 1.0,
                "node_types": list(KG_SOURCE_NODE_TYPES["drkg"]),
                "edge_types": list(KG_SOURCE_EDGE_TYPES["drkg"]),
            },
            "primekg": {
                "enabled": True,
                "weight": 1.0,
                "node_types": list(KG_SOURCE_NODE_TYPES["primekg"]),
                "edge_types": list(KG_SOURCE_EDGE_TYPES["primekg"]),
            },
            "include_side_effects": bool(include_side_effects) if include_side_effects is not None else False,
        }
    if resolved.get("include_side_effects"):
        primekg_policy = resolved["primekg"]
        node_types = list(primekg_policy.get("node_types", []))
        edge_types = list(primekg_policy.get("edge_types", []))
        if "side_effect" not in node_types:
            node_types.append("side_effect")
        if "drug_side_effect" not in edge_types:
            edge_types.append("drug_side_effect")
        primekg_policy["node_types"] = list(_validate_policy_values("primekg", "node_types", tuple(node_types), KG_SOURCE_NODE_TYPES["primekg"]))
        primekg_policy["edge_types"] = list(_validate_policy_values("primekg", "edge_types", tuple(edge_types), KG_SOURCE_EDGE_TYPES["primekg"]))
    return resolved


def resolve_chembl_source(paths: Any, prior_policy: Any | None = None) -> tuple[str, Path, Path]:
    policy = _policy_dict(prior_policy)
    chembl_policy = policy.get("chembl", {}) if isinstance(policy, dict) else {}
    if hasattr(chembl_policy, "to_dict"):
        chembl_policy = chembl_policy.to_dict()
    if not isinstance(chembl_policy, dict):
        chembl_policy = {}
    release = str(chembl_policy.get("release", CHEMBL_RELEASE_DEFAULT))
    tar_path = resolve_chembl_sqlite_tar(Path(paths.root), release=release, override=chembl_policy.get("sqlite_tar"))
    mapping_path = default_chembl_uniprot_mapping(Path(paths.root))
    return release, tar_path, mapping_path


def kg_source_paths(paths: Any, prior_policy: Any | None = None) -> list[Path]:
    _, chembl_tar_path, chembl_mapping_path = resolve_chembl_source(paths, prior_policy=prior_policy)
    root = Path(paths.root) / "KG_GAUGE_PublicData" / "drug"
    candidates = [
        chembl_tar_path,
        chembl_mapping_path,
        root / "drkg" / "drkg.tar.gz",
        root / "PrimeKG" / "kg.csv",
        paths.gdsc_smiles_cache,
    ]
    return [p for p in candidates if Path(p).exists()]


def build_multikg_graph_artifacts(
    paths: Any,
    drug_table: pd.DataFrame,
    cache_dir: Path,
    *,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    max_ppi_neighbors: int = 50,
    prior_policy: Any | None = None,
    include_side_effects: bool | None = None,
) -> MultiKGGraphArtifacts:
    cache_dir = Path(cache_dir) / "kg_prior"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(paths.root) / "KG_GAUGE_PublicData" / "drug"
    resolved_prior_policy = _resolve_prior_policy(prior_policy, include_side_effects)
    _, chembl_tar_path, _ = resolve_chembl_source(paths, prior_policy=resolved_prior_policy)
    source_files = kg_source_paths(paths, prior_policy=resolved_prior_policy)
    payload = {
        "kind": "multikg_graph_prior",
        "schema": KG_PRIOR_SCHEMA_VERSION,
        "source_files": files_signature(source_files),
        "drug_ids": sorted(map(int, drug_table["DRUG_ID"].tolist())),
        "drug_keys": sorted(map(str, drug_table.get("drug_key", pd.Series(dtype=str)).tolist())),
        "max_ppi_neighbors": int(max_ppi_neighbors),
        "resolved_prior_policy": resolved_prior_policy,
    }
    key = cache_key(payload)
    artifact_dir = cache_dir / key
    pickle_path = artifact_dir / "kg_artifacts.pkl"
    if use_cache and not rebuild_cache and pickle_path.exists():
        return pd.read_pickle(pickle_path)

    drug_records = _drug_records(drug_table)
    alias_index = _build_drug_alias_index(drug_records, artifact_dir, rebuild_cache=rebuild_cache)
    node_rows = []
    edge_rows = []
    audit_rows = []
    node_index: dict[tuple[str, str, str], int] = {}

    def node_id(source: str, node_type: str, name: str) -> int:
        key_tuple = (source, node_type, str(name))
        found = node_index.get(key_tuple)
        if found is not None:
            return found
        idx = len(node_rows)
        node_index[key_tuple] = idx
        node_rows.append(
            {
                "node_id": idx,
                "source": source,
                "node_type": node_type,
                "name": str(name),
                "normalized_name": normalize_name(name),
            }
        )
        return idx

    for rec in drug_records:
        rec["drug_node_ids"] = {source: node_id(source, "drug", rec["DRUG_NAME"]) for source in KG_SOURCES}

    _add_chembl_edges(
        source_root,
        drug_records,
        alias_index,
        node_id,
        edge_rows,
        audit_rows,
        artifact_dir / "chembl_sqlite",
        chembl_tar_path=chembl_tar_path,
        resolved_prior_policy=resolved_prior_policy["chembl"],
    )
    _add_drkg_edges(
        source_root,
        drug_records,
        alias_index,
        node_id,
        edge_rows,
        audit_rows,
        max_ppi_neighbors=max_ppi_neighbors,
        resolved_prior_policy=resolved_prior_policy["drkg"],
    )
    _add_primekg_edges(
        source_root,
        drug_records,
        node_id,
        edge_rows,
        audit_rows,
        max_ppi_neighbors=max_ppi_neighbors,
        resolved_prior_policy=resolved_prior_policy["primekg"],
    )
    node_table = pd.DataFrame(node_rows, columns=["node_id", "source", "node_type", "name", "normalized_name"])
    edge_table = pd.DataFrame(
        edge_rows,
        columns=["source", "src", "dst", "edge_type", "relation", "DRUG_ID", "weight"],
    ).drop_duplicates(["source", "src", "dst", "edge_type", "relation"])
    if edge_table.empty:
        edge_table = pd.DataFrame(columns=["edge_id", "source", "src", "dst", "edge_type", "relation", "DRUG_ID", "weight"])
    else:
        edge_table = edge_table.reset_index(drop=True)
        edge_table.insert(0, "edge_id", np.arange(len(edge_table), dtype=np.int64))
    coverage = _coverage_frame(drug_records, edge_table)
    edge_audit = pd.DataFrame(audit_rows, columns=["source", "event", "count", "detail"])
    artifacts = MultiKGGraphArtifacts(
        drug_ids=[int(x["DRUG_ID"]) for x in drug_records],
        drug_to_local={int(x["DRUG_ID"]): i for i, x in enumerate(drug_records)},
        node_table=node_table,
        edge_table=edge_table,
        coverage=coverage,
        edge_audit=edge_audit,
        cache_key=key,
        source_files=[str(p) for p in source_files],
        source_dir=str(source_root),
        resolved_prior_policy=resolved_prior_policy,
    )
    _write_artifact_reports(artifact_dir, artifacts, payload)
    artifacts.coverage.to_csv(artifact_dir / "kg_coverage_by_drug.csv", index=False)
    artifacts.node_table.to_csv(artifact_dir / "kg_node_index.csv", index=False)
    artifacts.edge_table.to_csv(artifact_dir / "kg_edges.csv", index=False)
    artifacts.edge_audit.to_csv(artifact_dir / "kg_edge_audit.csv", index=False)
    _write_source_edge_tables(artifact_dir, artifacts.edge_table)
    pd.to_pickle(artifacts, pickle_path)
    return artifacts


def write_kg_reports(out_dir: Path, artifacts: MultiKGGraphArtifacts | None) -> None:
    if artifacts is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.coverage.to_csv(out_dir / "kg_coverage_by_drug.csv", index=False)
    artifacts.node_table.to_csv(out_dir / "kg_node_index.csv", index=False)
    artifacts.edge_audit.to_csv(out_dir / "kg_edge_audit.csv", index=False)
    _write_source_edge_tables(out_dir, artifacts.edge_table)
    master = artifacts.coverage.copy()
    master.to_csv(out_dir / "drug_mapping_master.csv", index=False)


def _drug_records(drug_table: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in drug_table.itertuples(index=False):
        name = str(getattr(row, "DRUG_NAME"))
        drug_key = str(getattr(row, "drug_key", normalize_name(name)))
        records.append(
            {
                "DRUG_ID": int(getattr(row, "DRUG_ID")),
                "DRUG_NAME": name,
                "drug_key": drug_key,
                "canonical_name": normalize_name(name),
                "SMILES": str(getattr(row, "smiles", "")),
                "canonical_SMILES": str(getattr(row, "canonical_smiles", getattr(row, "smiles", ""))),
                "InChIKey": str(getattr(row, "inchikey", "")),
                "pubchem_cid": getattr(row, "pubchem_cid", pd.NA),
                "pubchem_title": str(getattr(row, "pubchem_title", "")),
            }
        )
    return records


def _build_drug_alias_index(
    drug_records: list[dict[str, Any]],
    artifact_dir: Path,
    *,
    rebuild_cache: bool,
) -> dict[str, dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    alias_cache = artifact_dir / "drkg_pubchem_aliases.csv"
    pubchem_cache: dict[int, list[str]] = {}
    if alias_cache.exists() and not rebuild_cache:
        try:
            cached = pd.read_csv(alias_cache)
            for row in cached.itertuples(index=False):
                pubchem_cache.setdefault(int(getattr(row, "pubchem_cid")), []).append(str(getattr(row, "alias")))
        except Exception:
            pubchem_cache = {}

    needed_cids = sorted(
        {
            int(rec["pubchem_cid"])
            for rec in drug_records
            if pd.notna(rec.get("pubchem_cid")) and str(rec.get("pubchem_cid")).strip() != ""
        }
    )
    missing_cids = [cid for cid in needed_cids if cid not in pubchem_cache]
    if missing_cids:
        rows: list[dict[str, Any]] = []
        max_workers = min(8, max(1, len(missing_cids)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_pubchem_cid_synonyms, cid): cid for cid in missing_cids}
            for fut in as_completed(futures):
                cid = futures[fut]
                aliases = fut.result()
                pubchem_cache.setdefault(cid, []).extend(aliases)
                for alias in aliases:
                    rows.append({"pubchem_cid": cid, "alias": alias})
        if rows:
            pd.DataFrame(rows, columns=["pubchem_cid", "alias"]).to_csv(alias_cache, index=False)

    alias_index: dict[str, dict[str, Any]] = {}
    for rec in drug_records:
        aliases = _drug_alias_candidates(rec)
        cid = rec.get("pubchem_cid")
        if pd.notna(cid) and str(cid).strip() != "":
            cid = int(cid)
            aliases.extend(pubchem_cache.get(cid, []))
            aliases.append(f"pubchem:{cid}")
            aliases.append(f"pubchem {cid}")
        for alias in aliases:
            key = normalize_name(alias)
            if key:
                alias_index.setdefault(key, rec)
    return alias_index


def _drug_alias_candidates(rec: dict[str, Any]) -> list[str]:
    aliases = [
        rec.get("DRUG_NAME", ""),
        rec.get("drug_key", ""),
        rec.get("canonical_name", ""),
        rec.get("canonical_SMILES", ""),
        rec.get("InChIKey", ""),
        rec.get("pubchem_title", ""),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = normalize_name(alias)
        if key and key not in seen:
            seen.add(key)
            out.append(str(alias))
    return out


def _fetch_pubchem_cid_synonyms(cid: int, retries: int = 2, timeout_seconds: int = 8) -> list[str]:
    req = Request(PUBCHEM_CID_SYNONYMS_URL.format(cid=int(cid)), method="GET")
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                body = json.load(resp)
            info = body.get("InformationList", {}).get("Information", [])
            if not info:
                return []
            synonyms = info[0].get("Synonym", [])
            out: list[str] = []
            seen: set[str] = set()
            for alias in synonyms:
                text = str(alias).strip()
                key = normalize_name(text)
                if text and key and key not in seen:
                    seen.add(key)
                    out.append(text)
            return out
        except (HTTPError, URLError, TimeoutError):
            if attempt < retries - 1:
                continue
            return []
        except Exception:
            if attempt < retries - 1:
                continue
            return []


def _write_artifact_reports(artifact_dir: Path, artifacts: MultiKGGraphArtifacts, payload: dict[str, Any]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kg_prior_schema_version": KG_PRIOR_SCHEMA_VERSION,
        "cache_key": artifacts.cache_key,
        "n_drugs": artifacts.n_drugs,
        "n_nodes": artifacts.n_nodes,
        "n_edges": int(len(artifacts.edge_table)),
        "resolved_prior_policy": artifacts.resolved_prior_policy,
        "payload": payload,
    }
    (artifact_dir / "kg_cache_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_source_edge_tables(out_dir: Path, edge_table: pd.DataFrame) -> None:
    mapping = {
        "ChEMBL": "chembl_moa_edges.csv",
        "DRKG": "drkg_filtered_edges.csv",
        "PrimeKG": "primekg_filtered_edges.csv",
    }
    for source, name in mapping.items():
        frame = edge_table.loc[edge_table["source"].eq(source)].copy()
        frame.to_csv(Path(out_dir) / name, index=False)


def _coverage_frame(drug_records: list[dict[str, Any]], edge_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in drug_records:
        row = {
            "GDSC_DRUG_ID": int(rec["DRUG_ID"]),
            "DRUG_ID": int(rec["DRUG_ID"]),
            "DRUG_NAME": rec["DRUG_NAME"],
            "canonical_name": rec["canonical_name"],
            "SMILES": rec["SMILES"],
            "canonical_SMILES": rec["canonical_SMILES"],
            "InChIKey": rec["InChIKey"],
        }
        for source in KG_SOURCES:
            sub = edge_table.loc[edge_table["source"].eq(source) & edge_table["DRUG_ID"].eq(int(rec["DRUG_ID"]))]
            row[f"{source}_node_id"] = rec["drug_node_ids"][source]
            row[f"has_{source}"] = int(not sub.empty)
            row[f"graph_degree_{source}"] = float(sub["weight"].sum()) if not sub.empty and "weight" in sub.columns else float(len(sub))
            row[f"source_weight_{source}"] = float(sub["weight"].iloc[0]) if not sub.empty and "weight" in sub.columns else float(0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _append_policy_audit(audit_rows: list[dict[str, Any]], source: str, resolved_prior_policy: dict[str, Any]) -> None:
    audit_rows.append(
        {
            "source": source,
            "event": "configured_node_types",
            "count": len(tuple(resolved_prior_policy.get("node_types", ()))),
            "detail": json.dumps(list(resolved_prior_policy.get("node_types", ())), sort_keys=True),
        }
    )
    audit_rows.append(
        {
            "source": source,
            "event": "configured_edge_types",
            "count": len(tuple(resolved_prior_policy.get("edge_types", ()))),
            "detail": json.dumps(list(resolved_prior_policy.get("edge_types", ())), sort_keys=True),
        }
    )


def _append_counter_audit(
    audit_rows: list[dict[str, Any]],
    source: str,
    event_prefix: str,
    counts: Counter[str],
) -> None:
    for key, value in sorted(counts.items()):
        audit_rows.append({"source": source, "event": event_prefix, "count": int(value), "detail": str(key)})


def _track_row_node_types(counter: Counter[str], *node_types: str) -> None:
    for node_type in node_types:
        counter[str(node_type)] += 1


def _drug_match_maps(
    drug_records: list[dict[str, Any]],
    alias_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in drug_records:
        keys = {
            rec["canonical_name"],
            normalize_name(rec["DRUG_NAME"]),
            normalize_name(rec["drug_key"]),
            normalize_name(rec.get("InChIKey", "")),
            normalize_name(rec.get("pubchem_title", "")),
        }
        cid = rec.get("pubchem_cid")
        if pd.notna(cid) and str(cid).strip() != "":
            cid = int(cid)
            keys.add(normalize_name(f"pubchem:{cid}"))
            keys.add(normalize_name(f"pubchem {cid}"))
        for key in keys:
            if key:
                out.setdefault(key, rec)
    if alias_index:
        for key, rec in alias_index.items():
            if key:
                out.setdefault(key, rec)
    return out


def _match_drug_record(drug_map: dict[str, dict[str, Any]], raw_value: str) -> dict[str, Any] | None:
    for key in _drkg_candidate_keys(raw_value):
        rec = drug_map.get(key)
        if rec is not None:
            return rec
    return None


def _drkg_candidate_keys(value: str) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []
    candidates = [raw]
    if "::" in raw:
        candidates.append(raw.split("::", 1)[1])
    if ":" in raw:
        candidates.append(raw.replace(":", " "))
    if "_" in raw:
        candidates.append(raw.replace("_", " "))
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = normalize_name(item)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _add_edge(
    edge_rows: list[dict[str, Any]],
    *,
    source: str,
    src: int,
    dst: int,
    edge_type: str,
    relation: str,
    drug_id: int,
    weight: float = 1.0,
) -> None:
    edge_rows.append(
        {
            "source": source,
            "src": int(src),
            "dst": int(dst),
            "edge_type": str(edge_type),
            "relation": str(relation),
            "DRUG_ID": int(drug_id),
            "weight": float(weight),
        }
    )
    edge_rows.append(
        {
            "source": source,
            "src": int(dst),
            "dst": int(src),
            "edge_type": f"rev_{edge_type}",
            "relation": str(relation),
            "DRUG_ID": int(drug_id),
            "weight": float(weight),
        }
    )


def _primekg_node_type(value: Any) -> str:
    low = str(value).strip().lower()
    if low == "drug" or "compound" in low:
        return "drug"
    if "gene/protein" in low or low == "protein" or "gene" in low:
        return "protein"
    if "side effect" in low or "adverse effect" in low:
        return "side_effect"
    if any(term in low for term in ("disease", "indication", "contraindication", "off-label", "off label")):
        return "disease"
    if "pathway" in low:
        return "pathway"
    if "phenotype" in low or "symptom" in low:
        return "phenotype"
    if "anatom" in low or "tissue" in low or "organ" in low:
        return "anatomy"
    if "biological process" in low:
        return "biological_process"
    if "molecular function" in low:
        return "molecular_function"
    if "cellular component" in low:
        return "cellular_component"
    if "exposure" in low:
        return "exposure"
    return "other"


def _drkg_kind(value: Any) -> str:
    prefix = str(value).split("::", 1)[0].strip().lower()
    if "gene" in prefix:
        return "gene"
    if "protein" in prefix:
        return "protein"
    if "compound" in prefix or "drug" in prefix:
        return "drug"
    if "disease" in prefix:
        return "disease"
    if "pathway" in prefix:
        return "pathway"
    if "side effect" in prefix or "side_effect" in prefix:
        return "side_effect"
    if "symptom" in prefix or "phenotype" in prefix:
        return "symptom"
    if "anatom" in prefix or "tissue" in prefix or "organ" in prefix:
        return "anatomy"
    if "biological process" in prefix:
        return "biological_process"
    if "molecular function" in prefix:
        return "molecular_function"
    if "cellular component" in prefix:
        return "cellular_component"
    if "taxonomy" in prefix:
        return "taxonomy"
    return "other"


def _drkg_name(value: Any) -> str:
    text = str(value)
    return text.split("::", 1)[1] if "::" in text else text


def _add_primekg_edges(
    source_root: Path,
    drug_records: list[dict[str, Any]],
    node_id,
    edge_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    max_ppi_neighbors: int,
    resolved_prior_policy: dict[str, Any],
) -> None:
    path = source_root / "PrimeKG" / "kg.csv"
    if not path.exists():
        audit_rows.append({"source": "PrimeKG", "event": "missing_source", "count": 1, "detail": str(path)})
        return
    if not bool(resolved_prior_policy.get("enabled", True)) or float(resolved_prior_policy.get("weight", 1.0)) <= 0.0:
        audit_rows.append({"source": "PrimeKG", "event": "disabled_by_config", "count": 1, "detail": json.dumps(resolved_prior_policy, sort_keys=True)})
        return
    allowed_node_types = set(resolved_prior_policy.get("node_types") or KG_SOURCE_NODE_TYPES["primekg"])
    allowed_edge_types = set(resolved_prior_policy.get("edge_types") or KG_SOURCE_EDGE_TYPES["primekg"])
    _append_policy_audit(audit_rows, "PrimeKG", resolved_prior_policy)
    drug_map = _drug_match_maps(drug_records)
    retained_nodes: dict[int, set[tuple[str, str]]] = {int(rec["DRUG_ID"]): set() for rec in drug_records}
    retained_edge_types: Counter[str] = Counter()
    retained_node_types: Counter[str] = Counter()
    skipped_node_types: Counter[str] = Counter()
    skipped_edge_types: Counter[str] = Counter()
    usecols = ["relation", "display_relation", "x_type", "x_name", "y_type", "y_name"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        for row in chunk.itertuples(index=False):
            x_key = normalize_name(row.x_name)
            y_key = normalize_name(row.y_name)
            x_type = _primekg_node_type(row.x_type)
            y_type = _primekg_node_type(row.y_type)
            relation = str(row.display_relation or row.relation)
            for drug_side in ("x", "y"):
                rec = drug_map.get(x_key if drug_side == "x" else y_key)
                if rec is None:
                    continue
                matched_type = x_type if drug_side == "x" else y_type
                other_node_type = y_type if drug_side == "x" else x_type
                other_name = row.y_name if drug_side == "x" else row.x_name
                if matched_type != "drug" or other_node_type == "drug":
                    continue
                edge_type = _canonical_pair_edge_type("drug", other_node_type)
                if "drug" not in allowed_node_types or other_node_type not in allowed_node_types:
                    skipped_node_types[edge_type] += 1
                    continue
                if edge_type not in allowed_edge_types:
                    skipped_edge_types[edge_type] += 1
                    continue
                drug_node = rec["drug_node_ids"]["PrimeKG"]
                other_node = node_id("PrimeKG", other_node_type, other_name)
                _add_edge(
                    edge_rows,
                    source="PrimeKG",
                    src=drug_node,
                    dst=other_node,
                    edge_type=edge_type,
                    relation=relation,
                    drug_id=rec["DRUG_ID"],
                    weight=float(resolved_prior_policy.get("weight", 1.0)),
                )
                retained_nodes[int(rec["DRUG_ID"])].add((other_node_type, normalize_name(other_name)))
                retained_edge_types[edge_type] += 1
                _track_row_node_types(retained_node_types, "drug", other_node_type)
    _add_source_expansion_edges(
        path,
        source_name="PrimeKG",
        retained_nodes=retained_nodes,
        node_id=node_id,
        edge_rows=edge_rows,
        max_ppi_neighbors=max_ppi_neighbors,
        node_type_fn=_primekg_node_type,
        name_fn=lambda value: str(value),
        allowed_node_types=allowed_node_types,
        allowed_edge_types=allowed_edge_types,
        weight=float(resolved_prior_policy.get("weight", 1.0)),
        retained_edge_types=retained_edge_types,
        retained_node_types=retained_node_types,
        skipped_node_types=skipped_node_types,
        skipped_edge_types=skipped_edge_types,
    )
    _append_counter_audit(audit_rows, "PrimeKG", "retained_edge_type", retained_edge_types)
    _append_counter_audit(audit_rows, "PrimeKG", "retained_node_type", retained_node_types)
    _append_counter_audit(audit_rows, "PrimeKG", "skipped_by_node_allowlist", skipped_node_types)
    _append_counter_audit(audit_rows, "PrimeKG", "skipped_by_edge_allowlist", skipped_edge_types)


def _add_source_expansion_edges(
    path: Path,
    *,
    source_name: str,
    retained_nodes: dict[int, set[tuple[str, str]]],
    node_id,
    edge_rows: list[dict[str, Any]],
    max_ppi_neighbors: int,
    node_type_fn,
    name_fn,
    allowed_node_types: set[str],
    allowed_edge_types: set[str],
    weight: float,
    retained_edge_types: Counter[str],
    retained_node_types: Counter[str],
    skipped_node_types: Counter[str],
    skipped_edge_types: Counter[str],
    read_kwargs: dict[str, Any] | None = None,
    field_names: tuple[str, str, str, str, str] = ("x_type", "x_name", "y_type", "y_name", "relation"),
) -> None:
    all_targets = {item for vals in retained_nodes.values() for item in vals}
    if not all_targets:
        return
    per_drug_count = {drug_id: 0 for drug_id in retained_nodes}
    options = {"chunksize": 250_000}
    if read_kwargs:
        options.update(read_kwargs)
    x_type_field, x_name_field, y_type_field, y_name_field, relation_field = field_names
    for chunk in pd.read_csv(path, **options):
        for row in chunk.itertuples(index=False):
            x_type = node_type_fn(getattr(row, x_type_field))
            y_type = node_type_fn(getattr(row, y_type_field))
            if "drug" in {x_type, y_type}:
                continue
            edge_type = _canonical_pair_edge_type(x_type, y_type)
            x_name = name_fn(getattr(row, x_name_field))
            y_name = name_fn(getattr(row, y_name_field))
            x_key = (x_type, normalize_name(x_name))
            y_key = (y_type, normalize_name(y_name))
            if x_key not in all_targets and y_key not in all_targets:
                continue
            for drug_id, names in retained_nodes.items():
                if per_drug_count[drug_id] >= max_ppi_neighbors:
                    continue
                if x_key not in names and y_key not in names:
                    continue
                if x_type not in allowed_node_types or y_type not in allowed_node_types:
                    skipped_node_types[edge_type] += 1
                    continue
                if edge_type not in allowed_edge_types:
                    skipped_edge_types[edge_type] += 1
                    continue
                src = node_id(source_name, x_type, x_name)
                dst = node_id(source_name, y_type, y_name)
                _add_edge(
                    edge_rows,
                    source=source_name,
                    src=src,
                    dst=dst,
                    edge_type=edge_type,
                    relation=str(getattr(row, relation_field)),
                    drug_id=drug_id,
                    weight=weight,
                )
                per_drug_count[drug_id] += 1
                retained_edge_types[edge_type] += 1
                _track_row_node_types(retained_node_types, x_type, y_type)


def _add_drkg_edges(
    source_root: Path,
    drug_records: list[dict[str, Any]],
    alias_index: dict[str, dict[str, Any]],
    node_id,
    edge_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    max_ppi_neighbors: int,
    resolved_prior_policy: dict[str, Any],
) -> None:
    path = source_root / "drkg" / "drkg.tar.gz"
    if not path.exists():
        audit_rows.append({"source": "DRKG", "event": "missing_source", "count": 1, "detail": str(path)})
        return
    if not bool(resolved_prior_policy.get("enabled", True)) or float(resolved_prior_policy.get("weight", 1.0)) <= 0.0:
        audit_rows.append({"source": "DRKG", "event": "disabled_by_config", "count": 1, "detail": json.dumps(resolved_prior_policy, sort_keys=True)})
        return
    allowed_node_types = set(resolved_prior_policy.get("node_types") or KG_SOURCE_NODE_TYPES["drkg"])
    allowed_edge_types = set(resolved_prior_policy.get("edge_types") or KG_SOURCE_EDGE_TYPES["drkg"])
    _append_policy_audit(audit_rows, "DRKG", resolved_prior_policy)
    drug_map = _drug_match_maps(drug_records, alias_index)
    retained_nodes: dict[int, set[tuple[str, str]]] = {int(rec["DRUG_ID"]): set() for rec in drug_records}
    retained_edge_types: Counter[str] = Counter()
    retained_node_types: Counter[str] = Counter()
    skipped_node_types: Counter[str] = Counter()
    skipped_edge_types: Counter[str] = Counter()
    try:
        with tarfile.open(path, "r:gz") as tar:
            member = tar.getmember("drkg.tsv")
            f = tar.extractfile(member)
            if f is None:
                raise FileNotFoundError("drkg.tsv not found in DRKG tarball")
            for chunk in pd.read_csv(f, sep="\t", names=["head", "relation", "tail"], chunksize=250_000):
                for row in chunk.itertuples(index=False):
                    h_name = _drkg_name(row.head)
                    t_name = _drkg_name(row.tail)
                    h_key = normalize_name(h_name)
                    t_key = normalize_name(t_name)
                    relation = str(row.relation)
                    h_type = _drkg_kind(row.head)
                    t_type = _drkg_kind(row.tail)
                    for drug_side in ("head", "tail"):
                        rec = _match_drug_record(drug_map, h_key if drug_side == "head" else t_key)
                        if rec is None:
                            continue
                        matched_type = h_type if drug_side == "head" else t_type
                        other_name = t_name if drug_side == "head" else h_name
                        other_kind = t_type if drug_side == "head" else h_type
                        if matched_type != "drug" or other_kind == "drug":
                            continue
                        edge_type = _canonical_pair_edge_type("drug", other_kind)
                        if "drug" not in allowed_node_types or other_kind not in allowed_node_types:
                            skipped_node_types[edge_type] += 1
                            continue
                        if edge_type not in allowed_edge_types:
                            skipped_edge_types[edge_type] += 1
                            continue
                        drug_node = rec["drug_node_ids"]["DRKG"]
                        other_node = node_id("DRKG", other_kind, other_name)
                        _add_edge(
                            edge_rows,
                            source="DRKG",
                            src=drug_node,
                            dst=other_node,
                            edge_type=edge_type,
                            relation=relation,
                            drug_id=rec["DRUG_ID"],
                            weight=float(resolved_prior_policy.get("weight", 1.0)),
                        )
                        retained_nodes[int(rec["DRUG_ID"])].add((other_kind, normalize_name(other_name)))
                        retained_edge_types[edge_type] += 1
                        _track_row_node_types(retained_node_types, "drug", other_kind)
    except Exception as exc:
        audit_rows.append({"source": "DRKG", "event": "read_error", "count": 1, "detail": repr(exc)})
    extracted_path = _extract_drkg_tsv(path)
    if extracted_path is not None and extracted_path.exists():
        _add_source_expansion_edges(
            extracted_path,
            source_name="DRKG",
            retained_nodes=retained_nodes,
            node_id=node_id,
            edge_rows=edge_rows,
            max_ppi_neighbors=max_ppi_neighbors,
            node_type_fn=_drkg_kind,
            name_fn=_drkg_name,
            allowed_node_types=allowed_node_types,
            allowed_edge_types=allowed_edge_types,
            weight=float(resolved_prior_policy.get("weight", 1.0)),
            retained_edge_types=retained_edge_types,
            retained_node_types=retained_node_types,
            skipped_node_types=skipped_node_types,
            skipped_edge_types=skipped_edge_types,
            read_kwargs={"sep": "\t", "names": ["head", "relation", "tail"]},
            field_names=("head", "head", "tail", "tail", "relation"),
        )
    _append_counter_audit(audit_rows, "DRKG", "retained_edge_type", retained_edge_types)
    _append_counter_audit(audit_rows, "DRKG", "retained_node_type", retained_node_types)
    _append_counter_audit(audit_rows, "DRKG", "skipped_by_node_allowlist", skipped_node_types)
    _append_counter_audit(audit_rows, "DRKG", "skipped_by_edge_allowlist", skipped_edge_types)


def _extract_drkg_tsv(path: Path) -> Path | None:
    if not path.exists():
        return None
    try:
        with tarfile.open(path, "r:gz") as tar:
            member = tar.getmember("drkg.tsv")
            extracted = tar.extractfile(member)
            if extracted is None:
                return None
            tmp_dir = Path(path).with_suffix("")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_path = tmp_dir / "drkg.tsv"
            with extracted, out_path.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            return out_path
    except Exception:
        return None


def _add_chembl_edges(
    source_root: Path,
    drug_records: list[dict[str, Any]],
    alias_index: dict[str, dict[str, Any]],
    node_id,
    edge_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    cache_root: Path,
    *,
    chembl_tar_path: Path,
    resolved_prior_policy: dict[str, Any],
) -> None:
    tar_path = Path(chembl_tar_path)
    db_path = _chembl_sqlite_path(tar_path, cache_root)
    if db_path is None or not db_path.exists():
        audit_rows.append({"source": "ChEMBL", "event": "missing_source", "count": 1, "detail": str(tar_path)})
        return
    if not bool(resolved_prior_policy.get("enabled", True)) or float(resolved_prior_policy.get("weight", 1.0)) <= 0.0:
        audit_rows.append({"source": "ChEMBL", "event": "disabled_by_config", "count": 1, "detail": json.dumps(resolved_prior_policy, sort_keys=True)})
        return
    allowed_node_types = set(resolved_prior_policy.get("node_types") or KG_SOURCE_NODE_TYPES["chembl"])
    allowed_edge_types = set(resolved_prior_policy.get("edge_types") or KG_SOURCE_EDGE_TYPES["chembl"])
    _append_policy_audit(audit_rows, "ChEMBL", resolved_prior_policy)
    drug_map = _drug_match_maps(drug_records, alias_index)
    retained_edge_types: Counter[str] = Counter()
    retained_node_types: Counter[str] = Counter()
    skipped_node_types: Counter[str] = Counter()
    skipped_edge_types: Counter[str] = Counter()
    try:
        conn = sqlite3.connect(str(db_path))
        query = """
            SELECT md.pref_name, md.chembl_id AS molecule_chembl_id,
                   cs.canonical_smiles, cs.standard_inchi_key,
                   td.chembl_id AS target_chembl_id, td.pref_name AS target_name,
                   td.target_type, dm.action_type, dm.mechanism_of_action
            FROM drug_mechanism dm
            JOIN molecule_dictionary md ON dm.molregno = md.molregno
            LEFT JOIN compound_structures cs ON dm.molregno = cs.molregno
            LEFT JOIN target_dictionary td ON dm.tid = td.tid
        """
        for chunk in pd.read_sql_query(query, conn, chunksize=100_000):
            for row in chunk.itertuples(index=False):
                candidates = {
                    normalize_name(row.pref_name),
                    normalize_name(row.molecule_chembl_id),
                    normalize_name(row.standard_inchi_key),
                }
                rec = next((drug_map[k] for k in candidates if k in drug_map), None)
                if rec is None:
                    continue
                drug_node = rec["drug_node_ids"]["ChEMBL"]
                target_name = str(row.target_name or row.target_chembl_id)
                action = str(row.action_type or "unknown_action")
                target_type_name = str(row.target_type or "unknown_target_type")
                mechanism_name = str(row.mechanism_of_action or "unknown_mechanism")
                edge_specs = [
                    ("action_type_target", "drug", rec["DRUG_NAME"], "target", target_name, action),
                    ("has_action_type", "drug", rec["DRUG_NAME"], "action_type", action, action),
                    ("acts_on_target", "action_type", action, "target", target_name, action),
                    ("target_has_target_type", "target", target_name, "target_type", target_type_name, target_type_name),
                    ("drug_has_mechanism", "drug", rec["DRUG_NAME"], "mechanism", mechanism_name, mechanism_name),
                    ("mechanism_targets_target", "mechanism", mechanism_name, "target", target_name, mechanism_name),
                ]
                for edge_type, src_type, src_name, dst_type, dst_name, relation in edge_specs:
                    if src_type not in allowed_node_types or dst_type not in allowed_node_types:
                        skipped_node_types[edge_type] += 1
                        continue
                    if edge_type not in allowed_edge_types:
                        skipped_edge_types[edge_type] += 1
                        continue
                    src_node = drug_node if src_type == "drug" and normalize_name(src_name) == rec["canonical_name"] else node_id("ChEMBL", src_type, src_name)
                    dst_node = drug_node if dst_type == "drug" and normalize_name(dst_name) == rec["canonical_name"] else node_id("ChEMBL", dst_type, dst_name)
                    _add_edge(
                        edge_rows,
                        source="ChEMBL",
                        src=src_node,
                        dst=dst_node,
                        edge_type=edge_type,
                        relation=relation,
                        drug_id=rec["DRUG_ID"],
                        weight=float(resolved_prior_policy.get("weight", 1.0)),
                    )
                    retained_edge_types[edge_type] += 1
                    _track_row_node_types(retained_node_types, src_type, dst_type)
        conn.close()
    except Exception as exc:
        audit_rows.append({"source": "ChEMBL", "event": "read_error", "count": 1, "detail": repr(exc)})
    _append_counter_audit(audit_rows, "ChEMBL", "retained_edge_type", retained_edge_types)
    _append_counter_audit(audit_rows, "ChEMBL", "retained_node_type", retained_node_types)
    _append_counter_audit(audit_rows, "ChEMBL", "skipped_by_node_allowlist", skipped_node_types)
    _append_counter_audit(audit_rows, "ChEMBL", "skipped_by_edge_allowlist", skipped_edge_types)


def _chembl_sqlite_path(tar_path: Path, cache_root: Path) -> Path | None:
    candidates = list(cache_root.glob("*.db")) + list(cache_root.glob("*.sqlite")) + list(cache_root.glob("**/*.db")) + list(cache_root.glob("**/*.sqlite"))
    healthy = [p for p in candidates if _sqlite_path_is_healthy(p)]
    if healthy:
        return healthy[0]
    for candidate in candidates:
        try:
            candidate.unlink()
        except OSError:
            pass
    if not tar_path.exists():
        return None
    extract_dir = cache_root / "sqlite_extracted"
    marker = extract_dir / ".extracted"
    if marker.exists():
        candidates = list(extract_dir.glob("**/*.db")) + list(extract_dir.glob("**/*.sqlite"))
        healthy = [p for p in candidates if _sqlite_path_is_healthy(p)]
        if healthy:
            return healthy[0]
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and (m.name.endswith(".db") or m.name.endswith(".sqlite"))]
        for member in members:
            rel_path = Path(member.name.lstrip("/"))
            target_path = extract_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with extracted, target_path.open("wb") as target:
                shutil.copyfileobj(extracted, target)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    candidates = list(extract_dir.glob("**/*.db")) + list(extract_dir.glob("**/*.sqlite"))
    healthy = [p for p in candidates if _sqlite_path_is_healthy(p)]
    return healthy[0] if healthy else None


def _sqlite_path_is_healthy(path: Path) -> bool:
    try:
        conn = sqlite3.connect(str(path))
        try:
            required = {"molecule_dictionary", "drug_mechanism", "target_dictionary"}
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            found = {str(row[0]) for row in rows}
            return required.issubset(found)
        finally:
            conn.close()
    except Exception:
        return False
