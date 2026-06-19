from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path("/mnt/raid5/xujing/KG")
GDSC_SOURCE_MODE_DEFAULT = "v2"
GDSC_SOURCE_MODES = ("v1", "v2", "both")
CHEMBL_RELEASE_DEFAULT = "36"
DATASET_ALIASES = {
    "ctrp": "ctrdb",
    "ctrp_v1": "ctrdb",
}


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _default_ctrp_dir() -> Path:
    return _env_path("GAUGE_CTRP_DIR", ROOT / "KG_GAUGE_PublicData" / "waibu" / "CTRP" / "v2")
DATASET_MODE_DEFAULT = "gdsc"
DATASET_MODES = ("gdsc", "beataml2", "ctrdb", "tcga", "pdx")
DATASET_DEFAULT = DATASET_MODE_DEFAULT
DATASETS = DATASET_MODES


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    shared_resources_dir: Path = ROOT / "shared_resources"
    dataset_smiles_dir: Path = shared_resources_dir / "dataset_smiles"
    gdsc_dir: Path = ROOT / "KG_GAUGE_PublicData" / "GDSC"
    gdsc_fitted_1: Path = gdsc_dir / "GDSC1_fitted_dose_response_27Oct23.xlsx"
    gdsc_fitted_2: Path = gdsc_dir / "GDSC2_fitted_dose_response_27Oct23.xlsx"
    gdsc_screened_compounds: Path = gdsc_dir / "screened_compounds_rel_8.5.csv"
    gdsc_expression: Path = gdsc_dir / "rnaseq_merged_rsem_tpm_20260323.csv"
    gdsc_gene_identifiers: Path = gdsc_dir / "gene_identifiers_20241212.csv"
    beataml2_dir: Path = ROOT / "KG_GAUGE_PublicData" / "waibu" / "BeatAML2"
    beataml2_curve_fits: Path = beataml2_dir / "beataml_probit_curve_fits_v4_dbgap.txt"
    beataml2_raw_inhibitor: Path = beataml2_dir / "beataml_wv1to4_raw_inhibitor_v4_dbgap.txt"
    beataml2_expression: Path = beataml2_dir / "beataml_waves1to4_norm_exp_dbgap.txt"
    beataml2_counts: Path = beataml2_dir / "beataml_waves1to4_counts_dbgap.txt"
    beataml2_drug_families: Path = beataml2_dir / "beataml_drug_families.xlsx"
    beataml2_sample_mapping: Path = beataml2_dir / "beataml_waves1to4_sample_mapping.xlsx"
    beataml2_clinical: Path = beataml2_dir / "beataml_wv1to4_clinical.xlsx"
    beataml2_smiles_cache: Path = dataset_smiles_dir / "beataml2_smiles.csv"
    primekg: Path = ROOT / "KG_GAUGE_PublicData" / "drug" / "PrimeKG" / "kg.csv"
    repurposing_dir: Path = ROOT / "KG_GAUGE_PublicData" / "repurposing"
    gdsc_smiles_cache: Path = dataset_smiles_dir / "gdsc_smiles.csv"
    ctrdb_smiles_cache: Path = dataset_smiles_dir / "ctrdb_smiles.csv"
    tcga_smiles_cache: Path = dataset_smiles_dir / "tcga_smiles.csv"
    ctrp_dir: Path = field(default_factory=_default_ctrp_dir)
    ctrp_response: Path = field(default_factory=lambda: _default_ctrp_dir() / "CTRPv1.csv")
    ctrp_gene_expression: Path = field(default_factory=lambda: _default_ctrp_dir() / "gene_expression.csv")
    ctrp_drug_names: Path = field(default_factory=lambda: _default_ctrp_dir() / "drug_names.csv")
    ctrp_drug_smiles: Path = field(default_factory=lambda: _default_ctrp_dir() / "drug_smiles.csv")
    pdx_dir: Path = ROOT / "KG_GAUGE_PublicData" / "waibu" / "PDX"
    pdx_response: Path = pdx_dir / "pdx_bruna_auc_info.csv"
    pdx_gene_expression: Path = pdx_dir / "gene_expression_pdx_bruna.csv"
    pdx_drug_names: Path = pdx_dir / "drug_names_pdx_bruna.csv"
    pdx_drug_smiles: Path = pdx_dir / "drug_smiles_pdx_bruna.csv"
    tcga_h5ad: Path = Path(
        "/mnt/raid5/xujing/Agent/Datasets/TCGA/h5ad_outputs/"
        "tcga_gene_expression_tpm_therapies_split.h5ad"
    )
    ctrdb_microarray_h5ad: Path = Path("/mnt/raid5/xujing/Agent/Datasets/ctrdbv2/CTR_Microarray.h5ad")


def normalize_gdsc_source_mode(value: str | None) -> str:
    mode = GDSC_SOURCE_MODE_DEFAULT if value is None else str(value).strip().lower()
    if mode not in GDSC_SOURCE_MODES:
        raise ValueError(f"gdsc_source_mode must be one of {', '.join(GDSC_SOURCE_MODES)}; got {value!r}")
    return mode


def normalize_dataset_mode(value: str | None) -> str:
    dataset = DATASET_MODE_DEFAULT if value is None else str(value).strip().lower()
    dataset = DATASET_ALIASES.get(dataset, dataset)
    if dataset not in DATASET_MODES:
        raise ValueError(f"dataset_mode must be one of {', '.join(DATASET_MODES)}; got {value!r}")
    return dataset


def normalize_dataset_name(value: str | None) -> str:
    return normalize_dataset_mode(value)


def gdsc_fitted_paths(paths: Paths, source_mode: str | None = None) -> list[Path]:
    mode = normalize_gdsc_source_mode(source_mode)
    if mode == "v1":
        return [paths.gdsc_fitted_1]
    if mode == "v2":
        return [paths.gdsc_fitted_2]
    return [paths.gdsc_fitted_1, paths.gdsc_fitted_2]


def dataset_response_paths(paths: Paths, dataset_mode: str | None = None) -> list[Path]:
    mode = normalize_dataset_mode(dataset_mode)
    if mode == "gdsc":
        return gdsc_fitted_paths(paths, GDSC_SOURCE_MODE_DEFAULT)
    if mode == "beataml2":
        return [paths.beataml2_curve_fits, paths.beataml2_raw_inhibitor, paths.beataml2_expression, paths.beataml2_counts, paths.beataml2_sample_mapping, paths.beataml2_clinical]
    if mode == "ctrdb":
        return [paths.ctrp_response]
    if mode == "tcga":
        return [paths.tcga_h5ad]
    if mode == "pdx":
        return [paths.pdx_response]
    raise ValueError(f"Unsupported dataset mode: {mode}")


def dataset_smiles_path(paths: Paths, dataset_mode: str | None = None) -> Path:
    mode = normalize_dataset_mode(dataset_mode)
    if mode == "gdsc":
        return paths.gdsc_smiles_cache
    if mode == "beataml2":
        return paths.beataml2_smiles_cache
    if mode == "ctrdb":
        return paths.ctrdb_smiles_cache
    if mode == "tcga":
        return paths.tcga_smiles_cache
    if mode == "pdx":
        return paths.dataset_smiles_dir / "pdx_smiles.csv"
    raise ValueError(f"Unsupported dataset mode: {mode}")


def normalize_chembl_release(value: str | int | None) -> str:
    release = CHEMBL_RELEASE_DEFAULT if value is None else str(value).strip()
    if not release.isdigit():
        raise ValueError(f"chembl release must be numeric; got {value!r}")
    return release


def default_chembl_sqlite_tar(root: Path, release: str | int | None = None) -> Path:
    normalized = normalize_chembl_release(release)
    subdir = "chembl" if normalized == CHEMBL_RELEASE_DEFAULT else f"chembl_{normalized}"
    return Path(root) / "KG_GAUGE_PublicData" / "drug" / subdir / f"chembl_{normalized}_sqlite.tar.gz"


def default_chembl_uniprot_mapping(root: Path) -> Path:
    return Path(root) / "KG_GAUGE_PublicData" / "drug" / "chembl" / "chembl_uniprot_mapping.txt"


def resolve_chembl_sqlite_tar(root: Path, release: str | int | None = None, override: str | None = None) -> Path:
    if override is None or str(override).strip() == "":
        return default_chembl_sqlite_tar(root, release)
    path = Path(str(override)).expanduser()
    if path.is_absolute():
        return path
    return Path(root) / path


DEFAULT_OUTPUT_DIR = ROOT / "GAUGE_runs"
DEFAULT_CACHE_DIR = ROOT / "GAUGE_cache"
