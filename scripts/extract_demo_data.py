"""One-time extraction of small, real demo datasets for the GAUGE app's
"load demo data" buttons. Pulls compact, representative slices from the
underlying public/research data (never the full files) so every feature
page has genuine example data to show, without bloating example_data/.

Usage:
    python extract_demo_data.py --target tcga
    python extract_demo_data.py --target drugcomb
    python extract_demo_data.py --target gtex
    python extract_demo_data.py --target design
    python extract_demo_data.py --target all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

from gauge_core import load_bundle  # noqa: E402  must precede numpy/pandas import, see gauge_core/_drugwm_path.py

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXAMPLE_DIR = SOFTWARE_ROOT / "example_data"
TCGA_H5AD = Path("/mnt/raid5/xujing/Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad")
DRUGCOMB_CSV = Path("/mnt/raid5/xujing/KG/KG_DrugWM_PublicData/DrugComb/drugcombs_scored.csv")
GTEX_MEDIAN_GCT = Path("/mnt/raid5/xujing/KG/KG_DrugWM_PublicData/GTEx_V11/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz")
DESIGN_DIR = Path("/mnt/raid5/xujing/KG/DrugDesign/05_tcga_egfr_erbb_design/results/my_full_run")


def _model_gene_panel() -> list[str]:
    genes: set[str] = set()
    for mode in ("gdsc_cell_split", "gdsc_drug_split", "prism_cell_split", "prism_drug_split"):
        genes |= set(load_bundle(mode).artifacts.genes)
    return sorted(genes)


def extract_tcga(n_per_drug: int = 4) -> None:
    import anndata as ad

    print("[tcga] loading h5ad (backed mode) ...")
    a = ad.read_h5ad(TCGA_H5AD, backed="r")
    genes = _model_gene_panel()

    single_drug = a.obs["drug"].astype(str)
    is_single = ~single_drug.str.contains(";", na=False) & single_drug.str.strip().ne("") & single_drug.str.lower().ne("nan")
    candidates = a.obs.loc[is_single].copy()
    candidates["drug_norm"] = candidates["drug"].str.strip()

    lib_names = set()
    for mode in ("gdsc_cell_split", "prism_cell_split", "prism_drug_split"):
        lib_names |= set(load_bundle(mode).drug_library["DRUG_NAME"].astype(str).str.lower().str.strip())
    candidates = candidates.loc[candidates["drug_norm"].str.lower().isin(lib_names)]

    # Spread across cancer types and drugs for a varied demo set.
    picked_idx: list[str] = []
    for drug_name, group in candidates.groupby("drug_norm"):
        picked_idx.extend(group.sample(n=min(n_per_drug, len(group)), random_state=0).index.tolist())
    picked_idx = sorted(set(picked_idx))[:200]
    print(f"[tcga] selected {len(picked_idx)} patient samples across {candidates.loc[picked_idx, 'drug_norm'].nunique()} drugs")

    var_gene_name = a.var["gene_name"].astype(str)
    gene_to_cols: dict[str, list[int]] = {}
    for pos, name in enumerate(var_gene_name.to_numpy()):
        if name in genes:
            gene_to_cols.setdefault(name, []).append(pos)

    full = a[picked_idx, :].to_memory()
    x = full.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    expr = pd.DataFrame(0.0, index=picked_idx, columns=sorted(gene_to_cols.keys()), dtype="float32")
    for gene, cols in gene_to_cols.items():
        expr[gene] = x[:, cols].mean(axis=1)
    expr.index.name = "sample_submitter_id"
    expr.to_csv(EXAMPLE_DIR / "example_tcga_patients.csv")

    meta_cols = ["case_submitter_id", "project_id", "primary_site", "disease_type", "vital_status", "drug"]
    meta = a.obs.loc[picked_idx, meta_cols].copy()
    meta.index.name = "sample_submitter_id"
    meta.to_csv(EXAMPLE_DIR / "example_tcga_patients_meta.csv")
    print(f"[tcga] wrote example_tcga_patients.csv ({expr.shape}) + meta ({meta.shape})")


def _normalize_cell_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("-", "", regex=False).str.replace(" ", "", regex=False)


def extract_drugcomb() -> None:
    print("[drugcomb] loading drugcombs_scored.csv ...")
    df = pd.read_csv(DRUGCOMB_CSV)
    # Restricted to the GDSC library specifically (not the PRISM union): the
    # live "compute GAUGE score" comparison in the app always scores against
    # the GDSC bundle (cell-line metadata matching is GDSC-specific too), so
    # both drugs in a demo pair must be resolvable there.
    gdsc_lib_names = set(load_bundle("gdsc_cell_split").drug_library["DRUG_NAME"].astype(str).str.lower().str.strip())

    df["drug1_norm"] = df["Drug1"].astype(str).str.lower().str.strip()
    df["drug2_norm"] = df["Drug2"].astype(str).str.lower().str.strip()
    matched = df.loc[df["drug1_norm"].isin(gdsc_lib_names) & df["drug2_norm"].isin(gdsc_lib_names)].copy()
    print(f"[drugcomb] {len(matched)}/{len(df)} rows have both drugs in the GDSC GAUGE library")

    gdsc_meta = load_bundle("gdsc_cell_split").cell_metadata[["SANGER_MODEL_ID", "model_name"]].copy()
    gdsc_meta["norm_name"] = _normalize_cell_name(gdsc_meta["model_name"])
    name_to_sidm = dict(zip(gdsc_meta["norm_name"], gdsc_meta["SANGER_MODEL_ID"]))
    matched["cell_line_norm"] = _normalize_cell_name(matched["Cell line"])
    matched["matched_gdsc_cell_id"] = matched["cell_line_norm"].map(name_to_sidm)
    n_matched_cells = matched["matched_gdsc_cell_id"].notna().sum()
    print(f"[drugcomb] {n_matched_cells}/{len(matched)} rows also have a bundled GDSC cell-line match")

    keep_cols = ["Drug1", "Drug2", "Cell line", "ZIP", "Bliss", "Loewe", "HSA", "drug1_norm", "drug2_norm", "matched_gdsc_cell_id"]
    matched = matched[keep_cols].dropna(subset=["Bliss"])
    # Bias the sample toward rows that *do* have a cell-line match so the live
    # GAUGE-vs-real-synergy demo comparison always has usable examples.
    with_match = matched.loc[matched["matched_gdsc_cell_id"].notna()]
    without_match = matched.loc[matched["matched_gdsc_cell_id"].isna()]
    n_with = min(len(with_match), 300)
    n_without = min(len(without_match), max(0, 500 - n_with))
    matched = pd.concat(
        [with_match.sample(n=n_with, random_state=0), without_match.sample(n=n_without, random_state=0)]
    )
    matched.to_csv(EXAMPLE_DIR / "example_drugcomb_pairs.csv", index=False)
    print(f"[drugcomb] wrote example_drugcomb_pairs.csv ({matched.shape})")


def extract_gtex() -> None:
    print("[gtex] loading median-TPM GCT (gzip) ...")
    df = pd.read_csv(GTEX_MEDIAN_GCT, sep="\t", skiprows=2)
    id_cols = [c for c in df.columns if c.lower() in {"name", "id", "description"}]
    gene_col = "Description" if "Description" in df.columns else id_cols[0]
    tissue_cols = [c for c in df.columns if c not in {"Name", "id", "Description"}]
    genes = set(_model_gene_panel())
    sub = df.loc[df[gene_col].isin(genes), [gene_col, *tissue_cols]].drop_duplicates(subset=gene_col)
    sub = sub.set_index(gene_col)
    sub.index.name = "gene"
    sub.to_csv(EXAMPLE_DIR / "example_gtex_median_tpm_by_tissue.csv")
    print(f"[gtex] wrote example_gtex_median_tpm_by_tissue.csv ({sub.shape})")


def extract_design() -> None:
    print("[design] copying EGFR/ERBB design candidate subset ...")
    for name in ("generated_candidates.csv", "ranked_candidates.csv", "seed_molecules.csv"):
        src = DESIGN_DIR / name
        if not src.exists():
            print(f"[design] WARNING: missing {src}")
            continue
        df = pd.read_csv(src)
        if len(df) > 300:
            df = df.sort_values(df.columns[-1], ascending=False).head(300)
        df.to_csv(EXAMPLE_DIR / f"example_design_{name}", index=False)
        print(f"[design] wrote example_design_{name} ({df.shape})")


EXTRACTORS = {
    "tcga": extract_tcga,
    "drugcomb": extract_drugcomb,
    "gtex": extract_gtex,
    "design": extract_design,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=[*EXTRACTORS.keys(), "all"], default="all")
    args = parser.parse_args()
    targets = list(EXTRACTORS.keys()) if args.target == "all" else [args.target]
    for t in targets:
        EXTRACTORS[t]()
