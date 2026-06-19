#!/usr/bin/env python3
"""
Layer 4 (part 1): Build LUAD and SKCM disease signatures.

Strategy: TCGA tumor vs GTEx normal tissue.
- LUAD: TCGA-LUAD tumor vs GTEx Lung_Normal
- SKCM: TCGA-SKCM tumor vs GTEx Skin_Sun_Exposed_Lower_leg + Skin_Not_Sun_Exposed_Suprapubic

Uses Mann-Whitney U test (Wilcoxon rank-sum) per gene; logFC = median(tumor) - median(normal).
Limits to LINCS L1000 landmark genes where possible.

Run with: conda run -n kg_GAUGE python v02_disease_signature.py
Output: data/disease_sig/
"""
from __future__ import annotations
import gzip, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
OUT_DIR = ROOT / "data" / "disease_sig"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TCGA_H5AD    = Path("/mnt/raid5/xujing/Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad")
GTEX_MEDIAN  = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/GTEx_V11/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz")
LINCS_GENE   = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/LINCS/GSE70138/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz")

CANCER_GTEX = {
    "TCGA-LUAD": ["Lung"],
    "TCGA-SKCM": ["Skin_Sun_Exposed_Lower_leg", "Skin_Not_Sun_Exposed_Suprapubic"],
}

TOP_N_GENES = 500   # top N up and N down by |logFC| with FDR < 0.05


def load_lincs_landmark_genes() -> set[str]:
    """Return set of LINCS landmark gene symbols."""
    try:
        df = pd.read_csv(LINCS_GENE, sep="\t", compression="gzip")
        # Column names vary; try common ones
        for col in ["gene_symbol", "pr_gene_symbol", "pert_iname"]:
            if col in df.columns:
                return set(df[col].dropna().tolist())
        # fallback: first string column
        return set(df.iloc[:, 1].dropna().tolist())
    except Exception as e:
        print(f"  WARN: could not load LINCS genes: {e}")
        return set()


def load_gtex_median(target_tissues: list[str]) -> pd.DataFrame:
    """Load GTEx median TPM for specific tissues. Returns DataFrame gene×tissue."""
    print(f"  Loading GTEx median for tissues: {target_tissues}")
    with gzip.open(GTEX_MEDIAN, "rt") as f:
        f.readline()  # version
        f.readline()  # dimensions
        header = f.readline().strip().split("\t")
        # Find tissue columns
        tissue_cols = []
        for t in target_tissues:
            for i, h in enumerate(header):
                if t.lower() == h.lower().replace(" ", "_").replace("(", "").replace(")", ""):
                    tissue_cols.append((h, i))
                    break
        if not tissue_cols:
            # Try partial match
            for t in target_tissues:
                for i, h in enumerate(header):
                    if t.lower().replace("_", "") in h.lower().replace(" ", ""):
                        tissue_cols.append((h, i))
                        break

        print(f"  Matched GTEx columns: {[t[0] for t in tissue_cols]}")
        if not tissue_cols:
            print("  WARNING: No GTEx tissue columns matched!")
            return pd.DataFrame()

        name_idx = header.index("Name") if "Name" in header else 0
        desc_idx = header.index("Description") if "Description" in header else 1

        rows = []
        for line in f:
            parts = line.strip().split("\t")
            gene_id   = parts[name_idx]
            gene_name = parts[desc_idx]
            vals = {tc[0]: float(parts[tc[1]]) for tc in tissue_cols}
            vals["gene_id"]   = gene_id
            vals["gene_name"] = gene_name
            rows.append(vals)

    df = pd.DataFrame(rows).set_index("gene_name")
    df.index.name = "gene_name"
    df = df[~df.index.duplicated(keep="first")]  # deduplicate gene names
    tissue_names = [tc[0] for tc in tissue_cols]
    # Mean across matched tissues if multiple
    df["gtex_median_tpm"] = df[tissue_names].mean(axis=1)
    return df[["gtex_median_tpm", "gene_id"]]


def build_disease_signature(
    cancer_type: str,
    gtex_tissues: list[str],
    n_top: int = TOP_N_GENES,
) -> pd.DataFrame:
    print(f"\n  [{cancer_type}] Building disease signature vs GTEx {gtex_tissues}")

    # ── Load TCGA expression ──────────────────────────────────────────────────
    import anndata as ad
    print("  Loading TCGA h5ad...")
    adata = ad.read_h5ad(str(TCGA_H5AD), backed="r")
    tumor_obs = adata.obs[adata.obs["project_id"] == cancer_type].index
    adata_tumor = adata[tumor_obs]
    gene_names  = list(adata.var["gene_name"])
    print(f"  TCGA {cancer_type}: {len(tumor_obs)} samples, {len(gene_names)} genes")

    # Convert to dense array (log TPM → raw TPM)
    tumor_X = np.array(adata_tumor.X)  # shape: (n_tumor, n_genes)
    print(f"  Tumor expression shape: {tumor_X.shape}")
    adata.file.close()

    # ── Load GTEx median ─────────────────────────────────────────────────────
    gtex = load_gtex_median(gtex_tissues)
    if len(gtex) == 0:
        print("  ERROR: could not load GTEx data")
        return pd.DataFrame()

    # ── Align genes ──────────────────────────────────────────────────────────
    common = [g for g in gene_names if g in gtex.index]
    print(f"  Common genes (TCGA ∩ GTEx): {len(common)}")

    gene2idx = {g: i for i, g in enumerate(gene_names)}
    tumor_sub = tumor_X[:, [gene2idx[g] for g in common]]
    gtex_vals = gtex.loc[common, "gtex_median_tpm"].values

    # ── Load LINCS landmark genes ─────────────────────────────────────────────
    landmark = load_lincs_landmark_genes()
    print(f"  LINCS landmark genes: {len(landmark)}")
    landmark_mask = [g in landmark for g in common]
    n_landmark = sum(landmark_mask)
    print(f"  Common genes in LINCS landmarks: {n_landmark}")

    # ── Per-gene test (tumor vs GTEx normal) ──────────────────────────────────
    print(f"  Computing Mann-Whitney U per gene (n_genes={len(common)})...")
    # For each gene: logFC = log2(median tumor + 1) - log2(gtex + 1)
    tumor_median  = np.median(tumor_sub, axis=0)
    log2fc        = np.log2(tumor_median + 1) - np.log2(gtex_vals + 1)

    # Use a sample of cells for speed if too large
    n_cells = tumor_sub.shape[0]
    if n_cells > 300:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, 300, replace=False)
        tumor_sample = tumor_sub[idx]
    else:
        tumor_sample = tumor_sub

    pvals = np.ones(len(common))
    for i in range(len(common)):
        tumor_g = tumor_sample[:, i]
        normal_g = np.array([gtex_vals[i]] * 10)  # GTEx is a single median value
        # If GTEx has only 1 value, compare tumor distribution vs that scalar
        # Use one-sample approach: compare tumor to gtex_val
        # Alternative: use ttest_1samp(tumor, gtex_val)
        from scipy.stats import ttest_1samp
        if np.std(tumor_g) > 0:
            _, p = ttest_1samp(tumor_g, gtex_vals[i])
        else:
            p = 1.0
        pvals[i] = p

    _, fdrs = fdrcorrection(pvals)

    # Build result
    sig_df = pd.DataFrame({
        "gene":      common,
        "logFC":     log2fc,
        "pvalue":    pvals,
        "fdr":       fdrs,
        "in_lincs":  landmark_mask,
    })
    sig_df["direction"] = np.where(sig_df["logFC"] > 0, "up", "down")

    # Filter to significant genes
    sig_fdr = sig_df[sig_df["fdr"] < 0.05].copy()
    print(f"  Significant genes (FDR<0.05): {len(sig_fdr)} "
          f"(up: {(sig_fdr['logFC']>0).sum()}, down: {(sig_fdr['logFC']<0).sum()})")

    # Top N by |logFC| in each direction
    up_genes   = sig_fdr[sig_fdr["logFC"] > 0].nlargest(n_top, "logFC")
    down_genes = sig_fdr[sig_fdr["logFC"] < 0].nsmallest(n_top, "logFC")

    print(f"  Final: top {len(up_genes)} up-genes, top {len(down_genes)} down-genes")
    return sig_df


def main():
    print("=" * 60)
    print("Layer 4 (Part 1): Disease Signature Construction")
    print("=" * 60)

    for cancer_type, tissues in CANCER_GTEX.items():
        sig = build_disease_signature(cancer_type, tissues)
        if len(sig) == 0:
            print(f"  SKIP {cancer_type}: no signature built")
            continue

        tag = cancer_type.replace("TCGA-", "")
        sig.to_csv(OUT_DIR / f"disease_sig_{tag}.csv", index=False)

        sig_fdr  = sig[sig["fdr"] < 0.05]
        up_top   = sig_fdr[sig_fdr["logFC"] > 0].nlargest(TOP_N_GENES, "logFC")
        down_top = sig_fdr[sig_fdr["logFC"] < 0].nsmallest(TOP_N_GENES, "logFC")

        up_top["gene"].to_csv(
            OUT_DIR / f"{tag}_up_genes.txt", index=False, header=False
        )
        down_top["gene"].to_csv(
            OUT_DIR / f"{tag}_down_genes.txt", index=False, header=False
        )

        # Save LINCS-landmark-only version
        sig_lincs = sig[sig["in_lincs"]]
        sig_lincs_fdr = sig_lincs[sig_lincs["fdr"] < 0.05]
        up_lincs   = sig_lincs_fdr[sig_lincs_fdr["logFC"] > 0].nlargest(min(200, len(sig_lincs_fdr)), "logFC")
        down_lincs = sig_lincs_fdr[sig_lincs_fdr["logFC"] < 0].nsmallest(min(200, len(sig_lincs_fdr)), "logFC")
        up_lincs["gene"].to_csv(OUT_DIR / f"{tag}_up_genes_lincs.txt", index=False, header=False)
        down_lincs["gene"].to_csv(OUT_DIR / f"{tag}_down_genes_lincs.txt", index=False, header=False)

        print(f"\n  {cancer_type} summary:")
        print(f"    All sig (FDR<0.05): {len(sig_fdr)} genes")
        print(f"    Top up: {len(up_top)}, top down: {len(down_top)}")
        print(f"    LINCS-subset up: {len(up_lincs)}, down: {len(down_lincs)}")
        print(f"    Saved to {OUT_DIR}/disease_sig_{tag}.csv")

    print(f"\nAll disease signature files → {OUT_DIR}")


if __name__ == "__main__":
    main()
