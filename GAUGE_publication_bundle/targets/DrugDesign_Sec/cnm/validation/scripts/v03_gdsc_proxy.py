#!/usr/bin/env python3
"""
Layer 4 (part 2, GDSC-based): Drug sensitivity gene signatures as transcriptome proxy.

Since LINCS L1000 gctx files require >20GB downloads (incomplete), we use GDSC
cell-line RNA-seq + drug sensitivity data to build drug response gene signatures.

For each parent drug (erlotinib/trametinib):
  1. Load GDSC RNAseq TPM for relevant cancer cell lines
  2. Compute per-gene Spearman correlation with LN_IC50 (drug sensitivity)
  3. Sensitivity signature: neg corr = sensitivity gene (high expr → more sensitive)

For analogues:
  proxy_signature = parent_sensitivity_signature × Tanimoto_to_parent
  (structural similarity scales the expected drug effect magnitude)

Run with: conda run -n kg_GAUGE python v03_gdsc_proxy.py
Output: data/lincs/   (same format as LINCS proxy for v04 compatibility)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
OUT_DIR = ROOT / "data" / "lincs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GDSC_DIR    = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/GDSC")
GDSC1_RESP  = GDSC_DIR / "GDSC1_fitted_dose_response_27Oct23.xlsx"
GDSC2_RESP  = GDSC_DIR / "GDSC2_fitted_dose_response_27Oct23.xlsx"
GDSC_RNASEQ = GDSC_DIR / "rnaseq_merged_rsem_tpm_20260323.csv"
GDSC_MODELS = GDSC_DIR / "model_list_20260420.csv"

DRUG_INFO = {
    "erlotinib": {
        "gdsc_file": GDSC1_RESP,
        "drug_id": 1,
        "cancer_types": ["Non-Small Cell Lung Carcinoma"],
        "cancer_type_tcga": "TCGA-LUAD",
    },
    "trametinib": {
        "gdsc_file": GDSC2_RESP,
        "drug_id": 1372,
        "cancer_types": ["Melanoma"],
        "cancer_type_tcga": "TCGA-SKCM",
    },
}

PARENT_SMILES = {
    "erlotinib":  "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "trametinib": "CC1=C(C(=O)N2CCN(CC2)C(=O)c2cc(I)c(F)c(NC(=O)c3ccc(F)cc3Cl)c2)C=NN1CC",
}

MAX_N_GENES = 978  # L1000 landmark genes or top variance genes to use
TOP_K = 5          # top-k nearest GDSC drugs for structural proxy (unused for now)
MIN_CELLS = 8      # minimum cell lines for correlation


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def load_rnaseq(rnaseq_path: Path, model_ids: list[str]) -> pd.DataFrame:
    """Load GDSC RNA-seq TPM for specific cell lines. Returns genes×cells DataFrame."""
    print(f"  Loading GDSC RNAseq for {len(model_ids)} cell lines...")
    # File has complex header: row 0 = model_id, row 1 = gene_id, row 2 = gene_name
    # Actually header row is 0, then 2 metadata rows, then data
    # Let's probe the structure first
    header = pd.read_csv(rnaseq_path, nrows=3, header=None)
    # Row 0: "model_id", "", "", SIDM00001, ...
    # Rows 1+: gene data
    # The first column might be gene identifiers
    # Read with model_id as header
    try:
        df = pd.read_csv(rnaseq_path, index_col=0, header=0, low_memory=False)
    except Exception as e:
        print(f"  Error loading rnaseq: {e}")
        return pd.DataFrame()

    # Get available model IDs
    available = [m for m in model_ids if m in df.columns]
    print(f"  Available cell lines: {len(available)} / {len(model_ids)}")
    if not available:
        return pd.DataFrame()

    sub = df[available].copy()
    # Convert to numeric, drop non-numeric rows (metadata rows)
    sub = sub.apply(pd.to_numeric, errors='coerce')
    sub = sub.dropna(how='all')
    print(f"  Genes after cleaning: {len(sub)}")
    return sub


def compute_drug_sensitivity_signature(
    drug_response: pd.DataFrame,
    rnaseq: pd.DataFrame,
    drug_name: str,
) -> pd.Series:
    """
    Compute gene-drug Spearman correlation across cell lines.
    Returns signed correlation: negative = sensitivity gene (high expr → sensitive).
    """
    common_cells = list(set(drug_response.index) & set(rnaseq.columns))
    print(f"  Common cell lines (drug sensitivity ∩ RNA-seq): {len(common_cells)}")
    if len(common_cells) < MIN_CELLS:
        print(f"  WARNING: too few cell lines ({len(common_cells)} < {MIN_CELLS})")
        return pd.Series(dtype=float)

    ic50 = drug_response.loc[common_cells, "LN_IC50"]
    expr = rnaseq[common_cells]

    # Select high-variance genes to speed up computation
    gene_var = expr.var(axis=1)
    top_var_genes = gene_var.nlargest(min(MAX_N_GENES * 2, len(gene_var))).index
    expr_sub = expr.loc[top_var_genes]

    print(f"  Computing Spearman correlation for {len(expr_sub)} high-variance genes...")
    correlations = {}
    ic50_vals = ic50.values
    for gene in expr_sub.index:
        g_vals = expr_sub.loc[gene, common_cells].values.astype(float)
        # Skip if no variance
        if np.nanstd(g_vals) == 0:
            continue
        # Handle NaN
        mask = ~(np.isnan(g_vals) | np.isnan(ic50_vals))
        if mask.sum() < MIN_CELLS:
            continue
        r, _ = spearmanr(ic50_vals[mask], g_vals[mask])
        correlations[gene] = r

    sig = pd.Series(correlations)
    sig = sig.sort_values()  # most negative first = strongest sensitivity genes
    print(f"  Drug sensitivity signature: {len(sig)} genes, "
          f"range [{sig.min():.3f}, {sig.max():.3f}]")
    return sig


def main():
    print("=" * 60)
    print("Layer 4 (Part 2, GDSC): Drug Sensitivity Gene Signatures")
    print("=" * 60)
    print("  Note: LINCS L1000 gctx requires >20GB download.")
    print("  Using GDSC RNAseq + drug sensitivity as transcriptome proxy.")

    # ── Load GDSC model metadata ──────────────────────────────────────────────
    print("\n[1] Loading GDSC cell line metadata...")
    models = pd.read_csv(GDSC_MODELS)
    print(f"  {len(models)} cell line models")

    # ── Load analogues ────────────────────────────────────────────────────────
    print("\n[2] Loading generated analogues...")
    luad = pd.read_csv(DATA / "candidates" / "generated_compounds_luad.csv")
    skcm = pd.read_csv(DATA / "candidates" / "generated_compounds_skcm.csv")
    luad = luad[luad["tanimoto"] >= 0.10].copy()
    skcm = skcm[skcm["tanimoto"] >= 0.10].copy()
    all_analogues = pd.concat([luad, skcm], ignore_index=True)
    print(f"  Total analogues: {len(all_analogues)}")

    # Pre-compute parent Morgan fingerprints
    parent_fps = {d: morgan_fp(s) for d, s in PARENT_SMILES.items()}

    parent_sigs = {}  # drug_name → gene sensitivity signature pd.Series
    proxy_rows  = []

    for drug_name, info in DRUG_INFO.items():
        print(f"\n[{drug_name.upper()}] Processing drug sensitivity signature...")

        # ── Load drug sensitivity ──────────────────────────────────────────────
        print(f"  Loading GDSC dose-response ({info['gdsc_file'].name})...")
        resp_df = pd.read_excel(info["gdsc_file"])
        drug_resp = resp_df[resp_df["DRUG_ID"] == info["drug_id"]].copy()
        # Filter to relevant cancer types
        cancer_resp = drug_resp[drug_resp["CANCER_TYPE"].isin(info["cancer_types"])]
        print(f"  {len(cancer_resp)} {info['cancer_types']} cell lines with {drug_name} data")
        if len(cancer_resp) == 0:
            print(f"  SKIP: no sensitivity data for {drug_name}")
            continue

        # Map to model_id via SANGER_MODEL_ID
        cancer_resp = cancer_resp.set_index("SANGER_MODEL_ID")

        # ── Get cell line model IDs for RNA-seq ───────────────────────────────
        model_ids = cancer_resp.index.tolist()

        # ── Load RNA-seq for these cell lines ─────────────────────────────────
        rnaseq = load_rnaseq(GDSC_RNASEQ, model_ids)
        if rnaseq.empty:
            print(f"  SKIP: no RNA-seq data matched for {drug_name}")
            continue

        # ── Compute drug sensitivity gene signature ────────────────────────────
        sig = compute_drug_sensitivity_signature(cancer_resp, rnaseq, drug_name)
        if sig.empty:
            continue
        parent_sigs[drug_name] = sig

        # Select top sensitivity genes (most negative = strongest sensitivity)
        n_genes = min(MAX_N_GENES, len(sig))
        top_genes = sig.head(n_genes // 2).index.tolist()  # top sensitivity
        top_resistance = sig.tail(n_genes // 2).index.tolist()  # top resistance
        all_sig_genes = top_genes + top_resistance
        gene_list = list(set(all_sig_genes))
        gene_list.sort()  # stable order

        sig_sub = sig.reindex(gene_list).fillna(0)

        # Save parent signature
        parent_row = {"DRUG_NAME": drug_name, "seed_drug": drug_name}
        for gene in gene_list:
            parent_row[gene] = float(sig_sub.get(gene, 0.0))
        pd.DataFrame([parent_row]).to_csv(
            OUT_DIR / f"parent_signature_{drug_name}.csv", index=False
        )
        print(f"  Parent signature saved: {len(gene_list)} genes")

        # ── Build proxy signatures for analogues ──────────────────────────────
        cancer_type = info["cancer_type_tcga"]
        seed_analogues = all_analogues[all_analogues["cancer_type"] == cancer_type]
        print(f"  Building proxy signatures for {len(seed_analogues)} analogues "
              f"({cancer_type})...")

        parent_fp = parent_fps[drug_name]

        for _, row in seed_analogues.iterrows():
            smi = row["smiles"]
            fp = morgan_fp(smi)
            if fp is None or parent_fp is None:
                continue
            # Tanimoto to parent drug
            tan = tanimoto(fp, parent_fp)

            # Proxy signature = parent signature × tanimoto
            proxy_vec = sig_sub * tan

            proxy_row = {
                "DRUG_NAME":          row["DRUG_NAME"],
                "DRUG_ID":            row["DRUG_ID"],
                "smiles":             smi,
                "seed_drug":          row["seed_drug"],
                "cancer_type":        row["cancer_type"],
                "delta_improvement":  row["delta_improvement"],
                "mean_value_hat":     row["mean_value_hat"],
                "tanimoto":           row["tanimoto"],
                "best_lincs_sim":     tan,
                "best_lincs_name":    drug_name,
            }
            for gene in gene_list:
                proxy_row[gene] = float(proxy_vec.get(gene, 0.0))
            proxy_rows.append(proxy_row)

    if not proxy_rows:
        print("\nERROR: No proxy signatures computed.")
        return

    proxy_df = pd.DataFrame(proxy_rows)
    meta_cols = ["DRUG_NAME", "DRUG_ID", "smiles", "seed_drug", "cancer_type",
                 "delta_improvement", "mean_value_hat", "tanimoto",
                 "best_lincs_sim", "best_lincs_name"]
    gene_cols = [c for c in proxy_df.columns if c not in meta_cols]
    # Fill NaN with 0: analogue from one cancer lacks gene cols from the other cancer
    proxy_df[gene_cols] = proxy_df[gene_cols].fillna(0.0)

    proxy_df.to_csv(OUT_DIR / "proxy_signatures.csv", index=False)
    print(f"\n  Proxy signatures: {len(proxy_df)} rows × {len(gene_cols)} gene cols")
    print(f"  Mean Tanimoto to parent: {proxy_df['best_lincs_sim'].mean():.3f}")

    # Save gene list
    pd.DataFrame({"gene_id": gene_cols, "gene_symbol": gene_cols}).to_csv(
        OUT_DIR / "lincs_landmark_genes.csv", index=False
    )

    print(f"\nAll outputs → {OUT_DIR}")
    print(f"  proxy_signatures.csv: {len(proxy_df)} analogues")
    for drug_name in DRUG_INFO:
        print(f"  parent_signature_{drug_name}.csv: "
              f"{'saved' if (OUT_DIR / f'parent_signature_{drug_name}.csv').exists() else 'missing'}")

    print("\n  Note: proxy_signature = parent_drug_sensitivity_signature × Tanimoto_to_parent")
    print("  This uses GDSC LN_IC50 vs RNAseq correlations as drug effect proxy.")


if __name__ == "__main__":
    main()
