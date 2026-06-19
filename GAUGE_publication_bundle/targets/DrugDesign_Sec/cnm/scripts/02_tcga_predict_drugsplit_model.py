#!/usr/bin/env python3
"""
TCGA drug sensitivity predictions using the drug-split model.

Model: /mnt/raid5/xujing/KG/PRISM/Secondary/cheml35/results/20260524_224312
This model was trained with a drug split (1040 train / 149 val / 298 test drugs).
ALL 1487 drugs are predicted (including val/test held-outs).

Pipeline:
  1. Load drug-split model and prepared.pkl (19,215 gene input → 477-dim PCA state)
  2. Load GDSC expression to calibrate quantile mapping
  3. Load TCGA h5ad, extract expression for target cancer types
  4. Gene-wise quantile mapping: TCGA → GDSC distribution (domain adaptation)
  5. Project: expression → imputer → scaler → PCA → 477-dim state (pad to 480)
  6. Run model for each (patient, drug) pair → auc_hat, value_hat, uncertainty
  7. Annotate with drug split (train/val/test)

Cancer types: LUAD, SKCM, BRCA, PRAD, HNSC (selected for biological relevance)

Outputs:
  results/tcga_drugsplit_predictions.parquet  — full predictions
  results/tcga_drugsplit_predictions_summary.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pickle
from GAUGE.train import load_model
from GAUGE.external import _predict_many_pairs

OUT_DIR  = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR  = ROOT / "PRISM/Secondary/cheml35/results/20260524_224312"
TCGA_H5AD = ROOT.parent / "Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad"
GDSC_EXPR = ROOT / "KG_GAUGE_PublicData/GDSC/rnaseq_merged_rsem_tpm_20260323.csv"
ARTIFACTS_PKL = RUN_DIR / "artifacts.pkl"
SPLIT_AUDIT   = RUN_DIR / "split_audit.csv"
PRED_CSV      = RUN_DIR / "predictions.csv"

TARGET_PROJECTS = ["TCGA-LUAD", "TCGA-SKCM", "TCGA-BRCA", "TCGA-PRAD", "TCGA-HNSC"]
DEVICE = "cuda:6"
BATCH_SIZE = 16384


# ── helpers ──────────────────────────────────────────────────────────────────

def _quantile_map_gene(source_col: np.ndarray, target_col: np.ndarray) -> np.ndarray:
    """Map target_col quantiles to match source_col distribution."""
    src = np.asarray(source_col, dtype=np.float32)
    tgt = np.asarray(target_col, dtype=np.float32)
    src_sorted = np.sort(src)
    ranks = np.argsort(np.argsort(tgt, kind="mergesort"), kind="mergesort").astype(np.float32)
    quantiles = (ranks + 0.5) / max(len(tgt), 1)
    grid = np.linspace(0.0, 1.0, num=len(src_sorted), endpoint=False, dtype=np.float32) + (
        0.5 / max(len(src_sorted), 1)
    )
    return np.interp(quantiles, grid, src_sorted,
                     left=src_sorted[0], right=src_sorted[-1]).astype(np.float32)


def load_gdsc_expression(gene_symbols: list[str]) -> np.ndarray:
    """Load GDSC expression matrix [cells × genes], aligned to gene_symbols."""
    print("  Loading GDSC expression for quantile mapping reference...")
    raw = pd.read_csv(GDSC_EXPR, index_col=0, header=None, low_memory=False)
    # Structure: 4 header rows (model_id, model_name, data_source, gene_symbol)
    # Columns 1-2 are ensembl_gene_id and gene_id (non-numeric metadata)
    # Gene symbols are the row index starting from row 4
    # Data values are at rows 4+, columns 2+ (shape: [n_genes, n_cells])
    gene_names = raw.index[4:].astype(str).tolist()
    data = raw.iloc[4:, 2:].to_numpy(dtype=np.float32)  # [n_genes × n_cells]

    gsym2row = {g: i for i, g in enumerate(gene_names)}
    n_cells = data.shape[1]
    mat = np.zeros((n_cells, len(gene_symbols)), dtype=np.float32)
    n_found = 0
    for pos, g in enumerate(gene_symbols):
        if g in gsym2row:
            mat[:, pos] = data[gsym2row[g], :]
            n_found += 1
    print(f"  GDSC expression loaded: {n_cells} cells × {len(gene_symbols)} genes "
          f"({n_found} genes found)")
    return mat  # shape: [n_cells × n_genes]


def load_tcga_expression(
    h5ad_path: Path,
    gene_symbols: list[str],
    target_projects: list[str],
    batch_size: int = 512,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load TCGA expression [patients × genes] for target projects."""
    print(f"  Loading TCGA h5ad: {h5ad_path}")
    data = ad.read_h5ad(h5ad_path, backed="r")

    if "gene_name" in data.var.columns:
        names = data.var["gene_name"].astype(str).tolist()
    else:
        names = [str(x).split(".")[0] for x in data.var_names]
    gene_to_idx: dict[str, int] = {}
    for i, g in enumerate(names):
        if g and g not in gene_to_idx:
            gene_to_idx[g] = i

    # Strip format like "TSPAN6 (7105)" → "TSPAN6"
    clean_syms = [g.split(" (")[0] if " (" in g else g for g in gene_symbols]
    present_pos = [(pos, gene_to_idx[g]) for pos, g in enumerate(clean_syms) if g in gene_to_idx]
    target_pos = [x[0] for x in present_pos]
    source_idx = [x[1] for x in present_pos]
    n_missing = len(gene_symbols) - len(present_pos)
    print(f"  Gene match: {len(present_pos)}/{len(gene_symbols)} found "
          f"({n_missing} missing → zero-padded)")

    # Filter to target projects
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    if target_projects:
        mask = obs["project_id"].isin(target_projects)
        obs = obs[mask]
    print(f"  Target samples: {len(obs)} across projects: "
          f"{obs['project_id'].value_counts().to_dict()}")

    sample_indices = [data.obs_names.get_loc(s) for s in obs.index]
    mat = np.zeros((len(sample_indices), len(gene_symbols)), dtype=np.float32)

    for b_start in range(0, len(sample_indices), batch_size):
        b_end = min(b_start + batch_size, len(sample_indices))
        b_idx = sample_indices[b_start:b_end]
        x = data.X[b_idx, :]
        if sparse.issparse(x):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        if source_idx:
            mat[b_start:b_end, target_pos] = x[:, source_idx]
        if b_start % (batch_size * 4) == 0:
            print(f"    Loaded {b_end}/{len(sample_indices)} patients...")

    return mat, obs


def apply_quantile_mapping(
    tcga_mat: np.ndarray,
    gdsc_mat: np.ndarray,
) -> np.ndarray:
    """Apply per-gene quantile mapping: TCGA → GDSC distribution."""
    print(f"  Quantile mapping {tcga_mat.shape[0]} patients × {tcga_mat.shape[1]} genes...")
    out = np.empty_like(tcga_mat)
    for j in range(tcga_mat.shape[1]):
        gdsc_col = gdsc_mat[:, j]
        tcga_col = tcga_mat[:, j]
        # Only map if GDSC column is non-constant
        if gdsc_col.std() < 1e-8 or tcga_col.std() < 1e-8:
            out[:, j] = tcga_col
        else:
            out[:, j] = _quantile_map_gene(gdsc_col, tcga_col)
        if j % 2000 == 0:
            print(f"    gene {j}/{tcga_mat.shape[1]}...")
    return out


def project_to_state(
    expr_mat: np.ndarray,
    artifacts,
    target_state_dim: int = 480,
) -> np.ndarray:
    """Project expression [n × 19215] → state [n × 480]."""
    imputed = artifacts.imputer.transform(expr_mat)
    scaled  = artifacts.scaler.transform(imputed)
    pca_out = artifacts.pca.transform(scaled).astype(np.float32)
    n, d = pca_out.shape
    if d < target_state_dim:
        pad = np.zeros((n, target_state_dim - d), dtype=np.float32)
        pca_out = np.concatenate([pca_out, pad], axis=1)
    elif d > target_state_dim:
        pca_out = pca_out[:, :target_state_dim]
    return pca_out


def build_all_drugs_payload(artifacts) -> list[dict]:
    """All drugs in the model's drug table as payload dicts."""
    payloads = []
    for row in artifacts.drug_table.itertuples(index=False):
        payloads.append({
            "DRUG_ID":    int(row.DRUG_ID),
            "DRUG_NAME":  row.DRUG_NAME,
            "fingerprint": row.fingerprint.astype(np.float32),
            "prior":       row.prior.astype(np.float32),
            "prior_mask":  float(row.prior_mask),
        })
    return payloads


def predict_tcga(
    model,
    states:   np.ndarray,
    obs:      pd.DataFrame,
    all_drugs: list[dict],
    device:   str,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    entity_ids     = obs.index.tolist()
    drugs_by_entity = [all_drugs] * len(entity_ids)
    print(f"  Running inference: {len(entity_ids)} patients × {len(all_drugs)} drugs "
          f"= {len(entity_ids)*len(all_drugs):,} pairs...")
    t0 = time.time()
    df = _predict_many_pairs(
        model, states, entity_ids, drugs_by_entity,
        device=device, batch_size=batch_size,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    df["project_id"] = df["entity_id"].map(obs["project_id"].to_dict())
    return df


def annotate_split(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Add train/val/test split label to each prediction row."""
    train_preds = pd.read_csv(
        PRED_CSV,
        usecols=["DRUG_ID", "split"],
    ).drop_duplicates("DRUG_ID").set_index("DRUG_ID")["split"].to_dict()
    pred_df["split"] = pred_df["DRUG_ID"].map(train_preds).fillna("unknown")
    return pred_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--projects", nargs="+", default=TARGET_PROJECTS)
    args = parser.parse_args()

    print("=" * 60)
    print("TCGA Virtual Drug Screening with Drug-Split Model")
    print(f"  Run dir: {RUN_DIR}")
    print(f"  Projects: {args.projects}")
    print("=" * 60)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("\n[1] Loading model and artifacts...")
    with open(ARTIFACTS_PKL, "rb") as _f:
        art = pickle.load(_f)
    model = load_model(RUN_DIR, art)
    model = model.to(args.device).eval()
    gene_symbols = list(art.genes)  # e.g. "TSPAN6 (7105)"
    clean_syms   = [g.split(" (")[0] if " (" in g else g for g in gene_symbols]
    print(f"  Genes: {len(gene_symbols)}, State dim: {art.pca.n_components_}")
    print(f"  Drugs: {len(art.drug_table)}")

    # ── 2. Load GDSC reference expression ────────────────────────────────────
    print("\n[2] Loading GDSC reference expression for quantile mapping...")
    gdsc_mat = load_gdsc_expression(clean_syms)

    # ── 3. Load TCGA expression ───────────────────────────────────────────────
    print("\n[3] Loading TCGA expression...")
    tcga_mat, obs = load_tcga_expression(TCGA_H5AD, gene_symbols, args.projects)
    print(f"  Loaded: {tcga_mat.shape[0]} patients × {tcga_mat.shape[1]} genes")

    # ── 4. Quantile mapping ───────────────────────────────────────────────────
    print("\n[4] Applying gene-wise quantile mapping (TCGA → GDSC distribution)...")
    tcga_mapped = apply_quantile_mapping(tcga_mat, gdsc_mat)
    del gdsc_mat  # free memory

    # ── 5. Project to state space ─────────────────────────────────────────────
    print("\n[5] Projecting to model state space...")
    states = project_to_state(tcga_mapped, art, target_state_dim=480)
    del tcga_mapped
    print(f"  States shape: {states.shape}")

    # ── 6. Build drug payloads ────────────────────────────────────────────────
    print("\n[6] Building drug payloads...")
    all_drugs = build_all_drugs_payload(art)
    print(f"  Total drugs: {len(all_drugs)}")

    # ── 7. Run model ──────────────────────────────────────────────────────────
    print("\n[7] Running model predictions...")
    pred_df = predict_tcga(model, states, obs, all_drugs, device=args.device)

    # ── 8. Annotate drug split ────────────────────────────────────────────────
    print("\n[8] Annotating drug splits...")
    pred_df = annotate_split(pred_df)

    # ── 9. Save ───────────────────────────────────────────────────────────────
    print("\n[9] Saving predictions...")
    pred_df.to_parquet(OUT_DIR / "tcga_drugsplit_predictions.parquet", index=False)
    print(f"  Saved {len(pred_df):,} rows to tcga_drugsplit_predictions.parquet")

    # Summary
    n_by_split = pred_df.groupby("split")["DRUG_ID"].nunique().to_dict()
    summary = {
        "model_run_dir": str(RUN_DIR),
        "n_patients": int(obs.shape[0]),
        "n_drugs": int(len(all_drugs)),
        "n_predictions": int(len(pred_df)),
        "cancer_types": args.projects,
        "n_patients_by_project": obs["project_id"].value_counts().to_dict(),
        "n_drugs_by_split": n_by_split,
        "value_hat_stats": {
            "mean":   round(float(pred_df["value_hat"].mean()), 4),
            "std":    round(float(pred_df["value_hat"].std()),  4),
            "median": round(float(pred_df["value_hat"].median()), 4),
        },
    }
    with open(OUT_DIR / "tcga_drugsplit_predictions_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
