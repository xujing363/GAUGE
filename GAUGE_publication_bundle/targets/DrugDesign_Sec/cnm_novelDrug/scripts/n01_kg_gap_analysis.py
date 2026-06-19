#!/usr/bin/env python3
"""
n01_kg_gap_analysis.py
======================
Quantify the contribution of the three KG prior networks (ChEMBL, DRKG, PrimeKG)
to drug sensitivity predictions, compared to fingerprint-only inference.

Scientific question:
  When we score novel BRICS-generated analogues with drug_idx=None (fingerprint-only),
  how much prediction signal are we losing relative to full KG-augmented inference?

Approach:
  For drugs that HAVE KG entries (ChEMBL/PrimeKG coverage > 0), compare:
    Mode A:  drug_idx=None  →  z_prior = 0  (fingerprint-only)
    Mode B:  drug_idx=local  →  z_prior = z_kg  (full KG augmentation)
  Focus on MoA classes relevant to the seed drugs:
    - EGFR inhibitors (for erlotinib analogues, LUAD)
    - MEK inhibitors  (for trametinib analogues, SKCM)

Key outputs:
  - Pearson/Spearman r between fp-only and full-KG per-patient value_hat
  - Per-MoA class comparison
  - Drug-level scatter of mean value_hat (fp-only vs full-KG)
  - Implication: if r is high, fp-only is a reliable proxy for novel drugs

Outputs:
  results/kg_contribution_gap.csv
  results/kg_contribution_gap_summary.json
"""
from __future__ import annotations

import json
import sys
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from GAUGE.train import load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
CNM_DIR      = ROOT / "DrugDesign_Sec/cnm"
OUT_DIR      = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR       = ROOT / "PRISM/Secondary/cheml35/results/20260524_224312"
ARTIFACTS_PKL = RUN_DIR / "artifacts.pkl"
PREDS_PARQUET = CNM_DIR / "results/tcga_drugsplit_predictions.parquet"
DRUG_SPLIT_CSV = CNM_DIR / "results/drug_split_validation.csv"

DEVICE = "cuda:6"

# MoA classes of interest and their cancer-type context
MOA_FOCUS = {
    "EGFR inhibitor": "TCGA-LUAD",
    "MEK inhibitor":  "TCGA-SKCM",
}

# Known MoA assignments (expand from cnm pipeline_description.md)
MOA_DRUGS = {
    "EGFR inhibitor": [
        "gefitinib", "osimertinib", "afatinib", "erlotinib",
        "dacomitinib", "neratinib", "lapatinib",
    ],
    "MEK inhibitor": [
        "trametinib", "cobimetinib", "selumetinib", "binimetinib",
        "PD-0325901", "AZD8330",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)


def score_drug(
    model: torch.nn.Module,
    states: np.ndarray,
    drug_fp: np.ndarray,
    drug_local_idx: int | None,
    device: str,
    precomputed_kg_payload: dict | None = None,
) -> np.ndarray:
    """
    Score a drug against a batch of patient states.

    drug_local_idx=None  → fingerprint-only mode (z_prior = 0)
    drug_local_idx=N     → full KG mode (z_prior = z_kg from KG encoder)
    """
    n_ent = states.shape[0]
    n_prior = model.prior_adapter.prior_dim if hasattr(model, "prior_adapter") else 11
    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    fp_t    = torch.tensor(drug_fp[np.newaxis, :], dtype=torch.float32, device=device).expand(n_ent, -1)
    prior_t = torch.zeros(n_ent, n_prior, dtype=torch.float32, device=device)
    mask_t  = torch.zeros(n_ent, 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        if drug_local_idx is None:
            out = model(state_t, fp_t, prior_t, mask_t, drug_idx=None)
        else:
            idx_t = torch.full((n_ent,), drug_local_idx, dtype=torch.long, device=device)
            out = model(
                state_t, fp_t, prior_t, mask_t,
                drug_idx=idx_t,
                precomputed_kg_payload=precomputed_kg_payload,
            )

    return out["value_hat"].cpu().numpy().astype(np.float32)


# ── Patient state loading (re-used from script 04) ───────────────────────────

GDSC_EXPR = ROOT / "KG_GAUGE_PublicData/GDSC/rnaseq_merged_rsem_tpm_20260323.csv"
TCGA_H5AD = ROOT.parent / "Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad"


def load_gdsc_efficiently():
    _hdr = pd.read_csv(GDSC_EXPR, header=None, nrows=1)
    n_cols = _hdr.shape[1]
    use_cols = [0] + list(range(3, n_cols))
    gdsc_raw = pd.read_csv(GDSC_EXPR, index_col=0, header=None, skiprows=4, usecols=use_cols)
    return gdsc_raw.index.astype(str).tolist(), gdsc_raw.to_numpy(dtype=np.float32)


def load_cancer_states(art, cancer_type: str, entity_ids: list[str]) -> np.ndarray | None:
    import anndata as ad
    from scipy import sparse

    gene_symbols = list(art.genes)
    clean_syms = [g.split(" (")[0] if " (" in g else g for g in gene_symbols]

    print(f"    Loading GDSC expression for quantile mapping...")
    gdsc_gene_names, gdsc_data = load_gdsc_efficiently()
    gsym2row = {g: i for i, g in enumerate(gdsc_gene_names)}
    n_gdsc_cells = gdsc_data.shape[1]
    gdsc_mat = np.zeros((n_gdsc_cells, len(clean_syms)), dtype=np.float32)
    for pos, g in enumerate(clean_syms):
        if g in gsym2row:
            gdsc_mat[:, pos] = gdsc_data[gsym2row[g], :]

    print(f"    Loading TCGA expression for {cancer_type}...")
    data = ad.read_h5ad(TCGA_H5AD, backed="r")
    obs = data.obs.copy()
    obs.index = data.obs_names.to_list()
    mask = obs.index.isin(entity_ids)
    obs_sub = obs[mask]

    if "gene_name" in data.var.columns:
        tcga_names = data.var["gene_name"].astype(str).tolist()
    else:
        tcga_names = [str(x).split(".")[0] for x in data.var_names]
    gene_to_idx: dict[str, int] = {}
    for i, g in enumerate(tcga_names):
        if g and g not in gene_to_idx:
            gene_to_idx[g] = i

    present_pos = [(pos, gene_to_idx[g]) for pos, g in enumerate(clean_syms) if g in gene_to_idx]
    target_pos = [x[0] for x in present_pos]
    source_idx = [x[1] for x in present_pos]

    sample_indices = [data.obs_names.get_loc(s) for s in obs_sub.index]
    tcga_mat = np.zeros((len(sample_indices), len(gene_symbols)), dtype=np.float32)
    CHUNK = 512
    for b in range(0, len(sample_indices), CHUNK):
        b_end = min(b + CHUNK, len(sample_indices))
        b_idx = sample_indices[b:b_end]
        x = data.X[b_idx, :]
        if sparse.issparse(x):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        if source_idx:
            tcga_mat[b:b_end, target_pos] = x[:, source_idx]

    out = np.empty_like(tcga_mat)
    for j in range(tcga_mat.shape[1]):
        gc = gdsc_mat[:, j]
        tc = tcga_mat[:, j]
        if gc.std() < 1e-8 or tc.std() < 1e-8:
            out[:, j] = tc
        else:
            src_sorted = np.sort(gc)
            ranks = np.argsort(np.argsort(tc, kind="mergesort")).astype(np.float32)
            quantiles = (ranks + 0.5) / max(len(tc), 1)
            grid = np.linspace(0.0, 1.0, num=len(src_sorted), endpoint=False) + 0.5 / max(len(src_sorted), 1)
            out[:, j] = np.interp(quantiles, grid, src_sorted, left=src_sorted[0], right=src_sorted[-1]).astype(np.float32)

    imputed = art.imputer.transform(out)
    scaled = art.scaler.transform(imputed)
    pca_out = art.pca.transform(scaled).astype(np.float32)
    n, d = pca_out.shape
    target_state_dim = 480
    if d < target_state_dim:
        pca_out = np.concatenate([pca_out, np.zeros((n, target_state_dim - d), dtype=np.float32)], axis=1)
    elif d > target_state_dim:
        pca_out = pca_out[:, :target_state_dim]
    return pca_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("KG Contribution Gap Analysis: Full-KG vs Fingerprint-Only Inference")
    print("=" * 70)

    # Load artifacts and model
    print("\n[1] Loading model and artifacts...")
    with open(ARTIFACTS_PKL, "rb") as f:
        art = pickle.load(f)
    model = load_model(RUN_DIR, art)
    model = model.to(DEVICE).eval()

    kg = art.kg_graph
    drug_df = art.drug_table
    drug_split_df = pd.read_csv(DRUG_SPLIT_CSV)

    print(f"  Total PRISM drugs: {len(drug_df)}")
    print(f"  Drugs in KG: {kg.n_drugs} (ChEMBL/PrimeKG coverage)")
    print(f"  ChEMBL coverage: {kg.coverage['has_ChEMBL'].sum()} / {kg.n_drugs}")
    print(f"  PrimeKG coverage: {kg.coverage['has_PrimeKG'].sum()} / {kg.n_drugs}")

    # Precompute KG payload (branch embeddings for all KG drugs) once
    print("\n[2] Precomputing KG branch embeddings...")
    t0 = time.time()
    precomputed_kg = model.precompute_kg_payload(device=DEVICE)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  branch_all shape: {precomputed_kg['branch_all'].shape}")

    # Load TCGA patient IDs per cancer type from predictions
    print("\n[3] Loading TCGA patient IDs from predictions parquet...")
    preds = pd.read_parquet(PREDS_PARQUET)
    cancer_entity_ids: dict[str, list[str]] = {}
    for ct in ["TCGA-LUAD", "TCGA-SKCM"]:
        cancer_entity_ids[ct] = preds[preds["project_id"] == ct]["entity_id"].unique().tolist()
        print(f"  {ct}: {len(cancer_entity_ids[ct])} patients")

    # Load patient states for each cancer type
    print("\n[4] Loading patient states...")
    cancer_states: dict[str, np.ndarray] = {}
    for ct in ["TCGA-LUAD", "TCGA-SKCM"]:
        print(f"\n  Loading {ct}...")
        cancer_states[ct] = load_cancer_states(art, ct, cancer_entity_ids[ct])
        print(f"  States: {cancer_states[ct].shape}")

    # Find drugs with KG coverage for each MoA class
    print("\n[5] Identifying drugs with KG coverage for MoA classes of interest...")

    moa_drugs_in_kg: dict[str, list[dict]] = {}
    for moa_class, cancer_type in MOA_FOCUS.items():
        drug_names = MOA_DRUGS[moa_class]
        found = []
        for dname in drug_names:
            rows = drug_df[drug_df["DRUG_NAME"].str.lower().str.contains(dname.lower())]
            for _, row in rows.iterrows():
                drug_id = int(row["DRUG_ID"])
                in_kg = drug_id in kg.drug_to_local
                split_rows = drug_split_df[drug_split_df["drug_id"] == drug_id]
                split = split_rows.iloc[0]["split"] if len(split_rows) > 0 else "unknown"
                pcc = float(split_rows.iloc[0]["pcc_value_hat"]) if len(split_rows) > 0 else None
                smiles = row.get("canonical_smiles") or row.get("smiles", "")
                fp = morgan_fp(str(smiles)) if smiles else None
                entry = {
                    "drug_id": drug_id,
                    "drug_name": str(row["DRUG_NAME"]),
                    "split": split,
                    "pcc_value_hat": pcc,
                    "in_kg": in_kg,
                    "local_idx": kg.drug_to_local.get(drug_id),
                    "smiles": str(smiles),
                    "fp": fp,
                    "moa_class": moa_class,
                    "cancer_type": cancer_type,
                }
                # KG coverage details
                if in_kg:
                    cov_row = kg.coverage[kg.coverage["DRUG_ID"] == drug_id]
                    if len(cov_row) > 0:
                        c = cov_row.iloc[0]
                        entry["has_ChEMBL"] = float(c.get("has_ChEMBL", 0))
                        entry["has_PrimeKG"] = float(c.get("has_PrimeKG", 0))
                        entry["kg_degree_ChEMBL"] = float(c.get("graph_degree_ChEMBL", 0))
                        entry["kg_degree_PrimeKG"] = float(c.get("graph_degree_PrimeKG", 0))
                    else:
                        entry["has_ChEMBL"] = 0.0
                        entry["has_PrimeKG"] = 0.0
                        entry["kg_degree_ChEMBL"] = 0.0
                        entry["kg_degree_PrimeKG"] = 0.0
                else:
                    entry["has_ChEMBL"] = 0.0
                    entry["has_PrimeKG"] = 0.0
                    entry["kg_degree_ChEMBL"] = 0.0
                    entry["kg_degree_PrimeKG"] = 0.0
                found.append(entry)
        moa_drugs_in_kg[moa_class] = found
        n_with_kg = sum(1 for d in found if d["in_kg"])
        print(f"  {moa_class}: {len(found)} drug entries, {n_with_kg} with KG coverage")
        for d in found:
            kg_tag = f"KG(ChEMBL={d['has_ChEMBL']:.0f},PrimeKG={d['has_PrimeKG']:.0f},deg={d['kg_degree_ChEMBL']:.0f}/{d['kg_degree_PrimeKG']:.0f})" if d["in_kg"] else "NO_KG"
            print(f"    [{d['split']:5s}] {d['drug_name'][:35]:<35} {kg_tag}")

    # Score drugs: fp-only vs full-KG
    print("\n[6] Scoring drugs: fp-only vs full-KG (precomputed branch embeddings)...")
    records = []

    for moa_class, cancer_type in MOA_FOCUS.items():
        states = cancer_states[cancer_type]
        n_pat = states.shape[0]
        drugs = moa_drugs_in_kg[moa_class]

        for drug_info in drugs:
            if drug_info["fp"] is None:
                print(f"  SKIP {drug_info['drug_name']}: no valid SMILES")
                continue

            print(f"  [{cancer_type}] {drug_info['drug_name']} ({drug_info['split']}, KG={drug_info['in_kg']})...")

            # Mode A: fingerprint-only (drug_idx=None)
            vh_fp = score_drug(
                model, states, drug_info["fp"],
                drug_local_idx=None, device=DEVICE,
            )

            # Mode B: full KG (drug_idx=local_idx) — only if drug is in KG
            if drug_info["in_kg"] and drug_info["local_idx"] is not None:
                vh_kg = score_drug(
                    model, states, drug_info["fp"],
                    drug_local_idx=drug_info["local_idx"], device=DEVICE,
                    precomputed_kg_payload=precomputed_kg,
                )
                has_kg = True
            else:
                vh_kg = np.full(n_pat, np.nan, dtype=np.float32)
                has_kg = False

            # Compare
            from scipy.stats import pearsonr, spearmanr
            valid = ~np.isnan(vh_kg)
            if valid.sum() > 10:
                r_pearson, p_pearson = pearsonr(vh_fp[valid], vh_kg[valid])
                r_spearman, p_spearman = spearmanr(vh_fp[valid], vh_kg[valid])
                mean_abs_diff = float(np.abs(vh_fp[valid] - vh_kg[valid]).mean())
            else:
                r_pearson = r_spearman = p_pearson = p_spearman = mean_abs_diff = None

            rec = {
                "drug_id": drug_info["drug_id"],
                "drug_name": drug_info["drug_name"],
                "moa_class": moa_class,
                "cancer_type": cancer_type,
                "split": drug_info["split"],
                "in_kg": has_kg,
                "has_ChEMBL": drug_info["has_ChEMBL"],
                "has_PrimeKG": drug_info["has_PrimeKG"],
                "kg_degree_ChEMBL": drug_info["kg_degree_ChEMBL"],
                "kg_degree_PrimeKG": drug_info["kg_degree_PrimeKG"],
                "n_patients": n_pat,
                "mean_vh_fp": float(vh_fp.mean()),
                "std_vh_fp": float(vh_fp.std()),
                "mean_vh_kg": float(np.nanmean(vh_kg)) if has_kg else None,
                "std_vh_kg": float(np.nanstd(vh_kg)) if has_kg else None,
                "pearson_r": r_pearson,
                "pearson_p": p_pearson,
                "spearman_r": r_spearman,
                "spearman_p": p_spearman,
                "mean_abs_diff": mean_abs_diff,
                "delta_mean_vh": float(np.nanmean(vh_kg) - vh_fp.mean()) if has_kg else None,
            }
            records.append(rec)

            if has_kg:
                print(f"    mean_vh: fp={vh_fp.mean():.4f}, kg={np.nanmean(vh_kg):.4f}  "
                      f"Pearson r={r_pearson:.4f}  Spearman r={r_spearman:.4f}  MAD={mean_abs_diff:.4f}")
            else:
                print(f"    mean_vh (fp-only): {vh_fp.mean():.4f}")

    # Save results
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "kg_contribution_gap.csv", index=False)
    print(f"\n  Saved kg_contribution_gap.csv")

    # Summary statistics
    has_kg_df = df[df["in_kg"] == True].copy()
    print("\n[7] Summary statistics:")

    summary = {
        "n_drugs_analyzed": len(df),
        "n_drugs_with_kg": int(has_kg_df["in_kg"].sum()),
        "per_moa": {},
    }

    for moa_class in MOA_FOCUS.keys():
        sub = has_kg_df[has_kg_df["moa_class"] == moa_class]
        if len(sub) > 0:
            stats = {
                "n_drugs_in_kg": len(sub),
                "mean_pearson_r_fp_vs_kg": round(float(sub["pearson_r"].mean()), 4),
                "mean_spearman_r_fp_vs_kg": round(float(sub["spearman_r"].mean()), 4),
                "mean_abs_delta_vh": round(float(sub["delta_mean_vh"].abs().mean()), 6),
                "drugs": [
                    {
                        "drug_name": r["drug_name"],
                        "split": r["split"],
                        "pearson_r": round(r["pearson_r"], 4) if r["pearson_r"] is not None else None,
                        "spearman_r": round(r["spearman_r"], 4) if r["spearman_r"] is not None else None,
                        "mean_vh_fp": round(r["mean_vh_fp"], 4),
                        "mean_vh_kg": round(r["mean_vh_kg"], 4) if r["mean_vh_kg"] is not None else None,
                        "delta": round(r["delta_mean_vh"], 6) if r["delta_mean_vh"] is not None else None,
                    }
                    for _, r in sub.iterrows()
                ],
            }
            summary["per_moa"][moa_class] = stats
            print(f"\n  {moa_class} ({len(sub)} drugs with KG):")
            print(f"    Mean Pearson r (fp-only vs full-KG): {stats['mean_pearson_r_fp_vs_kg']:.4f}")
            print(f"    Mean Spearman r: {stats['mean_spearman_r_fp_vs_kg']:.4f}")
            print(f"    Mean |Δ mean_vh|: {stats['mean_abs_delta_vh']:.6f}")

    # Also add all-drugs statistics
    if len(has_kg_df) > 0:
        summary["overall_mean_pearson_r"] = round(float(has_kg_df["pearson_r"].mean()), 4)
        summary["overall_mean_spearman_r"] = round(float(has_kg_df["spearman_r"].mean()), 4)
        print(f"\n  Overall: mean Pearson r = {summary['overall_mean_pearson_r']:.4f}")

    with open(OUT_DIR / "kg_contribution_gap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved kg_contribution_gap_summary.json")
    print(f"\n  → {OUT_DIR}")


if __name__ == "__main__":
    main()
