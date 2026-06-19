#!/usr/bin/env python3
"""
n02_kg_proxy_scoring.py
=======================
KG-proxy inference for novel BRICS-generated drug analogues.

Problem:
  Novel BRICS analogues are not registered in ChEMBL/DRKG/PrimeKG.
  When scored with drug_idx=None (fingerprint-only), the model uses:
    z_prior = prior_adapter(zeros) * zero_mask = 0
    z_a = z_chem + 0 = z_chem   (only fingerprint branch)
  The three KG networks are completely bypassed.

Solution — Structural Similarity KG Proxy:
  For each novel analogue, find its most structurally similar KG-covered
  training drug (top-1 by Tanimoto similarity on Morgan fingerprints).
  Then score with:
    z_chem = drug_encoder(novel_fp)          ← novel drug's chemistry
    z_prior = z_kg[nearest_training_drug]    ← KG embedding borrowed from NN
  This is the "proxy KG" inference:
    model(..., drug_latent=novel_z_chem, drug_idx=nn_local_idx)

Biological rationale:
  Structurally similar drugs tend to share molecular targets and pathway
  effects. Borrowing the KG embedding from the nearest known drug transfers
  its target-interaction network as a prior for the novel analogue.

Validation:
  Compare fp-only vs KG-proxy rankings for all analogues.
  Specifically check: are the "improved" analogues (delta_vs_seed > 0)
  from the original analysis still improved under KG-proxy inference?

Outputs:
  results/novel_drug_kg_proxy_luad.csv
  results/novel_drug_kg_proxy_skcm.csv
  results/novel_drug_proxy_summary.json
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

GENERATED_LUAD = CNM_DIR / "results/generated_compounds_luad.csv"
GENERATED_SKCM = CNM_DIR / "results/generated_compounds_skcm.csv"

DEVICE = "cuda:6"
TOP_K_NN = 5   # top-k nearest neighbors for weighted proxy (also use top-1)
MIN_TANIMOTO_FOR_PROXY = 0.05   # if NN Tanimoto < this, report but don't exclude


# ── Fingerprint helpers ───────────────────────────────────────────────────────

def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)


def batch_tanimoto(fp_query: np.ndarray, fp_bank: np.ndarray) -> np.ndarray:
    """
    Compute Tanimoto similarity between a single query FP and all bank FPs.

    fp_query: [n_bits]
    fp_bank:  [n_drugs, n_bits]
    Returns:  [n_drugs] Tanimoto similarities
    """
    ab = fp_query @ fp_bank.T          # [n_drugs]
    a_sum = fp_query.sum()             # scalar
    b_sum = fp_bank.sum(axis=1)        # [n_drugs]
    denom = a_sum + b_sum - ab
    denom = np.where(denom < 1e-6, 1e-6, denom)
    return ab / denom


# ── Patient state loader (same as n01 / script 04) ───────────────────────────

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

    gdsc_gene_names, gdsc_data = load_gdsc_efficiently()
    gsym2row = {g: i for i, g in enumerate(gdsc_gene_names)}
    n_gdsc_cells = gdsc_data.shape[1]
    gdsc_mat = np.zeros((n_gdsc_cells, len(clean_syms)), dtype=np.float32)
    for pos, g in enumerate(clean_syms):
        if g in gsym2row:
            gdsc_mat[:, pos] = gdsc_data[gsym2row[g], :]

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


# ── KG proxy scoring ──────────────────────────────────────────────────────────

def score_novel_batch(
    model: torch.nn.Module,
    states: np.ndarray,
    analogue_fps: list[np.ndarray],
    nn_local_idxs: list[int | None],
    device: str,
    precomputed_kg_payload: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score a batch of novel analogues using two modes:
      Mode A: fp-only (drug_idx=None)
      Mode B: KG-proxy (drug_latent=novel, drug_idx=NN's local_idx)

    Returns:
      vh_fp_means: [n_analogues] mean value_hat under fp-only
      vh_proxy_means: [n_analogues] mean value_hat under KG-proxy
    """
    n_ent = states.shape[0]
    n_prior = model.prior_adapter.prior_dim if hasattr(model, "prior_adapter") else 11
    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    prior_t = torch.zeros(n_ent, n_prior, dtype=torch.float32, device=device)
    mask_t  = torch.zeros(n_ent, 1, dtype=torch.float32, device=device)

    vh_fp_means    = np.zeros(len(analogue_fps), dtype=np.float32)
    vh_proxy_means = np.zeros(len(analogue_fps), dtype=np.float32)

    with torch.no_grad():
        for i, (fp, nn_idx) in enumerate(zip(analogue_fps, nn_local_idxs)):
            fp_t = torch.tensor(fp[np.newaxis, :], dtype=torch.float32, device=device).expand(n_ent, -1)

            # Mode A: fingerprint-only
            out_a = model(state_t, fp_t, prior_t, mask_t, drug_idx=None)
            vh_fp_means[i] = float(out_a["value_hat"].mean().cpu())

            # Mode B: KG proxy (only if a valid NN index exists)
            if nn_idx is not None:
                # Compute novel drug's latent from its own fingerprint
                novel_z_chem = model.drug_encoder(fp_t[:1, :])  # [1, 128]
                novel_z_chem_expanded = novel_z_chem.expand(n_ent, -1)  # [n_ent, 128]
                idx_t = torch.full((n_ent,), nn_idx, dtype=torch.long, device=device)
                out_b = model(
                    state_t, fp_t, prior_t, mask_t,
                    drug_latent=novel_z_chem_expanded,  # novel drug's chemistry
                    drug_idx=idx_t,                     # NN's KG embedding
                    precomputed_kg_payload=precomputed_kg_payload,
                )
                vh_proxy_means[i] = float(out_b["value_hat"].mean().cpu())
            else:
                # No valid NN → fall back to fp-only
                vh_proxy_means[i] = vh_fp_means[i]

    return vh_fp_means, vh_proxy_means


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("KG-Proxy Scoring for Novel BRICS-Generated Drug Analogues")
    print("=" * 70)

    # Load artifacts and model
    print("\n[1] Loading model and artifacts...")
    with open(ARTIFACTS_PKL, "rb") as f:
        art = pickle.load(f)
    model = load_model(RUN_DIR, art)
    model = model.to(DEVICE).eval()

    kg = art.kg_graph
    drug_df = art.drug_table

    # Get training drug fingerprints from the KG encoder's fingerprint bank
    # This is [n_kg_drugs, 2048] — same fingerprints used during training
    train_fp_bank = model.kg_action_encoder.drug_fingerprint_bank.cpu().numpy()
    kg_drug_ids = list(kg.drug_ids)  # local_idx → global DRUG_ID
    print(f"  KG fingerprint bank: {train_fp_bank.shape} ({len(kg_drug_ids)} drugs)")
    print(f"  KG coverage — ChEMBL: {kg.coverage['has_ChEMBL'].sum()}, PrimeKG: {kg.coverage['has_PrimeKG'].sum()}")

    # Build lookup: local_idx → drug_name and KG coverage mask
    cov_df = kg.coverage.set_index("DRUG_ID") if not kg.coverage.empty else pd.DataFrame()
    local_to_info = {}
    for drug_id, local_idx in kg.drug_to_local.items():
        row = drug_df[drug_df["DRUG_ID"] == drug_id]
        dname = str(row.iloc[0]["DRUG_NAME"]) if len(row) > 0 else f"DRUG_{drug_id}"
        if drug_id in cov_df.index:
            c = cov_df.loc[drug_id]
            has_kg_any = float(c.get("has_ChEMBL", 0)) + float(c.get("has_PrimeKG", 0)) > 0
        else:
            has_kg_any = False
        local_to_info[local_idx] = {
            "drug_id": drug_id,
            "drug_name": dname,
            "has_kg": has_kg_any,
        }

    # Build mask: which local indices have ANY KG coverage (ChEMBL OR PrimeKG)
    kg_covered_local_idxs = np.array([
        local_idx for local_idx, info in local_to_info.items()
        if info["has_kg"]
    ], dtype=np.int64)
    kg_covered_fps = train_fp_bank[kg_covered_local_idxs]
    print(f"  Local indices with KG coverage: {len(kg_covered_local_idxs)}")

    # Precompute KG payload (GNN branch embeddings for all training drugs)
    print("\n[2] Precomputing KG branch embeddings (one-time)...")
    t0 = time.time()
    precomputed_kg = model.precompute_kg_payload(device=DEVICE)
    print(f"  Done in {time.time()-t0:.1f}s — branch_all: {precomputed_kg['branch_all'].shape}")

    # Load TCGA patient states
    print("\n[3] Loading TCGA patient states...")
    preds = pd.read_parquet(PREDS_PARQUET)
    cancer_entity_ids = {
        "TCGA-LUAD": preds[preds["project_id"] == "TCGA-LUAD"]["entity_id"].unique().tolist(),
        "TCGA-SKCM": preds[preds["project_id"] == "TCGA-SKCM"]["entity_id"].unique().tolist(),
    }

    cancer_states: dict[str, np.ndarray] = {}
    for ct, eids in cancer_entity_ids.items():
        print(f"  Loading {ct} ({len(eids)} patients)...")
        cancer_states[ct] = load_cancer_states(art, ct, eids)
        print(f"  States shape: {cancer_states[ct].shape}")

    # ── Process each seed drug ────────────────────────────────────────────────
    all_results = []
    summary_data = {"seeds": []}

    for seed_name, cancer_type, generated_csv in [
        ("erlotinib",  "TCGA-LUAD", GENERATED_LUAD),
        ("trametinib", "TCGA-SKCM", GENERATED_SKCM),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {seed_name} → {cancer_type}")

        # Load generated compounds
        gen_df = pd.read_csv(generated_csv)
        gen_df = gen_df[gen_df["seed_drug"] == seed_name].copy()
        print(f"  Loaded {len(gen_df)} analogues for {seed_name}")

        states = cancer_states[cancer_type]
        n_pat = states.shape[0]

        # Parse analogue fingerprints
        analogue_fps = []
        valid_mask = []
        for _, row in gen_df.iterrows():
            fp = morgan_fp(str(row["smiles"]))
            if fp is not None:
                analogue_fps.append(fp)
                valid_mask.append(True)
            else:
                analogue_fps.append(None)
                valid_mask.append(False)

        valid_fps = [fp for fp in analogue_fps if fp is not None]
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        print(f"  Valid analogue FPs: {len(valid_fps)} / {len(gen_df)}")

        if len(valid_fps) == 0:
            print("  No valid FPs, skipping")
            continue

        # Find nearest KG-covered training drug for each analogue
        print(f"  Computing Tanimoto similarities to {len(kg_covered_local_idxs)} KG-covered training drugs...")
        fp_matrix = np.stack(valid_fps, axis=0)  # [n_valid, 2048]

        # Vectorized Tanimoto: [n_valid, n_kg_covered]
        ab = fp_matrix @ kg_covered_fps.T
        a_sum = fp_matrix.sum(axis=1, keepdims=True)  # [n_valid, 1]
        b_sum = kg_covered_fps.sum(axis=1)             # [n_kg_covered]
        denom = a_sum + b_sum - ab
        denom = np.where(denom < 1e-6, 1e-6, denom)
        tanimoto_mat = ab / denom   # [n_valid, n_kg_covered]

        # Top-1 NN per analogue
        top1_pos = np.argmax(tanimoto_mat, axis=1)        # [n_valid]
        top1_tanimoto = tanimoto_mat[np.arange(len(valid_fps)), top1_pos]  # [n_valid]
        top1_local_idx = kg_covered_local_idxs[top1_pos]  # [n_valid] local indices

        print(f"  Top-1 Tanimoto: mean={top1_tanimoto.mean():.4f}, "
              f"median={np.median(top1_tanimoto):.4f}, min={top1_tanimoto.min():.4f}")

        # Get top-1 NN drug names
        nn_drug_names = [local_to_info[li]["drug_name"] for li in top1_local_idx]

        # Score all valid analogues: fp-only vs KG-proxy
        print(f"  Scoring {len(valid_fps)} analogues × {n_pat} patients (2 modes)...")
        t0 = time.time()
        vh_fp_means, vh_proxy_means = score_novel_batch(
            model, states,
            analogue_fps=valid_fps,
            nn_local_idxs=[int(li) for li in top1_local_idx],
            device=DEVICE,
            precomputed_kg_payload=precomputed_kg,
        )
        print(f"  Scoring done in {time.time()-t0:.1f}s")

        # Compute seed drug baselines (fp-only AND KG-proxy)
        seed_rows = drug_df[drug_df["DRUG_NAME"].str.lower() == seed_name.lower()]
        seed_fp = morgan_fp(str(seed_rows.iloc[0]["canonical_smiles"])) if len(seed_rows) > 0 else None
        if seed_fp is not None:
            n_prior = model.prior_adapter.prior_dim if hasattr(model, "prior_adapter") else 11
            state_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            fp_t = torch.tensor(seed_fp[np.newaxis], dtype=torch.float32, device=DEVICE).expand(n_pat, -1)
            prior_t = torch.zeros(n_pat, n_prior, dtype=torch.float32, device=DEVICE)
            mask_t  = torch.zeros(n_pat, 1, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                out_seed = model(state_t, fp_t, prior_t, mask_t, drug_idx=None)
            seed_vh_fp = float(out_seed["value_hat"].mean().cpu())

            # Compute seed's KG-proxy score using the SAME NN-based approach as analogues.
            # This ensures a fair apples-to-apples comparison under KG-proxy mode.
            # Key: seed's nearest KG-covered drug (by Tanimoto) is used as its proxy.
            # For erlotinib test (1130, no KG): nearest KG drug is erlotinib (1129), Tanimoto=1.0
            # For trametinib test (366, no KG): nearest KG drug is trametinib (365), Tanimoto=1.0
            seed_fp_arr = seed_fp[np.newaxis, :]  # [1, 2048]
            ab_seed = seed_fp_arr @ kg_covered_fps.T  # [1, n_kg_covered]
            a_sum_seed = seed_fp_arr.sum()
            b_sum_all = kg_covered_fps.sum(axis=1)
            denom_seed = a_sum_seed + b_sum_all - ab_seed[0]
            denom_seed = np.where(denom_seed < 1e-6, 1e-6, denom_seed)
            tani_seed = ab_seed[0] / denom_seed
            seed_nn_pos = int(np.argmax(tani_seed))
            seed_nn_local_idx = int(kg_covered_local_idxs[seed_nn_pos])
            seed_nn_tanimoto = float(tani_seed[seed_nn_pos])
            seed_nn_name = local_to_info[seed_nn_local_idx]["drug_name"]
            print(f"  Seed {seed_name} NN: {seed_nn_name} (Tanimoto={seed_nn_tanimoto:.4f})")

            seed_z_chem = model.drug_encoder(fp_t[:1, :])
            seed_z_chem_exp = seed_z_chem.expand(n_pat, -1)
            seed_idx_t = torch.full((n_pat,), seed_nn_local_idx, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                out_seed_proxy = model(
                    state_t, fp_t, prior_t, mask_t,
                    drug_latent=seed_z_chem_exp,
                    drug_idx=seed_idx_t,
                    precomputed_kg_payload=precomputed_kg,
                )
            seed_vh_proxy = float(out_seed_proxy["value_hat"].mean().cpu())
            print(f"  Seed {seed_name}: vh_fp={seed_vh_fp:.4f}, vh_proxy={seed_vh_proxy:.4f}, "
                  f"Δ(proxy-fp)={seed_vh_proxy-seed_vh_fp:+.4f}")
        else:
            seed_vh_fp = float(gen_df["seed_baseline_value_hat"].iloc[0])
            seed_vh_proxy = seed_vh_fp
            seed_nn_name = "unknown"
            seed_nn_tanimoto = 0.0

        # Build output dataframe
        result_rows = []
        for i, vidx in enumerate(valid_indices):
            row = gen_df.iloc[vidx].to_dict()
            result_rows.append({
                # Original columns from generated_compounds csv
                "DRUG_ID":            row.get("DRUG_ID"),
                "DRUG_NAME":          row.get("DRUG_NAME"),
                "smiles":             row.get("smiles"),
                "cancer_type":        row.get("cancer_type"),
                "seed_drug":          row.get("seed_drug"),
                "mw":                 row.get("mw"),
                "logp":               row.get("logp"),
                "qed":                row.get("qed"),
                "tanimoto_to_seed":   row.get("tanimoto"),
                # Original fp-only scoring (from cnm/)
                "mean_vh_fp_original": row.get("mean_value_hat"),
                # New fp-only scoring (recomputed, should match)
                "mean_vh_fp":         round(float(vh_fp_means[i]), 6),
                # KG-proxy scoring
                "mean_vh_proxy":      round(float(vh_proxy_means[i]), 6),
                # Nearest training drug info (for analogue)
                "nn_tanimoto":        round(float(top1_tanimoto[i]), 6),
                "nn_drug_name":       nn_drug_names[i],
                "nn_local_idx":       int(top1_local_idx[i]),
                "nn_has_kg":          local_to_info[int(top1_local_idx[i])]["has_kg"],
                # Seed baselines
                "seed_vh_fp":         round(float(seed_vh_fp), 6),
                "seed_vh_proxy":      round(float(seed_vh_proxy), 6),
                "seed_nn_drug_name":  seed_nn_name,
                "seed_nn_tanimoto":   round(float(seed_nn_tanimoto), 6),
                # FP-only improvement: analogue fp-only vs seed fp-only (internally consistent)
                "delta_fp":           round(float(seed_vh_fp - vh_fp_means[i]), 6),
                "improved_fp":        bool(seed_vh_fp > vh_fp_means[i]),
                # FAIR KG-proxy improvement: analogue proxy vs seed proxy (internally consistent)
                # Both analogue and seed scored with their respective NN's KG embedding
                "delta_proxy_fair":   round(float(seed_vh_proxy - vh_proxy_means[i]), 6),
                "improved_proxy_fair": bool(seed_vh_proxy > vh_proxy_means[i]),
            })

        result_df = pd.DataFrame(result_rows).sort_values("mean_vh_proxy", ascending=True)

        # Compute rank correlation between fp-only and KG-proxy
        from scipy.stats import spearmanr, pearsonr
        ranks_fp    = result_df["mean_vh_fp"].rank()
        ranks_proxy = result_df["mean_vh_proxy"].rank()
        rho, p_rho = spearmanr(ranks_fp, ranks_proxy)
        r_pearson, p_pearson = pearsonr(result_df["mean_vh_fp"].values,
                                         result_df["mean_vh_proxy"].values)

        n_improved_fp         = int(result_df["improved_fp"].sum())
        n_improved_proxy_fair = int(result_df["improved_proxy_fair"].sum())
        n_improved_both       = int((result_df["improved_fp"] & result_df["improved_proxy_fair"]).sum())

        print(f"\n  Ranking comparison: Spearman ρ={rho:.4f} (p={p_rho:.2e})")
        print(f"  Pearson r={r_pearson:.4f} (p={p_pearson:.2e})")
        print(f"  Improved fp-only (vs fp seed {seed_vh_fp:.4f}): {n_improved_fp} / {len(result_df)}")
        print(f"  Improved proxy-fair (vs proxy seed {seed_vh_proxy:.4f}): {n_improved_proxy_fair} / {len(result_df)}")
        print(f"  Improved in BOTH (fp-only AND proxy-fair): {n_improved_both}")
        print(f"  Concordance fp↔proxy_fair: "
              f"{(result_df['improved_fp'] == result_df['improved_proxy_fair']).mean()*100:.1f}%")

        print(f"\n  Top 10 by KG-proxy score (lower = better):")
        top10 = result_df.head(10)
        for _, r in top10.iterrows():
            flag = "✓BOTH" if r["improved_fp"] and r["improved_proxy_fair"] else (
                   "fp✓" if r["improved_fp"] else (
                   "kgp_fair✓" if r["improved_proxy_fair"] else ""))
            print(f"    {str(r['DRUG_NAME'])[:35]:<35} "
                  f"fp={r['mean_vh_fp']:.4f} proxy={r['mean_vh_proxy']:.4f} "
                  f"NN={str(r['nn_drug_name'])[:20]:<20} tani={r['nn_tanimoto']:.3f}  {flag}")

        # Save
        cancer_tag = cancer_type.split("-")[1].lower()
        out_file = OUT_DIR / f"novel_drug_kg_proxy_{cancer_tag}.csv"
        result_df.to_csv(out_file, index=False)
        print(f"\n  Saved → {out_file.name}")

        all_results.append(result_df)

        # Collect summary
        seed_summary = {
            "seed_drug": seed_name,
            "cancer_type": cancer_type,
            "n_analogues": len(result_df),
            "seed_vh_fp": round(float(seed_vh_fp), 6),
            "seed_vh_proxy": round(float(seed_vh_proxy), 6),
            "seed_nn_drug": seed_nn_name,
            "seed_nn_tanimoto": round(float(seed_nn_tanimoto), 4),
            "analogue_nn_tanimoto_mean": round(float(top1_tanimoto.mean()), 4),
            "analogue_nn_tanimoto_median": round(float(np.median(top1_tanimoto)), 4),
            "analogue_nn_tanimoto_min": round(float(top1_tanimoto.min()), 4),
            "analogue_nn_tanimoto_max": round(float(top1_tanimoto.max()), 4),
            "top_analogue_nn_drug": nn_drug_names[int(np.argmax(top1_tanimoto))],
            "ranking_spearman_rho": round(float(rho), 4),
            "ranking_spearman_p": float(p_rho),
            "ranking_pearson_r": round(float(r_pearson), 4),
            "ranking_pearson_p": float(p_pearson),
            "n_improved_fp_only": n_improved_fp,
            "n_improved_proxy_fair": n_improved_proxy_fair,
            "n_improved_both": n_improved_both,
            "pct_concordant_fp_proxy": round(float((result_df["improved_fp"] == result_df["improved_proxy_fair"]).mean() * 100), 1),
            "interpretation": (
                "Both fp-only and KG-proxy use their respective seed baselines. "
                "n_improved_both = analogues superior to BOTH baselines."
            ),
            "top5_proxy": [
                {
                    "drug_name": r["DRUG_NAME"],
                    "mean_vh_proxy": round(r["mean_vh_proxy"], 6),
                    "mean_vh_fp": round(r["mean_vh_fp"], 6),
                    "delta_proxy_fair": round(r["delta_proxy_fair"], 6),
                    "nn_drug_name": r["nn_drug_name"],
                    "nn_tanimoto": round(r["nn_tanimoto"], 4),
                    "improved_both": bool(r["improved_fp"] and r["improved_proxy_fair"]),
                }
                for _, r in result_df.head(5).iterrows()
            ],
        }
        summary_data["seeds"].append(seed_summary)

    # Save summary
    with open(OUT_DIR / "novel_drug_proxy_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n  Saved novel_drug_proxy_summary.json")
    print(f"\n  → All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
