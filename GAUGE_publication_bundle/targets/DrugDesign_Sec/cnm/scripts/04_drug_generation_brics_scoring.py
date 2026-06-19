#!/usr/bin/env python3
"""
BRICS-based drug analogue generation + GAUGE scoring.

Pipeline:
  1. Select seed drugs: erlotinib (LUAD, test split), trametinib (SKCM, test split)
  2. BRICS fragmentation of seed drugs + fragment pool from all 1487 model drugs
  3. Generate analogues via fragment recombination; filter by Lipinski / drug-likeness
  4. Score each generated molecule against TCGA patients using the drug-split model
     - Novel drugs scored via fingerprint branch (drug_idx=None, bypasses KG bank)
     - This enables zero-shot scoring for unseen chemical space
  5. Rank candidates by mean value_hat across target cancer patients
  6. Compare top candidates against seed drug baseline

Outputs (cnm/results/):
  generated_compounds.csv          — all generated molecules with properties
  generated_compounds_luad.csv     — LUAD scoring results
  generated_compounds_skcm.csv     — SKCM scoring results
  drug_generation_summary.json
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
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, BRICS, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")  # suppress RDKit C++ warnings (e.g. valence violations)

from GAUGE.train import load_model

OUT_DIR      = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR      = ROOT / "PRISM/Secondary/cheml35/results/20260524_224312"
ARTIFACTS_PKL = RUN_DIR / "artifacts.pkl"
PREDS_PARQUET = OUT_DIR / "tcga_drugsplit_predictions.parquet"

DEVICE    = "cuda:6"
N_JOBS    = 1
BATCH_SIZE = 4096
MAX_FRAGS = 300   # fragment pool size — 300 is sufficient for diverse analogues

# Target cancer types for each seed drug
SEED_TARGETS = {
    "erlotinib":  "TCGA-LUAD",
    "trametinib": "TCGA-SKCM",
}

# Lipinski filter parameters
MW_MAX    = 600.0
LOGP_MAX  = 5.5
HBD_MAX   = 5
HBA_MAX   = 10
TPSA_MAX  = 140.0


# ── chemistry helpers ─────────────────────────────────────────────────────────

def morgan_fp(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits),
        dtype=np.float32,
    )


def drug_likeness_ok(mol) -> bool:
    try:
        Chem.SanitizeMol(mol)  # ensures valence is computed before descriptor calls
    except Exception:
        return False
    try:
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = Descriptors.NumHDonors(mol)
        hba  = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
    except Exception:
        return False
    return (
        mw   <= MW_MAX
        and logp <= LOGP_MAX
        and hbd <= HBD_MAX
        and hba <= HBA_MAX
        and tpsa <= TPSA_MAX
        and mol.GetNumAtoms() >= 10
    )


def enumerate_brics_analogues(
    seed_smiles: str,
    fragment_pool: list[str],
    max_generated: int = 5000,
    seed_rng: int = 42,
) -> list[tuple[str, str]]:
    """
    Generate BRICS analogues of a seed molecule.
    Returns list of (smiles, parent_fragment) tuples.
    """
    rng = np.random.default_rng(seed_rng)
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        return []

    # Fragment the seed molecule
    try:
        frags = list(BRICS.BRICSDecompose(seed_mol, returnMols=True))
    except Exception:
        frags = []
    if not frags:
        return []

    # Convert fragment pool smiles → mols
    pool_mols = []
    for smi in fragment_pool:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            pool_mols.append((smi, mol))

    # Use BRICS build: combine seed fragments with pool fragments.
    # IMPORTANT: only use fragments with exactly 1 attachment point — multi-attachment
    # fragments cause BRICSBuild to enumerate exponentially many products (very slow).
    all_frags = [Chem.MolToSmiles(f) for f in frags]
    all_frags_unique = list(dict.fromkeys(all_frags))  # deduplicate
    # Filter to single-attachment-point fragments only
    def _n_attach(smi: str) -> int:
        return smi.count("*")
    all_frags_unique = [s for s in all_frags_unique if _n_attach(s) == 1]
    if not all_frags_unique:
        return []

    generated: list[tuple[str, str]] = []
    seen: set[str] = {Chem.MolToSmiles(seed_mol) or ""}

    # Try building from each seed fragment + pool fragments
    for seed_frag_smi in all_frags_unique:
        seed_frag_mol = Chem.MolFromSmiles(seed_frag_smi)
        if seed_frag_mol is None:
            continue
        for pool_smi, pool_mol in pool_mols:
            if len(generated) >= max_generated:
                break
            try:
                new_mols = list(BRICS.BRICSBuild([seed_frag_mol, pool_mol], scrambleReagents=False))
                for nm in new_mols[:5]:
                    try:
                        Chem.SanitizeMol(nm)
                        smi = Chem.MolToSmiles(nm)
                    except Exception:
                        continue
                    if smi and smi not in seen and drug_likeness_ok(nm):
                        seen.add(smi)
                        generated.append((smi, seed_frag_smi))
            except Exception:
                pass
        if len(generated) >= max_generated:
            break

    return generated[:max_generated]


def build_fragment_pool(
    drug_table: pd.DataFrame,
    max_frags: int = 2000,
) -> list[str]:
    """
    Extract BRICS fragments from all 1487 model drugs.
    Only keeps single-attachment-point fragments — multi-attachment fragments
    cause BRICSBuild to enumerate exponentially many products (very slow).
    """
    frags: set[str] = set()
    for row in drug_table.itertuples(index=False):
        smi = getattr(row, "canonical_smiles", None) or getattr(row, "smiles", None)
        if not smi:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            frag_mols = BRICS.BRICSDecompose(mol, returnMols=True)
            for f in frag_mols:
                fsmi = Chem.MolToSmiles(f)
                if fsmi and fsmi.count("*") == 1:   # single-attachment only
                    frags.add(fsmi)
        except Exception:
            pass
        if len(frags) >= max_frags:
            break
    return list(frags)


# ── model inference for novel compounds ──────────────────────────────────────

def _score_novel_drugs_batch(
    model: torch.nn.Module,
    states_batch: np.ndarray,      # [n_entities × state_dim]
    fps_batch: np.ndarray,         # [n_drugs × fp_dim]
    drug_names: list[str],
    drug_ids:   list[int],
    device: str,
) -> pd.DataFrame:
    """
    Score novel drugs using fingerprint branch (drug_idx=None → bypasses KG bank).
    Returns DataFrame with value_hat per (entity, drug) pair.
    """
    n_ent  = states_batch.shape[0]
    n_drug = fps_batch.shape[0]
    rows   = []

    state_t = torch.tensor(states_batch, dtype=torch.float32, device=device)
    n_prior = model.prior_adapter.prior_dim if hasattr(model, "prior_adapter") else 11

    with torch.no_grad():
        for di in range(n_drug):
            fp_t    = torch.tensor(fps_batch[[di]], dtype=torch.float32, device=device).expand(n_ent, -1)
            prior_t = torch.zeros(n_ent, n_prior, dtype=torch.float32, device=device)
            mask_t  = torch.zeros(n_ent, 1, dtype=torch.float32, device=device)
            out = model(
                state_t, fp_t, prior_t, mask_t,
                drug_idx=None,  # use fingerprint branch directly
            )
            vh = out["value_hat"].cpu().numpy().astype(np.float32)
            rows.append({
                "DRUG_ID":    drug_ids[di],
                "DRUG_NAME":  drug_names[di],
                "value_hat":  float(vh.mean()),
                "value_hat_std": float(vh.std()),
                "n_patients": n_ent,
            })
    return pd.DataFrame(rows)


def score_generated_compounds(
    model: torch.nn.Module,
    states: np.ndarray,
    entity_ids: list[str],
    generated: list[tuple[str, str]],  # (smiles, parent_frag)
    cancer_type: str,
    seed_drug_name: str,
    device: str,
    drug_batch_size: int = 64,
) -> pd.DataFrame:
    """
    Score all generated compounds by iterating drug-by-drug.
    Per drug: run all patients in a single forward pass (efficient on GPU).
    """
    valid: list[tuple[str, str, np.ndarray]] = []
    for smiles, parent_frag in generated:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = morgan_fp(mol)
        valid.append((smiles, parent_frag, fp))

    if not valid:
        print(f"  No valid generated molecules for {seed_drug_name}")
        return pd.DataFrame()

    n_ent  = states.shape[0]
    n_drug = len(valid)
    n_prior = model.prior_adapter.prior_dim if hasattr(model, "prior_adapter") else 11

    print(f"  Scoring {n_drug} analogues × {n_ent} {cancer_type} patients...")
    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    prior_t = torch.zeros(n_ent, n_prior, dtype=torch.float32, device=device)
    mask_t  = torch.zeros(n_ent, 1, dtype=torch.float32, device=device)

    t0 = time.time()
    vh_means = np.empty(n_drug, dtype=np.float32)
    vh_stds  = np.empty(n_drug, dtype=np.float32)

    with torch.no_grad():
        for di in range(n_drug):
            fp_t = torch.tensor(valid[di][2][np.newaxis], dtype=torch.float32, device=device
                                ).expand(n_ent, -1)
            out = model(state_t, fp_t, prior_t, mask_t, drug_idx=None)
            vh = out["value_hat"].cpu().numpy().astype(np.float32)
            vh_means[di] = vh.mean()
            vh_stds[di]  = vh.std()

    print(f"    Done in {time.time() - t0:.1f}s")

    out_rows = []
    for di, (smiles, parent_frag, _) in enumerate(valid):
        mol = Chem.MolFromSmiles(smiles)
        out_rows.append({
            "DRUG_ID":         90000 + di,
            "DRUG_NAME":       f"{seed_drug_name}_analogue_{di:04d}",
            "smiles":          smiles,
            "parent_fragment": parent_frag,
            "cancer_type":     cancer_type,
            "seed_drug":       seed_drug_name,
            "mean_value_hat":  round(float(vh_means[di]), 6),
            "std_value_hat":   round(float(vh_stds[di]), 6),
            "n_patients":      n_ent,
            "mw":              round(Descriptors.MolWt(mol), 2) if mol else None,
            "logp":            round(Descriptors.MolLogP(mol), 3) if mol else None,
            "qed":             round(QED.qed(mol), 4) if mol else None,
            "tpsa":            round(Descriptors.TPSA(mol), 2) if mol else None,
        })

    # Sort ascending: LOWER value_hat = more cancer cell killing = better drug candidate
    return pd.DataFrame(out_rows).sort_values("mean_value_hat", ascending=True)


def get_seed_drug_baseline(
    model: torch.nn.Module,
    states: np.ndarray,
    drug_name: str,
    drug_table: pd.DataFrame,
    device: str,
) -> float:
    """Get mean value_hat of a seed drug over a set of patient states."""
    rows = drug_table[drug_table["DRUG_NAME"].str.lower() == drug_name.lower()]
    if len(rows) == 0:
        return 0.5
    smiles = rows.iloc[0]["canonical_smiles"] or rows.iloc[0]["smiles"]
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0.5
    fp = morgan_fp(mol)
    fp_t = torch.tensor(fp[np.newaxis, :], dtype=torch.float32, device=device).expand(len(states), -1)
    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    n_prior = model.prior_adapter.prior_dim if hasattr(model, "prior_adapter") else 11
    prior_t = torch.zeros(len(states), n_prior, dtype=torch.float32, device=device)
    mask_t  = torch.zeros(len(states), 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(state_t, fp_t, prior_t, mask_t, drug_idx=None)
    return float(out["value_hat"].mean().cpu())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max_per_seed", type=int, default=1000,
                        help="Max generated analogues per seed drug")
    parser.add_argument("--seed_name", default=None,
                        help="Run only this seed drug (e.g. erlotinib). Default: all.")
    args = parser.parse_args()

    # Optionally restrict to a single seed (for parallel execution on multiple GPUs)
    seed_targets = SEED_TARGETS
    if args.seed_name is not None:
        if args.seed_name not in SEED_TARGETS:
            print(f"ERROR: seed_name '{args.seed_name}' not in {list(SEED_TARGETS.keys())}")
            sys.exit(1)
        seed_targets = {args.seed_name: SEED_TARGETS[args.seed_name]}

    print("=" * 60)
    print("BRICS Drug Generation + GAUGE Scoring")
    print(f"  Seed drugs: {list(seed_targets.keys())}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[1] Loading model and artifacts...")
    with open(ARTIFACTS_PKL, "rb") as f:
        art = pickle.load(f)
    model = load_model(RUN_DIR, art)
    model = model.to(args.device).eval()
    print(f"  Drugs in model: {len(art.drug_table)}")

    # ── Load TCGA states ─────────────────────────────────────────────────────
    print("\n[2] Loading pre-computed TCGA patient states from predictions...")
    # We reconstruct per-cancer states by sampling unique entity_ids from predictions
    # (states are stored in the parquet via entity_id → we need to recompute from the
    # parquet or load from the h5ad; using parquet unique entity_ids as proxy)
    preds = pd.read_parquet(PREDS_PARQUET)
    print(f"  Total predictions loaded: {len(preds):,}")

    # ── Load GDSC expression once (fast path) ────────────────────────────────
    print("\n[3] Pre-loading GDSC expression for quantile mapping...")
    gdsc_gene_names, gdsc_data = load_gdsc_efficiently()

    # ── Build fragment pool ──────────────────────────────────────────────────
    print("\n[4] Building BRICS fragment pool from all model drugs...")
    frag_pool = build_fragment_pool(art.drug_table, max_frags=MAX_FRAGS)
    print(f"  Fragment pool size: {len(frag_pool)}")

    # ── Per-seed drug generation + scoring ──────────────────────────────────
    all_results = []
    seed_baselines: dict[str, float] = {}
    n_seeds = len(seed_targets)

    for i, (seed_name, target_cancer) in enumerate(seed_targets.items(), 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{n_seeds}] Processing seed: {seed_name} → {target_cancer}")

        # Get seed SMILES from drug table
        seed_rows = art.drug_table[art.drug_table["DRUG_NAME"].str.lower() == seed_name.lower()]
        if len(seed_rows) == 0:
            print(f"  Seed drug {seed_name} not found in drug table, skipping")
            continue
        seed_smiles = seed_rows.iloc[0]["canonical_smiles"] or seed_rows.iloc[0]["smiles"]
        print(f"  Seed SMILES: {seed_smiles}")

        # Get TCGA patient states for this cancer type
        cancer_entity_ids = preds[preds["project_id"] == target_cancer]["entity_id"].unique().tolist()
        if len(cancer_entity_ids) == 0:
            print(f"  No patients for {target_cancer}, skipping")
            continue
        print(f"  Target patients: {len(cancer_entity_ids)} {target_cancer}")

        print(f"  Loading patient states for {target_cancer}...")
        states_cancer = _load_cancer_states(
            art, target_cancer, cancer_entity_ids, args.device,
            gdsc_gene_names, gdsc_data,
        )
        if states_cancer is None or len(states_cancer) == 0:
            print(f"  Could not load states for {target_cancer}")
            continue
        print(f"  States shape: {states_cancer.shape}")

        # Seed drug baseline (using fingerprint branch for fair comparison)
        baseline = get_seed_drug_baseline(model, states_cancer, seed_name, art.drug_table, args.device)
        seed_baselines[seed_name] = baseline
        print(f"  Seed drug mean value_hat (fp branch): {baseline:.4f}")

        # Generate analogues
        print(f"  Generating BRICS analogues (max={args.max_per_seed})...")
        generated = enumerate_brics_analogues(
            seed_smiles, frag_pool,
            max_generated=args.max_per_seed,
        )
        print(f"  Generated {len(generated)} drug-like analogues")

        if len(generated) == 0:
            continue

        # Score
        result_df = score_generated_compounds(
            model, states_cancer, cancer_entity_ids,
            generated, target_cancer, seed_name, args.device,
        )
        result_df["seed_baseline_value_hat"] = baseline
        # delta_vs_seed: positive = more cancer cell killing = better (lower vh = better)
        result_df["delta_vs_seed"] = baseline - result_df["mean_value_hat"]
        all_results.append(result_df)

        # Print top candidates (lowest value_hat = best)
        print(f"\n  Top 20 {seed_name} candidates (lowest value_hat = most effective):")
        print(result_df.head(20)[["DRUG_NAME", "mean_value_hat", "delta_vs_seed",
                                   "mw", "logp", "qed"]].to_string(index=False))
        print(f"\n  Candidates improving on seed (delta > 0): "
              f"{(result_df['delta_vs_seed'] > 0).sum()} / {len(result_df)}")

        # Save per-cancer
        out_file = OUT_DIR / f"generated_compounds_{target_cancer.split('-')[1].lower()}.csv"
        result_df.to_csv(out_file, index=False)
        print(f"  Saved to {out_file.name}")

    # ── Consolidate ──────────────────────────────────────────────────────────
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUT_DIR / "generated_compounds.csv", index=False)

        # Summary stats
        summary = {
            "seed_drugs": list(seed_targets.keys()),
            "seed_baselines": {
                k: round(float(v), 4) for k, v in seed_baselines.items()
            },
            "per_seed": [],
        }
        for df in all_results:
            seed = df["seed_drug"].iloc[0]
            cancer = df["cancer_type"].iloc[0]
            baseline = float(df["seed_baseline_value_hat"].iloc[0])
            # delta_vs_seed = baseline - mean_value_hat: positive = more effective than seed
            n_improved = int((df["delta_vs_seed"] > 0).sum())
            top1 = df.iloc[0]  # sorted ascending by value_hat
            summary["per_seed"].append({
                "seed_drug":        seed,
                "cancer_type":      cancer,
                "n_generated":      len(df),
                "seed_baseline":    round(baseline, 4),
                "top1_mean_vh":     round(float(top1["mean_value_hat"]), 4),
                "top1_delta":       round(float(top1["delta_vs_seed"]), 4),
                "top1_smiles":      str(top1["smiles"]),
                "top1_qed":         round(float(top1["qed"]) if top1["qed"] else 0.0, 4),
                "n_improved":       n_improved,
                "frac_improved":    round(float(n_improved / len(df)), 4),
                "mean_delta":       round(float(df["delta_vs_seed"].mean()), 4),
            })

        with open(OUT_DIR / "drug_generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n[Summary]")
        print(json.dumps(summary, indent=2))
        print(f"\nAll outputs → {OUT_DIR}")
    else:
        print("No results generated.")


# ── State loading helper ──────────────────────────────────────────────────────

GDSC_EXPR = ROOT / "KG_GAUGE_PublicData/GDSC/rnaseq_merged_rsem_tpm_20260323.csv"
TCGA_H5AD = ROOT.parent / "Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad"


def load_gdsc_efficiently() -> tuple[list[str], np.ndarray]:
    """
    Load GDSC expression matrix fast.
    CSV layout: 4 header rows, then gene rows.
    Col 0: gene_symbol (index), Col 1: ensembl_id (skip), Col 2: gene_id (skip), Cols 3+: samples.
    Uses skiprows=4 + usecols to avoid object-dtype conversion bottleneck.
    """
    print("    Loading GDSC expression (fast path)...")
    _hdr = pd.read_csv(GDSC_EXPR, header=None, nrows=1)
    n_cols = _hdr.shape[1]
    use_cols = [0] + list(range(3, n_cols))
    gdsc_raw = pd.read_csv(
        GDSC_EXPR, index_col=0, header=None,
        skiprows=4, usecols=use_cols,
    )
    gdsc_gene_names = gdsc_raw.index.astype(str).tolist()
    gdsc_data = gdsc_raw.to_numpy(dtype=np.float32)  # float64→float32: fast
    print(f"    GDSC: {len(gdsc_gene_names)} genes × {gdsc_data.shape[1]} cells")
    return gdsc_gene_names, gdsc_data


def _load_cancer_states(
    art,
    cancer_type: str,
    entity_ids: list[str],
    device: str,
    gdsc_gene_names: list[str],
    gdsc_data: np.ndarray,
    target_state_dim: int = 480,
) -> np.ndarray | None:
    """
    Load TCGA patient states for a specific cancer type.
    Re-runs the projection pipeline: TCGA expression → quantile map → PCA → state.
    gdsc_gene_names / gdsc_data must be pre-loaded (avoids re-loading the 339MB CSV twice).
    """
    import anndata as ad
    from scipy import sparse

    # Gene symbols (clean)
    gene_symbols = list(art.genes)
    clean_syms = [g.split(" (")[0] if " (" in g else g for g in gene_symbols]

    gsym2row = {g: i for i, g in enumerate(gdsc_gene_names)}

    n_gdsc_cells = gdsc_data.shape[1]
    gdsc_mat = np.zeros((n_gdsc_cells, len(clean_syms)), dtype=np.float32)
    for pos, g in enumerate(clean_syms):
        if g in gsym2row:
            gdsc_mat[:, pos] = gdsc_data[gsym2row[g], :]  # shape: [n_cells]

    # Load TCGA expression for this cancer type
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

    # Quantile mapping
    print(f"    Applying quantile mapping ({len(sample_indices)} patients)...")
    out = np.empty_like(tcga_mat)
    for j in range(tcga_mat.shape[1]):
        gc = gdsc_mat[:, j]
        tc = tcga_mat[:, j]
        if gc.std() < 1e-8 or tc.std() < 1e-8:
            out[:, j] = tc
        else:
            src_sorted = np.sort(gc)
            ranks = np.argsort(np.argsort(tc, kind="mergesort"), kind="mergesort").astype(np.float32)
            quantiles = (ranks + 0.5) / max(len(tc), 1)
            grid = np.linspace(0.0, 1.0, num=len(src_sorted), endpoint=False) + 0.5 / max(len(src_sorted), 1)
            out[:, j] = np.interp(quantiles, grid, src_sorted,
                                   left=src_sorted[0], right=src_sorted[-1]).astype(np.float32)

    # Project to state
    imputed = art.imputer.transform(out)
    scaled  = art.scaler.transform(imputed)
    pca_out = art.pca.transform(scaled).astype(np.float32)
    n, d = pca_out.shape
    if d < target_state_dim:
        pca_out = np.concatenate([pca_out, np.zeros((n, target_state_dim - d), dtype=np.float32)], axis=1)
    elif d > target_state_dim:
        pca_out = pca_out[:, :target_state_dim]
    return pca_out


if __name__ == "__main__":
    main()
