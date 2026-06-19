#!/usr/bin/env python3
"""
Layer 4 (part 2): LINCS chemical-neighbor proxy signatures.

For each generated analogue, finds the top-k nearest LINCS compounds
by Morgan fingerprint Tanimoto similarity. Uses their L1000 Level5
signatures (weighted by Tanimoto) as a proxy drug perturbation signature.

Also fetches real LINCS signatures for erlotinib and trametinib as parent baselines.

Run with: conda run -n kg_GAUGE python v03_lincs_proxy.py
Output: data/lincs/
"""
from __future__ import annotations
import gzip, json, struct
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
OUT_DIR = ROOT / "data" / "lincs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINCS_DIR    = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/LINCS")
# Use GSE92742 (full LINCS Phase I+II, 473K signatures)
GCTX_GZ   = LINCS_DIR / "GSE92742" / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
PERT_INFO = LINCS_DIR / "GSE92742" / "GSE92742_Broad_LINCS_pert_info.txt.gz"
SIG_INFO  = LINCS_DIR / "GSE92742" / "GSE92742_Broad_LINCS_sig_info.txt.gz"
GENE_INFO = LINCS_DIR / "GSE92742" / "GSE92742_Broad_LINCS_gene_info.txt.gz"

TOP_K    = 5       # top-k neighbors per analogue
MIN_SIM  = 0.1    # minimum Tanimoto for inclusion
CANCER_CELL_LINES = {
    "TCGA-LUAD": ["A549", "HCC827", "PC9", "H1299", "H1650"],   # LUAD lines in LINCS
    "TCGA-SKCM": ["SKMEL5", "A375", "SKMEL28", "COLO829"],      # SKCM lines in LINCS
}
PARENT_SMILES = {
    "erlotinib":  "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "trametinib": "CC1=C(C(=O)N2CCN(CC2)C(=O)c2cc(I)c(F)c(NC(=O)c3ccc(F)cc3Cl)c2)C=NN1CC",
}


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def batch_tanimoto(query_fp, library_fps: list) -> np.ndarray:
    return np.array(DataStructs.BulkTanimotoSimilarity(query_fp, library_fps))


def load_gctx_gunzipped(gctx_path: Path) -> str:
    """
    Read a .gctx.gz file: decompress to sibling path, return path string.
    Uses Python gzip module to avoid OS gunzip issues with large files.
    """
    import gzip, shutil
    # Check if already decompressed
    ungz = gctx_path.parent / gctx_path.stem
    if not ungz.exists():
        print(f"  Decompressing {gctx_path.name} → {ungz.name} ...")
        chunk = 64 * 1024 * 1024  # 64MB chunks
        with gzip.open(str(gctx_path), "rb") as f_in:
            with open(str(ungz), "wb") as f_out:
                while True:
                    data = f_in.read(chunk)
                    if not data:
                        break
                    f_out.write(data)
        print(f"  Decompressed → {ungz} ({ungz.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"  Using existing decompressed {ungz.name}")
    return str(ungz)


def main():
    print("=" * 60)
    print("Layer 4 (Part 2): LINCS Chemical-Neighbor Proxy Signatures")
    print("=" * 60)

    # ── Load LINCS compound info with SMILES ──────────────────────────────────
    print("\n[1] Loading LINCS pert_info (compounds with SMILES)...")
    pert = pd.read_csv(PERT_INFO, sep="\t", compression="gzip")
    cmpd = pert[
        (pert["pert_type"] == "trt_cp") &
        pert["canonical_smiles"].notna() &
        (pert["canonical_smiles"] != "-666")
    ].copy()
    cmpd = cmpd.drop_duplicates("pert_id")
    print(f"  {len(cmpd)} unique LINCS compounds with SMILES")

    # ── Compute Morgan fingerprints for LINCS compounds ───────────────────────
    print("\n[2] Computing fingerprints for LINCS compounds...")
    lib_fps, lib_ids, lib_smiles, lib_names = [], [], [], []
    for _, row in cmpd.iterrows():
        fp = morgan_fp(row["canonical_smiles"])
        if fp is not None:
            lib_fps.append(fp)
            lib_ids.append(row["pert_id"])
            lib_smiles.append(row["canonical_smiles"])
            lib_names.append(row["pert_iname"])
    print(f"  Valid fingerprints: {len(lib_fps)} / {len(cmpd)}")

    # ── Load signature info ───────────────────────────────────────────────────
    print("\n[3] Loading LINCS sig_info...")
    sig_info = pd.read_csv(SIG_INFO, sep="\t", compression="gzip")
    # Keep only compound perturbations
    sig_cmpd = sig_info[sig_info["pert_type"] == "trt_cp"].copy()
    print(f"  Compound signatures: {len(sig_cmpd)}")

    # Build pert_id → list of sig_ids mapping
    pert_to_sigs = sig_cmpd.groupby("pert_id")["sig_id"].apply(list).to_dict()

    # ── Decompress and open gctx ──────────────────────────────────────────────
    print("\n[4] Opening LINCS gctx...")
    gctx_path = load_gctx_gunzipped(GCTX_GZ)
    f = h5py.File(gctx_path, "r")
    matrix    = f["0"]["DATA"]["0"]["matrix"]        # shape: (n_genes, n_sigs)
    row_ids   = [x.decode() if isinstance(x, bytes) else x
                 for x in f["0"]["META"]["ROW"]["id"][:]]   # gene ids
    col_ids   = [x.decode() if isinstance(x, bytes) else x
                 for x in f["0"]["META"]["COL"]["id"][:]]   # sig ids
    col_idx   = {sid: i for i, sid in enumerate(col_ids)}
    print(f"  Matrix shape: {matrix.shape} (genes × sigs)")

    # ── Load gene info for landmark genes ────────────────────────────────────
    gene_info = pd.read_csv(GENE_INFO, sep="\t", compression="gzip")
    landmark_col = None
    for c in ["pr_is_lm", "is_lm", "landmark"]:
        if c in gene_info.columns:
            landmark_col = c
            break
    if landmark_col:
        lm_ids = set(gene_info[gene_info[landmark_col] == 1]["pr_gene_id"].astype(str).tolist())
        lm_mask = [rid in lm_ids for rid in row_ids]
        print(f"  Landmark genes in matrix: {sum(lm_mask)} / {len(row_ids)}")
    else:
        lm_mask = [True] * len(row_ids)
        print("  Could not identify landmark genes; using all genes")

    lm_indices = np.where(lm_mask)[0]
    lm_gene_ids = [row_ids[i] for i in lm_indices]
    gene_id_to_symbol = gene_info.set_index("pr_gene_id").get("pr_gene_symbol",
                        gene_info.set_index("pr_gene_id").iloc[:, 0]).to_dict() \
        if "pr_gene_id" in gene_info.columns else {}
    lm_gene_symbols = [str(gene_id_to_symbol.get(int(gid) if gid.isdigit() else gid, gid))
                       for gid in lm_gene_ids]

    def get_sig_mean(sig_ids_list: list[str]) -> np.ndarray | None:
        """Average multiple signatures for same compound."""
        valid_sigs = [sid for sid in sig_ids_list if sid in col_idx]
        if not valid_sigs:
            return None
        cols = [col_idx[sid] for sid in valid_sigs]
        # Read selected columns from landmark rows
        arr = matrix[np.ix_(lm_indices, cols)]  # [n_lm, n_sigs]
        return arr.mean(axis=1)

    # ── Find parent drug signatures ───────────────────────────────────────────
    print("\n[5] Finding parent drug (erlotinib / trametinib) signatures in LINCS...")
    parent_sigs = {}
    for drug, smi in PARENT_SMILES.items():
        query_fp = morgan_fp(smi)
        if query_fp is None:
            continue
        sims = batch_tanimoto(query_fp, lib_fps)
        top_idx = np.argsort(sims)[::-1][:5]
        print(f"  {drug}: top-5 LINCS matches by Tanimoto:")
        for ii in top_idx:
            print(f"    {lib_names[ii]} ({lib_ids[ii]}): Tanimoto={sims[ii]:.3f}")
        # Use top match if sim > 0.3
        best_i = top_idx[0]
        best_sim = sims[best_i]
        if best_sim >= 0.3:
            pid = lib_ids[best_i]
            sig_list = pert_to_sigs.get(pid, [])
            sig_vec = get_sig_mean(sig_list)
            if sig_vec is not None:
                parent_sigs[drug] = sig_vec
                print(f"  → Using {lib_names[best_i]} (sim={best_sim:.3f}, {len(sig_list)} sigs) as {drug} proxy")
        else:
            print(f"  → No high-similarity LINCS match for {drug} (best={best_sim:.3f})")
            # Use the best match anyway
            pid = lib_ids[best_i]
            sig_list = pert_to_sigs.get(pid, [])
            sig_vec = get_sig_mean(sig_list)
            if sig_vec is not None:
                parent_sigs[drug] = sig_vec
                print(f"  → Using {lib_names[best_i]} (sim={best_sim:.3f}) as fallback proxy")

    # ── Compute proxy signatures for each analogue ────────────────────────────
    print("\n[6] Computing proxy signatures for generated analogues...")
    luad = pd.read_csv(DATA / "candidates" / "generated_compounds_luad.csv")
    skcm = pd.read_csv(DATA / "candidates" / "generated_compounds_skcm.csv")
    luad = luad[luad["tanimoto"] >= 0.10].copy()
    skcm = skcm[skcm["tanimoto"] >= 0.10].copy()
    all_analogues = pd.concat([luad, skcm], ignore_index=True)
    print(f"  Processing {len(all_analogues)} analogues (Tanimoto≥0.10)...")

    proxy_rows = []
    for _, row in all_analogues.iterrows():
        smi = row["smiles"]
        query_fp = morgan_fp(smi)
        if query_fp is None:
            continue
        sims = batch_tanimoto(query_fp, lib_fps)
        top_k_idx = np.argsort(sims)[::-1][:TOP_K]
        top_k_sims = sims[top_k_idx]

        # Filter by minimum similarity
        valid = top_k_sims >= MIN_SIM
        if not any(valid):
            valid = np.array([True] + [False]*(TOP_K-1))  # at least use best

        # Weighted average signature
        w_sims = top_k_sims[valid]
        proxy_vec = None
        for ii, w in zip(top_k_idx[valid], w_sims):
            pid = lib_ids[ii]
            sigs = pert_to_sigs.get(pid, [])
            sig_vec = get_sig_mean(sigs)
            if sig_vec is not None:
                if proxy_vec is None:
                    proxy_vec = sig_vec * w
                    total_w = w
                else:
                    proxy_vec += sig_vec * w
                    total_w += w

        if proxy_vec is None or total_w == 0:
            continue
        proxy_vec = proxy_vec / total_w

        proxy_row = {
            "DRUG_NAME":        row["DRUG_NAME"],
            "DRUG_ID":          row["DRUG_ID"],
            "smiles":           smi,
            "seed_drug":        row["seed_drug"],
            "cancer_type":      row["cancer_type"],
            "delta_improvement":row["delta_improvement"],
            "mean_value_hat":   row["mean_value_hat"],
            "tanimoto":         row["tanimoto"],
            "best_lincs_sim":   float(top_k_sims[0]),
            "best_lincs_name":  lib_names[top_k_idx[0]],
        }
        for gi, sym in enumerate(lm_gene_symbols):
            proxy_row[sym] = float(proxy_vec[gi])
        proxy_rows.append(proxy_row)

    proxy_df = pd.DataFrame(proxy_rows)
    print(f"  Proxy signatures computed: {len(proxy_df)} / {len(all_analogues)}")
    print(f"  Mean best LINCS similarity: {proxy_df['best_lincs_sim'].mean():.3f}")
    proxy_df.to_csv(OUT_DIR / "proxy_signatures.csv", index=False)

    # Save parent signatures
    for drug, sig_vec in parent_sigs.items():
        parent_row = {"DRUG_NAME": drug, "seed_drug": drug}
        for gi, sym in enumerate(lm_gene_symbols):
            parent_row[sym] = float(sig_vec[gi])
        pd.DataFrame([parent_row]).to_csv(
            OUT_DIR / f"parent_signature_{drug}.csv", index=False
        )

    # Save gene list
    pd.DataFrame({"gene_id": lm_gene_ids, "gene_symbol": lm_gene_symbols}).to_csv(
        OUT_DIR / "lincs_landmark_genes.csv", index=False
    )

    f.close()
    print(f"\nAll outputs → {OUT_DIR}")
    print(f"  proxy_signatures.csv: {len(proxy_df)} rows × {proxy_df.shape[1]} cols")


if __name__ == "__main__":
    main()
