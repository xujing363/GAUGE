#!/usr/bin/env python3
"""
Layer 5: Molecular docking validation using GNINA.

Targets:
  - EGFR (for erlotinib and LUAD analogues): PDB 4HJO (erlotinib bound)
  - MAP2K1/MEK1 (for trametinib and SKCM analogues): PDB 4LMN (trametinib bound)

Steps:
  1. Download PDB structures
  2. Prepare ligands (3D embedding via RDKit)
  3. Define docking box from co-crystal ligand
  4. Run GNINA docking (CNN scoring)
  5. Parse scores, compare analogue vs parent

Run with: conda run -n kg_GAUGE python v05_docking.py
Requires: gnina binary at /mnt/raid5/xujing/KG/DrugDesign_Sec/docking/gnina/gnina.1.3.2
Output: results/layer5_docking/
"""
from __future__ import annotations
import json, os, subprocess, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "docking"
OUT_DIR = ROOT / "results" / "layer5_docking"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LIG_SDF  = DATA / "ligands" / "sdf"
LIG_SDF.mkdir(parents=True, exist_ok=True)

GNINA    = Path("/mnt/raid5/xujing/KG/DrugDesign_Sec/docking/gnina/gnina.1.3.2")
GNINA_CUDA = Path("/mnt/raid5/xujing/KG/DrugDesign_Sec/docking/gnina/gnina.1.3.2.cuda12.8")

# gnina.1.3.2 needs libcudnn.so.9 but works on this glibc; cuda12.8 requires GLIBC_2.35
# Always use gnina.1.3.2 with explicit LD_LIBRARY_PATH
_CUDNN_DIR = "/mnt/raid5/xujing/miniconda3/lib/python3.10/site-packages/nvidia/cudnn/lib"
import os as _os
_GNINA_ENV = dict(_os.environ)
_GNINA_ENV["LD_LIBRARY_PATH"] = _CUDNN_DIR + ":/usr/local/cuda/lib64:" + _GNINA_ENV.get("LD_LIBRARY_PATH", "")

GNINA_BIN = GNINA  # use gnina.1.3.2 (cuda12.8 requires newer GLIBC)

TARGETS = {
    "EGFR": {
        "pdb_id":  "4HJO",
        "seed":    "erlotinib",
        "cancer":  "TCGA-LUAD",
        "ligand_resname": "AQ4",  # erlotinib in 4HJO is AQ4
        # Box center from co-crystal erlotinib binding site (EGFR kinase domain)
        "center": (-9.42, 1.73, -3.57),  # approximate; updated after prep
        "size":   (22, 22, 22),
    },
    "MAP2K1": {
        "pdb_id":  "4LMN",
        "seed":    "trametinib",
        "cancer":  "TCGA-SKCM",
        "ligand_resname": "EUI",  # trametinib in 4LMN
        "center": (0.0, 0.0, 0.0),  # updated after structure download
        "size":   (22, 22, 22),
    },
}

PARENT_SMILES = {
    "erlotinib":  "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "trametinib": "CC1=C(C(=O)N2CCN(CC2)C(=O)c2cc(I)c(F)c(NC(=O)c3ccc(F)cc3Cl)c2)C=NN1CC",
}


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        print(f"  {pdb_id}.pdb already exists, skipping download")
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, str(out_path))
    print(f"  Saved → {out_path}")
    return out_path


def extract_box_from_pdb(pdb_path: Path, ligand_resname: str) -> tuple[tuple, tuple]:
    """Extract docking box center from co-crystal ligand in PDB."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("HETATM") or line.startswith("ATOM")) and \
               line[17:20].strip() == ligand_resname:
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append([x, y, z])
                except Exception:
                    pass
    if not coords:
        print(f"  WARN: ligand {ligand_resname} not found in {pdb_path.name}")
        return (0.0, 0.0, 0.0), (22, 22, 22)
    center = np.mean(coords, axis=0)
    span   = np.max(coords, axis=0) - np.min(coords, axis=0)
    size   = tuple(int(max(s + 10, 18)) for s in span)  # add 10Å padding
    print(f"  Box center: {center.round(2)}, size: {size}")
    return tuple(center.round(3)), size


def prepare_receptor(pdb_path: Path, out_dir: Path, target_name: str) -> Path:
    """Extract protein only (remove HETATM except important cofactors)."""
    out_path = out_dir / f"{target_name}_receptor.pdb"
    if out_path.exists():
        return out_path
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("END"):
                lines.append(line)
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"  Receptor written → {out_path}")
    return out_path


def smiles_to_sdf(name: str, smiles: str, out_dir: Path) -> Path | None:
    """Convert SMILES to 3D SDF using RDKit ETKDGv3."""
    out_path = out_dir / f"{name}.sdf"
    if out_path.exists():
        return out_path
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  WARN: invalid SMILES for {name}")
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        # Try ETKDG fallback
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        print(f"  WARN: 3D embedding failed for {name}")
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return out_path


def run_gnina(receptor: Path, ligand_sdf: Path, center: tuple, size: tuple,
              out_sdf: Path, cuda_device: int = 0) -> dict | None:
    """Run GNINA docking and return best CNN score."""
    if out_sdf.exists():
        return parse_gnina_output(out_sdf)

    cmd = [
        str(GNINA_BIN),
        "-r", str(receptor),
        "-l", str(ligand_sdf),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(size[0]),
        "--size_y", str(size[1]),
        "--size_z", str(size[2]),
        "--exhaustiveness", "8",
        "--num_modes", "5",
        "--cnn_scoring", "rescore",
        "-o", str(out_sdf),
        "--device", str(cuda_device),
        "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                env=_GNINA_ENV)
        if result.returncode != 0:
            print(f"  GNINA error: {result.stderr[:300]}")
            return None
        return parse_gnina_output(out_sdf)
    except subprocess.TimeoutExpired:
        print(f"  GNINA timeout for {ligand_sdf.name}")
        return None
    except Exception as e:
        print(f"  GNINA exception: {e}")
        return None


def parse_gnina_output(sdf_path: Path) -> dict | None:
    """Parse GNINA output SDF for CNN scores."""
    if not sdf_path.exists():
        return None
    scores = []
    cnn_affin = []
    with open(sdf_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "minimizedAffinity" in line:
            if i+1 < len(lines):
                try:
                    scores.append(float(lines[i+1].strip()))
                except Exception:
                    pass
        elif "CNNaffinity" in line and "variance" not in line:
            if i+1 < len(lines):
                try:
                    cnn_affin.append(float(lines[i+1].strip()))
                except Exception:
                    pass
        i += 1

    if not scores:
        return None
    return {
        "best_vina_score":  min(scores),  # most negative = best
        "best_cnn_affinity": max(cnn_affin) if cnn_affin else None,
        "n_poses":           len(scores),
    }


def main():
    print("=" * 60)
    print("Layer 5: Molecular Docking Validation (GNINA)")
    print("=" * 60)
    print(f"  GNINA binary: {GNINA_BIN} (exists: {GNINA_BIN.exists()})")

    if not GNINA_BIN.exists():
        print(f"  ERROR: GNINA binary not found at {GNINA_BIN}")
        print("  Please ensure gnina is at the expected path.")
        return

    # Load analogues
    luad = pd.read_csv(ROOT / "data" / "candidates" / "generated_compounds_luad.csv")
    skcm = pd.read_csv(ROOT / "data" / "candidates" / "generated_compounds_skcm.csv")
    luad = luad[luad["tanimoto"] >= 0.10].copy()
    skcm = skcm[skcm["tanimoto"] >= 0.10].copy()

    docking_rows = []

    for target_name, target_info in TARGETS.items():
        pdb_id  = target_info["pdb_id"]
        seed    = target_info["seed"]
        cancer  = target_info["cancer"]
        lig_res = target_info["ligand_resname"]

        print(f"\n  === {target_name} ({pdb_id}) / {seed} ===")

        # ── Setup target directory ─────────────────────────────────────────────
        tgt_dir = DATA / "targets" / target_name
        tgt_dir.mkdir(parents=True, exist_ok=True)
        results_dir = OUT_DIR / target_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # ── Download PDB ──────────────────────────────────────────────────────
        pdb_path = download_pdb(pdb_id, tgt_dir)
        receptor_path = prepare_receptor(pdb_path, tgt_dir, target_name)

        # ── Extract box from co-crystal ligand ────────────────────────────────
        center, size = extract_box_from_pdb(pdb_path, lig_res)
        if center == (0.0, 0.0, 0.0):
            # Use known binding site coordinates
            if target_name == "EGFR":
                center = (-9.42, 1.73, -3.57)  # EGFR ATP binding site
            else:
                center = (0.92, 3.62, -1.44)   # MEK1 allosteric site

        # ── Prepare parent drug ───────────────────────────────────────────────
        parent_smi = PARENT_SMILES[seed]
        parent_sdf = smiles_to_sdf(f"parent_{seed}", parent_smi, LIG_SDF)

        parent_scores = None
        if parent_sdf:
            parent_out = results_dir / f"parent_{seed}_gnina.sdf"
            parent_scores = run_gnina(receptor_path, parent_sdf, center, size,
                                      parent_out, cuda_device=0)
            if parent_scores:
                print(f"  Parent {seed}: vina={parent_scores['best_vina_score']:.2f}, "
                      f"cnn_affin={parent_scores.get('best_cnn_affinity'):.3f}")

        # ── Select analogues to dock ──────────────────────────────────────────
        analogue_df = luad if seed == "erlotinib" else skcm
        improved = analogue_df[analogue_df["delta_improvement"] > 0].copy()
        # Also dock top-10 by delta_improvement (even if delta > 0)
        top10 = analogue_df.nlargest(10, "delta_improvement")
        to_dock = pd.concat([improved, top10]).drop_duplicates("DRUG_ID")
        print(f"  Docking {len(to_dock)} analogues "
              f"({len(improved)} improved + top-10 by delta)")

        for _, analogue in to_dock.iterrows():
            name = str(analogue["DRUG_NAME"]).replace("/", "_")
            drug_id = analogue["DRUG_ID"]
            smi = analogue["smiles"]

            lig_sdf = smiles_to_sdf(f"{name}_{drug_id}", smi, LIG_SDF)
            if lig_sdf is None:
                continue

            out_sdf = results_dir / f"{name}_{drug_id}_gnina.sdf"
            scores = run_gnina(receptor_path, lig_sdf, center, size,
                               out_sdf, cuda_device=0)

            delta_vina = None
            delta_cnn  = None
            if scores and parent_scores:
                delta_vina = scores["best_vina_score"] - parent_scores["best_vina_score"]
                if scores.get("best_cnn_affinity") and parent_scores.get("best_cnn_affinity"):
                    delta_cnn = scores["best_cnn_affinity"] - parent_scores["best_cnn_affinity"]

            row = {
                "target":           target_name,
                "pdb_id":           pdb_id,
                "DRUG_NAME":        analogue["DRUG_NAME"],
                "DRUG_ID":          drug_id,
                "seed_drug":        seed,
                "cancer_type":      cancer,
                "smiles":           smi,
                "delta_improvement": analogue["delta_improvement"],
                "tanimoto":         analogue["tanimoto"],
                "is_improved":      analogue["delta_improvement"] > 0,
                "vina_score":       scores["best_vina_score"]  if scores else None,
                "cnn_affinity":     scores.get("best_cnn_affinity") if scores else None,
                "n_poses":          scores["n_poses"] if scores else None,
                "parent_vina":      parent_scores["best_vina_score"] if parent_scores else None,
                "parent_cnn":       parent_scores.get("best_cnn_affinity") if parent_scores else None,
                "delta_vina_vs_parent": delta_vina,
                "delta_cnn_vs_parent":  delta_cnn,
            }
            docking_rows.append(row)
            if scores:
                print(f"    {name}: vina={scores['best_vina_score']:.2f} "
                      f"(Δ={delta_vina:.2f}), cnn={scores.get('best_cnn_affinity'):.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    if docking_rows:
        df = pd.DataFrame(docking_rows)
        df.to_csv(OUT_DIR / "docking_scores.csv", index=False)
        print(f"\n  Saved → {OUT_DIR}/docking_scores.csv ({len(df)} rows)")

        # Summary
        summary = {}
        for target_name in TARGETS:
            sub = df[df["target"] == target_name]
            imp = sub[sub["is_improved"]]
            parent_vina = sub["parent_vina"].iloc[0] if len(sub) else None
            summary[target_name] = {
                "pdb_id":        TARGETS[target_name]["pdb_id"],
                "seed_drug":     TARGETS[target_name]["seed"],
                "parent_vina_score": float(parent_vina) if parent_vina else None,
                "n_docked_analogues": int(len(sub)),
                "n_improved_analogues": int(len(imp)),
                "n_better_than_parent_vina": int((sub["delta_vina_vs_parent"] < 0).sum())
                                              if "delta_vina_vs_parent" in sub.columns else 0,
                "improved_details": [
                    {
                        "DRUG_NAME":     r["DRUG_NAME"],
                        "delta_improvement": round(float(r["delta_improvement"]), 6),
                        "vina_score":    round(float(r["vina_score"]), 3) if r["vina_score"] else None,
                        "delta_vina":    round(float(r["delta_vina_vs_parent"]), 3)
                                         if r["delta_vina_vs_parent"] is not None else None,
                    }
                    for _, r in imp.iterrows()
                ]
            }

        with open(OUT_DIR / "docking_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
