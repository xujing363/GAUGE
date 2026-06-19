#!/usr/bin/env python3
"""
Layer 6: Chemistry & ADMET validation of generated analogues.

Computes drug-likeness, synthesizability, PAINS alerts, and structural
similarity metrics for all analogues vs parent drugs.

Run with: conda run -n kg_GAUGE python v01_chemistry_admet.py
Output: results/layer6_chemistry/
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, AllChem, FilterCatalog, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import sys as _sys
_sys.path.insert(0, "/mnt/raid5/xujing/miniconda3/envs/kg_GAUGE/lib/python3.10/site-packages/rdkit/Contrib/SA_Score")
import sascorer  # SA_Score from RDKit Contrib

ROOT    = Path(__file__).resolve().parents[1]   # cnm/validation/
DATA    = ROOT / "data" / "candidates"
OUT     = ROOT / "results" / "layer6_chemistry"
OUT.mkdir(parents=True, exist_ok=True)

PARENT_SMILES = {
    "erlotinib":  "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "trametinib": "CC1=C(C(=O)N2CCN(CC2)C(=O)c2cc(I)c(F)c(NC(=O)c3ccc(F)cc3Cl)c2)C=NN1CC",
}

# ── PAINS catalog ─────────────────────────────────────────────────────────────
_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog.FilterCatalog(_pains_params)


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto(smi1: str, smi2: str) -> float:
    fp1, fp2 = morgan_fp(smi1), morgan_fp(smi2)
    if fp1 is None or fp2 is None:
        return float("nan")
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def compute_chemistry(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False}
    mol = Chem.AddHs(mol)
    try:
        sa = sascorer.calculateScore(mol)
    except Exception:
        sa = float("nan")
    mol_noh = Chem.RemoveHs(mol)

    mw       = Descriptors.MolWt(mol_noh)
    logp     = Descriptors.MolLogP(mol_noh)
    hbd      = rdMolDescriptors.CalcNumHBD(mol_noh)
    hba      = rdMolDescriptors.CalcNumHBA(mol_noh)
    rotb     = rdMolDescriptors.CalcNumRotatableBonds(mol_noh)
    tpsa     = rdMolDescriptors.CalcTPSA(mol_noh)
    qed_val  = QED.qed(mol_noh)
    n_rings  = rdMolDescriptors.CalcNumRings(mol_noh)
    n_arom   = rdMolDescriptors.CalcNumAromaticRings(mol_noh)
    fsp3     = rdMolDescriptors.CalcFractionCSP3(mol_noh)

    # Lipinski violations
    lip_viols = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    # PAINS alerts
    n_pains = len(PAINS_CATALOG.GetMatches(mol_noh))

    # Veber filter (oral bioavailability): rotb ≤ 10 and TPSA ≤ 140
    veber_ok = (rotb <= 10) and (tpsa <= 140)

    # Murcko scaffold
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol_noh)
    except Exception:
        scaffold = ""

    # Hard filter: fail if invalid, severe PAINS, or extreme properties
    hard_fail = (
        mol is None or
        n_pains > 0 or
        mw > 700 or
        logp > 7 or
        lip_viols >= 3
    )

    return {
        "valid":          True,
        "mw":             round(mw, 2),
        "logp":           round(logp, 3),
        "hbd":            hbd,
        "hba":            hba,
        "rotb":           rotb,
        "tpsa":           round(tpsa, 2),
        "qed":            round(qed_val, 4),
        "sa_score":       round(float(sa), 4),
        "n_rings":        n_rings,
        "n_aromatic_rings": n_arom,
        "fsp3":           round(fsp3, 4),
        "lipinski_violations": lip_viols,
        "n_pains_alerts": n_pains,
        "veber_ok":       veber_ok,
        "scaffold":       scaffold,
        "hard_fail":      hard_fail,
    }


def chemistry_score(row: dict, parent_row: dict) -> float:
    """Composite chemistry score: higher = better drug-like analogue."""
    if not row.get("valid", False):
        return float("-inf")
    # Normalise relative to parent
    dQED  =  row["qed"]      - parent_row["qed"]          # higher QED = better
    dSA   = -(row["sa_score"] - parent_row["sa_score"])    # lower SA = better (negate)
    dLip  = -(row["lipinski_violations"] - parent_row["lipinski_violations"])
    dPAIN = -(row["n_pains_alerts"] - parent_row["n_pains_alerts"])
    return round(float(dQED + 0.5*dSA + 0.3*dLip + 0.2*dPAIN), 5)


def main():
    print("=" * 60)
    print("Layer 6: Chemistry & ADMET Validation")
    print("=" * 60)

    luad = pd.read_csv(DATA / "generated_compounds_luad.csv")
    skcm = pd.read_csv(DATA / "generated_compounds_skcm.csv")

    # Keep only Tanimoto-filtered analogues (tanimoto >= 0.10)
    luad = luad[luad["tanimoto"] >= 0.10].copy()
    skcm = skcm[skcm["tanimoto"] >= 0.10].copy()
    df   = pd.concat([luad, skcm], ignore_index=True)
    print(f"  Analogues (Tanimoto≥0.10): {len(df)} "
          f"(erlotinib/LUAD: {len(luad)}, trametinib/SKCM: {len(skcm)})")

    # Add parent rows
    parent_rows = []
    for seed, smi in PARENT_SMILES.items():
        chem = compute_chemistry(smi)
        chem.update({"DRUG_NAME": seed, "smiles": smi, "seed_drug": seed,
                     "delta_improvement": 0, "is_parent": True,
                     "tanimoto_to_parent": 1.0})
        parent_rows.append(chem)

    # Compute chemistry for all analogues
    rows = []
    parent_chem = {p: compute_chemistry(s) for p, s in PARENT_SMILES.items()}

    for _, analogue in df.iterrows():
        smi = analogue["smiles"]
        seed = analogue["seed_drug"]
        chem = compute_chemistry(smi)
        chem["DRUG_NAME"]         = analogue["DRUG_NAME"]
        chem["DRUG_ID"]           = analogue["DRUG_ID"]
        chem["smiles"]            = smi
        chem["seed_drug"]         = seed
        chem["cancer_type"]       = analogue["cancer_type"]
        chem["delta_improvement"] = analogue["delta_improvement"]
        chem["tanimoto_to_parent"]= analogue["tanimoto"]
        chem["mean_value_hat"]    = analogue["mean_value_hat"]
        chem["is_parent"]         = False
        chem["is_improved"]       = analogue["delta_improvement"] > 0
        # Tanimoto to parent (already have it)
        chem["chem_score_vs_parent"] = chemistry_score(chem, parent_chem[seed])
        rows.append(chem)

    result = pd.DataFrame(rows)

    # Summary
    n_valid      = result["valid"].sum()
    n_hard_fail  = result["hard_fail"].sum()
    n_pains_any  = (result["n_pains_alerts"] > 0).sum()
    n_veber_ok   = result["veber_ok"].sum()
    n_lip_ok     = (result["lipinski_violations"] == 0).sum()

    print(f"\n  Total analogues: {len(result)}")
    print(f"  Valid SMILES:    {n_valid} / {len(result)}")
    print(f"  Hard fail:       {n_hard_fail} ({100*n_hard_fail/len(result):.1f}%)")
    print(f"  PAINS alerts:    {n_pains_any} ({100*n_pains_any/len(result):.1f}%)")
    print(f"  Lipinski OK (0 violations): {n_lip_ok} ({100*n_lip_ok/len(result):.1f}%)")
    print(f"  Veber OK:        {n_veber_ok} ({100*n_veber_ok/len(result):.1f}%)")

    # Improved vs non-improved chemistry comparison
    improved = result[result["is_improved"]]
    non_imp  = result[~result["is_improved"]]
    print(f"\n  Improved analogues (delta > 0): {len(improved)}")
    if len(improved) > 0:
        print("  Chemistry of improved analogues:")
        for col in ["qed", "sa_score", "lipinski_violations", "n_pains_alerts",
                    "tanimoto_to_parent", "chem_score_vs_parent"]:
            vals = improved[col].dropna()
            print(f"    {col}: mean={vals.mean():.4f}, range=[{vals.min():.3f}, {vals.max():.3f}]")

    result.to_csv(OUT / "chemistry_admet.csv", index=False)
    print(f"\n  Saved → {OUT}/chemistry_admet.csv")

    # Per-seed summary
    summary = {}
    for seed in ["erlotinib", "trametinib"]:
        sub = result[result["seed_drug"] == seed]
        imp = sub[sub["is_improved"]]
        summary[seed] = {
            "n_total":          int(len(sub)),
            "n_improved":       int(len(imp)),
            "n_hard_fail":      int(sub["hard_fail"].sum()),
            "n_pains":          int((sub["n_pains_alerts"] > 0).sum()),
            "mean_qed":         round(float(sub["qed"].mean()), 4),
            "mean_sa_score":    round(float(sub["sa_score"].mean()), 4),
            "mean_lipinski_violations": round(float(sub["lipinski_violations"].mean()), 4),
            "mean_tanimoto_to_parent":  round(float(sub["tanimoto_to_parent"].mean()), 4),
            "improved_analogues": [
                {
                    "DRUG_NAME":     r["DRUG_NAME"],
                    "smiles":        r["smiles"],
                    "delta_improvement": round(float(r["delta_improvement"]), 6),
                    "qed":           round(float(r["qed"]), 4),
                    "sa_score":      round(float(r["sa_score"]), 4),
                    "lipinski_violations": int(r["lipinski_violations"]),
                    "n_pains_alerts": int(r["n_pains_alerts"]),
                    "tanimoto_to_parent": round(float(r["tanimoto_to_parent"]), 4),
                    "hard_fail":     bool(r["hard_fail"]),
                    "chem_score_vs_parent": round(float(r["chem_score_vs_parent"]), 5),
                }
                for _, r in imp.iterrows()
            ],
        }

    with open(OUT / "chemistry_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Filter: pass hard filter and are improved
    tier_A_chem = improved[~improved["hard_fail"]]
    print(f"\n  Improved analogues passing hard chemistry filter: {len(tier_A_chem)}")
    if len(tier_A_chem) > 0:
        print(tier_A_chem[["DRUG_NAME","seed_drug","delta_improvement","qed","sa_score",
                            "lipinski_violations","n_pains_alerts","tanimoto_to_parent"]].to_string(index=False))

    print(f"\nAll outputs → {OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
