"""Molecular-design utilities for the GAUGE Molecular Design Scoring page.

This is the offline, in-app analogue of the paper's REINVENT4 design task:
- `generate_analogs` performs a lightweight *de novo* generation from one or
  more seed molecules by BRICS-fragmenting them and recombining the fragments
  into new, valid, drug-like structures (no heavy generative model needed, so
  it runs anywhere GAUGE runs).
- `molecular_properties` / `nearest_library_drug` enrich each candidate with
  physicochemical descriptors and its similarity to the closest GAUGE-library
  drug, so the design page reports *what* a molecule is, not only its score.

GAUGE itself still supplies the relative-sensitive-value reward for ranking
(via `gauge_core.predict`); nothing here predicts drug response.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _drugwm_path  # noqa: F401  must precede rdkit/numpy-heavy imports
from .bundle import ModelBundle

from rdkit import Chem
from rdkit.Chem import BRICS, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from drugwm.features import morgan_fp  # noqa: E402


def canonical(smiles: str) -> str | None:
    """Canonical SMILES, or None if RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def generate_analogs(
    seed_smiles: list[str],
    n: int = 20,
    max_iter: int = 4000,
) -> list[str]:
    """Generate up to `n` novel, valid analog molecules from seed SMILES.

    Uses BRICS decomposition + recombination: each seed is broken into
    retrosynthetic fragments, and fragments are reassembled into new molecules.
    Exact seed structures are excluded, so every returned SMILES is a genuinely
    new candidate. Returns canonical SMILES strings.
    """
    fragments: set[str] = set()
    seeds_canon: set[str] = set()
    for smi in seed_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        seeds_canon.add(Chem.MolToSmiles(mol))
        try:
            fragments.update(BRICS.BRICSDecompose(mol))
        except Exception:  # noqa: BLE001 - skip seeds BRICS can't fragment
            continue

    frag_mols = [m for m in (Chem.MolFromSmiles(f) for f in fragments) if m is not None]
    if len(frag_mols) < 2:
        return []

    out: list[str] = []
    seen: set[str] = set(seeds_canon)
    try:
        builder = BRICS.BRICSBuild(frag_mols, scrambleReagents=True, maxDepth=3)
    except TypeError:  # pragma: no cover - older rdkit signatures
        builder = BRICS.BRICSBuild(frag_mols)

    for i, mol in enumerate(builder):
        if i >= max_iter or len(out) >= n:
            break
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
        except Exception:  # noqa: BLE001 - reject invalid recombinations
            continue
        if smi and smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def molecular_properties(smiles: str) -> dict[str, Any]:
    """Drug-likeness / physicochemical descriptors for one molecule."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"valid": False}
    return {
        "valid": True,
        "mol_weight": round(Descriptors.MolWt(mol), 1),
        "logp": round(Crippen.MolLogP(mol), 2),
        "qed": round(QED.qed(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "h_donors": int(Lipinski.NumHDonors(mol)),
        "h_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "lipinski_ok": bool(
            Descriptors.MolWt(mol) <= 500
            and Crippen.MolLogP(mol) <= 5
            and Lipinski.NumHDonors(mol) <= 5
            and Lipinski.NumHAcceptors(mol) <= 10
        ),
    }


_LIB_FP_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}


def _library_fps(bundle: ModelBundle) -> tuple[list[str], np.ndarray]:
    """(drug names, boolean Morgan-fingerprint matrix) for the bundle library, cached."""
    if bundle.mode in _LIB_FP_CACHE:
        return _LIB_FP_CACHE[bundle.mode]
    names: list[str] = []
    mats: list[np.ndarray] = []
    for row in bundle.drug_library.itertuples(index=False):
        smi = getattr(row, "canonical_smiles", None) or getattr(row, "smiles", None)
        fp = morgan_fp(str(smi)) if smi is not None else None
        if fp is None:
            continue
        names.append(str(row.DRUG_NAME))
        mats.append(fp.astype(bool))
    matrix = np.vstack(mats) if mats else np.zeros((0, 2048), dtype=bool)
    _LIB_FP_CACHE[bundle.mode] = (names, matrix)
    return names, matrix


def nearest_library_drug(bundle: ModelBundle, smiles: str) -> dict[str, Any] | None:
    """Closest GAUGE-library drug to a molecule by Tanimoto over Morgan fingerprints."""
    q = morgan_fp(str(smiles))
    if q is None:
        return None
    names, matrix = _library_fps(bundle)
    if matrix.shape[0] == 0:
        return None
    qb = q.astype(bool)
    inter = np.logical_and(matrix, qb).sum(axis=1)
    union = np.logical_or(matrix, qb).sum(axis=1)
    sim = np.where(union > 0, inter / union, 0.0)
    i = int(sim.argmax())
    return {"nearest_drug": names[i], "tanimoto": round(float(sim[i]), 3)}
