"""Tests for the molecular-design utilities (generation, properties, similarity).

All network-free: generation and descriptors are pure RDKit, similarity is
computed against the bundled GAUGE drug library.
"""
import pytest

from gauge_core import load_bundle
from gauge_core import moldesign as md

# Erlotinib (an EGFR inhibitor) -- a good, BRICS-fragmentable seed.
ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle("gdsc_cell_split")


def test_canonical_roundtrip_and_invalid():
    assert md.canonical("c1ccccc1") == md.canonical("C1=CC=CC=C1")
    assert md.canonical("not_a_smiles") is None


def test_generate_analogs_returns_novel_valid_molecules():
    gen = md.generate_analogs([ERLOTINIB], n=10)
    assert len(gen) > 0
    seed_canon = md.canonical(ERLOTINIB)
    for smi in gen:
        assert md.canonical(smi) is not None  # every candidate is a valid molecule
        assert smi != seed_canon              # and is not just the seed echoed back


def test_generate_analogs_empty_for_unfragmentable_seed():
    # A trivial molecule with no BRICS bonds yields no recombinations.
    assert md.generate_analogs(["C"], n=5) == []


def test_molecular_properties():
    props = md.molecular_properties(ERLOTINIB)
    assert props["valid"] is True
    assert 350 < props["mol_weight"] < 420
    assert 0.0 <= props["qed"] <= 1.0
    assert isinstance(props["lipinski_ok"], bool)
    assert md.molecular_properties("xyz_not_smiles") == {"valid": False}


def test_nearest_library_drug(bundle):
    near = md.nearest_library_drug(bundle, ERLOTINIB)
    assert near is not None
    assert 0.0 <= near["tanimoto"] <= 1.0
    # An exact library member resolves to itself at Tanimoto 1.0.
    first_smiles = str(bundle.drug_library.iloc[0]["canonical_smiles"])
    exact = md.nearest_library_drug(bundle, first_smiles)
    assert exact["tanimoto"] == 1.0
