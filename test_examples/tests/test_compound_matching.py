import pytest
from unittest.mock import patch
from rdkit import Chem
from scoring_optimizer.compound_matching import morgan_tanimoto, assign_to_crystal
import scoring_optimizer.compound_matching as _cm


def _mol(smi):
    return Chem.MolFromSmiles(smi)


def test_identical_molecules_score_one():
    mol = _mol("c1ccccc1")
    assert morgan_tanimoto(mol, mol) == pytest.approx(1.0)


def test_dissimilar_molecules_score_low():
    assert morgan_tanimoto(_mol("c1ccccc1"), _mol("CCC")) < 0.3


def test_assign_returns_none_below_threshold():
    query = _mol("CCC")
    crystals = [_mol("c1ccccc1"), _mol("c1ccncc1")]
    idx, sim = assign_to_crystal(query, crystals, threshold=0.6)
    assert idx is None
    assert sim < 0.6


def test_assign_picks_exact_match():
    query = _mol("c1ccncc1")  # pyridine
    crystals = [_mol("CCC"), _mol("c1ccncc1"), _mol("c1ccccc1")]
    idx, sim = assign_to_crystal(query, crystals, threshold=0.6)
    assert idx == 1
    assert sim == pytest.approx(1.0)


def test_assign_returns_best_when_multiple_match():
    query = _mol("c1ccc(N)cc1")   # aniline
    # phenol is more similar to aniline than propane
    crystals = [_mol("CCC"), _mol("c1ccc(O)cc1")]
    idx, sim = assign_to_crystal(query, crystals, threshold=0.3)
    assert idx == 1  # phenol, not propane


def test_assign_tie_returns_first_index():
    """Verify that ties are broken by returning the lowest-index crystal."""
    mol = _mol("c1ccccc1")
    idx, sim = assign_to_crystal(mol, [_mol("c1ccccc1"), _mol("c1ccccc1")], threshold=0.5)
    assert idx == 0
    assert sim == pytest.approx(1.0)


def test_assign_empty_crystal_list_returns_none():
    """Verify that empty crystal list returns None with 0.0 similarity."""
    idx, sim = assign_to_crystal(_mol("c1ccccc1"), [], threshold=0.5)
    assert idx is None
    assert sim == pytest.approx(0.0)


def test_morgan_tanimoto_none_mol_raises():
    with pytest.raises(ValueError, match="mol_a and mol_b"):
        morgan_tanimoto(None, _mol("c1ccccc1"))


def test_assign_none_sar_mol_raises():
    with pytest.raises(ValueError, match="sar_mol"):
        assign_to_crystal(None, [_mol("c1ccccc1")])


def test_morgan_tanimoto_non_default_params_uses_fresh_generator():
    """Covers the non-default-params branch (lines 40-45): radius != 2 triggers new generator."""
    mol_a = _mol("c1ccccc1")
    mol_b = _mol("c1ccncc1")
    sim = morgan_tanimoto(mol_a, mol_b, radius=3, n_bits=1024)
    assert 0.0 <= sim <= 1.0


def test_morgan_tanimoto_fallback_when_gen_is_none():
    """Covers the else/fallback branch (lines 46-48): _DEFAULT_MORGAN_GEN patched to None."""
    mol_a = _mol("c1ccccc1")
    mol_b = _mol("c1ccccc1")
    with patch.object(_cm, "_DEFAULT_MORGAN_GEN", None):
        sim = morgan_tanimoto(mol_a, mol_b)
    assert sim == pytest.approx(1.0)
