# tests/test_crystal_processing.py
import numpy as np
import pytest
from pathlib import Path
from rdkit import Chem
from scoring_optimizer.crystal_processing import (
    find_binding_waters,
    pharmacophore_features,
    write_pharma_restr,
    crystal_ligand_coords,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_find_waters_within_cutoff():
    waters = find_binding_waters(
        FIXTURES / "mini_crystal.pdb",
        FIXTURES / "mini_ligand.sdf",
        cutoff_angstrom=5.0,
        min_occupancy=0.5,
    )
    # HOH 101 (1.0 occ, near), HOH 102 (1.0 occ, near), HOH 103 (0.5 occ, near)
    # HOH 104 is >5Å away — excluded
    assert len(waters) == 3
    for w in waters:
        assert w.shape == (3,)


def test_occupancy_filter():
    waters = find_binding_waters(
        FIXTURES / "mini_crystal.pdb",
        FIXTURES / "mini_ligand.sdf",
        cutoff_angstrom=5.0,
        min_occupancy=0.6,  # HOH 103 has occupancy=0.5 → excluded
    )
    assert len(waters) == 2


def test_pharmacophore_features_aromatic_rings():
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    assert len(feats["aro_centers"]) >= 1


def test_pharmacophore_features_includes_aromatic_acceptors():
    # pyridine N is aromatic — must be included as acceptor
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    assert len(feats["acceptors"]) >= 1


def test_write_pharma_restr_line_count(tmp_path):
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    out = tmp_path / "test.restr"
    write_pharma_restr(feats, out, n_aro=1, n_acc=1, tolerance=1.0)
    lines = [l for l in out.read_text().strip().splitlines() if l.strip()]
    assert len(lines) == 2


def test_write_pharma_restr_format(tmp_path):
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    out = tmp_path / "test.restr"
    write_pharma_restr(feats, out, n_aro=1, n_acc=1, tolerance=1.0)
    for line in out.read_text().strip().splitlines():
        parts = line.split()
        assert len(parts) == 5
        float(parts[0]); float(parts[1]); float(parts[2]); float(parts[3])
        assert parts[4] in ("Aro", "Acc", "Don", "Hyd")


def test_crystal_ligand_coords_shape():
    coords = crystal_ligand_coords(FIXTURES / "mini_ligand.sdf")
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert len(coords) > 0


def test_find_waters_invalid_sdf_raises(tmp_path):
    bad_sdf = tmp_path / "bad.sdf"
    bad_sdf.write_text("this is not a valid SDF file")
    with pytest.raises(ValueError, match="Could not parse"):
        find_binding_waters(
            FIXTURES / "mini_crystal.pdb",
            bad_sdf,
        )


def test_crystal_ligand_coords_invalid_sdf_raises(tmp_path):
    bad_sdf = tmp_path / "bad.sdf"
    bad_sdf.write_text("this is not a valid SDF file")
    with pytest.raises(ValueError, match="Could not parse"):
        crystal_ligand_coords(bad_sdf)
