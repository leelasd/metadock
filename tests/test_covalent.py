"""
Unit tests for Covalent Docking module in openmm-dock.
Verifies automated warhead perception, nucleophile residue resolution,
and GPU-accelerated covalent bond/angle restraint formulation.
"""
from pathlib import Path
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openmm_dock import DockingEngine, CavityDefinition
from openmm_dock.covalent import (
    detect_ligand_warhead,
    find_receptor_nucleophile,
    create_covalent_restraint,
    CovalentRestraint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "test_examples"


def test_warhead_detection():
    # 1. Acrylamide (Michael acceptor)
    mol_acryl = Chem.MolFromSmiles("CC(=O)Nc1ccccc1NC(=O)C=C")
    res_acryl = detect_ligand_warhead(mol_acryl)
    assert res_acryl is not None
    assert "Acrylamide" in res_acryl[0]
    assert res_acryl[2] == "CYS"
    assert res_acryl[3] == 0.182

    # 2. Haloacetamide
    mol_halo = Chem.MolFromSmiles("CC(=O)Nc1ccccc1NC(=O)CCl")
    res_halo = detect_ligand_warhead(mol_halo)
    assert res_halo is not None
    assert "Haloacetamide" in res_halo[0]
    assert res_halo[2] == "CYS"

    # 3. Aldehyde
    mol_ald = Chem.MolFromSmiles("c1ccccc1C=O")
    res_ald = detect_ligand_warhead(mol_ald)
    assert res_ald is not None
    assert "Aldehyde" in res_ald[0]

    # 4. Nitrile
    mol_nitrile = Chem.MolFromSmiles("c1ccccc1C#N")
    res_nitrile = detect_ligand_warhead(mol_nitrile)
    assert res_nitrile is not None
    assert "Nitrile" in res_nitrile[0]

    # 5. Boronic Acid
    mol_boron = Chem.MolFromSmiles("c1ccccc1B(O)O")
    res_boron = detect_ligand_warhead(mol_boron)
    assert res_boron is not None
    assert "Boronic Acid" in res_boron[0]
    assert res_boron[2] == "SER"


def test_receptor_nucleophile_resolution():
    rec_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity)

    # Resolve CYS
    nucl_idx, anchor_idx, res_name = find_receptor_nucleophile(engine.receptor, "CYS")
    assert res_name == "CYS"
    assert nucl_idx is not None
    assert anchor_idx is not None


def test_covalent_minimization_and_bond_formation():
    rec_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity, covalent_res="CYS")

    smi = "CC(=O)Nc1ccccc1NC(=O)C=C"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    cov_restr = create_covalent_restraint(engine.receptor, mol, "CYS")
    res = engine.minimize(mol)

    conf_res = res.mol.GetConformer()
    nucl_pos = engine.receptor.atoms[cov_restr.rec_nucleophile_idx].coord
    el_pos_min = np.array(conf_res.GetAtomPosition(cov_restr.lig_electrophile_idx))
    dist_angstrom = float(np.linalg.norm(nucl_pos - el_pos_min))

    # Verify covalent bond formed within 0.05 Å of target equilibrium
    assert abs(dist_angstrom - cov_restr.r0_nm * 10) < 0.05
