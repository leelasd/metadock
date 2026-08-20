"""
Comprehensive test suite for openmm_dock covering all 6 use cases from rxdock-deepdive-examples.
"""
from pathlib import Path
import pytest
import numpy as np
from rdkit import Chem

from openmm_dock.core import Mol2Parser, SDFParser, PDBParser
from openmm_dock.cavity import CavityDefinition
from openmm_dock.pharmacophore import parse_pharma_restr, find_ligand_pharma_features
from openmm_dock.tether import find_tethered_atoms_mcs
from openmm_dock.engine import DockingEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "test_examples"


def test_core_parsers():
    # 1. Mol2 Parser
    mol2_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    rec_sys = Mol2Parser.parse(mol2_path)
    assert len(rec_sys.atoms) > 0
    assert rec_sys.coordinates.shape == (len(rec_sys.atoms), 3)

    # 2. SDF Parser
    sdf_path = EXAMPLES_DIR / "score" / "ii.sd"
    mols = SDFParser.load_molecules(sdf_path)
    assert len(mols) > 0
    lig_sys = SDFParser.mol_to_system(mols[0])
    assert len(lig_sys.atoms) == mols[0].GetNumAtoms()


def test_cavity_definition():
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    cavity = CavityDefinition.from_prm_file(prm_path)
    assert cavity.radius > 0
    assert len(cavity.center) == 3


def test_use_case_score():
    rec_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    sdf_path = EXAMPLES_DIR / "score" / "ii.sd"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity)

    mols = SDFParser.load_molecules(sdf_path)
    scores = engine.score(mols[0])
    assert "SCORE" in scores
    assert "SCORE.INTER" in scores
    assert "SCORE.RESTR.CAVITY" in scores


def test_score_decomposition_is_additive():
    """
    Each SCORE.INTER.* / SCORE.RESTR.* term must be a real, independently
    computed energy that sums exactly to its parent total -- not a fixed
    fraction of one combined nonbonded blob.
    """
    rec_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    sdf_path = EXAMPLES_DIR / "score" / "ii.sd"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity)

    mols = SDFParser.load_molecules(sdf_path)
    scores = engine.score(mols[0])

    inter_terms = [
        "SCORE.INTER.VDW",
        "SCORE.INTER.POLAR",
        "SCORE.INTER.REPUL",
        "SCORE.INTER.HYD",
        "SCORE.INTER.CONST",
        "SCORE.INTER.ROT",
    ]
    for term in inter_terms:
        assert term in scores
    assert scores["SCORE.INTER"] == pytest.approx(sum(scores[t] for t in inter_terms), abs=1e-6)

    restr_terms = ["SCORE.RESTR.CAVITY", "SCORE.RESTR.PHARMA", "SCORE.RESTR.TETHER"]
    assert scores["SCORE.RESTR"] == pytest.approx(sum(scores[t] for t in restr_terms), abs=1e-6)

    assert scores["SCORE"] == pytest.approx(
        scores["SCORE.INTER"] + scores["SCORE.INTRA"] + scores["SCORE.RESTR"] + scores["SCORE.SYSTEM"],
        abs=1e-6,
    )

    # VDW is a genuinely distinct term from POLAR, not a fixed 5:3 fractional split
    # of one combined nonbonded energy.
    assert scores["SCORE.INTER.VDW"] != pytest.approx(scores["SCORE.INTER.POLAR"] * (5.0 / 3.0))


def test_rotatable_bond_penalty_reflects_topology():
    """SCORE.INTER.ROT must come from a real rotatable-bond count, not nb_e*const."""
    rec_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    sdf_path = EXAMPLES_DIR / "score" / "ii.sd"

    from rdkit.Chem import rdMolDescriptors

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity)
    mol = SDFParser.load_molecules(sdf_path)[0]

    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    scores = engine.score(mol)
    assert scores["SCORE.INTER.ROT"] == pytest.approx(engine.weights.rot * n_rot)


def test_use_case_minimize():
    rec_path = EXAMPLES_DIR / "minimize" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "minimize" / "cavity.prm"
    sdf_path = EXAMPLES_DIR / "minimize" / "ii.sd"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity)

    mols = SDFParser.load_molecules(sdf_path)
    initial_score = engine.score(mols[0])["SCORE"]
    min_res = engine.minimize(mols[0])

    assert min_res.score < initial_score
    assert min_res.mol.GetNumAtoms() == mols[0].GetNumAtoms()


def test_use_case_solvent():
    rec_path = EXAMPLES_DIR / "solvent" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "solvent" / "cavity.prm"
    wat_path = EXAMPLES_DIR / "solvent" / "test_waters.pdb"
    sdf_path = EXAMPLES_DIR / "solvent" / "lig.sdf"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity, waters_pdb_path=wat_path)

    mols = SDFParser.load_molecules(sdf_path)
    min_res = engine.minimize(mols[0])
    assert min_res.score < 0.0


def test_use_case_pharmacophores():
    rec_path = EXAMPLES_DIR / "pharmacophores" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "pharmacophores" / "cavity.prm"
    restr_path = EXAMPLES_DIR / "pharmacophores" / "pharma.restr"
    sdf_path = EXAMPLES_DIR / "pharmacophores" / "xtal-lig.sd"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity, pharma_restr_path=restr_path)

    mols = SDFParser.load_molecules(sdf_path)
    feats = find_ligand_pharma_features(mols[0])
    assert "Aro" in feats
    assert "Acc" in feats

    min_res = engine.minimize(mols[0])
    assert "SCORE" in min_res.scores
    assert "SCORE.RESTR.PHARMA" in min_res.scores
    assert min_res.scores["SCORE.RESTR.PHARMA"] >= 0.0

    dock_res = engine.dock_simulated_annealing(mols[0], n_runs=2, anneal_steps=5, steps_per_temp=50)
    assert len(dock_res) == 2
    assert dock_res[0].mol.GetNumAtoms() == mols[0].GetNumAtoms()


def test_use_case_tethered():
    rec_path = EXAMPLES_DIR / "tethered" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "tethered" / "cavity.prm"
    ref_path = EXAMPLES_DIR / "tethered" / "xtal-lig.sd"
    query_path = EXAMPLES_DIR / "tethered" / "query_ligands.sdf"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity)

    ref_mol = SDFParser.load_molecules(ref_path)[0]
    query_mol = SDFParser.load_molecules(query_path)[0]

    aligned_mol, constraints = find_tethered_atoms_mcs(query_mol, ref_mol)
    assert len(constraints) > 0
    assert aligned_mol is not None

    dock_res = engine.dock_simulated_annealing(
        aligned_mol, tether_constraints=constraints, n_runs=2, anneal_steps=5, steps_per_temp=50
    )
    assert len(dock_res) == 2


def test_use_case_rna_docking():
    rec_path = EXAMPLES_DIR / "rna_docking_example" / "1nem" / "1nem_rdock.mol2"
    prm_path = EXAMPLES_DIR / "rna_docking_example" / "1nem" / "1nem_rdock.prm"
    lig_path = EXAMPLES_DIR / "rna_docking_example" / "1nem" / "1nem_lig.sd"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity)

    lig_mol = SDFParser.load_molecules(lig_path)[0]
    min_res = engine.minimize(lig_mol)
    assert min_res.score < -100.0


def test_monte_carlo_basin_hopping():
    rec_path = EXAMPLES_DIR / "solvent" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "solvent" / "cavity.prm"
    wat_path = EXAMPLES_DIR / "solvent" / "test_waters.pdb"
    lig_path = EXAMPLES_DIR / "solvent" / "lig.sdf"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity, waters_pdb_path=wat_path)

    lig_mol = SDFParser.load_molecules(lig_path)[0]
    mc_res = engine.dock_monte_carlo(lig_mol, n_steps=20, temperature_k=300.0)

    assert mc_res.score < -100.0
    assert mc_res.trajectory is not None
    assert len(mc_res.trajectory) == 21
    assert "MC_FRAME" in mc_res.trajectory[0].GetPropNames()


def test_fluorobenzene_and_aromatic_planarity():
    lig_path = EXAMPLES_DIR / "pharmacophores" / "xtal-lig.sd"
    prm_path = EXAMPLES_DIR / "pharmacophores" / "cavity.prm"
    rec_path = EXAMPLES_DIR / "pharmacophores" / "receptor.mol2"
    restr_path = EXAMPLES_DIR / "pharmacophores" / "pharma.restr"

    cavity = CavityDefinition.from_prm_file(prm_path)
    engine = DockingEngine(receptor_path=rec_path, cavity=cavity, pharma_restr_path=restr_path)

    lig_mol = SDFParser.load_molecules(lig_path)[0]
    res = engine.minimize(lig_mol)
    conf = res.mol.GetConformer()

    # Verify all aromatic rings remain strictly planar (max deviation < 0.05 Å)
    for ring in res.mol.GetRingInfo().AtomRings():
        if all(res.mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            r_coords = np.array([conf.GetAtomPosition(i) for i in ring])
            centered = r_coords - np.mean(r_coords, axis=0)
            _, _, vh = np.linalg.svd(centered)
            normal = vh[2]
            dist_to_plane = np.abs(np.dot(centered, normal))
            assert np.max(dist_to_plane) < 0.05
