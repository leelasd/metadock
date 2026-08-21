"""
Unit tests for Collaborative Multi-Swarm Kinematic Metadynamics Engine in openmm_dock.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem
import pytest

from openmm_dock.collaborative_kinematic_metadynamics import (
    SharedMetadynamicsArchive,
    CollaborativeKinematicMetaDEngine,
    CollaborativeMetaDParams,
    SharedBasin,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "test_examples"


def test_shared_metadynamics_archive():
    archive = SharedMetadynamicsArchive(
        initial_height_w0=10.0,
        gaussian_sigma=1.0,
        bias_factor_gamma=5.0,
        min_basin_rmsd=1.0
    )
    
    dummy_coords1 = np.zeros((10, 3))
    dummy_coords2 = np.ones((10, 3)) * 3.0
    
    # 1. Register basin from Island 1
    basin1 = archive.register_basin(
        island_id=1,
        iteration=1,
        trans=np.zeros(3),
        rot_vec=np.zeros(3),
        ring_drivers=np.zeros(4),
        exo_dihedrals=np.zeros(9),
        coords=dummy_coords1,
        phys_score=-150.0
    )
    assert basin1 is not None
    assert len(archive.basins) == 1
    assert basin1.height_w == 10.0
    
    # Bias at basin1 center should be ~10.0 kcal/mol
    bias_at_center = archive.compute_bias(dummy_coords1)
    assert bias_at_center == pytest.approx(10.0, abs=1e-3)
    
    # Bias far away should be near 0
    bias_far = archive.compute_bias(dummy_coords2)
    assert bias_far < 0.1
    
    # 2. Register distinct basin from Island 2
    basin2 = archive.register_basin(
        island_id=2,
        iteration=2,
        trans=np.ones(3),
        rot_vec=np.ones(3),
        ring_drivers=np.ones(4),
        exo_dihedrals=np.ones(9),
        coords=dummy_coords2,
        phys_score=-160.0
    )
    assert basin2 is not None
    assert len(archive.basins) == 2


def test_collaborative_kinematic_engine_run():
    rec_path = EXAMPLES_DIR / "macrocycle_6z6a" / "receptor.pdb"
    xtal_path = EXAMPLES_DIR / "macrocycle_6z6a" / "q9e_crystal_pose.sdf"
    
    xtal_mol = Chem.SDMolSupplier(str(xtal_path), removeHs=False)[0]
    conf_x = xtal_mol.GetConformer()
    coords_x = np.array([conf_x.GetAtomPosition(i) for i in range(xtal_mol.GetNumAtoms())])
    pocket_center = np.mean(coords_x, axis=0)
    
    engine = CollaborativeKinematicMetaDEngine(
        receptor_pdb_path=rec_path,
        pocket_center=pocket_center,
        ligand_mol=xtal_mol,
        num_conformer_seeds=2
    )
    
    assert engine.num_ring_drivers == 4
    assert engine.num_exo == 9
    assert engine.num_dofs == 19
    
    # Test evaluation
    guide_s, phys_s, coords = engine.evaluate_kinematics(
        trans=np.zeros(3),
        rot_vec=np.zeros(3),
        ring_drivers=np.zeros(4),
        exo_dihedrals=np.zeros(9),
        conformer_seed_id=0
    )
    assert isinstance(guide_s, float)
    assert isinstance(phys_s, float)
    assert coords.shape == (xtal_mol.GetNumAtoms(), 3)

    
    # Run short 2-iteration collaborative test
    params = CollaborativeMetaDParams(
        num_islands=2,
        particles_per_island=4,
        n_iterations=2,
        search_radius=3.0,
        basin_deposit_interval=1
    )
    
    best_mol, best_score, trajectory_mols, summary = engine.run_collaborative_docking(
        params=params,
        reference_xtal_mol=xtal_mol
    )
    
    assert best_mol is not None
    assert len(trajectory_mols) == 2 * 4 * 2 # 2 islands * 4 particles * 2 iterations
    assert "PHYS_SCORE_KCAL" in best_mol.GetPropNames()
    assert isinstance(summary["best_phys_score_kcal"], float)
    assert summary["best_phys_score_kcal"] < 5000.0
