"""
End-to-end correctness test for scoring.create_grid_search_force (Phase 2):
builds a real OpenMM System with the grid-based nonbonded force and checks
its total energy against the exact pairwise create_combined_search_force on
a real docked pose (the "score" test system's crystal ligand).

Scoped to a small cavity radius and only the VDW element types actually
present in the test ligand to keep grid computation (the one-time,
amortized-over-search cost) affordable for routine test runs.
"""
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from openmm_dock.core import Mol2Parser, SDFParser
from openmm_dock.cavity import CavityDefinition
from openmm_dock.gridding import GridBox, compute_potential_grids
from openmm_dock.scoring import ScoreWeights, create_combined_search_force, create_grid_search_force

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "test_examples"


def _build_system(receptor, lig_mol, lig_sys, nb_force):
    system = mm.System()
    all_atoms = list(receptor.atoms) + list(lig_sys.atoms)
    lig_start = len(receptor.atoms)
    for _ in receptor.atoms:
        system.addParticle(0.0)
    for _ in lig_sys.atoms:
        system.addParticle(12.0)
    for i, a in enumerate(all_atoms):
        is_lig = 1.0 if i >= lig_start else 0.0
        is_hyd = 1.0 if a.element.upper() in ["C", "CL", "BR", "I", "F"] and not a.is_polar else 0.0
        nb_force.addParticle([
            a.charge, a.sigma, a.epsilon,
            1.0 if a.is_donor else 0.0, 1.0 if a.is_acceptor else 0.0, is_hyd, is_lig,
        ])
    excluded = set()

    def add_excl(i1, i2):
        pair = (min(i1, i2), max(i1, i2))
        if pair not in excluded:
            excluded.add(pair)
            nb_force.addExclusion(pair[0], pair[1])

    for b in lig_sys.bonds:
        add_excl(lig_start + b.atom1, lig_start + b.atom2)
    for atom in lig_mol.GetAtoms():
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                add_excl(lig_start + nbrs[i], lig_start + nbrs[j])

    for f in nb_force.forces:
        system.addForce(f)
    return system


def _energy(system, receptor, lig_mol):
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = mm.Context(system, integrator)
    rec_coords = receptor.coordinates * 0.1
    lig_coords = np.array([lig_mol.GetConformer().GetAtomPosition(i) for i in range(lig_mol.GetNumAtoms())]) * 0.1
    context.setPositions(np.vstack([rec_coords, lig_coords]) * unit.nanometers)
    e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    del context, integrator
    return e


def test_grid_search_force_matches_exact_pairwise_on_real_docked_pose():
    rec_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    lig_path = EXAMPLES_DIR / "score" / "xtal-lig.sd"

    receptor = Mol2Parser.parse(rec_path)
    cavity_full = CavityDefinition.from_prm_file(prm_path)
    # Small radius (vs. cavity.prm's real ~16A) to keep this test's one-time
    # grid-computation cost affordable for routine test runs.
    cavity = CavityDefinition(
        center=cavity_full.center, radius=8.0,
        min_coords=cavity_full.center - 8.0, max_coords=cavity_full.center + 8.0,
        name="test_tight",
    )
    lig_mol = SDFParser.load_molecules(lig_path)[0]
    lig_sys = SDFParser.mol_to_system(lig_mol)

    # Only the VDW element types actually present in this ligand (C, F, N, O).
    probe_types = ["C", "F", "N", "O"]
    box = GridBox.from_cavity(cavity, ligand_margin_ang=6.0, spacing_ang=0.375)
    grids = compute_potential_grids(receptor, box, vdw_probe_types=probe_types)

    weights = ScoreWeights()
    exact_system = _build_system(receptor, lig_mol, lig_sys, create_combined_search_force(weights))
    grid_system = _build_system(
        receptor, lig_mol, lig_sys,
        create_grid_search_force(grids, box, grids, box, weights, vdw_probe_types=probe_types),
    )

    e_exact = _energy(exact_system, receptor, lig_mol)
    e_grid = _energy(grid_system, receptor, lig_mol)

    assert e_exact < 0  # sanity: this is a real, favorable docked pose
    rel_error = abs(e_grid - e_exact) / abs(e_exact)
    assert rel_error < 0.15, f"grid energy {e_grid:.2f} vs exact {e_exact:.2f} ({100*rel_error:.1f}% off)"
