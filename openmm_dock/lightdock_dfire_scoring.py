"""
Wraps LightDock's own installed DFIRE scoring function (a real statistical
potential, not reimplemented -- called as an external library dependency,
same as using scipy/numpy, not copied into this repository) as a
(trans, quat) -> energy callable compatible with
openmm_dock.glowworm_swarm.GlowwormSwarmOptimizer.

This lets the SEARCH ALGORITHM (our GSO re-implementation vs. LightDock's
own) be tested as the only variable, with the SCORING FUNCTION held fixed
and identical on both sides -- decoupling the two questions "is our search
worse?" and "is our scoring worse?" that a same-scoring, same-algorithm,
different-tool comparison otherwise conflates.

Requires `pip install lightdock` (already a dependency of the
protein_protein_1brs comparison work).
"""
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from lightdock.prep.simulation import read_input_structure
from lightdock.scoring.dfire.driver import DFIRE, DFIREAdapter
from lightdock.structure.space import SpacePoints

PoseEnergyFn = Callable[[np.ndarray, np.ndarray], float]


def build_dfire_energy_fn(receptor_pdb: str, ligand_pdb: str) -> Tuple[PoseEnergyFn, np.ndarray]:
    """
    Returns (energy_fn, ligand_local_coords) where energy_fn(trans, quat) ->
    -DFIRE_score (negated so lower is better, matching this codebase's
    energy-minimization convention -- DFIRE's own convention, per its source
    comment, is "higher is better" since the raw GSO luciferin update uses
    the score directly with no sign flip).
    """
    receptor_complex = read_input_structure(receptor_pdb)
    ligand_complex = read_input_structure(ligand_pdb)

    adapter = DFIREAdapter(receptor_complex, ligand_complex)
    receptor_model = adapter.receptor_model
    ligand_model = adapter.ligand_model
    scoring = DFIRE()

    receptor_coords = receptor_model.coordinates[0]
    base_ligand_coords = np.array(ligand_model.coordinates[0].coordinates)
    ligand_local = base_ligand_coords - base_ligand_coords.mean(axis=0)

    def energy_fn(trans: np.ndarray, quat: np.ndarray) -> float:
        rot = ScipyRotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
        world = ligand_local.dot(rot.T) + trans
        score = scoring(receptor_model, receptor_coords, ligand_model, SpacePoints(world))
        return -float(score)

    return energy_fn, ligand_local
