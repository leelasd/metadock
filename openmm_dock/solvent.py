"""
Flexible active-site solvent / explicit water handling in OpenMM docking.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import openmm as mm
from .core import MolecularSystem, DockAtom, PDBParser
from .scoring import GROUP_SOLVENT


@dataclass
class SolventWater:
    oxygen_idx: int
    hydrogen_indices: List[int]
    crystal_coord: np.ndarray  # (3,) in Angstroms
    radius_tether: float = 0.8  # Max translation sphere radius in Angstroms


def load_solvent_waters(water_pdb_path: Path | str) -> MolecularSystem:
    """Loads active-site water molecules from PDB."""
    return PDBParser.parse(water_pdb_path)


def create_solvent_tether_force(
    water_indices: List[int],
    initial_coords: np.ndarray,
    radius_tether: float = 0.8,
    k_tether: float = 1000.0,
) -> mm.CustomExternalForce:
    """
    Creates a flat-bottom harmonic tether on solvent oxygen atoms,
    allowing flexible rotation and small translation near crystallographic positions.
    """
    expr = (
        "0.5 * k_solv * step(r_dist - r_tol) * (r_dist - r_tol)^2;"
        "r_dist = sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    force = mm.CustomExternalForce(expr)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addGlobalParameter("k_solv", k_tether)
    force.addGlobalParameter("r_tol", radius_tether * 0.1)  # Å to nm
    force.setForceGroup(GROUP_SOLVENT)
    force.setName("SolventTetherForce")

    for local_i, sys_idx in enumerate(water_indices):
        x0_nm = initial_coords[local_i][0] * 0.1
        y0_nm = initial_coords[local_i][1] * 0.1
        z0_nm = initial_coords[local_i][2] * 0.1
        force.addParticle(sys_idx, [x0_nm, y0_nm, z0_nm])

    return force
