"""
OpenMM Docking Suite: GPU-accelerated molecular docking and scoring powered by OpenMM.
"""
from .core import MolecularSystem, DockAtom, Mol2Parser, SDFParser, PDBParser
from .cavity import CavityDefinition, create_cavity_restraint_force
from .scoring import ScoreWeights, create_unified_rdock_force
from .pharmacophore import PharmaPoint, parse_pharma_restr, find_ligand_pharma_features
from .tether import find_tethered_atoms_mcs, TetherConstraint
from .solvent import load_solvent_waters
from .engine import DockingEngine, DockingResult

__version__ = "0.1.0"
__all__ = [
    "MolecularSystem",
    "DockAtom",
    "Mol2Parser",
    "SDFParser",
    "PDBParser",
    "CavityDefinition",
    "create_cavity_restraint_force",
    "ScoreWeights",
    "create_unified_rdock_force",
    "PharmaPoint",
    "parse_pharma_restr",
    "find_ligand_pharma_features",
    "find_tethered_atoms_mcs",
    "TetherConstraint",
    "load_solvent_waters",
    "DockingEngine",
    "DockingResult",
]
