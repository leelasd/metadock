"""
Unified Kinematic Particle Swarm Optimization Engine for openmm-dock.
Unifies:
1. Multi-Driver Macrocycle Inverse Kinematics (IK Loop Closure on 4 Ring Hinges)
2. Exocyclic Ligand Forward Kinematics (FK on Side-Chain Arms)
3. Receptor Side-Chain Kinematics (chi1-chi4 Articulation on Pocket Residues)
4. Multi-Conformer Seeded Particle Swarm Optimization (PSO on Coupled SE(3) x T^k Manifold)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
from rdkit import Chem
from rdkit.Geometry import Point3D
import openmm as mm
from openmm import unit

from .engine import DockingEngine
from .inverse_kinematics import TwoTierMacrocycleEngine
from .receptor_kinematics import ReceptorSideChainKinematics


@dataclass
class UnifiedSwarmParticle:
    """Represents a coupled particle with articulated macrocycle and articulated receptor."""
    particle_id: int
    conformer_seed_id: int       # Multi-conformer template ID
    trans: np.ndarray            # (3,) Ligand translation
    rot_vec: np.ndarray          # (3,) Ligand orientation
    ring_driver_angles: np.ndarray # (4,) 4 Ring IK driver dihedrals
    exo_dihedrals: np.ndarray    # (k_exo,) Ligand side-chain angles
    rec_chi_angles: np.ndarray   # (k_rec,) Receptor chi angles
    
    # Velocities
    v_trans: np.ndarray
    v_rot: np.ndarray
    v_ring: np.ndarray
    v_exo: np.ndarray
    v_rec: np.ndarray
    
    # Personal best
    p_best_trans: np.ndarray
    p_best_rot: np.ndarray
    p_best_ring: np.ndarray
    p_best_exo: np.ndarray
    p_best_rec: np.ndarray
    p_best_score: float
    current_score: float


class UnifiedKinematicPSOEngine:
    """
    Coordinates coupled Particle Swarm Optimization across multi-driver macrocycle ring (IK),
    exocyclic arms (FK), and receptor pocket side chains (chi1-chi4).
    """
    def __init__(
        self,
        receptor_pdb_path: Path | str,
        pocket_center: np.ndarray,
        ligand_mol: Chem.Mol,
        flex_radius: float = 9.0
    ):
        self.lig_mol = Chem.Mol(ligand_mol)
        self.two_tier_lig = TwoTierMacrocycleEngine(ligand_mol)
        self.rec_kin = ReceptorSideChainKinematics(receptor_pdb_path, pocket_center, flex_radius)
        
        # Upgraded to 4 Ring Drivers (Joints 1, 3, 5, 8) for Full Pucker Envelope Exploration
        self.driver_joint_indices = [1, 3, 5, min(8, len(self.two_tier_lig.ik_engine.joints) - 1)]
        self.num_ring_drivers = len(self.driver_joint_indices)
        self.num_exo = len(self.two_tier_lig.exo_joints)
        
        # Flatten all chi joints
        self.all_chi_keys: List[Tuple[str, int, int]] = []
        for r in self.rec_kin.flex_residues:
            for j in r.chi_joints:
                self.all_chi_keys.append((r.res_name, r.res_num, j.chi_idx))
        self.num_rec_chi = len(self.all_chi_keys)
        
        # OpenMM Docking Engine
        self.engine = DockingEngine(receptor_path=receptor_pdb_path)
        self.system, _, self.lig_start, self.lig_n = self.engine._build_system(self.lig_mol)
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        self.context = (
            mm.Context(self.system, self.integrator, self.engine.platform)
            if self.engine.platform
            else mm.Context(self.system, self.integrator)
        )
        
        print(f"[*] Upgraded Unified Kinematic Engine Initialized:")
        print(f"    • Macrocycle: 10 Ring Joints ({self.num_ring_drivers} Active IK Drivers) + {self.num_exo} Exocyclic FK Joints")
        print(f"    • Receptor  : {len(self.rec_kin.flex_residues)} Pocket Residues ({self.num_rec_chi} Chi Joints)")
        print(f"    • Total Coupled Degrees of Freedom: {6 + self.num_ring_drivers + self.num_exo + self.num_rec_chi}")

    @staticmethod
    def _toroidal_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def evaluate_coupled_state(
        self,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        rec_chi: np.ndarray,
        base_coords: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluates coupled energy on OpenMM GPU:
        1. Solves Multi-Driver Macrocycle IK loop closure (4 Drivers)
        2. Rotates Exocyclic Ligand Arms (9 FK Joints)
        3. Rotates Receptor Active-Site Side Chains (31 Chi Joints)
        Returns: (score_kcal, lig_coords, rec_coords)
        """
        # A. Receptor Coordinates
        chi_dict = {key: float(rec_chi[i]) for i, key in enumerate(self.all_chi_keys)}
        rec_coords = self.rec_kin.forward_kinematics_sidechains(chi_dict)
        
        # B. Macrocycle Coordinates (Multi-Driver Ring IK)
        start_c = base_coords if base_coords is not None else self.two_tier_lig.base_coords
        d_dict = {self.driver_joint_indices[i]: float(ring_drivers[i]) for i in range(min(len(ring_drivers), self.num_ring_drivers))}
        
        c_lig, _, _ = self.two_tier_lig.ik_engine.solve_loop_closure(
            start_c, driver_angles=d_dict
        )
        
        # C. Exocyclic Ligand Arms (FK)
        for j_idx in range(min(len(exo_dihedrals), self.num_exo)):
            c_lig = self.two_tier_lig.apply_exocyclic_rotation(c_lig, j_idx, float(exo_dihedrals[j_idx]))
            
        # D. Global Ligand Rigid Body
        if not np.allclose(rot_vec, [0, 0, 0]):
            q = ScipyRotation.from_rotvec(rot_vec).as_matrix()
            center = np.mean(c_lig, axis=0)
            c_lig = (c_lig - center).dot(q.T) + center
        c_lig += trans
        
        # E. Update OpenMM GPU Context
        full_pos = self.engine._full_positions_from_coords(c_lig)
        for idx in range(min(len(rec_coords), self.lig_start)):
            full_pos[idx] = mm.Vec3(rec_coords[idx][0], rec_coords[idx][1], rec_coords[idx][2]) * unit.angstroms
            
        self.context.setPositions(full_pos)
        state = self.context.getState(getEnergy=True)
        score_kcal = float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) * 0.239006)
        return score_kcal, c_lig, rec_coords
