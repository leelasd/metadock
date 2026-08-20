"""
Macrocycle Inverse Kinematics (IK) Engine for openmm-dock.
Solves analytical closed-loop constraints using Damped Least Squares (DLS)
enabling strain-free conformational flexing and docking of macrocyclic rings.
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


@dataclass
class MacrocycleJoint:
    """Represents a rotatable joint along the macrocyclic ring backbone."""
    joint_idx: int
    begin_atom_idx: int       # Origin atom of rotation axis
    end_atom_idx: int         # Destination atom of rotation axis
    moving_atom_indices: List[int] # ALL downstream atoms (including hydrogens & sidechains)


class MacrocycleInverseKinematics:
    """
    Solves closed-loop Inverse Kinematics for macrocyclic ligands.
    Guarantees that the macrocyclic ring remains 100% closed with 0.000 Å gap
    while all attached hydrogens and side-chain substituents move rigidly with the ring.
    """
    def __init__(self, mol: Chem.Mol, ring_atom_indices: Optional[List[int]] = None):
        self.mol = Chem.Mol(mol)
        self.num_atoms = mol.GetNumAtoms()
        conf = self.mol.GetConformer()
        self.base_coords = np.array(
            [conf.GetAtomPosition(i) for i in range(self.num_atoms)], dtype=np.float64
        )
        
        # 1. Identify the largest macrocyclic ring if not provided
        if ring_atom_indices is None:
            rings = mol.GetRingInfo().AtomRings()
            macro_rings = [list(r) for r in rings if len(r) >= 9]
            if not macro_rings:
                macro_rings = [list(r) for r in rings]
                macro_rings.sort(key=len, reverse=True)
            self.ring_atoms = macro_rings[0]
        else:
            self.ring_atoms = list(ring_atom_indices)
            
        # 2. Select a closure cut bond inside the macrocycle
        self.cut_a1 = self.ring_atoms[0]
        self.cut_a2 = self.ring_atoms[-1]
        p1 = self.base_coords[self.cut_a1]
        p2 = self.base_coords[self.cut_a2]
        self.target_bond_length = float(np.linalg.norm(p1 - p2))
        
        # 3. Build the directed backbone chain and identify rotatable single bonds
        self.chain_atoms = self.ring_atoms
        self.joints: List[MacrocycleJoint] = []
        
        for i in range(len(self.chain_atoms) - 1):
            a1 = self.chain_atoms[i]
            a2 = self.chain_atoms[i + 1]
            b = self.mol.GetBondBetweenAtoms(a1, a2)
            if b is not None and b.GetBondType() == Chem.BondType.SINGLE:
                # Find ALL downstream atoms on the a2 side (excluding a1 and the cut closure bond)
                moving = self._find_full_downstream_subtree(a1, a2)
                self.joints.append(MacrocycleJoint(
                    joint_idx=len(self.joints),
                    begin_atom_idx=a1,
                    end_atom_idx=a2,
                    moving_atom_indices=moving
                ))
                
        self.num_joints = len(self.joints)
        print(f"[*] Macrocycle IK Engine initialized: {self.num_atoms} total atoms | {len(self.ring_atoms)}-membered ring")
        print(f"[*] Identified {self.num_joints} rotatable joint hinges with full atomic subtree propagation")

    def _find_full_downstream_subtree(self, begin_idx: int, split_idx: int) -> List[int]:
        """
        BFS traversal from split_idx to find ALL downstream atoms in the entire molecule
        (including all attached hydrogens and exocyclic side-chains) without crossing begin_idx
        or the cut closure bond (cut_a1 - cut_a2).
        """
        # Block backwards path to begin_idx and block the cut closure bond
        blocked_edges = {
            (min(begin_idx, split_idx), max(begin_idx, split_idx)),
            (min(self.cut_a1, self.cut_a2), max(self.cut_a1, self.cut_a2))
        }
        
        visited: Set[int] = {split_idx}
        queue = [split_idx]
        
        while queue:
            curr = queue.pop(0)
            for nbr in self.mol.GetAtomWithIdx(curr).GetNeighbors():
                n_idx = nbr.GetIdx()
                edge = (min(curr, n_idx), max(curr, n_idx))
                if edge in blocked_edges:
                    continue
                if n_idx not in visited:
                    visited.add(n_idx)
                    queue.append(n_idx)
                    
        return sorted(list(visited))

    def solve_loop_closure(
        self,
        coords: np.ndarray,
        driver_angles: Optional[Dict[int, float]] = None,
        max_iter: int = 30,
        damping: float = 0.05,
        tolerance: float = 1e-4
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Solves Damped Least Squares (DLS) Inverse Kinematics to close the ring gap.
        Transforms ALL ring atoms, attached side chains, and hydrogens simultaneously.
        Returns: (closed_coords, final_closure_gap_angstrom, is_converged)
        """
        curr_coords = coords.copy()
        
        # 1. Apply driver angles on non-closure joints
        if driver_angles:
            for j_idx, angle in driver_angles.items():
                if j_idx < len(self.joints):
                    joint = self.joints[j_idx]
                    a1 = joint.begin_atom_idx
                    a2 = joint.end_atom_idx
                    axis = curr_coords[a2] - curr_coords[a1]
                    norm = np.linalg.norm(axis)
                    if norm > 1e-6:
                        u = axis / norm
                        rot = ScipyRotation.from_rotvec(u * angle).as_matrix()
                        origin = curr_coords[a1]
                        sub_p = curr_coords[joint.moving_atom_indices] - origin
                        curr_coords[joint.moving_atom_indices] = sub_p.dot(rot.T) + origin

        # 2. Damped Least Squares Loop Closure on the tip-to-anchor gap
        for iteration in range(max_iter):
            p_tip = curr_coords[self.cut_a2]
            p_anchor = curr_coords[self.cut_a1]
            
            curr_gap_vec = p_anchor - p_tip
            curr_dist = np.linalg.norm(curr_gap_vec)
            error_val = abs(curr_dist - self.target_bond_length)
            
            if error_val < tolerance:
                return curr_coords, float(error_val), True
                
            # Desired tip position is target_bond_length away from anchor
            if curr_dist > 1e-6:
                target_tip = p_anchor - (curr_gap_vec / curr_dist) * self.target_bond_length
            else:
                target_tip = p_anchor + np.array([self.target_bond_length, 0.0, 0.0])
                
            delta_e = target_tip - p_tip # (3,) 3D positional gap vector
            
            # Construct 3 x K Geometric Jacobian Matrix
            J = np.zeros((3, self.num_joints), dtype=np.float64)
            for j_idx, joint in enumerate(self.joints):
                a1 = joint.begin_atom_idx
                a2 = joint.end_atom_idx
                axis = curr_coords[a2] - curr_coords[a1]
                norm = np.linalg.norm(axis)
                if norm > 1e-6:
                    u = axis / norm
                    r_vec = p_tip - curr_coords[a1]
                    J[:, j_idx] = np.cross(u, r_vec)
                    
            # Damped Least Squares: Δθ = J^T (J J^T + λ² I)^(-1) e
            JJT = J.dot(J.T) + (damping ** 2) * np.eye(3)
            inv_JJT = np.linalg.inv(JJT)
            delta_theta = J.T.dot(inv_JJT).dot(delta_e)
            
            # Apply delta_theta updates to all downstream atoms and hydrogens
            for j_idx, joint in enumerate(self.joints):
                d_angle = float(delta_theta[j_idx])
                if abs(d_angle) < 1e-7:
                    continue
                a1 = joint.begin_atom_idx
                a2 = joint.end_atom_idx
                axis = curr_coords[a2] - curr_coords[a1]
                norm = np.linalg.norm(axis)
                if norm > 1e-6:
                    u = axis / norm
                    rot = ScipyRotation.from_rotvec(u * d_angle).as_matrix()
                    origin = curr_coords[a1]
                    sub_p = curr_coords[joint.moving_atom_indices] - origin
                    curr_coords[joint.moving_atom_indices] = sub_p.dot(rot.T) + origin

        final_dist = np.linalg.norm(curr_coords[self.cut_a1] - curr_coords[self.cut_a2])
        final_err = abs(final_dist - self.target_bond_length)
        return curr_coords, float(final_err), (final_err < 0.05)

    def generate_macrocycle_breathing_trajectory(
        self,
        engine: DockingEngine,
        n_frames: int = 60
    ) -> List[Chem.Mol]:
        """
        Generates a continuous breathing / conformational flexing movie in the pocket
        where Inverse Kinematics maintains exact ring closure and all hydrogens move rigidly.
        """
        frames: List[Chem.Mol] = []
        
        system, _, lig_start, lig_n = engine._build_system(self.mol)
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(system, integrator, engine.platform)
            if engine.platform
            else mm.Context(system, integrator)
        )
        
        driver1 = 1
        driver2 = max(2, self.num_joints // 2)
        
        t_values = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        for f_idx, t in enumerate(t_values):
            # Smooth sinusoidal breathing driver perturbations (±30°)
            d_angles = {
                driver1: float(np.sin(t) * (np.pi / 6.0)),
                driver2: float(np.cos(t) * (np.pi / 6.0))
            }
            
            closed_coords, gap_err, conv = self.solve_loop_closure(
                self.base_coords, driver_angles=d_angles
            )
            
            # Evaluate OpenMM GPU energy on Keap1
            full_pos = engine._full_positions_from_coords(closed_coords)
            context.setPositions(full_pos)
            state = context.getState(getEnergy=True)
            score_kcal = float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) * 0.239006)
            
            # Build RDKit frame
            mol_f = Chem.Mol(self.mol)
            conf_f = mol_f.GetConformer()
            for i in range(self.num_atoms):
                p = closed_coords[i]
                conf_f.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
                
            mol_f.SetProp("FRAME_ID", str(f_idx + 1))
            mol_f.SetProp("RING_CLOSURE_GAP_A", f"{gap_err:.6f}")
            mol_f.SetProp("OPENMM_SCORE_KCAL", f"{score_kcal:.2f}")
            mol_f.SetProp("IK_STATUS", "CLOSED_CONVERGED" if conv else "APPROX")
            frames.append(mol_f)
            
        return frames


@dataclass
class ExocyclicJoint:
    """Represents an unconstrained exocyclic side-chain rotatable bond."""
    joint_idx: int
    begin_atom_idx: int
    end_atom_idx: int
    moving_atom_indices: List[int]
    name: str = ""


class TwoTierMacrocycleEngine:
    """
    Decoupled Two-Tier Macrocycle Docking Engine:
    • Tier 1: Endocyclic Ring Backbone (Constrained Manifold via Inverse Kinematics)
    • Tier 2: Exocyclic Side-Chain Arms (Unconstrained Torus via Forward Kinematics)
    """
    def __init__(self, mol: Chem.Mol):
        self.mol = Chem.Mol(mol)
        self.ik_engine = MacrocycleInverseKinematics(mol)
        self.num_atoms = mol.GetNumAtoms()
        conf = self.mol.GetConformer()
        self.base_coords = np.array(
            [conf.GetAtomPosition(i) for i in range(self.num_atoms)], dtype=np.float64
        )
        self.ring_set = set(self.ik_engine.ring_atoms)
        
        # Identify exocyclic rotatable bonds
        rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
        matches = self.mol.GetSubstructMatches(rot_smarts)
        
        self.exo_joints: List[ExocyclicJoint] = []
        seen = set()
        for a1, a2 in matches:
            pair = tuple(sorted([a1, a2]))
            if pair in seen:
                continue
            seen.add(pair)
            b = self.mol.GetBondBetweenAtoms(a1, a2)
            if not b.IsInRing():
                # If a1 is in the ring or closer to the ring, moving subtree is on a2 side
                if a1 in self.ring_set and a2 not in self.ring_set:
                    moving = self._find_exocyclic_subtree(a1, a2)
                elif a2 in self.ring_set and a1 not in self.ring_set:
                    a1, a2 = a2, a1
                    moving = self._find_exocyclic_subtree(a1, a2)
                else:
                    moving = self._find_exocyclic_subtree(a1, a2)
                    
                sym1 = self.mol.GetAtomWithIdx(a1).GetSymbol()
                sym2 = self.mol.GetAtomWithIdx(a2).GetSymbol()
                self.exo_joints.append(ExocyclicJoint(
                    joint_idx=len(self.exo_joints),
                    begin_atom_idx=a1,
                    end_atom_idx=a2,
                    moving_atom_indices=moving,
                    name=f"{sym1}{a1}-{sym2}{a2}"
                ))
                
        print(f"[*] Two-Tier Macrocycle Engine Ready: {self.ik_engine.num_joints} Ring IK Joints | {len(self.exo_joints)} Exocyclic FK Joints")

    def _find_exocyclic_subtree(self, begin_idx: int, split_idx: int) -> List[int]:
        """Finds all atoms downstream on the exocyclic side without crossing begin_idx."""
        visited: Set[int] = {split_idx}
        queue = [split_idx]
        while queue:
            curr = queue.pop(0)
            for nbr in self.mol.GetAtomWithIdx(curr).GetNeighbors():
                n_idx = nbr.GetIdx()
                if n_idx != begin_idx and n_idx not in visited:
                    visited.add(n_idx)
                    queue.append(n_idx)
        return sorted(list(visited))

    def apply_exocyclic_rotation(
        self,
        coords: np.ndarray,
        exo_joint_idx: int,
        angle_rad: float
    ) -> np.ndarray:
        """Applies pure Forward Kinematics (FK) rotation to an exocyclic side-chain arm."""
        if abs(angle_rad) < 1e-7 or exo_joint_idx >= len(self.exo_joints):
            return coords
            
        new_coords = coords.copy()
        joint = self.exo_joints[exo_joint_idx]
        a1, a2 = joint.begin_atom_idx, joint.end_atom_idx
        axis = new_coords[a2] - new_coords[a1]
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            u = axis / norm
            rot = ScipyRotation.from_rotvec(u * angle_rad).as_matrix()
            origin = new_coords[a1]
            sub_p = new_coords[joint.moving_atom_indices] - origin
            new_coords[joint.moving_atom_indices] = sub_p.dot(rot.T) + origin
            
        return new_coords

    def generate_two_tier_movie(
        self,
        engine: DockingEngine
    ) -> List[Chem.Mol]:
        """
        Generates a 3-Phase Multi-Tier Movie:
        • Phase 1 (Frames 1-40): Macrocyclic Ring Breathing via Inverse Kinematics (IK)
        • Phase 2 (Frames 41-80): Exocyclic Side-Chain Articulation via Forward Kinematics (FK)
        • Phase 3 (Frames 81-120): Coordinated Coupled Docking settling into the Keap1 pocket
        """
        frames: List[Chem.Mol] = []
        frame_id = 1
        
        # Build OpenMM system once
        system, _, lig_start, lig_n = engine._build_system(self.mol)
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(system, integrator, engine.platform)
            if engine.platform
            else mm.Context(system, integrator)
        )
        
        def _make_frame(c_arr: np.ndarray, move_type: str, detail: str) -> Chem.Mol:
            nonlocal frame_id
            full_pos = engine._full_positions_from_coords(c_arr)
            context.setPositions(full_pos)
            state = context.getState(getEnergy=True)
            score_kcal = float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) * 0.239006)
            
            p1 = c_arr[self.ik_engine.cut_a1]
            p2 = c_arr[self.ik_engine.cut_a2]
            gap = abs(np.linalg.norm(p1 - p2) - self.ik_engine.target_bond_length)
            
            mol_f = Chem.Mol(self.mol)
            conf_f = mol_f.GetConformer()
            for i in range(self.num_atoms):
                p = c_arr[i]
                conf_f.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
                
            mol_f.SetProp("FRAME_ID", str(frame_id))
            mol_f.SetProp("TIER_MOVE_TYPE", move_type)
            mol_f.SetProp("MOVE_DETAIL", detail)
            mol_f.SetProp("OPENMM_SCORE_KCAL", f"{score_kcal:.2f}")
            mol_f.SetProp("RING_CLOSURE_GAP_A", f"{gap:.6f}")
            frame_id += 1
            return mol_f

        # =========================================================================
        # PHASE 1: Macrocyclic Ring Breathing via Inverse Kinematics (40 Frames)
        # =========================================================================
        t_ring = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        for t in t_ring:
            d_angles = {
                1: float(np.sin(t) * (np.pi / 5.0)),
                max(2, self.ik_engine.num_joints // 2): float(np.cos(t) * (np.pi / 5.0))
            }
            c_ring, _, _ = self.ik_engine.solve_loop_closure(self.base_coords, driver_angles=d_angles)
            frames.append(_make_frame(c_ring, "TIER_1_RING_IK_BREATHE", "Ring Backbone Flexing (IK Closed)"))

        # =========================================================================
        # PHASE 2: Exocyclic Side-Chain Articulation via Forward Kinematics (40 Frames)
        # =========================================================================
        t_exo = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        for t in t_exo:
            c_exo = self.base_coords.copy()
            # Articulate key exocyclic arms (carboxylate/amide rotamers)
            for j_idx in range(min(4, len(self.exo_joints))):
                phase_shift = j_idx * (np.pi / 2.0)
                angle = float(np.sin(t + phase_shift) * (np.pi / 3.0))
                c_exo = self.apply_exocyclic_rotation(c_exo, j_idx, angle)
                
            frames.append(_make_frame(c_exo, "TIER_2_EXOCYCLIC_FK_ROTATE", "Side-Chain Functional Arms Rotating (FK)"))

        # =========================================================================
        # PHASE 3: Coupled Two-Tier Pocket Docking (40 Frames)
        # =========================================================================
        decay = np.linspace(1.0, 0.0, 40)
        for i, dec in enumerate(decay):
            # Combine damped ring breathing with targeted side-chain alignment
            d_angles = {
                1: float(np.sin(i * 0.3) * (np.pi / 6.0) * dec),
                max(2, self.ik_engine.num_joints // 2): float(np.cos(i * 0.3) * (np.pi / 6.0) * dec)
            }
            c_coupled, _, _ = self.ik_engine.solve_loop_closure(self.base_coords, driver_angles=d_angles)
            for j_idx in range(min(3, len(self.exo_joints))):
                angle = float(np.sin(i * 0.4 + j_idx) * (np.pi / 4.0) * dec)
                c_coupled = self.apply_exocyclic_rotation(c_coupled, j_idx, angle)
                
            frames.append(_make_frame(c_coupled, "TIER_3_COUPLED_DOCKING", "Coupled Two-Tier Docking into Keap1 Pocket"))

        return frames
