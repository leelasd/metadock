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
    ring_driver_angles: np.ndarray # (k_ring,) Ring IK driver dihedrals
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
        
        # Adaptive Ring Drivers: only for macrocyclic rings (>=8 atoms)
        num_joints = len(self.two_tier_lig.ik_engine.joints)
        if num_joints >= 4:
            self.driver_joint_indices = [1, 3, 5, min(8, num_joints - 1)]
        elif num_joints > 0:
            self.driver_joint_indices = [0]
        else:
            self.driver_joint_indices = []
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
        
        print(f"[*] Unified Kinematic Engine Initialized:")
        print(f"    • Ligand    : {self.num_ring_drivers} Ring IK Drivers + {self.num_exo} Rotatable FK Joints")
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
        1. Solves Macrocycle IK loop closure (if macrocyclic)
        2. Rotates Ligand Rotatable Arms (FK Joints)
        3. Rotates Receptor Active-Site Side Chains (Chi Joints)
        Returns: (score_kcal, lig_coords, rec_coords)
        """
        # A. Receptor Coordinates
        chi_dict = {key: float(rec_chi[i]) for i, key in enumerate(self.all_chi_keys)}
        rec_coords = self.rec_kin.forward_kinematics_sidechains(chi_dict)
        
        # B. Macrocycle Coordinates (Ring IK)
        start_c = base_coords if base_coords is not None else self.two_tier_lig.base_coords
        if self.num_ring_drivers > 0:
            d_dict = {self.driver_joint_indices[i]: float(ring_drivers[i]) for i in range(min(len(ring_drivers), self.num_ring_drivers))}
            c_lig, _, _ = self.two_tier_lig.ik_engine.solve_loop_closure(start_c, driver_angles=d_dict)
        else:
            c_lig = start_c.copy()
        
        # C. Exocyclic / Small Molecule Ligand Arms (FK)
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

    def run_unified_pso(
        self,
        n_particles: int = 15,
        n_iterations: int = 20,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494
    ) -> Tuple[Chem.Mol, np.ndarray, float, List[Chem.Mol], List[np.ndarray]]:
        """
        Executes Unified Kinematic PSO across coupled Macrocycle + Receptor space.
        Returns: (best_lig_mol, best_rec_coords, best_score, lig_swarm_frames, rec_movie_frames)
        """
        particles: List[UnifiedSwarmParticle] = []
        g_best_score = 999999.0
        g_best_trans = np.zeros(3)
        g_best_rot = np.zeros(3)
        g_best_ring = np.zeros(self.num_ring_drivers)
        g_best_exo = np.zeros(self.num_exo)
        g_best_rec = np.zeros(self.num_rec_chi)

        # Initialize swarm particles
        for p_id in range(n_particles):
            if p_id == 0:
                t = np.zeros(3)
                r = np.zeros(3)
                ring = np.zeros(self.num_ring_drivers)
                exo = np.zeros(self.num_exo)
                rec = np.zeros(self.num_rec_chi)
            else:
                t = np.random.uniform(-1.5, 1.5, 3)
                r = np.random.uniform(-0.5, 0.5, 3)
                ring = np.random.uniform(-np.pi / 6, np.pi / 6, self.num_ring_drivers)
                exo = np.random.uniform(-np.pi / 4, np.pi / 4, self.num_exo)
                rec = np.random.uniform(-np.pi / 6, np.pi / 6, self.num_rec_chi)

            score, _, _ = self.evaluate_coupled_state(t, r, ring, exo, rec)

            p = UnifiedSwarmParticle(
                particle_id=p_id,
                conformer_seed_id=0,
                trans=t.copy(),
                rot_vec=r.copy(),
                ring_driver_angles=ring.copy(),
                exo_dihedrals=exo.copy(),
                rec_chi_angles=rec.copy(),
                v_trans=np.random.uniform(-0.5, 0.5, 3),
                v_rot=np.random.uniform(-0.3, 0.3, 3),
                v_ring=np.random.uniform(-0.3, 0.3, self.num_ring_drivers),
                v_exo=np.random.uniform(-0.3, 0.3, self.num_exo),
                v_rec=np.random.uniform(-0.3, 0.3, self.num_rec_chi),
                p_best_trans=t.copy(),
                p_best_rot=r.copy(),
                p_best_ring=ring.copy(),
                p_best_exo=exo.copy(),
                p_best_rec=rec.copy(),
                p_best_score=score,
                current_score=score
            )
            particles.append(p)

            if score < g_best_score:
                g_best_score = score
                g_best_trans = t.copy()
                g_best_rot = r.copy()
                g_best_ring = ring.copy()
                g_best_exo = exo.copy()
                g_best_rec = rec.copy()

        lig_frames: List[Chem.Mol] = []
        rec_frames: List[np.ndarray] = []

        # Evolution Loop
        for it in range(n_iterations):
            for p in particles:
                r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)

                # Update Velocities
                p.v_trans = w * p.v_trans + c1 * r1 * (p.p_best_trans - p.trans) + c2 * r2 * (g_best_trans - p.trans)
                p.trans += np.clip(p.v_trans, -1.5, 1.5)

                diff_r_p = self._toroidal_sub(p.p_best_rot, p.rot_vec)
                diff_r_g = self._toroidal_sub(g_best_rot, p.rot_vec)
                p.v_rot = w * p.v_rot + c1 * r1 * diff_r_p + c2 * r2 * diff_r_g
                p.rot_vec = (p.rot_vec + np.clip(p.v_rot, -0.6, 0.6) + np.pi) % (2 * np.pi) - np.pi

                diff_ring_p = self._toroidal_sub(p.p_best_ring, p.ring_driver_angles)
                diff_ring_g = self._toroidal_sub(g_best_ring, p.ring_driver_angles)
                p.v_ring = w * p.v_ring + c1 * r1 * diff_ring_p + c2 * r2 * diff_ring_g
                p.ring_driver_angles = (p.ring_driver_angles + np.clip(p.v_ring, -0.5, 0.5) + np.pi) % (2 * np.pi) - np.pi

                diff_exo_p = self._toroidal_sub(p.p_best_exo, p.exo_dihedrals)
                diff_exo_g = self._toroidal_sub(g_best_exo, p.exo_dihedrals)
                p.v_exo = w * p.v_exo + c1 * r1 * diff_exo_p + c2 * r2 * diff_exo_g
                p.exo_dihedrals = (p.exo_dihedrals + np.clip(p.v_exo, -0.6, 0.6) + np.pi) % (2 * np.pi) - np.pi

                diff_rec_p = self._toroidal_sub(p.p_best_rec, p.rec_chi_angles)
                diff_rec_g = self._toroidal_sub(g_best_rec, p.rec_chi_angles)
                p.v_rec = w * p.v_rec + c1 * r1 * diff_rec_p + c2 * r2 * diff_rec_g
                p.rec_chi_angles = (p.rec_chi_angles + np.clip(p.v_rec, -0.5, 0.5) + np.pi) % (2 * np.pi) - np.pi

                # Evaluate Coupled Energy
                score, c_lig, c_rec = self.evaluate_coupled_state(
                    p.trans, p.rot_vec, p.ring_driver_angles, p.exo_dihedrals, p.rec_chi_angles
                )
                p.current_score = score

                if score < p.p_best_score:
                    p.p_best_score = score
                    p.p_best_trans = p.trans.copy()
                    p.p_best_rot = p.rot_vec.copy()
                    p.p_best_ring = p.ring_driver_angles.copy()
                    p.p_best_exo = p.exo_dihedrals.copy()
                    p.p_best_rec = p.rec_chi_angles.copy()

                if score < g_best_score:
                    g_best_score = score
                    g_best_trans = p.trans.copy()
                    g_best_rot = p.rot_vec.copy()
                    g_best_ring = p.ring_driver_angles.copy()
                    g_best_exo = p.exo_dihedrals.copy()
                    g_best_rec = p.rec_chi_angles.copy()

                # Save Synchronized Movie Frames
                mol_f = Chem.Mol(self.lig_mol)
                conf_f = mol_f.GetConformer()
                for i in range(mol_f.GetNumAtoms()):
                    conf_f.SetAtomPosition(i, Point3D(float(c_lig[i][0]), float(c_lig[i][1]), float(c_lig[i][2])))
                mol_f.SetProp("ITERATION", str(it + 1))
                mol_f.SetProp("PARTICLE_ID", str(p.particle_id + 1))
                mol_f.SetProp("COUPLED_SCORE_KCAL", f"{score:.2f}")
                mol_f.SetProp("SWARM_BEST_SCORE", f"{g_best_score:.2f}")
                lig_frames.append(mol_f)
                rec_frames.append(c_rec)

        # Build final best complex
        _, best_lig_coords, best_rec_coords = self.evaluate_coupled_state(
            g_best_trans, g_best_rot, g_best_ring, g_best_exo, g_best_rec
        )
        best_mol = Chem.Mol(self.lig_mol)
        conf_b = best_mol.GetConformer()
        for i in range(best_mol.GetNumAtoms()):
            conf_b.SetAtomPosition(i, Point3D(float(best_lig_coords[i][0]), float(best_lig_coords[i][1]), float(best_lig_coords[i][2])))
        best_mol.SetProp("FINAL_SCORE_KCAL", f"{g_best_score:.3f}")

        return best_mol, best_rec_coords, g_best_score, lig_frames, rec_frames
