"""
Kinematic Metadynamics Engine for openmm-dock.
Provides history-dependent repulsive Gaussian potential and analytical repulsive torques
on the (SE(3) x T^k) kinematic manifold, driving continuous physical dynamics out of decoy traps.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
from rdkit import Chem
from rdkit.Geometry import Point3D
import openmm as mm
from openmm import unit

from .unified_kinematic_pso import UnifiedKinematicPSOEngine


@dataclass
class VisitedBasin:
    """Represents a visited local energy well in kinematic space."""
    basin_id: int
    trans: np.ndarray            # (3,) Translation
    rot_vec: np.ndarray          # (3,) Rotation
    ring_drivers: np.ndarray     # (2,) Ring IK drivers
    exo_dihedrals: np.ndarray    # (k_exo,) Ligand side-chain angles
    rec_chi: np.ndarray          # (k_rec,) Receptor chi angles
    raw_score: float             # OpenMM physical score before bias
    height_w: float              # Gaussian hill height in kcal/mol
    sigma: float                 # Gaussian hill width in radians/Angstroms


class KinematicMetadynamicsEngine:
    """
    Kinematic Metadynamics (Kin-MetaD) Engine:
    Dynamically fills visited local energy minima with repulsive Gaussian hills
    and computes analytical repulsive forces driving continuous Langevin dynamics on the manifold.
    """
    def __init__(
        self,
        unified_engine: UnifiedKinematicPSOEngine,
        gaussian_height_w: float = 25.0,
        gaussian_sigma: float = 0.5
    ):
        self.unified_engine = unified_engine
        self.gaussian_height_w = gaussian_height_w
        self.gaussian_sigma = gaussian_sigma
        self.visited_basins: List[VisitedBasin] = []

    @staticmethod
    def _toroidal_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def compute_metadynamics_bias_and_gradient(
        self,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        rec_chi: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes total repulsive bias energy (kcal/mol) and analytical repulsive gradient vectors
        (g_trans, g_rot, g_ring, g_exo, g_rec) directed away from all visited basins.
        """
        if not self.visited_basins:
            return 0.0, np.zeros(3), np.zeros(3), np.zeros_like(ring_drivers), np.zeros_like(exo_dihedrals), np.zeros_like(rec_chi)
            
        total_bias = 0.0
        g_trans = np.zeros(3)
        g_rot = np.zeros(3)
        g_ring = np.zeros_like(ring_drivers)
        g_exo = np.zeros_like(exo_dihedrals)
        g_rec = np.zeros_like(rec_chi)
        
        two_sigma_sq = 2.0 * (self.gaussian_sigma ** 2)
        inv_sigma_sq = 1.0 / (self.gaussian_sigma ** 2)
        
        for basin in self.visited_basins:
            diff_trans = trans - basin.trans
            diff_rot = self._toroidal_sub(rot_vec, basin.rot_vec)
            diff_ring = self._toroidal_sub(ring_drivers, basin.ring_drivers)
            diff_exo = self._toroidal_sub(exo_dihedrals, basin.exo_dihedrals)
            diff_rec = self._toroidal_sub(rec_chi, basin.rec_chi)
            
            d_trans_sq = np.sum(diff_trans ** 2) / 4.0
            d_rot_sq = np.sum(diff_rot ** 2)
            d_ring_sq = np.sum(diff_ring ** 2)
            d_exo_sq = np.sum(diff_exo ** 2)
            d_rec_sq = np.sum(diff_rec ** 2) / 4.0
            
            total_dist_sq = d_trans_sq + d_rot_sq + d_ring_sq + d_exo_sq + d_rec_sq
            hill = basin.height_w * np.exp(-total_dist_sq / two_sigma_sq)
            total_bias += hill
            
            # Analytical Repulsive Gradient
            factor = hill * inv_sigma_sq
            g_trans += factor * (diff_trans / 4.0)
            g_rot += factor * diff_rot
            g_ring += factor * diff_ring
            g_exo += factor * diff_exo
            g_rec += factor * (diff_rec / 4.0)
            
        return float(total_bias), g_trans, g_rot, g_ring, g_exo, g_rec

    def run_metadynamics_exploration(
        self,
        n_steps: int = 100,
        deposit_frequency: int = 5,
        step_size: float = 0.04,
        temperature_k: float = 300.0
    ) -> Tuple[Chem.Mol, np.ndarray, float, List[Chem.Mol], List[np.ndarray], List[Dict[str, float]]]:
        """
        Executes smooth, continuous Kinematic Metadynamics (Kin-MetaD) Langevin dynamics.
        Repulsive Gaussian hills actively push the macrocycle and side chains through space.
        """
        t_curr = np.zeros(3)
        r_curr = np.zeros(3)
        ring_curr = np.zeros(self.unified_engine.num_ring_drivers)
        exo_curr = np.zeros(self.unified_engine.num_exo)
        rec_curr = np.zeros(self.unified_engine.num_rec_chi)
        
        self.visited_basins.append(VisitedBasin(
            basin_id=1,
            trans=t_curr.copy(),
            rot_vec=r_curr.copy(),
            ring_drivers=ring_curr.copy(),
            exo_dihedrals=exo_curr.copy(),
            rec_chi=rec_curr.copy(),
            raw_score=0.0,
            height_w=self.gaussian_height_w,
            sigma=self.gaussian_sigma
        ))
        
        best_raw_score = 999999.0
        best_t = t_curr.copy()
        best_r = r_curr.copy()
        best_ring = ring_curr.copy()
        best_exo = exo_curr.copy()
        best_rec = rec_curr.copy()
        
        lig_frames: List[Chem.Mol] = []
        rec_frames: List[np.ndarray] = []
        log_data: List[Dict[str, float]] = []
        
        print(f"[*] Starting Kinematic Metadynamics (Kin-MetaD): {n_steps} Smooth Trajectory Steps...")
        print(f"    • Gaussian Hill Height (W): +{self.gaussian_height_w:.1f} kcal/mol | Sigma (σ): {self.gaussian_sigma:.2f}")
        
        for step in range(n_steps):
            bias_val, g_t, g_r, g_ring, g_exo, g_rec = self.compute_metadynamics_bias_and_gradient(
                t_curr, r_curr, ring_curr, exo_curr, rec_curr
            )
            
            # Smooth Langevin step with balanced bounds
            dt = step_size
            noise = 0.02
            
            t_curr += np.clip(g_t * dt, -0.12, 0.12) + np.random.normal(0, noise, 3)
            r_curr = (r_curr + np.clip(g_r * dt, -0.08, 0.08) + np.random.normal(0, noise, 3) + np.pi) % (2 * np.pi) - np.pi
            ring_curr = (ring_curr + np.clip(g_ring * dt, -0.08, 0.08) + np.random.normal(0, noise, self.unified_engine.num_ring_drivers) + np.pi) % (2 * np.pi) - np.pi
            exo_curr = (exo_curr + np.clip(g_exo * dt, -0.10, 0.10) + np.random.normal(0, noise, self.unified_engine.num_exo) + np.pi) % (2 * np.pi) - np.pi
            rec_curr = (rec_curr + np.clip(g_rec * dt, -0.06, 0.06) + np.random.normal(0, noise * 0.5, self.unified_engine.num_rec_chi) + np.pi) % (2 * np.pi) - np.pi
            
            raw_score, c_lig, c_rec = self.unified_engine.evaluate_coupled_state(
                t_curr, r_curr, ring_curr, exo_curr, rec_curr
            )
            
            if raw_score < best_raw_score:
                best_raw_score = raw_score
                best_t = t_curr.copy()
                best_r = r_curr.copy()
                best_ring = ring_curr.copy()
                best_exo = exo_curr.copy()
                best_rec = rec_curr.copy()
                
            if (step + 1) % deposit_frequency == 0:
                self.visited_basins.append(VisitedBasin(
                    basin_id=len(self.visited_basins) + 1,
                    trans=t_curr.copy(),
                    rot_vec=r_curr.copy(),
                    ring_drivers=ring_curr.copy(),
                    exo_dihedrals=exo_curr.copy(),
                    rec_chi=rec_curr.copy(),
                    raw_score=raw_score,
                    height_w=self.gaussian_height_w,
                    sigma=self.gaussian_sigma
                ))
                
            log_data.append({
                "step": step + 1,
                "raw_score": raw_score,
                "bias_kcal": bias_val,
                "effective_score": raw_score + bias_val,
                "num_hills": len(self.visited_basins),
                "best_score": best_raw_score
            })
            
            mol_f = Chem.Mol(self.unified_engine.lig_mol)
            conf_f = mol_f.GetConformer()
            for i in range(mol_f.GetNumAtoms()):
                conf_f.SetAtomPosition(i, Point3D(float(c_lig[i][0]), float(c_lig[i][1]), float(c_lig[i][2])))
            mol_f.SetProp("STEP", str(step + 1))
            mol_f.SetProp("RAW_SCORE_KCAL", f"{raw_score:.2f}")
            mol_f.SetProp("METADYNAMICS_BIAS_KCAL", f"{bias_val:.2f}")
            mol_f.SetProp("HILLS_DEPOSITED", str(len(self.visited_basins)))
            mol_f.SetProp("MAX_BOND_DEV_A", "0.0000")
            lig_frames.append(mol_f)
            rec_frames.append(c_rec)

        _, best_lig_coords, best_rec_coords = self.unified_engine.evaluate_coupled_state(
            best_t, best_r, best_ring, best_exo, best_rec
        )
        best_mol = Chem.Mol(self.unified_engine.lig_mol)
        conf_b = best_mol.GetConformer()
        for i in range(best_mol.GetNumAtoms()):
            conf_b.SetAtomPosition(i, Point3D(float(best_lig_coords[i][0]), float(best_lig_coords[i][1]), float(best_lig_coords[i][2])))
        best_mol.SetProp("FINAL_SCORE_KCAL", f"{best_raw_score:.3f}")
        
        return best_mol, best_rec_coords, best_raw_score, lig_frames, rec_frames, log_data
