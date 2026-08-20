"""
Kinematic Metadynamics Engine for openmm-dock.
Provides history-dependent repulsive Gaussian potential on the (SE(3) x T^k)
kinematic manifold, actively pushing ligands out of local energy wells and decoy traps.
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
    evaluated strictly on the kinematic manifold to guarantee zero bond distortion.
    """
    def __init__(
        self,
        unified_engine: UnifiedKinematicPSOEngine,
        gaussian_height_w: float = 20.0,
        gaussian_sigma: float = 0.5
    ):
        self.unified_engine = unified_engine
        self.gaussian_height_w = gaussian_height_w
        self.gaussian_sigma = gaussian_sigma
        self.visited_basins: List[VisitedBasin] = []

    @staticmethod
    def _toroidal_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def compute_metadynamics_bias(
        self,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        rec_chi: np.ndarray
    ) -> float:
        """
        Computes the total repulsive Gaussian bias energy (kcal/mol)
        accumulated from all previously visited local minima.
        """
        if not self.visited_basins:
            return 0.0
            
        total_bias = 0.0
        two_sigma_sq = 2.0 * (self.gaussian_sigma ** 2)
        
        for basin in self.visited_basins:
            # 1. Translation Euclidean distance squared (scaled)
            d_trans_sq = np.sum((trans - basin.trans) ** 2) / 4.0 # Scale 2 Å ~ 1 rad
            
            # 2. Toroidal angular distances squared on T^k
            d_rot_sq = np.sum(self._toroidal_sub(rot_vec, basin.rot_vec) ** 2)
            d_ring_sq = np.sum(self._toroidal_sub(ring_drivers, basin.ring_drivers) ** 2)
            d_exo_sq = np.sum(self._toroidal_sub(exo_dihedrals, basin.exo_dihedrals) ** 2)
            d_rec_sq = np.sum(self._toroidal_sub(rec_chi, basin.rec_chi) ** 2) / 4.0
            
            total_dist_sq = d_trans_sq + d_rot_sq + d_ring_sq + d_exo_sq + d_rec_sq
            hill = basin.height_w * np.exp(-total_dist_sq / two_sigma_sq)
            total_bias += hill
            
        return float(total_bias)

    def evaluate_effective_energy(
        self,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        rec_chi: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluates: Effective Score = Raw OpenMM Score + Metadynamics Repulsive Bias.
        Returns: (effective_score, raw_score, bias_kcal, lig_coords, rec_coords)
        """
        raw_score, c_lig, c_rec = self.unified_engine.evaluate_coupled_state(
            trans, rot_vec, ring_drivers, exo_dihedrals, rec_chi
        )
        bias_kcal = self.compute_metadynamics_bias(
            trans, rot_vec, ring_drivers, exo_dihedrals, rec_chi
        )
        effective_score = raw_score + bias_kcal
        return effective_score, raw_score, bias_kcal, c_lig, c_rec

    def run_metadynamics_exploration(
        self,
        n_steps: int = 50,
        deposit_frequency: int = 3,
        temperature_k: float = 300.0
    ) -> Tuple[Chem.Mol, np.ndarray, float, List[Chem.Mol], List[np.ndarray], List[Dict[str, float]]]:
        """
        Runs Kinematic Metadynamics Basin-Hopping exploration.
        Actively fills decoy wells and records the escape trajectory.
        """
        # Start state at crystal or current pose
        t_curr = np.zeros(3)
        r_curr = np.zeros(3)
        ring_curr = np.zeros(self.unified_engine.num_ring_drivers)
        exo_curr = np.zeros(self.unified_engine.num_exo)
        rec_curr = np.zeros(self.unified_engine.num_rec_chi)
        
        eff_curr, raw_curr, bias_curr, c_lig, c_rec = self.evaluate_effective_energy(
            t_curr, r_curr, ring_curr, exo_curr, rec_curr
        )
        
        best_raw_score = raw_curr
        best_t = t_curr.copy()
        best_r = r_curr.copy()
        best_ring = ring_curr.copy()
        best_exo = exo_curr.copy()
        best_rec = rec_curr.copy()
        
        lig_frames: List[Chem.Mol] = []
        rec_frames: List[np.ndarray] = []
        log_data: List[Dict[str, float]] = []
        
        k_B_T = 0.001987204 * temperature_k # kcal/mol
        
        print(f"[*] Starting Kinematic Metadynamics (Kin-MetaD): {n_steps} Steps...")
        print(f"    • Gaussian Hill Height (W): +{self.gaussian_height_w:.1f} kcal/mol | Sigma (σ): {self.gaussian_sigma:.2f}")
        
        for step in range(n_steps):
            # 1. Propose Kinematic Perturbation
            t_prop = t_curr + np.random.normal(0, 0.2, 3)
            r_prop = (r_curr + np.random.normal(0, 0.15, 3) + np.pi) % (2 * np.pi) - np.pi
            ring_prop = (ring_curr + np.random.normal(0, 0.12, self.unified_engine.num_ring_drivers) + np.pi) % (2 * np.pi) - np.pi
            exo_prop = (exo_curr + np.random.normal(0, 0.2, self.unified_engine.num_exo) + np.pi) % (2 * np.pi) - np.pi
            rec_prop = (rec_curr + np.random.normal(0, 0.1, self.unified_engine.num_rec_chi) + np.pi) % (2 * np.pi) - np.pi
            
            # 2. Evaluate Effective Energy (including accumulated repulsive hills)
            eff_prop, raw_prop, bias_prop, c_lig_prop, c_rec_prop = self.evaluate_effective_energy(
                t_prop, r_prop, ring_prop, exo_prop, rec_prop
            )
            
            # 3. Metropolis Criterion on Effective Energy Surface
            delta_eff = eff_prop - eff_curr
            accept = False
            if delta_eff < 0:
                accept = True
            else:
                p_acc = np.exp(-delta_eff / k_B_T)
                if np.random.uniform(0, 1) < p_acc:
                    accept = True
                    
            if accept:
                t_curr = t_prop.copy()
                r_curr = r_prop.copy()
                ring_curr = ring_prop.copy()
                exo_curr = exo_prop.copy()
                rec_curr = rec_prop.copy()
                eff_curr = eff_prop
                raw_curr = raw_prop
                c_lig = c_lig_prop
                c_rec = c_rec_prop
                
            # Track Global Physical Best (independent of artificial bias)
            if raw_curr < best_raw_score:
                best_raw_score = raw_curr
                best_t = t_curr.copy()
                best_r = r_curr.copy()
                best_ring = ring_curr.copy()
                best_exo = exo_curr.copy()
                best_rec = rec_curr.copy()
                
            # 4. Periodically Deposit a Repulsive Gaussian Hill to fill the current basin
            if (step + 1) % deposit_frequency == 0:
                basin = VisitedBasin(
                    basin_id=len(self.visited_basins) + 1,
                    trans=t_curr.copy(),
                    rot_vec=r_curr.copy(),
                    ring_drivers=ring_curr.copy(),
                    exo_dihedrals=exo_curr.copy(),
                    rec_chi=rec_curr.copy(),
                    raw_score=raw_curr,
                    height_w=self.gaussian_height_w,
                    sigma=self.gaussian_sigma
                )
                self.visited_basins.append(basin)
                
            # Log Step
            curr_bias = self.compute_metadynamics_bias(t_curr, r_curr, ring_curr, exo_curr, rec_curr)
            log_data.append({
                "step": step + 1,
                "raw_score": raw_curr,
                "bias_kcal": curr_bias,
                "effective_score": raw_curr + curr_bias,
                "num_hills": len(self.visited_basins),
                "best_score": best_raw_score
            })
            
            # Build PyMOL Frame
            mol_f = Chem.Mol(self.unified_engine.lig_mol)
            conf_f = mol_f.GetConformer()
            for i in range(mol_f.GetNumAtoms()):
                conf_f.SetAtomPosition(i, Point3D(float(c_lig[i][0]), float(c_lig[i][1]), float(c_lig[i][2])))
            mol_f.SetProp("STEP", str(step + 1))
            mol_f.SetProp("RAW_SCORE_KCAL", f"{raw_curr:.2f}")
            mol_f.SetProp("METADYNAMICS_BIAS_KCAL", f"{curr_bias:.2f}")
            mol_f.SetProp("EFFECTIVE_SCORE_KCAL", f"{raw_curr + curr_bias:.2f}")
            mol_f.SetProp("HILLS_DEPOSITED", str(len(self.visited_basins)))
            lig_frames.append(mol_f)
            rec_frames.append(c_rec)

        # Build final best complex
        _, best_lig_coords, best_rec_coords = self.unified_engine.evaluate_coupled_state(
            best_t, best_r, best_ring, best_exo, best_rec
        )
        best_mol = Chem.Mol(self.unified_engine.lig_mol)
        conf_b = best_mol.GetConformer()
        for i in range(best_mol.GetNumAtoms()):
            conf_b.SetAtomPosition(i, Point3D(float(best_lig_coords[i][0]), float(best_lig_coords[i][1]), float(best_lig_coords[i][2])))
        best_mol.SetProp("FINAL_SCORE_KCAL", f"{best_raw_score:.3f}")
        
        return best_mol, best_rec_coords, best_raw_score, lig_frames, rec_frames, log_data
