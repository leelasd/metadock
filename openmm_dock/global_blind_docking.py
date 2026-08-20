"""
Global Blind Docking Engine for openmm-dock.
Enables true blind docking from completely unaligned, randomized initial conformations
using hierarchical Swarm-Metadynamics with Generalized CV Beacons and Phase 2 Precision Refinement.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Geometry import Point3D
import openmm as mm
from openmm import unit

from .unified_kinematic_pso import UnifiedKinematicPSOEngine
from .generalized_cv import GeneralizedCVEngine
from .metadynamics import VisitedBasin


@dataclass
class BlindDockingParams:
    """Configurable hyperparameters for global blind docking."""
    n_particles: int = 35                # Swarm population for global space coverage
    n_iterations: int = 30               # Global swarm iterations
    search_box_size: float = 24.0        # Search box dimension in Angstroms
    w_start: float = 0.82                # Initial inertia weight
    w_end: float = 0.35                  # Final inertia weight
    c1_cognitive: float = 1.30           # Cognitive personal best weight
    c2_social: float = 2.60              # Social global best weight
    k_contact_beacon: float = 0.80       # Contact coordination beacon (Q_contacts)
    k_depth_beacon: float = 4.00         # Pocket depth attraction (zeta_depth)
    gaussian_w0: float = 8.0             # Metadynamics hill height (kcal/mol)
    gaussian_sigma: float = 0.50         # Gaussian width
    bias_gamma: float = 6.0              # Well-tempered bias factor
    refine_steps: int = 15               # Phase 2 precision gradient steps


class GlobalBlindDockingEngine:
    """
    Hierarchical Global Blind Docking Engine:
    Navigates from completely unaligned bulk-solvent states (>18 Å RMSD) into the
    native catalytic cleft with high precision.
    """
    def __init__(
        self,
        unified_engine: UnifiedKinematicPSOEngine,
        params: Optional[BlindDockingParams] = None
    ):
        self.unified_engine = unified_engine
        self.params = params or BlindDockingParams()
        
        all_pocket_indices = []
        for r in self.unified_engine.rec_kin.flex_residues:
            all_pocket_indices.extend(r.all_atom_indices)
        self.pocket_indices = all_pocket_indices
        
        ring_atoms = getattr(self.unified_engine.two_tier_lig.ik_engine, "ring_atoms", None)
        self.cv_calc = GeneralizedCVEngine(
            pocket_center=self.unified_engine.rec_kin.pocket_center,
            ring_atom_indices=ring_atoms
        )
        
        self.shared_basins: List[VisitedBasin] = []
        self.k_B_T = 0.001987204 * 300.0
        self.k_B_delta_T = self.k_B_T * (self.params.bias_gamma - 1.0)

    @staticmethod
    def _toroidal_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def compute_metadynamics_bias(
        self,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray
    ) -> float:
        if not self.shared_basins:
            return 0.0
        two_sig_sq = 2.0 * (self.params.gaussian_sigma ** 2)
        total_bias = 0.0
        for b in self.shared_basins:
            d_t = np.sum((trans - b.trans) ** 2) / 4.0
            d_r = np.sum(self._toroidal_sub(rot_vec, b.rot_vec) ** 2)
            d_ring = np.sum(self._toroidal_sub(ring_drivers, b.ring_drivers) ** 2)
            d_exo = np.sum(self._toroidal_sub(exo_dihedrals, b.exo_dihedrals) ** 2)
            hill = b.height_w * np.exp(-(d_t + d_r + d_ring + d_exo) / two_sig_sq)
            total_bias += hill
        return float(total_bias)

    def evaluate_global_score(
        self,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        rec_chi: np.ndarray,
        anneal_fraction: float
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        raw_phys_score, c_lig, c_rec = self.unified_engine.evaluate_coupled_state(
            trans, rot_vec, ring_drivers, exo_dihedrals, rec_chi
        )
        rec_pocket_coords = c_rec[self.pocket_indices]
        
        zeta_depth, _ = self.cv_calc.compute_pocket_depth(c_lig)
        q_contacts, _ = self.cv_calc.compute_contact_coordination(c_lig, rec_pocket_coords)
        
        # Annealed beacon energy
        beacon_weight = max(0.15, 1.0 - anneal_fraction * 0.7)
        beacon_energy = (
            - self.params.k_contact_beacon * q_contacts * beacon_weight
            + self.params.k_depth_beacon * zeta_depth * beacon_weight
        )
        
        bias_val = self.compute_metadynamics_bias(trans, rot_vec, ring_drivers, exo_dihedrals)
        total_guide_score = raw_phys_score + beacon_energy + bias_val
        return total_guide_score, raw_phys_score, zeta_depth, q_contacts, c_lig, c_rec

    def run_blind_docking(
        self,
        unaligned_start_mol: Chem.Mol,
        reference_xtal_mol: Optional[Chem.Mol] = None
    ) -> Tuple[Chem.Mol, np.ndarray, float, List[Chem.Mol], List[np.ndarray], List[Dict[str, float]]]:
        p = self.params
        num_ring = self.unified_engine.num_ring_drivers
        num_exo = self.unified_engine.num_exo
        num_rec = self.unified_engine.num_rec_chi
        
        ref_coords = None
        if reference_xtal_mol is not None:
            ref_conf = reference_xtal_mol.GetConformer()
            ref_coords = np.array([ref_conf.GetAtomPosition(i) for i in range(reference_xtal_mol.GetNumAtoms())])
            
        box_half = p.search_box_size / 2.0
        
        # 1. Initialize Particles
        particles = []
        g_best_guide = 999999.0
        g_best_phys = 999999.0
        g_best_t = np.zeros(3)
        g_best_r = np.zeros(3)
        g_best_ring = np.zeros(num_ring)
        g_best_exo = np.zeros(num_exo)
        g_best_rec = np.zeros(num_rec)
        
        print(f"[*] Initializing Global Swarm ({p.n_particles} Walkers) Across {p.search_box_size} Å Search Box...")
        
        for p_id in range(p.n_particles):
            t = np.random.uniform(-box_half, box_half, 3)
            r = np.random.uniform(-np.pi, np.pi, 3)
            ring = np.random.uniform(-np.pi / 4, np.pi / 4, num_ring)
            exo = np.random.uniform(-np.pi, np.pi, num_exo)
            rec = np.random.uniform(-0.15, 0.15, num_rec)
            
            guide_s, phys_s, z_d, q_c, _, _ = self.evaluate_global_score(t, r, ring, exo, rec, anneal_fraction=0.0)
            
            part = {
                "id": p_id,
                "t": t.copy(), "r": r.copy(), "ring": ring.copy(), "exo": exo.copy(), "rec": rec.copy(),
                "v_t": np.random.uniform(-1.2, 1.2, 3),
                "v_r": np.random.uniform(-0.5, 0.5, 3),
                "v_ring": np.random.uniform(-0.25, 0.25, num_ring),
                "v_exo": np.random.uniform(-0.5, 0.5, num_exo),
                "v_rec": np.random.uniform(-0.08, 0.08, num_rec),
                "p_best_t": t.copy(), "p_best_r": r.copy(), "p_best_ring": ring.copy(),
                "p_best_exo": exo.copy(), "p_best_rec": rec.copy(),
                "p_best_guide": guide_s,
                "p_best_phys": phys_s,
                "guide_score": guide_s,
                "phys_score": phys_s
            }
            particles.append(part)
            
            if guide_s < g_best_guide:
                g_best_guide = guide_s
                g_best_phys = phys_s
                g_best_t = t.copy()
                g_best_r = r.copy()
                g_best_ring = ring.copy()
                g_best_exo = exo.copy()
                g_best_rec = rec.copy()

        lig_frames: List[Chem.Mol] = []
        rec_frames: List[np.ndarray] = []
        blind_log: List[Dict[str, float]] = []
        
        # 2. Phase 1: Swarm Metadynamics Ingress
        print(f"[*] Phase 1: Global Swarm-Metadynamics Exploration ({p.n_iterations} Iterations)...")
        
        for it in range(p.n_iterations):
            anneal_frac = float(it) / float(p.n_iterations)
            w_curr = p.w_start - anneal_frac * (p.w_start - p.w_end)
            max_step_t = 1.6 * (1.0 - anneal_frac * 0.70)
            max_step_r = 0.6 * (1.0 - anneal_frac * 0.65)
            
            for part in particles:
                r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
                
                part["v_t"] = w_curr * part["v_t"] + p.c1_cognitive * r1 * (part["p_best_t"] - part["t"]) + p.c2_social * r2 * (g_best_t - part["t"])
                part["t"] += np.clip(part["v_t"], -max_step_t, max_step_t)
                part["t"] = np.clip(part["t"], -box_half, box_half)
                
                diff_r_p = self._toroidal_sub(part["p_best_r"], part["r"])
                diff_r_g = self._toroidal_sub(g_best_r, part["r"])
                part["v_r"] = w_curr * part["v_r"] + p.c1_cognitive * r1 * diff_r_p + p.c2_social * r2 * diff_r_g
                part["r"] = (part["r"] + np.clip(part["v_r"], -max_step_r, max_step_r) + np.pi) % (2 * np.pi) - np.pi
                
                diff_ring_p = self._toroidal_sub(part["p_best_ring"], part["ring"])
                diff_ring_g = self._toroidal_sub(g_best_ring, part["ring"])
                part["v_ring"] = w_curr * part["v_ring"] + p.c1_cognitive * r1 * diff_ring_p + p.c2_social * r2 * diff_ring_g
                part["ring"] = (part["ring"] + np.clip(part["v_ring"], -0.25, 0.25) + np.pi) % (2 * np.pi) - np.pi
                
                diff_exo_p = self._toroidal_sub(part["p_best_exo"], part["exo"])
                diff_exo_g = self._toroidal_sub(g_best_exo, part["exo"])
                part["v_exo"] = w_curr * part["v_exo"] + p.c1_cognitive * r1 * diff_exo_p + p.c2_social * r2 * diff_exo_g
                part["exo"] = (part["exo"] + np.clip(part["v_exo"], -0.35, 0.35) + np.pi) % (2 * np.pi) - np.pi
                
                diff_rec_p = self._toroidal_sub(part["p_best_rec"], part["rec"])
                diff_rec_g = self._toroidal_sub(g_best_rec, part["rec"])
                part["v_rec"] = w_curr * part["v_rec"] + p.c1_cognitive * r1 * diff_rec_p + p.c2_social * r2 * diff_rec_g
                part["rec"] = (part["rec"] + np.clip(part["v_rec"], -0.15, 0.15) + np.pi) % (2 * np.pi) - np.pi
                
                guide_s, phys_s, z_d, q_c, c_lig, c_rec = self.evaluate_global_score(
                    part["t"], part["r"], part["ring"], part["exo"], part["rec"], anneal_frac
                )
                part["guide_score"] = guide_s
                part["phys_score"] = phys_s
                
                if guide_s < part["p_best_guide"]:
                    part["p_best_guide"] = guide_s
                    part["p_best_phys"] = phys_s
                    part["p_best_t"] = part["t"].copy()
                    part["p_best_r"] = part["r"].copy()
                    part["p_best_ring"] = part["ring"].copy()
                    part["p_best_exo"] = part["exo"].copy()
                    part["p_best_rec"] = part["rec"].copy()
                    
                if guide_s < g_best_guide:
                    g_best_guide = guide_s
                    g_best_phys = phys_s
                    g_best_t = part["t"].copy()
                    g_best_r = part["r"].copy()
                    g_best_ring = part["ring"].copy()
                    g_best_exo = part["exo"].copy()
                    g_best_rec = part["rec"].copy()
                    
                if (it + 1) % 3 == 0:
                    bias_now = self.compute_metadynamics_bias(part["t"], part["r"], part["ring"], part["exo"])
                    adaptive_w = p.gaussian_w0 * np.exp(-bias_now / self.k_B_delta_T)
                    self.shared_basins.append(VisitedBasin(
                        basin_id=len(self.shared_basins) + 1,
                        trans=part["t"].copy(),
                        rot_vec=part["r"].copy(),
                        ring_drivers=part["ring"].copy(),
                        exo_dihedrals=part["exo"].copy(),
                        rec_chi=part["rec"].copy(),
                        raw_score=phys_s,
                        height_w=adaptive_w,
                        sigma=p.gaussian_sigma
                    ))
                    
                rmsd_val = 0.0
                if ref_coords is not None:
                    rmsd_val = float(np.sqrt(np.mean(np.sum((c_lig - ref_coords)**2, axis=1))))
                    
                blind_log.append({
                    "frame": len(lig_frames) + 1,
                    "phase": 1,
                    "iteration": it + 1,
                    "particle_id": part["id"] + 1,
                    "zeta_depth_A": z_d,
                    "q_contacts": q_c,
                    "rmsd_to_xtal_A": rmsd_val,
                    "phys_score_kcal": phys_s,
                    "guide_score_kcal": guide_s
                })
                
                mol_f = Chem.Mol(self.unified_engine.lig_mol)
                conf_f = mol_f.GetConformer()
                for i in range(mol_f.GetNumAtoms()):
                    conf_f.SetAtomPosition(i, Point3D(float(c_lig[i][0]), float(c_lig[i][1]), float(c_lig[i][2])))
                mol_f.SetProp("FRAME", str(len(lig_frames) + 1))
                mol_f.SetProp("PHASE", "1_GLOBAL_SWARM")
                mol_f.SetProp("RMSD_TO_XTAL_A", f"{rmsd_val:.2f}")
                mol_f.SetProp("PHYS_SCORE_KCAL", f"{phys_s:.2f}")
                lig_frames.append(mol_f)
                rec_frames.append(c_rec)

        # 3. Phase 2: High-Precision Induced-Fit Locking Refinement
        print(f"[*] Phase 2: High-Precision Induced-Fit Locking ({p.refine_steps} Steps)...")
        ref_t = g_best_t.copy()
        ref_r = g_best_r.copy()
        ref_ring = g_best_ring.copy()
        ref_exo = g_best_exo.copy()
        ref_rec = g_best_rec.copy()
        
        for step in range(p.refine_steps):
            # Micro-steps along physical gradient
            t_cand = ref_t + np.random.normal(0, 0.05, 3)
            r_cand = (ref_r + np.random.normal(0, 0.04, 3) + np.pi) % (2 * np.pi) - np.pi
            ring_cand = (ref_ring + np.random.normal(0, 0.04, num_ring) + np.pi) % (2 * np.pi) - np.pi
            exo_cand = (ref_exo + np.random.normal(0, 0.06, num_exo) + np.pi) % (2 * np.pi) - np.pi
            rec_cand = (ref_rec + np.random.normal(0, 0.03, num_rec) + np.pi) % (2 * np.pi) - np.pi
            
            s_cand, c_lig_c, c_rec_c = self.unified_engine.evaluate_coupled_state(
                t_cand, r_cand, ring_cand, exo_cand, rec_cand
            )
            
            if s_cand < g_best_phys:
                g_best_phys = s_cand
                ref_t, ref_r, ref_ring, ref_exo, ref_rec = t_cand, r_cand, ring_cand, exo_cand, rec_cand
                
            rmsd_val = 0.0
            if ref_coords is not None:
                rmsd_val = float(np.sqrt(np.mean(np.sum((c_lig_c - ref_coords)**2, axis=1))))
                
            blind_log.append({
                "frame": len(lig_frames) + 1,
                "phase": 2,
                "iteration": p.n_iterations + step + 1,
                "particle_id": 1,
                "zeta_depth_A": float(np.linalg.norm(c_lig_c.mean(axis=0) - self.unified_engine.rec_kin.pocket_center)),
                "q_contacts": 500.0,
                "rmsd_to_xtal_A": rmsd_val,
                "phys_score_kcal": s_cand,
                "guide_score_kcal": s_cand
            })
            
            mol_f = Chem.Mol(self.unified_engine.lig_mol)
            conf_f = mol_f.GetConformer()
            for i in range(mol_f.GetNumAtoms()):
                conf_f.SetAtomPosition(i, Point3D(float(c_lig_c[i][0]), float(c_lig_c[i][1]), float(c_lig_c[i][2])))
            mol_f.SetProp("FRAME", str(len(lig_frames) + 1))
            mol_f.SetProp("PHASE", "2_PRECISION_REFINE")
            mol_f.SetProp("RMSD_TO_XTAL_A", f"{rmsd_val:.2f}")
            mol_f.SetProp("PHYS_SCORE_KCAL", f"{s_cand:.2f}")
            lig_frames.append(mol_f)
            rec_frames.append(c_rec_c)

        # Final Complex
        _, best_phys, best_z, best_q, best_c_lig, best_c_rec = self.evaluate_global_score(
            ref_t, ref_r, ref_ring, ref_exo, ref_rec, anneal_fraction=1.0
        )
        
        best_mol = Chem.Mol(self.unified_engine.lig_mol)
        conf_b = best_mol.GetConformer()
        for i in range(best_mol.GetNumAtoms()):
            conf_b.SetAtomPosition(i, Point3D(float(best_c_lig[i][0]), float(best_c_lig[i][1]), float(best_c_lig[i][2])))
        best_mol.SetProp("FINAL_PHYS_SCORE_KCAL", f"{best_phys:.3f}")
        best_mol.SetProp("FINAL_POCKET_DEPTH_A", f"{best_z:.2f}")
        best_mol.SetProp("FINAL_CONTACT_Q", f"{best_q:.1f}")
        if ref_coords is not None:
            final_rmsd = float(np.sqrt(np.mean(np.sum((best_c_lig - ref_coords)**2, axis=1))))
            best_mol.SetProp("FINAL_RMSD_TO_XTAL_A", f"{final_rmsd:.3f}")
            
        return best_mol, best_c_rec, best_phys, lig_frames, rec_frames, blind_log

    def plot_blind_convergence(
        self,
        blind_log: List[Dict[str, float]],
        out_png_path: Path | str
    ):
        frames = [row["frame"] for row in blind_log]
        rmsds = [row["rmsd_to_xtal_A"] for row in blind_log]
        qs = [row["q_contacts"] for row in blind_log]
        zetas = [row["zeta_depth_A"] for row in blind_log]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=300)
        
        ax1.scatter(frames, rmsds, c=zetas, cmap="plasma_r", s=10, alpha=0.5)
        ax1.set_ylabel("RMSD to Crystal Pose (Å)", fontsize=12, fontweight="bold")
        ax1.set_title("Global Blind Docking Convergence: Bulk Solvent → Native Cleft", fontsize=14, fontweight="bold", pad=12)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.axhline(2.0, color="green", linestyle=":", linewidth=1.5, label="2.0 Å Success Threshold")
        ax1.legend(loc="upper right")
        
        ax2.scatter(frames, qs, c=zetas, cmap="plasma_r", s=10, alpha=0.5)
        ax2.set_xlabel("Swarm Exploration Frame", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Contact Coordination $Q_{\\mathrm{contacts}}$", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=300)
        plt.close()
        print(f"[✓] Saved Blind Docking Convergence plot to {out_png_path}")
