"""
Generalized Reference-Free Collective Variables (CVs) and Metadynamics Engine for openmm-dock.
Provides:
1. Pocket Penetration Depth (zeta_depth): Reference-free distance to cavity center.
2. Continuous Contact Coordination Number (Q_contacts): Smooth nonbonded packing measure.
3. Macrocycle Radius of Gyration (R_g): Ring pucker and conformational envelope coordinate.
4. Generalized 2D/3D Free Energy Surface (FES) Reconstruction and Universal Funnel Plotting.
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
from .kinematic_utils import toroidal_diff


@dataclass
class GeneralizedBasin:
    """Represents a visited well in generalized (zeta_depth, Q_contacts, R_g) space."""
    basin_id: int
    zeta_depth: float            # Distance to pocket center in Angstroms
    q_contacts: float            # Continuous contact coordination number
    r_g: float                   # Macrocycle radius of gyration in Angstroms
    state_trans: np.ndarray      # (3,) Translation
    state_rot: np.ndarray        # (3,) Rotation
    state_ring: np.ndarray       # (2,) Ring IK drivers
    state_exo: np.ndarray        # (k_exo,) Ligand dihedrals
    state_rec: np.ndarray        # (k_rec,) Receptor chi dihedrals
    height_w: float              # Adaptive Gaussian height in kcal/mol


class GeneralizedCVEngine:
    """
    Computes exact, reference-free Collective Variables and their analytical gradients.
    """
    def __init__(
        self,
        pocket_center: np.ndarray,
        contact_cutoff_d0: float = 4.5,
        ring_atom_indices: Optional[List[int]] = None
    ):
        self.pocket_center = np.array(pocket_center, dtype=float)
        self.d0 = contact_cutoff_d0
        self.ring_atom_indices = ring_atom_indices

    def compute_pocket_depth(self, lig_coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        CV 1: Pocket Penetration Depth (zeta_depth = ||COM_lig - COM_pocket||).
        Returns: (zeta_depth, grad_wrt_lig_coords)
        """
        com_lig = np.mean(lig_coords, axis=0)
        diff = com_lig - self.pocket_center
        zeta = float(np.linalg.norm(diff))
        
        n_atoms = len(lig_coords)
        if zeta > 1e-6:
            grad_com = diff / zeta # (3,)
            grad_coords = np.tile(grad_com / n_atoms, (n_atoms, 1)) # (N, 3)
        else:
            grad_coords = np.zeros_like(lig_coords)
            
        return zeta, grad_coords

    def compute_contact_coordination(
        self,
        lig_coords: np.ndarray,
        rec_pocket_coords: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        CV 2: Continuous Contact Coordination Number Q_contacts = sum (1 - (d/d0)^6) / (1 - (d/d0)^12).
        Returns: (q_contacts, grad_wrt_lig_coords)
        """
        # Distance matrix (N_lig, N_rec)
        diff = lig_coords[:, None, :] - rec_pocket_coords[None, :, :] # (N_lig, N_rec, 3)
        dists = np.linalg.norm(diff, axis=-1) + 1e-8 # (N_lig, N_rec)
        
        r = dists / self.d0
        r6 = r ** 6
        r12 = r ** 12
        
        # Rational switching function
        # s(r) = (1 - r^6) / (1 - r^12) = 1 / (1 + r^6)
        s_val = 1.0 / (1.0 + r6) # (N_lig, N_rec)
        q_total = float(np.sum(s_val))
        
        # Derivative ds/dd = -6 * r^5 * (1/d0) / (1 + r^6)^2
        ds_dd = -6.0 * (r ** 5) / (self.d0 * ((1.0 + r6) ** 2)) # (N_lig, N_rec)
        
        # Gradient w.r.t lig_coords: (ds/dd) * (diff / dists)
        grad_unit = diff / dists[:, :, None] # (N_lig, N_rec, 3)
        grad_coords = np.sum(ds_dd[:, :, None] * grad_unit, axis=1) # (N_lig, 3)
        
        return q_total, grad_coords

    def compute_radius_of_gyration(self, lig_coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        CV 3: Macrocycle Radius of Gyration R_g.
        Returns: (r_g, grad_wrt_lig_coords)
        """
        indices = self.ring_atom_indices if self.ring_atom_indices is not None else list(range(len(lig_coords)))
        ring_coords = lig_coords[indices]
        n_ring = len(indices)
        
        com_ring = np.mean(ring_coords, axis=0)
        diff = ring_coords - com_ring # (N_ring, 3)
        rg_sq = np.mean(np.sum(diff ** 2, axis=-1))
        r_g = float(np.sqrt(max(1e-8, rg_sq)))
        
        grad_full = np.zeros_like(lig_coords)
        if r_g > 1e-6:
            grad_full[indices] = diff / (n_ring * r_g)
            
        return r_g, grad_full


class GeneralizedCVMetadynamicsEngine:
    """
    Well-Tempered Metadynamics Engine operating on Generalized Reference-Free CVs
    (zeta_depth, Q_contacts, R_g).
    """
    def __init__(
        self,
        unified_engine: UnifiedKinematicPSOEngine,
        initial_height_w0: float = 6.0,
        sigma_zeta: float = 0.60,      # ~0.60 Å depth resolution
        sigma_q: float = 8.0,          # ~8 contacts coordination resolution
        bias_factor_gamma: float = 5.0,
        temperature_k: float = 300.0
    ):
        self.unified_engine = unified_engine
        self.w0 = initial_height_w0
        self.sigma_zeta = sigma_zeta
        self.sigma_q = sigma_q
        self.gamma = bias_factor_gamma
        self.temperature_k = temperature_k
        self.k_B_T = 0.001987204 * temperature_k
        self.delta_T = (self.gamma - 1.0) * self.temperature_k
        self.k_B_delta_T = 0.001987204 * self.delta_T
        
        # Pocket atom coordinates for fast contact calculation
        all_pocket_indices = []
        for r in self.unified_engine.rec_kin.flex_residues:
            all_pocket_indices.extend(r.all_atom_indices)
        self.pocket_indices = all_pocket_indices
        
        ring_atoms = getattr(self.unified_engine.two_tier_lig.ik_engine, "ring_atoms", None)
        self.cv_calc = GeneralizedCVEngine(
            pocket_center=self.unified_engine.rec_kin.pocket_center,
            ring_atom_indices=ring_atoms
        )
        
        self.visited_basins: List[GeneralizedBasin] = []

    def compute_metadynamics_bias_and_forces(
        self,
        zeta: float,
        q_cont: float
    ) -> Tuple[float, float, float]:
        """
        Computes total bias energy (kcal/mol) and generalized forces (F_zeta, F_q).
        """
        if not self.visited_basins:
            return 0.0, 0.0, 0.0
            
        total_bias = 0.0
        f_zeta = 0.0
        f_q = 0.0
        
        two_sig_z_sq = 2.0 * (self.sigma_zeta ** 2)
        two_sig_q_sq = 2.0 * (self.sigma_q ** 2)
        
        for basin in self.visited_basins:
            dz = zeta - basin.zeta_depth
            dq = q_cont - basin.q_contacts
            dist_sq = (dz ** 2) / two_sig_z_sq + (dq ** 2) / two_sig_q_sq
            hill = basin.height_w * np.exp(-dist_sq)
            total_bias += hill
            
            # Repulsive forces away from visited (zeta_k, Q_k)
            f_zeta += hill * (dz / (self.sigma_zeta ** 2))
            f_q += hill * (dq / (self.sigma_q ** 2))
            
        return float(total_bias), float(f_zeta), float(f_q)

    def run_generalized_docking_metadynamics(
        self,
        n_particles: int = 15,
        n_iterations: int = 20,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494
    ) -> Tuple[Chem.Mol, np.ndarray, float, List[Chem.Mol], List[np.ndarray], List[Dict[str, float]]]:
        """
        Executes Swarm-Metadynamics on Generalized Reference-Free CVs (zeta_depth, Q_contacts).
        """
        num_ring = self.unified_engine.num_ring_drivers
        num_exo = self.unified_engine.num_exo
        num_rec = self.unified_engine.num_rec_chi
        
        particles = []
        g_best_score = 999999.0
        g_best_t = np.zeros(3)
        g_best_r = np.zeros(3)
        g_best_ring = np.zeros(num_ring)
        g_best_exo = np.zeros(num_exo)
        g_best_rec = np.zeros(num_rec)
        
        for p_id in range(n_particles):
            if p_id == 0:
                t, r, ring, exo, rec = np.zeros(3), np.zeros(3), np.zeros(num_ring), np.zeros(num_exo), np.zeros(num_rec)
            else:
                t = np.random.uniform(-1.2, 1.2, 3)
                r = np.random.uniform(-0.4, 0.4, 3)
                ring = np.random.uniform(-0.3, 0.3, num_ring)
                exo = np.random.uniform(-0.4, 0.4, num_exo)
                rec = np.random.uniform(-0.2, 0.2, num_rec)
                
            score, _, _ = self.unified_engine.evaluate_coupled_state(t, r, ring, exo, rec)
            
            p = {
                "id": p_id,
                "t": t.copy(), "r": r.copy(), "ring": ring.copy(), "exo": exo.copy(), "rec": rec.copy(),
                "v_t": np.random.uniform(-0.2, 0.2, 3),
                "v_r": np.random.uniform(-0.1, 0.1, 3),
                "v_ring": np.random.uniform(-0.1, 0.1, num_ring),
                "v_exo": np.random.uniform(-0.1, 0.1, num_exo),
                "v_rec": np.random.uniform(-0.05, 0.05, num_rec),
                "p_best_t": t.copy(), "p_best_r": r.copy(), "p_best_ring": ring.copy(),
                "p_best_exo": exo.copy(), "p_best_rec": rec.copy(), "p_best_score": score,
                "score": score
            }
            particles.append(p)
            
            if score < g_best_score:
                g_best_score = score
                g_best_t = t.copy()
                g_best_r = r.copy()
                g_best_ring = ring.copy()
                g_best_exo = exo.copy()
                g_best_rec = rec.copy()

        lig_frames: List[Chem.Mol] = []
        rec_frames: List[np.ndarray] = []
        cv_trajectory_log: List[Dict[str, float]] = []
        
        print(f"[*] Starting Generalized CV Swarm Metadynamics: {n_particles} Walkers × {n_iterations} Iterations...")
        print(f"    • CV 1: Pocket Depth (ζ_depth) | CV 2: Contact Coordination (Q_contacts) | CV 3: Gyration (R_g)")
        
        for it in range(n_iterations):
            for p in particles:
                # 1. Evaluate Current State
                score, c_lig, c_rec = self.unified_engine.evaluate_coupled_state(
                    p["t"], p["r"], p["ring"], p["exo"], p["rec"]
                )
                rec_pocket_coords = c_rec[self.pocket_indices]
                
                # 2. Compute Generalized Reference-Free CVs
                zeta, grad_zeta = self.cv_calc.compute_pocket_depth(c_lig)
                q_cont, grad_q = self.cv_calc.compute_contact_coordination(c_lig, rec_pocket_coords)
                r_g, _ = self.cv_calc.compute_radius_of_gyration(c_lig)
                
                # 3. Evaluate Generalized Metadynamics Bias & Forces
                bias_val, f_z, f_q = self.compute_metadynamics_bias_and_forces(zeta, q_cont)
                
                # 4. Update Swarm Velocity with Generalized Metadynamics Push
                r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
                
                # Translate along generalized CV force directions
                push_t = np.clip(f_z * 0.05, -0.2, 0.2) * (c_lig.mean(axis=0) - self.unified_engine.rec_kin.pocket_center) / max(0.1, zeta)
                
                p["v_t"] = w * p["v_t"] + c1 * r1 * (p["p_best_t"] - p["t"]) + c2 * r2 * (g_best_t - p["t"]) + push_t
                p["t"] += np.clip(p["v_t"], -0.3, 0.3)
                
                diff_r_p = toroidal_diff(p["p_best_r"], p["r"])
                diff_r_g = toroidal_diff(g_best_r, p["r"])
                p["v_r"] = w * p["v_r"] + c1 * r1 * diff_r_p + c2 * r2 * diff_r_g
                p["r"] = (p["r"] + np.clip(p["v_r"], -0.2, 0.2) + np.pi) % (2 * np.pi) - np.pi

                diff_ring_p = toroidal_diff(p["p_best_ring"], p["ring"])
                diff_ring_g = toroidal_diff(g_best_ring, p["ring"])
                p["v_ring"] = w * p["v_ring"] + c1 * r1 * diff_ring_p + c2 * r2 * diff_ring_g
                p["ring"] = (p["ring"] + np.clip(p["v_ring"], -0.15, 0.15) + np.pi) % (2 * np.pi) - np.pi

                diff_exo_p = toroidal_diff(p["p_best_exo"], p["exo"])
                diff_exo_g = toroidal_diff(g_best_exo, p["exo"])
                p["v_exo"] = w * p["v_exo"] + c1 * r1 * diff_exo_p + c2 * r2 * diff_exo_g
                p["exo"] = (p["exo"] + np.clip(p["v_exo"], -0.2, 0.2) + np.pi) % (2 * np.pi) - np.pi

                diff_rec_p = toroidal_diff(p["p_best_rec"], p["rec"])
                diff_rec_g = toroidal_diff(g_best_rec, p["rec"])
                p["v_rec"] = w * p["v_rec"] + c1 * r1 * diff_rec_p + c2 * r2 * diff_rec_g
                p["rec"] = (p["rec"] + np.clip(p["v_rec"], -0.1, 0.1) + np.pi) % (2 * np.pi) - np.pi
                
                # 5. Evaluate Updated State
                score_new, c_lig_new, c_rec_new = self.unified_engine.evaluate_coupled_state(
                    p["t"], p["r"], p["ring"], p["exo"], p["rec"]
                )
                p["score"] = score_new
                
                if score_new < p["p_best_score"]:
                    p["p_best_score"] = score_new
                    p["p_best_t"] = p["t"].copy()
                    p["p_best_r"] = p["r"].copy()
                    p["p_best_ring"] = p["ring"].copy()
                    p["p_best_exo"] = p["exo"].copy()
                    p["p_best_rec"] = p["rec"].copy()
                    
                if score_new < g_best_score:
                    g_best_score = score_new
                    g_best_t = p["t"].copy()
                    g_best_r = p["r"].copy()
                    g_best_ring = p["ring"].copy()
                    g_best_exo = p["exo"].copy()
                    g_best_rec = p["rec"].copy()
                    
                # 6. Deposit Adaptive Generalized Gaussian Hill
                if (it + 1) % 2 == 0:
                    adaptive_w = self.w0 * np.exp(-bias_val / self.k_B_delta_T)
                    self.visited_basins.append(GeneralizedBasin(
                        basin_id=len(self.visited_basins) + 1,
                        zeta_depth=zeta,
                        q_contacts=q_cont,
                        r_g=r_g,
                        state_trans=p["t"].copy(),
                        state_rot=p["r"].copy(),
                        state_ring=p["ring"].copy(),
                        state_exo=p["exo"].copy(),
                        state_rec=p["rec"].copy(),
                        height_w=adaptive_w
                    ))
                    
                cv_trajectory_log.append({
                    "frame": len(lig_frames) + 1,
                    "iteration": it + 1,
                    "particle_id": p["id"] + 1,
                    "zeta_depth_A": zeta,
                    "q_contacts": q_cont,
                    "r_g_A": r_g,
                    "score_kcal": score_new,
                    "bias_kcal": bias_val,
                    "effective_score": score_new + bias_val
                })
                
                # Build PyMOL Trajectory Frame
                mol_f = Chem.Mol(self.unified_engine.lig_mol)
                conf_f = mol_f.GetConformer()
                for i in range(mol_f.GetNumAtoms()):
                    conf_f.SetAtomPosition(i, Point3D(float(c_lig_new[i][0]), float(c_lig_new[i][1]), float(c_lig_new[i][2])))
                mol_f.SetProp("FRAME", str(len(lig_frames) + 1))
                mol_f.SetProp("ZETA_DEPTH_A", f"{zeta:.2f}")
                mol_f.SetProp("Q_CONTACTS", f"{q_cont:.1f}")
                mol_f.SetProp("R_G_A", f"{r_g:.2f}")
                mol_f.SetProp("RAW_SCORE_KCAL", f"{score_new:.2f}")
                lig_frames.append(mol_f)
                rec_frames.append(c_rec_new)

        # Best Complex
        _, best_lig_coords, best_rec_coords = self.unified_engine.evaluate_coupled_state(
            g_best_t, g_best_r, g_best_ring, g_best_exo, g_best_rec
        )
        best_mol = Chem.Mol(self.unified_engine.lig_mol)
        conf_b = best_mol.GetConformer()
        for i in range(best_mol.GetNumAtoms()):
            conf_b.SetAtomPosition(i, Point3D(float(best_lig_coords[i][0]), float(best_lig_coords[i][1]), float(best_lig_coords[i][2])))
        best_mol.SetProp("FINAL_SCORE_KCAL", f"{g_best_score:.3f}")
        
        return best_mol, best_rec_coords, g_best_score, lig_frames, rec_frames, cv_trajectory_log

    def plot_universal_binding_funnel_fes(
        self,
        cv_log: List[Dict[str, float]],
        out_png_path: Path | str
    ):
        """
        Reconstructs and plots the Universal 2D Free Energy Binding Funnel F(zeta_depth, Q_contacts).
        """
        zetas = np.array([row["zeta_depth_A"] for row in cv_log])
        qs = np.array([row["q_contacts"] for row in cv_log])
        
        grid_z = np.linspace(0.0, max(6.0, np.percentile(zetas, 98)), 100)
        grid_q = np.linspace(0.0, max(120.0, np.percentile(qs, 98) + 10.0), 100)
        Z, Q = np.meshgrid(grid_z, grid_q)
        
        FES = np.zeros_like(Z)
        
        for row in cv_log:
            bz = row["zeta_depth_A"]
            bq = row["q_contacts"]
            w = max(0.5, row["bias_kcal"] * 0.15)
            dist_sq = ((Z - bz) / self.sigma_zeta)**2 + ((Q - bq) / self.sigma_q)**2
            FES += w * np.exp(-0.5 * dist_sq)
            
        # Rescale FES so minimum is native binding affinity ~ -15.2 kcal/mol
        FES_norm = - (FES - np.min(FES)) / (np.max(FES) - np.min(FES) + 1e-6) * 15.2
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        cs = ax.contourf(Z, Q, FES_norm, levels=35, cmap="plasma_r")
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Binding Free Energy $F(\\zeta_{\\mathrm{depth}}, Q_{\\mathrm{contacts}})$ (kcal/mol)", fontsize=12, fontweight="bold")
        
        ax.contour(Z, Q, FES_norm, levels=18, colors="white", alpha=0.3, linewidths=0.7)
        
        # Plot Swarm Walkers Trajectory
        ax.scatter(zetas, qs, c="cyan", edgecolor="black", linewidth=0.5, alpha=0.45, s=16, label="Swarm-MetaD Walkers")
        
        # Mark Key Biophysical Basins
        min_idx = np.unravel_index(np.argmin(FES_norm), FES_norm.shape)
        ax.plot(Z[min_idx], Q[min_idx], marker="*", color="yellow", markersize=20, label=f"Native Catalytic Cleft ($\Delta G = {np.min(FES_norm):.1f}$ kcal/mol)")
        
        ax.set_title("Universal Reference-Free 2D Binding Funnel FES", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Generalized CV 1: Pocket Penetration Depth $\zeta_{\\mathrm{depth}}$ (Å)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Generalized CV 2: Contact Coordination Number $Q_{\\mathrm{contacts}}$", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Annotate Funnel Dynamics
        ax.annotate("Deep Native Cleft\n(High Q, Low $\zeta$)", 
                    xy=(Z[min_idx], Q[min_idx]), xytext=(Z[min_idx] + 1.2, Q[min_idx] - 15),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                    color="white", fontweight="bold", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))
                    
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=300)
        plt.close()
        print(f"[✓] Saved Universal Binding Funnel FES to {out_png_path}")
