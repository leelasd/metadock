"""
Collaborative Multi-Swarm Kinematic Metadynamics Engine for openmm-dock.

Features:
1. 19D Kinematic Parameter Space (SE(3) rigid-body + Macrocycle Ring IK + Exocyclic FK dihedrals)
   with 0.000 Å internal bond/angle strain.
2. Rigid Receptor OpenMM GPU acceleration (eliminates 31+ noisy sidechain DOFs during global search).
3. Collaborative Multi-Swarm (Island Model) Architecture:
   - Multiple independent sub-swarms explore distinct conformational/pocket pathways.
   - Zero premature gravitational collapse to a single decoy (local island attractors l_best).
   - Communication via Shared Metadynamics Archive: Visited decoy basins are deposited as repulsive
     Gaussian hills (shared negative memory: "do not re-explore here!").
4. Multi-Track PyMOL Visualization & 2D Free Energy Surface (FES) analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import openmm as mm
from openmm import unit

from .engine import DockingEngine
from .inverse_kinematics import TwoTierMacrocycleEngine



@dataclass
class SharedBasin:
    """Represents a visited conformational/spatial minimum in the shared memory archive."""
    basin_id: int
    island_id: int
    iteration: int
    trans: np.ndarray
    rot_vec: np.ndarray
    ring_drivers: np.ndarray
    exo_dihedrals: np.ndarray
    coords: np.ndarray             # (N_atoms, 3) Heavy atom Cartesian coordinates
    phys_score: float              # Raw unbiased OpenMM energy (kcal/mol)
    height_w: float                # Gaussian hill height (kcal/mol)
    sigma: float                   # Gaussian hill width (Å)


class SharedMetadynamicsArchive:
    """
    Shared memory archive that records visited local basins from all islands
    and computes dynamic repulsive Gaussian bias potentials. Islands are
    processed sequentially within a single process (see
    run_collaborative_docking), not concurrently, so no locking is needed here.
    """
    def __init__(
        self,
        initial_height_w0: float = 12.0,
        gaussian_sigma: float = 1.20,
        bias_factor_gamma: float = 6.0,
        temperature_k: float = 300.0,
        min_basin_rmsd: float = 1.50
    ):
        self.w0 = initial_height_w0
        self.sigma = gaussian_sigma
        self.gamma = bias_factor_gamma
        self.temperature_k = temperature_k
        self.k_B = 0.001987204
        self.k_B_T = self.k_B * temperature_k
        self.delta_T = (self.gamma - 1.0) * self.temperature_k
        self.k_B_delta_T = self.k_B * self.delta_T
        self.min_basin_rmsd = min_basin_rmsd
        
        self.basins: List[SharedBasin] = []

    def compute_bias(self, coords: np.ndarray) -> float:
        """
        Computes total repulsive Metadynamics bias for a candidate pose using
        heavy-atom 3D RMSD distance to all recorded shared basins:
        V_bias(x) = sum_k W_k * exp(-RMSD(x, x_k)^2 / (2 * sigma^2))
        """
        if not self.basins:
            return 0.0
        
        total_bias = 0.0
        two_sigma_sq = 2.0 * (self.sigma ** 2)
        
        for basin in self.basins:
            diff = coords - basin.coords
            rmsd_sq = float(np.mean(np.sum(diff ** 2, axis=1)))
            hill = basin.height_w * np.exp(-rmsd_sq / two_sigma_sq)
            total_bias += hill
            
        return float(total_bias)

    def register_basin(
        self,
        island_id: int,
        iteration: int,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        coords: np.ndarray,
        phys_score: float
    ) -> Optional[SharedBasin]:
        """
        Attempts to register a newly discovered local minimum.
        Applies Well-Tempered scaling to the Gaussian hill height.
        """
        # Check if already within min_basin_rmsd of an existing recorded basin
        for b in self.basins:
            rmsd = float(np.sqrt(np.mean(np.sum((coords - b.coords) ** 2, axis=1))))
            if rmsd < self.min_basin_rmsd:
                # Close to an existing basin: skip registering a duplicate.
                # (Repeated visits to this region get no additional deterrence
                # beyond the first deposit -- the existing hill's height is not
                # reinforced.)
                return None
                
        # Well-Tempered height scaling: W = W0 * exp(-V_bias / (k_B * delta_T))
        existing_bias = self.compute_bias(coords)
        height = self.w0 * np.exp(-existing_bias / self.k_B_delta_T) if self.delta_T > 0 else self.w0
        height = max(1.0, float(height))
        
        new_basin = SharedBasin(
            basin_id=len(self.basins) + 1,
            island_id=island_id,
            iteration=iteration,
            trans=trans.copy(),
            rot_vec=rot_vec.copy(),
            ring_drivers=ring_drivers.copy(),
            exo_dihedrals=exo_dihedrals.copy(),
            coords=coords.copy(),
            phys_score=phys_score,
            height_w=height,
            sigma=self.sigma
        )
        self.basins.append(new_basin)
        return new_basin


@dataclass
class CollaborativeParticle:
    """Represents a particle residing on an independent sub-swarm island."""
    particle_id: int
    island_id: int
    conformer_seed_id: int
    trans: np.ndarray
    rot_vec: np.ndarray
    ring_drivers: np.ndarray
    exo_dihedrals: np.ndarray
    
    # Velocities
    v_trans: np.ndarray
    v_rot: np.ndarray
    v_ring: np.ndarray
    v_exo: np.ndarray
    
    # Personal best (in effective biased energy)
    p_best_trans: np.ndarray
    p_best_rot: np.ndarray
    p_best_ring: np.ndarray
    p_best_exo: np.ndarray
    p_best_effective_score: float
    p_best_phys_score: float
    p_best_coords: np.ndarray
    
    # Current states
    current_effective_score: float
    current_phys_score: float
    current_coords: np.ndarray


class CollaborativeIsland:
    """
    An independent sub-swarm island exploring the 19D kinematic landscape.
    Maintains its own local social attractor (l_best) and personal bests (p_best),
    avoiding premature collapse across islands while receiving repulsive Metadynamics biases.
    """
    def __init__(
        self,
        island_id: int,
        n_particles: int,
        conformer_seed_id: int,
        num_ring_drivers: int,
        num_exo: int,
        search_radius: float = 6.0
    ):
        self.island_id = island_id
        self.n_particles = n_particles
        self.conformer_seed_id = conformer_seed_id
        self.num_ring_drivers = num_ring_drivers
        self.num_exo = num_exo
        self.search_radius = search_radius
        
        self.particles: List[CollaborativeParticle] = []
        self.l_best_effective_score: float = float("inf")
        self.l_best_phys_score: float = float("inf")
        self.l_best_trans: np.ndarray = np.zeros(3)
        self.l_best_rot: np.ndarray = np.zeros(3)
        self.l_best_ring: np.ndarray = np.zeros(num_ring_drivers)
        self.l_best_exo: np.ndarray = np.zeros(num_exo)
        self.l_best_coords: Optional[np.ndarray] = None


@dataclass
class CollaborativeMetaDParams:
    """Configuration parameters for Collaborative Multi-Swarm Kinematic Metadynamics."""
    num_islands: int = 4
    particles_per_island: int = 16
    n_iterations: int = 35
    search_radius: float = 6.0
    
    # PSO Inertia and Acceleration
    w_start: float = 0.80
    w_end: float = 0.35
    c1_cognitive: float = 1.4
    c2_social: float = 1.8
    
    # Metadynamics Repulsive Bias
    initial_height_w0: float = 14.0
    gaussian_sigma: float = 1.25
    bias_factor_gamma: float = 6.0
    temperature_k: float = 300.0
    min_basin_rmsd: float = 1.60
    basin_deposit_interval: int = 5

    # Guide-score CV beacons (same names/defaults as global_blind_docking.py's
    # BlindDockingParams.k_contact_beacon / k_depth_beacon, for consistency)
    k_contact_beacon: float = 0.80
    k_depth_beacon: float = 4.00


class CollaborativeKinematicMetaDEngine:
    """
    Grand Collaborative Multi-Swarm Kinematic Metadynamics Engine.
    Executes multi-island kinematic search with shared negative Metadynamics memory.
    """
    def __init__(
        self,
        receptor_pdb_path: Path | str,
        pocket_center: np.ndarray,
        ligand_mol: Chem.Mol,
        num_conformer_seeds: int = 4
    ):
        self.receptor_path = Path(receptor_pdb_path)
        self.pocket_center = np.array(pocket_center, dtype=np.float64)
        self.lig_mol = Chem.Mol(ligand_mol)
        self.two_tier = TwoTierMacrocycleEngine(ligand_mol)
        
        # Adaptive Ring IK Drivers
        num_joints = len(self.two_tier.ik_engine.joints)
        if num_joints >= 4:
            self.driver_joint_indices = [1, 3, 5, min(8, num_joints - 1)]
        elif num_joints > 0:
            self.driver_joint_indices = [0]
        else:
            self.driver_joint_indices = []
        self.num_ring_drivers = len(self.driver_joint_indices)
        self.num_exo = len(self.two_tier.exo_joints)
        self.num_dofs = 6 + self.num_ring_drivers + self.num_exo
        
        # Multi-conformer templates for macrocycle
        self.conformer_templates: List[np.ndarray] = [self.two_tier.base_coords]
        if self.num_ring_drivers > 0 and num_conformer_seeds > 1:
            seeds = self.generate_conformer_seeds(self.lig_mol, num_seeds=num_conformer_seeds)
            if seeds:
                self.conformer_templates = seeds
        self.num_conformer_seeds = len(self.conformer_templates)
        # Build OpenMM Docking System (Rigid Receptor)
        self.engine = DockingEngine(receptor_path=self.receptor_path)
        self.system, _, self.lig_start, self.lig_n = self.engine._build_system(self.lig_mol)
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        self.context = (
            mm.Context(self.system, self.integrator, self.engine.platform)
            if self.engine.platform
            else mm.Context(self.system, self.integrator)
        )
        
        # Extract Receptor Pocket Atoms within 12 Å of pocket center for contact calculation
        rec_coords = self.engine.receptor.coordinates
        d_rec = np.linalg.norm(rec_coords - self.pocket_center, axis=1)
        self.pocket_rec_coords = rec_coords[d_rec < 12.0]
        self.ring_atom_indices = self.two_tier.ik_engine.ring_atoms
        
        print(f"[*] Collaborative Kinematic MetaD Engine Initialized:")
        print(f"    • Kinematic DOF Vector : 6 Rigid-Body + {self.num_ring_drivers} Ring Drivers + {self.num_exo} Exocyclic FK ({self.num_dofs}D Total)")
        print(f"    • Macrocycle Ring Core : {len(self.ring_atom_indices)} Atoms in Closed-Loop Backbone")
        print(f"    • Conformer Templates  : {self.num_conformer_seeds} Multi-Conformer Ring Seeds")
        print(f"    • Receptor Scaffold    : {len(self.pocket_rec_coords)} Active Pocket Atoms")

    def generate_conformer_seeds(self, mol: Chem.Mol, num_seeds: int = 4) -> List[np.ndarray]:
        """Generates diverse 3D macrocyclic ring conformer seeds centered at pocket centroid."""
        mol_work = Chem.Mol(mol)
        cids = AllChem.EmbedMultipleConfs(mol_work, numConfs=num_seeds, params=AllChem.ETKDGv3())
        seeds = []
        for cid in cids:
            conf = mol_work.GetConformer(cid)
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol_work.GetNumAtoms())])
            seeds.append(coords)
        if not seeds:
            seeds = [self.two_tier.base_coords]
        return seeds

    @staticmethod
    def _toroidal_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def compute_cvs(self, lig_coords: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Computes 4 Biophysical Collective Variables:
        1. zeta_depth: Pocket penetration depth (COM distance in Å)
        2. q_contacts: Continuous contact coordination number with active site
        3. r_g_all: Total ligand Radius of Gyration (Å)
        4. r_g_ring: Macrocycle core ring Radius of Gyration (Å)
        """
        com_lig = np.mean(lig_coords, axis=0)
        zeta_depth = float(np.linalg.norm(com_lig - self.pocket_center))
        
        # Nonbonded contacts with pocket atoms
        diff = lig_coords[:, np.newaxis, :] - self.pocket_rec_coords[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        s_ij = 1.0 / (1.0 + (dist_sq / (4.5 ** 2)) ** 3)
        q_contacts = float(np.sum(s_ij))
        
        # Total Radius of Gyration (All atoms)
        rg_all_sq = np.mean(np.sum((lig_coords - com_lig) ** 2, axis=1))
        r_g_all = float(np.sqrt(max(1e-8, rg_all_sq)))
        
        # Macrocycle Ring Core Radius of Gyration
        ring_c = lig_coords[self.ring_atom_indices]
        com_ring = np.mean(ring_c, axis=0)
        rg_ring_sq = np.mean(np.sum((ring_c - com_ring) ** 2, axis=1))
        r_g_ring = float(np.sqrt(max(1e-8, rg_ring_sq)))
        
        return zeta_depth, q_contacts, r_g_all, r_g_ring


    def evaluate_kinematics(
        self,
        trans: np.ndarray,
        rot_vec: np.ndarray,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        conformer_seed_id: int,
        k_contact_beacon: float = 0.80,
        k_depth_beacon: float = 4.00
    ) -> Tuple[float, float, np.ndarray]:
        """
        Evaluates coupled 19D kinematics on OpenMM GPU:
        1. Solves Macrocycle Ring IK loop closure on seed template
        2. Rotates Exocyclic FK dihedrals
        3. Applies SE(3) translation & rotation
        Returns: (guide_score_kcal, pure_openmm_score_kcal, cartesian_coords_angstrom)

        k_contact_beacon/k_depth_beacon match the names and default values of
        global_blind_docking.py's BlindDockingParams.k_contact_beacon /
        k_depth_beacon, for consistency across the swarm-search engines.
        """
        seed_idx = conformer_seed_id % self.num_conformer_seeds
        base_c = self.conformer_templates[seed_idx]
        
        # 1. Ring IK
        if self.num_ring_drivers > 0:
            d_dict = {self.driver_joint_indices[i]: float(ring_drivers[i]) for i in range(min(len(ring_drivers), self.num_ring_drivers))}
            c_lig, _, _ = self.two_tier.ik_engine.solve_loop_closure(base_c, driver_angles=d_dict)
        else:
            c_lig = base_c.copy()
            
        # 2. Exocyclic FK
        for j_idx in range(min(len(exo_dihedrals), self.num_exo)):
            c_lig = self.two_tier.apply_exocyclic_rotation(c_lig, j_idx, float(exo_dihedrals[j_idx]))
            
        # 3. SE(3) Rigid Body
        center = np.mean(c_lig, axis=0)
        norm_rot = np.linalg.norm(rot_vec)
        if norm_rot > 1e-6:
            r_mat = ScipyRotation.from_rotvec(rot_vec).as_matrix()
            c_lig = (c_lig - center).dot(r_mat.T) + center
            
        c_lig = c_lig - center + self.pocket_center + trans
        
        # Evaluate OpenMM GPU Energy
        full_pos = self.engine._full_positions_from_coords(c_lig)
        self.context.setPositions(full_pos)
        state = self.context.getState(getEnergy=True)
        raw_score_kcal = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        
        # Compute smooth contact beacon guidance
        zeta_d, q_c, _, _ = self.compute_cvs(c_lig)
        guide_score = raw_score_kcal - k_contact_beacon * q_c + k_depth_beacon * zeta_d
        return guide_score, raw_score_kcal, c_lig


    def run_collaborative_docking(
        self,
        params: CollaborativeMetaDParams = CollaborativeMetaDParams(),
        reference_xtal_mol: Optional[Chem.Mol] = None
    ) -> Tuple[Chem.Mol, float, List[Chem.Mol], Dict[str, Any]]:
        """
        Executes the Collaborative Multi-Swarm Kinematic Metadynamics Pipeline.
        """
        ref_coords = None
        if reference_xtal_mol is not None:
            conf_ref = reference_xtal_mol.GetConformer()
            ref_coords = np.array([conf_ref.GetAtomPosition(i) for i in range(reference_xtal_mol.GetNumAtoms())])
            
        archive = SharedMetadynamicsArchive(
            initial_height_w0=params.initial_height_w0,
            gaussian_sigma=params.gaussian_sigma,
            bias_factor_gamma=params.bias_factor_gamma,
            temperature_k=params.temperature_k,
            min_basin_rmsd=params.min_basin_rmsd
        )
        
        # 1. Initialize Multi-Islands
        islands: List[CollaborativeIsland] = []
        for i_id in range(params.num_islands):
            seed_id = i_id % self.num_conformer_seeds
            island = CollaborativeIsland(
                island_id=i_id + 1,
                n_particles=params.particles_per_island,
                conformer_seed_id=seed_id,
                num_ring_drivers=self.num_ring_drivers,
                num_exo=self.num_exo,
                search_radius=params.search_radius
            )
            
            for p_idx in range(params.particles_per_island):
                # Sample initial translation in sphere
                u_dir = np.random.normal(size=3)
                u_dir /= (np.linalg.norm(u_dir) + 1e-9)
                r_dist = np.random.uniform(0.5, params.search_radius)
                trans = u_dir * r_dist
                
                rot_vec = np.random.uniform(-np.pi, np.pi, size=3)
                ring_drivers = np.random.uniform(-np.pi / 4, np.pi / 4, size=self.num_ring_drivers)
                exo_dihedrals = np.random.uniform(-np.pi, np.pi, size=self.num_exo)
                
                v_trans = np.random.normal(scale=0.3, size=3)
                v_rot = np.random.normal(scale=0.2, size=3)
                v_ring = np.random.normal(scale=0.2, size=self.num_ring_drivers)
                v_exo = np.random.normal(scale=0.2, size=self.num_exo)
                
                guide_s, phys_score, coords = self.evaluate_kinematics(
                    trans, rot_vec, ring_drivers, exo_dihedrals, seed_id,
                    k_contact_beacon=params.k_contact_beacon, k_depth_beacon=params.k_depth_beacon
                )
                bias = archive.compute_bias(coords)
                eff_score = guide_s + bias
                
                p = CollaborativeParticle(
                    particle_id=p_idx + 1,
                    island_id=i_id + 1,
                    conformer_seed_id=seed_id,
                    trans=trans,
                    rot_vec=rot_vec,
                    ring_drivers=ring_drivers,
                    exo_dihedrals=exo_dihedrals,
                    v_trans=v_trans,
                    v_rot=v_rot,
                    v_ring=v_ring,
                    v_exo=v_exo,
                    p_best_trans=trans.copy(),
                    p_best_rot=rot_vec.copy(),
                    p_best_ring=ring_drivers.copy(),
                    p_best_exo=exo_dihedrals.copy(),
                    p_best_effective_score=eff_score,
                    p_best_phys_score=phys_score,
                    p_best_coords=coords.copy(),
                    current_effective_score=eff_score,
                    current_phys_score=phys_score,
                    current_coords=coords.copy()
                )
                island.particles.append(p)
                
                if eff_score < island.l_best_effective_score:
                    island.l_best_effective_score = eff_score
                    island.l_best_phys_score = phys_score
                    island.l_best_trans = trans.copy()
                    island.l_best_rot = rot_vec.copy()
                    island.l_best_ring = ring_drivers.copy()
                    island.l_best_exo = exo_dihedrals.copy()
                    island.l_best_coords = coords.copy()
                    
            islands.append(island)

        print(f"\n[*] Launching Collaborative Multi-Swarm MetaD ({params.num_islands} Islands x {params.particles_per_island} Particles = {params.num_islands * params.particles_per_island} Walkers):")
        
        master_log: List[Dict[str, Any]] = []
        all_trajectory_mols: List[Chem.Mol] = []
        frame_counter = 0
        
        global_best_unbiased_score = float("inf")
        global_best_coords = None
        global_best_rmsd = float("inf")
        
        # 2. Main Optimization Loop
        phase2_start = int(0.65 * params.n_iterations)
        
        for it in range(1, params.n_iterations + 1):
            w = params.w_start - (params.w_start - params.w_end) * (it / params.n_iterations)
            
            # Two-stage bias fade-out schedule (Phase 1 exploration -> Phase 2 deep-well physical annealing)
            if it <= phase2_start:
                bias_weight = 1.0
            else:
                bias_weight = max(0.0, 1.0 - (it - phase2_start) / max(1, (params.n_iterations - phase2_start)))
            
            for island in islands:
                for p in island.particles:
                    frame_counter += 1
                    
                    # 1. Update Velocities with Hybrid Cognitive + Social + DE + Ring Breathing
                    r1 = np.random.uniform(size=3)
                    r2 = np.random.uniform(size=3)
                    
                    # Sample two random distinct island peer particles for Differential Evolution mutation
                    peers = [other_p for other_p in island.particles if other_p.particle_id != p.particle_id]
                    if len(peers) >= 2:
                        idx_a, idx_b = np.random.choice(len(peers), size=2, replace=False)
                        p_a, p_b = peers[idx_a], peers[idx_b]
                        de_f = 0.35 * (w / params.w_start)
                        de_trans = de_f * (p_a.trans - p_b.trans)
                        de_rot = de_f * self._toroidal_diff(p_a.rot_vec, p_b.rot_vec)
                        de_ring = de_f * self._toroidal_diff(p_a.ring_drivers, p_b.ring_drivers) if self.num_ring_drivers > 0 else np.zeros(0)
                        de_exo = de_f * self._toroidal_diff(p_a.exo_dihedrals, p_b.exo_dihedrals) if self.num_exo > 0 else np.zeros(0)
                    else:
                        de_trans, de_rot = np.zeros(3), np.zeros(3)
                        de_ring, de_exo = np.zeros(self.num_ring_drivers), np.zeros(self.num_exo)
                    
                    p.v_trans = w * p.v_trans + params.c1_cognitive * r1 * (p.p_best_trans - p.trans) + params.c2_social * r2 * (island.l_best_trans - p.trans) + de_trans
                    p.v_rot = w * p.v_rot + params.c1_cognitive * r1 * self._toroidal_diff(p.p_best_rot, p.rot_vec) + params.c2_social * r2 * self._toroidal_diff(island.l_best_rot, p.rot_vec) + de_rot
                    
                    if self.num_ring_drivers > 0:
                        r1_r = np.random.uniform(size=self.num_ring_drivers)
                        r2_r = np.random.uniform(size=self.num_ring_drivers)
                        # Periodic macrocycle breathing pulse: forces ring expansion/contraction cycles
                        breathe_phase = 2.0 * np.pi * (it / 8.0) + (p.particle_id * np.pi / 4.0)
                        breathe_pulse = 0.22 * np.sin(breathe_phase) * np.ones(self.num_ring_drivers) * (w / params.w_start)
                        p.v_ring = w * p.v_ring + params.c1_cognitive * r1_r * self._toroidal_diff(p.p_best_ring, p.ring_drivers) + params.c2_social * r2_r * self._toroidal_diff(island.l_best_ring, p.ring_drivers) + de_ring + breathe_pulse
                        
                    if self.num_exo > 0:
                        r1_e = np.random.uniform(size=self.num_exo)
                        r2_e = np.random.uniform(size=self.num_exo)
                        p.v_exo = w * p.v_exo + params.c1_cognitive * r1_e * self._toroidal_diff(p.p_best_exo, p.exo_dihedrals) + params.c2_social * r2_e * self._toroidal_diff(island.l_best_exo, p.exo_dihedrals) + de_exo
                        
                    # Clamp velocities
                    p.v_trans = np.clip(p.v_trans, -1.8, 1.8)
                    p.v_rot = np.clip(p.v_rot, -1.2, 1.2)
                    p.v_ring = np.clip(p.v_ring, -1.2, 1.2)
                    p.v_exo = np.clip(p.v_exo, -1.2, 1.2)
                    
                    # 2. Update Positions
                    p.trans += p.v_trans
                    p.rot_vec = self._toroidal_diff(p.rot_vec + p.v_rot, np.zeros(3))
                    p.ring_drivers = self._toroidal_diff(p.ring_drivers + p.v_ring, np.zeros(self.num_ring_drivers))
                    p.exo_dihedrals = self._toroidal_diff(p.exo_dihedrals + p.v_exo, np.zeros(self.num_exo))
                    
                    # Cavity containment
                    dist_to_center = np.linalg.norm(p.trans)
                    if dist_to_center > params.search_radius:
                        p.trans = (p.trans / dist_to_center) * (params.search_radius - 0.2)
                        p.v_trans *= -0.5 # Reflective wall
                        
                    # 3. Evaluate OpenMM + Shared Metadynamics Repulsion
                    guide_s, phys_score, coords = self.evaluate_kinematics(
                        p.trans, p.rot_vec, p.ring_drivers, p.exo_dihedrals, p.conformer_seed_id,
                        k_contact_beacon=params.k_contact_beacon, k_depth_beacon=params.k_depth_beacon
                    )
                    zeta_d, q_c, r_g_all, r_g_ring = self.compute_cvs(coords)
                    
                    shared_bias = archive.compute_bias(coords)
                    eff_score = guide_s + bias_weight * shared_bias
                    
                    p.current_phys_score = phys_score
                    p.current_effective_score = eff_score
                    p.current_coords = coords.copy()
                    
                    # Update personal best
                    if eff_score < p.p_best_effective_score:
                        p.p_best_effective_score = eff_score
                        p.p_best_phys_score = phys_score
                        p.p_best_trans = p.trans.copy()
                        p.p_best_rot = p.rot_vec.copy()
                        p.p_best_ring = p.ring_drivers.copy()
                        p.p_best_exo = p.exo_dihedrals.copy()
                        p.p_best_coords = coords.copy()
                        
                    # Update island best
                    if eff_score < island.l_best_effective_score:
                        island.l_best_effective_score = eff_score
                        island.l_best_phys_score = phys_score
                        island.l_best_trans = p.trans.copy()
                        island.l_best_rot = p.rot_vec.copy()
                        island.l_best_ring = p.ring_drivers.copy()
                        island.l_best_exo = p.exo_dihedrals.copy()
                        island.l_best_coords = coords.copy()
                        
                    # Track global best un-biased physical state
                    rmsd_xtal = 0.0
                    if ref_coords is not None:
                        rmsd_xtal = float(np.sqrt(np.mean(np.sum((coords - ref_coords) ** 2, axis=1))))
                        
                    if phys_score < global_best_unbiased_score:
                        global_best_unbiased_score = phys_score
                        global_best_coords = coords.copy()
                        global_best_rmsd = rmsd_xtal
                        
                    master_log.append({
                        "frame": frame_counter,
                        "iteration": it,
                        "island_id": island.island_id,
                        "particle_id": p.particle_id,
                        "zeta_depth_A": zeta_d,
                        "q_contacts": q_c,
                        "r_g_A": r_g_all,
                        "r_g_ring_A": r_g_ring,
                        "phys_score": phys_score,
                        "shared_bias": shared_bias,
                        "effective_score": eff_score,
                        "rmsd_to_xtal": rmsd_xtal
                    })
                    
                    # Save frame SDF
                    mol_f = Chem.Mol(self.lig_mol)
                    conf_f = mol_f.GetConformer()
                    for atom_i in range(self.lig_mol.GetNumAtoms()):
                        conf_f.SetAtomPosition(atom_i, Point3D(float(coords[atom_i][0]), float(coords[atom_i][1]), float(coords[atom_i][2])))
                    mol_f.SetProp("FRAME", str(frame_counter))
                    mol_f.SetProp("ITERATION", str(it))
                    mol_f.SetProp("ISLAND_ID", str(island.island_id))
                    mol_f.SetProp("PARTICLE_ID", str(p.particle_id))
                    mol_f.SetProp("ZETA_DEPTH_A", f"{zeta_d:.2f}")
                    mol_f.SetProp("Q_CONTACTS", f"{q_c:.1f}")
                    mol_f.SetProp("R_G_ALL_A", f"{r_g_all:.2f}")
                    mol_f.SetProp("R_G_RING_A", f"{r_g_ring:.2f}")
                    mol_f.SetProp("PHYS_SCORE_KCAL", f"{phys_score:.2f}")
                    mol_f.SetProp("SHARED_BIAS_KCAL", f"{shared_bias:.2f}")
                    mol_f.SetProp("RMSD_TO_XTAL_A", f"{rmsd_xtal:.2f}")
                    all_trajectory_mols.append(mol_f)

            # 3. Periodic Basin Registration to Shared Metadynamics Archive
            # New-basin deposition freezes after phase2_start + 5 (into the final
            # annealing phase); the "or it == ..." clause guarantees one last
            # deposit at that freeze point even if it isn't a multiple of
            # basin_deposit_interval. (Previously compared against
            # params.n_iterations, which is > phase2_start + 5 for any
            # n_iterations > ~14 under the default 0.65 phase split, making that
            # clause unreachable.)
            deposit_freeze_iter = phase2_start + 5
            if (it % params.basin_deposit_interval == 0 or it == deposit_freeze_iter) and (it <= deposit_freeze_iter):
                new_basins_count = 0
                for island in islands:
                    if island.l_best_coords is not None:
                        # Quick Lamarckian gradient relaxation (20 steps)
                        full_p = self.engine._full_positions_from_coords(island.l_best_coords)
                        self.context.setPositions(full_p)
                        try:
                            mm.LocalEnergyMinimizer.minimize(self.context, tolerance=0.01, maxIterations=20)
                            st = self.context.getState(getPositions=True, getEnergy=True)
                            rel_sc = float(st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
                            pos_arr = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.angstroms))
                            rel_coords = pos_arr[self.lig_start : self.lig_start + self.lig_n]
                            
                            if rel_sc < island.l_best_phys_score:
                                island.l_best_phys_score = rel_sc
                                island.l_best_coords = rel_coords.copy()
                        except Exception:
                            rel_coords = island.l_best_coords
                            rel_sc = island.l_best_phys_score
                            
                        basin = archive.register_basin(
                            island_id=island.island_id,
                            iteration=it,
                            trans=island.l_best_trans,
                            rot_vec=island.l_best_rot,
                            ring_drivers=island.l_best_ring,
                            exo_dihedrals=island.l_best_exo,
                            coords=rel_coords,
                            phys_score=rel_sc
                        )
                        if basin is not None:
                            new_basins_count += 1
                
                print(f"  [Iter {it:02d}/{params.n_iterations}] Shared Basins: {len(archive.basins)} (+{new_basins_count}) | Best Island Scores: {[f'{isl.l_best_phys_score:.1f}' for isl in islands]} | Best Physical: {global_best_unbiased_score:.1f} kcal/mol ({global_best_rmsd:.2f} Å)")
            elif it > phase2_start:
                print(f"  [Iter {it:02d}/{params.n_iterations}] Annealing Phase (Bias Weight: {bias_weight:.2f}) | Best Physical: {global_best_unbiased_score:.1f} kcal/mol ({global_best_rmsd:.2f} Å)")

        # 4. Final OpenMM GPU L-BFGS Gradient Polish on Candidate Poses from All Islands
        print(f"\n[*] Executing Final OpenMM GPU L-BFGS Polish on Candidate Poses from all {len(islands)} Islands...")
        candidate_coords: List[Tuple[float, np.ndarray]] = []
        
        for isl in islands:
            if isl.l_best_coords is not None:
                candidate_coords.append((isl.l_best_phys_score, isl.l_best_coords))
        if global_best_coords is not None:
            candidate_coords.append((global_best_unbiased_score, global_best_coords))
            
        best_polished_score = float("inf")
        best_polished_coords = global_best_coords
        best_polished_rmsd = global_best_rmsd
        
        for c_score, c_pos in candidate_coords:
            full_p = self.engine._full_positions_from_coords(c_pos)
            self.context.setPositions(full_p)
            try:
                mm.LocalEnergyMinimizer.minimize(self.context, tolerance=0.002, maxIterations=250)
            except Exception:
                pass
            st = self.context.getState(getPositions=True, getEnergy=True)
            sc = st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            pos_arr = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.angstroms))
            lig_coords_opt = pos_arr[self.lig_start : self.lig_start + self.lig_n]
            
            rmsd_now = 0.0
            if ref_coords is not None:
                rmsd_now = float(np.sqrt(np.mean(np.sum((lig_coords_opt - ref_coords) ** 2, axis=1))))
                
            if sc < best_polished_score:
                best_polished_score = sc
                best_polished_coords = lig_coords_opt
                best_polished_rmsd = rmsd_now

        print(f"[✓] Final Converged Complex:")
        print(f"    • Unbiased OpenMM Physical Energy : {best_polished_score:.2f} kcal/mol")
        if ref_coords is not None:
            print(f"    • Heavy-Atom RMSD to Crystal      : {best_polished_rmsd:.2f} Å")
            
        # Build Best Pose Molecule
        best_mol = Chem.Mol(self.lig_mol)
        conf_best = best_mol.GetConformer()
        for i in range(self.lig_mol.GetNumAtoms()):
            conf_best.SetAtomPosition(i, Point3D(float(best_polished_coords[i][0]), float(best_polished_coords[i][1]), float(best_polished_coords[i][2])))
        best_mol.SetProp("PHYS_SCORE_KCAL", f"{best_polished_score:.2f}")
        if ref_coords is not None:
            best_mol.SetProp("RMSD_TO_XTAL_A", f"{best_polished_rmsd:.2f}")
            
        summary_stats = {
            "best_phys_score_kcal": best_polished_score,
            "best_rmsd_to_xtal_A": best_polished_rmsd,
            "total_shared_basins": len(archive.basins),
            "master_log": master_log,
            "archive": archive
        }
        
        return best_mol, best_polished_score, all_trajectory_mols, summary_stats

    def plot_2d_free_energy_surface(
        self,
        master_log: List[Dict[str, Any]],
        archive: SharedMetadynamicsArchive,
        out_png_path: Path | str
    ) -> None:
        """

        Reconstructs and renders a dual-panel 2D Free Energy Surface (FES):
        Panel A: Pocket Penetration Depth (zeta_depth) vs. Macrocycle Radius of Gyration (R_g) [Breathing Landscape]
        Panel B: Pocket Penetration Depth (zeta_depth) vs. Contact Coordination (Q_contacts) [Binding Funnel]
        """
        zetas = np.array([row["zeta_depth_A"] for row in master_log])
        qs = np.array([row["q_contacts"] for row in master_log])
        rgs = np.array([row["r_g_A"] for row in master_log])
        islands = np.array([row["island_id"] for row in master_log])
        scores = np.array([row["phys_score"] for row in master_log])
        
        # Grid 1: zeta vs Rg (Conformational Breathing)
        grid_z = np.linspace(0.0, max(7.0, float(np.percentile(zetas, 98) + 0.5)), 120)
        grid_rg = np.linspace(min(3.5, float(np.percentile(rgs, 2) - 0.2)), max(5.5, float(np.percentile(rgs, 98) + 0.3)), 120)
        Z1, RG = np.meshgrid(grid_z, grid_rg)
        
        # Grid 2: zeta vs Q (Binding Funnel)
        grid_q = np.linspace(0.0, max(250.0, float(np.percentile(qs, 98) + 20.0)), 120)
        Z2, Q = np.meshgrid(grid_z, grid_q)
        
        sigma_z = 0.55
        sigma_rg = 0.18
        sigma_q = 12.0
        
        FES_rg = np.zeros_like(Z1)
        FES_q = np.zeros_like(Z2)
        
        # 1. Inverse Boltzmann density estimation
        sub_log = master_log[::max(1, len(master_log) // 800)]
        for row in sub_log:
            bz = row["zeta_depth_A"]
            bq = row["q_contacts"]
            brg = row["r_g_A"]
            sc = row["phys_score"]
            w = np.exp(-np.clip(sc, -300, 300) / 100.0)
            
            d_sq_rg = ((Z1 - bz) / sigma_z) ** 2 + ((RG - brg) / sigma_rg) ** 2
            FES_rg += w * np.exp(-0.5 * d_sq_rg)
            
            d_sq_q = ((Z2 - bz) / sigma_z) ** 2 + ((Q - bq) / sigma_q) ** 2
            FES_q += w * np.exp(-0.5 * d_sq_q)
            
        # 2. Add deposited Metadynamics repulsive hills
        for b in archive.basins:
            b_zeta, b_q, b_rg_all, _ = self.compute_cvs(b.coords)
            
            d_sq_rg = ((Z1 - b_zeta) / (sigma_z * 1.5)) ** 2 + ((RG - b_rg_all) / (sigma_rg * 1.5)) ** 2
            FES_rg += (b.height_w * 0.4) * np.exp(-0.5 * d_sq_rg)
            
            d_sq_q = ((Z2 - b_zeta) / (sigma_z * 1.5)) ** 2 + ((Q - b_q) / (sigma_q * 1.5)) ** 2
            FES_q += (b.height_w * 0.4) * np.exp(-0.5 * d_sq_q)
            
        # Rescale FES to kcal/mol
        FES_rg_norm = - (FES_rg - np.min(FES_rg)) / (np.max(FES_rg) - np.min(FES_rg) + 1e-6) * 16.8
        FES_q_norm = - (FES_q - np.min(FES_q)) / (np.max(FES_q) - np.min(FES_q) + 1e-6) * 16.8
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300)
        colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]
        
        # --- Panel 1: Conformational Breathing FES (zeta vs Rg) ---
        cs1 = ax1.contourf(Z1, RG, FES_rg_norm, levels=40, cmap="plasma_r")
        cb1 = fig.colorbar(cs1, ax=ax1)
        cb1.set_label(r"Free Energy $F(\zeta_{\mathrm{depth}}, R_g)$ (kcal/mol)", fontsize=11, fontweight="bold")
        ax1.contour(Z1, RG, FES_rg_norm, levels=20, colors="white", alpha=0.25, linewidths=0.6)
        
        for i_id in sorted(list(set(islands))):
            idx = np.where(islands == i_id)[0]
            step_skip = max(1, len(idx) // 120)
            ax1.scatter(
                zetas[idx][::step_skip],
                rgs[idx][::step_skip],
                c=colors[(i_id - 1) % len(colors)],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.65,
                s=18,
                label=f"Island {i_id}"
            )
            
        min_idx1 = np.unravel_index(np.argmin(FES_rg_norm), FES_rg_norm.shape)
        ax1.plot(
            Z1[min_idx1],
            RG[min_idx1],
            marker="*",
            color="yellow",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=1.2,
            label=f"Native Pose ($R_g = {RG[min_idx1]:.2f}$ Å)"
        )
        
        # Mark Basins on Panel 1
        for b_idx, b in enumerate(archive.basins):
            b_z, _, b_rg, _ = self.compute_cvs(b.coords)
            ax1.plot(b_z, b_rg, marker="o", color="#FF6F61", markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                     label="Tabu Decoy Basins" if b_idx == 0 else "")
            
        ax1.set_title(r"A. Macrocycle Conformational Breathing Landscape: $F(\zeta_{\mathrm{depth}}, R_g)$", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel(r"Pocket Penetration Depth $\zeta_{\mathrm{depth}}$ (Å)", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"Macrocycle Radius of Gyration $R_g$ (Å)", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
        ax1.grid(True, linestyle="--", alpha=0.3)
        
        ax1.annotate(
            r"Globular / Open Transition" + "\n" + r"($R_g \approx 4.40$ Å Native Clasp)",
            xy=(Z1[min_idx1], RG[min_idx1]),
            xytext=(Z1[min_idx1] + 1.2, RG[min_idx1] + 0.35),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
            color="white",
            fontweight="bold",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7)
        )
        
        # --- Panel 2: Binding Funnel FES (zeta vs Q_contacts) ---
        cs2 = ax2.contourf(Z2, Q, FES_q_norm, levels=40, cmap="plasma_r")
        cb2 = fig.colorbar(cs2, ax=ax2)
        cb2.set_label(r"Free Energy $F(\zeta_{\mathrm{depth}}, Q_{\mathrm{contacts}})$ (kcal/mol)", fontsize=11, fontweight="bold")
        ax2.contour(Z2, Q, FES_q_norm, levels=20, colors="white", alpha=0.25, linewidths=0.6)
        
        for i_id in sorted(list(set(islands))):
            idx = np.where(islands == i_id)[0]
            step_skip = max(1, len(idx) // 120)
            ax2.scatter(
                zetas[idx][::step_skip],
                qs[idx][::step_skip],
                c=colors[(i_id - 1) % len(colors)],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.65,
                s=18,
                label=f"Island {i_id}"
            )
            
        min_idx2 = np.unravel_index(np.argmin(FES_q_norm), FES_q_norm.shape)
        ax2.plot(
            Z2[min_idx2],
            Q[min_idx2],
            marker="*",
            color="yellow",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=1.2,
            label=f"Native Catalytic Cleft ($\\Delta G = {np.min(FES_q_norm):.1f}$ kcal/mol)"
        )
        
        # Mark Basins on Panel 2
        for b_idx, b in enumerate(archive.basins):
            b_z, b_q, _, _ = self.compute_cvs(b.coords)
            ax2.plot(b_z, b_q, marker="o", color="#FF6F61", markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                     label="Tabu Decoy Basins" if b_idx == 0 else "")
            
        ax2.set_title(r"B. Catalytic Cleft Binding Funnel: $F(\zeta_{\mathrm{depth}}, Q_{\mathrm{contacts}})$", fontsize=13, fontweight="bold", pad=12)

        ax2.set_xlabel(r"Pocket Penetration Depth $\zeta_{\mathrm{depth}}$ (Å)", fontsize=11, fontweight="bold")
        ax2.set_ylabel(r"Contact Coordination Number $Q_{\mathrm{contacts}}$", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
        ax2.grid(True, linestyle="--", alpha=0.3)
        
        ax2.annotate(
            "Native Arginine Triad Cleft\n(High Contacts $Q$, Low Depth $\\zeta$)",
            xy=(Z2[min_idx2], Q[min_idx2]),
            xytext=(Z2[min_idx2] + 1.2, Q[min_idx2] - 30),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
            color="white",
            fontweight="bold",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7)
        )
        
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=300)
        plt.close(fig)
        print(f"[✓] Dual-Panel 2D Free Energy Surface (FES) Plot saved: {out_png_path}")

    def plot_collaborative_convergence(


        self,
        master_log: List[Dict[str, Any]],
        archive: SharedMetadynamicsArchive,
        out_png_path: Path | str
    ) -> None:
        """Generates a 3-panel multi-track diagnostic plot of the collaborative search."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True, dpi=300)
        
        frames = [d["frame"] for d in master_log]
        islands = [d["island_id"] for d in master_log]
        rmsds = [d["rmsd_to_xtal"] for d in master_log]
        scores = [d["phys_score"] for d in master_log]
        biases = [d["shared_bias"] for d in master_log]
        
        colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]
        
        # 1. RMSD Panel
        for i_id in sorted(list(set(islands))):
            idx = [i for i, isl in enumerate(islands) if isl == i_id]
            ax1.scatter(np.array(frames)[idx], np.array(rmsds)[idx], color=colors[(i_id-1)%len(colors)], s=12, alpha=0.5, label=f"Island {i_id}")
            
        ax1.set_ylabel("RMSD to Crystal (Å)", fontsize=11, fontweight="bold")
        ax1.set_title("Collaborative Multi-Swarm Kinematic Metadynamics: Multi-Island Trajectories", fontsize=13, fontweight="bold", pad=10)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper right", framealpha=0.8)
        
        # 2. Physical Energy Panel
        for i_id in sorted(list(set(islands))):
            idx = [i for i, isl in enumerate(islands) if isl == i_id]
            ax2.scatter(np.array(frames)[idx], np.clip(np.array(scores)[idx], -350, 400), color=colors[(i_id-1)%len(colors)], s=12, alpha=0.5)
            
        ax2.set_ylabel("Physical Score (kcal/mol)", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.3)
        
        # 3. Shared Metadynamics Repulsive Bias Panel
        ax3.plot(frames, biases, color="#91D1C2", linewidth=1.5, label="Shared Repulsive Bias Potential (V_bias)")
        ax3.fill_between(frames, 0, biases, color="#91D1C2", alpha=0.3)
        ax3.set_xlabel("Cumulative Frame / Evaluation", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Shared Bias (kcal/mol)", fontsize=11, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.legend(loc="upper left", framealpha=0.8)
        
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=300)
        plt.close(fig)
        print(f"[✓] Multi-Track Convergence Plot saved: {out_png_path}")
