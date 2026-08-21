"""
Macrocycle Kinematic Metadynamics Collective Variable (CV) Benchmark Engine.
Compares enhanced sampling efficiency across:
1. Radius of Gyration (Rg)
2. Principal Moments of Inertia / Relative Shape Anisotropy (kappa^2 / PMI)
3. Macrocyclic Ring Puckering Amplitude (Q_puck)
4. Coupled 2D (Rg + kappa^2)
5. Unbiased Kinematic Baseline
"""
from __future__ import annotations
from dataclasses import dataclass
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

from openmm_dock.inverse_kinematics import TwoTierMacrocycleEngine
from openmm_dock.engine import DockingEngine


@dataclass
class ConformationalCVState:
    """Stores the multi-dimensional CV coordinates of a macrocycle conformation."""
    r_g_total: float              # Total Radius of Gyration (Å)
    r_g_ring: float               # Macrocycle Core Ring Radius of Gyration (Å)
    pmi_eigenvals: np.ndarray     # (3,) Principal Moments of Inertia I1 <= I2 <= I3
    npr1: float                   # Normalized PMI ratio I1 / I3
    npr2: float                   # Normalized PMI ratio I2 / I3
    kappa_sq: float               # Relative Shape Anisotropy (0=spherical, 1=linear)
    asphericity: float            # Asphericity (Δ)
    q_puck: float                 # Ring Puckering Amplitude (Å)
    q_intra: float                # Intramolecular contact fraction


class MacrocycleCVCalculator:
    """Computes exact 3D conformational Collective Variables for macrocyclic ligands."""
    def __init__(self, mol: Chem.Mol):
        self.mol = Chem.Mol(mol)
        self.num_atoms = mol.GetNumAtoms()
        
        # Identify macrocycle ring atoms
        rings = [list(r) for r in mol.GetRingInfo().AtomRings() if len(r) >= 9]
        if not rings:
            rings = [list(r) for r in mol.GetRingInfo().AtomRings()]
            rings.sort(key=len, reverse=True)
        self.ring_atoms = rings[0] if rings else list(range(self.num_atoms))
        
        # Identify non-bonded intramolecular atom pairs (|i - j| >= 4)
        self.intra_pairs = []
        for i in range(self.num_atoms):
            for j in range(i + 4, self.num_atoms):
                self.intra_pairs.append((i, j))

    def compute_all_cvs(self, coords: np.ndarray) -> ConformationalCVState:
        # 1. Radius of Gyration (Total)
        com = np.mean(coords, axis=0)
        c = coords - com
        rg_all = float(np.sqrt(np.mean(np.sum(c ** 2, axis=1))))
        
        # 2. Macrocycle Ring Rg
        ring_c = coords[self.ring_atoms] - np.mean(coords[self.ring_atoms], axis=0)
        rg_ring = float(np.sqrt(np.mean(np.sum(ring_c ** 2, axis=1))))
        
        # 3. Principal Moments of Inertia (PMI)
        Ixx = np.sum(c[:, 1]**2 + c[:, 2]**2)
        Iyy = np.sum(c[:, 0]**2 + c[:, 2]**2)
        Izz = np.sum(c[:, 0]**2 + c[:, 1]**2)
        Ixy = -np.sum(c[:, 0] * c[:, 1])
        Ixz = -np.sum(c[:, 0] * c[:, 2])
        Iyz = -np.sum(c[:, 1] * c[:, 2])
        
        I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        eigvals = np.sort(np.linalg.eigvalsh(I))
        I1, I2, I3 = max(1e-6, eigvals[0]), max(1e-6, eigvals[1]), max(1e-6, eigvals[2])
        
        npr1 = float(I1 / I3)
        npr2 = float(I2 / I3)
        I_mean = np.mean(eigvals)
        kappa_sq = float(1.5 * np.sum((eigvals - I_mean)**2) / (np.sum(eigvals)**2 + 1e-9))
        asphericity = float(((I1 - I2)**2 + (I2 - I3)**2 + (I3 - I1)**2) / (2 * (I1 + I2 + I3)**2 + 1e-9))
        
        # 4. Ring Puckering Amplitude (Q_puck)
        _, _, vh = np.linalg.svd(ring_c)
        normal = vh[2]
        z_displacements = np.dot(ring_c, normal)
        q_puck = float(np.sqrt(np.mean(z_displacements ** 2)))
        
        # 5. Intramolecular Compactness (Q_intra)
        if self.intra_pairs:
            p_i = [p[0] for p in self.intra_pairs]
            p_j = [p[1] for p in self.intra_pairs]
            dists = np.linalg.norm(coords[p_i] - coords[p_j], axis=1)
            q_intra = float(np.mean(1.0 / (1.0 + (dists / 4.5) ** 6)))
        else:
            q_intra = 0.0
            
        return ConformationalCVState(
            r_g_total=rg_all,
            r_g_ring=rg_ring,
            pmi_eigenvals=eigvals,
            npr1=npr1,
            npr2=npr2,
            kappa_sq=kappa_sq,
            asphericity=asphericity,
            q_puck=q_puck,
            q_intra=q_intra
        )


@dataclass
class MetadynamicsHill:
    """Represents a 1D or 2D Gaussian repulsive hill deposited in CV space."""
    iteration: int
    cv_values: np.ndarray        # Coordinates in chosen CV space
    height_w: float              # Well-Tempered hill height in kcal/mol
    sigmas: np.ndarray           # Width per CV dimension


class MacrocycleKinematicSampler:
    """
    Simulates macrocycle conformational exploration using Kinematic Metadynamics.
    Evaluates 13D internal conformational kinematics (4 Ring IK Drivers + 9 Exocyclic FK Dihedrals).
    """
    def __init__(self, ligand_mol: Chem.Mol, reference_xtal_mol: Optional[Chem.Mol] = None):
        self.lig_mol = Chem.Mol(ligand_mol)
        self.two_tier = TwoTierMacrocycleEngine(ligand_mol)
        self.cv_calc = MacrocycleCVCalculator(ligand_mol)
        
        # Driver Joint selection
        num_joints = len(self.two_tier.ik_engine.joints)
        if num_joints >= 4:
            self.driver_joint_indices = [1, 3, 5, min(8, num_joints - 1)]
        elif num_joints > 0:
            self.driver_joint_indices = [0]
        else:
            self.driver_joint_indices = []
            
        self.num_ring_drivers = len(self.driver_joint_indices)
        self.num_exo = len(self.two_tier.exo_joints)
        self.num_internal_dofs = self.num_ring_drivers + self.num_exo
        
        # Reference Crystal Coords
        self.ref_coords = None
        if reference_xtal_mol is not None:
            conf_ref = reference_xtal_mol.GetConformer()
            self.ref_coords = np.array([conf_ref.GetAtomPosition(i) for i in range(reference_xtal_mol.GetNumAtoms())])
            # Center ref coords at origin
            self.ref_coords -= np.mean(self.ref_coords, axis=0)

        # Build Isolated OpenMM Ligand Energy System
        self.system = self._build_openmm_ligand_system(self.lig_mol)
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            plat = mm.Platform.getPlatformByName("OpenCL")
            self.context = mm.Context(self.system, self.integrator, plat)
        except Exception:
            self.context = mm.Context(self.system, self.integrator)

    def _build_openmm_ligand_system(self, mol: Chem.Mol) -> mm.System:
        """Builds a fast OpenMM nonbonded-only internal strain potential.
        Bond lengths/angles are not modeled here since the IK/FK kinematics
        engine enforces valid bonded geometry by construction; only nonbonded
        steric clash beyond 1-4 is scored.
        """
        system = mm.System()
        num_atoms = mol.GetNumAtoms()
        for _ in range(num_atoms):
            system.addParticle(12.0)

        # Nonbonded force for intramolecular sterics & electrostatics
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        for i in range(num_atoms):
            nb.addParticle(0.0, 0.35, 0.20)

        # Fully exclude 1-2/1-3 pairs and zero out 1-4 pairs by walking the
        # bond graph (a manual GetBonds()-only loop only catches 1-2 pairs,
        # leaving 1-3 angle-distance pairs ~2.3-2.5 A apart unexcluded --
        # deep inside the sigma=3.5 A repulsive wall, injecting ~1000+
        # kcal/mol of spurious topology-only "clash" into every evaluation).
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        nb.createExceptionsFromBonds(bonds, 0.0, 0.0)

        system.addForce(nb)
        return system

    def generate_randomized_etkdg_conformers(self, n_confs: int = 16) -> List[np.ndarray]:
        """Generates completely randomized, diverse 3D macrocyclic conformers from ETKDGv3."""
        mol_work = Chem.Mol(self.lig_mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(np.random.randint(1, 1000000))
        params.useRandomCoords = True
        params.numThreads = 0
        cids = AllChem.EmbedMultipleConfs(mol_work, numConfs=n_confs, params=params)
        confs = []
        for cid in cids:
            conf = mol_work.GetConformer(cid)
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol_work.GetNumAtoms())])
            coords -= np.mean(coords, axis=0)
            confs.append(coords)
        if not confs:
            confs = [self.two_tier.base_coords]
        return confs

    def evaluate_internal_state(
        self,
        ring_drivers: np.ndarray,
        exo_dihedrals: np.ndarray,
        base_coords: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, ConformationalCVState]:
        """
        Solves Macrocycle Ring IK + FK and evaluates internal physical energy and CVs.
        """
        base_c = base_coords if base_coords is not None else self.two_tier.base_coords
        
        # 1. Solve Ring IK
        if self.num_ring_drivers > 0:
            d_dict = {self.driver_joint_indices[i]: float(ring_drivers[i]) for i in range(self.num_ring_drivers)}
            c_lig, _, _ = self.two_tier.ik_engine.solve_loop_closure(base_c, driver_angles=d_dict)
        else:
            c_lig = base_c.copy()
            
        # 2. Exocyclic FK
        for j_idx in range(self.num_exo):
            c_lig = self.two_tier.apply_exocyclic_rotation(c_lig, j_idx, float(exo_dihedrals[j_idx]))
            
        # Center coordinates
        c_lig -= np.mean(c_lig, axis=0)
        
        # 3. OpenMM Energy
        pos_vec = [mm.Vec3(c_lig[i][0], c_lig[i][1], c_lig[i][2]) * unit.angstroms for i in range(len(c_lig))]
        self.context.setPositions(pos_vec)
        state = self.context.getState(getEnergy=True)
        phys_score = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        
        # 4. Compute all CVs
        cv_state = self.cv_calc.compute_all_cvs(c_lig)
        
        return phys_score, c_lig, cv_state

    def run_metadynamics_cv_experiment(
        self,
        cv_mode: str = "rg",          # 'unbiased', 'rg', 'kappa_sq', 'qpuck', 'coupled_rg_kappa'
        n_iterations: int = 120,
        n_particles: int = 16,
        w0_height: float = 10.0,
        gamma_well_temper: float = 4.0
    ) -> Dict[str, Any]:
        """
        Executes a Kinematic Metadynamics run along the selected Collective Variable
        with all particles initialized from totally randomized ETKDGv3 conformations.
        """
        print(f"[*] Generating {n_particles} Randomized 3D Conformer Seeds via ETKDGv3...")
        randomized_confs = self.generate_randomized_etkdg_conformers(n_confs=n_particles)
        print(f"[*] Running Experiment: CV Mode = '{cv_mode.upper()}' ({n_particles} Walkers x {n_iterations} Iterations)...")
        
        # Initialize swarm with randomized conformer seeds
        particles = []
        for p_id in range(n_particles):
            seed_c = randomized_confs[p_id % len(randomized_confs)]
            ring = np.random.uniform(-np.pi / 3, np.pi / 3, self.num_ring_drivers)
            exo = np.random.uniform(-np.pi, np.pi, self.num_exo)
            v_ring = np.random.normal(scale=0.2, size=self.num_ring_drivers)
            v_exo = np.random.normal(scale=0.2, size=self.num_exo)
            
            sc, coords, cv_s = self.evaluate_internal_state(ring, exo, base_coords=seed_c)
            particles.append({
                "id": p_id + 1,
                "seed_coords": seed_c,
                "ring": ring, "exo": exo,
                "v_ring": v_ring, "v_exo": v_exo,
                "p_best_ring": ring.copy(), "p_best_exo": exo.copy(),
                "p_best_score": sc,
                "phys_score": sc,
                "coords": coords,
                "cv_state": cv_s
            })


        hills: List[MetadynamicsHill] = []
        exp_log: List[Dict[str, Any]] = []
        conformations_explored: List[np.ndarray] = []
        frame_idx = 0
        
        # Define CV mapping functions
        def get_cv_vector(cv_s: ConformationalCVState) -> np.ndarray:
            if cv_mode == "rg":
                return np.array([cv_s.r_g_total])
            elif cv_mode == "kappa_sq":
                return np.array([cv_s.kappa_sq])
            elif cv_mode == "qpuck":
                return np.array([cv_s.q_puck])
            elif cv_mode == "coupled_rg_kappa":
                return np.array([cv_s.r_g_total, cv_s.kappa_sq])
            else:
                return np.zeros(1)

        def get_cv_sigmas() -> np.ndarray:
            if cv_mode == "rg":
                return np.array([0.15])          # 0.15 Å
            elif cv_mode == "kappa_sq":
                return np.array([0.035])         # 0.035 anisotropy
            elif cv_mode == "qpuck":
                return np.array([0.06])          # 0.06 Å puckering
            elif cv_mode == "coupled_rg_kappa":
                return np.array([0.15, 0.035])
            else:
                return np.array([1.0])

        sigmas = get_cv_sigmas()
        k_B_T = 0.596 # kcal/mol at 300K
        delta_T = (gamma_well_temper - 1.0) * 300.0
        k_B_delta_T = max(1e-3, k_B_T * (gamma_well_temper - 1.0))
        
        min_rmsd_to_crystal = float("inf")
        min_rmsd_coords = None
        min_rmsd_iter = -1
        
        for it in range(1, n_iterations + 1):
            w_inertia = 0.78 - (0.78 - 0.28) * (it / n_iterations)
            
            for p in particles:
                frame_idx += 1
                
                # 1. Update velocities with DE peer mutation
                other_idx = np.random.choice([x for x in range(n_particles) if x != p["id"] - 1])
                other_p = particles[other_idx]
                
                de_ring = 0.3 * np.arctan2(np.sin(p["ring"] - other_p["ring"]), np.cos(p["ring"] - other_p["ring"]))
                de_exo = 0.3 * np.arctan2(np.sin(p["exo"] - other_p["exo"]), np.cos(p["exo"] - other_p["exo"]))
                
                r1, r2 = np.random.uniform(size=2)
                diff_ring = np.arctan2(np.sin(p["p_best_ring"] - p["ring"]), np.cos(p["p_best_ring"] - p["ring"]))
                diff_exo = np.arctan2(np.sin(p["p_best_exo"] - p["exo"]), np.cos(p["p_best_exo"] - p["exo"]))
                
                p["v_ring"] = w_inertia * p["v_ring"] + 1.4 * r1 * diff_ring + de_ring
                p["v_exo"] = w_inertia * p["v_exo"] + 1.4 * r2 * diff_exo + de_exo
                
                # Clamp
                p["v_ring"] = np.clip(p["v_ring"], -1.2, 1.2)
                p["v_exo"] = np.clip(p["v_exo"], -1.2, 1.2)
                
                # 2. Update Dihedrals
                p["ring"] = (p["ring"] + p["v_ring"] + np.pi) % (2 * np.pi) - np.pi
                p["exo"] = (p["exo"] + p["v_exo"] + np.pi) % (2 * np.pi) - np.pi
                
                # 3. Evaluate State
                phys_sc, coords, cv_s = self.evaluate_internal_state(p["ring"], p["exo"], base_coords=p["seed_coords"])
                conformations_explored.append(coords)

                
                # 4. Compute Metadynamics Bias
                cv_vec = get_cv_vector(cv_s)
                bias_val = 0.0
                if cv_mode != "unbiased" and hills:
                    for h in hills:
                        d_norm_sq = np.sum(((cv_vec - h.cv_values) / h.sigmas) ** 2)
                        bias_val += h.height_w * np.exp(-0.5 * d_norm_sq)
                        
                eff_score = phys_sc + bias_val
                p["phys_score"] = phys_sc
                p["coords"] = coords
                p["cv_state"] = cv_s
                
                if eff_score < p["p_best_score"]:
                    p["p_best_score"] = eff_score
                    p["p_best_ring"] = p["ring"].copy()
                    p["p_best_exo"] = p["exo"].copy()
                    
                # RMSD to reference crystal pose (pure internal heavy-atom Kabsch alignment)
                rmsd_xtal = 0.0
                rmsd_ring_xtal = 0.0
                if self.ref_coords is not None:
                    # 1. Total 64-Atom RMSD
                    c_aln = coords - np.mean(coords, axis=0)
                    r_ref = self.ref_coords
                    h_mat = c_aln.T.dot(r_ref)
                    u, s, vt = np.linalg.svd(h_mat)
                    d = np.sign(np.linalg.det(vt.T.dot(u.T)))
                    k_rot = vt.T.dot(np.diag([1, 1, d])).dot(u.T)
                    c_rot = c_aln.dot(k_rot.T)
                    rmsd_xtal = float(np.sqrt(np.mean(np.sum((c_rot - r_ref) ** 2, axis=1))))
                    
                    # 2. Ring Backbone Core RMSD (< 1.0 Å Milestone)
                    ring_ats = self.two_tier.ik_engine.ring_atoms
                    c_ring = coords[ring_ats] - np.mean(coords[ring_ats], axis=0)
                    r_ring = self.ref_coords[ring_ats] - np.mean(self.ref_coords[ring_ats], axis=0)
                    h_r = c_ring.T.dot(r_ring)
                    u_r, _, vt_r = np.linalg.svd(h_r)
                    d_r = np.sign(np.linalg.det(vt_r.T.dot(u_r.T)))
                    k_r = vt_r.T.dot(np.diag([1, 1, d_r])).dot(u_r.T)
                    c_ring_rot = c_ring.dot(k_r.T)
                    rmsd_ring_xtal = float(np.sqrt(np.mean(np.sum((c_ring_rot - r_ring) ** 2, axis=1))))
                    
                    if rmsd_xtal < min_rmsd_to_crystal:
                        min_rmsd_to_crystal = rmsd_xtal
                        min_rmsd_coords = coords.copy()
                        min_rmsd_iter = it
                        
                exp_log.append({
                    "frame": frame_idx,
                    "iteration": it,
                    "particle_id": p["id"],
                    "phys_score": phys_sc,
                    "bias_kcal": bias_val,
                    "effective_score": eff_score,
                    "r_g_total": cv_s.r_g_total,
                    "r_g_ring": cv_s.r_g_ring,
                    "npr1": cv_s.npr1,
                    "npr2": cv_s.npr2,
                    "kappa_sq": cv_s.kappa_sq,
                    "asphericity": cv_s.asphericity,
                    "q_puck": cv_s.q_puck,
                    "q_intra": cv_s.q_intra,
                    "rmsd_to_xtal": rmsd_xtal,
                    "rmsd_ring_xtal": rmsd_ring_xtal
                })


            # 5. Periodic Metadynamics Hill Deposition (every 2 iterations)
            if cv_mode != "unbiased" and it % 2 == 0:
                for p in particles:
                    cv_vec = get_cv_vector(p["cv_state"])
                    # Compute current bias at particle position
                    cur_bias = 0.0
                    for h in hills:
                        d_norm_sq = np.sum(((cv_vec - h.cv_values) / h.sigmas) ** 2)
                        cur_bias += h.height_w * np.exp(-0.5 * d_norm_sq)
                    # Well-Tempered attenuation
                    w_hill = w0_height * np.exp(-cur_bias / k_B_delta_T)
                    if w_hill > 0.05:
                        hills.append(MetadynamicsHill(
                            iteration=it,
                            cv_values=cv_vec.copy(),
                            height_w=w_hill,
                            sigmas=sigmas.copy()
                        ))

        all_rgs = [row["r_g_total"] for row in exp_log]
        all_kappas = [row["kappa_sq"] for row in exp_log]
        all_qpucks = [row["q_puck"] for row in exp_log]
        all_ring_rmsds = [row["rmsd_ring_xtal"] for row in exp_log]
        
        min_ring_rmsd = float(np.min(all_ring_rmsds))
        sub_1_ring_iter = next((row["iteration"] for row in exp_log if row["rmsd_ring_xtal"] < 1.0), None)
        
        # Dihedral phase space volume estimate (standard deviation product)
        rg_span = float(np.percentile(all_rgs, 95) - np.percentile(all_rgs, 5))
        kappa_span = float(np.percentile(all_kappas, 95) - np.percentile(all_kappas, 5))
        qpuck_span = float(np.percentile(all_qpucks, 95) - np.percentile(all_qpucks, 5))
        
        results = {
            "cv_mode": cv_mode,
            "min_rmsd_to_crystal_A": min_rmsd_to_crystal,
            "min_rmsd_discovered_iter": min_rmsd_iter,
            "min_ring_rmsd_A": min_ring_rmsd,
            "sub_1_ring_iter": sub_1_ring_iter,
            "min_rmsd_coords": min_rmsd_coords,
            "total_hills_deposited": len(hills),
            "rg_span_A": rg_span,
            "kappa_span": kappa_span,
            "qpuck_span_A": qpuck_span,
            "exp_log": exp_log,
            "hills": hills
        }
        
        print(f"[✓] Completed '{cv_mode.upper()}': Total RMSD = {min_rmsd_to_crystal:.2f} Å | Ring Backbone Core = {min_ring_rmsd:.3f} Å (<1.0Å at Iter #{sub_1_ring_iter}) | Hills = {len(hills)}")
        return results

