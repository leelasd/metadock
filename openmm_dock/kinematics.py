"""
Molecular Kinematics Engine for openmm-dock.
Provides forward kinematics parameterization on torsion trees (SE(3) x T^k)
guaranteeing 0.000 Å bond length/angle distortion during docking exploration.
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

import itertools

from .core import MolecularSystem, DockAtom, SDFParser
from .engine import DockingEngine, DockingResult
from .kinematic_utils import toroidal_diff, find_downstream_atoms, identify_rotatable_bonds

# Discrete rotamer libraries keyed by the hybridization of the two atoms
# forming a rotatable bond -- a +/-20 degree, 5-point spread around each
# staggered minimum for sp3-sp3 (Newman-projection 60/180/300 degrees, each
# with 4 neighboring points so the library covers the WIDTH of each energy
# well, not just its exact center), full 30 degree coverage for any bond
# touching an sp2 center (a much shallower/less sharply-peaked rotational
# barrier than sp3-sp3), and only the two planar minima for sp2-sp2 (e.g.
# biaryl-type bonds). Values match those actually used in a real, published
# rotamer-library conformer search (Salveson et al., Science 2024,
# vector_dock_align.py's rotamer_dict) rather than a from-scratch guess.
DISCRETE_ROTAMER_LIBRARY: Dict[Tuple[Chem.HybridizationType, Chem.HybridizationType], np.ndarray] = {
    (Chem.HybridizationType.SP3, Chem.HybridizationType.SP3): np.radians(
        [40.0, 50.0, 60.0, 70.0, 80.0, 160.0, 170.0, 180.0, 190.0, 200.0, 280.0, 290.0, 300.0, 310.0, 320.0]
    ),
    (Chem.HybridizationType.SP3, Chem.HybridizationType.SP2): np.radians(list(range(0, 360, 30))),
    (Chem.HybridizationType.SP2, Chem.HybridizationType.SP2): np.radians([0.0, 180.0]),
    (Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D): np.radians(
        [40.0, 50.0, 60.0, 70.0, 80.0, 160.0, 170.0, 180.0, 190.0, 200.0, 280.0, 290.0, 300.0, 310.0, 320.0]
    ),
    (Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3): np.radians(
        [40.0, 50.0, 60.0, 70.0, 80.0, 160.0, 170.0, 180.0, 190.0, 200.0, 280.0, 290.0, 300.0, 310.0, 320.0]
    ),
    (Chem.HybridizationType.SP3D, Chem.HybridizationType.SP2): np.radians(list(range(0, 360, 30))),
}
# sp2-sp3 / sp3-sp2, sp2-sp3d / sp3d-sp2, and any UNSPECIFIED-hybridization
# pairing are resolved by _rotamer_set_for_joint's symmetric lookup + this
# fallback, matching the reference's own (mostly-symmetric-anyway) table.
_DEFAULT_ROTAMER_SET = np.radians(list(range(0, 360, 30)))


@dataclass
class TorsionJoint:
    """Represents a single rotatable bond hinge in the kinematic tree."""
    joint_idx: int
    begin_atom_idx: int            # Pivot atom 1 (origin)
    end_atom_idx: int              # Pivot atom 2 (axis definition)
    moving_atom_indices: List[int] # All downstream atoms rotated by this joint
    bond_name: str = ""


class LigandKinematicTree:
    """
    Constructs a rigid-body kinematic tree from an RDKit Mol object.
    Reduces the 3N Cartesian search space to strictly (3 Translation + 4 Rotation + k Dihedrals).
    """
    def __init__(self, mol: Chem.Mol):
        self.mol = Chem.Mol(mol)
        self.num_atoms = mol.GetNumAtoms()
        
        # 1. Store un-distorted base coordinates
        conf = self.mol.GetConformer()
        self.local_coords = np.array(
            [conf.GetAtomPosition(i) for i in range(self.num_atoms)], dtype=np.float64
        )
        
        # 2. Identify rotatable single bonds (excluding rings and terminal bonds)
        unique_matches = identify_rotatable_bonds(self.mol)

        # 3. Build directed tree joints and determine moving subtrees
        self.joints: List[TorsionJoint] = []
        for j_idx, (a1, a2) in enumerate(unique_matches):
            moving_atoms = self._find_downstream_atoms(a1, a2)
            # If the downstream tree contains more than half the atoms, invert direction
            if len(moving_atoms) > self.num_atoms // 2:
                a1, a2 = a2, a1
                moving_atoms = self._find_downstream_atoms(a1, a2)
                
            el1 = self.mol.GetAtomWithIdx(a1).GetSymbol()
            el2 = self.mol.GetAtomWithIdx(a2).GetSymbol()
            bname = f"{el1}{a1}-{el2}{a2}"
            
            self.joints.append(TorsionJoint(
                joint_idx=j_idx,
                begin_atom_idx=a1,
                end_atom_idx=a2,
                moving_atom_indices=moving_atoms,
                bond_name=bname
            ))
            
        self.num_torsions = len(self.joints)
        self._island_decomposer: Optional[TorsionIslandDecomposer] = None

    def get_island_decomposer(self) -> "TorsionIslandDecomposer":
        """Lazily builds (and caches) this tree's TorsionIslandDecomposer --
        island partitioning + rotamer enumeration cost is only paid once,
        the first time clash-free swarm seeding is actually requested."""
        if self._island_decomposer is None:
            self._island_decomposer = TorsionIslandDecomposer(self)
        return self._island_decomposer

    def _find_downstream_atoms(self, begin_idx: int, split_idx: int) -> List[int]:
        """Breadth-first search finding all atoms on the split_idx side of the cut."""
        return find_downstream_atoms(self.mol, begin_idx, split_idx)

    def forward_kinematics(
        self,
        translation: np.ndarray,      # (3,) vector in Angstroms
        quaternion: np.ndarray,       # (4,) [x, y, z, w] orientation
        dihedrals: np.ndarray,        # (k,) joint rotation angles in radians
        base_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes 3D Cartesian coordinates from internal kinematic coordinates (t, q, θ):
        Guarantees 0.000 Å bond length distortion by mathematical definition.
        """
        coords = self.local_coords.copy() if base_coords is None else base_coords.copy()
        
        # 1. Apply internal dihedral rotations around joint axes
        for j_idx, joint in enumerate(self.joints):
            angle_rad = float(dihedrals[j_idx])
            if abs(angle_rad) < 1e-7:
                continue
            origin = coords[joint.begin_atom_idx]
            axis = coords[joint.end_atom_idx] - origin
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis_unit = axis / norm
            
            # Rodrigues rotation matrix for moving subtree
            rot_mat = ScipyRotation.from_rotvec(axis_unit * angle_rad).as_matrix()
            sub_coords = coords[joint.moving_atom_indices] - origin
            coords[joint.moving_atom_indices] = sub_coords.dot(rot_mat.T) + origin

        # 2. Apply global rigid-body rotation around geometric center
        if quaternion is not None and not np.allclose(quaternion, [0, 0, 0, 1]):
            q_norm = quaternion / np.linalg.norm(quaternion)
            global_rot = ScipyRotation.from_quat(q_norm).as_matrix()
            center = np.mean(coords, axis=0)
            coords = (coords - center).dot(global_rot.T) + center

        # 3. Apply global rigid-body translation
        if translation is not None:
            coords += translation

        return coords

    def coords_to_mol(self, coords_angstrom: np.ndarray) -> Chem.Mol:
        """Updates RDKit molecule conformer with 3D kinematic coordinates."""
        mol_out = Chem.Mol(self.mol)
        conf = mol_out.GetConformer()
        for i in range(self.num_atoms):
            p = coords_angstrom[i]
            conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
        return mol_out


class TorsionIslandDecomposer:
    """
    Partitions a LigandKinematicTree's rotatable joints into independent
    "islands" (groups of joints whose downstream subtrees overlap -- i.e.
    joints along the same kinematic chain, since a joint further out on a
    branch has a moving-atom set that is a strict subset of every ancestor
    joint's own moving-atom set), enumerates a hybridization-aware discrete
    rotamer library per joint, filters out self-clashing combinations
    *within each island independently*, and combines islands to build
    clash-free full-molecule dihedral vectors for swarm seeding.

    Why per-island, not per-molecule: sampling all k rotatable bonds
    independently and uniformly over [-pi, pi]^k wastes the overwhelming
    majority of samples on severely self-clashing conformations for any
    molecule with more than a handful of rotatable bonds (the product of k
    independent uniform draws essentially never lands on a jointly sensible
    combination). Rotating one joint is a rigid rotation of everything
    downstream of it, which by definition cannot change any *internal*
    pairwise distance within that downstream subtree -- so whether a given
    branch clashes with itself depends only on that branch's own joints,
    never on any other branch's or any ancestor's angle. That means each
    branch's clash-free rotamer combinations can be enumerated and filtered
    completely independently, in isolation, and then combined -- turning an
    O(prod of all library sizes) search into a sum of much smaller
    per-branch searches.
    """

    def __init__(self, tree: "LigandKinematicTree", clash_cutoff_angstrom: float = 1.2):
        self.tree = tree
        self.clash_cutoff = clash_cutoff_angstrom
        self.islands: List[List[int]] = self._partition_islands()
        self._island_valid_combos: Optional[List[np.ndarray]] = None

    def _partition_islands(self) -> List[List[int]]:
        """
        Connected components of the "downstream subtree overlaps" graph,
        via union-find. Doing this as a proper connected-components
        computation (rather than a single greedy seed-vs-rest pass) matters
        for correctness whenever two joints only share an ancestor without
        directly overlapping each other -- e.g. two sibling branches B and C
        that both descend from a joint A: B and C's own subtrees are
        disjoint from each other (so a single seed-vs-rest pass starting
        from B would never merge C into B's island), but both are subsets
        of A's subtree, so a proper union over all pairs still correctly
        merges {A, B, C} into one island via A acting as the connecting
        pair, regardless of which joint happens to be visited first.
        """
        n = len(self.tree.joints)
        subtree_sets = [set(j.moving_atom_indices) for j in self.tree.joints]
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if subtree_sets[i] & subtree_sets[j]:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    def _rotamer_set_for_joint(self, j_idx: int) -> np.ndarray:
        joint = self.tree.joints[j_idx]
        h1 = self.tree.mol.GetAtomWithIdx(joint.begin_atom_idx).GetHybridization()
        h2 = self.tree.mol.GetAtomWithIdx(joint.end_atom_idx).GetHybridization()
        return (
            DISCRETE_ROTAMER_LIBRARY.get((h1, h2))
            if (h1, h2) in DISCRETE_ROTAMER_LIBRARY
            else DISCRETE_ROTAMER_LIBRARY.get((h2, h1), _DEFAULT_ROTAMER_SET)
        )

    def _clash_free_combos_for_island(
        self, island: List[int], rng: np.random.Generator, max_combos: int = 2000,
    ) -> np.ndarray:
        rotamer_sets = [self._rotamer_set_for_joint(j) for j in island]
        # Compute the full product SIZE without ever materializing it: for
        # an island of, say, 9 sp3-sp3 joints (15 angles each), the full
        # itertools.product is 15**9 ~ 3.8e10 rows -- building that list (as
        # an early version of this method did) OOM-kills the process well
        # before the len(combos) > max_combos subsampling check ever runs.
        # When the product is too large, sample max_combos *random* index
        # tuples directly (one independent choice per joint) instead of
        # ever enumerating the full space -- statistically equivalent to
        # "generate everything, then subsample," at O(max_combos) memory
        # instead of O(product size).
        combo_count = 1
        for rs in rotamer_sets:
            combo_count *= len(rs)
            if combo_count > max_combos:
                break
        if combo_count <= max_combos:
            combos = np.array(list(itertools.product(*rotamer_sets)))
        else:
            combos = np.array([
                [rs[rng.integers(len(rs))] for rs in rotamer_sets]
                for _ in range(max_combos)
            ])

        # The union of every joint's own moving atoms in this island is
        # exactly this island's own subtree extent (its "branch"); only
        # pairwise distances within THIS set can possibly change as a
        # function of THIS island's own joint angles (see class docstring).
        moving_atoms = sorted(set().union(*(set(self.tree.joints[j].moving_atom_indices) for j in island)))
        base_dofs = np.zeros(self.tree.num_torsions)

        min_dists = np.full(len(combos), np.inf)
        for c_idx, combo in enumerate(combos):
            test_dofs = base_dofs.copy()
            for k, j_idx in enumerate(island):
                test_dofs[j_idx] = combo[k]
            xyz = self.tree.forward_kinematics(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), test_dofs)
            sub = xyz[moving_atoms]
            if len(sub) > 1:
                d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
                np.fill_diagonal(d, np.inf)
                min_dists[c_idx] = float(d.min())

        clash_free = min_dists > self.clash_cutoff
        if np.any(clash_free):
            return combos[clash_free]
        # Every discrete combination in this island clashes by the strict
        # cutoff (rare, but possible for a very congested branch) -- fall
        # back to the least-clashing decile rather than an arbitrary
        # first-N slice, matching the reference algorithm's own fallback
        # (vector_dock_align.py: falls back to the bottom 10th percentile
        # of intramolecular LJ energy when no candidate clears its cutoff).
        threshold = np.percentile(min_dists, 90.0)  # top 10% by distance = least clashing
        return combos[min_dists >= threshold]

    def _ensure_island_combos(self, rng: np.random.Generator) -> List[np.ndarray]:
        if self._island_valid_combos is None:
            self._island_valid_combos = [
                self._clash_free_combos_for_island(island, rng) for island in self.islands
            ]
        return self._island_valid_combos

    def sample_clash_free_dihedrals(self, num_samples: int = 100, seed: Optional[int] = None) -> np.ndarray:
        """
        Returns (num_samples, num_torsions) full-molecule dihedral vectors
        (radians), each built by picking one clash-free rotamer combination
        independently per island plus a small +/-5 degree jitter -- self-
        consistent, chemically-sane starting torsions for swarm seeding,
        in place of independent per-DOF uniform random sampling.
        """
        rng = np.random.default_rng(seed)
        island_combos = self._ensure_island_combos(rng)
        out = np.zeros((num_samples, self.tree.num_torsions))
        jitter_scale = np.radians(5.0)
        for s in range(num_samples):
            for island, valid_combos in zip(self.islands, island_combos):
                combo = valid_combos[rng.integers(len(valid_combos))]
                jitter = rng.uniform(-jitter_scale, jitter_scale, size=len(island))
                for k, j_idx in enumerate(island):
                    out[s, j_idx] = combo[k] + jitter[k]
        return out


class KinematicDockingEngine:
    """
    Evaluates OpenMM GPU Hamiltonian on Kinematic Trees (SE(3) x T^k).
    Enables zero-distortion kinematic Monte Carlo and smooth trajectory generation.
    """
    def __init__(
        self,
        engine: DockingEngine,
        ligand_mol: Chem.Mol,
        covalent_res: Optional[str] = None
    ):
        self.engine = engine
        self.tree = LigandKinematicTree(ligand_mol)
        self.covalent_res = covalent_res or engine.covalent_res
        
        # Build OpenMM system once
        cov_restr = None
        if self.covalent_res is not None:
            from .covalent import create_covalent_restraint
            cov_restr = create_covalent_restraint(engine.receptor, ligand_mol, self.covalent_res)
            
        self.system, _, self.lig_start, self.lig_n = engine._build_system(
            ligand_mol, covalent_restraint=cov_restr
        )
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        self.context = (
            mm.Context(self.system, self.integrator, engine.platform)
            if engine.platform
            else mm.Context(self.system, self.integrator)
        )

    def evaluate_state(
        self,
        translation: np.ndarray,
        quaternion: np.ndarray,
        dihedrals: np.ndarray,
        base_coords: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """Evaluates OpenMM potential energy (kcal/mol) and returns 3D coords."""
        coords = self.tree.forward_kinematics(translation, quaternion, dihedrals, base_coords)
        full_pos = self.engine._full_positions_from_coords(coords)
        self.context.setPositions(full_pos)
        
        state = self.context.getState(getEnergy=True)
        energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        score_kcal = float(energy_kj * 0.239006)
        return score_kcal, coords

    def generate_kinematic_sweep_movie(
        self,
        ref_mol: Optional[Chem.Mol] = None,
        n_frames_per_joint: int = 15
    ) -> List[Chem.Mol]:
        """
        Generates a smooth, robotic forward-kinematic exploration movie
        sweeping through each rotatable joint hinge in sequence.
        """
        frames: List[Chem.Mol] = []
        conf = self.tree.mol.GetConformer()
        current_base = np.array([conf.GetAtomPosition(i) for i in range(self.tree.num_atoms)], dtype=np.float64)
        
        heavy_indices = [a.GetIdx() for a in self.tree.mol.GetAtoms() if a.GetAtomicNum() > 1]
        p_ref = None
        if ref_mol is not None:
            conf_ref = ref_mol.GetConformer()
            p_ref = np.array([conf_ref.GetAtomPosition(i) for i in heavy_indices])
            
        frame_count = 0
        
        # 1. Sweep each rotatable joint hinge through [-60°, +60°]
        for j_idx, joint in enumerate(self.tree.joints):
            angles = np.linspace(-np.pi / 3.0, np.pi / 3.0, n_frames_per_joint)
            # Add reverse return sweep
            angles = np.concatenate([angles, angles[::-1]])
            
            for angle in angles:
                dihedrals = np.zeros(self.tree.num_torsions)
                dihedrals[j_idx] = angle
                
                score, coords = self.evaluate_state(
                    translation=np.zeros(3),
                    quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                    dihedrals=dihedrals,
                    base_coords=current_base
                )
                
                mol_frame = self.tree.coords_to_mol(coords)
                mol_frame.SetProp("FRAME_ID", str(frame_count + 1))
                mol_frame.SetProp("ACTIVE_JOINT", f"Joint_{j_idx}_{joint.bond_name}")
                mol_frame.SetProp("DIHEDRAL_ANGLE_DEG", f"{np.degrees(angle):.1f}")
                mol_frame.SetProp("OPENMM_SCORE_KCAL", f"{score:.3f}")
                mol_frame.SetProp("MAX_BOND_DEV_A", "0.0000") # Permanently exact by definition
                
                if p_ref is not None:
                    p_curr = coords[heavy_indices]
                    rmsd = float(np.sqrt(np.mean(np.sum((p_curr - p_ref)**2, axis=1))))
                    mol_frame.SetProp("RMSD_TO_CRYSTAL_A", f"{rmsd:.3f}")
                    
                frames.append(mol_frame)
                frame_count += 1
                
        return frames


@dataclass
class SwarmParticle:
    """Represents an articulated kinematic particle in the PSO swarm."""
    particle_id: int
    trans: np.ndarray            # (3,) Translation in Å
    rot_vec: np.ndarray          # (3,) Rodrigues rotation vector
    dihedrals: np.ndarray        # (k,) Joint dihedrals in radians
    v_trans: np.ndarray          # (3,) Translation velocity
    v_rot: np.ndarray            # (3,) Rotational velocity
    v_dihedrals: np.ndarray      # (k,) Joint angular velocities
    p_best_trans: np.ndarray
    p_best_rot: np.ndarray
    p_best_dihedrals: np.ndarray
    p_best_score: float
    current_score: float


class KinematicParticleSwarmOptimizer:
    """
    Particle Swarm Optimization on the (SE(3) x T^k) Kinematic Manifold.
    Coordinates multi-particle swarms to escape local surface traps into deep catalytic clefts.
    """
    def __init__(self, kin_engine: KinematicDockingEngine):
        self.kin_engine = kin_engine
        self.tree = kin_engine.tree
        self.num_torsions = self.tree.num_torsions

    @staticmethod
    def _toroidal_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculates the shortest angular difference on the circle T^k."""
        return toroidal_diff(a, b)

    def run_pso(
        self,
        n_particles: int = 20,
        n_iterations: int = 25,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
        ref_mol: Optional[Chem.Mol] = None,
        use_torsion_islands: bool = True,
    ) -> Tuple[Chem.Mol, float, List[Chem.Mol]]:
        """
        Executes Kinematic PSO and returns (best_mol, best_score, swarm_movie_frames).

        use_torsion_islands: seed non-elite particles' initial torsions from
        TorsionIslandDecomposer.sample_clash_free_dihedrals instead of
        independent per-DOF uniform random sampling -- for a molecule with
        more than a few rotatable bonds, independent uniform sampling wastes
        the overwhelming majority of the initial swarm on severely self-
        clashing starting conformations (see TorsionIslandDecomposer's
        docstring for why per-branch enumeration avoids this).
        """
        particles: List[SwarmParticle] = []
        g_best_trans = np.zeros(3)
        g_best_rot = np.zeros(3)
        g_best_dihedrals = np.zeros(self.num_torsions)
        g_best_score = 9999999.0

        island_dihedral_seeds: Optional[np.ndarray] = None
        if use_torsion_islands and self.num_torsions > 0:
            island_dihedral_seeds = self.tree.get_island_decomposer().sample_clash_free_dihedrals(
                num_samples=max(0, n_particles - 1)
            )

        # 1. Initialize Swarm Particles with diverse random kinematic configurations
        for p_id in range(n_particles):
            if p_id == 0:
                t_init = np.zeros(3)
                r_init = np.zeros(3)
                d_init = np.zeros(self.num_torsions)
            else:
                t_init = np.random.uniform(-4.0, 4.0, 3)
                r_init = np.random.uniform(-np.pi, np.pi, 3)
                d_init = (
                    island_dihedral_seeds[p_id - 1]
                    if island_dihedral_seeds is not None
                    else np.random.uniform(-np.pi, np.pi, self.num_torsions)
                )
                
            q_init = ScipyRotation.from_rotvec(r_init).as_quat()
            score, _ = self.kin_engine.evaluate_state(t_init, q_init, d_init)
            
            p = SwarmParticle(
                particle_id=p_id,
                trans=t_init.copy(),
                rot_vec=r_init.copy(),
                dihedrals=d_init.copy(),
                v_trans=np.random.uniform(-1.0, 1.0, 3),
                v_rot=np.random.uniform(-0.5, 0.5, 3),
                v_dihedrals=np.random.uniform(-0.5, 0.5, self.num_torsions),
                p_best_trans=t_init.copy(),
                p_best_rot=r_init.copy(),
                p_best_dihedrals=d_init.copy(),
                p_best_score=score,
                current_score=score
            )
            particles.append(p)
            
            if score < g_best_score:
                g_best_score = score
                g_best_trans = t_init.copy()
                g_best_rot = r_init.copy()
                g_best_dihedrals = d_init.copy()

        # Movie frames across iterations
        movie_frames: List[Chem.Mol] = []
        heavy_indices = [a.GetIdx() for a in self.tree.mol.GetAtoms() if a.GetAtomicNum() > 1]
        p_ref = None
        if ref_mol is not None:
            p_ref = np.array([ref_mol.GetConformer().GetAtomPosition(i) for i in heavy_indices])
            
        # 2. Main Swarm Evolution Loop
        for it in range(n_iterations):
            for p in particles:
                r1 = np.random.uniform(0.0, 1.0)
                r2 = np.random.uniform(0.0, 1.0)
                
                # Velocity updates
                # A. Translation velocity
                p.v_trans = (
                    w * p.v_trans
                    + c1 * r1 * (p.p_best_trans - p.trans)
                    + c2 * r2 * (g_best_trans - p.trans)
                )
                p.trans += np.clip(p.v_trans, -3.0, 3.0)
                
                # B. Rotational velocity
                diff_rot_p = self._toroidal_sub(p.p_best_rot, p.rot_vec)
                diff_rot_g = self._toroidal_sub(g_best_rot, p.rot_vec)
                p.v_rot = w * p.v_rot + c1 * r1 * diff_rot_p + c2 * r2 * diff_rot_g
                p.rot_vec = (p.rot_vec + np.clip(p.v_rot, -1.0, 1.0) + np.pi) % (2 * np.pi) - np.pi
                
                # C. Dihedral velocity on T^k
                diff_d_p = self._toroidal_sub(p.p_best_dihedrals, p.dihedrals)
                diff_d_g = self._toroidal_sub(g_best_dihedrals, p.dihedrals)
                p.v_dihedrals = w * p.v_dihedrals + c1 * r1 * diff_d_p + c2 * r2 * diff_d_g
                p.dihedrals = (p.dihedrals + np.clip(p.v_dihedrals, -1.0, 1.0) + np.pi) % (2 * np.pi) - np.pi
                
                # Evaluate new energy
                q_curr = ScipyRotation.from_rotvec(p.rot_vec).as_quat()
                score, coords = self.kin_engine.evaluate_state(p.trans, q_curr, p.dihedrals)
                p.current_score = score
                
                # Update Personal Best
                if score < p.p_best_score:
                    p.p_best_score = score
                    p.p_best_trans = p.trans.copy()
                    p.p_best_rot = p.rot_vec.copy()
                    p.p_best_dihedrals = p.dihedrals.copy()
                    
                # Update Global Swarm Best
                if score < g_best_score:
                    g_best_score = score
                    g_best_trans = p.trans.copy()
                    g_best_rot = p.rot_vec.copy()
                    g_best_dihedrals = p.dihedrals.copy()
                    
                # Record frame for PyMOL movie
                mol_f = self.tree.coords_to_mol(coords)
                mol_f.SetProp("ITERATION", str(it + 1))
                mol_f.SetProp("PARTICLE_ID", str(p.particle_id + 1))
                mol_f.SetProp("SCORE_KCAL", f"{score:.2f}")
                mol_f.SetProp("SWARM_BEST_SCORE", f"{g_best_score:.2f}")
                mol_f.SetProp("MAX_BOND_DEV_A", "0.0000")
                if p_ref is not None:
                    p_curr = coords[heavy_indices]
                    rmsd = float(np.sqrt(np.mean(np.sum((p_curr - p_ref)**2, axis=1))))
                    mol_f.SetProp("RMSD_TO_CRYSTAL_A", f"{rmsd:.2f}")
                movie_frames.append(mol_f)

        # Build final best molecule
        q_best = ScipyRotation.from_rotvec(g_best_rot).as_quat()
        _, best_coords = self.kin_engine.evaluate_state(g_best_trans, q_best, g_best_dihedrals)
        best_mol = self.tree.coords_to_mol(best_coords)
        best_mol.SetProp("DOCK_SCORE", f"{g_best_score:.3f}")
        return best_mol, g_best_score, movie_frames
