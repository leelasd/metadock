"""
Docking engine integrating OpenMM simulation, custom forces, minimization,
and search protocols (Simulated Annealing & Torsion Monte Carlo).
"""
from __future__ import annotations
import math
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import openmm as mm
from openmm import unit
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdMolDescriptors
from scipy.spatial.transform import Rotation as ScipyRotation

from .core import MolecularSystem, DockAtom, Mol2Parser, SDFParser, PDBParser
from .cavity import CavityDefinition, create_cavity_restraint_force
from .scoring import (
    ScoreWeights,
    create_rdock_nonbonded_forces,
    create_combined_search_force,
    create_grid_search_force,
    GROUP_VDW_INTER,
    GROUP_VDW_INTRA,
    GROUP_POLAR_INTER,
    GROUP_POLAR_INTRA,
    GROUP_REPUL,
    GROUP_HYD,
    GROUP_VALENCE,
    GROUP_CAVITY,
    GROUP_PHARMA,
    GROUP_TETHER,
    GROUP_SOLVENT,
)
from .gridding import GridBox, compute_potential_grids, STANDARD_VDW_ELEMENTS
from .pharmacophore import (
    PharmaPoint,
    parse_pharma_restr,
    create_pharmacophore_restraint_forces,
    align_ligand_to_pharmacophore,
    find_ligand_pharma_features,
)
from .tether import (
    TetherConstraint,
    find_tethered_atoms_mcs,
    create_tether_restraint_force,
)
from .solvent import load_solvent_waters, create_solvent_tether_force
from .kinematic_utils import find_downstream_atoms
from .gradient_minimizer import lbfgs_minimize
from .covalent import (
    CovalentRestraint,
    create_covalent_restraint,
    create_covalent_bond_force,
    prealign_ligand_for_covalent_docking,
    detect_ligand_warhead,
)


@dataclass
class DockingResult:
    mol: Chem.Mol
    score: float
    scores: Dict[str, float]
    run_idx: int = 0
    trajectory: Optional[List[Chem.Mol]] = None


# --- Genetic Algorithm chromosome helpers -----------------------------------
# rDock's actual default search engine is a population-based Genetic Algorithm
# over a compact chromosome: 3 rigid-body translation genes + 3 rigid-body
# rotation genes + one torsion gene per rotatable bond. These free functions
# implement that chromosome (RDKit supplies the ligand topology/rotatable
# bonds; OpenMM supplies the receptor physics used to score each individual).

def identify_torsion_dofs(ligand_mol: Chem.Mol) -> List[Dict[str, Any]]:
    """
    Finds rotatable single bonds and, for each, the rigid subtree of atoms
    that moves when the torsion is changed, plus a pair of reference atoms
    used to measure/set the absolute dihedral angle.
    """
    num_atoms = ligand_mol.GetNumAtoms()
    dofs: List[Dict[str, Any]] = []
    for b in ligand_mol.GetBonds():
        if b.IsInRing() or b.GetBondTypeAsDouble() != 1.0:
            continue
        a1_idx, a2_idx = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        a1_atom, a2_atom = ligand_mol.GetAtomWithIdx(a1_idx), ligand_mol.GetAtomWithIdx(a2_idx)
        if a1_atom.GetDegree() < 2 or a2_atom.GetDegree() < 2:
            continue

        # Subtree of atoms on the a2 side of the bond (a2 inclusive, a1 excluded).
        # If that side holds more than half the atoms, rotate the smaller a1
        # side instead (same physical dihedral, fewer atoms to transform).
        subtree = find_downstream_atoms(ligand_mol, a1_idx, a2_idx)
        if len(subtree) > num_atoms // 2:
            a1_idx, a2_idx = a2_idx, a1_idx
            a1_atom, a2_atom = a2_atom, a1_atom
            subtree = find_downstream_atoms(ligand_mol, a1_idx, a2_idx)

        ref0 = next((n.GetIdx() for n in a1_atom.GetNeighbors() if n.GetIdx() != a2_idx), None)
        ref3 = next((n.GetIdx() for n in a2_atom.GetNeighbors() if n.GetIdx() != a1_idx), None)
        if ref0 is None or ref3 is None:
            continue

        dofs.append({"a1": a1_idx, "a2": a2_idx, "subtree": subtree, "ref0": ref0, "ref3": ref3})
    return dofs


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def decode_chromosome(
    chromosome: np.ndarray,
    base_local_coords: np.ndarray,
    torsion_dofs: List[Dict[str, Any]],
    cavity_center: np.ndarray,
) -> np.ndarray:
    """
    Decodes a chromosome into absolute Cartesian ligand coordinates (Angstroms):
    first sets each rotatable-bond torsion to its absolute gene value (internal
    DOFs), then places the resulting rigid conformation via the 6 rigid-body
    genes relative to the cavity center (external DOFs).
    """
    coords = base_local_coords.copy()
    n_t = len(torsion_dofs)
    trans = chromosome[0:3]
    euler = chromosome[3:6]
    torsions = chromosome[6:6 + n_t]

    for dof, target_angle in zip(torsion_dofs, torsions):
        p1, p2 = coords[dof["a1"]], coords[dof["a2"]]
        axis = p2 - p1
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            continue
        axis = axis / norm
        current = _dihedral_deg(coords[dof["ref0"]], p1, p2, coords[dof["ref3"]])
        delta = float(target_angle) - current
        rot = ScipyRotation.from_rotvec(np.radians(delta) * axis)
        idx = dof["subtree"]
        coords[idx] = p1 + rot.apply(coords[idx] - p1)

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    rot_mat = ScipyRotation.from_euler("xyz", euler, degrees=True).as_matrix()
    return cavity_center + trans + centered.dot(rot_mat.T)


def encode_chromosome(
    coords: np.ndarray,
    base_local_coords: np.ndarray,
    torsion_dofs: List[Dict[str, Any]],
    cavity_center: np.ndarray,
) -> np.ndarray:
    """
    Inverse of decode_chromosome: given actual Cartesian ligand coordinates
    (e.g. the result of a Cartesian local minimization), recovers the
    chromosome that would decode to (approximately) those coordinates. This
    is what makes local search *Lamarckian* rather than Baldwinian -- a
    locally-optimized phenotype can be written back into the genotype and
    inherited by future generations/proposals, instead of the improvement
    being discarded the moment fitness is scored (AutoDock's LGA does this
    via Solis-Wets local search operating directly in genotype space; we get
    the same effect by re-deriving the genotype via Kabsch alignment after a
    Cartesian minimization).

    Torsions are read directly off the actual coordinates (real dihedral
    measurements). The 6 rigid-body genes are recovered by first replaying
    those torsions onto base_local_coords (reproducing decode_chromosome's
    torsion-only intermediate, centered at the origin), then finding the
    rotation + translation that best superimposes that intermediate onto the
    actual coordinates (Kabsch algorithm) -- the same superposition method
    already used by pharmacophore.align_ligand_to_pharmacophore.
    """
    n_t = len(torsion_dofs)
    torsions = np.array([
        _dihedral_deg(coords[d["ref0"]], coords[d["a1"]], coords[d["a2"]], coords[d["ref3"]])
        for d in torsion_dofs
    ])

    zero_chrom = np.concatenate([np.zeros(3), np.zeros(3), torsions])
    intermediate = decode_chromosome(zero_chrom, base_local_coords, torsion_dofs, np.zeros(3))

    q_centroid = coords.mean(axis=0)
    trans = q_centroid - cavity_center

    P = intermediate  # already centered at the origin by construction above
    Q = coords - q_centroid
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    euler = ScipyRotation.from_matrix(R).as_euler("xyz", degrees=True)
    return np.concatenate([trans, euler, torsions])


def tournament_select(
    rng: np.random.Generator, population: List[np.ndarray], scores: List[float], k: int
) -> np.ndarray:
    idxs = rng.integers(0, len(population), size=k)
    best = idxs[0]
    for i in idxs[1:]:
        if scores[i] < scores[best]:
            best = i
    return population[best]


def crossover_chromosomes(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    mask = rng.random(p1.shape[0]) < 0.5
    return np.where(mask, p1, p2)


def mutate_chromosome(
    rng: np.random.Generator,
    chromosome: np.ndarray,
    mutation_rate: float,
    n_torsions: int,
    trans_sigma: float = 0.5,
    rot_sigma: float = 20.0,
    torsion_sigma: float = 30.0,
) -> np.ndarray:
    child = chromosome.copy()
    n_genes = child.shape[0]
    mut_mask = rng.random(n_genes) < mutation_rate
    noise = np.zeros(n_genes)
    noise[0:3] = rng.normal(0.0, trans_sigma, size=3)
    noise[3:6] = rng.normal(0.0, rot_sigma, size=3)
    if n_torsions > 0:
        noise[6:6 + n_torsions] = rng.normal(0.0, torsion_sigma, size=n_torsions)
    child[mut_mask] += noise[mut_mask]
    return child


def mutate_chromosome_vina_style(
    rng: np.random.Generator,
    chromosome: np.ndarray,
    n_torsions: int,
    trans_amplitude: float = 2.0,
    rot_amplitude_deg: float = 60.0,
) -> np.ndarray:
    """
    AutoDock Vina/smina-style coarse single-entity mutation (see mutate.cpp's
    mutate_conf): picks exactly ONE of {translation, rotation, torsion_1, ...,
    torsion_k} uniformly at random and applies one full-amplitude move to
    just that entity, leaving every other gene untouched. This is
    deliberately coarser and more localized than mutate_chromosome's
    every-gene-at-once Gaussian jitter -- the coarseness is the point: paired
    with an immediate local minimization (see dock_monte_carlo_minimization),
    a single big jump followed by relaxation explores a genuinely different
    basin each step, rather than a small perturbation of the current one.
    """
    child = chromosome.copy()
    n_entities = 2 + n_torsions
    which = int(rng.integers(0, n_entities))
    if which == 0:
        direction = rng.normal(size=3)
        direction /= (np.linalg.norm(direction) + 1e-12)
        mag = trans_amplitude * (rng.random() ** (1.0 / 3.0))
        child[0:3] += direction * mag
    elif which == 1:
        axis = rng.normal(size=3)
        axis /= (np.linalg.norm(axis) + 1e-12)
        mag = rot_amplitude_deg * (rng.random() ** (1.0 / 3.0))
        child[3:6] += axis * mag
    else:
        child[6 + (which - 2)] = rng.uniform(-180.0, 180.0)
    return child


class DockingEngine:
    """
    OpenMM Docking Engine implementing rDock-style scoring and multi-protocol sampling.
    """

    def __init__(
        self,
        receptor_path: Path | str,
        cavity: Optional[CavityDefinition] = None,
        cavity_prm_path: Optional[Path | str] = None,
        waters_pdb_path: Optional[Path | str] = None,
        pharma_restr_path: Optional[Path | str] = None,
        flexible_radius: Optional[float] = None,
        flexible_residues: Optional[List[int | str]] = None,
        covalent_res: Optional[str | int] = None,
        weights: Optional[ScoreWeights] = None,
        platform_name: Optional[str] = None,
    ):
        self.receptor_path = Path(receptor_path)
        self.flexible_radius = flexible_radius
        self.flexible_residues = flexible_residues
        self.covalent_res = covalent_res
        self.weights = weights or ScoreWeights()
        # Lazily populated by _ensure_grid_cache: {"box": GridBox, "shared":
        # {name: ndarray}, "vdw": {element: ndarray}}. Grid values are
        # weight-independent (weights apply at force-eval time), and the
        # receptor/cavity never change after __init__, so this cache is safe
        # for the engine's whole lifetime -- computed once, reused by every
        # dock_* call, every GA generation, every SA move (see gridding.py).
        self._grid_cache: Optional[Dict[str, Any]] = None

        # 1. Load Receptor
        if self.receptor_path.suffix.lower() == ".mol2":
            self.receptor = Mol2Parser.parse(self.receptor_path)
        elif self.receptor_path.suffix.lower() in [".pdb", ".ent"]:
            self.receptor = PDBParser.parse(self.receptor_path)
        else:
            raise ValueError(f"Unsupported receptor format: {self.receptor_path.suffix}")

        # 2. Cavity definition
        if cavity is not None:
            self.cavity = cavity
        elif cavity_prm_path is not None:
            self.cavity = CavityDefinition.from_prm_file(cavity_prm_path)
        else:
            center = self.receptor.get_center()
            self.cavity = CavityDefinition(
                center=center,
                radius=15.0,
                min_coords=center - 15.0,
                max_coords=center + 15.0,
                name="DefaultCavity",
            )

        # 3. Optional Solvent Waters
        self.waters: Optional[MolecularSystem] = None
        if waters_pdb_path is not None and Path(waters_pdb_path).exists():
            self.waters = load_solvent_waters(waters_pdb_path)

        # 4. Optional Pharmacophores
        self.pharma_points: List[PharmaPoint] = []
        if pharma_restr_path is not None and Path(pharma_restr_path).exists():
            self.pharma_points = parse_pharma_restr(pharma_restr_path)

        # Platform selection
        self.platform = None
        if platform_name:
            self.platform = mm.Platform.getPlatformByName(platform_name)
        else:
            for plat in ["Metal", "OpenCL", "CPU", "Reference"]:
                try:
                    self.platform = mm.Platform.getPlatformByName(plat)
                    break
                except Exception:
                    continue

    def _prepare_covalent(
        self, ligand_mol: Chem.Mol, prealign_threshold_nm: float = 0.5
    ) -> Tuple[Chem.Mol, Optional[CovalentRestraint]]:
        """
        Resolves self.covalent_res into a CovalentRestraint (if set). Only
        rigidly pre-aligns the ligand (covalent.prealign_ligand_for_covalent_docking)
        when the input pose's electrophile atom is *not already* near the
        target attack geometry (beyond `prealign_threshold_nm`). Forcing the
        heuristic two-point alignment onto a pose that's already close --
        e.g. a co-crystal structure, or an already-docked/refined pose --
        would throw away a perfectly good (often better than the heuristic's)
        starting orientation. Called at the top of every public docking
        method so every search protocol, not just minimize(), gets a
        chemically sensible starting pose when it actually needs one.
        """
        if self.covalent_res is None:
            return ligand_mol, None
        restraint = create_covalent_restraint(self.receptor, ligand_mol, self.covalent_res)

        conf = ligand_mol.GetConformer()
        el_pos = np.array(conf.GetAtomPosition(restraint.lig_electrophile_idx))
        nucl_pos = self.receptor.atoms[restraint.rec_nucleophile_idx].coord
        current_dist_nm = float(np.linalg.norm(el_pos - nucl_pos)) * 0.1

        if current_dist_nm <= prealign_threshold_nm:
            return ligand_mol, restraint

        aligned_mol = prealign_ligand_for_covalent_docking(ligand_mol, self.receptor, restraint)
        aligned_mol = self._resolve_covalent_rotation(aligned_mol, restraint)
        return aligned_mol, restraint

    def _resolve_covalent_rotation(
        self,
        ligand_mol: Chem.Mol,
        restraint: CovalentRestraint,
        n_candidates: int = 24,
        pharma_weight: float = 15.0,
    ) -> Chem.Mol:
        """
        prealign_ligand_for_covalent_docking fixes 5 of the ligand's 6 rigid-body
        DOFs (electrophile position + 2 rotational, via the two-point
        alignment); rotation about the forming bond axis itself is left
        arbitrary. That single remaining DOF is exactly what determines
        whether the rest of the ligand swings into open space or straight
        into the receptor -- and it's cheap to resolve directly (1-D grid
        search) rather than hoping a generic search protocol's small step
        sizes stumble onto a clash-free angle. AutoDock's GA instead treats
        this as just another torsion-tree DOF explored by the full population
        search; since ours are local-refinement (see dock_genetic_algorithm),
        we resolve it once, up front, the same way a real user would pick a
        starting rotamer.

        When self.pharma_points is set, candidates are scored on steric clash
        *and* pharmacophore-feature proximity. This is the useful synergy with
        covalent docking: the covalent bond already pins 5 of 6 DOFs to near
        machine precision, so pharmacophore points no longer need the normal
        3-point minimum to fully resolve orientation (see
        pharmacophore.align_ligand_to_pharmacophore, which needs >=3 points
        for its independent Kabsch fit) -- a single point can be enough to
        break the one remaining rotational ambiguity here.
        """
        conf = ligand_mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
        el_pos = coords[restraint.lig_electrophile_idx]
        nucl_pos = self.receptor.atoms[restraint.rec_nucleophile_idx].coord
        axis = el_pos - nucl_pos
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return ligand_mol
        axis = axis / axis_norm

        rec_coords = self.receptor.coordinates
        near_mask = np.linalg.norm(rec_coords - el_pos, axis=1) < 12.0
        rec_near = rec_coords[near_mask]
        if rec_near.shape[0] == 0:
            return ligand_mol

        pharma_features = find_ligand_pharma_features(ligand_mol) if self.pharma_points else {}

        best_angle_rad = 0.0
        best_score = float("inf")
        for angle_deg in np.linspace(0.0, 360.0, n_candidates, endpoint=False):
            angle_rad = math.radians(angle_deg)
            rot = ScipyRotation.from_rotvec(angle_rad * axis)
            trial = el_pos + rot.apply(coords - el_pos)
            d = np.linalg.norm(trial[:, None, :] - rec_near[None, :, :], axis=2)
            clash = float(np.sum(np.clip(3.0 - d, 0.0, None) ** 2))

            pharma_penalty = 0.0
            for p in self.pharma_points:
                feats = pharma_features.get(p.ptype, [])
                if not feats:
                    continue
                best_feat_d = min(
                    float(np.linalg.norm(trial[feat].mean(axis=0) - p.coords)) for feat in feats
                )
                pharma_penalty += best_feat_d ** 2

            total = clash + pharma_weight * pharma_penalty
            if total < best_score:
                best_score = total
                best_angle_rad = angle_rad

        rot = ScipyRotation.from_rotvec(best_angle_rad * axis)
        new_coords = el_pos + rot.apply(coords - el_pos)
        mol_copy = Chem.Mol(ligand_mol)
        conf2 = mol_copy.GetConformer()
        for i in range(mol_copy.GetNumAtoms()):
            p = new_coords[i]
            conf2.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
        return mol_copy

    def _ensure_grid_cache(self, required_types: set) -> None:
        """
        Lazily computes and extends self._grid_cache so it covers at least
        `required_types` VDW element channels. First call also computes the
        5 shared (elec/hbdon/hbacc/hyd/repul) grids once, over a box sized
        for self.cavity + a ligand margin. Deliberately does *not* precompute
        all 10 STANDARD_VDW_ELEMENTS up front (the plan's original phrasing)
        -- a receptor-wide grid channel for an element type no ligand in the
        session ever uses would be wasted time/memory, and per-atom windowed
        accumulation cost only depends on the number of receptor atoms and
        the cutoff (not on how many channels are requested), so extending
        the cache incrementally as new elements show up is strictly cheaper
        with no accuracy cost -- results are identical to computing
        everything up front.
        """
        if self._grid_cache is None:
            box = GridBox.from_cavity(self.cavity, ligand_margin_ang=6.0, spacing_ang=0.375)
            shared = compute_potential_grids(
                self.receptor, box, vdw_probe_types=[], compute_vdw=False, compute_shared=True
            )
            self._grid_cache = {"box": box, "shared": shared, "vdw": {}}

        missing = [t for t in required_types if t not in self._grid_cache["vdw"]]
        if missing:
            new_vdw = compute_potential_grids(
                self.receptor, self._grid_cache["box"], vdw_probe_types=missing,
                compute_vdw=True, compute_shared=False,
            )
            for t in missing:
                self._grid_cache["vdw"][t] = new_vdw[f"vdw_{t}"]

    def _get_grid_search_forces(self, lig_sys: MolecularSystem):
        """
        Returns a scoring.GridSearchForces for this ligand's element set, or
        None if the ligand contains an element outside
        gridding.STANDARD_VDW_ELEMENTS (e.g. a metal) -- the documented,
        explicit fallback signal create_grid_search_force's docstring asks
        callers to check for, rather than relying on its own silent
        per-atom omission as the safety net.
        """
        required = {a.element.upper() for a in lig_sys.atoms}
        if not required.issubset(set(STANDARD_VDW_ELEMENTS)):
            return None

        self._ensure_grid_cache(required)
        vdw_subset = {f"vdw_{t}": self._grid_cache["vdw"][t] for t in required}
        box = self._grid_cache["box"]
        return create_grid_search_force(
            vdw_subset, box, self._grid_cache["shared"], box, self.weights,
            vdw_probe_types=list(required),
        )

    def _build_system(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        covalent_restraint: Optional[CovalentRestraint] = None,
        fast_search: bool | str = False,
    ) -> Tuple[mm.System, MolecularSystem, int, int]:
        """
        Assembles OpenMM System with receptor, waters, ligand, and all scoring forces.
        Returns: (system, combined_mol_sys, ligand_start_idx, ligand_num_atoms)

        fast_search selects among three nonbonded backends:
        - False (default): six-way decomposed CustomNonbondedForces (real
          per-term SCORE.INTER.* energies) -- use for anything whose score is
          actually reported.
        - True or "pairwise": one combined CustomNonbondedForce with
          identical physics (scoring.create_combined_search_force). Six
          separate forces means six separate neighbor-list rebuilds against
          the full receptor per setPositions() call -- irrelevant for one-off
          scoring, but decisive for an inner search loop evaluating thousands
          of large-jump candidate poses.
        - "grid": AutoDock-style precomputed-grid nonbonded force
          (scoring.create_grid_search_force) -- turns the O(N_ligand x
          N_receptor) pairwise sum into O(N_ligand) grid interpolation
          lookups, at the cost of a one-time (cached on this DockingEngine
          instance, see _ensure_grid_cache) grid-computation pass. Falls
          back to "pairwise" transparently if the ligand contains an element
          outside gridding.STANDARD_VDW_ELEMENTS.

        Use "grid" or "pairwise" only for search-loop fitness ranking; use
        the default (real decomposed terms) for anything whose score is
        actually reported.
        """
        lig_sys = SDFParser.mol_to_system(ligand_mol)
        system = mm.System()

        rec_n = len(self.receptor.atoms)
        wat_n = len(self.waters.atoms) if self.waters else 0
        lig_start = rec_n + wat_n
        lig_n = len(lig_sys.atoms)

        # 1. Identify flexible residues if requested
        flex_res_indices = set()
        if self.flexible_radius is not None:
            cav_center = self.cavity.center
            for a in self.receptor.atoms:
                if np.linalg.norm(a.coord - cav_center) <= self.flexible_radius:
                    flex_res_indices.add(a.residue_idx)
        elif self.flexible_residues is not None:
            flex_res_indices = set(self.flexible_residues)

        # 1. Add particles
        # Receptor atoms: Mass = 0.0 for static atoms, non-zero for flexible pocket residues
        backbone_restraint = mm.CustomExternalForce("0.5*k_bb*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        backbone_restraint.addPerParticleParameter("x0")
        backbone_restraint.addPerParticleParameter("y0")
        backbone_restraint.addPerParticleParameter("z0")
        backbone_restraint.addGlobalParameter("k_bb", 5000.0)
        backbone_restraint.setForceGroup(GROUP_VALENCE)
        backbone_restraint.setName("ReceptorBackboneRestraint")
        has_bb_restraints = False

        for idx, a in enumerate(self.receptor.atoms):
            if a.residue_idx in flex_res_indices:
                el = a.element.upper()
                m = 12.011 if el == "C" else (1.008 if el == "H" else (15.999 if el == "O" else (14.007 if el == "N" else 32.06)))
                system.addParticle(m * unit.dalton)
                
                # If backbone, tether with strong harmonic position restraint
                if a.name.upper() in ["CA", "C", "N", "O", "H", "HA", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]:
                    pos_nm = a.coord * 0.1
                    backbone_restraint.addParticle(idx, [pos_nm[0], pos_nm[1], pos_nm[2]])
                    has_bb_restraints = True
            else:
                system.addParticle(0.0 * unit.dalton)

        if has_bb_restraints:
            system.addForce(backbone_restraint)

        # Receptor internal valence bonds for flexible residues
        if flex_res_indices:
            rec_bond_force = mm.HarmonicBondForce()
            for b in self.receptor.bonds:
                a1 = self.receptor.atoms[b.atom1]
                a2 = self.receptor.atoms[b.atom2]
                if a1.residue_idx in flex_res_indices and a2.residue_idx in flex_res_indices:
                    r0_nm = float(np.linalg.norm(a1.coord - a2.coord) * 0.1)
                    rec_bond_force.addBond(b.atom1, b.atom2, r0_nm, 500000.0)
            rec_bond_force.setForceGroup(GROUP_VALENCE)
            system.addForce(rec_bond_force)

        # Water atoms: Mass = 16.0 / 1.0
        if self.waters:
            for a in self.waters.atoms:
                mass = 16.0 if a.element == "O" else 1.0
                system.addParticle(mass * unit.dalton)

        # Ligand atoms: standard atomic masses
        for a in lig_sys.atoms:
            el = a.element.upper()
            m = 12.011 if el == "C" else (1.008 if el == "H" else (15.999 if el == "O" else (14.007 if el == "N" else 32.06)))
            system.addParticle(m * unit.dalton)

        # 2. Nonbonded Forces (separate VDW / POLAR / REPUL / HYD terms for real score decomposition,
        #    or one combined/grid force for fast search-loop ranking -- see fast_search docstring above)
        nb_force = None
        if fast_search == "grid":
            nb_force = self._get_grid_search_forces(lig_sys)
        if nb_force is None:
            nb_force = (
                create_combined_search_force(self.weights)
                if fast_search
                else create_rdock_nonbonded_forces(self.weights)
            )
        all_atoms = list(self.receptor.atoms) + (list(self.waters.atoms) if self.waters else []) + list(lig_sys.atoms)

        for i, a in enumerate(all_atoms):
            is_lig = 1.0 if i >= lig_start else 0.0
            is_hyd = 1.0 if a.element.upper() in ["C", "CL", "BR", "I", "F"] and not a.is_polar else 0.0
            nb_force.addParticle([
                a.charge,
                a.sigma,
                a.epsilon,
                1.0 if a.is_donor else 0.0,
                1.0 if a.is_acceptor else 0.0,
                is_hyd,
                is_lig,
            ])

        # Track unique exclusions
        excluded_pairs = set()

        def add_unique_exclusion(i1: int, i2: int):
            pair = (min(i1, i2), max(i1, i2))
            if pair not in excluded_pairs:
                excluded_pairs.add(pair)
                nb_force.addExclusion(pair[0], pair[1])

        # Exclude 1-2 bonded pairs and 1-3 angle pairs within ligand
        for b in lig_sys.bonds:
            add_unique_exclusion(lig_start + b.atom1, lig_start + b.atom2)

        for atom in ligand_mol.GetAtoms():
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    add_unique_exclusion(lig_start + nbrs[i], lig_start + nbrs[j])

        # Compute fused ring systems (connected components of rings)
        ring_info = ligand_mol.GetRingInfo()
        rings = [set(r) for r in ring_info.AtomRings()]
        fused_systems: List[set[int]] = []
        for r in rings:
            merged = False
            for f in fused_systems:
                if f & r:
                    f.update(r)
                    merged = True
                    break
            if not merged:
                fused_systems.append(r)

        # Exclude all intra-ring and intra-fused-system atom pairs from nonbonded force to prevent ring buckling
        for f in fused_systems:
            f_list = sorted(list(f))
            for i in range(len(f_list)):
                for j in range(i + 1, len(f_list)):
                    add_unique_exclusion(lig_start + f_list[i], lig_start + f_list[j])

        for f in nb_force.forces:
            system.addForce(f)

        # 3. Cavity Restraint Force
        lig_indices = list(range(lig_start, lig_start + lig_n))
        cav_force = create_cavity_restraint_force(self.cavity, lig_indices, k_cavity=1000.0)
        cav_force.setForceGroup(GROUP_CAVITY)
        system.addForce(cav_force)

        # 4. Optional Pharmacophore Restraints
        if self.pharma_points:
            pharma_forces = create_pharmacophore_restraint_forces(
                self.pharma_points, ligand_mol, ligand_offset_in_system=lig_start, k_pharma=2000.0
            )
            for pf in pharma_forces:
                system.addForce(pf)

        # 5. Optional Tether Restraints
        if tether_constraints:
            teth_force = create_tether_restraint_force(
                tether_constraints, ligand_offset_in_system=lig_start, k_tether=5000.0
            )
            system.addForce(teth_force)

        # 6. Optional Solvent Restraints
        if self.waters:
            wat_indices = list(range(rec_n, rec_n + wat_n))
            wat_coords = np.array([a.coord for a in self.waters.atoms])
            solv_force = create_solvent_tether_force(wat_indices, wat_coords)
            system.addForce(solv_force)

        # 7. Optional Covalent Adduct Restraints
        if covalent_restraint is None and self.covalent_res is not None:
            covalent_restraint = create_covalent_restraint(self.receptor, ligand_mol, self.covalent_res)

        if covalent_restraint is not None:
            nucl_idx = covalent_restraint.rec_nucleophile_idx
            anchor_idx = covalent_restraint.rec_nucleophile_anchor_idx
            el_idx = lig_start + covalent_restraint.lig_electrophile_idx

            cov_bond = create_covalent_bond_force(covalent_restraint, nucl_idx, el_idx)
            cov_bond.setForceGroup(GROUP_VALENCE)
            system.addForce(cov_bond)

            cov_angle = mm.HarmonicAngleForce()
            cov_angle.addAngle(anchor_idx, nucl_idx, el_idx, covalent_restraint.theta0_rad, covalent_restraint.k_angle)
            cov_angle.setForceGroup(GROUP_VALENCE)
            cov_angle.setName("CovalentAdductAngle")
            system.addForce(cov_angle)

            add_unique_exclusion(nucl_idx, el_idx)
            add_unique_exclusion(anchor_idx, el_idx)

        # 8. Ligand Valence Forces (Bonds, Angles, Ring Triangulation, Stereocenter Locks, Dihedrals)
        self._add_ligand_valence_forces(system, ligand_mol, lig_start, fused_systems)

        combined_sys = MolecularSystem(
            name=f"{self.receptor.name}_{lig_sys.name}",
            atoms=all_atoms,
            bonds=[],
        )

        return system, combined_sys, lig_start, lig_n

    def _add_ligand_valence_forces(
        self,
        system: mm.System,
        ligand_mol: Chem.Mol,
        lig_start: int,
        fused_systems: List[set[int]],
    ) -> None:
        """
        Constructs and adds harmonic bond, angle, and torsional forces to the OpenMM System
        to strictly preserve the ligand's chemical geometry (bond lengths, bond angles,
        chiral stereocenters, aromatic & aliphatic ring geometries, and substituent orientations)
        during Cartesian minimization and simulated annealing.
        """
        conf = ligand_mol.GetConformer()
        bond_force = mm.HarmonicBondForce()
        angle_force = mm.HarmonicAngleForce()
        torsion_force = mm.PeriodicTorsionForce()

        # Helper to ensure unique harmonic bond springs
        added_bonds = set()

        def add_unique_bond(a1_idx: int, a2_idx: int, k_val: float = 500000.0) -> None:
            pair = (min(a1_idx, a2_idx), max(a1_idx, a2_idx))
            if pair not in added_bonds:
                added_bonds.add(pair)
                p1 = np.array(conf.GetAtomPosition(pair[0] - lig_start))
                p2 = np.array(conf.GetAtomPosition(pair[1] - lig_start))
                r0_nm = float(np.linalg.norm(p1 - p2) * 0.1)
                bond_force.addBond(pair[0], pair[1], r0_nm, k_val)

        # 1. Harmonic Bonds (k = 500,000 kJ/(mol*nm^2))
        for b in ligand_mol.GetBonds():
            add_unique_bond(b.GetBeginAtomIdx() + lig_start, b.GetEndAtomIdx() + lig_start)

        # 2. Complete Fused/Individual Ring Triangulation
        for f in fused_systems:
            f_list = sorted(list(f))
            for i in range(len(f_list)):
                for j in range(i + 1, len(f_list)):
                    add_unique_bond(f_list[i] + lig_start, f_list[j] + lig_start)

            # Lock orientation of all substituents directly attached to the ring
            for a in f_list:
                ring_nbrs = [n.GetIdx() for n in ligand_mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in f]
                for nbr in ligand_mol.GetAtomWithIdx(a).GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx not in f:
                        for rn in ring_nbrs:
                            add_unique_bond(nbr_idx + lig_start, rn + lig_start)

        # 3. Proper Ring Perimeter Torsions for every ring (ensures exact ring planarity)
        for ring in ligand_mol.GetRingInfo().AtomRings():
            r_list = list(ring)
            N = len(r_list)
            for i in range(N):
                a1 = r_list[i] + lig_start
                a2 = r_list[(i + 1) % N] + lig_start
                a3 = r_list[(i + 2) % N] + lig_start
                a4 = r_list[(i + 3) % N] + lig_start
                phi0_rad = float(rdMolTransforms.GetDihedralRad(conf, r_list[i], r_list[(i + 1) % N], r_list[(i + 2) % N], r_list[(i + 3) % N]))
                phase = 2.0 * phi0_rad - math.pi
                torsion_force.addTorsion(a1, a2, a3, a4, 2, phase, 2000.0)

        # 4. Tetrahedral Stereocenter Triangulation: locks all chiral sp3 centers rigid against inversion
        for atom in ligand_mol.GetAtoms():
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) == 4:  # Tetrahedral sp3 atom
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        add_unique_bond(nbrs[i] + lig_start, nbrs[j] + lig_start)

        # 5. Harmonic Angles for all angle triplets (k = 2000 kJ/(mol*rad^2))
        for atom in ligand_mol.GetAtoms():
            c_idx = atom.GetIdx()
            neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a1 = neighbors[i]
                    a3 = neighbors[j]
                    theta0_rad = float(rdMolTransforms.GetAngleRad(conf, a1, c_idx, a3))
                    angle_force.addAngle(
                        a1 + lig_start,
                        c_idx + lig_start,
                        a3 + lig_start,
                        theta0_rad,
                        2000.0,
                    )

        # 6. OpenFF / MMFF-style Improper Torsions for all trivalent sp2 and aromatic centers
        for atom in ligand_mol.GetAtoms():
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) == 3 and (atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP2):
                c = atom.GetIdx()
                a1, a2, a3 = nbrs[0], nbrs[1], nbrs[2]
                phi0_rad = float(rdMolTransforms.GetDihedralRad(conf, a1, c, a2, a3))
                phase = 2.0 * phi0_rad - math.pi
                torsion_force.addTorsion(
                    a1 + lig_start, c + lig_start, a2 + lig_start, a3 + lig_start,
                    2, phase, 2000.0
                )

        # 7. Exocyclic Halogen & Terminal Substituent Planarity Locks (e.g. F-benzene, Cl-benzene)
        for atom in ligand_mol.GetAtoms():
            if atom.GetSymbol() in ["F", "CL", "BR", "I", "O", "N", "Cl", "Br"]:
                nbrs = atom.GetNeighbors()
                if len(nbrs) == 1 and nbrs[0].GetIsAromatic():
                    c_nbrs = [n.GetIdx() for n in nbrs[0].GetNeighbors() if n.GetIdx() != atom.GetIdx()]
                    for cn in c_nbrs:
                        add_unique_bond(atom.GetIdx() + lig_start, cn + lig_start)

        # 8. Flexible Rotatable Single Bonds (allows smooth torsional search during annealing)
        for b in ligand_mol.GetBonds():
            if not b.IsInRing() and b.GetBondTypeAsDouble() == 1.0:
                a2 = b.GetBeginAtomIdx()
                a3 = b.GetEndAtomIdx()
                n2 = [n.GetIdx() for n in ligand_mol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a3]
                n3 = [n.GetIdx() for n in ligand_mol.GetAtomWithIdx(a3).GetNeighbors() if n.GetIdx() != a2]
                if n2 and n3:
                    torsion_force.addTorsion(
                        n2[0] + lig_start, a2 + lig_start, a3 + lig_start, n3[0] + lig_start,
                        3, 0.0, 4.0
                    )

        bond_force.setForceGroup(GROUP_VALENCE)
        angle_force.setForceGroup(GROUP_VALENCE)
        torsion_force.setForceGroup(GROUP_VALENCE)

        system.addForce(bond_force)
        system.addForce(angle_force)
        system.addForce(torsion_force)

    def _get_system_positions(self, ligand_mol: Chem.Mol) -> unit.Quantity:
        """Returns full (N, 3) OpenMM positions quantity in nanometers."""
        conf = ligand_mol.GetConformer()
        lig_coords = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
        
        rec_coords = self.receptor.coordinates
        wat_coords = self.waters.coordinates if self.waters else np.zeros((0, 3))
        
        all_coords = np.vstack([rec_coords, wat_coords, lig_coords]) * 0.1  # Å -> nm
        return all_coords * unit.nanometers

    def _full_positions_from_coords(self, lig_coords_angstrom: np.ndarray) -> unit.Quantity:
        """Like _get_system_positions, but takes raw ligand coordinates (Å) directly, avoiding
        the cost of building an RDKit conformer for every candidate pose in a GA population."""
        rec_coords = self.receptor.coordinates
        wat_coords = self.waters.coordinates if self.waters else np.zeros((0, 3))
        all_coords = np.vstack([rec_coords, wat_coords, lig_coords_angstrom]) * 0.1
        return all_coords * unit.nanometers

    def _update_ligand_conformer(
        self,
        ligand_mol: Chem.Mol,
        state_positions: unit.Quantity,
        lig_start: int,
        lig_n: int,
    ) -> Chem.Mol:
        """Updates RDKit ligand conformer with OpenMM state positions."""
        mol_copy = Chem.Mol(ligand_mol)
        conf = mol_copy.GetConformer()
        pos_nm = state_positions.value_in_unit(unit.nanometers)
        for i in range(lig_n):
            p = pos_nm[lig_start + i] * 10.0  # nm -> Å
            conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
        return mol_copy

    def _extract_decomposed_scores(self, context: mm.Context, ligand_mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculates decomposed energy terms from OpenMM Context force groups.
        Every SCORE.INTER.* term is read from its own dedicated force group
        (real per-term physics), not derived as a fixed fraction of a combined
        nonbonded energy.
        """

        def group_e(group: int) -> float:
            return context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )

        vdw_inter_e = group_e(GROUP_VDW_INTER)
        vdw_intra_e = group_e(GROUP_VDW_INTRA)
        polar_inter_e = group_e(GROUP_POLAR_INTER)
        polar_intra_e = group_e(GROUP_POLAR_INTRA)
        repul_e = group_e(GROUP_REPUL)
        hyd_e = group_e(GROUP_HYD)
        val_e = group_e(GROUP_VALENCE)
        cav_e = group_e(GROUP_CAVITY)
        pharma_e = group_e(GROUP_PHARMA)
        tether_e = group_e(GROUP_TETHER)
        solv_e = group_e(GROUP_SOLVENT)

        conv = 1.0 / 4.184
        n_waters = len(self.waters.atoms) // 3 if self.waters else 0
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(ligand_mol)

        inter_vdw = vdw_inter_e * conv
        inter_polar = polar_inter_e * conv
        inter_repul = repul_e * conv
        inter_hyd = hyd_e * conv
        inter_const = self.weights.const * n_waters
        inter_rot = self.weights.rot * n_rot
        score_inter = inter_vdw + inter_polar + inter_repul + inter_hyd + inter_const + inter_rot

        score_intra = (vdw_intra_e + polar_intra_e) * conv

        restr_cavity = cav_e * conv
        restr_pharma = pharma_e * conv
        restr_tether = tether_e * conv
        score_restr = restr_cavity + restr_pharma + restr_tether

        score_system = solv_e * conv

        score_total = score_inter + score_intra + score_restr + score_system

        return {
            "SCORE": score_total,
            "SCORE.INTER": score_inter,
            "SCORE.INTER.VDW": inter_vdw,
            "SCORE.INTER.POLAR": inter_polar,
            "SCORE.INTER.REPUL": inter_repul,
            "SCORE.INTER.HYD": inter_hyd,
            "SCORE.INTER.CONST": inter_const,
            "SCORE.INTER.ROT": inter_rot,
            "SCORE.INTRA": score_intra,
            "SCORE.VALENCE": val_e * conv,
            "SCORE.RESTR": score_restr,
            "SCORE.RESTR.CAVITY": restr_cavity,
            "SCORE.RESTR.PHARMA": restr_pharma,
            "SCORE.RESTR.TETHER": restr_tether,
            "SCORE.SYSTEM": score_system,
        }

    def score(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
    ) -> Dict[str, float]:
        """Scores a single ligand pose without moving coordinates."""
        # Note: unlike the search methods below, score() must not reposition the
        # input pose -- it only needs the covalent restraint *force* included
        # (so covalent bond/angle strain is reflected in the score), resolved
        # against the pose as given.
        covalent_restraint = (
            create_covalent_restraint(self.receptor, ligand_mol, self.covalent_res)
            if self.covalent_res is not None
            else None
        )
        system, _, _, _ = self._build_system(ligand_mol, tether_constraints, covalent_restraint)
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(system, integrator, self.platform)
            if self.platform
            else mm.Context(system, integrator)
        )
        try:
            context.setPositions(self._get_system_positions(ligand_mol))
            scores = self._extract_decomposed_scores(context, ligand_mol)
            return scores
        finally:
            del context, integrator

    def minimize(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        covalent_restraint: Optional[CovalentRestraint] = None,
        max_iterations: int = 500,
        tolerance: float = 0.1,
    ) -> DockingResult:
        """Performs local L-BFGS gradient minimization of ligand pose in cavity."""
        ligand_mol = Chem.Mol(ligand_mol)
        if covalent_restraint is None and self.covalent_res is not None:
            ligand_mol, covalent_restraint = self._prepare_covalent(ligand_mol)

        system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints, covalent_restraint)
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(system, integrator, self.platform)
            if self.platform
            else mm.Context(system, integrator)
        )
        try:
            context.setPositions(self._get_system_positions(ligand_mol))
            mm.LocalEnergyMinimizer.minimize(
                context,
                tolerance=tolerance * (unit.kilojoules_per_mole / unit.nanometer),
                maxIterations=max_iterations,
            )
            state = context.getState(getPositions=True, getEnergy=True)
            scores = self._extract_decomposed_scores(context, ligand_mol)
            min_mol = self._update_ligand_conformer(ligand_mol, state.getPositions(), lig_start, lig_n)

            for k, v in scores.items():
                min_mol.SetProp(k, f"{v:.4f}")

            return DockingResult(
                mol=min_mol,
                score=scores["SCORE"],
                scores=scores,
                run_idx=1,
            )
        finally:
            del context, integrator

    def dock_simulated_annealing(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        n_runs: int = 10,
        t_high: float = 800.0,
        t_low: float = 10.0,
        anneal_steps: int = 10,
        steps_per_temp: int = 100,
        trans_sigma: float = 1.0,
        rot_sigma: float = 25.0,
        torsion_sigma: float = 35.0,
        lamarck_interval: int = 25,
        lamarck_iterations: int = 15,
        minimize_clash_ceiling_kj: float = 2000.0,
        seed: int = 42,
    ) -> List[DockingResult]:
        """
        Chromosome-space Simulated Annealing docking (AutoDock-style): discrete
        Metropolis moves over the same rigid-body + torsion chromosome
        dock_genetic_algorithm uses, with a cooling temperature schedule --
        not literal MD integration.

        Why not literal MD: this used to run real Langevin dynamics
        (LangevinMiddleIntegrator, 1fs timestep) starting from a randomized
        pose. A randomized orientation frequently starts with a severe steric
        clash, and integrating that against 500,000 kJ/(mol*nm^2) valence
        bond springs at 800K is a classic MD instability -- large initial
        forces can blow the integration up rather than anneal away from it
        (empirically: 10/10 blind runs on a simple test system ended in
        unphysical positive energies, regardless of search-box size). A
        discrete chromosome-space proposal can never do this: a rotation/
        translation/torsion change is always a chemically valid rigid-body
        configuration -- there's nothing to integrate, only a score to
        evaluate and a Metropolis accept/reject decision to make. This
        mirrors AutoDock, which never runs literal dynamics either.

        Also Lamarckian (AutoDock LGA-style), like dock_genetic_algorithm:
        every `lamarck_interval` accepted-or-not moves, the current
        chromosome is locally minimized and re-encoded (encode_chromosome),
        so within-run local optimization actually compounds instead of being
        immediately perturbed away by the next proposal.

        `minimize_clash_ceiling_kj`: same guard, same reason, as
        dock_genetic_algorithm's parameter of the same name (see its
        docstring for the measured grid energy-vs-gradient accuracy gap this
        protects against) -- lamark()'s periodic minimize() only runs when the
        chromosome's current raw grid energy is already below this ceiling;
        the Metropolis proposal loop itself never minimizes (energy_of() is a
        pure energy read, no gradient), so it was never exposed to this
        failure mode, but the periodic Lamarckian step is exactly as exposed
        as dock_genetic_algorithm's per-generation one and gets the same fix.
        """
        rng = np.random.default_rng(seed)

        if self.pharma_points and not tether_constraints:
            ligand_mol = align_ligand_to_pharmacophore(ligand_mol, self.pharma_points)

        ligand_mol, covalent_restraint = self._prepare_covalent(ligand_mol)

        torsion_dofs = identify_torsion_dofs(ligand_mol)
        n_t = len(torsion_dofs)

        conf = ligand_mol.GetConformer()
        base_coords = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
        base_local = base_coords - base_coords.mean(axis=0)

        input_torsions = [
            _dihedral_deg(base_coords[d["ref0"]], base_coords[d["a1"]], base_coords[d["a2"]], base_coords[d["ref3"]])
            for d in torsion_dofs
        ]
        input_centroid_offset = base_coords.mean(axis=0) - self.cavity.center
        input_chrom = np.concatenate([input_centroid_offset, [0.0, 0.0, 0.0], input_torsions])

        # Pharma/tether/covalent guidance already pins down (most of) the rigid-body
        # placement, so each run only needs to jitter around it, same as
        # dock_genetic_algorithm's local refinement. Otherwise, this is genuinely
        # blind global search: fully random orientation, small random offset from
        # the cavity center (position is unconstrained beyond "somewhere in the
        # defined pocket").
        guided = bool(self.pharma_points or tether_constraints or covalent_restraint is not None)

        # Two systems: a cheap grid-scored system for the O(runs x anneal_steps x
        # steps_per_temp) inner search loop (falls back to pairwise if the
        # ligand has a non-standard element -- see _build_system), and the real
        # decomposed-score system used only once per run (for the winning pose)
        # so reported SCORE.INTER.* fields stay genuine.
        search_system, _, lig_start, lig_n = self._build_system(
            ligand_mol, tether_constraints, covalent_restraint, fast_search="grid"
        )
        search_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(search_system, search_integrator, self.platform)
            if self.platform
            else mm.Context(search_system, search_integrator)
        )

        report_system, _, _, _ = self._build_system(
            ligand_mol, tether_constraints, covalent_restraint, fast_search=False
        )
        report_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        report_context = (
            mm.Context(report_system, report_integrator, self.platform)
            if self.platform
            else mm.Context(report_system, report_integrator)
        )

        R_GAS_KJ = 0.0083145  # kJ/(mol*K)

        def decode(chrom: np.ndarray) -> np.ndarray:
            return decode_chromosome(chrom, base_local, torsion_dofs, self.cavity.center)

        def energy_of(chrom: np.ndarray) -> float:
            coords = decode(chrom)
            context.setPositions(self._full_positions_from_coords(coords))
            return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        def lamarck(chrom: np.ndarray) -> Tuple[float, np.ndarray]:
            coords = decode(chrom)
            context.setPositions(self._full_positions_from_coords(coords))
            raw_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            if raw_energy > minimize_clash_ceiling_kj:
                return raw_energy, chrom

            mm.LocalEnergyMinimizer.minimize(
                context,
                tolerance=1.0 * (unit.kilojoules_per_mole / unit.nanometer),
                maxIterations=lamarck_iterations,
            )
            state = context.getState(getPositions=True, getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            minimized = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)) * 10.0
            lig_coords = minimized[lig_start:lig_start + lig_n]
            new_chrom = encode_chromosome(lig_coords, base_local, torsion_dofs, self.cavity.center)
            return energy, new_chrom

        results: List[DockingResult] = []
        try:
            for run in range(n_runs):
                if guided:
                    chrom = mutate_chromosome(
                        rng, input_chrom, mutation_rate=1.0, n_torsions=n_t,
                        trans_sigma=0.5, rot_sigma=15.0, torsion_sigma=20.0,
                    )
                else:
                    trans = rng.uniform(-1.5, 1.5, size=3)
                    euler = ScipyRotation.random(random_state=rng).as_euler("xyz", degrees=True)
                    chrom = np.concatenate([trans, euler, input_torsions])

                energy = energy_of(chrom)

                temps = np.linspace(t_high, t_low, num=anneal_steps)
                move_count = 0
                for t in temps:
                    beta = 1.0 / (R_GAS_KJ * max(t, 1.0))
                    for _ in range(steps_per_temp):
                        trial = mutate_chromosome(
                            rng, chrom, mutation_rate=1.0, n_torsions=n_t,
                            trans_sigma=trans_sigma, rot_sigma=rot_sigma, torsion_sigma=torsion_sigma,
                        )
                        trial_energy = energy_of(trial)
                        delta = trial_energy - energy
                        if delta <= 0.0 or rng.random() < math.exp(-beta * delta):
                            chrom, energy = trial, trial_energy

                        move_count += 1
                        if lamarck_interval > 0 and move_count % lamarck_interval == 0:
                            energy, chrom = lamarck(chrom)

                best_coords = decode(chrom)
                mol_variant = copy.deepcopy(ligand_mol)
                vconf = mol_variant.GetConformer()
                for i in range(lig_n):
                    p = best_coords[i]
                    vconf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))

                report_context.setPositions(self._full_positions_from_coords(best_coords))
                mm.LocalEnergyMinimizer.minimize(
                    report_context,
                    tolerance=0.1 * (unit.kilojoules_per_mole / unit.nanometer),
                    maxIterations=500,
                )
                state = report_context.getState(getPositions=True, getEnergy=True)
                scores = self._extract_decomposed_scores(report_context, mol_variant)
                docked_mol = self._update_ligand_conformer(mol_variant, state.getPositions(), lig_start, lig_n)

                for k, v in scores.items():
                    docked_mol.SetProp(k, f"{v:.4f}")

                results.append(
                    DockingResult(
                        mol=docked_mol,
                        score=scores["SCORE"],
                        scores=scores,
                        run_idx=run + 1,
                    )
                )
        finally:
            del context, search_integrator
            del report_context, report_integrator

        results.sort(key=lambda r: r.score)
        return results

    def dock_monte_carlo_minimization(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        n_runs: int = 10,
        num_steps: int = 30,
        temperature: float = 1.2,
        trans_amplitude: float = 2.0,
        rot_amplitude_deg: float = 60.0,
        lbfgs_maxiter: int = 15,
        lbfgs_maxiter_final: int = 40,
        seed: int = 42,
    ) -> List[DockingResult]:
        """
        Monte-Carlo-with-Minimization (MCM / basin-hopping) docking, directly
        modeled on AutoDock Vina/smina's core search loop (monte_carlo.cpp:
        single_run -- mutate_conf, then quasi_newton BFGS minimize, THEN
        Metropolis-accept the *minimized* energy, every single step).

        This is a structurally different search than dock_simulated_annealing:
        that method takes many small-Gaussian-jitter Metropolis steps and only
        periodically (every lamarck_interval moves) locally minimizes, so most
        visited states are raw, un-minimized proposal energies -- noisy
        compared to a converged local optimum, and the chain can drift through
        many similar-scoring-but-not-actually-relaxed poses before a Lamarckian
        polish ever fires. Here every step is: one coarse single-DOF jump
        (mutate_chromosome_vina_style) -> immediate local minimization via
        lbfgs_minimize (gradient_minimizer.py's finite-difference L-BFGS-B,
        operating on the same cheap grid-scored energy this class's other
        search methods use) -> Metropolis test on the MINIMIZED energy. Every
        state actually compared by the Markov chain is therefore already a
        genuine local-basin minimum, which is the mechanistic reason Vina-
        family tools converge to precise poses far more reliably than a
        periodically-polished raw random walk: the search is over BASINS, not
        over noisy raw conformations.

        lbfgs_maxiter is deliberately small (mirrors Vina's cheap hunt_cap-
        capped look-ahead minimize used to score every candidate); once a run
        finishes, the single best chromosome found gets one more, fuller
        lbfgs_maxiter_final pass (mirrors Vina's authentic_v full minimize)
        before the final real decomposed-score report step.
        """
        rng = np.random.default_rng(seed)

        if self.pharma_points and not tether_constraints:
            ligand_mol = align_ligand_to_pharmacophore(ligand_mol, self.pharma_points)

        ligand_mol, covalent_restraint = self._prepare_covalent(ligand_mol)

        torsion_dofs = identify_torsion_dofs(ligand_mol)
        n_t = len(torsion_dofs)

        conf = ligand_mol.GetConformer()
        base_coords = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
        base_local = base_coords - base_coords.mean(axis=0)

        input_torsions = [
            _dihedral_deg(base_coords[d["ref0"]], base_coords[d["a1"]], base_coords[d["a2"]], base_coords[d["ref3"]])
            for d in torsion_dofs
        ]
        input_centroid_offset = base_coords.mean(axis=0) - self.cavity.center
        input_chrom = np.concatenate([input_centroid_offset, [0.0, 0.0, 0.0], input_torsions])
        guided = bool(self.pharma_points or tether_constraints or covalent_restraint is not None)

        search_system, _, lig_start, lig_n = self._build_system(
            ligand_mol, tether_constraints, covalent_restraint, fast_search="grid"
        )
        search_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(search_system, search_integrator, self.platform)
            if self.platform
            else mm.Context(search_system, search_integrator)
        )

        report_system, _, _, _ = self._build_system(
            ligand_mol, tether_constraints, covalent_restraint, fast_search=False
        )
        report_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        report_context = (
            mm.Context(report_system, report_integrator, self.platform)
            if self.platform
            else mm.Context(report_system, report_integrator)
        )

        def decode(chrom: np.ndarray) -> np.ndarray:
            return decode_chromosome(chrom, base_local, torsion_dofs, self.cavity.center)

        def energy_of(chrom: np.ndarray) -> float:
            coords = decode(chrom)
            context.setPositions(self._full_positions_from_coords(coords))
            return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Finite-difference step sizes: coarser for degrees than Angstroms,
        # matching the very different natural scales of this chromosome.
        step_vec = np.concatenate([
            np.full(3, 0.02),                 # translation, Angstroms
            np.full(3, 1.0),                  # euler angles, degrees
            np.full(max(n_t, 0), 1.0),        # torsions, degrees
        ])

        def local_minimize(chrom: np.ndarray, maxiter: int) -> Tuple[np.ndarray, float]:
            res = lbfgs_minimize(energy_of, chrom, step_size=step_vec, max_iterations=maxiter)
            return res.x, res.fun

        results: List[DockingResult] = []
        try:
            for run in range(n_runs):
                if guided:
                    chrom = mutate_chromosome(
                        rng, input_chrom, mutation_rate=1.0, n_torsions=n_t,
                        trans_sigma=0.5, rot_sigma=15.0, torsion_sigma=20.0,
                    )
                else:
                    trans = rng.uniform(-1.5, 1.5, size=3)
                    euler = ScipyRotation.random(random_state=rng).as_euler("xyz", degrees=True)
                    chrom = np.concatenate([trans, euler, input_torsions])

                # Mirrors monte_carlo.cpp's single_run exactly: one initial
                # cheap minimize, num_steps of (coarse move -> cheap minimize
                # -> Metropolis-accept-the-minimized-energy) tracking the best
                # CHEAP-minimized chromosome seen, then ONE final, fuller
                # "authentic" minimize on that best chromosome at the very
                # end -- not a full re-minimize on every improvement, which
                # is what made the first version of this method expensive
                # without a matching accuracy benefit (Vina's own single_run
                # only pays the expensive authentic_v minimize once per run).
                chrom, energy = local_minimize(chrom, lbfgs_maxiter)
                best_chrom, best_energy = chrom, energy

                for _ in range(num_steps):
                    candidate = mutate_chromosome_vina_style(
                        rng, chrom, n_torsions=n_t,
                        trans_amplitude=trans_amplitude, rot_amplitude_deg=rot_amplitude_deg,
                    )
                    candidate, cand_energy = local_minimize(candidate, lbfgs_maxiter)

                    if cand_energy <= energy or rng.random() < math.exp(-(cand_energy - energy) / max(temperature, 1e-6)):
                        chrom, energy = candidate, cand_energy
                        if energy < best_energy:
                            best_chrom, best_energy = chrom, energy

                best_chrom, best_energy = local_minimize(best_chrom, lbfgs_maxiter_final)
                best_coords = decode(best_chrom)
                mol_variant = copy.deepcopy(ligand_mol)
                vconf = mol_variant.GetConformer()
                for i in range(lig_n):
                    p = best_coords[i]
                    vconf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))

                report_context.setPositions(self._full_positions_from_coords(best_coords))
                mm.LocalEnergyMinimizer.minimize(
                    report_context,
                    tolerance=0.1 * (unit.kilojoules_per_mole / unit.nanometer),
                    maxIterations=500,
                )
                state = report_context.getState(getPositions=True, getEnergy=True)
                scores = self._extract_decomposed_scores(report_context, mol_variant)
                docked_mol = self._update_ligand_conformer(mol_variant, state.getPositions(), lig_start, lig_n)

                for k, v in scores.items():
                    docked_mol.SetProp(k, f"{v:.4f}")

                results.append(
                    DockingResult(
                        mol=docked_mol,
                        score=scores["SCORE"],
                        scores=scores,
                        run_idx=run + 1,
                    )
                )
        finally:
            del context, search_integrator
            del report_context, report_integrator

        results.sort(key=lambda r: r.score)
        return results

    def dock_monte_carlo(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        n_steps: int = 100,
        temperature_k: float = 300.0,
        translation_scale: float = 0.5,  # Å
        rotation_scale: float = 15.0,     # degrees
        torsion_scale: float = 30.0,      # degrees
        minimize_each_step: bool = True,
        seed: int = 42,
    ) -> DockingResult:
        """
        Runs Metropolis Monte Carlo with Minimization (MCM / Basin-Hopping) docking.
        Explores discrete rigid-body translations, rotations, and internal rotatable bond torsions.
        """
        random.seed(seed)
        np.random.seed(seed)

        if self.pharma_points and not tether_constraints:
            ligand_mol = align_ligand_to_pharmacophore(ligand_mol, self.pharma_points)

        ligand_mol, covalent_restraint = self._prepare_covalent(ligand_mol)

        # 1. Identify rotatable single bonds and their moving subtrees
        rot_bonds = []
        for b in ligand_mol.GetBonds():
            if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE:
                a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if len(ligand_mol.GetAtomWithIdx(a1).GetNeighbors()) > 1 and len(ligand_mol.GetAtomWithIdx(a2).GetNeighbors()) > 1:
                    visited = {a1}
                    q = [a2]
                    st = set()
                    while q:
                        curr = q.pop(0)
                        if curr not in visited:
                            visited.add(curr)
                            st.add(curr)
                            for nbr in ligand_mol.GetAtomWithIdx(curr).GetNeighbors():
                                if nbr.GetIdx() not in visited:
                                    q.append(nbr.GetIdx())
                    rot_bonds.append((a1, a2, list(st)))

        system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints, covalent_restraint)
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(system, integrator, self.platform)
            if self.platform
            else mm.Context(system, integrator)
        )

        try:
            curr_mol = copy.deepcopy(ligand_mol)
            context.setPositions(self._get_system_positions(curr_mol))

            # Initial relaxation
            mm.LocalEnergyMinimizer.minimize(
                context,
                tolerance=0.1 * (unit.kilojoules_per_mole / unit.nanometer),
                maxIterations=200,
            )
            state = context.getState(getPositions=True, getEnergy=True)
            curr_mol = self._update_ligand_conformer(curr_mol, state.getPositions(), lig_start, lig_n)
            curr_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / 4.184
            curr_coords = np.array([curr_mol.GetConformer().GetAtomPosition(i) for i in range(lig_n)])

            best_mol = copy.deepcopy(curr_mol)
            best_energy = curr_energy
            best_scores = self._extract_decomposed_scores(context, curr_mol)

            # Record trajectory frames
            trajectory_frames: List[Chem.Mol] = []
            f0 = copy.deepcopy(curr_mol)
            f0.SetProp("MC_FRAME", "0")
            f0.SetProp("MOVE_TYPE", "INITIAL")
            f0.SetProp("ACCEPTED", "1")
            f0.SetProp("ENERGY", f"{curr_energy:.4f}")
            f0.SetProp("BEST_ENERGY", f"{best_energy:.4f}")
            trajectory_frames.append(f0)

            from scipy.spatial.transform import Rotation as ScipyRotation
            beta = 1.0 / (0.001987 * temperature_k)

            for step in range(n_steps):
                trial_coords = np.copy(curr_coords)
                move_type = np.random.choice(
                    ["trans", "rot", "torsion"],
                    p=[0.2, 0.2, 0.6] if rot_bonds else [0.5, 0.5, 0.0],
                )

                if move_type == "trans":
                    trial_coords += np.random.uniform(-translation_scale, translation_scale, size=3)
                elif move_type == "rot":
                    axis = np.random.normal(size=3)
                    axis /= (np.linalg.norm(axis) + 1e-12)
                    angle_rad = math.radians(np.random.uniform(-rotation_scale, rotation_scale))
                    rot_mat = ScipyRotation.from_rotvec(angle_rad * axis).as_matrix()
                    c = np.mean(trial_coords, axis=0)
                    trial_coords = c + np.dot(trial_coords - c, rot_mat.T)
                elif move_type == "torsion" and rot_bonds:
                    a1, a2, st = rot_bonds[np.random.randint(len(rot_bonds))]
                    p1 = trial_coords[a1]
                    p2 = trial_coords[a2]
                    v = p2 - p1
                    v /= (np.linalg.norm(v) + 1e-12)
                    d_phi = math.radians(np.random.uniform(-torsion_scale, torsion_scale))
                    rot_mat = ScipyRotation.from_rotvec(d_phi * v).as_matrix()
                    trial_coords[st] = p1 + np.dot(trial_coords[st] - p1, rot_mat.T)

                trial_mol = copy.deepcopy(curr_mol)
                conf = trial_mol.GetConformer()
                for i in range(lig_n):
                    conf.SetAtomPosition(i, (float(trial_coords[i, 0]), float(trial_coords[i, 1]), float(trial_coords[i, 2])))

                context.setPositions(self._get_system_positions(trial_mol))

                if minimize_each_step:
                    mm.LocalEnergyMinimizer.minimize(
                        context,
                        tolerance=0.5 * (unit.kilojoules_per_mole / unit.nanometer),
                        maxIterations=50,
                    )

                state = context.getState(getPositions=True, getEnergy=True)
                trial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / 4.184
                delta_e = trial_energy - curr_energy

                is_accepted = (delta_e <= 0.0 or random.random() < math.exp(- beta * delta_e))

                if is_accepted:
                    curr_energy = trial_energy
                    curr_mol = self._update_ligand_conformer(trial_mol, state.getPositions(), lig_start, lig_n)
                    curr_coords = np.array([curr_mol.GetConformer().GetAtomPosition(i) for i in range(lig_n)])

                    if curr_energy < best_energy:
                        best_energy = curr_energy
                        best_mol = copy.deepcopy(curr_mol)
                        best_scores = self._extract_decomposed_scores(context, trial_mol)

                # Record frame in trajectory
                frame_mol = copy.deepcopy(curr_mol)
                frame_mol.SetProp("MC_FRAME", str(step + 1))
                frame_mol.SetProp("MOVE_TYPE", move_type.upper())
                frame_mol.SetProp("ACCEPTED", "1" if is_accepted else "0")
                frame_mol.SetProp("ENERGY", f"{curr_energy:.4f}")
                frame_mol.SetProp("TRIAL_ENERGY", f"{trial_energy:.4f}")
                frame_mol.SetProp("DELTA_E", f"{delta_e:.4f}")
                frame_mol.SetProp("BEST_ENERGY", f"{best_energy:.4f}")
                trajectory_frames.append(frame_mol)

            for k, v in best_scores.items():
                best_mol.SetProp(k, f"{v:.4f}")

            return DockingResult(
                mol=best_mol,
                score=best_scores["SCORE"],
                scores=best_scores,
                run_idx=1,
                trajectory=trajectory_frames,
            )
        finally:
            del context, integrator

    def dock_genetic_algorithm(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        population_size: int = 20,
        n_generations: int = 15,
        mutation_rate: float = 0.2,
        mutation_trans_sigma: float = 0.3,
        mutation_rot_sigma: float = 8.0,
        mutation_torsion_sigma: float = 15.0,
        init_trans_sigma: float = 1.0,
        init_rot_sigma: float = 15.0,
        init_torsion_sigma: float = 30.0,
        elite_fraction: float = 0.15,
        tournament_size: int = 3,
        fitness_minimize_iterations: int = 10,
        minimize_clash_ceiling_kj: float = 2000.0,
        n_runs: int = 5,
        seed: int = 42,
    ) -> List[DockingResult]:
        """
        Genetic Algorithm *local refinement* docking: rDock's own default search
        engine is a population-based GA over rigid-body + torsional DOFs, but
        blind global GA search (random start anywhere in the cavity) turned out
        not to reliably recover the binding pose in a practical compute budget
        even with Lamarckian local-minimization fitness -- a genuinely hard
        global-optimization problem that real docking codes throw much larger
        populations/generations of compiled code at. Rather than chase that
        further, this GA refines the population *around the input ligand_mol's
        given pose* (e.g. already pharmacophore-aligned or tether-aligned, or
        simply a reasonable starting geometry the caller supplies), the same
        "aligned pose + jitter" convention dock_simulated_annealing already
        uses for its pharma/tether case. This is a much better-posed problem:
        find the best nearby pose, not find the pocket from scratch.

        Each chromosome is [3 translation genes (Å), 3 rigid-body rotation genes
        (Euler degrees), 1 torsion gene per rotatable bond (absolute dihedral,
        degrees)], encoded relative to the input pose (all-zero rigid-body genes
        exactly reproduce it). The initial population is the input pose plus
        Gaussian jitter (`init_*_sigma`); each of `n_runs` independent
        populations then evolves for `n_generations` generations via tournament
        selection, uniform crossover, smaller-scale Gaussian mutation
        (`mutation_*_sigma`), and elitism. Fitness is Lamarckian/Baldwinian: a
        short local minimization (`fitness_minimize_iterations`) is applied
        before reading back each candidate's energy, since local refinement
        still benefits from smoothing out torsion-induced steric noise. The
        fittest individual of each run gets a full final minimization and is
        returned with genuinely decomposed SCORE.INTER.* fields.

        `minimize_clash_ceiling_kj` guards the grid-scored search system
        (fast_search="grid", see _build_system) against a real, measured
        failure mode: OpenMM's Continuous3DFunction grid reproduces *energy*
        values quite accurately even for a badly clashing pose (measured 4.75%
        relative error on a real severely-clashed test pose, +47,284 kJ/mol
        exact vs +49,531 kJ/mol grid), but its *gradient* is far less
        trustworthy in the steep soft-core-4-8 VDW repulsive wall -- mean
        per-atom force error was ~36,500 kJ/mol/nm at that same pose (vs. 0-370
        for every other scoring term at the same pose), because a small
        interpolation error at a steep point translates into a large slope
        error. Widening the VDW smoothing window (gridding._smoothed_vdw_curve)
        does not fix this -- it was swept from 0.008nm up to 0.75nm and only
        made the *energy* error worse (4.75% to >100%), so the fix is not
        another smoothing recalibration. Since mutate_chromosome's proposals
        are large, unconstrained jumps (unlike AutoDock/Vina's own small local
        moves), a freshly mutated individual is often still in a severe clash
        -- exactly the regime where LocalEnergyMinimizer.minimize(), which
        follows the grid's gradient, would get misled before the pose ever
        reaches a region the grid can be trusted in (empirically this once
        left a GA run 4.1A from the crystal pose it should have refined onto,
        vs 0.024A under exact pairwise scoring). The fix: evaluate() checks
        each candidate's *raw* (unminimized) grid energy first -- accurate per
        the 4.75% figure above -- and only runs the gradient-following
        minimization step when that raw energy is already below this ceiling,
        i.e. only on candidates in the region the grid's forces can be
        trusted. Candidates still above the ceiling are ranked by their raw
        grid energy instead (correctly deprioritized without ever asking the
        grid for an unreliable gradient). 2000 kJ/mol is a pragmatic
        heuristic -- comfortably below the ~47,000 kJ/mol scale of the severe
        clash actually measured, comfortably above typical bound-pose energies
        (tens to a few hundred kJ/mol) -- not a value derived from a precise
        force-vs-energy error curve.
        """
        rng = np.random.default_rng(seed)

        if self.pharma_points and not tether_constraints:
            ligand_mol = align_ligand_to_pharmacophore(ligand_mol, self.pharma_points)

        ligand_mol, covalent_restraint = self._prepare_covalent(ligand_mol)

        torsion_dofs = identify_torsion_dofs(ligand_mol)
        n_t = len(torsion_dofs)

        conf = ligand_mol.GetConformer()
        base_coords = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
        base_local = base_coords - base_coords.mean(axis=0)

        # Chromosome encoding the input pose exactly: zero rigid-body genes (the
        # frame is centered on base_local already) + the ligand's own current
        # torsion angles as absolute gene values.
        input_torsions = [
            _dihedral_deg(base_coords[d["ref0"]], base_coords[d["a1"]], base_coords[d["a2"]], base_coords[d["ref3"]])
            for d in torsion_dofs
        ]
        input_centroid_offset = base_coords.mean(axis=0) - self.cavity.center
        base_chrom = np.concatenate([input_centroid_offset, [0.0, 0.0, 0.0], input_torsions])

        # Two systems: a cheap grid-scored system for the O(pop x gens x runs) inner
        # search loop (falls back to pairwise for non-standard elements -- see
        # _build_system), and the real decomposed-score system used only once per
        # run (for the winning individual) so reported SCORE.INTER.* fields stay genuine.
        search_system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints, covalent_restraint, fast_search="grid")
        search_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(search_system, search_integrator, self.platform)
            if self.platform
            else mm.Context(search_system, search_integrator)
        )

        report_system, _, _, _ = self._build_system(ligand_mol, tether_constraints, covalent_restraint, fast_search=False)
        report_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        report_context = (
            mm.Context(report_system, report_integrator, self.platform)
            if self.platform
            else mm.Context(report_system, report_integrator)
        )

        n_elite = max(1, int(round(elite_fraction * population_size)))

        def decode(chrom: np.ndarray) -> np.ndarray:
            return decode_chromosome(chrom, base_local, torsion_dofs, self.cavity.center)

        def evaluate(chrom: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Lamarckian evaluation (AutoDock LGA-style): after the short local
            minimization, the improved Cartesian result is re-encoded back into
            the chromosome via encode_chromosome (Kabsch superposition), so the
            improvement is inherited by future generations instead of being
            discarded the moment fitness is scored (the previous Baldwinian
            behavior -- minimize to *score* the individual, but keep evolving
            the original, un-improved genotype).

            Always reads the raw (unminimized) grid energy first and only runs
            LocalEnergyMinimizer when it's below minimize_clash_ceiling_kj --
            see that parameter's docstring above for why: the grid's gradient
            is unreliable precisely on the badly-clashing poses a fresh large
            mutation often produces, even though its raw energy value is
            accurate there, so a still-clashing candidate is ranked on that
            accurate raw energy instead of being minimized with an untrustworthy
            gradient.
            """
            coords = decode(chrom)
            context.setPositions(self._full_positions_from_coords(coords))
            raw_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            if fitness_minimize_iterations > 0 and raw_energy <= minimize_clash_ceiling_kj:
                mm.LocalEnergyMinimizer.minimize(
                    context,
                    tolerance=1.0 * (unit.kilojoules_per_mole / unit.nanometer),
                    maxIterations=fitness_minimize_iterations,
                )
                state = context.getState(getPositions=True, getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                minimized_coords = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer) * 10.0
                lig_coords = np.array(minimized_coords[lig_start:lig_start + lig_n])
                new_chrom = encode_chromosome(lig_coords, base_local, torsion_dofs, self.cavity.center)
                return energy, new_chrom
            return raw_energy, chrom

        results: List[DockingResult] = []
        try:
            for run in range(n_runs):
                population = [base_chrom.copy()] + [
                    mutate_chromosome(
                        rng, base_chrom, mutation_rate=1.0, n_torsions=n_t,
                        trans_sigma=init_trans_sigma,
                        rot_sigma=init_rot_sigma,
                        torsion_sigma=init_torsion_sigma,
                    )
                    for _ in range(population_size - 1)
                ]
                evaluated = [evaluate(c) for c in population]
                scores = [e for e, _ in evaluated]
                population = [c for _, c in evaluated]

                for gen in range(n_generations):
                    order = np.argsort(scores)
                    new_population = [population[i].copy() for i in order[:n_elite]]

                    while len(new_population) < population_size:
                        p1 = tournament_select(rng, population, scores, tournament_size)
                        p2 = tournament_select(rng, population, scores, tournament_size)
                        child = crossover_chromosomes(rng, p1, p2)
                        child = mutate_chromosome(
                            rng, child, mutation_rate, n_t,
                            trans_sigma=mutation_trans_sigma,
                            rot_sigma=mutation_rot_sigma,
                            torsion_sigma=mutation_torsion_sigma,
                        )
                        new_population.append(child)

                    evaluated = [evaluate(c) for c in new_population]
                    scores = [e for e, _ in evaluated]
                    population = [c for _, c in evaluated]

                best_idx = int(np.argmin(scores))
                best_chrom = population[best_idx]
                best_coords = decode(best_chrom)

                mol_variant = copy.deepcopy(ligand_mol)
                vconf = mol_variant.GetConformer()
                for i in range(lig_n):
                    vconf.SetAtomPosition(i, (float(best_coords[i, 0]), float(best_coords[i, 1]), float(best_coords[i, 2])))

                report_context.setPositions(self._full_positions_from_coords(best_coords))
                mm.LocalEnergyMinimizer.minimize(
                    report_context,
                    tolerance=0.1 * (unit.kilojoules_per_mole / unit.nanometer),
                    maxIterations=500,
                )
                state = report_context.getState(getPositions=True, getEnergy=True)
                final_scores = self._extract_decomposed_scores(report_context, mol_variant)
                final_mol = self._update_ligand_conformer(mol_variant, state.getPositions(), lig_start, lig_n)

                for k, v in final_scores.items():
                    final_mol.SetProp(k, f"{v:.4f}")

                results.append(
                    DockingResult(
                        mol=final_mol,
                        score=final_scores["SCORE"],
                        scores=final_scores,
                        run_idx=run + 1,
                    )
                )
        finally:
            del context, search_integrator
            del report_context, report_integrator

        results.sort(key=lambda r: r.score)
        return results
