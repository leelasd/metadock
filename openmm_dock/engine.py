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
from .pharmacophore import (
    PharmaPoint,
    parse_pharma_restr,
    create_pharmacophore_restraint_forces,
    align_ligand_to_pharmacophore,
)
from .tether import (
    TetherConstraint,
    find_tethered_atoms_mcs,
    create_tether_restraint_force,
)
from .solvent import load_solvent_waters, create_solvent_tether_force
from .covalent import (
    CovalentRestraint,
    create_covalent_restraint,
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
    dofs: List[Dict[str, Any]] = []
    for b in ligand_mol.GetBonds():
        if b.IsInRing() or b.GetBondTypeAsDouble() != 1.0:
            continue
        a1_idx, a2_idx = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        a1_atom, a2_atom = ligand_mol.GetAtomWithIdx(a1_idx), ligand_mol.GetAtomWithIdx(a2_idx)
        if a1_atom.GetDegree() < 2 or a2_atom.GetDegree() < 2:
            continue

        # BFS subtree of atoms on the a2 side of the bond (a2 inclusive, a1 excluded)
        visited = {a1_idx}
        queue = [a2_idx]
        subtree: List[int] = []
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            subtree.append(curr)
            for nbr in ligand_mol.GetAtomWithIdx(curr).GetNeighbors():
                if nbr.GetIdx() not in visited:
                    queue.append(nbr.GetIdx())

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


def random_chromosome(rng: np.random.Generator, trans_range: float, n_torsions: int) -> np.ndarray:
    return np.concatenate([
        rng.uniform(-trans_range, trans_range, size=3),
        rng.uniform(-180.0, 180.0, size=3),
        rng.uniform(-180.0, 180.0, size=n_torsions),
    ])


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

    def _build_system(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        covalent_restraint: Optional[CovalentRestraint] = None,
        fast_search: bool = False,
    ) -> Tuple[mm.System, MolecularSystem, int, int]:
        """
        Assembles OpenMM System with receptor, waters, ligand, and all scoring forces.
        Returns: (system, combined_mol_sys, ligand_start_idx, ligand_num_atoms)

        fast_search=True swaps the six-way decomposed nonbonded force for one
        combined CustomNonbondedForce with identical physics (see
        scoring.create_combined_search_force). Six separate forces means six
        separate neighbor-list rebuilds against the full receptor per
        setPositions() call -- irrelevant for one-off scoring, but decisive for
        an inner search loop (GA/MC) evaluating thousands of large-jump
        candidate poses. Use fast_search=True only for search-loop fitness
        ranking; use the default (real decomposed terms) for anything whose
        score is actually reported.
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
        #    or one combined force for fast search-loop ranking -- see fast_search docstring above)
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

            cov_bond = mm.HarmonicBondForce()
            cov_bond.addBond(nucl_idx, el_idx, covalent_restraint.r0_nm, covalent_restraint.k_bond)
            cov_bond.setForceGroup(GROUP_VALENCE)
            cov_bond.setName("CovalentAdductBond")
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
        system, _, _, _ = self._build_system(ligand_mol, tether_constraints)
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
            covalent_restraint = create_covalent_restraint(self.receptor, ligand_mol, self.covalent_res)

        if covalent_restraint is not None:
            conf = ligand_mol.GetConformer()
            nucl_pos = self.receptor.atoms[covalent_restraint.rec_nucleophile_idx].coord
            el_pos = np.array(conf.GetAtomPosition(covalent_restraint.lig_electrophile_idx))
            shift = (nucl_pos + np.array([0.05, 0.05, 0.05])) - el_pos
            for i in range(ligand_mol.GetNumAtoms()):
                p = np.array(conf.GetAtomPosition(i)) + shift
                conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))

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
        seed: int = 42,
    ) -> List[DockingResult]:
        """GPU-accelerated Simulated Annealing Molecular Dynamics (SAMD) Docking."""
        random.seed(seed)
        np.random.seed(seed)

        # If pharmacophore points exist and no tether, align ligand to pharmacophore before building system
        if self.pharma_points and not tether_constraints:
            ligand_mol = align_ligand_to_pharmacophore(ligand_mol, self.pharma_points)

        system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints)
        results: List[DockingResult] = []

        for run in range(n_runs):
            mol_rand = copy.deepcopy(ligand_mol)
            if self.pharma_points or tether_constraints:
                # Pharmacophore-constrained or Tethered docking: sample around aligned pose with rotational/translational jitter
                from scipy.spatial.transform import Rotation as ScipyRotation
                angles = np.random.uniform(-15.0, 15.0, size=3)
                rot_mat = ScipyRotation.from_euler("xyz", angles, degrees=True).as_matrix()
                trans = np.random.uniform(-0.5, 0.5, size=3)
                conf = mol_rand.GetConformer()
                coords = np.array([conf.GetAtomPosition(j) for j in range(mol_rand.GetNumAtoms())])
                mean_p = np.mean(coords, axis=0)
                centered = coords - mean_p
                rotated = np.dot(centered, rot_mat.T)
                new_coords = mean_p + rotated + trans
                for i in range(mol_rand.GetNumAtoms()):
                    conf.SetAtomPosition(i, (float(new_coords[i, 0]), float(new_coords[i, 1]), float(new_coords[i, 2])))
            else:
                # Unconstrained docking: full rigid-body 3D rotation (SO3) and translation into cavity
                from scipy.spatial.transform import Rotation as ScipyRotation
                rot_mat = ScipyRotation.random().as_matrix()
                trans = np.random.uniform(-1.5, 1.5, size=3)
                conf = mol_rand.GetConformer()
                center = self.cavity.center
                lig_atoms_count = mol_rand.GetNumAtoms()
                coords = np.array([conf.GetAtomPosition(j) for j in range(lig_atoms_count)])
                mean_p = np.mean(coords, axis=0)
                centered = coords - mean_p
                rotated = np.dot(centered, rot_mat.T)
                new_coords = center + rotated + trans
                for i in range(lig_atoms_count):
                    conf.SetAtomPosition(i, (float(new_coords[i, 0]), float(new_coords[i, 1]), float(new_coords[i, 2])))

            integrator = mm.LangevinMiddleIntegrator(
                t_high * unit.kelvin,
                2.0 / unit.picoseconds,
                1.0 * unit.femtoseconds,
            )
            context = (
                mm.Context(system, integrator, self.platform)
                if self.platform
                else mm.Context(system, integrator)
            )

            try:
                context.setPositions(self._get_system_positions(mol_rand))

                # Annealing schedule
                temps = np.linspace(t_high, t_low, num=anneal_steps)
                for t in temps:
                    integrator.setTemperature(t * unit.kelvin)
                    integrator.step(steps_per_temp)

                # Local Minimization
                mm.LocalEnergyMinimizer.minimize(
                    context,
                    tolerance=0.1 * (unit.kilojoules_per_mole / unit.nanometer),
                    maxIterations=500,
                )

                state = context.getState(getPositions=True, getEnergy=True)
                scores = self._extract_decomposed_scores(context, mol_rand)
                docked_mol = self._update_ligand_conformer(mol_rand, state.getPositions(), lig_start, lig_n)

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
                del context, integrator

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

        system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints)
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
        population_size: int = 40,
        n_generations: int = 30,
        mutation_rate: float = 0.2,
        mutation_trans_sigma: float = 1.0,
        mutation_rot_sigma: float = 25.0,
        mutation_torsion_sigma: float = 35.0,
        elite_fraction: float = 0.15,
        tournament_size: int = 3,
        translation_search_radius: Optional[float] = None,
        fitness_minimize_iterations: int = 25,
        n_runs: int = 10,
        seed: int = 42,
    ) -> List[DockingResult]:
        """
        rDock-style Genetic Algorithm docking: this is rDock's own default search
        engine (population-based GA over rigid-body + torsional DOFs), as opposed
        to the SAMD / Monte Carlo protocols above, which are OpenMM-native stand-ins.

        Each chromosome is [3 translation genes (Å), 3 rigid-body rotation genes
        (Euler degrees), 1 torsion gene per rotatable bond (absolute dihedral,
        degrees)]. RDKit supplies the ligand topology and rotatable-bond
        identification; OpenMM (via the shared rDock-scoring Context) supplies
        the fitness function. Each of `n_runs` independent GA populations evolves
        for `n_generations` generations with tournament selection, uniform
        crossover, Gaussian mutation, and elitism; the fittest individual of each
        run is locally minimized and returned.

        Fitness is evaluated Lamarckian/Baldwinian-style: every candidate pose is
        given a short local minimization (`fitness_minimize_iterations`) before
        its energy is read back. A raw, unrelaxed soft-core score is dominated by
        random steric clashes (any random torsion/orientation typically produces
        energies orders of magnitude worse than the true minimum), which makes
        the unrelaxed landscape too noisy for selection to act on -- this mirrors
        why AutoDock's Lamarckian GA and rDock's own GA+simplex hybrid both
        couple global GA search with local refinement rather than scoring raw
        poses. `translation_search_radius` bounds the initial/mutated centroid
        search box; it defaults to a modest fraction of the cavity radius rather
        than the full cavity radius, since the cavity restraint's radius already
        has to enclose the whole ligand extent (much bigger than the sensible
        range for the ligand *centroid* to wander).
        """
        rng = np.random.default_rng(seed)

        if self.pharma_points and not tether_constraints:
            ligand_mol = align_ligand_to_pharmacophore(ligand_mol, self.pharma_points)

        torsion_dofs = identify_torsion_dofs(ligand_mol)
        n_t = len(torsion_dofs)

        conf = ligand_mol.GetConformer()
        base_coords = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
        base_local = base_coords - base_coords.mean(axis=0)

        # Two systems: a cheap single-force system for the O(pop x gens x runs) inner
        # search loop, and the real decomposed-score system used only once per run
        # (for the winning individual) so reported SCORE.INTER.* fields stay genuine.
        search_system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints, fast_search=True)
        search_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = (
            mm.Context(search_system, search_integrator, self.platform)
            if self.platform
            else mm.Context(search_system, search_integrator)
        )

        report_system, _, _, _ = self._build_system(ligand_mol, tether_constraints, fast_search=False)
        report_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        report_context = (
            mm.Context(report_system, report_integrator, self.platform)
            if self.platform
            else mm.Context(report_system, report_integrator)
        )

        if translation_search_radius is not None:
            trans_range = translation_search_radius
        else:
            trans_range = max(min(self.cavity.radius, 8.0), 1.0)
        n_elite = max(1, int(round(elite_fraction * population_size)))

        def decode(chrom: np.ndarray) -> np.ndarray:
            return decode_chromosome(chrom, base_local, torsion_dofs, self.cavity.center)

        def fitness(chrom: np.ndarray) -> float:
            coords = decode(chrom)
            context.setPositions(self._full_positions_from_coords(coords))
            if fitness_minimize_iterations > 0:
                mm.LocalEnergyMinimizer.minimize(
                    context,
                    tolerance=1.0 * (unit.kilojoules_per_mole / unit.nanometer),
                    maxIterations=fitness_minimize_iterations,
                )
            return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        results: List[DockingResult] = []
        try:
            for run in range(n_runs):
                population = [random_chromosome(rng, trans_range, n_t) for _ in range(population_size)]
                scores = [fitness(c) for c in population]

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

                    population = new_population
                    scores = [fitness(c) for c in population]

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
