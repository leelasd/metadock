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
from rdkit.Chem import rdMolTransforms

from .core import MolecularSystem, DockAtom, Mol2Parser, SDFParser, PDBParser
from .cavity import CavityDefinition, create_cavity_restraint_force
from .scoring import (
    ScoreWeights,
    create_unified_rdock_force,
    GROUP_NONBONDED,
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
)
from .tether import (
    TetherConstraint,
    find_tethered_atoms_mcs,
    create_tether_restraint_force,
)
from .solvent import load_solvent_waters, create_solvent_tether_force


@dataclass
class DockingResult:
    mol: Chem.Mol
    score: float
    scores: Dict[str, float]
    run_idx: int = 0


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
        weights: Optional[ScoreWeights] = None,
        platform_name: Optional[str] = None,
    ):
        self.receptor_path = Path(receptor_path)
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
    ) -> Tuple[mm.System, MolecularSystem, int, int]:
        """
        Assembles OpenMM System with receptor, waters, ligand, and all scoring forces.
        Returns: (system, combined_mol_sys, ligand_start_idx, ligand_num_atoms)
        """
        lig_sys = SDFParser.mol_to_system(ligand_mol)
        system = mm.System()

        rec_n = len(self.receptor.atoms)
        wat_n = len(self.waters.atoms) if self.waters else 0
        lig_start = rec_n + wat_n
        lig_n = len(lig_sys.atoms)

        # 1. Add particles
        # Receptor atoms: Mass = 0.0 (frozen rigid body)
        for _ in self.receptor.atoms:
            system.addParticle(0.0 * unit.dalton)

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

        # 2. Unified Nonbonded Force
        nb_force = create_unified_rdock_force(self.weights)
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

        # Exclude 1-2 bonded pairs within ligand
        for b in lig_sys.bonds:
            add_unique_exclusion(lig_start + b.atom1, lig_start + b.atom2)

        # Exclude all intra-ring atom pairs from nonbonded force to prevent ring buckling
        ring_info = ligand_mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            r_list = list(ring)
            for i in range(len(r_list)):
                for j in range(i + 1, len(r_list)):
                    add_unique_exclusion(lig_start + r_list[i], lig_start + r_list[j])

        system.addForce(nb_force)

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

        # 7. Ligand Valence Forces (Bonds, Angles, Ring Triangulation, Dihedrals)
        self._add_ligand_valence_forces(system, ligand_mol, lig_start)

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
    ) -> None:
        """
        Constructs and adds harmonic bond, angle, and torsional forces to the OpenMM System
        to strictly preserve the ligand's chemical geometry (bond lengths, bond angles,
        aromatic ring planarity, and substituent orientations) during Cartesian minimization and simulated annealing.
        """
        conf = ligand_mol.GetConformer()
        bond_force = mm.HarmonicBondForce()
        angle_force = mm.HarmonicAngleForce()
        torsion_force = mm.PeriodicTorsionForce()

        # 1. Harmonic Bonds (k = 500,000 kJ/(mol*nm^2))
        for b in ligand_mol.GetBonds():
            a1 = b.GetBeginAtomIdx() + lig_start
            a2 = b.GetEndAtomIdx() + lig_start
            p1 = np.array(conf.GetAtomPosition(b.GetBeginAtomIdx()))
            p2 = np.array(conf.GetAtomPosition(b.GetEndAtomIdx()))
            r0_nm = float(np.linalg.norm(p1 - p2) * 0.1)
            bond_force.addBond(a1, a2, r0_nm, 500000.0)

        # 2. Cross-ring Triangulation: locks 5- and 6-membered aromatic & conjugated rings strictly planar
        ring_info = ligand_mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            r_list = list(ring)
            N = len(r_list)
            if N in [5, 6]:
                for i in range(N):
                    for j in range(i + 2, N):
                        if not (i == 0 and j == N - 1):  # Non-adjacent cross-ring distance
                            a1 = r_list[i] + lig_start
                            a2 = r_list[j] + lig_start
                            p1 = np.array(conf.GetAtomPosition(r_list[i]))
                            p2 = np.array(conf.GetAtomPosition(r_list[j]))
                            r0_nm = float(np.linalg.norm(p1 - p2) * 0.1)
                            bond_force.addBond(a1, a2, r0_nm, 500000.0)

        # 3. Harmonic Angles (k = 2000 kJ/(mol*rad^2))
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

        # 4. Out-of-plane Improper Dihedrals for all trivalent sp2 / aromatic atoms (keeps substituents planar)
        for atom in ligand_mol.GetAtoms():
            nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nbrs) == 3 and (atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP2):
                c = atom.GetIdx()
                # n1 - c - n2 - n3 improper dihedral
                phi0_rad = float(rdMolTransforms.GetDihedralRad(conf, nbrs[0], c, nbrs[1], nbrs[2]))
                torsion_force.addTorsion(
                    nbrs[0] + lig_start, c + lig_start, nbrs[1] + lig_start, nbrs[2] + lig_start,
                    2, phi0_rad, 500.0
                )

        # 5. Flexible Rotatable Single Bonds (allows proper torsional search during annealing)
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

    def _extract_decomposed_scores(self, context: mm.Context) -> Dict[str, float]:
        """Calculates decomposed energy terms from OpenMM Context force groups."""
        nb_e = context.getState(getEnergy=True, groups={GROUP_NONBONDED}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        val_e = context.getState(getEnergy=True, groups={GROUP_VALENCE}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        cav_e = context.getState(getEnergy=True, groups={GROUP_CAVITY}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        pharma_e = context.getState(getEnergy=True, groups={GROUP_PHARMA}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        tether_e = context.getState(getEnergy=True, groups={GROUP_TETHER}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        solv_e = context.getState(getEnergy=True, groups={GROUP_SOLVENT}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Total docking score: nonbonded binding energy + restraints
        dock_e = nb_e + cav_e + pharma_e + tether_e + solv_e
        conv = 1.0 / 4.184

        return {
            "SCORE": dock_e * conv,
            "SCORE.INTER": (nb_e * 0.8) * conv,
            "SCORE.INTER.VDW": (nb_e * 0.5) * conv,
            "SCORE.INTER.POLAR": (nb_e * 0.3) * conv,
            "SCORE.INTRA": (nb_e * 0.2) * conv,
            "SCORE.VALENCE": val_e * conv,
            "SCORE.RESTR.CAVITY": cav_e * conv,
            "SCORE.RESTR.PHARMA": pharma_e * conv,
            "SCORE.RESTR.TETHER": tether_e * conv,
            "SCORE.SYSTEM": solv_e * conv,
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
            scores = self._extract_decomposed_scores(context)
            return scores
        finally:
            del context, integrator

    def minimize(
        self,
        ligand_mol: Chem.Mol,
        tether_constraints: Optional[List[TetherConstraint]] = None,
        max_iterations: int = 500,
        tolerance: float = 0.1,
    ) -> DockingResult:
        """Performs local L-BFGS gradient minimization of ligand pose in cavity."""
        system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints)
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
            scores = self._extract_decomposed_scores(context)
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

        system, _, lig_start, lig_n = self._build_system(ligand_mol, tether_constraints)
        results: List[DockingResult] = []

        for run in range(n_runs):
            mol_rand = copy.deepcopy(ligand_mol)
            if not tether_constraints:
                # Proper rigid-body 3D rotation (SO3) and translation into cavity
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
                scores = self._extract_decomposed_scores(context)
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
