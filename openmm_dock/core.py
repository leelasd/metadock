"""
Core data structures and parsers for receptor and ligand topologies and parameters.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Default Van der Waals parameters (sigma in nm, epsilon in kJ/mol)
# Mapped by element symbol and Sybyl atom types
VDW_PARAMS: Dict[str, Tuple[float, float]] = {
    # Elements
    "H": (0.20, 0.10),
    "C": (0.36, 0.40),
    "N": (0.33, 0.70),
    "O": (0.30, 0.85),
    "F": (0.31, 0.25),
    "P": (0.40, 0.84),
    "S": (0.40, 1.05),
    "CL": (0.39, 1.15),
    "BR": (0.41, 1.35),
    "I": (0.44, 1.65),
    "MG": (0.25, 0.50),
    "CA": (0.28, 0.50),
    "ZN": (0.24, 0.50),
    "FE": (0.24, 0.50),
    # Specific Sybyl types
    "C.3": (0.38, 0.40),
    "C.2": (0.36, 0.45),
    "C.AR": (0.36, 0.45),
    "C.1": (0.34, 0.45),
    "N.3": (0.33, 0.70),
    "N.2": (0.33, 0.70),
    "N.AR": (0.33, 0.70),
    "N.AM": (0.33, 0.70),
    "N.PL3": (0.33, 0.70),
    "N.4": (0.33, 0.70),
    "O.3": (0.30, 0.85),
    "O.2": (0.30, 0.85),
    "O.CO2": (0.30, 0.85),
    "S.3": (0.40, 1.05),
    "S.2": (0.40, 1.05),
    "S.O": (0.40, 1.05),
    "S.O2": (0.40, 1.05),
    "P.3": (0.40, 0.84),
}


@dataclass
class DockAtom:
    idx: int
    name: str
    element: str
    sybyl_type: str
    charge: float
    coord: np.ndarray  # Shape (3,) in Angstroms
    is_donor: bool = False
    is_acceptor: bool = False
    is_polar: bool = False
    is_rotatable_root: bool = False
    residue_name: str = ""
    residue_idx: int = 1
    chain: str = "A"

    @property
    def sigma(self) -> float:
        """Van der Waals sigma in nanometers."""
        st = self.sybyl_type.upper()
        el = self.element.upper()
        if st in VDW_PARAMS:
            return VDW_PARAMS[st][0]
        if el in VDW_PARAMS:
            return VDW_PARAMS[el][0]
        return 0.35  # default nm

    @property
    def epsilon(self) -> float:
        """Van der Waals epsilon in kJ/mol."""
        st = self.sybyl_type.upper()
        el = self.element.upper()
        if st in VDW_PARAMS:
            return VDW_PARAMS[st][1]
        if el in VDW_PARAMS:
            return VDW_PARAMS[el][1]
        return 0.40  # default kJ/mol


def _perceive_bonds_by_distance(atoms: List["DockAtom"], cutoff: float = 1.7) -> List["DockBond"]:
    """
    Simple distance-based bond perception for parsers (PDBParser) that don't
    get explicit connectivity from their source file: any atom pair closer
    than `cutoff` Angstroms is treated as bonded. Used only to recover
    accurate is_donor flags (need to know which N/O/S atoms have a real
    bonded H) -- not a general-purpose topology (no bond order/rotatability
    is inferred from it).
    """
    coords = np.array([a.coord for a in atoms])
    n = len(atoms)
    bonds: List[DockBond] = []
    for i in range(n):
        d2 = np.sum((coords[i + 1:] - coords[i]) ** 2, axis=1)
        close = np.nonzero(d2 < cutoff ** 2)[0]
        for c in close:
            bonds.append(DockBond(atom1=i, atom2=i + 1 + int(c), order="1"))
    return bonds


# Standard amino acid sidechain donor atom names, for structures with no
# explicit hydrogens at all (a common case for a raw, unprotonated
# receptor.pdb) -- bond-based detection can never find a donor when the H
# atom simply isn't in the file, so this is a real fallback, not a
# duplicate check. Backbone amide N is a donor for every residue except
# proline (its ring nitrogen is a secondary/tertiary amine with no N-H).
# HIS includes both ND1 and NE2 since tautomer state (which one is
# protonated) isn't resolved from heavy-atom coordinates alone.
_PROLINE_RESIDUES = {"PRO"}
_SIDECHAIN_DONOR_ATOMS: Dict[str, Tuple[str, ...]] = {
    "SER": ("OG",), "THR": ("OG1",), "CYS": ("SG",), "TYR": ("OH",),
    "ASN": ("ND2",), "GLN": ("NE2",), "LYS": ("NZ",),
    "ARG": ("NE", "NH1", "NH2"), "HIS": ("ND1", "NE2"), "TRP": ("NE1",),
}


def _assign_standard_residue_donor_fallback(atoms: List["DockAtom"]) -> None:
    """Sets is_donor=True for known standard-amino-acid donor atoms
    (backbone amide N, defined sidechain donors) WITHOUT requiring an
    explicit bonded H -- see module comment above. Only ever turns a flag
    ON, never off, so it's safe to run after (or instead of) the bond-based
    pass: a structure that DOES have real H atoms keeps that more precise
    signal; one that doesn't still gets correct standard-chemistry donors
    instead of silently reporting zero."""
    for a in atoms:
        if a.name.strip().upper() == "N" and a.residue_name.strip().upper() not in _PROLINE_RESIDUES:
            a.is_donor = True
        elif a.name.strip().upper() in _SIDECHAIN_DONOR_ATOMS.get(a.residue_name.strip().upper(), ()):
            a.is_donor = True


def _assign_donor_flags_from_bonds(atoms: List["DockAtom"], bonds: List["DockBond"]) -> None:
    """
    Overwrites each atom's is_donor flag based on REAL bonded connectivity
    (does this N/O/S atom have an attached hydrogen?) instead of a fragile
    heuristic based on the atom's own name/sybyl-type string containing the
    literal letter "H" -- which silently misses, for example, every
    backbone amide N (PDB atom name is just "N", not "NH") despite it being
    the single most common hydrogen-bond donor in any protein, and every
    ligand atom whose hydrogens were added as explicit graph neighbors
    (same underlying bug class as the GetTotalNumHs() vs.
    GetTotalNumHs(includeNeighbors=True) fix already applied in
    pharmacophore.py's find_ligand_pharma_features -- here duplicated
    independently in three different from-scratch parsers rather than
    shared code, so each needed its own fix). Mutates atoms in place.
    """
    has_h_neighbor = [False] * len(atoms)
    for b in bonds:
        a1, a2 = atoms[b.atom1], atoms[b.atom2]
        if a2.element.upper() == "H":
            has_h_neighbor[b.atom1] = True
        if a1.element.upper() == "H":
            has_h_neighbor[b.atom2] = True
    for i, a in enumerate(atoms):
        if a.element.upper() in ("N", "O", "S"):
            a.is_donor = has_h_neighbor[i]


@dataclass
class DockBond:
    atom1: int  # 0-indexed
    atom2: int  # 0-indexed
    order: str = "1"
    is_rotatable: bool = False


@dataclass
class MolecularSystem:
    name: str
    atoms: List[DockAtom] = field(default_factory=list)
    bonds: List[DockBond] = field(default_factory=list)

    @property
    def coordinates(self) -> np.ndarray:
        """Coordinates array in Angstroms with shape (N, 3)."""
        if not self.atoms:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([a.coord for a in self.atoms], dtype=np.float64)

    @coordinates.setter
    def coordinates(self, coords: np.ndarray):
        for i, a in enumerate(self.atoms):
            a.coord = coords[i]

    def get_center(self) -> np.ndarray:
        coords = self.coordinates
        if len(coords) == 0:
            return np.zeros(3)
        return np.mean(coords, axis=0)


class Mol2Parser:
    """Robust Tripos Mol2 parser supporting multi-residue proteins, RNA, and ligands."""

    @staticmethod
    def parse(filepath: Path | str) -> MolecularSystem:
        filepath = Path(filepath)
        lines = filepath.read_text().splitlines()
        
        atoms: List[DockAtom] = []
        bonds: List[DockBond] = []
        mol_name = filepath.stem
        
        section = None
        atom_idx_map = {}  # 1-indexed in mol2 -> 0-indexed internal

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("@<TRIPOS>"):
                section = line_str[9:].strip().upper()
                continue

            if section == "MOLECULE":
                if mol_name == filepath.stem and not line_str.startswith("#"):
                    mol_name = line_str
            elif section == "ATOM":
                parts = line_str.split()
                if len(parts) >= 6:
                    raw_id = int(parts[0])
                    name = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    sybyl_type = parts[5]
                    
                    res_idx = 1
                    res_name = "LIG"
                    charge = 0.0
                    
                    if len(parts) >= 7:
                        try:
                            res_idx = int(parts[6])
                        except ValueError:
                            res_name = parts[6]
                    if len(parts) >= 8:
                        res_name = parts[7]
                    if len(parts) >= 9:
                        try:
                            charge = float(parts[8])
                        except ValueError:
                            pass

                    # Extract element from sybyl type (e.g., C.3 -> C, N.ar -> N)
                    element = sybyl_type.split(".")[0]
                    if len(element) > 2 or not element.isalpha():
                        # Try to extract from name
                        element = "".join([c for c in name if c.isalpha()])[:2]
                        if not element:
                            element = "C"

                    idx = len(atoms)
                    atom_idx_map[raw_id] = idx
                    
                    is_don = element.upper() in ["N", "O"] and ("H" in name or "H" in sybyl_type)
                    is_acc = element.upper() in ["O", "N", "F"] and not sybyl_type.startswith("N.4")
                    is_pol = element.upper() in ["N", "O", "S", "P", "F", "CL"]

                    atoms.append(
                        DockAtom(
                            idx=idx,
                            name=name,
                            element=element,
                            sybyl_type=sybyl_type,
                            charge=charge,
                            coord=np.array([x, y, z], dtype=np.float64),
                            is_donor=is_don,
                            is_acceptor=is_acc,
                            is_polar=is_pol,
                            residue_name=res_name,
                            residue_idx=res_idx,
                        )
                    )
            elif section == "BOND":
                parts = line_str.split()
                if len(parts) >= 4:
                    try:
                        a1_raw = int(parts[1])
                        a2_raw = int(parts[2])
                        order = parts[3]
                        if a1_raw in atom_idx_map and a2_raw in atom_idx_map:
                            bonds.append(DockBond(atom1=atom_idx_map[a1_raw], atom2=atom_idx_map[a2_raw], order=order))
                    except ValueError:
                        pass

        _assign_donor_flags_from_bonds(atoms, bonds)
        _assign_standard_residue_donor_fallback(atoms)
        return MolecularSystem(name=mol_name, atoms=atoms, bonds=bonds)


def sanitize_hydrogens_3d(mol: Chem.Mol) -> Chem.Mol:
    """
    Checks and repairs 3D hydrogen positions if missing or distorted,
    optimizing hydrogen positions with all heavy atoms rigidly locked.
    """
    needs_repair = False
    h_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    if len(h_atoms) == 0:
        needs_repair = True
    else:
        conf = mol.GetConformer()
        for b in mol.GetBonds():
            a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx())
            a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx())
            if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
                p1 = np.array(conf.GetAtomPosition(b.GetBeginAtomIdx()))
                p2 = np.array(conf.GetAtomPosition(b.GetEndAtomIdx()))
                d = np.linalg.norm(p1 - p2)
                if d < 0.75 or d > 1.35:
                    needs_repair = True
                    break

    if not needs_repair:
        return mol

    mol_no_h = Chem.RemoveHs(mol)
    mol_h = Chem.AddHs(mol_no_h, addCoords=True)
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol_h)
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp)
            for i, a in enumerate(mol_h.GetAtoms()):
                if a.GetAtomicNum() > 1:
                    ff.AddFixedPoint(i)
            ff.Minimize(maxIts=500)
    except Exception:
        pass
    return mol_h


class SDFParser:
    """Parser for SDF / MOL files using RDKit."""

    @staticmethod
    def load_molecules(filepath: Path | str, sanitize: bool = True) -> List[Chem.Mol]:
        filepath = Path(filepath)
        suppl = Chem.SDMolSupplier(str(filepath), removeHs=False, sanitize=sanitize)
        mols = []
        for i, mol in enumerate(suppl):
            if mol is not None:
                mols.append(sanitize_hydrogens_3d(mol))
            else:
                try:
                    mol_relaxed = Chem.MolFromMolFile(str(filepath), removeHs=False, sanitize=False)
                    if mol_relaxed is not None:
                        Chem.Kekulize(mol_relaxed, clearAromaticFlags=True)
                        mols.append(sanitize_hydrogens_3d(mol_relaxed))
                except Exception:
                    pass
        return mols

    @staticmethod
    def mol_to_system(mol: Chem.Mol, name: str = "LIG") -> MolecularSystem:
        """Converts an RDKit Mol object into a MolecularSystem with Gasteiger charges."""
        mol = sanitize_hydrogens_3d(mol)
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass

        conf = mol.GetConformer()
        atoms: List[DockAtom] = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            element = atom.GetSymbol()
            
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                if math.isnan(charge) or math.isinf(charge):
                    charge = 0.0
            except Exception:
                charge = float(atom.GetFormalCharge())

            is_acc = element in ["O", "N", "F"] and atom.GetFormalCharge() <= 0
            # includeNeighbors=True is required: GetTotalNumHs() alone only
            # reports the packed implicit/explicit-H *count* property and
            # returns 0 once a molecule's hydrogens exist as explicit graph
            # neighbor atoms (e.g. after Chem.AddHs()), silently finding
            # zero donors -- same bug class already fixed in
            # pharmacophore.py's find_ligand_pharma_features.
            is_don = element in ["N", "O"] and atom.GetTotalNumHs(includeNeighbors=True) > 0
            is_pol = element in ["N", "O", "S", "P", "F", "CL"]

            atoms.append(
                DockAtom(
                    idx=i,
                    name=f"{element}{i+1}",
                    element=element,
                    sybyl_type=f"{element}.3",
                    charge=charge,
                    coord=np.array([pos.x, pos.y, pos.z], dtype=np.float64),
                    is_donor=is_don,
                    is_acceptor=is_acc,
                    is_polar=is_pol,
                    residue_name="LIG",
                    residue_idx=1,
                )
            )

        bonds: List[DockBond] = []
        for b in mol.GetBonds():
            bonds.append(
                DockBond(
                    atom1=b.GetBeginAtomIdx(),
                    atom2=b.GetEndAtomIdx(),
                    order=str(b.GetBondTypeAsDouble()),
                    is_rotatable=not b.IsInRing() and b.GetBondTypeAsDouble() == 1.0,
                )
            )

        return MolecularSystem(name=name, atoms=atoms, bonds=bonds)


class PDBParser:
    """Parser for PDB files (proteins, waters, ligands)."""

    @staticmethod
    def parse(filepath: Path | str) -> MolecularSystem:
        filepath = Path(filepath)
        lines = filepath.read_text().splitlines()
        atoms: List[DockAtom] = []
        mol_name = filepath.stem

        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    raw_id = int(line[6:11].strip())
                    name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain = line[21:22].strip()
                    res_idx = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    element = line[76:78].strip()
                    if not element:
                        element = "".join([c for c in name if c.isalpha()])[:2]
                        if not element:
                            element = "C"

                    idx = len(atoms)
                    is_pol = element.upper() in ["N", "O", "S", "P", "F", "CL"]
                    # Placeholder -- overwritten below by
                    # _assign_donor_flags_from_bonds once every atom (and a
                    # perceived bond graph) is available; the name-substring
                    # check this used to use directly missed e.g. every
                    # backbone amide N (PDB atom name is just "N").
                    is_don = False
                    is_acc = element.upper() in ["O", "N"]

                    atoms.append(
                        DockAtom(
                            idx=idx,
                            name=name,
                            element=element,
                            sybyl_type=f"{element}.3",
                            charge=0.0,
                            coord=np.array([x, y, z], dtype=np.float64),
                            is_donor=is_don,
                            is_acceptor=is_acc,
                            is_polar=is_pol,
                            residue_name=res_name,
                            residue_idx=res_idx,
                            chain=chain,
                        )
                    )
                except Exception:
                    continue

        bonds = _perceive_bonds_by_distance(atoms) if atoms else []
        _assign_donor_flags_from_bonds(atoms, bonds)
        _assign_standard_residue_donor_fallback(atoms)
        return MolecularSystem(name=mol_name, atoms=atoms, bonds=bonds)
