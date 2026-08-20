"""
Receptor Side-Chain Kinematics Engine for openmm-dock.
Parameterizes active-site amino acid side chains as robotic kinematic chains
driven by torsional joint hinges (chi1, chi2, chi3, chi4) with zero backbone distortion.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
import openmm as mm
from openmm import unit

from .core import MolecularSystem, DockAtom
from .engine import DockingEngine


# Standard amino acid chi dihedral atom definitions (IUPAC nomenclature)
CHI_DEFINITIONS = {
    # 1 chi angle
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "VAL": [("N", "CA", "CB", "CG1")],
    # 2 chi angles
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    # 3 chi angles
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    # 4 chi angles
    "LYS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
    "ARG": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
}


@dataclass
class ChiJoint:
    """Represents a single chi dihedral hinge in an amino acid side chain."""
    chi_idx: int                 # 1 for chi1, 2 for chi2, etc.
    res_name: str
    res_num: int
    chain_id: str
    atom_names: Tuple[str, str, str, str] # (A, B, C, D)
    axis_atom_indices: Tuple[int, int]    # Indices of B and C (rotation axis)
    moving_atom_indices: List[int]        # All downstream side-chain atoms rotated


@dataclass
class FlexibleResidue:
    """Represents an active-site amino acid with articulated kinematic side chains."""
    res_name: str
    res_num: int
    chain_id: str
    ca_atom_idx: int
    all_atom_indices: List[int]
    chi_joints: List[ChiJoint]


class ReceptorSideChainKinematics:
    """
    Kinematic articulation engine for receptor pocket side chains.
    Enables induced-fit side-chain rotamer flexing (chi1 - chi4) on GPU.
    """
    def __init__(
        self,
        receptor_pdb_path: Path | str,
        pocket_center: np.ndarray,
        flex_radius: float = 6.0
    ):
        self.pdb_path = Path(receptor_pdb_path)
        self.pocket_center = np.asarray(pocket_center, dtype=np.float64)
        self.flex_radius = flex_radius
        
        # 1. Parse PDB lines and atoms
        self.pdb_lines = self.pdb_path.read_text().splitlines()
        self.atom_lines: List[str] = []
        self.atom_names: List[str] = []
        self.res_names: List[str] = []
        self.res_nums: List[int] = []
        self.chain_ids: List[str] = []
        self.atom_coords: List[np.ndarray] = []
        
        for l in self.pdb_lines:
            if l.startswith("ATOM  ") or l.startswith("HETATM"):
                self.atom_lines.append(l)
                self.atom_names.append(l[12:16].strip())
                self.res_names.append(l[17:20].strip())
                self.chain_ids.append(l[21])
                self.res_nums.append(int(l[22:26]))
                x = float(l[30:38])
                y = float(l[38:46])
                z = float(l[46:54])
                self.atom_coords.append(np.array([x, y, z]))
                
        self.num_atoms = len(self.atom_coords)
        self.base_coords = np.array(self.atom_coords, dtype=np.float64)
        
        # 2. Identify active-site residues within flex_radius of pocket_center
        self.flex_residues: List[FlexibleResidue] = []
        self._build_flexible_residues()
        
        total_chi = sum(len(r.chi_joints) for r in self.flex_residues)
        print(f"[*] Receptor Side-Chain Kinematics Initialized on {self.pdb_path.name}")
        print(f"[*] Pocket Centroid: {self.pocket_center.round(2)} | Flex Radius: {self.flex_radius:.1f} Å")
        print(f"[*] Identified {len(self.flex_residues)} flexible active-site residues ({total_chi} total chi joint hinges)")

    def _build_flexible_residues(self):
        """Finds active-site residues and constructs their kinematic chi chains."""
        # Group atom indices by residue
        res_map: Dict[Tuple[str, int, str], List[int]] = {}
        for idx in range(self.num_atoms):
            key = (self.res_names[idx], self.res_nums[idx], self.chain_ids[idx])
            res_map.setdefault(key, []).append(idx)
            
        for (rname, rnum, ch), atom_indices in res_map.items():
            if rname not in CHI_DEFINITIONS:
                continue
                
            # Check if any atom in residue is within flex_radius of pocket center
            coords_res = self.base_coords[atom_indices]
            dists = np.linalg.norm(coords_res - self.pocket_center, axis=1)
            if np.min(dists) > self.flex_radius:
                continue
                
            # Find CA atom
            ca_idx = None
            for idx in atom_indices:
                if self.atom_names[idx] == "CA":
                    ca_idx = idx
                    break
            if ca_idx is None:
                continue
                
            # Map atom names to indices for this residue
            name_to_idx = {self.atom_names[i]: i for i in atom_indices}
            
            # Build Chi joints
            chi_defs = CHI_DEFINITIONS[rname]
            chi_joints: List[ChiJoint] = []
            
            for chi_num, (aA, aB, aC, aD) in enumerate(chi_defs, start=1):
                if aB not in name_to_idx or aC not in name_to_idx or aD not in name_to_idx:
                    continue
                idxB = name_to_idx[aB]
                idxC = name_to_idx[aC]
                
                # Downstream side-chain atoms are all atoms beyond C in the residue hierarchy
                # For standard amino acids, downstream atoms contain atom C and all subsequent side-chain atoms
                downstream = [
                    i for i in atom_indices
                    if self.atom_names[i] not in ["N", "CA", "C", "O", "H", "HA"]
                    and self._is_downstream(rname, chi_num, self.atom_names[i])
                ]
                
                if not downstream:
                    continue
                    
                chi_joints.append(ChiJoint(
                    chi_idx=chi_num,
                    res_name=rname,
                    res_num=rnum,
                    chain_id=ch,
                    atom_names=(aA, aB, aC, aD),
                    axis_atom_indices=(idxB, idxC),
                    moving_atom_indices=downstream
                ))
                
            if chi_joints:
                self.flex_residues.append(FlexibleResidue(
                    res_name=rname,
                    res_num=rnum,
                    chain_id=ch,
                    ca_atom_idx=ca_idx,
                    all_atom_indices=atom_indices,
                    chi_joints=chi_joints
                ))

    def _is_downstream(self, res_name: str, chi_idx: int, atom_name: str) -> bool:
        """Determines if atom_name is downstream of chi_idx in standard amino acid side chains."""
        hierarchy = {
            "CB": 0,
            "CG": 1, "CG1": 1, "CG2": 1, "OG": 1, "OG1": 1, "SG": 1,
            "CD": 2, "CD1": 2, "CD2": 2, "SD": 2, "OD1": 2, "OD2": 2, "ND1": 2, "ND2": 2,
            "CE": 3, "CE1": 3, "CE2": 3, "CE3": 3, "NE": 3, "NE1": 3, "NE2": 3, "OE1": 3, "OE2": 3,
            "CZ": 4, "CZ2": 4, "CZ3": 4, "NZ": 4, "NH1": 4, "NH2": 4, "OH": 4,
            "CH2": 5,
        }
        # Also include hydrogen atoms on downstream heavy atoms
        for h_prefix, level in [("HB", 0), ("HG", 1), ("HD", 2), ("HE", 3), ("HZ", 4), ("HH", 4)]:
            if atom_name.startswith(h_prefix):
                return level >= chi_idx
                
        atom_level = hierarchy.get(atom_name, -1)
        return atom_level >= chi_idx

    def forward_kinematics_sidechains(
        self,
        chi_perturbations: Dict[Tuple[str, int, int], float], # ((res_name, res_num, chi_num) -> angle_rad)
        base_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes 3D receptor coordinates after applying forward kinematic rotations
        to active-site side-chain chi hinges. Backbone remains 100% rigid.
        """
        coords = self.base_coords.copy() if base_coords is None else base_coords.copy()
        
        for res in self.flex_residues:
            for joint in res.chi_joints:
                key = (res.res_name, res.res_num, joint.chi_idx)
                angle_rad = chi_perturbations.get(key, 0.0)
                if abs(angle_rad) < 1e-7:
                    continue
                    
                idxB, idxC = joint.axis_atom_indices
                origin = coords[idxB]
                axis = coords[idxC] - origin
                norm = np.linalg.norm(axis)
                if norm < 1e-6:
                    continue
                u = axis / norm
                rot = ScipyRotation.from_rotvec(u * angle_rad).as_matrix()
                
                sub_p = coords[joint.moving_atom_indices] - origin
                coords[joint.moving_atom_indices] = sub_p.dot(rot.T) + origin
                
        return coords

    def write_pdb_frame(self, coords: np.ndarray, out_path: Path | str):
        """Writes updated receptor coordinates to a clean PDB file."""
        lines_out = []
        for idx, l in enumerate(self.atom_lines):
            x, y, z = coords[idx]
            # Format PDB coordinate columns (30-54)
            line_mod = f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}"
            lines_out.append(line_mod)
        lines_out.append("END")
        Path(out_path).write_text("\n".join(lines_out) + "\n")
