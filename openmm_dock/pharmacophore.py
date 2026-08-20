"""
Pharmacophore feature extraction and OpenMM restraint forces.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import openmm as mm
from rdkit import Chem
from .scoring import GROUP_PHARMA


@dataclass
class PharmaPoint:
    x: float          # in Angstroms
    y: float          # in Angstroms
    z: float          # in Angstroms
    tolerance: float  # in Angstroms
    ptype: str        # 'Aro', 'Acc', 'Don', 'Hyd'
    is_optional: bool = False

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


def parse_pharma_restr(filepath: Path | str) -> List[PharmaPoint]:
    """Parses an rDock pharma.restr or optional_pharma.restr file."""
    filepath = Path(filepath)
    points: List[PharmaPoint] = []
    lines = filepath.read_text().splitlines()

    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue
        parts = line_str.split()
        if len(parts) >= 5:
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                tol = float(parts[3])
                ptype = parts[4].strip()
                points.append(PharmaPoint(x=x, y=y, z=z, tolerance=tol, ptype=ptype))
            except ValueError:
                continue

    return points


def find_ligand_pharma_features(mol: Chem.Mol) -> Dict[str, List[List[int]]]:
    """
    Finds atom indices corresponding to pharmacophore feature groups in an RDKit Mol:
    - 'Aro': List of list of atom indices for each aromatic ring
    - 'Acc': List of [atom_idx] for hydrogen bond acceptors
    - 'Don': List of [atom_idx] for hydrogen bond donors
    - 'Hyd': List of [atom_idx] for hydrophobic atoms
    """
    features: Dict[str, List[List[int]]] = {
        "Aro": [],
        "Acc": [],
        "Don": [],
        "Hyd": [],
    }

    # 1. Aromatic rings
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            features["Aro"].append(list(ring))

    # Fallback: if no strictly aromatic ring found, include all planar 5/6-membered rings
    if not features["Aro"]:
        for ring in ring_info.AtomRings():
            if len(ring) in [5, 6]:
                features["Aro"].append(list(ring))

    # 2. Acceptors, Donors, Hydrophobic
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        elem = atom.GetSymbol()

        # Acceptors: O, N, F
        if elem in ["O", "N", "F"] and atom.GetFormalCharge() <= 0:
            features["Acc"].append([idx])

        # Donors: N, O with attached hydrogen
        if elem in ["N", "O"] and atom.GetTotalNumHs() > 0:
            features["Don"].append([idx])

        # Hydrophobic
        if elem in ["Cl", "Br", "I"]:
            features["Hyd"].append([idx])
        elif elem == "C":
            has_hetero = any(nbr.GetSymbol() in ["N", "O", "S", "P", "F"] for nbr in atom.GetNeighbors())
            if not has_hetero:
                features["Hyd"].append([idx])

    return features


def create_pharmacophore_restraint_forces(
    pharma_points: List[PharmaPoint],
    ligand_mol: Chem.Mol,
    ligand_offset_in_system: int = 0,
    k_pharma: float = 2000.0,  # kJ/(mol * nm^2)
) -> List[mm.Force]:
    """
    Creates OpenMM restraint force applying flat-bottom penalties for pharmacophore points.
    Uses per-particle parameters to prevent global parameter naming conflicts.
    """
    features = find_ligand_pharma_features(ligand_mol)
    conf = ligand_mol.GetConformer()

    expr = (
        "0.5 * k_pharma * weight_scale * step(r_dist - tol) * (r_dist - tol)^2;"
        "r_dist = sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    force = mm.CustomExternalForce(expr)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addPerParticleParameter("tol")
    force.addPerParticleParameter("weight_scale")
    force.addGlobalParameter("k_pharma", k_pharma)
    force.setForceGroup(GROUP_PHARMA)
    force.setName("PharmacophoreRestraintForce")

    for point in pharma_points:
        matching_feats = features.get(point.ptype, [])
        if not matching_feats and point.ptype == "Aro":
            ring_info = ligand_mol.GetRingInfo()
            matching_feats = [list(r) for r in ring_info.AtomRings()]

        if not matching_feats:
            continue

        # Find closest matching feature to the reference pharmacophore point
        best_feat = None
        min_dist = float("inf")

        for feat in matching_feats:
            coords = np.array([conf.GetAtomPosition(a) for a in feat])
            center = np.mean(coords, axis=0)
            dist = np.linalg.norm(center - point.coords)
            if dist < min_dist:
                min_dist = dist
                best_feat = feat

        if best_feat is None:
            continue

        x0_nm = point.x * 0.1
        y0_nm = point.y * 0.1
        z0_nm = point.z * 0.1
        tol_nm = point.tolerance * 0.1
        weight = 1.0 / float(len(best_feat))

        for atom_idx in best_feat:
            sys_idx = atom_idx + ligand_offset_in_system
            force.addParticle(sys_idx, [x0_nm, y0_nm, z0_nm, tol_nm, weight])

    return [force]
