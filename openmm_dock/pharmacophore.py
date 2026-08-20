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
    Creates OpenMM restraint forces applying flat-bottom penalties for pharmacophore points.
    - Uses CustomCentroidBondForce for multi-atom features (Aromatic rings) to translate the
      ring center-of-mass without creating radial inward squeezing forces on individual ring atoms.
    - Uses CustomExternalForce for single-atom features (Acceptors, Donors, Hydrophobic).
    """
    features = find_ligand_pharma_features(ligand_mol)
    conf = ligand_mol.GetConformer()
    forces: List[mm.Force] = []

    # 1. Centroid Force for Multi-Atom Ring Features (Aro)
    expr_centroid = (
        "0.5 * k_pharma * step(dist - tol) * (dist - tol)^2;"
        "dist = sqrt((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
    )
    centroid_force = mm.CustomCentroidBondForce(1, expr_centroid)
    centroid_force.addPerBondParameter("x0")
    centroid_force.addPerBondParameter("y0")
    centroid_force.addPerBondParameter("z0")
    centroid_force.addPerBondParameter("tol")
    centroid_force.addGlobalParameter("k_pharma", k_pharma)
    centroid_force.setForceGroup(GROUP_PHARMA)
    centroid_force.setName("PharmacophoreCentroidForce")

    # 2. External Force for Single-Atom Features (Acc, Don, Hyd)
    expr_ext = (
        "0.5 * k_pharma * step(r_dist - tol) * (r_dist - tol)^2;"
        "r_dist = sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    ext_force = mm.CustomExternalForce(expr_ext)
    ext_force.addPerParticleParameter("x0")
    ext_force.addPerParticleParameter("y0")
    ext_force.addPerParticleParameter("z0")
    ext_force.addPerParticleParameter("tol")
    ext_force.addGlobalParameter("k_pharma", k_pharma)
    ext_force.setForceGroup(GROUP_PHARMA)
    ext_force.setName("PharmacophoreExternalForce")

    has_centroid = False
    has_ext = False

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

        if len(best_feat) > 1:
            # Multi-atom ring centroid restraint
            sys_indices = [a + ligand_offset_in_system for a in best_feat]
            g_idx = centroid_force.addGroup(sys_indices)
            centroid_force.addBond([g_idx], [x0_nm, y0_nm, z0_nm, tol_nm])
            has_centroid = True
        else:
            # Single-atom restraint
            sys_idx = best_feat[0] + ligand_offset_in_system
            ext_force.addParticle(sys_idx, [x0_nm, y0_nm, z0_nm, tol_nm])
            has_ext = True

    if has_centroid:
        forces.append(centroid_force)
    if has_ext:
        forces.append(ext_force)

    return forces


def align_ligand_to_pharmacophore(mol: Chem.Mol, pharma_points: List[PharmaPoint]) -> Chem.Mol:
    """
    Superimposes the ligand's matching pharmacophore feature centers onto the target pharmacophore points
    using a least-squares rigid-body Kabsch transformation.
    """
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()
    features = find_ligand_pharma_features(mol_copy)

    # 1. Translate molecule center to the center of the pharmacophore points first
    target_center = np.mean([p.coords for p in pharma_points], axis=0)
    current_center = np.mean([conf.GetAtomPosition(i) for i in range(mol_copy.GetNumAtoms())], axis=0)
    trans = target_center - current_center
    for i in range(mol_copy.GetNumAtoms()):
        p = np.array(conf.GetAtomPosition(i)) + trans
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))

    # 2. Match distinct features for each pharmacophore point
    used_features = set()
    pt_coords = []
    feat_coords = []
    for p in pharma_points:
        m_feats = features.get(p.ptype, [])
        if not m_feats and p.ptype == "Aro":
            ring_info = mol_copy.GetRingInfo()
            m_feats = [list(r) for r in ring_info.AtomRings()]
        if not m_feats:
            continue

        best_f_key = None
        best_c = None
        min_d = float("inf")
        for f in m_feats:
            f_key = tuple(sorted(f))
            if f_key in used_features and len(m_feats) > len(used_features):
                continue
            fc = np.mean([conf.GetAtomPosition(a) for a in f], axis=0)
            d = np.linalg.norm(fc - p.coords)
            if d < min_d:
                min_d = d
                best_c = fc
                best_f_key = f_key
        if best_c is not None:
            used_features.add(best_f_key)
            pt_coords.append(p.coords)
            feat_coords.append(best_c)

    if len(pt_coords) >= 3:
        P = np.array(feat_coords)
        Q = np.array(pt_coords)
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_c = P - centroid_P
        Q_c = Q - centroid_Q
        H = np.dot(P_c.T, Q_c)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_Q - np.dot(centroid_P, R.T)

        for i in range(mol_copy.GetNumAtoms()):
            p_orig = np.array(conf.GetAtomPosition(i))
            p_new = np.dot(p_orig, R.T) + t
            conf.SetAtomPosition(i, (float(p_new[0]), float(p_new[1]), float(p_new[2])))

    return mol_copy
