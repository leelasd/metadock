# scoring_optimizer/crystal_processing.py
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def find_binding_waters(
    receptor_pdb: Path,
    ligand_sdf: Path,
    cutoff_angstrom: float = 5.0,
    min_occupancy: float = 0.5,
) -> list[np.ndarray]:
    """Return xyz of crystallographic waters within cutoff_angstrom of the ligand.

    Note: distances are computed to all atoms (including explicit H) if the ligand SDF
    contains them. Pass removeHs=False (default) to preserve explicit H.
    """
    ligand_mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False)
    if ligand_mol is None:
        raise ValueError(f"Could not parse ligand SDF: {ligand_sdf}")
    conf = ligand_mol.GetConformer()
    lig_coords = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(ligand_mol.GetNumAtoms())
    ])

    waters = []
    with open(receptor_pdb, encoding="utf-8") as f:
        for line in f:
            if not (line.startswith("HETATM") or line.startswith("ATOM")):
                continue
            if line[17:20].strip() != "HOH" or line[12:16].strip() != "O":
                continue
            try:
                occupancy = float(line[54:60].strip())
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if occupancy < min_occupancy:
                continue
            if np.linalg.norm(lig_coords - xyz, axis=1).min() <= cutoff_angstrom:
                waters.append(xyz)
    return waters


def pharmacophore_features(mol: Chem.Mol) -> dict:
    """
    Extract pharmacophore features from a 3D ligand molecule.

    Acceptors include both non-aromatic and aromatic N/O (e.g. pyridine N).
    """
    try:
        conf = mol.GetConformer()
    except ValueError as e:
        raise ValueError("mol must have a 3D conformer — pass removeHs=False when loading from SDF") from e

    def pos(idx) -> np.ndarray:
        p = conf.GetAtomPosition(idx)
        return np.array([p.x, p.y, p.z])

    aro_centers = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            aro_centers.append(np.mean([pos(a) for a in ring], axis=0))

    acceptors, donors = [], []
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum in (7, 8):
            acceptors.append(pos(atom.GetIdx()))
        if anum in (7, 8) and atom.GetTotalNumHs() > 0:
            donors.append(pos(atom.GetIdx()))

    return {"aro_centers": aro_centers, "acceptors": acceptors, "donors": donors}


def write_pharma_restr(
    features: dict,
    output_path: Path,
    n_aro: int = 2,
    n_acc: int = 2,
    tolerance: float = 1.0,
) -> None:
    """Write a pharma.restr file from pharmacophore features."""
    lines = []
    actual_aro = min(n_aro, len(features["aro_centers"]))
    actual_acc = min(n_acc, len(features["acceptors"]))
    if actual_aro < n_aro:
        warnings.warn(f"Only {actual_aro} aromatic centers available, requested {n_aro}")
    if actual_acc < n_acc:
        warnings.warn(f"Only {actual_acc} acceptors available, requested {n_acc}")
    for c in features["aro_centers"][:n_aro]:
        lines.append(f"{c[0]:7.2f} {c[1]:7.2f} {c[2]:7.2f} {tolerance:.2f} Aro")
    for c in features["acceptors"][:n_acc]:
        lines.append(f"{c[0]:7.2f} {c[1]:7.2f} {c[2]:7.2f} {tolerance:.2f} Acc")
    output_path.write_text("\n".join(lines) + "\n")


def crystal_ligand_coords(ligand_sdf: Path) -> np.ndarray:
    """Return heavy-atom 3D coordinates of the co-crystal ligand as (n, 3) array."""
    mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=True)
    if mol is None:
        raise ValueError(f"Could not parse ligand SDF: {ligand_sdf}")
    conf = mol.GetConformer()
    return np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])
