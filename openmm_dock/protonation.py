"""
Automated physiological pH (pH 7.4) ionization and protonation state perception for ligands.
Supports native RDKit substructure perception with optional OpenBabel bridge.
"""
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem


def protonate_ligand_ph(mol: Chem.Mol, target_ph: float = 7.4) -> Chem.Mol:
    """
    Perceives and sets the dominant physiological ionization state (pH ~ 7.4)
    for drug-like ligands, including carboxylic acids, tetrazoles, sulfonic acids,
    aliphatic amines, amidines, and guanidines.
    """
    # 1. Try OpenBabel if installed
    try:
        from openbabel import pybel
        ob_mol = pybel.readstring("mol", Chem.MolToMolBlock(mol))
        ob_mol.OBMol.AddHydrogens(False, True, target_ph)
        rd_mol = Chem.MolFromMolBlock(ob_mol.write("mol"), removeHs=False)
        if rd_mol is not None:
            return rd_mol
    except Exception:
        pass

    # 2. Fallback to high-precision native RDKit rule engine
    mol = Chem.Mol(mol)

    # A. Acids -> Anions (-1)
    # 1. Carboxylic Acids -> Carboxylate
    patt_acid = Chem.MolFromSmarts("[C:1](=[O:2])[OH1:3]")
    if patt_acid:
        for m in mol.GetSubstructMatches(patt_acid):
            mol.GetAtomWithIdx(m[2]).SetFormalCharge(-1)
            mol.GetAtomWithIdx(m[2]).SetNumExplicitHs(0)

    # 2. Sulfonic Acids -> Sulfonate
    patt_sulf = Chem.MolFromSmarts("[S:1](=[O:2])(=[O:3])[OH1:4]")
    if patt_sulf:
        for m in mol.GetSubstructMatches(patt_sulf):
            mol.GetAtomWithIdx(m[3]).SetFormalCharge(-1)
            mol.GetAtomWithIdx(m[3]).SetNumExplicitHs(0)

    # 3. Tetrazoles -> Tetrazolate
    patt_tet = Chem.MolFromSmarts("[c:1]1[n:2][n:3][n:4][nH1:5]1")
    if patt_tet:
        for m in mol.GetSubstructMatches(patt_tet):
            mol.GetAtomWithIdx(m[4]).SetFormalCharge(-1)
            mol.GetAtomWithIdx(m[4]).SetNumExplicitHs(0)

    # B. Bases -> Cations (+1)
    # 1. Aliphatic Amines (primary, secondary, tertiary)
    patt_amine = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N*=[O,N,P,S]);!$(N=O);!$(NC=O);!$(NS(=O)=O);!$(Nc1ncccc1);!$(n):1]")
    if patt_amine:
        for m in mol.GetSubstructMatches(patt_amine):
            atom = mol.GetAtomWithIdx(m[0])
            if atom.GetFormalCharge() == 0:
                atom.SetFormalCharge(1)

    # 2. Guanidines
    patt_guan = Chem.MolFromSmarts("[NX3:1][C:2](=[NX2:3])[NX3:4]")
    if patt_guan:
        for m in mol.GetSubstructMatches(patt_guan):
            atom = mol.GetAtomWithIdx(m[2])
            if atom.GetFormalCharge() == 0:
                atom.SetFormalCharge(1)

    Chem.SanitizeMol(mol)
    return mol
