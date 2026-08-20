# scripts/make_fixtures.py
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy

# Anchor all paths to repo root for reproducibility
ROOT = Path(__file__).parent.parent
(ROOT / "tests/fixtures").mkdir(parents=True, exist_ok=True)
(ROOT / "scripts").mkdir(exist_ok=True)

# --- mini_poses.sdf: 3 compounds x 3 poses, SCORE.* fields + real 3D mol blocks ---
compounds = [
    ("CPD001", "c1ccc2ncccc2c1"),
    ("CPD002", "c1ccncc1"),
    ("CPD003", "c1ccc(N)cc1"),
    ("CPD005", "c1cccc2ccccc12"),
    ("CPD006", "c1ccc(Cl)cc1"),
]

lines = []
for name, smi in compounds:
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    for rank in range(1, 4):
        score = -10.0 + rank * 2.0  # rank 1 = best (lowest) score
        # Create a copy and perturb coordinates for distinct 3D poses
        mol_copy = deepcopy(mol)
        conf = mol_copy.GetConformer()
        rng = np.random.default_rng(rank * 100)
        for i in range(mol_copy.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            delta = rng.uniform(-0.5, 0.5, 3)
            conf.SetAtomPosition(i, (pos.x + delta[0], pos.y + delta[1], pos.z + delta[2]))
        mol_block = Chem.MolToMolBlock(mol_copy)
        lines.append(mol_block)
        lines.append(f">  <Name>\n{name}\n\n")
        lines.append(f">  <SCORE>\n{score:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.VDW>\n{-3.0 + rank:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.POLAR>\n{-2.0 + rank * 0.5:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.REPUL>\n{0.5 + rank * 0.1:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.CONST>\n5.400\n\n")
        lines.append(f">  <SCORE.INTER.ROT>\n{1.0 + rank * 0.2:.3f}\n\n")
        lines.append(f">  <SCORE.RESTR>\n{0.0 if rank == 1 else 2.0:.3f}\n\n")
        lines.append(f">  <SCORE.RESTR.CAVITY>\n0.000\n\n")
        lines.append(f">  <SCORE.SYSTEM.VDW>\n{-1.0 + rank * 0.3:.3f}\n\n")
        lines.append(f">  <SCORE.SYSTEM.POLAR>\n{-0.5 + rank * 0.1:.3f}\n\n")
        lines.append("$$$$\n")

(ROOT / "tests/fixtures/mini_poses.sdf").write_text("".join(lines))
print("Wrote mini_poses.sdf")

# --- mini_crystal.pdb: ligand at origin, 3 waters within 5Å, 1 water far away ---
pdb = """\
HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A   1       1.400   0.000   0.000  1.00  0.00           C
HETATM    3  N1  LIG A   1       0.700   1.212   0.000  1.00  0.00           N
HETATM    4  O   HOH A 101       2.000   0.500   0.000  1.00  0.00           O
HETATM    5  O   HOH A 102      -1.500   0.500   0.000  1.00  0.00           O
HETATM    6  O   HOH A 103       0.500   3.000   0.000  0.50  0.00           O
HETATM    7  O   HOH A 104       5.500   5.500   5.500  1.00  0.00           O
END
"""
(ROOT / "tests/fixtures/mini_crystal.pdb").write_text(pdb)
print("Wrote mini_crystal.pdb")

# --- mini_ligand.sdf: pyridine in Kekulé form, centred near origin ---
mol = Chem.MolFromSmiles("c1ccncc1")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
mol = Chem.RemoveHs(mol)
Chem.Kekulize(mol, clearAromaticFlags=True)
w = Chem.SDWriter(str(ROOT / "tests/fixtures/mini_ligand.sdf"))
w.SetKekulize(True)
w.write(mol)
w.close()
print("Wrote mini_ligand.sdf")
