#!/usr/bin/env python
"""
Generates pharma.restr for the 9Z1L system from the crystal ligand pose,
using openmm_dock's own find_ligand_pharma_features -- same feature
detector used by DockingEngine(pharma_restr_path=...) internally, so the
restraints are guaranteed self-consistent with what the scoring/restraint
code itself considers a pharmacophore feature.

Includes: all 3 aromatic ring centroids (pyrimidine, pyridine, pyrazole --
BLU-654's core scaffold) and the two aniline N-H donor nitrogens that form
the classic kinase-hinge donor/acceptor pair for this 2,4-diaminopyrimidine
chemotype, matching the density of the existing pharma.restr examples
elsewhere in this repo (2-4 points).
"""
from pathlib import Path
from rdkit import Chem
from openmm_dock.pharmacophore import find_ligand_pharma_features

ROOT = Path(__file__).resolve().parent
TOLERANCE_A = 1.5

mol = Chem.SDMolSupplier(str(ROOT / "a1czz_crystal_pose.sdf"), removeHs=False)[0]
conf = mol.GetConformer()
feats = find_ligand_pharma_features(mol)

lines = []
for ring in feats["Aro"]:
    coords = [conf.GetAtomPosition(i) for i in ring]
    cx = sum(c.x for c in coords) / len(coords)
    cy = sum(c.y for c in coords) / len(coords)
    cz = sum(c.z for c in coords) / len(coords)
    lines.append(f"{cx:6.2f} {cy:6.2f} {cz:6.2f} {TOLERANCE_A:.2f} Aro")

# Donor nitrogens (aniline N-H linkers to the pyrimidine hinge-binding core)
donor_idx = {idx for [idx] in feats["Don"]}
for idx in sorted(donor_idx):
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() == "N" and not atom.GetIsAromatic():
        c = conf.GetAtomPosition(idx)
        lines.append(f"{c.x:6.2f} {c.y:6.2f} {c.z:6.2f} {TOLERANCE_A:.2f} Don")

(ROOT / "pharma.restr").write_text("\n".join(lines) + "\n")
print(f"[✓] Wrote pharma.restr ({len(lines)} points)")
for line in lines:
    print("   ", line)
