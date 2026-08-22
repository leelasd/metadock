#!/usr/bin/env python
"""
Generates pharma.restr for the 9Z1L system from the crystal ligand pose,
using openmm_dock's own find_ligand_pharma_features -- same feature
detector used by DockingEngine(pharma_restr_path=...) internally, so the
restraints are guaranteed self-consistent with what the scoring/restraint
code itself considers a pharmacophore feature.

Includes all 3 aromatic ring centroids (pyrimidine, pyridine, pyrazole --
BLU-654's core scaffold), all 8 heavy-atom acceptor points (every ring/
exocyclic N and O), the 2 aniline N-H donor nitrogens, AND all 6 hydrophobic
(pure-carbon, no heteroatom neighbor) points -- full coverage of every
feature find_ligand_pharma_features detects, with a MUCH tighter flat-bottom
tolerance (0.4 A vs. the earlier 1.5 A). Each restraint is a flat-bottom
harmonic (see create_pharmacophore_restraint_forces in pharmacophore.py):
zero force inside the tolerance radius, so a loose tolerance lets a pose
satisfy every restraint while still sitting several A from native. Tightening
it is the direct lever to force close convergence.

The Hyd points matter specifically because they're the aliphatic atoms NOT
already anchored by any Aro/Acc/Don restraint -- an earlier 13-point version
(Aro+Acc+Don only) left those parts of the ligand's shape/orientation
under-constrained, which plausibly explains why even the best fine-refined
poses still carried nonzero restraint-score residual.
"""
from pathlib import Path
from rdkit import Chem
from openmm_dock.pharmacophore import find_ligand_pharma_features

ROOT = Path(__file__).resolve().parent
TOLERANCE_A = 0.4

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

# All heavy-atom acceptors (ring + exocyclic N/O) -- comprehensive coverage
for [idx] in feats["Acc"]:
    c = conf.GetAtomPosition(idx)
    lines.append(f"{c.x:6.2f} {c.y:6.2f} {c.z:6.2f} {TOLERANCE_A:.2f} Acc")

# Donor nitrogens (aniline N-H linkers to the pyrimidine hinge-binding core)
for [idx] in feats["Don"]:
    c = conf.GetAtomPosition(idx)
    lines.append(f"{c.x:6.2f} {c.y:6.2f} {c.z:6.2f} {TOLERANCE_A:.2f} Don")

# Hydrophobic atoms (pure-carbon substituents with no heteroatom neighbor) --
# the only feature type the earlier version of this script omitted.
for [idx] in feats["Hyd"]:
    c = conf.GetAtomPosition(idx)
    lines.append(f"{c.x:6.2f} {c.y:6.2f} {c.z:6.2f} {TOLERANCE_A:.2f} Hyd")

(ROOT / "pharma.restr").write_text("\n".join(lines) + "\n")
print(f"[✓] Wrote pharma.restr ({len(lines)} points, tolerance={TOLERANCE_A} A)")
for line in lines:
    print("   ", line)
