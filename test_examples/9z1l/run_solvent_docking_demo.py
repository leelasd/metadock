"""
Explicit Flexible Active-Site Water Docking on PDB 9Z1L (KIT V654A + BLU-654/A1CZZ).

Compares docking WITH vs WITHOUT the 11 crystallographic waters found within
5 A of the ligand (active_site_waters.pdb, extracted by extract_from_pdb.py) --
these waters are treated as explicit, flexible OpenMM particles during the
search, not just static receptor decoration.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock.cavity import CavityDefinition
from openmm_dock.engine import DockingEngine
from openmm_dock.core import SDFParser

DEMO_DIR = Path(__file__).resolve().parent

print("=" * 95)
print("   OPENMM-DOCK: EXPLICIT FLEXIBLE ACTIVE-SITE WATER DOCKING (PDB 9Z1L: KIT V654A + BLU-654)")
print("=" * 95)

cavity = CavityDefinition.from_prm_file(DEMO_DIR / "cavity.prm")
xtal_mol = SDFParser.load_molecules(DEMO_DIR / "a1czz_crystal_pose.sdf")[0]
xtal_coords = xtal_mol.GetConformer().GetPositions()


def rmsd_to_xtal(mol: Chem.Mol) -> float:
    coords = mol.GetConformer().GetPositions()
    return float(np.sqrt(np.mean(np.sum((coords - xtal_coords) ** 2, axis=1))))


# 1. WITHOUT explicit waters (dry pocket)
print("\n[1] Monte Carlo Basin-Hopping WITHOUT explicit waters (dry pocket)...")
engine_dry = DockingEngine(receptor_path=DEMO_DIR / "receptor.mol2", cavity=cavity)
res_dry = engine_dry.dock_monte_carlo(xtal_mol, n_steps=100, temperature_k=300.0)
print(f"  Best Pose (dry): Score = {res_dry.score:.3f} | RMSD to Xtal = {rmsd_to_xtal(res_dry.mol):.2f} Å")

# 2. WITH explicit flexible active-site waters
print("\n[2] Monte Carlo Basin-Hopping WITH 11 explicit flexible waters...")
engine_wet = DockingEngine(
    receptor_path=DEMO_DIR / "receptor.mol2", cavity=cavity, waters_pdb_path=DEMO_DIR / "active_site_waters.pdb"
)
res_wet = engine_wet.dock_monte_carlo(xtal_mol, n_steps=100, temperature_k=300.0)
print(f"  Best Pose (wet): Score = {res_wet.score:.3f} | RMSD to Xtal = {rmsd_to_xtal(res_wet.mol):.2f} Å")

# 3. Local minimization with waters, for a clean converged reference
print("\n[3] Local L-BFGS minimization of the crystal pose WITH explicit waters...")
min_wet = engine_wet.minimize(xtal_mol)
print(f"  Minimized (wet): Score = {min_wet.score:.3f} | RMSD to Xtal = {rmsd_to_xtal(min_wet.mol):.2f} Å")

print("\n" + "=" * 80)
print("COMPARISON: Dry Pocket vs. Explicit Flexible Waters")
print(f"  Dry MC best   : Score {res_dry.score:8.2f} | RMSD {rmsd_to_xtal(res_dry.mol):.2f} Å")
print(f"  Wet MC best   : Score {res_wet.score:8.2f} | RMSD {rmsd_to_xtal(res_wet.mol):.2f} Å")
print(f"  Wet minimized : Score {min_wet.score:8.2f} | RMSD {rmsd_to_xtal(min_wet.mol):.2f} Å")
print("=" * 80)

Chem.SDWriter(str(DEMO_DIR / "solvent_dock_dry_out.sdf")).write(res_dry.mol)
Chem.SDWriter(str(DEMO_DIR / "solvent_dock_wet_out.sdf")).write(res_wet.mol)
Chem.SDWriter(str(DEMO_DIR / "solvent_min_wet_out.sdf")).write(min_wet.mol)
print("\n[✓] Saved solvent_dock_dry_out.sdf, solvent_dock_wet_out.sdf, solvent_min_wet_out.sdf")

pml_content = """# PyMOL Script for Explicit Water Docking Comparison (PDB 9Z1L)
reinitialize
load receptor.mol2, kit_receptor
load active_site_waters.pdb, active_waters
load a1czz_crystal_pose.sdf, reference_crystal_pose
load solvent_dock_dry_out.sdf, dry_pose
load solvent_dock_wet_out.sdf, wet_pose

hide everything, kit_receptor
show cartoon, kit_receptor
color wheat, kit_receptor

hide everything, active_waters
show nb_spheres, active_waters
color skyblue, active_waters

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose

hide everything, dry_pose
show sticks, dry_pose
color orange, dry_pose
set stick_radius, 0.18, dry_pose

hide everything, wet_pose
show sticks, wet_pose
color cyan, wet_pose
set stick_radius, 0.22, wet_pose

zoom reference_crystal_pose, 8.0

print "================================================================="
print "  Blue spheres: 11 crystallographic active-site waters"
print "  Green: crystal reference | Orange: dry-pocket dock | Cyan: wet-pocket dock"
print "================================================================="
"""
(DEMO_DIR / "visualize_solvent_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_solvent_pymol.pml'}")
