"""
Global Blind Docking from Scratch on PDB 9Z1L (KIT V654A kinase + BLU-654/A1CZZ).
Starts from a completely unaligned, scrambled pose in bulk solvent and lets the
multi-conformer swarm-metadynamics + PSO engine (global_blind_docking.py) find
its own way back into the ATP-binding cleft, with no prior knowledge of the pose.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
from rdkit import Chem
from rdkit.Geometry import Point3D

from openmm_dock.unified_kinematic_pso import UnifiedKinematicPSOEngine
from openmm_dock.global_blind_docking import GlobalBlindDockingEngine, BlindDockingParams

DEMO_DIR = Path(__file__).resolve().parent
rec_path = DEMO_DIR / "receptor.pdb"
pocket_center = np.array([16.92, -31.66, 18.54])

print("=" * 95)
print("   OPENMM-DOCK: GLOBAL BLIND DOCKING FROM SCRATCH (PDB 9Z1L: KIT V654A + BLU-654)")
print("=" * 95)

# 1. Load Reference Crystal Pose and Create a Completely Scrambled Unaligned Starting Mol
xtal_path = DEMO_DIR / "a1czz_crystal_pose.sdf"
xtal_mol = Chem.SDMolSupplier(str(xtal_path), removeHs=False)[0]

unaligned_mol = Chem.Mol(xtal_mol)
conf_u = unaligned_mol.GetConformer()
coords_xtal = np.array([conf_u.GetAtomPosition(i) for i in range(unaligned_mol.GetNumAtoms())])

# Invert orientation and translate ~18 A into bulk solvent, away from the pocket
q_rot = ScipyRotation.from_euler("zyx", [160, 55, 95], degrees=True).as_matrix()
center = np.mean(coords_xtal, axis=0)
coords_scrambled = (coords_xtal - center).dot(q_rot.T) + center + np.array([-11.0, 9.0, -9.0])

for i in range(unaligned_mol.GetNumAtoms()):
    conf_u.SetAtomPosition(i, Point3D(float(coords_scrambled[i][0]), float(coords_scrambled[i][1]), float(coords_scrambled[i][2])))

init_rmsd = float(np.sqrt(np.mean(np.sum((coords_scrambled - coords_xtal) ** 2, axis=1))))
print(f"[*] Generated Unaligned Starting Ligand in Bulk Solvent:")
print(f"    • Initial Distance from Pocket Center: {np.linalg.norm(np.mean(coords_scrambled, axis=0) - pocket_center):.2f} Å")
print(f"    • Initial RMSD to Native Crystal Pose : {init_rmsd:.2f} Å (Completely Blind!)")

out_unaligned = DEMO_DIR / "a1czz_unaligned_start.sdf"
w_u = Chem.SDWriter(str(out_unaligned))
w_u.write(unaligned_mol)
w_u.close()

# 2. Setup Unified Engine and Blind Docking Engine
unified_engine = UnifiedKinematicPSOEngine(
    receptor_pdb_path=rec_path,
    pocket_center=pocket_center,
    ligand_mol=xtal_mol,
    flex_radius=8.0
)

params = BlindDockingParams(
    n_particles=30,
    n_iterations=20,
    search_box_size=22.0,
    w_start=0.85,
    w_end=0.40,
    c1_cognitive=1.4,
    c2_social=2.1,
    k_contact_beacon=0.50,
    k_depth_beacon=3.0,
    gaussian_w0=8.0,
    gaussian_sigma=0.50,
    bias_gamma=6.0
)

blind_engine = GlobalBlindDockingEngine(unified_engine, params)

# 3. Execute Global Blind Docking
print(f"\n[*] Launching Global Blind Swarm Metadynamics (30 Walkers × 20 Iterations = 600 Frames)...")
best_lig, best_rec_coords, best_score, lig_frames, rec_frames, blind_log = blind_engine.run_blind_docking(
    unaligned_start_mol=unaligned_mol,
    reference_xtal_mol=xtal_mol
)

final_rmsd = float(best_lig.GetProp("FINAL_RMSD_TO_XTAL_A"))
print(f"\n[✓] Global Blind Docking Completed!")
print(f"    • Starting RMSD : {init_rmsd:.2f} Å (Bulk Solvent)")
print(f"    • Converged RMSD: {final_rmsd:.2f} Å")
print(f"    • Final Physical Score: {best_score:.3f} kcal/mol")

# 4. Plot Convergence Curve
conv_png = DEMO_DIR / "blind_docking_convergence.png"
print("\n[*] Plotting Blind Docking Convergence Trajectory...")
blind_engine.plot_blind_convergence(blind_log, conv_png)

# 5. Save Multi-Track Movie
out_lig_movie = DEMO_DIR / "blind_docking_swarm_trajectory.sdf"
w_traj = Chem.SDWriter(str(out_lig_movie))
for f in lig_frames:
    w_traj.write(f)
w_traj.close()

pdb_models = []
for f_idx, r_coords in enumerate(rec_frames):
    lines_m = [f"MODEL     {f_idx + 1:4d}"]
    for a_idx, l in enumerate(unified_engine.rec_kin.atom_lines):
        x, y, z = r_coords[a_idx]
        lines_m.append(f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}")
    lines_m.append("ENDMDL")
    pdb_models.append("\n".join(lines_m))

out_rec_movie = DEMO_DIR / "blind_docking_receptor_trajectory.pdb"
out_rec_movie.write_text("\n".join(pdb_models) + "\nEND\n")

# Save Converged Complex
w_best = Chem.SDWriter(str(DEMO_DIR / "blind_docking_best_pose.sdf"))
w_best.write(best_lig)
w_best.close()
unified_engine.rec_kin.write_pdb_frame(best_rec_coords, DEMO_DIR / "blind_docking_best_receptor.pdb")

print(f"\n[✓] Saved Synchronized 600-Frame Blind Docking Trajectory:")
print(f"    • Swarm Movie   : {out_lig_movie.name}")
print(f"    • Receptor Movie: {out_rec_movie.name}")
print(f"    • Plot File     : blind_docking_convergence.png")
print(f"    • Best Complex  : blind_docking_best_pose.sdf + blind_docking_best_receptor.pdb")

# 6. Generate PyMOL Script
flex_res_str = " or ".join(f"(resi {r.res_num} and resn {r.res_name})" for r in unified_engine.rec_kin.flex_residues)
pml_content = f"""# PyMOL Script for Global Blind Docking Demonstration (PDB 9Z1L)
# Run directly in PyMOL: pymol visualize_blind_docking_pymol.pml

reinitialize
load blind_docking_receptor_trajectory.pdb, kit_blind_receptor
load blind_docking_swarm_trajectory.sdf, a1czz_blind_swarm
load a1czz_unaligned_start.sdf, start_unaligned_ligand
load a1czz_crystal_pose.sdf, reference_crystal_pose
load blind_docking_best_pose.sdf, best_docked_ligand

hide everything, kit_blind_receptor
show cartoon, kit_blind_receptor
color wheat, kit_blind_receptor
show surface, kit_blind_receptor
set transparency, 0.70, kit_blind_receptor

select active_pocket, kit_blind_receptor and ({flex_res_str})
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

hide everything, start_unaligned_ligand
show sticks, start_unaligned_ligand
color red, start_unaligned_ligand
set stick_radius, 0.22, start_unaligned_ligand

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose
set stick_radius, 0.22, reference_crystal_pose

hide everything, a1czz_blind_swarm
show sticks, a1czz_blind_swarm
color magenta, a1czz_blind_swarm
set stick_radius, 0.24, a1czz_blind_swarm

hide everything, best_docked_ligand
show sticks, best_docked_ligand
color cyan, best_docked_ligand
set stick_radius, 0.28, best_docked_ligand

zoom kit_blind_receptor, 8.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 600-Frame GLOBAL BLIND DOCKING MOVIE! (KIT V654A + BLU-654)"
print "  • Red Sticks    : Initial Unaligned Starting Pose in Solvent"
print "  • Green Sticks  : Reference Co-Crystal Pose (0.0 Å RMSD)"
print "  • Magenta Sticks: 30-Walker Swarm Navigating from Solvent to Cleft"
print "  • Cyan Sticks   : Final Converged Complex"
print "  Press Play (bottom right) or Spacebar to watch blind docking!"
print "================================================================="
"""
(DEMO_DIR / "visualize_blind_docking_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_blind_docking_pymol.pml'}")
