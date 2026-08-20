"""
Demonstration of Kinematic Metadynamics (Kin-MetaD) on PDB 6Z6A (Keap1 + Q9E Macrocycle).
Deposits history-dependent repulsive Gaussian hills to fill local energy traps and
force smooth, continuous physical dynamics across the kinematic manifold.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock.unified_kinematic_pso import UnifiedKinematicPSOEngine
from openmm_dock.metadynamics import KinematicMetadynamicsEngine

DEMO_DIR = Path(__file__).resolve().parent
rec_path = DEMO_DIR / "receptor.pdb"
pocket_center = np.array([-21.46, 22.44, -24.18])

print("=" * 90)
print("   OPENMM-DOCK: KINEMATIC METADYNAMICS (Kin-MetaD) ON PDB 6Z6A (Keap1 + Q9E)")
print("=" * 90)

# 1. Load Macrocycle Ligand
lig_path = DEMO_DIR / "q9e_crystal_pose.sdf"
lig_mol = Chem.SDMolSupplier(str(lig_path), removeHs=False)[0]

# 2. Setup Unified Engine and Kinematic Metadynamics Engine
unified_engine = UnifiedKinematicPSOEngine(
    receptor_pdb_path=rec_path,
    pocket_center=pocket_center,
    ligand_mol=lig_mol,
    flex_radius=9.0
)

metad_engine = KinematicMetadynamicsEngine(
    unified_engine,
    gaussian_height_w=25.0,  # +25 kcal/mol per Gaussian hill
    gaussian_sigma=0.50      # Smooth Gaussian width
)

# 3. Run Smooth Kinematic Metadynamics Exploration (100 Steps)
print("\n[*] Launching Smooth Kinematic Metadynamics (Kin-MetaD, 100 Frames)...")
best_lig, best_rec_coords, best_score, lig_frames, rec_frames, logs = metad_engine.run_metadynamics_exploration(
    n_steps=100,
    deposit_frequency=5,
    step_size=0.04
)

print(f"\n[✓] Kin-MetaD Exploration Completed!")
print(f"    • Total Gaussian Hills Deposited: {len(metad_engine.visited_basins)} hills")
print(f"    • Global Best Physical Score     : {best_score:.3f} kcal/mol")

# 4. Save Multi-Track PyMOL Movie
out_lig_movie = DEMO_DIR / "metadynamics_macrocycle_trajectory.sdf"
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

out_rec_movie = DEMO_DIR / "metadynamics_receptor_trajectory.pdb"
out_rec_movie.write_text("\n".join(pdb_models) + "\nEND\n")

# Save Best Poses
w_best = Chem.SDWriter(str(DEMO_DIR / "metadynamics_best_pose.sdf"))
w_best.write(best_lig)
w_best.close()
unified_engine.rec_kin.write_pdb_frame(best_rec_coords, DEMO_DIR / "metadynamics_best_receptor.pdb")

print(f"\n[✓] Saved Synchronized 100-Frame Metadynamics Trajectory:")
print(f"    • Macrocycle Movie: {out_lig_movie.name}")
print(f"    • Receptor Movie  : {out_rec_movie.name}")

# 5. Generate PyMOL Script
flex_res_str = " or ".join(f"(resi {r.res_num} and resn {r.res_name})" for r in unified_engine.rec_kin.flex_residues)
pml_content = f"""# PyMOL Script for Kinematic Metadynamics (Kin-MetaD) Movie
# Run directly in PyMOL: pymol visualize_metadynamics_pymol.pml

reinitialize
load metadynamics_receptor_trajectory.pdb, keap1_metad_receptor
load metadynamics_macrocycle_trajectory.sdf, q9e_metad_macrocycle
load metadynamics_best_receptor.pdb, best_receptor
load metadynamics_best_pose.sdf, best_macrocycle

# Style Receptor Backbone
hide everything, keap1_metad_receptor
show cartoon, keap1_metad_receptor
color wheat, keap1_metad_receptor
show surface, keap1_metad_receptor
set transparency, 0.70, keap1_metad_receptor

# Style Active Pocket Residues
select active_pocket, keap1_metad_receptor and ({flex_res_str})
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

select arginines, keap1_metad_receptor and (resn ARG and resi 415+483+380+336+601)
color marine, arginines
select tyrosines, keap1_metad_receptor and (resn TYR and resi 334+572+525)
color tv_green, tyrosines
select serines, keap1_metad_receptor and (resn SER and resi 602+555+363+338)
color orange, serines

# Style Macrocycle Trajectory
hide everything, q9e_metad_macrocycle
show sticks, q9e_metad_macrocycle
color magenta, q9e_metad_macrocycle
set stick_radius, 0.24, q9e_metad_macrocycle

# Style Best Complex (Shown on pause)
hide everything, best_receptor
hide everything, best_macrocycle
show sticks, best_macrocycle
color cyan, best_macrocycle
set stick_radius, 0.28, best_macrocycle

# Dynamic Distance Contacts to Arginines
distance sb_arg415, (keap1_metad_receptor and resi 415 and name NH1), (q9e_metad_macrocycle and name O28), 3.5
distance sb_arg483, (keap1_metad_receptor and resi 483 and name NH2), (q9e_metad_macrocycle and name O19), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
set dash_width, 3.0

zoom q9e_metad_macrocycle, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 100-Frame KINEMATIC METADYNAMICS (Kin-MetaD) MOVIE!"
print "  • Repulsive Gaussian Hills fill visited decoy traps (+25 kcal/mol)"
print "  • Watch the 16-membered macrocycle continuously glide & escape traps!"
print "  • Press Play (bottom right) or Spacebar to watch active 3D dynamics!"
print "================================================================="
"""
(DEMO_DIR / "visualize_metadynamics_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_metadynamics_pymol.pml'}")
