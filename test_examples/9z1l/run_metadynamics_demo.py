"""
Well-Tempered Kinematic Metadynamics (WT-Kin-MetaD) on PDB 9Z1L (KIT V654A +
BLU-654/A1CZZ) -- used here to ASSESS BINDING POSE STRENGTH rather than to
search for a pose: starting exactly at the crystal pose, repulsive Gaussian
hills are deposited there and nearby, actively pushing the ligand away from
its own native basin. How much cumulative bias is needed before the pose
is forced out, and how far the physical score degrades as it's pushed,
is a direct read on how deep/robust that binding basin is.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock.unified_kinematic_pso import UnifiedKinematicPSOEngine
from openmm_dock.metadynamics import KinematicMetadynamicsEngine

DEMO_DIR = Path(__file__).resolve().parent
rec_path = DEMO_DIR / "receptor.pdb"
pocket_center = np.array([16.92, -31.66, 18.54])

print("=" * 90)
print("   OPENMM-DOCK: WELL-TEMPERED KINEMATIC METADYNAMICS -- POSE STRENGTH ASSAY (PDB 9Z1L)")
print("=" * 90)

# 1. Load Crystal Ligand Pose
lig_path = DEMO_DIR / "a1czz_crystal_pose.sdf"
lig_mol = Chem.SDMolSupplier(str(lig_path), removeHs=False)[0]

# 2. Setup Unified Engine and Well-Tempered Metadynamics Engine
unified_engine = UnifiedKinematicPSOEngine(
    receptor_pdb_path=rec_path,
    pocket_center=pocket_center,
    ligand_mol=lig_mol,
    flex_radius=8.0
)

metad_engine = KinematicMetadynamicsEngine(
    unified_engine,
    initial_height_w0=8.0,
    gaussian_sigma=0.50,
    bias_factor_gamma=5.0
)

# 3. Run Clash-Free Metadynamics Exploration Starting From the Crystal Pose
print("\n[*] Launching Clash-Free Well-Tempered Metadynamics from the crystal pose (150 Steps)...")
best_lig, best_rec_coords, best_score, lig_frames, rec_frames, logs = metad_engine.run_metadynamics_exploration(
    n_steps=150,
    deposit_frequency=5,
    step_size=0.03
)

print(f"\n[✓] WT-Kin-MetaD Exploration Completed!")
print(f"    • Total Adaptive Gaussian Hills Deposited: {len(metad_engine.visited_basins)} hills")
print(f"    • Global Best Physical Score              : {best_score:.3f} kcal/mol")

# Print Summary Log Table
print("\n" + "=" * 85)
print(f"{'Step':<6} | {'Physical Score (kcal)':<22} | {'Bias Energy (kcal)':<20} | {'Effective Score':<16} | {'Hills':<6}")
print("-" * 85)
for row in logs[::10]:
    print(f"{row['step']:<6d} | {row['raw_score']:<22.2f} | {row['bias_kcal']:<20.2f} | {row['effective_score']:<16.2f} | {row['num_hills']:<6d}")
print("=" * 85)

# 4. Pose-Strength Metric: how much did the physical score degrade once the
# bias forced the ligand out of the native basin, and how quickly?
native_score = logs[0]["raw_score"]
worst_score_first_third = max(row["raw_score"] for row in logs[: len(logs) // 3])
# A basin is deposited at the native pose itself before step 1, so bias at
# the native pose is high from the very first step by construction --
# "when does bias first exceed X" is not a meaningful escape metric here.
# Instead: how many steps of active repulsion does it take before the RAW
# physical score (unbiased -- what the receptor actually "feels") first
# crosses from favorable into unfavorable (> 0 kcal/mol)? Longer = the
# native basin resists being pushed out = stronger pose.
escape_step = next((row["step"] for row in logs if row["raw_score"] > 0.0), None)
print("\n" + "=" * 85)
print("POSE STRENGTH ASSAY")
print(f"  Native basin physical score              : {native_score:.2f} kcal/mol")
print(f"  Worst score while still escaping (1/3)    : {worst_score_first_third:.2f} kcal/mol")
print(f"  Score penalty for leaving native basin    : {worst_score_first_third - native_score:.2f} kcal/mol")
if escape_step is not None:
    print(f"  Steps of repulsion before score turns unfavorable (>0): {escape_step} (more steps = stronger pose)")
else:
    print(f"  Score never turned unfavorable over {len(logs)} steps -- an exceptionally strong/deep binding basin")
print("=" * 85)

# 5. Save Multi-Track PyMOL Movie
out_lig_movie = DEMO_DIR / "metadynamics_trajectory.sdf"
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

w_best = Chem.SDWriter(str(DEMO_DIR / "metadynamics_best_pose.sdf"))
w_best.write(best_lig)
w_best.close()
unified_engine.rec_kin.write_pdb_frame(best_rec_coords, DEMO_DIR / "metadynamics_best_receptor.pdb")

print(f"\n[✓] Saved Synchronized {len(lig_frames)}-Frame Clash-Free Metadynamics Trajectory:")
print(f"    • Ligand Movie  : {out_lig_movie.name}")
print(f"    • Receptor Movie: {out_rec_movie.name}")
print(f"    • Best Complex  : metadynamics_best_pose.sdf + metadynamics_best_receptor.pdb")

# 6. Generate PyMOL Script
flex_res_str = " or ".join(f"(resi {r.res_num} and resn {r.res_name})" for r in unified_engine.rec_kin.flex_residues)
pml_content = f"""# PyMOL Script for Well-Tempered Kinematic Metadynamics Pose-Strength Assay (PDB 9Z1L)
# Run directly in PyMOL: pymol visualize_metadynamics_pymol.pml

reinitialize
load metadynamics_receptor_trajectory.pdb, kit_metad_receptor
load metadynamics_trajectory.sdf, a1czz_metad_trajectory
load metadynamics_best_receptor.pdb, best_receptor
load metadynamics_best_pose.sdf, best_ligand
load a1czz_crystal_pose.sdf, reference_crystal_pose

hide everything, kit_metad_receptor
show cartoon, kit_metad_receptor
color wheat, kit_metad_receptor
show surface, kit_metad_receptor
set transparency, 0.70, kit_metad_receptor

select active_pocket, kit_metad_receptor and ({flex_res_str})
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose
set stick_radius, 0.24, reference_crystal_pose

hide everything, a1czz_metad_trajectory
show sticks, a1czz_metad_trajectory
color magenta, a1czz_metad_trajectory
set stick_radius, 0.24, a1czz_metad_trajectory

hide everything, best_receptor
hide everything, best_ligand
show sticks, best_ligand
color cyan, best_ligand
set stick_radius, 0.28, best_ligand

zoom reference_crystal_pose, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded WELL-TEMPERED METADYNAMICS POSE-STRENGTH MOVIE!"
print "  • Green: native crystal pose (starting point, hills deposited here)"
print "  • Magenta: trajectory being actively repelled out of the native basin"
print "  Press Play (bottom right) or Spacebar to watch it escape (or resist escaping)!"
print "================================================================="
"""
(DEMO_DIR / "visualize_metadynamics_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_metadynamics_pymol.pml'}")
