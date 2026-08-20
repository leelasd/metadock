"""
Demonstration of Grand Unified Kinematic Particle Swarm Optimization (Kin-PSO).
Unifies:
1. Macrocycle Ring Breathing (Inverse Kinematics)
2. Ligand Exocyclic Arm Articulation (Forward Kinematics)
3. Receptor Active-Site Side-Chain Flexing (chi1 - chi4 Rotamers)
4. Particle Swarm Optimization (Coupled Swarm Intelligence)
"""
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock.unified_kinematic_pso import UnifiedKinematicPSOEngine

DEMO_DIR = Path(__file__).resolve().parent
rec_path = DEMO_DIR / "receptor.pdb"
pocket_center = np.array([-21.46, 22.44, -24.18])

print("=" * 90)
print("   OPENMM-DOCK: GRAND UNIFIED KINEMATIC PARTICLE SWARM OPTIMIZATION (6Z6A)")
print("=" * 90)

# 1. Load Macrocycle
lig_path = DEMO_DIR / "q9e_crystal_pose.sdf"
lig_mol = Chem.SDMolSupplier(str(lig_path), removeHs=False)[0]

# 2. Initialize Unified Engine
unified_engine = UnifiedKinematicPSOEngine(
    receptor_pdb_path=rec_path,
    pocket_center=pocket_center,
    ligand_mol=lig_mol,
    flex_radius=9.0
)

# 3. Launch Unified PSO: 15 Particles x 15 Iterations (225 Synchronized Poses)
print("\n[*] Launching Unified Kin-PSO (15 Particles x 15 Iterations)...")
best_lig, best_rec_coords, best_score, lig_frames, rec_frames = unified_engine.run_unified_pso(
    n_particles=15,
    n_iterations=15
)

print(f"\n[✓] Unified Kin-PSO Converged to Coupled Best Score: {best_score:.3f} kcal/mol")

# 4. Save Best Pose & Best Induced-Fit Receptor
out_best_lig = DEMO_DIR / "unified_best_ligand.sdf"
w = Chem.SDWriter(str(out_best_lig))
w.write(best_lig)
w.close()

unified_engine.rec_kin.write_pdb_frame(best_rec_coords, DEMO_DIR / "unified_best_receptor.pdb")
print(f"[✓] Saved Best Docked Complex: {out_best_lig.name} + unified_best_receptor.pdb")

# 5. Save Synchronized Multi-Track Movie
out_lig_movie = DEMO_DIR / "unified_macrocycle_swarm.sdf"
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

out_rec_movie = DEMO_DIR / "unified_receptor_trajectory.pdb"
out_rec_movie.write_text("\n".join(pdb_models) + "\nEND\n")
print(f"[✓] Saved Synchronized 225-Frame Multi-Track Movie:")
print(f"    • Ligand Swarm:   {out_lig_movie.name}")
print(f"    • Receptor Track: {out_rec_movie.name}")

# 6. Generate Master PyMOL Script
flex_res_str = " or ".join(f"(resi {r.res_num} and resn {r.res_name})" for r in unified_engine.rec_kin.flex_residues)
pml_content = f"""# PyMOL Script for Grand Unified Kinematic PSO Demonstration
# Run directly in PyMOL: pymol visualize_unified_pso_pymol.pml

reinitialize
load unified_receptor_trajectory.pdb, keap1_swarm_receptor
load unified_macrocycle_swarm.sdf, q9e_swarm_ligand
load unified_best_receptor.pdb, best_receptor
load unified_best_ligand.sdf, best_ligand

# Style Receptor Swarm Track
hide everything, keap1_swarm_receptor
show cartoon, keap1_swarm_receptor
color wheat, keap1_swarm_receptor
show surface, keap1_swarm_receptor
set transparency, 0.70, keap1_swarm_receptor

# Style Active-Site Arginine Triad & Polar Network
select active_pocket, keap1_swarm_receptor and ({flex_res_str})
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

select arginines, keap1_swarm_receptor and (resn ARG and resi 415+483+380+336+601)
color marine, arginines
select tyrosines, keap1_swarm_receptor and (resn TYR and resi 334+572+525)
color tv_green, tyrosines
select serines, keap1_swarm_receptor and (resn SER and resi 602+555+363+338)
color orange, serines

# Style Ligand Swarm Track
hide everything, q9e_swarm_ligand
show sticks, q9e_swarm_ligand
color magenta, q9e_swarm_ligand
set stick_radius, 0.24, q9e_swarm_ligand

# Style Converged Best Complex (Hidden initially, show on pause)
hide everything, best_receptor
hide everything, best_ligand
show sticks, best_ligand
color cyan, best_ligand
set stick_radius, 0.28, best_ligand

# Dynamic Contact Distances
distance sb_arg415, (keap1_swarm_receptor and resi 415 and name NH1), (q9e_swarm_ligand and name O28), 3.5
distance sb_arg483, (keap1_swarm_receptor and resi 483 and name NH2), (q9e_swarm_ligand and name O19), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
set dash_width, 3.0

zoom q9e_swarm_ligand, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 225-Frame GRAND UNIFIED KINEMATIC PSO MOVIE!"
print "  • Wheat Cartoon : Keap1 Kelch β-Propeller Backbone"
print "  • Marine Sticks : Arginine Triad (Arg415, 483, 380) Flexing (χ₁-χ₄)"
print "  • Magenta Sticks: 16-Membered Macrocycle Swarm (Ring IK + Arm FK)"
print "  • Yellow Dashes : Dynamic Salt Bridges Formed in Real-Time"
print "  Press Play (bottom right) or Spacebar to watch the coupled swarm!"
print "================================================================="
"""
(DEMO_DIR / "visualize_unified_pso_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_unified_pso_pymol.pml'}")
