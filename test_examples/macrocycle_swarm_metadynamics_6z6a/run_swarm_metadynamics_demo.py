"""
Demonstration of Swarm Metadynamics (MetaD-PSO) and 2D Free Energy Surface Reconstruction.
Combines Multiple-Walker Well-Tempered Metadynamics with Particle Swarm Optimization (PSO)
to generate 300 exploration frames, reconstruct F(RMSD, d_SB), and plot per-residue footprints.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock.unified_kinematic_pso import UnifiedKinematicPSOEngine
from openmm_dock.swarm_metadynamics import SwarmMetadynamicsEngine

DEMO_DIR = Path(__file__).resolve().parent
rec_path = DEMO_DIR / "receptor.pdb"
pocket_center = np.array([-21.46, 22.44, -24.18])

print("=" * 90)
print("   OPENMM-DOCK: SWARM METADYNAMICS & 2D FREE ENERGY RECONSTRUCTION (PDB 6Z6A)")
print("=" * 90)

# 1. Load Macrocycle
lig_path = DEMO_DIR / "q9e_crystal_pose.sdf"
lig_mol = Chem.SDMolSupplier(str(lig_path), removeHs=False)[0]

# 2. Setup Unified Engine and Swarm Metadynamics Engine
unified_engine = UnifiedKinematicPSOEngine(
    receptor_pdb_path=rec_path,
    pocket_center=pocket_center,
    ligand_mol=lig_mol,
    flex_radius=9.0
)

swarm_metad = SwarmMetadynamicsEngine(
    unified_engine,
    initial_height_w0=6.0,
    gaussian_sigma=0.50,
    bias_factor_gamma=5.0
)

# 3. Launch Swarm Metadynamics (15 Particles x 20 Iterations = 300 Exploration Frames)
print("\n[*] Launching Swarm Metadynamics: 15 Walkers x 20 Iterations (300 Frames)...")
best_lig, best_rec_coords, best_score, lig_frames, rec_frames, cv_log = swarm_metad.run_swarm_metadynamics(
    n_particles=15,
    n_iterations=20
)

print(f"\n[✓] Swarm Metadynamics Completed!")
print(f"    • Total Exploration Frames Generated: {len(lig_frames)} frames")
print(f"    • Total Shared Gaussian Hills       : {len(swarm_metad.shared_basins)} hills")
print(f"    • Global Best Physical Score        : {best_score:.3f} kcal/mol")

# 4. Reconstruct 2D Free Energy Surface F(RMSD, d_Arg415)
fes_png = DEMO_DIR / "free_energy_surface.png"
print("\n[*] Reconstructing 2D Free Energy Surface F(RMSD, d_Arg415)...")
swarm_metad.reconstruct_free_energy_surface_2d(cv_log, fes_png)

# 5. Compute Per-Residue Binding Energy Footprint
footprint_png = DEMO_DIR / "per_residue_energy_footprint.png"
print("\n[*] Computing Keap1 Per-Residue Interaction Energy Footprint...")
res_energies = swarm_metad.compute_per_residue_energy_footprint(best_lig, best_rec_coords, footprint_png)

print("\n" + "=" * 60)
print(f"{'Active-Site Residue':<25} | {'Interaction Energy (kcal/mol)':<28}")
print("-" * 60)
for rname, e_val in res_energies[:8]:
    print(f"{rname:<25} | {e_val:<28.2f}")
print("=" * 60)

# 6. Save Multi-Track Trajectories
out_lig_movie = DEMO_DIR / "swarm_metadynamics_trajectory.sdf"
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

out_rec_movie = DEMO_DIR / "swarm_receptor_trajectory.pdb"
out_rec_movie.write_text("\n".join(pdb_models) + "\nEND\n")

# Save Best Pose
w_best = Chem.SDWriter(str(DEMO_DIR / "swarm_metadynamics_best_pose.sdf"))
w_best.write(best_lig)
w_best.close()
unified_engine.rec_kin.write_pdb_frame(best_rec_coords, DEMO_DIR / "swarm_metadynamics_best_receptor.pdb")

print(f"\n[✓] Saved Synchronized 300-Frame Multi-Track Trajectory:")
print(f"    • Ligand Movie  : {out_lig_movie.name}")
print(f"    • Receptor Movie: {out_rec_movie.name}")
print(f"    • Free Energy 2D: free_energy_surface.png")
print(f"    • Energy Barplot: per_residue_energy_footprint.png")

# 7. Generate PyMOL Script
flex_res_str = " or ".join(f"(resi {r.res_num} and resn {r.res_name})" for r in unified_engine.rec_kin.flex_residues)
pml_content = f"""# PyMOL Script for Swarm Metadynamics (MetaD-PSO) Movie
# Run directly in PyMOL: pymol visualize_swarm_metadynamics_pymol.pml

reinitialize
load swarm_receptor_trajectory.pdb, keap1_swarm_receptor
load swarm_metadynamics_trajectory.sdf, q9e_swarm_macrocycle
load swarm_metadynamics_best_receptor.pdb, best_receptor
load swarm_metadynamics_best_pose.sdf, best_macrocycle

# Style Receptor Backbone
hide everything, keap1_swarm_receptor
show cartoon, keap1_swarm_receptor
color wheat, keap1_swarm_receptor
show surface, keap1_swarm_receptor
set transparency, 0.70, keap1_swarm_receptor

# Style Active Pocket Residues
select active_pocket, keap1_swarm_receptor and ({flex_res_str})
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

select arginines, keap1_swarm_receptor and (resn ARG and resi 415+483+380+336+601)
color marine, arginines
select tyrosines, keap1_swarm_receptor and (resn TYR and resi 334+572+525)
color tv_green, tyrosines
select serines, keap1_swarm_receptor and (resn SER and resi 602+555+363+338)
color orange, serines

# Style Macrocycle Trajectory
hide everything, q9e_swarm_macrocycle
show sticks, q9e_swarm_macrocycle
color magenta, q9e_swarm_macrocycle
set stick_radius, 0.24, q9e_swarm_macrocycle

# Style Best Complex (Shown on pause)
hide everything, best_receptor
hide everything, best_macrocycle
show sticks, best_macrocycle
color cyan, best_macrocycle
set stick_radius, 0.28, best_macrocycle

# Dynamic Distance Contacts to Arginines
distance sb_arg415, (keap1_swarm_receptor and resi 415 and name NH1), (q9e_swarm_macrocycle and name O28), 3.5
distance sb_arg483, (keap1_swarm_receptor and resi 483 and name NH2), (q9e_swarm_macrocycle and name O19), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
set dash_width, 3.0

zoom q9e_swarm_macrocycle, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 300-Frame SWARM METADYNAMICS (MetaD-PSO) MOVIE!"
print "  • Multiple-Walker Well-Tempered Metadynamics + Swarm Intelligence"
print "  • Energetics: Generated free_energy_surface.png and footprint plot"
print "  • Press Play (bottom right) or Spacebar to watch 3D swarm dynamics!"
print "================================================================="
"""
(DEMO_DIR / "visualize_swarm_metadynamics_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_swarm_metadynamics_pymol.pml'}")
