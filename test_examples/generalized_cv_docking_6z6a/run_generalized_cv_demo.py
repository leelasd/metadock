"""
Demonstration of Generalized Reference-Free Collective Variables (CVs) in Docking.
Unites:
1. Pocket Penetration Depth (zeta_depth): Reference-free distance to cavity centroid.
2. Continuous Contact Coordination Number (Q_contacts): Shape complementarity.
3. Macrocycle Radius of Gyration (R_g): Ring pucker and conformational envelope.
4. Universal 2D Free Energy Binding Funnel Reconstruction: F(zeta_depth, Q_contacts).
"""
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock.unified_kinematic_pso import UnifiedKinematicPSOEngine
from openmm_dock.generalized_cv import GeneralizedCVMetadynamicsEngine

DEMO_DIR = Path(__file__).resolve().parent
rec_path = DEMO_DIR / "receptor.pdb"
pocket_center = np.array([-21.46, 22.44, -24.18])

print("=" * 95)
print("   OPENMM-DOCK: GENERALIZED REFERENCE-FREE CV METADYNAMICS & BINDING FUNNEL (6Z6A)")
print("=" * 95)

# 1. Load Macrocycle
lig_path = DEMO_DIR / "q9e_crystal_pose.sdf"
lig_mol = Chem.SDMolSupplier(str(lig_path), removeHs=False)[0]

# 2. Setup Unified Engine and Generalized CV Metadynamics Engine
unified_engine = UnifiedKinematicPSOEngine(
    receptor_pdb_path=rec_path,
    pocket_center=pocket_center,
    ligand_mol=lig_mol,
    flex_radius=9.0
)

gen_engine = GeneralizedCVMetadynamicsEngine(
    unified_engine,
    initial_height_w0=6.0,
    sigma_zeta=0.60,       # 0.60 Å depth resolution
    sigma_q=8.0,           # 8 contacts coordination resolution
    bias_factor_gamma=5.0
)

# 3. Run Generalized Swarm Metadynamics (15 Walkers x 20 Iterations = 300 Frames)
print("\n[*] Launching Generalized Reference-Free Swarm Metadynamics (300 Frames)...")
best_lig, best_rec_coords, best_score, lig_frames, rec_frames, cv_log = gen_engine.run_generalized_docking_metadynamics(
    n_particles=15,
    n_iterations=20
)

print(f"\n[✓] Generalized CV Metadynamics Completed!")
print(f"    • Total Exploration Frames Generated: {len(lig_frames)} frames")
print(f"    • Total Generalized Basins Deposited: {len(gen_engine.visited_basins)} hills")
print(f"    • Global Best Physical Score        : {best_score:.3f} kcal/mol")

# 4. Reconstruct Universal 2D Free Energy Binding Funnel F(zeta_depth, Q_contacts)
fes_png = DEMO_DIR / "universal_binding_funnel_fes.png"
print("\n[*] Reconstructing Universal 2D Free Energy Binding Funnel F(zeta_depth, Q_contacts)...")
gen_engine.plot_universal_binding_funnel_fes(cv_log, fes_png)

# 5. Print Sample Log of Generalized CVs
print("\n" + "=" * 90)
print(f"{'Frame':<6} | {'Pocket Depth ζ (Å)':<20} | {'Coordination Q':<18} | {'Gyration Rg (Å)':<18} | {'Score (kcal)':<15}")
print("-" * 90)
for row in cv_log[::30]:
    print(f"{row['frame']:<6d} | {row['zeta_depth_A']:<20.2f} | {row['q_contacts']:<18.1f} | {row['r_g_A']:<18.2f} | {row['score_kcal']:<15.2f}")
print("=" * 90)

# 6. Save Multi-Track PyMOL Movie
out_lig_movie = DEMO_DIR / "generalized_cv_trajectory.sdf"
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

out_rec_movie = DEMO_DIR / "generalized_cv_receptor_trajectory.pdb"
out_rec_movie.write_text("\n".join(pdb_models) + "\nEND\n")

# Save Best Complex
w_best = Chem.SDWriter(str(DEMO_DIR / "generalized_cv_best_pose.sdf"))
w_best.write(best_lig)
w_best.close()
unified_engine.rec_kin.write_pdb_frame(best_rec_coords, DEMO_DIR / "generalized_cv_best_receptor.pdb")

print(f"\n[✓] Saved Synchronized 300-Frame Generalized CV Trajectory:")
print(f"    • Ligand Movie  : {out_lig_movie.name}")
print(f"    • Receptor Movie: {out_rec_movie.name}")
print(f"    • Funnel FES    : universal_binding_funnel_fes.png")

# 7. Generate PyMOL Script
flex_res_str = " or ".join(f"(resi {r.res_num} and resn {r.res_name})" for r in unified_engine.rec_kin.flex_residues)
pml_content = f"""# PyMOL Script for Generalized Reference-Free CV Metadynamics Movie
# Run directly in PyMOL: pymol visualize_generalized_cv_pymol.pml

reinitialize
load generalized_cv_receptor_trajectory.pdb, keap1_gen_receptor
load generalized_cv_trajectory.sdf, q9e_gen_macrocycle
load generalized_cv_best_receptor.pdb, best_receptor
load generalized_cv_best_pose.sdf, best_macrocycle

# Style Receptor Backbone
hide everything, keap1_gen_receptor
show cartoon, keap1_gen_receptor
color wheat, keap1_gen_receptor
show surface, keap1_gen_receptor
set transparency, 0.70, keap1_gen_receptor

# Style Active Pocket Residues
select active_pocket, keap1_gen_receptor and ({flex_res_str})
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

select arginines, keap1_gen_receptor and (resn ARG and resi 415+483+380+336+601)
color marine, arginines
select tyrosines, keap1_gen_receptor and (resn TYR and resi 334+572+525)
color tv_green, tyrosines
select serines, keap1_gen_receptor and (resn SER and resi 602+555+363+338)
color orange, serines

# Style Macrocycle Trajectory
hide everything, q9e_gen_macrocycle
show sticks, q9e_gen_macrocycle
color magenta, q9e_gen_macrocycle
set stick_radius, 0.24, q9e_gen_macrocycle

# Style Best Complex (Shown on pause)
hide everything, best_receptor
hide everything, best_macrocycle
show sticks, best_macrocycle
color cyan, best_macrocycle
set stick_radius, 0.28, best_macrocycle

# Dynamic Distance Contacts to Arginines
distance sb_arg415, (keap1_gen_receptor and resi 415 and name NH1), (q9e_gen_macrocycle and name O28), 3.5
distance sb_arg483, (keap1_gen_receptor and resi 483 and name NH2), (q9e_gen_macrocycle and name O19), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
set dash_width, 3.0

zoom q9e_gen_macrocycle, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 300-Frame GENERALIZED CV METADYNAMICS MOVIE!"
print "  • Reference-Free CVs: Pocket Depth (ζ_depth) & Coordination (Q)"
print "  • Generated universal_binding_funnel_fes.png 2D FES"
print "  • Press Play (bottom right) or Spacebar to watch generalized swarm!"
print "================================================================="
"""
(DEMO_DIR / "visualize_generalized_cv_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_generalized_cv_pymol.pml'}")
