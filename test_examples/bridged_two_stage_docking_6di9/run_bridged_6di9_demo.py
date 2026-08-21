#!/usr/bin/env python
"""
Automated Bridged Two-Stage Docking on Non-Macrocyclic System (PDB 6DI9: BTK Kinase + GJJ Inhibitor).
Demonstrates:
Stage 1: Global Swarm Ingress from Bulk Solvent (14D Search Space, 18+ Å RMSD).
The Bridge Gate: Automated detection of cavity entry.
Stage 2: In-Pocket Kinematic Induced-Fit Refinement (Ligand FK + Active-Site χ Plasticity).
"""
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation as ScipyRotation

from openmm_dock.bridged_docking import BridgedTwoStageDockingEngine
from openmm_dock.global_blind_docking import BlindDockingParams

def main():
    root = Path(__file__).parent
    src_dir = root.parent / "covalent_docking" / "6di9"
    
    rec_path = root / "receptor.pdb"
    xtal_path = root / "gjj_crystal_pose.sdf"
    unaligned_path = root / "gjj_unaligned_start.sdf"
    
    import shutil
    shutil.copy(src_dir / "receptor.pdb", rec_path)
    shutil.copy(src_dir / "xtal_ligand.sdf", xtal_path)
    if (src_dir / "cavity.prm").exists():
        shutil.copy(src_dir / "cavity.prm", root / "cavity.prm")
        
    xtal_mol = Chem.SDMolSupplier(str(xtal_path), removeHs=False)[0]
    conf_x = xtal_mol.GetConformer()
    coords_x = np.array([conf_x.GetAtomPosition(i) for i in range(xtal_mol.GetNumAtoms())])
    pocket_center = np.mean(coords_x, axis=0)
    
    # Create unaligned starting pose in bulk solvent (+18.0 A translation + 180° inversion)
    unaligned_mol = Chem.Mol(xtal_mol)
    conf_u = unaligned_mol.GetConformer()
    center = np.mean(coords_x, axis=0)
    q_rot = ScipyRotation.from_euler("zyx", [135, -45, 110], degrees=True).as_matrix()
    coords_u = (coords_x - center).dot(q_rot.T) + center + np.array([12.0, 10.0, -11.0])
    
    for i in range(xtal_mol.GetNumAtoms()):
        conf_u.SetAtomPosition(i, Point3D(float(coords_u[i][0]), float(coords_u[i][1]), float(coords_u[i][2])))
        
    w = Chem.SDWriter(str(unaligned_path))
    w.write(unaligned_mol)
    w.close()
    
    init_rmsd = float(np.sqrt(np.mean(np.sum((coords_u - coords_x)**2, axis=1))))
    print(f"[*] Generated Non-Macrocyclic Starting Pose (BTK Inhibitor GJJ):")
    print(f"    • Initial Distance to Pocket Center: {np.linalg.norm(np.mean(coords_u, axis=0) - pocket_center):.3f} Å")
    print(f"    • Initial RMSD in Bulk Solvent     : {init_rmsd:.3f} Å")
    
    engine = BridgedTwoStageDockingEngine(
        receptor_pdb_path=rec_path,
        pocket_center=pocket_center,
        ligand_mol=xtal_mol,
        flex_radius=8.0
    )
    
    params = BlindDockingParams(
        n_particles=35,
        n_iterations=20,
        num_conformer_seeds=1, # Non-macrocyclic single conformer seed
        search_box_size=22.0,
        w_start=0.82,
        w_end=0.35,
        c1_cognitive=1.3,
        c2_social=2.6,
        k_contact_beacon=1.0,
        k_depth_beacon=4.5,
        gaussian_w0=8.0,
        gaussian_sigma=0.50,
        bias_gamma=6.0,
        lbfgs_iterations=100
    )
    
    best_lig, best_rec_coords, best_phys, all_lig_frames, all_rec_frames, master_log = engine.run_bridged_docking_pipeline(
        unaligned_start_mol=unaligned_mol,
        reference_xtal_mol=xtal_mol,
        stage1_params=params
    )
    
    # Save Outputs
    out_swarm_sdf = root / "bridged_6di9_swarm_trajectory.sdf"
    out_rec_pdb = root / "bridged_6di9_receptor_trajectory.pdb"
    out_best_sdf = root / "bridged_6di9_best_pose.sdf"
    out_best_rec = root / "bridged_6di9_best_receptor.pdb"
    out_plot = root / "bridged_6di9_stage_transition.png"
    
    w = Chem.SDWriter(str(out_swarm_sdf))
    for m in all_lig_frames:
        w.write(m)
    w.close()
    
    w_best = Chem.SDWriter(str(out_best_sdf))
    w_best.write(best_lig)
    w_best.close()
    
    engine.engine_stage2.rec_kin.write_multi_model_trajectory(
        all_rec_frames[::max(1, len(all_rec_frames)//100)],
        out_rec_pdb
    )
    
    engine.engine_stage2.rec_kin.write_pdb_frame(
        best_rec_coords,
        out_best_rec
    )
    
    engine.plot_stage_transition(master_log, out_plot)
    
    # Write PyMOL script
    pml_path = root / "visualize_bridged_6di9_pymol.pml"
    with open(pml_path, "w") as f:
        f.write(f"""# PyMOL Visualization Script: Bridged Two-Stage Docking on BTK Kinase (PDB 6DI9)
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2

# Load Receptor and Crystal Reference
load receptor.pdb, btk_static
load gjj_crystal_pose.sdf, gjj_crystal_reference
load gjj_unaligned_start.sdf, gjj_unaligned_solvent_start

# Load 2-Stage Multi-Track Trajectories
load bridged_6di9_swarm_trajectory.sdf, bridged_swarm_movie
load bridged_6di9_receptor_trajectory.pdb, bridged_receptor_movie
load bridged_6di9_best_pose.sdf, bridged_converged_pose
load bridged_6di9_best_receptor.pdb, bridged_converged_receptor

# Style Static Receptor
hide everything, btk_static
show cartoon, btk_static
color gray85, btk_static
set cartoon_transparency, 0.4, btk_static

# Style Crystal Reference
show sticks, gjj_crystal_reference
color forest, gjj_crystal_reference
set stick_radius, 0.28, gjj_crystal_reference

# Style Unaligned Solvent Start (Bulk Solvent)
show sticks, gjj_unaligned_solvent_start
color firebrick, gjj_unaligned_solvent_start
set stick_radius, 0.25, gjj_unaligned_solvent_start

# Style Swarm Movie
show sticks, bridged_swarm_movie
color magenta, bridged_swarm_movie
set stick_radius, 0.18, bridged_swarm_movie
set stick_transparency, 0.2, bridged_swarm_movie

# Style Converged Pose
show sticks, bridged_converged_pose
color cyan, bridged_converged_pose
set stick_radius, 0.32, bridged_converged_pose

# Active-Site Pocket Residues (BTK Kinase Hinge & Catalytic Cleft)
select btk_hinge, resi 474+475+477+481+430 and btk_static
show sticks, btk_hinge
color marine, btk_hinge
set stick_radius, 0.22, btk_hinge
label name CA and resi 477, '"Met-477 (Hinge)"'
label name CA and resi 481, '"Cys-481 (Covalent)"'
label name CA and resi 430, '"Lys-430"'
set label_size, 14
set label_color, black

zoom resi 477+481+430 or gjj_crystal_reference, 10.0
mplay
""")
    print(f"\n[✓] PyMOL visualizer created: {pml_path}")

if __name__ == "__main__":
    main()
