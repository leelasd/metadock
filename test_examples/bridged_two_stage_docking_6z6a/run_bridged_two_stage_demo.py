#!/usr/bin/env python
"""
Automated Bridged Two-Stage Docking Demonstration on PDB 6Z6A (Keap1 + Q9E Macrocycle).
Shows how the algorithm automatically connects:
Stage 1: Global Swarm-Metadynamics Ingress from Bulk Solvent (19D Search Space).
The Ingress Gate: Handover trigger when ligand enters the pocket cavity.
Stage 2: In-Pocket Kinematic Induced-Fit Relaxation (Two-Tier IK/FK + Receptor χ₁–χ₄ Plasticity).
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
    rec_path = root / "receptor.pdb"
    xtal_path = root / "q9e_crystal_pose.sdf"
    unaligned_path = root / "q9e_unaligned_start.sdf"
    
    # Copy assets if missing
    if not rec_path.exists():
        import shutil
        src_dir = root.parent / "macrocycle_6z6a"
        shutil.copy(src_dir / "receptor.pdb", rec_path)
        shutil.copy(src_dir / "q9e_crystal_pose.sdf", xtal_path)
        shutil.copy(src_dir / "cavity.prm", root / "cavity.prm")
        
    xtal_mol = Chem.SDMolSupplier(str(xtal_path), removeHs=False)[0]
    conf_x = xtal_mol.GetConformer()
    coords_x = np.array([conf_x.GetAtomPosition(i) for i in range(xtal_mol.GetNumAtoms())])
    pocket_center = np.array([-21.46, 22.44, -24.18])
    
    # Create unaligned starting pose in bulk solvent (18.97 A away, inverted)
    if not unaligned_path.exists():
        unaligned_mol = Chem.Mol(xtal_mol)
        conf_u = unaligned_mol.GetConformer()
        center = np.mean(coords_x, axis=0)
        q_rot = ScipyRotation.from_euler("zyx", [140, 60, -90], degrees=True).as_matrix()
        coords_u = (coords_x - center).dot(q_rot.T) + center + np.array([12.0, -11.0, 9.0])
        for i in range(xtal_mol.GetNumAtoms()):
            conf_u.SetAtomPosition(i, Point3D(float(coords_u[i][0]), float(coords_u[i][1]), float(coords_u[i][2])))
        w = Chem.SDWriter(str(unaligned_path))
        w.write(unaligned_mol)
        w.close()
    else:
        unaligned_mol = Chem.SDMolSupplier(str(unaligned_path), removeHs=False)[0]
        
    engine = BridgedTwoStageDockingEngine(
        receptor_pdb_path=rec_path,
        pocket_center=pocket_center,
        ligand_mol=xtal_mol,
        flex_radius=9.0
    )
    
    params = BlindDockingParams(
        n_particles=40,
        n_iterations=20,
        num_conformer_seeds=6,
        search_box_size=24.0,
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
    out_swarm_sdf = root / "bridged_docking_swarm_trajectory.sdf"
    out_rec_pdb = root / "bridged_docking_receptor_trajectory.pdb"
    out_best_sdf = root / "bridged_docking_best_pose.sdf"
    out_best_rec = root / "bridged_docking_best_receptor.pdb"
    out_plot = root / "bridged_docking_stage_transition.png"
    
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
    pml_path = root / "visualize_bridged_docking_pymol.pml"
    with open(pml_path, "w") as f:
        f.write(f"""# PyMOL Visualization Script: Automated Bridged Two-Stage Docking Pipeline (PDB 6Z6A)
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2

# Load Receptor and Crystal Reference
load receptor.pdb, keap1_static
load q9e_crystal_pose.sdf, q9e_crystal_reference
load q9e_unaligned_start.sdf, q9e_unaligned_solvent_start

# Load 2-Stage Multi-Track Trajectories
load bridged_docking_swarm_trajectory.sdf, bridged_swarm_movie
load bridged_docking_receptor_trajectory.pdb, bridged_receptor_movie
load bridged_docking_best_pose.sdf, bridged_converged_pose
load bridged_docking_best_receptor.pdb, bridged_converged_receptor

# Style Static Receptor
hide everything, keap1_static
show cartoon, keap1_static
color gray85, keap1_static
set cartoon_transparency, 0.4, keap1_static

# Style Crystal Reference
show sticks, q9e_crystal_reference
color forest, q9e_crystal_reference
set stick_radius, 0.28, q9e_crystal_reference

# Style Unaligned Solvent Start (Bulk Solvent)
show sticks, q9e_unaligned_solvent_start
color firebrick, q9e_unaligned_solvent_start
set stick_radius, 0.25, q9e_unaligned_solvent_start

# Style Swarm Movie
show sticks, bridged_swarm_movie
color magenta, bridged_swarm_movie
set stick_radius, 0.18, bridged_swarm_movie
set stick_transparency, 0.2, bridged_swarm_movie

# Style Converged Pose
show sticks, bridged_converged_pose
color cyan, bridged_converged_pose
set stick_radius, 0.32, bridged_converged_pose

# Active-Site Pocket Residues (Arginine Triad)
select active_site, resi 415+483+380+334+572 and keap1_static
show sticks, active_site
color marine, active_site
set stick_radius, 0.22, active_site
label name CA and resi 415, '"Arg-415"'
label name CA and resi 483, '"Arg-483"'
label name CA and resi 380, '"Arg-380"'
set label_size, 14
set label_color, black

zoom resi 415+483+380 or q9e_crystal_reference, 10.0
mplay
""")
    print(f"\n[✓] PyMOL visualizer created: {pml_path}")

if __name__ == "__main__":
    main()
