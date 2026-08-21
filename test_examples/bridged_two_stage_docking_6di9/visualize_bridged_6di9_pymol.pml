# PyMOL Visualization Script: Bridged Two-Stage Docking on BTK Kinase (PDB 6DI9)
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
