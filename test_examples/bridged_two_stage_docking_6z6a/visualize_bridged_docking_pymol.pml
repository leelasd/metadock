# PyMOL Visualization Script: Automated Bridged Two-Stage Docking Pipeline (PDB 6Z6A)
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
