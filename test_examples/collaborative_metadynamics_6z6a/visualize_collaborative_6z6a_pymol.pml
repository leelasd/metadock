# PyMOL Visualization Script: Collaborative Multi-Swarm Kinematic MetaD on Keap1 + Q9E (PDB 6Z6A)
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2

# 1. Load Static Receptor and Reference
load receptor.pdb, keap1_scaffold
load q9e_crystal_pose.sdf, q9e_crystal_reference

# 2. Load Trajectories and Shared Basins
load collaborative_trajectory.sdf, multi_island_movie
load collaborative_shared_basins.sdf, shared_metad_basins
load collaborative_best_pose.sdf, collaborative_converged_pose

# Style Receptor Scaffold
hide everything, keap1_scaffold
show cartoon, keap1_scaffold
color gray85, keap1_scaffold
set cartoon_transparency, 0.45, keap1_scaffold

# Style Crystal Reference
show sticks, q9e_crystal_reference
color forest, q9e_crystal_reference
set stick_radius, 0.28, q9e_crystal_reference

# Style Converged Best Pose
show sticks, collaborative_converged_pose
color gold, collaborative_converged_pose
set stick_radius, 0.35, collaborative_converged_pose

# Style Shared Metadynamics Repulsive Basins (Negative Memory)
show spheres, shared_metad_basins
color warmpink, shared_metad_basins
set sphere_scale, 0.25, shared_metad_basins
set sphere_transparency, 0.35, shared_metad_basins

# Style Multi-Island Swarm Movie
show sticks, multi_island_movie
set stick_radius, 0.16, multi_island_movie
set stick_transparency, 0.2, multi_island_movie

# Color Swarm by Discovering Island
color coral, multi_island_movie and (elem C)
# Sub-selections for islands in movie
select island1, multi_island_movie and prop ISLAND_ID == 1
select island2, multi_island_movie and prop ISLAND_ID == 2
select island3, multi_island_movie and prop ISLAND_ID == 3
select island4, multi_island_movie and prop ISLAND_ID == 4
color tv_red, island1
color cyan, island2
color mediumseagreen, island3
color slate, island4

# Active-Site Pocket Residues (Keap1 Arginine Triad & Aromatic Clasp)
select keap1_pocket, resi 415+483+380+441+334+525+572+577 and keap1_scaffold
show sticks, keap1_pocket
color marine, keap1_pocket
set stick_radius, 0.22, keap1_pocket
label name CA and resi 415, '"Arg-415"'
label name CA and resi 483, '"Arg-483"'
label name CA and resi 380, '"Arg-380"'
label name CA and resi 525, '"Tyr-525"'
set label_size, 14
set label_color, black

zoom q9e_crystal_reference, 10.0
mplay
