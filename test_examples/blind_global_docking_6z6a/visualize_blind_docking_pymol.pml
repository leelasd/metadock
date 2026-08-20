# PyMOL Script for Global Blind Docking Demonstration
# Run directly in PyMOL: pymol visualize_blind_docking_pymol.pml

reinitialize
load blind_docking_receptor_trajectory.pdb, keap1_blind_receptor
load blind_docking_swarm_trajectory.sdf, q9e_blind_swarm
load q9e_unaligned_start.sdf, start_unaligned_ligand
load q9e_crystal_pose.sdf, reference_crystal_pose
load blind_docking_best_pose.sdf, best_docked_macrocycle

# Style Receptor Backbone
hide everything, keap1_blind_receptor
show cartoon, keap1_blind_receptor
color wheat, keap1_blind_receptor
show surface, keap1_blind_receptor
set transparency, 0.70, keap1_blind_receptor

# Style Active Pocket Residues
select active_pocket, keap1_blind_receptor and ((resi 334 and resn TYR) or (resi 335 and resn PHE) or (resi 338 and resn SER) or (resi 363 and resn SER) or (resi 380 and resn ARG) or (resi 382 and resn ASN) or (resi 414 and resn ASN) or (resi 415 and resn ARG) or (resi 483 and resn ARG) or (resi 530 and resn GLN) or (resi 555 and resn SER) or (resi 572 and resn TYR) or (resi 577 and resn PHE) or (resi 602 and resn SER))
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

select arginines, keap1_blind_receptor and (resn ARG and resi 415+483+380+336+601)
color marine, arginines
select tyrosines, keap1_blind_receptor and (resn TYR and resi 334+572+525)
color tv_green, tyrosines
select serines, keap1_blind_receptor and (resn SER and resi 602+555+363+338)
color orange, serines

# Style Starting Pose (Red) & Reference Crystal (Green)
hide everything, start_unaligned_ligand
show sticks, start_unaligned_ligand
color red, start_unaligned_ligand
set stick_radius, 0.22, start_unaligned_ligand

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose
set stick_radius, 0.22, reference_crystal_pose

# Style Swarm Trajectory (Magenta)
hide everything, q9e_blind_swarm
show sticks, q9e_blind_swarm
color magenta, q9e_blind_swarm
set stick_radius, 0.24, q9e_blind_swarm

# Style Best Docked Macrocycle (Cyan)
hide everything, best_docked_macrocycle
show sticks, best_docked_macrocycle
color cyan, best_docked_macrocycle
set stick_radius, 0.28, best_docked_macrocycle

# Dynamic Distance Contacts to Arginines
distance sb_arg415, (keap1_blind_receptor and resi 415 and name NH1), (q9e_blind_swarm and name O28), 3.5
distance sb_arg483, (keap1_blind_receptor and resi 483 and name NH2), (q9e_blind_swarm and name O19), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
set dash_width, 3.0

zoom keap1_blind_receptor, 8.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 600-Frame GLOBAL BLIND DOCKING MOVIE!"
print "  • Red Sticks   : Initial Unaligned Starting Pose in Solvent (19.5 Å RMSD)"
print "  • Green Sticks : Reference Co-Crystal Pose (0.0 Å RMSD)"
print "  • Magenta Sticks: 30-Walker Swarm Navigating from Solvent to Cleft"
print "  • Cyan Sticks  : Final Converged Complex"
print "  Press Play (bottom right) or Spacebar to watch blind docking!"
print "================================================================="
