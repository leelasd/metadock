# PyMOL Script for Global Blind Docking Demonstration (PDB 9Z1L)
# Run directly in PyMOL: pymol visualize_blind_docking_pymol.pml

reinitialize
load blind_docking_receptor_trajectory.pdb, kit_blind_receptor
load blind_docking_swarm_trajectory.sdf, a1czz_blind_swarm
load a1czz_unaligned_start.sdf, start_unaligned_ligand
load a1czz_crystal_pose.sdf, reference_crystal_pose
load blind_docking_best_pose.sdf, best_docked_ligand

hide everything, kit_blind_receptor
show cartoon, kit_blind_receptor
color wheat, kit_blind_receptor
show surface, kit_blind_receptor
set transparency, 0.70, kit_blind_receptor

select active_pocket, kit_blind_receptor and ((resi 39 and resn LEU) or (resi 47 and resn VAL) or (resi 48 and resn VAL) or (resi 49 and resn GLU) or (resi 66 and resn VAL) or (resi 114 and resn THR) or (resi 115 and resn GLU) or (resi 116 and resn TYR) or (resi 117 and resn CYS) or (resi 118 and resn CYS) or (resi 119 and resn TYR) or (resi 121 and resn ASP) or (resi 177 and resn LEU) or (resi 178 and resn LEU) or (resi 187 and resn CYS) or (resi 188 and resn ASP) or (resi 189 and resn PHE))
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

hide everything, start_unaligned_ligand
show sticks, start_unaligned_ligand
color red, start_unaligned_ligand
set stick_radius, 0.22, start_unaligned_ligand

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose
set stick_radius, 0.22, reference_crystal_pose

hide everything, a1czz_blind_swarm
show sticks, a1czz_blind_swarm
color magenta, a1czz_blind_swarm
set stick_radius, 0.24, a1czz_blind_swarm

hide everything, best_docked_ligand
show sticks, best_docked_ligand
color cyan, best_docked_ligand
set stick_radius, 0.28, best_docked_ligand

zoom kit_blind_receptor, 8.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 600-Frame GLOBAL BLIND DOCKING MOVIE! (KIT V654A + BLU-654)"
print "  • Red Sticks    : Initial Unaligned Starting Pose in Solvent"
print "  • Green Sticks  : Reference Co-Crystal Pose (0.0 Å RMSD)"
print "  • Magenta Sticks: 30-Walker Swarm Navigating from Solvent to Cleft"
print "  • Cyan Sticks   : Final Converged Complex"
print "  Press Play (bottom right) or Spacebar to watch blind docking!"
print "================================================================="
