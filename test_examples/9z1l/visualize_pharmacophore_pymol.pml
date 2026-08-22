# PyMOL Script for Pharmacophore-Restraint Docking Comparison (PDB 9Z1L)
reinitialize
load receptor.mol2, kit_receptor
load a1czz_crystal_pose.sdf, reference_crystal_pose
load pharma_dock_free_out.sdf, free_docked
load pharma_dock_restrained_out.sdf, restrained_docked
load pharma_dock_final_out.sdf, final_docked
load pharma_dock_top5_diverse_out.sdf, top5_diverse

hide everything, kit_receptor
show cartoon, kit_receptor
color wheat, kit_receptor

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose

hide everything, free_docked
show sticks, free_docked
color red, free_docked
set stick_radius, 0.18, free_docked

hide everything, restrained_docked
show sticks, restrained_docked
color orange, restrained_docked
set stick_radius, 0.20, restrained_docked

hide everything, final_docked
show sticks, final_docked
color cyan, final_docked
set stick_radius, 0.24, final_docked

hide everything, top5_diverse
show sticks, top5_diverse
util.cbaw top5_diverse
set stick_radius, 0.16, top5_diverse
set all_states, on, top5_diverse

zoom reference_crystal_pose, 8.0

print "================================================================="
print "  Green: crystal reference | Red: free-docked (no restraints)"
print "  Orange: restrained SA only | Cyan: final (restrained + polish + fine refine)"
print "  White/multi: top5_diverse -- all 5 diverse final candidates shown at once"
print "  (all_states on: every candidate visible simultaneously, not as movie frames)"
print "================================================================="
