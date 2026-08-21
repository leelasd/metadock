# PyMOL Script for Pharmacophore-Restraint Docking Comparison (PDB 9Z1L)
reinitialize
load receptor.mol2, kit_receptor
load a1czz_crystal_pose.sdf, reference_crystal_pose
load pharma_dock_free_out.sdf, free_docked
load pharma_dock_restrained_out.sdf, restrained_docked

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
color cyan, restrained_docked
set stick_radius, 0.22, restrained_docked

pseudoatom pharma_pt1, pos=[13.59, -33.58, 18.62]

zoom reference_crystal_pose, 8.0

print "================================================================="
print "  Green: crystal reference | Red: free-docked (no restraints)"
print "  Cyan:  pharmacophore-restrained docking"
print "================================================================="
