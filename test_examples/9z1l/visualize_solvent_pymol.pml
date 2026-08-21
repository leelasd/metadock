# PyMOL Script for Explicit Water Docking Comparison (PDB 9Z1L)
reinitialize
load receptor.mol2, kit_receptor
load active_site_waters.pdb, active_waters
load a1czz_crystal_pose.sdf, reference_crystal_pose
load solvent_dock_dry_out.sdf, dry_pose
load solvent_dock_wet_out.sdf, wet_pose

hide everything, kit_receptor
show cartoon, kit_receptor
color wheat, kit_receptor

hide everything, active_waters
show nb_spheres, active_waters
color skyblue, active_waters

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose

hide everything, dry_pose
show sticks, dry_pose
color orange, dry_pose
set stick_radius, 0.18, dry_pose

hide everything, wet_pose
show sticks, wet_pose
color cyan, wet_pose
set stick_radius, 0.22, wet_pose

zoom reference_crystal_pose, 8.0

print "================================================================="
print "  Blue spheres: 11 crystallographic active-site waters"
print "  Green: crystal reference | Orange: dry-pocket dock | Cyan: wet-pocket dock"
print "================================================================="
