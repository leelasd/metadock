# PyMOL Visualization Script for openmm-dock Kinematics (PDB 9Z1L)
# Run directly in PyMOL: pymol visualize_kinematics_pymol.pml

reinitialize
load receptor.mol2, kit_receptor
load kinematics_joint_sweep.sdf, kit_kinematics

hide everything, kit_receptor
show cartoon, kit_receptor
color slate, kit_receptor
show surface, kit_receptor
set transparency, 0.65, kit_receptor

select pocket, kit_receptor within 6.0 of kit_kinematics
show sticks, pocket
color gray80, pocket

hide everything, kit_kinematics
show sticks, kit_kinematics
color cyan, kit_kinematics
set stick_radius, 0.22, kit_kinematics

zoom kit_kinematics, 8.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 240-frame OpenMM Forward Kinematics Movie!"
print "  Press Play (bottom right) or Spacebar to watch joint rotations."
print "================================================================="
