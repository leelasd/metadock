# PyMOL Visualization Script for openmm-dock Kinematics
# Run directly in PyMOL: pymol visualize_pymol.pml

reinitialize
load receptor.pdb, receptor
load kinematics_joint_sweep.sdf, kinase_kinematics

# Style receptor
hide everything, receptor
show cartoon, receptor
color slate, receptor
show surface, receptor
set transparency, 0.65, receptor

# Style active site pocket residues
select pocket, receptor within 6.0 of kinase_kinematics
show sticks, pocket
color gray80, pocket
select cys481, (resn CYS and resi 481)
show sticks, cys481
color yellow, cys481

# Style Kinematic Ligand Trajectory
hide everything, kinase_kinematics
show sticks, kinase_kinematics
color cyan, kinase_kinematics
set stick_radius, 0.22, kinase_kinematics

# Display Covalent Bond to Cys481 SG
distance cov_bond, (cys481 and name SG), (kinase_kinematics and name C33), 2.5
color magenta, cov_bond
set dash_width, 3.0, cov_bond

# Setup Movie Camera and Play
zoom kinase_kinematics, 8.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 336-frame OpenMM Robotic Forward Kinematics Movie!"
print "  Press Play (bottom right) or Spacebar to watch joint rotations."
print "================================================================="
