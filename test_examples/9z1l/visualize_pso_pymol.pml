# PyMOL Script for Kinematic Particle Swarm Optimization Movie (PDB 9Z1L)
# Run directly in PyMOL: pymol visualize_pso_pymol.pml

reinitialize
load receptor.mol2, kit_receptor
load pso_swarm_trajectory.sdf, pso_swarm
load pso_best_pose.sdf, best_docked_pose

hide everything, kit_receptor
show cartoon, kit_receptor
color slate, kit_receptor
show surface, kit_receptor
set transparency, 0.65, kit_receptor

select pocket, kit_receptor within 6.0 of best_docked_pose
show sticks, pocket
color gray80, pocket

hide everything, pso_swarm
show sticks, pso_swarm
color cyan, pso_swarm
set stick_radius, 0.18, pso_swarm

show sticks, best_docked_pose
color green, best_docked_pose
set stick_radius, 0.28, best_docked_pose

zoom best_docked_pose, 7.0
set movie_fps, 30
mplay

print "================================================================="
print "  Loaded 400-frame Kinematic Particle Swarm (Kin-PSO) Movie!"
print "  Cyan: 20 Swarm Particles evolving through iterations."
print "  Green: Final Converged Global Best Pose."
print "  Press Play (bottom right) or Spacebar to watch swarm collapse."
print "================================================================="
