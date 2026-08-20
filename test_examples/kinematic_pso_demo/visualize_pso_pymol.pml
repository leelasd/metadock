# PyMOL Script for Kinematic Particle Swarm Optimization Movie
# Run directly in PyMOL: pymol visualize_pso_pymol.pml

reinitialize
load receptor.pdb, receptor
load pso_swarm_trajectory.sdf, pso_swarm
load pso_best_pose.sdf, best_docked_pose

hide everything, receptor
show cartoon, receptor
color slate, receptor
show surface, receptor
set transparency, 0.65, receptor

# Pocket residue highlight
select pocket, receptor within 6.0 of best_docked_pose
show sticks, pocket
color gray80, pocket
select cys481, (resn CYS and resi 481)
show sticks, cys481
color yellow, cys481

# Style Swarm Trajectory
hide everything, pso_swarm
show sticks, pso_swarm
color cyan, pso_swarm
set stick_radius, 0.18, pso_swarm

# Style Best Pose
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
