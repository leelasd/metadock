# PyMOL Script for Well-Tempered Kinematic Metadynamics Pose-Strength Assay (PDB 9Z1L)
# Run directly in PyMOL: pymol visualize_metadynamics_pymol.pml

reinitialize
load metadynamics_receptor_trajectory.pdb, kit_metad_receptor
load metadynamics_trajectory.sdf, a1czz_metad_trajectory
load metadynamics_best_receptor.pdb, best_receptor
load metadynamics_best_pose.sdf, best_ligand
load a1czz_crystal_pose.sdf, reference_crystal_pose

hide everything, kit_metad_receptor
show cartoon, kit_metad_receptor
color wheat, kit_metad_receptor
show surface, kit_metad_receptor
set transparency, 0.70, kit_metad_receptor

select active_pocket, kit_metad_receptor and ((resi 39 and resn LEU) or (resi 47 and resn VAL) or (resi 48 and resn VAL) or (resi 49 and resn GLU) or (resi 66 and resn VAL) or (resi 114 and resn THR) or (resi 115 and resn GLU) or (resi 116 and resn TYR) or (resi 117 and resn CYS) or (resi 118 and resn CYS) or (resi 119 and resn TYR) or (resi 121 and resn ASP) or (resi 177 and resn LEU) or (resi 178 and resn LEU) or (resi 187 and resn CYS) or (resi 188 and resn ASP) or (resi 189 and resn PHE))
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

hide everything, reference_crystal_pose
show sticks, reference_crystal_pose
color green, reference_crystal_pose
set stick_radius, 0.24, reference_crystal_pose

hide everything, a1czz_metad_trajectory
show sticks, a1czz_metad_trajectory
color magenta, a1czz_metad_trajectory
set stick_radius, 0.24, a1czz_metad_trajectory

hide everything, best_receptor
hide everything, best_ligand
show sticks, best_ligand
color cyan, best_ligand
set stick_radius, 0.28, best_ligand

zoom reference_crystal_pose, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded WELL-TEMPERED METADYNAMICS POSE-STRENGTH MOVIE!"
print "  • Green: native crystal pose (starting point, hills deposited here)"
print "  • Magenta: trajectory being actively repelled out of the native basin"
print "  Press Play (bottom right) or Spacebar to watch it escape (or resist escaping)!"
print "================================================================="
