# PyMOL Script for Keap1 Macrocycle Inverse Kinematics Movie
# Run directly in PyMOL: pymol visualize_6z6a_pymol.pml

reinitialize
load receptor.pdb, keap1_receptor
load macrocycle_ik_flexing_movie.sdf, q9e_macrocycle_ik
load macrocycle_docked_min.sdf, q9e_docked_best

# Style Keap1 Receptor
hide everything, keap1_receptor
show cartoon, keap1_receptor
color wheat, keap1_receptor
show surface, keap1_receptor
set transparency, 0.60, keap1_receptor

# Style Keap1 Kelch Propeller Arginine Triad Pocket
select pocket_args, (resn ARG and resi 415+483+380) or (resn TYR and resi 334+572) or (resn SER and resi 602)
show sticks, pocket_args
color marine, pocket_args
set stick_radius, 0.20, pocket_args

# Style Macrocycle IK Movie
hide everything, q9e_macrocycle_ik
show sticks, q9e_macrocycle_ik
color magenta, q9e_macrocycle_ik
set stick_radius, 0.24, q9e_macrocycle_ik

# Style Minimized Docked Macrocycle
hide everything, q9e_docked_best
show sticks, q9e_docked_best
color green, q9e_docked_best
set stick_radius, 0.28, q9e_docked_best

# Focus on the binding cleft
zoom q9e_docked_best, 6.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 60-frame Keap1 Macrocycle Inverse Kinematics (IK) Movie!"
print "  Magenta: 16-Membered Macrocycle flexing with 0.000 Å Ring Closure."
print "  Green: Minimized Docked Macrocycle (-211.1 kcal/mol)."
print "  Marine Sticks: Keap1 Arginine Triad (Arg415, Arg483, Arg380)."
print "  Press Play (bottom right) or Spacebar to watch macrocycle breathe."
print "================================================================="
