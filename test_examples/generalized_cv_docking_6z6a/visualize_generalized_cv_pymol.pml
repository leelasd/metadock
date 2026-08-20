# PyMOL Script for Generalized Reference-Free CV Metadynamics Movie
# Run directly in PyMOL: pymol visualize_generalized_cv_pymol.pml

reinitialize
load generalized_cv_receptor_trajectory.pdb, keap1_gen_receptor
load generalized_cv_trajectory.sdf, q9e_gen_macrocycle
load generalized_cv_best_receptor.pdb, best_receptor
load generalized_cv_best_pose.sdf, best_macrocycle

# Style Receptor Backbone
hide everything, keap1_gen_receptor
show cartoon, keap1_gen_receptor
color wheat, keap1_gen_receptor
show surface, keap1_gen_receptor
set transparency, 0.70, keap1_gen_receptor

# Style Active Pocket Residues
select active_pocket, keap1_gen_receptor and ((resi 334 and resn TYR) or (resi 335 and resn PHE) or (resi 338 and resn SER) or (resi 363 and resn SER) or (resi 380 and resn ARG) or (resi 382 and resn ASN) or (resi 414 and resn ASN) or (resi 415 and resn ARG) or (resi 483 and resn ARG) or (resi 530 and resn GLN) or (resi 555 and resn SER) or (resi 572 and resn TYR) or (resi 577 and resn PHE) or (resi 602 and resn SER))
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

select arginines, keap1_gen_receptor and (resn ARG and resi 415+483+380+336+601)
color marine, arginines
select tyrosines, keap1_gen_receptor and (resn TYR and resi 334+572+525)
color tv_green, tyrosines
select serines, keap1_gen_receptor and (resn SER and resi 602+555+363+338)
color orange, serines

# Style Macrocycle Trajectory
hide everything, q9e_gen_macrocycle
show sticks, q9e_gen_macrocycle
color magenta, q9e_gen_macrocycle
set stick_radius, 0.24, q9e_gen_macrocycle

# Style Best Complex (Shown on pause)
hide everything, best_receptor
hide everything, best_macrocycle
show sticks, best_macrocycle
color cyan, best_macrocycle
set stick_radius, 0.28, best_macrocycle

# Dynamic Distance Contacts to Arginines
distance sb_arg415, (keap1_gen_receptor and resi 415 and name NH1), (q9e_gen_macrocycle and name O28), 3.5
distance sb_arg483, (keap1_gen_receptor and resi 483 and name NH2), (q9e_gen_macrocycle and name O19), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
set dash_width, 3.0

zoom q9e_gen_macrocycle, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 300-Frame GENERALIZED CV METADYNAMICS MOVIE!"
print "  • Reference-Free CVs: Pocket Depth (ζ_depth) & Coordination (Q)"
print "  • Generated universal_binding_funnel_fes.png 2D FES"
print "  • Press Play (bottom right) or Spacebar to watch generalized swarm!"
print "================================================================="
