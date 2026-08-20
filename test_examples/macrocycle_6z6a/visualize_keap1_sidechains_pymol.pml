# PyMOL Script for Keap1 Receptor Side-Chain Kinematics Movie
# Run directly in PyMOL: pymol visualize_keap1_sidechains_pymol.pml

reinitialize
load keap1_sidechains_movie.pdb, keap1_receptor_kinematics
load q9e_crystal_pose.sdf, q9e_macrocycle

# Style Receptor Backbone
hide everything, keap1_receptor_kinematics
show cartoon, keap1_receptor_kinematics
color wheat, keap1_receptor_kinematics
show surface, keap1_receptor_kinematics
set transparency, 0.70, keap1_receptor_kinematics

# Select and Style Flexible Active-Site Side Chains
select flex_sidechains, keap1_receptor_kinematics and ((resi 334 and resn TYR) or (resi 335 and resn PHE) or (resi 336 and resn ARG) or (resi 337 and resn GLN) or (resi 338 and resn SER) or (resi 363 and resn SER) or (resi 380 and resn ARG) or (resi 382 and resn ASN) or (resi 414 and resn ASN) or (resi 415 and resn ARG) or (resi 461 and resn ILE) or (resi 483 and resn ARG) or (resi 525 and resn TYR) or (resi 530 and resn GLN) or (resi 555 and resn SER) or (resi 557 and resn LEU) or (resi 572 and resn TYR) or (resi 573 and resn ASP) or (resi 576 and resn THR) or (resi 577 and resn PHE) or (resi 601 and resn ARG) or (resi 602 and resn SER) or (resi 604 and resn VAL))
show sticks, flex_sidechains
set stick_radius, 0.22, flex_sidechains

# Color Key Pocket Residues
select arginine_triad, keap1_receptor_kinematics and (resn ARG and resi 415+483+380+336+601)
color marine, arginine_triad
select aromatic_tyrs, keap1_receptor_kinematics and (resn TYR and resi 334+572+525)
color tv_green, aromatic_tyrs
select polar_serines, keap1_receptor_kinematics and (resn SER and resi 602+555+363+338)
color orange, polar_serines
select other_flex, flex_sidechains and not (resn ARG or resn TYR or resn SER)
color salmon, other_flex

# Style Docked Macrocycle
hide everything, q9e_macrocycle
show sticks, q9e_macrocycle
color magenta, q9e_macrocycle
set stick_radius, 0.26, q9e_macrocycle

# Salt Bridge & Hydrogen Bond Distances to Arginines
distance sb_arg415, (keap1_receptor_kinematics and resi 415 and name NH1), (q9e_macrocycle and name O28), 3.5
distance sb_arg483, (keap1_receptor_kinematics and resi 483 and name NH2), (q9e_macrocycle and name O19), 3.5
distance hb_tyr334, (keap1_receptor_kinematics and resi 334 and name OH),  (q9e_macrocycle and name N18), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
color yellow, hb_tyr334
set dash_width, 3.0

zoom q9e_macrocycle, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 60-frame Keap1 Receptor Side-Chain Kinematics Movie!"
print "  • Wheat Cartoon : Keap1 Kelch β-Propeller Backbone (100% Rigid)"
print "  • Marine Sticks : Arginine Triad (Arg415, Arg483, Arg380, 4 Chi)"
print "  • Green Sticks  : Aromatic Gates (Tyr334, Tyr572)"
print "  • Orange Sticks : Polar Network (Ser602, Ser555)"
print "  • Magenta Sticks: 16-Membered Macrocycle Q9E"
print "  Press Play (bottom right) or Spacebar to watch side chains flex!"
print "================================================================="
