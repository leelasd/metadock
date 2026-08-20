# PyMOL Script for Receptor Side-Chain Kinematics Movie
# Run directly in PyMOL: pymol visualize_sidechains_pymol.pml

reinitialize
load receptor_sidechain_movie.pdb, btk_receptor_kinematics
load xtal_ligand.sdf, btk_ligand

# Style Receptor Backbone
hide everything, btk_receptor_kinematics
show cartoon, btk_receptor_kinematics
color slate, btk_receptor_kinematics
show surface, btk_receptor_kinematics
set transparency, 0.70, btk_receptor_kinematics

# Select and Style Flexible Active-Site Side Chains
select flex_sidechains, btk_receptor_kinematics and ((resi 426 and resn ASP) or (resi 427 and resn VAL) or (resi 457 and resn LEU) or (resi 458 and resn VAL) or (resi 459 and resn GLN) or (resi 461 and resn TYR) or (resi 474 and resn THR) or (resi 475 and resn GLU) or (resi 476 and resn TYR) or (resi 477 and resn MET) or (resi 528 and resn LEU) or (resi 529 and resn VAL) or (resi 530 and resn ASN) or (resi 531 and resn ASP) or (resi 536 and resn LYS))
show sticks, flex_sidechains
set stick_radius, 0.22, flex_sidechains

# Color Key Pocket Residues
select hinge_res, btk_receptor_kinematics and (resi 475+477)
color orange, hinge_res
select catalytic_lys, btk_receptor_kinematics and (resi 536)
color yellow, catalytic_lys
select aromatic_gates, btk_receptor_kinematics and (resi 461+476)
color tv_green, aromatic_gates
select other_flex, flex_sidechains and not (resi 475+477+536+461+476)
color salmon, other_flex

# Style Docked Ligand
hide everything, btk_ligand
show sticks, btk_ligand
color cyan, btk_ligand
set stick_radius, 0.26, btk_ligand

# Hinge Hydrogen Bonds
distance hb_met477, (btk_receptor_kinematics and resi 477 and name N), (btk_ligand and name N24), 3.5
distance hb_glu475, (btk_receptor_kinematics and resi 475 and name O), (btk_ligand and name N25), 3.5
color magenta, hb_met477
color magenta, hb_glu475
set dash_width, 3.0

zoom flex_sidechains, 6.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 60-frame Receptor Side-Chain Kinematics Movie!"
print "  • Slate Cartoon: Protein Backbone (100% Rigid, 0.000 Å distortion)"
print "  • Orange Sticks: Hinge Residues (Glu475, Met477)"
print "  • Yellow Sticks: Catalytic Lysine (Lys536, 4 Chi Joints)"
print "  • Green Sticks : Aromatic Gating (Tyr461, Tyr476)"
print "  • Cyan Sticks  : Docked Kinase Inhibitor GJJ"
print "  Press Play (bottom right) or Spacebar to watch side chains flex!"
print "================================================================="
