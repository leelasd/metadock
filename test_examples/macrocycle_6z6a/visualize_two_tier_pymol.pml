# PyMOL Script for Two-Tier Decoupled Macrocycle Kinematics
# Run directly in PyMOL: pymol visualize_two_tier_pymol.pml

reinitialize
load receptor.pdb, keap1_receptor
load macrocycle_two_tier_trajectory.sdf, q9e_two_tier

# Style Keap1 Receptor
hide everything, keap1_receptor
show cartoon, keap1_receptor
color wheat, keap1_receptor
show surface, keap1_receptor
set transparency, 0.65, keap1_receptor

# Style Keap1 Kelch Arginine Triad Pocket
select pocket_args, (resn ARG and resi 415+483+380) or (resn TYR and resi 334+572) or (resn SER and resi 602)
show sticks, pocket_args
color marine, pocket_args
set stick_radius, 0.20, pocket_args

# Style Macrocycle Trajectory: Decoupled Ring vs. Side Chains
hide everything, q9e_two_tier
show sticks, q9e_two_tier
set stick_radius, 0.24, q9e_two_tier

# Color Macrocyclic Ring Backbone MAGENTA
select ring_backbone, q9e_two_tier and (id 32+33+3+4+8+12+14+21+22+23+24+26)
color magenta, ring_backbone

# Color Exocyclic Functional Arms YELLOW
select exocyclic_arms, q9e_two_tier and not (id 32+33+3+4+8+12+14+21+22+23+24+26) and not (elem H)
color yellow, exocyclic_arms

# Style Hydrogens White
select hydrogens, q9e_two_tier and (elem H)
color white, hydrogens
set stick_radius, 0.12, hydrogens

zoom q9e_two_tier, 6.0
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 120-frame Decoupled Two-Tier Macrocycle Movie!"
print "  • MAGENTA Sticks: 16-Membered Macrocyclic Ring (Tier 1: IK)"
print "  • YELLOW Sticks : Exocyclic Side-Chain Arms (Tier 2: FK)"
print "  • MARINE Sticks : Keap1 Arginine Pocket Triad (Arg415, 483, 380)"
print "  Press Play (bottom right) or Spacebar to watch the 3 phases:"
print "    - Phase 1 (Frames 1-40) : Ring Breathing (IK Closed)"
print "    - Phase 2 (Frames 41-80): Side-Chain Articulation (FK)"
print "    - Phase 3 (Frames 81-120): Coupled Pocket Docking"
print "================================================================="
