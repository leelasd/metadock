# PyMOL Script for Grand Unified Kinematic PSO Demonstration
# Run directly in PyMOL: pymol visualize_unified_pso_pymol.pml

reinitialize
load unified_receptor_trajectory.pdb, keap1_swarm_receptor
load unified_macrocycle_swarm.sdf, q9e_swarm_ligand
load unified_best_receptor.pdb, best_receptor
load unified_best_ligand.sdf, best_ligand

# Style Receptor Swarm Track
hide everything, keap1_swarm_receptor
show cartoon, keap1_swarm_receptor
color wheat, keap1_swarm_receptor
show surface, keap1_swarm_receptor
set transparency, 0.70, keap1_swarm_receptor

# Style Active-Site Arginine Triad & Polar Network
select active_pocket, keap1_swarm_receptor and ((resi 334 and resn TYR) or (resi 335 and resn PHE) or (resi 338 and resn SER) or (resi 363 and resn SER) or (resi 380 and resn ARG) or (resi 382 and resn ASN) or (resi 414 and resn ASN) or (resi 415 and resn ARG) or (resi 483 and resn ARG) or (resi 530 and resn GLN) or (resi 555 and resn SER) or (resi 572 and resn TYR) or (resi 577 and resn PHE) or (resi 602 and resn SER))
show sticks, active_pocket
set stick_radius, 0.20, active_pocket

select arginines, keap1_swarm_receptor and (resn ARG and resi 415+483+380+336+601)
color marine, arginines
select tyrosines, keap1_swarm_receptor and (resn TYR and resi 334+572+525)
color tv_green, tyrosines
select serines, keap1_swarm_receptor and (resn SER and resi 602+555+363+338)
color orange, serines

# Style Ligand Swarm Track
hide everything, q9e_swarm_ligand
show sticks, q9e_swarm_ligand
color magenta, q9e_swarm_ligand
set stick_radius, 0.24, q9e_swarm_ligand

# Style Converged Best Complex (Hidden initially, show on pause)
hide everything, best_receptor
hide everything, best_ligand
show sticks, best_ligand
color cyan, best_ligand
set stick_radius, 0.28, best_ligand

# Dynamic Contact Distances
distance sb_arg415, (keap1_swarm_receptor and resi 415 and name NH1), (q9e_swarm_ligand and name O28), 3.5
distance sb_arg483, (keap1_swarm_receptor and resi 483 and name NH2), (q9e_swarm_ligand and name O19), 3.5
color yellow, sb_arg415
color yellow, sb_arg483
set dash_width, 3.0

zoom q9e_swarm_ligand, 6.5
set movie_fps, 24
mplay

print "================================================================="
print "  Loaded 225-Frame GRAND UNIFIED KINEMATIC PSO MOVIE!"
print "  • Wheat Cartoon : Keap1 Kelch β-Propeller Backbone"
print "  • Marine Sticks : Arginine Triad (Arg415, 483, 380) Flexing (χ₁-χ₄)"
print "  • Magenta Sticks: 16-Membered Macrocycle Swarm (Ring IK + Arm FK)"
print "  • Yellow Dashes : Dynamic Salt Bridges Formed in Real-Time"
print "  Press Play (bottom right) or Spacebar to watch the coupled swarm!"
print "================================================================="
