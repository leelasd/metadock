# PyMOL Script: Macrocycle CV Benchmark Comparison
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2

load q9e_crystal_pose.sdf, crystal_reference
show sticks, crystal_reference
color forest, crystal_reference
set stick_radius, 0.35, crystal_reference

load best_conformer_unbiased.sdf, conf_unbiased
show sticks, conf_unbiased
color gray60, conf_unbiased
set stick_radius, 0.22, conf_unbiased

load best_conformer_rg.sdf, conf_rg
show sticks, conf_rg
color tv_red, conf_rg
set stick_radius, 0.22, conf_rg

load best_conformer_kappa_sq.sdf, conf_pmi_kappa
show sticks, conf_pmi_kappa
color cyan, conf_pmi_kappa
set stick_radius, 0.22, conf_pmi_kappa

load best_conformer_qpuck.sdf, conf_qpuck
show sticks, conf_qpuck
color mediumseagreen, conf_qpuck
set stick_radius, 0.22, conf_qpuck

load best_conformer_coupled_rg_kappa.sdf, conf_coupled_rg_kappa
show sticks, conf_coupled_rg_kappa
color gold, conf_coupled_rg_kappa
set stick_radius, 0.28, conf_coupled_rg_kappa

zoom crystal_reference, 4.0
