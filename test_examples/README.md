# `openmm-dock` Benchmark Examples & Demonstration Suites

This directory contains comprehensive, reproducible test examples and PyMOL demonstration suites covering all docking paradigms implemented in `openmm-dock`.

---

## Master Directory Index

```
┌────────────────────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Example Directory                      │ Target System / Paradigm                  │ Key Features & Algorithms                              │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. `covalent_docking/6di9/`            │ Human BTK Kinase + Acrylamide GJJ         │ Native HarmonicBondForce (k=2,000,000) on Cys-481      │
│                                        │ (PDB: 6DI9)                               │ Sub-angstrom RMSD (0.941 Å), Score: -236.6 kcal/mol    │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 2. `kinematics_workflow/`              │ Robotic Forward Kinematics Tree Engine    │ Parameterizes ligand in (SE(3) × T^k) space            │
│                                        │ (PDB: 6DI9)                               │ 336-frame joint sweep with 0.000000 Å bond distortion  │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 3. `kinematic_pso_demo/`               │ Kinematic Particle Swarm Optimization     │ 20-particle swarm on (SE(3) × T^14) manifold           │
│                                        │ (Kin-PSO) on BTK Kinase                   │ Escapes surface decoy traps into deep catalytic cleft  │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 4. `macrocycle_6z6a/`                  │ Human Keap1 Kelch + 16-Membered Macrocycle│ Damped Least Squares (DLS) Inverse Kinematics (IK)     │
│                                        │ (PDB: 6Z6A, Ligand: Q9E)                  │ Decoupled Two-Tier Kinematics (Ring IK vs Side-Chain FK│
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 5. `blind_global_docking_6z6a/`        │ Global Blind Docking from Scratch         │ 30 Walkers flying from bulk solvent (18.97 Å RMSD)     │
│                                        │ (PDB: 6Z6A Keap1 Macrocycle)              │ Converges into catalytic cleft via Generalized Beacons │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 6. `generalized_cv_docking_6z6a/`      │ Generalized Reference-Free CVs (ζ, Q, Rg) │ Universal 2D Free Energy Binding Funnel Reconstruction │
│                                        │ (PDB: 6Z6A Keap1 Macrocycle)              │ True blind predictive docking without crystal reference│
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 7. `macrocycle_swarm_metadynamics_6z6a/│ Swarm Metadynamics (MetaD-PSO) + FES      │ 15 Walkers x 20 Iterations (300 frames)                │
│                                        │ (PDB: 6Z6A Keap1 Macrocycle)              │ 2D Free Energy Surface & Per-Residue Energy Footprint  │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 8. `receptor_sidechain_kinematics/`    │ Receptor Pocket Side-Chain Kinematics     │ Parameterizes active-site residues by χ₁, χ₂, χ₃, χ₄   │
│                                        │ (PDB: 6DI9 BTK Kinase)                    │ 60-frame side-chain flexing with 0.000 Å backbone dev  │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 9. `pharmacophores/`                   │ Kinase Hinge Restraints                   │ Donor/Acceptor flat-bottom harmonic guiding funnels    │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 10.`solvent/`                          │ Solvated Explicit Waters                  │ Displaceable water scoring (neutral & charged states)  │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 11.`tethered/`                         │ Core Tethered MCS Docking                 │ Maximum Common Substructure core coordinate tethering  │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 12.`rna_docking_example/`              │ HIV-1 TAR RNA Binding                     │ Nucleic acid scoring & RNA pocket docking              │
└────────────────────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 3D PyMOL Demonstration Playback Quick Reference

| Benchmark Demo | PyMOL Command | Visualized Features |
| :--- | :--- | :--- |
| **Global Blind Docking** | `pymol test_examples/blind_global_docking_6z6a/visualize_blind_docking_pymol.pml` | 30-walker swarm flying from bulk solvent (18.97 Å RMSD) into Keap1 pocket |
| **Generalized CV Docking & Funnel** | `pymol test_examples/generalized_cv_docking_6z6a/visualize_generalized_cv_pymol.pml` | 300-frame swarm guided by reference-free CVs (ζ, Q, Rg) with 2D Binding Funnel |
| **Swarm Metadynamics (MetaD-PSO)** | `pymol test_examples/macrocycle_swarm_metadynamics_6z6a/visualize_swarm_metadynamics_pymol.pml` | 300-frame swarm movie with 2D Free Energy Surface & Footprint plots |
| **Macrocycle Two-Tier IK** | `pymol test_examples/macrocycle_6z6a/visualize_two_tier_pymol.pml` | Decoupled Ring Breathing (Magenta) vs. Side Chains (Yellow) in Keap1 |
| **Kinematic Particle Swarm** | `pymol test_examples/kinematic_pso_demo/visualize_pso_pymol.pml` | 400-frame 20-particle swarm collapsing into the catalytic cleft |
| **Robotic Kinematics Sweep** | `pymol test_examples/kinematics_workflow/visualize_pymol.pml` | 336-frame joint-by-joint robotic articulation inside BTK pocket |
| **Receptor Side-Chain Kinematics** | `pymol test_examples/receptor_sidechain_kinematics/visualize_sidechains_pymol.pml` | Active-site side chains flexing (χ₁–χ₄) around docked inhibitor GJJ |
