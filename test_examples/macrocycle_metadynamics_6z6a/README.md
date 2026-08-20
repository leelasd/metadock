# PDB 6Z6A: Well-Tempered Kinematic Metadynamics (WT-Kin-MetaD) on Keap1 Macrocycle

This benchmark demonstrates **Well-Tempered Kinematic Metadynamics (WT-Kin-MetaD)** with **Physical Steric Force Balance** on **PDB 6Z6A** (human Keap1 Kelch domain in complex with the 16-membered macrocycle **Q9E**).

---

## 1. How We Prevent Protein Clashes (Well-Tempered Metadynamics)

Standard Metadynamics with fixed-height hills can overfill wells and push the ligand into protein walls ($> 100{,}000\text{ kcal/mol}$ steric clashes).

WT-Kin-MetaD prevents this using two biophysical principles:

### A. Well-Tempered Adaptive Gaussian Heights $W(t)$
Instead of depositing rigid $+25\text{ kcal/mol}$ hills, hill heights decay exponentially as the local free energy fills:

$$W(t) = W_0 \cdot \exp\left( -\frac{V_{\text{meta}}(\mathbf{S})}{k_B \Delta T} \right)$$

* **Initial Hill Height ($W_0$):** $+8.0\text{ kcal/mol}$
* **Bias Factor ($\gamma = 5.0$):** $k_B \Delta T = 2.38\text{ kcal/mol}$
* **Result:** The bias potential smoothly plateaus without ever over-pushing into unphysical steric walls.

### B. Physical Steric Force Balance
The step direction balances the Metadynamics repulsive push with the **OpenMM Physical Restoring Gradient**:

$$\Delta \mathbf{S} = \eta_{\text{meta}} \boldsymbol{\tau}_{\text{meta}}(\mathbf{S}) - \eta_{\text{phys}} \nabla_{\mathbf{S}} \mathcal{V}_{\text{OpenMM}}(\mathbf{S}) + \boldsymbol{\xi}_{\text{thermal}}$$

* If the macrocycle approaches a protein atom ($r < 2.5\text{ \AA}$), OpenMM's Lennard-Jones repulsion creates a restoring force that **steers the ligand around the obstacle and along the pocket channel**, eliminating atomic interpenetration.

---

## 2. Files in This Directory

* [`receptor.pdb`](receptor.pdb): Cleaned Keap1 Kelch domain.
* [`cavity.prm`](cavity.prm): Active-site pocket definition.
* [`q9e_crystal_pose.sdf`](q9e_crystal_pose.sdf): Reference co-crystal macrocycle.
* [`run_metadynamics_macrocycle_demo.py`](run_metadynamics_macrocycle_demo.py): Automated WT-Kin-MetaD runner.
* [`metadynamics_macrocycle_trajectory.sdf`](metadynamics_macrocycle_trajectory.sdf): 100-frame clash-free macrocycle escape trajectory.
* [`metadynamics_receptor_trajectory.pdb`](metadynamics_receptor_trajectory.pdb): 100-frame synchronized Keap1 side-chain track.
* [`metadynamics_best_pose.sdf`](metadynamics_best_pose.sdf): Best discovered macrocycle conformation.
* [`visualize_metadynamics_pymol.pml`](visualize_metadynamics_pymol.pml): PyMOL script for live 3D playback.

---

## 3. Running the Demo & PyMOL Visualization

```bash
cd test_examples/macrocycle_metadynamics_6z6a

# Run WT-Kin-MetaD
python run_metadynamics_macrocycle_demo.py

# Open 3D multi-track movie in PyMOL
pymol visualize_metadynamics_pymol.pml
```
