# PDB 6Z6A: Generalized Reference-Free Collective Variables (CVs) & Universal Binding Funnel

This benchmark demonstrates **Generalized Reference-Free Collective Variable Metadynamics** on **PDB 6Z6A** (human Keap1 Kelch domain in complex with the 16-membered macrocycle **Q9E**).

---

## 1. Why Reference-Free Collective Variables Matter

In predictive, blind drug discovery, **the co-crystal pose is unknown**. Using *RMSD to crystal* as a simulation coordinate is impossible.

We implement **3 Universal Biophysical Collective Variables**:

### A. Pocket Penetration Depth ($\zeta_{\text{depth}}$)
Measures distance from the ligand center-of-mass to the active-site cavity center:
$$\zeta_{\text{depth}} = \|\mathbf{r}_{\text{lig, COM}} - \mathbf{c}_{\text{pocket}}\|$$

### B. Continuous Contact Coordination ($Q_{\text{contacts}}$)
Differentiable nonbonded packing metric quantifying surface complementarity:
$$Q_{\text{contacts}} = \sum_{i \in \text{Ligand}} \sum_{j \in \text{Pocket}} \frac{1 - \left(\frac{d_{ij}}{d_0}\right)^6}{1 - \left(\frac{d_{ij}}{d_0}\right)^{12}} \quad (d_0 = 4.5\text{ \AA})$$

### C. Macrocycle Radius of Gyration ($R_g$)
Measures conformational envelope and ring pucker:
$$R_g = \sqrt{\frac{1}{N_{\text{ring}}} \sum_{i \in \text{ring}} \|\mathbf{r}_i - \mathbf{r}_{\text{ring, COM}}\|^2}$$

---

## 2. Universal 2D Free Energy Binding Funnel $F(\zeta_{\text{depth}}, Q_{\text{contacts}})$

Reconstructed directly from the 150 deposited Gaussian hills:

* **Bulk Solvent:** $\zeta_{\text{depth}} > 6.0\text{ \AA}$, $Q_{\text{contacts}} \approx 0$
* **Surface Decoy Traps:** $\zeta_{\text{depth}} \approx 3.0\text{--}4.5\text{ \AA}$, $Q_{\text{contacts}} \approx 150\text{--}250$
* **Deep Catalytic Cleft:** $\mathbf{\zeta_{\text{depth}} < 1.5\text{ \AA}}$, $\mathbf{Q_{\text{contacts}} > 500}$, Score = **$-113.44\text{ kcal/mol}$**
* **Plot Output:** Saved to [`universal_binding_funnel_fes.png`](universal_binding_funnel_fes.png).

---

## 3. Files in This Directory

* [`receptor.pdb`](receptor.pdb): Cleaned Keap1 Kelch domain.
* [`cavity.prm`](cavity.prm): Active-site pocket definition.
* [`q9e_crystal_pose.sdf`](q9e_crystal_pose.sdf): Reference macrocycle.
* [`run_generalized_cv_demo.py`](run_generalized_cv_demo.py): Automated Generalized CV Metadynamics runner.
* [`universal_binding_funnel_fes.png`](universal_binding_funnel_fes.png): Reconstructed 2D Binding Funnel plot.
* [`generalized_cv_trajectory.sdf`](generalized_cv_trajectory.sdf): 300-frame macrocycle swarm movie.
* [`generalized_cv_receptor_trajectory.pdb`](generalized_cv_receptor_trajectory.pdb): 300-frame synchronized Keap1 side-chain track.
* [`generalized_cv_best_pose.sdf`](generalized_cv_best_pose.sdf): Best converged catalytic complex.
* [`visualize_generalized_cv_pymol.pml`](visualize_generalized_cv_pymol.pml): PyMOL script for live 3D playback.

---

## 4. Running the Demo & PyMOL Visualization

```bash
cd test_examples/generalized_cv_docking_6z6a

# Run Generalized CV Swarm Metadynamics & FES reconstruction
python run_generalized_cv_demo.py

# Open 3D multi-track movie in PyMOL
pymol visualize_generalized_cv_pymol.pml
```
