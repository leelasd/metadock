# PDB 6Z6A: Kinematic Metadynamics (Kin-MetaD) on Keap1 + Macrocycle

This benchmark demonstrates **Kinematic Metadynamics (Kin-MetaD)** on **PDB 6Z6A** (human Keap1 Kelch domain in complex with the 16-membered macrocycle **Q9E**).

---

## 1. How Kinematic Metadynamics Works

In standard docking, an optimizer often gets stuck in a local energy minimum (decoy well). Kin-MetaD adds a **history-dependent repulsive Gaussian potential** directly onto the **kinematic manifold** $(\mathbf{t}, \mathbf{q}, \boldsymbol{\theta}_{\text{ring\_ik}}, \boldsymbol{\theta}_{\text{exo\_fk}}, \boldsymbol{\chi}_{\text{rec}})$:

$$V_{\text{meta}}(\mathbf{S}) = \sum_{k=1}^{N_{\text{visited}}} W \cdot \exp\left( -\frac{\|\mathbf{S} \ominus \mathbf{S}_k\|^2_{\mathbb{T}}}{2\sigma^2} \right)$$

* **Hill Height ($W$):** $+25.0\text{ kcal/mol}$ deposited every 3 steps.
* **Gaussian Width ($\sigma$):** $0.45\text{ rad}$ on the Toroidal Manifold $\mathbb{T}^k$.
* **The Effect:** As the simulation revisits the same pocket conformation, the accumulated Gaussian hills progressively raise the effective energy ($+50 \to +150 \to +300\text{ kcal/mol}$), **physically pushing the macrocycle out of local traps and forcing it into new pocket sub-states**.

---

## 2. Files in This Directory

* [`receptor.pdb`](receptor.pdb): Cleaned Keap1 Kelch domain.
* [`cavity.prm`](cavity.prm): Active-site pocket definition.
* [`q9e_crystal_pose.sdf`](q9e_crystal_pose.sdf): Reference co-crystal macrocycle.
* [`run_metadynamics_macrocycle_demo.py`](run_metadynamics_macrocycle_demo.py): Automated Kin-MetaD runner.
* [`metadynamics_macrocycle_trajectory.sdf`](metadynamics_macrocycle_trajectory.sdf): 50-frame macrocycle escape trajectory.
* [`metadynamics_receptor_trajectory.pdb`](metadynamics_receptor_trajectory.pdb): 50-frame synchronized Keap1 side-chain track.
* [`metadynamics_best_pose.sdf`](metadynamics_best_pose.sdf): Best discovered macrocycle conformation.
* [`visualize_metadynamics_pymol.pml`](visualize_metadynamics_pymol.pml): PyMOL script for live 3D playback.

---

## 3. Running the Demo & PyMOL Visualization

```bash
cd test_examples/macrocycle_metadynamics_6z6a

# Run Kin-MetaD
python run_metadynamics_macrocycle_demo.py

# Open 3D multi-track movie in PyMOL
pymol visualize_metadynamics_pymol.pml
```

#### What You Will See in PyMOL:
* **Wheat Cartoon:** Keap1 Kelch $\beta$-propeller fold (100% rigid backbone).
* **Marine Sticks:** Arginine Triad (`Arg-415`, `Arg-483`, `Arg-380`) dynamically articulating.
* **Magenta Sticks:** The 16-membered macrocycle progressively escaping filled decoy wells and exploring diverse pocket geometries.
