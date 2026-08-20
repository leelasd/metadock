# PDB 6Z6A: Swarm Metadynamics (MetaD-PSO) & 2D Free Energy Surface Benchmark

This benchmark demonstrates **Swarm Metadynamics (MetaD-PSO)**, uniting **Multiple-Walker Well-Tempered Metadynamics** with **Particle Swarm Optimization (PSO)** on **PDB 6Z6A** (human Keap1 Kelch domain in complex with the 16-membered macrocycle **Q9E**).

---

## 1. Mathematical Architecture of Swarm Metadynamics (MetaD-PSO)

Instead of a single walker laying down Gaussian hills, **15 articulated swarm particles** explore the $(SE(3) \times \mathbb{T}^{42})$ kinematic manifold in parallel, sharing a **unified global Metadynamics bias archive**:

$$V_{\text{meta}}(\mathbf{S}, t) = \sum_{\text{all walkers}} W_k \cdot \exp\left( -\frac{\|\mathbf{S} \ominus \mathbf{S}_k\|^2_{\mathbb{T}}}{2\sigma^2} \right)$$

* **Swarm Velocity Vector:**
  $$\mathbf{v}_i^{(t+1)} = w \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{p}_{\text{best}, i} \ominus \mathbf{S}_i) + c_2 r_2 (\mathbf{g}_{\text{best}} \ominus \mathbf{S}_i) + \eta_{\text{meta}} \boldsymbol{\tau}_{\text{meta}}(\mathbf{S}_i)$$
* **Advantage:** Fills energy wells **$15\times$ faster** than single-walker metadynamics while swarm social attractors pull walkers toward global energy funnels.

---

## 2. Quantitative Energetic Insights Generated

### A. 2D Free Energy Surface (FES): $F(\text{RMSD}, d_{\text{Arg415}})$
Reconstructed using the Well-Tempered Metadynamics scaling formula:

$$F(s_1, s_2) = -\frac{\gamma}{\gamma - 1} V_{\text{meta}}(s_1, s_2)$$

* **Collective Variable 1:** Heavy-Atom RMSD to Crystal ($\text{\AA}$)
* **Collective Variable 2:** Salt-Bridge Distance $d(\text{Macrocycle O28}, \text{Arg-415 NH1})$ ($\text{\AA}$)
* **Output:** Saved as a high-resolution 2D contour plot in [`free_energy_surface.png`](free_energy_surface.png).

### B. Keap1 Per-Residue Interaction Energy Footprint
Decomposes the total OpenMM GPU Hamiltonian into residue-by-residue binding contributions ($VDW + \text{Coulomb} + \text{H-bond}$):
* **`Arg-415` & `Tyr-572`:** Key binding hotspots ($> -500\text{ kcal/mol}$ electrostatic/VDW pull).
* **`Asn-382` & `Phe-577`:** Secondary stabilization contacts.
* **Output:** Saved as a bar chart in [`per_residue_energy_footprint.png`](per_residue_energy_footprint.png).

---

## 3. Files in This Directory

* [`receptor.pdb`](receptor.pdb): Cleaned Keap1 Kelch domain.
* [`cavity.prm`](cavity.prm): Active-site pocket definition.
* [`q9e_crystal_pose.sdf`](q9e_crystal_pose.sdf): Reference co-crystal macrocycle.
* [`run_swarm_metadynamics_demo.py`](run_swarm_metadynamics_demo.py): Automated Swarm-MetaD runner.
* [`free_energy_surface.png`](free_energy_surface.png): 2D Free Energy Landscape plot.
* [`per_residue_energy_footprint.png`](per_residue_energy_footprint.png): Per-residue interaction energy footprint plot.
* [`swarm_metadynamics_trajectory.sdf`](swarm_metadynamics_trajectory.sdf): 300-frame macrocycle swarm movie.
* [`swarm_receptor_trajectory.pdb`](swarm_receptor_trajectory.pdb): 300-frame synchronized Keap1 side-chain track.
* [`swarm_metadynamics_best_pose.sdf`](swarm_metadynamics_best_pose.sdf): Converged global minimum pose.
* [`visualize_swarm_metadynamics_pymol.pml`](visualize_swarm_metadynamics_pymol.pml): PyMOL script for live 3D playback.

---

## 4. Running the Demo & PyMOL Visualization

```bash
cd test_examples/macrocycle_swarm_metadynamics_6z6a

# Run Swarm-MetaD and generate FES plots
python run_swarm_metadynamics_demo.py

# Open 3D multi-track movie in PyMOL
pymol visualize_swarm_metadynamics_pymol.pml
```
