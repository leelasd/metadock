# Kinematic Particle Swarm Optimization (Kin-PSO) Benchmark

This benchmark demonstrates **Particle Swarm Optimization on the $(SE(3) \times \mathbb{T}^k)$ Kinematic Manifold** using **PDB 6DI9** (human Bruton's Tyrosine Kinase + covalent inhibitor GJJ).

---

## 1. Mathematical Formulation of Kin-PSO

In Kin-PSO, $N_{\text{particles}} = 20$ articulated molecular robots fly simultaneously through the internal coordinate space $\mathbf{x} = (\mathbf{t}, \mathbf{q}, \theta_1, \dots, \theta_k) \in \mathbb{R}^{6+k}$.

At each iteration $t$, every particle updates its velocity vector:

$$\mathbf{v}_i^{(t+1)} = w \cdot \mathbf{v}_i^{(t)} + c_1 r_1 \cdot \left( \mathbf{p}_{\text{best}, i} \ominus \mathbf{x}_i^{(t)} \right) + c_2 r_2 \cdot \left( \mathbf{g}_{\text{best}} \ominus \mathbf{x}_i^{(t)} \right)$$
$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} \oplus \mathbf{v}_i^{(t+1)}$$

* **Inertia ($w = 0.729$):** Maintains momentum along promising exploration vectors.
* **Cognitive Pull ($c_1 = 1.494$):** Attracts the particle back to its personal historical best score $\mathbf{p}_{\text{best}, i}$.
* **Social Swarm Pull ($c_2 = 1.494$):** Attracts all particles toward the global colony best $\mathbf{g}_{\text{best}}$.
* **Toroidal Difference ($\ominus$):** Calculates the shortest geodesic angle on the circle $\mathbb{T}^1 = [-\pi, \pi]$:
  $$\theta_a \ominus \theta_b = \text{atan2}\left( \sin(\theta_a - \theta_b), \; \cos(\theta_a - \theta_b) \right)$$

---

## 2. Why Kin-PSO Escapes Surface Decoy Traps

In elongated $41\text{ \AA}$ drug molecules like GJJ, single-trajectory Monte Carlo often becomes trapped in shallow solvent-exposed surface wells ($-77\text{ kcal/mol}$). 

In Kin-PSO:
1. 20 particles explore diverse pocket orientations simultaneously.
2. The moment **any single particle** enters the deep catalytic cleft ($-236\text{ kcal/mol}$), it becomes the new $\mathbf{g}_{\text{best}}$.
3. The social gravitational term $c_2 r_2 (\mathbf{g}_{\text{best}} \ominus \mathbf{x}_i)$ immediately pulls all other particles out of surface decoys and funnels them directly into the native binding cleft.

---

## 3. Files in This Directory

* [`receptor.pdb`](receptor.pdb): Cleaned BTK catalytic domain.
* [`cavity.prm`](cavity.prm): Active-site cavity definition centered at `(-12.16, 4.01, 0.43)`.
* [`xtal_ligand.sdf`](xtal_ligand.sdf): Reference X-ray crystal inhibitor GJJ with sanitized 3D hydrogens.
* [`run_pso_demo.py`](run_pso_demo.py): Automated runner for Kin-PSO.
* [`pso_swarm_trajectory.sdf`](pso_swarm_trajectory.sdf): 400-frame multi-particle evolution trajectory.
* [`pso_best_pose.sdf`](pso_best_pose.sdf): Final converged global minimum pose.
* [`visualize_pso_pymol.pml`](visualize_pso_pymol.pml): PyMOL script for live 3D swarm playback.

---

## 4. Running the Demo & PyMOL Visualization

```bash
cd test_examples/kinematic_pso_demo

# Run the Kin-PSO algorithm
python run_pso_demo.py

# Open 3D multi-particle movie in PyMOL
pymol visualize_pso_pymol.pml
```
