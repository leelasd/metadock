# Architectural Insights & Suggested Improvements for `openmm-dock`

This document summarizes the core architectural insights, mathematical formulations, and future roadmap enhancements derived from comparative analysis with **OpenDock**, **Metadynamics**, and real-world complex benchmarks (such as **PDB 6DI9**).

---

## 1. Executive Summary of Proposed Enhancements

```
┌────────────────────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Enhancement                            │ Primary Benefit                           │ Implementation Strategy                                │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. Particle Swarm Optimization (PSO)   │ Prevents large elongated ligands from     │ Multi-particle swarm in (SE(3) × T^N) with cognitive   │
│                                        │ getting trapped in surface decoy wells    │ (p_best) and social (g_best) velocity updates          │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 2. Metadynamics / Tabu Repulsive Bias  │ Actively pushes single-trajectory MC out  │ History-dependent Gaussian penalty on visited          │
│                                        │ of local minima into unexplored clefts    │ local coordinate basins (RMSD Gaussian kernels)        │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 3. Forward Kinematics Torsion Tree     │ Eliminates internal spring distortion and │ Parameterizes ligand strictly in (t, q, θ) space       │
│                                        │ reduces search to exactly 6 + N_torsions  │ with analytical SE(3) kinematic matrix transforms      │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 4. Three-Stage Chained Pipeline        │ Combines broad global search with         │ Global Swarm/GA ──► Annealing MC ──► GPU L-BFGS       │
│                                        │ sub-angstrom gradient refinement          │ with seamless pose hand-off                            │
├────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 5. Differentiable Contact Constraints  │ Converts rugged multiminima landscapes    │ Pairwise distance matrix soft-wall potentials          │
│                                        │ into smooth directional funnels           │ on key pocket anchor residues (e.g. kinase hinge)      │
└────────────────────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Deep Dive into Suggested Enhancements

### Enhancement 1: Particle Swarm Optimization (PSO) for Large/Covalent Systems

#### The Challenge
In large, extended drug molecules (such as the $41\text{ \AA}$ covalent BTK inhibitor GJJ in PDB 6DI9), single-trajectory Monte Carlo or Simulated Annealing often settles on outer solvent-exposed protein surfaces (the $-77\text{ kcal/mol}$ decoy trap) because the narrow entrance into the catalytic cleft is flanked by steep steric clash walls ($> +500\text{ kcal/mol}$).

#### The Mathematical Solution
Maintain a swarm of $N_{\text{particles}} = 30\text{--}50$ individuals in the search space $\mathbf{x}_i \in \mathbb{R}^{6 + N_{\text{torsions}}}$. At each step $t$:

$$\mathbf{v}_i^{(t+1)} = w \mathbf{v}_i^{(t)} + c_1 r_1 \left(\mathbf{p}_{\text{best}, i} - \mathbf{x}_i^{(t)}\right) + c_2 r_2 \left(\mathbf{g}_{\text{best}} - \mathbf{x}_i^{(t)}\right)$$
$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$

* $w$: Inertia weight ($\approx 0.7$)
* $c_1, c_2$: Cognitive and social acceleration coefficients ($\approx 1.5$)
* $r_1, r_2 \sim \mathcal{U}(0, 1)$: Random uniform sampling

**Why this works:** If **even a single particle** in the swarm discovers the entrance to the deep catalytic hinge cleft ($-236\text{ kcal/mol}$), the global social attractor $\mathbf{g}_{\text{best}}$ exerts a strong gravitational pull that steers the entire swarm out of surface traps and into the active site.

---

### Enhancement 2: History-Dependent Metadynamics / Tabu Scoring

#### The Mathematical Solution
During Monte Carlo Basin-Hopping, store each converged local minimum $\mathbf{x}_k$ in a `visited_basins` archive. Modify the effective docking score evaluated by the Metropolis acceptance criterion:

$$\mathcal{V}_{\text{effective}}(\mathbf{x}) = \mathcal{V}_{\text{OpenMM}}(\mathbf{x}) + \sum_{k=1}^{N_{\text{visited}}} W \cdot \exp\left( -\frac{\text{RMSD}(\mathbf{x}, \mathbf{x}_k)^2}{2\sigma^2} \right)$$

* **Gaussian Height ($W$):** $+15\text{ to }+50\text{ kcal/mol}$
* **Gaussian Width ($\sigma$):** $1.5\text{ to }2.0\text{ \AA}$

**Why this works:** As the simulation revisits the $-77\text{ kcal/mol}$ decoy well, the accumulated repulsive Gaussian hill raises the local free energy of that trap above zero, forcing subsequent Monte Carlo perturbations to explore new pocket volumes.

---

### Enhancement 3: Forward Kinematics Torsion Tree Representation

#### Current vs. Proposed Representation
* **Current OpenMM-Dock:** Internal bond lengths ($C\text{--}C, C\text{--}H$), valence angles, and rings are kept rigid by stiff harmonic spring potentials ($k_{\text{bond}} = 500{,}000\text{ kJ/(mol nm}^2)$).
* **Proposed Kinematics:** Construct a directed acyclic kinematic tree where every rotatable bond $j$ defines a relative transformation matrix $\mathbf{T}_j(\theta_j) \in SE(3)$:

$$\mathbf{T}_j(\theta_j) = \begin{bmatrix} \mathbf{R}(\hat{\mathbf{u}}_j, \theta_j) & \mathbf{d}_j \\ \mathbf{0} & 1 \end{bmatrix}$$

$$\mathbf{r}_i(\mathbf{t}, \mathbf{q}, \boldsymbol{\theta}) = \mathbf{T}_{\text{root}}(\mathbf{t}, \mathbf{q}) \prod_{j \in \text{path}(i)} \mathbf{T}_j(\theta_j) \, \mathbf{r}_{i, \text{local}}$$

**Key Advantage:** Guarantees **$0.000\text{ \AA}$ bond length or angle distortion** by mathematical definition, reducing the parameter search space to strictly $6 + N_{\text{torsions}}$ dimensions.

---

### Enhancement 4: Three-Stage Chained Sampling Pipeline

To maximize both search exhaustiveness and sub-angstrom accuracy:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                THREE-STAGE CHAINED SAMPLING PIPELINE                                   │
├────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────┤
│ Stage 1: Global Swarm / GA     │ Stage 2: Thermal Annealing MC         │ Stage 3: GPU L-BFGS Gradient  │
│                                │                                       │          Minimization         │
│ • Explores global rotational & │ • Relaxes flexible ligand side chains │ • Converges directly into the │
│   torsional space around the   │   and receptor pocket residues        │   deepest stationary point    │
│   covalent/pocket anchor       │   (1000K → 300K Langevin cooling)     │   (||∇V|| < 0.01 kcal/mol/Å)  │
│ • Output: Cleft-localized seed │ • Output: Sub-native pose (< 2.5 Å)   │ • Output: Final pose (< 1.0 Å)│
└────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────┘
```

---

### Enhancement 5: Differentiable Contact Matrix Guidance

For protein targets with known key interaction motifs (e.g. kinase hinge donors/acceptors, catalytic triads in proteases, or metal-coordinating spheres):

$$\mathcal{V}_{\text{contact}}(\mathbf{R}_{\text{lig}}, \mathbf{R}_{\text{rec}}) = \sum_{(i, j) \in \text{contacts}} \frac{1}{2} k_{\text{contact}} \max\left(0, \|\mathbf{r}_i^{\text{lig}} - \mathbf{r}_j^{\text{rec}}\| - d_{\text{target}}\right)^2$$

Adding this soft potential directly into OpenMM force groups turns an otherwise flat or deceptive surface landscape into a **steep, monotonic directional funnel** directly guiding the ligand core into the catalytic binding site.
