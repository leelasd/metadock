# PDB 6Z6A: Collaborative Multi-Swarm Kinematic Metadynamics (19D Shared Negative Memory)

This benchmark demonstrates the **Collaborative Multi-Swarm Kinematic Metadynamics Engine** on **PDB 6Z6A** (Keap1 Kelch Domain + Q9E 16-Membered Macrocycle).

---

## 1. Architectural Highlights

```
                       COLLABORATIVE MULTI-SWARM KINEMATIC METADYNAMICS
                       
      ISLAND 1 (16 Walkers)             SHARED REH-METADYNAMICS ARCHIVE           ISLAND 2 (16 Walkers)
    ┌───────────────────────┐                  ┌──────────────────────┐         ┌───────────────────────┐
    │ Conformer Seed #1     │ ─── Deposits ──► │ Basin Archive:       │ ◄───Reads │ Conformer Seed #2     │
    │ Explores Channel Alpha│     Gaussian     │ • Decoy 1 (+16 kcal) │   Biased│ Explores Channel Beta │
    │ Personal Bests p_best │     Hills        │ • Decoy 2 (+14 kcal) │    Space│ Personal Bests p_best │
    │ Island Best l_best    │                  │ • Decoy 3 (+12 kcal) │         │ Island Best l_best    │
    └───────────────────────┘                  └──────────────────────┘         └───────────────────────┘
                                                  ▲                ▲
                                          Deposits│                │Reads
                                                  │                │
                                       ┌──────────────────────────────┐
                                       │ ISLAND 3 & ISLAND 4 (32 Wk.) │
                                       │ Actively repelled from       │
                                       │ Decoys 1, 2, 3               │
                                       │ Forced into Native Basin!    │
                                       └──────────────────────────────┘
```

1. **Dimensionality Reduction ($50\text{D} \to 19\text{D}$):**
   * Keeping the receptor rigid during global swarm exploration eliminates $31$ noisy side-chain $\chi$ degrees of freedom and cuts coordinate rebuild overhead to zero.
   * Search space is strictly **$19$ Kinematic DOFs**:
     $$\mathbf{x} = \underbrace{[x, y, z, \theta_x, \theta_y, \theta_z]}_{6\text{ Rigid Body}} + \underbrace{[\theta_1, \theta_3, \theta_5, \theta_8]}_{4\text{ Ring IK Drivers}} + \underbrace{[\phi_1, \dots, \phi_9]}_{9\text{ Exocyclic FK Dihedrals}}$$
   * Guaranteed **$0.000\text{ \AA}$ internal bond/angle strain**.

2. **Collaborative Multi-Swarm (Island Model):**
   * $4$ independent sub-swarms ($64$ particles total) seeded with diverse macrocyclic conformer templates.
   * Local social attractors ($\mathbf{l}_{\text{best}}$) prevent premature gravitational collapse to a single false decoy.

3. **Shared Negative Memory (Metadynamics Repulsion):**
   * Whenever an island identifies a local minimum / decoy well, it deposits a repulsive Gaussian hill in the `SharedMetadynamicsArchive`.
   * Other islands evaluating that region feel a repulsive penalty $+V_{\text{bias}}(\mathbf{x})$, forcing them to explore unvisited pocket crevices.

4. **Final OpenMM GPU L-BFGS Polish:**
   * Candidate poses from all islands are relaxed with analytical gradients in OpenMM to reach sub-angstrom / crystallographic resolution.

---

## 2. Running the Benchmark

```bash
cd test_examples/collaborative_metadynamics_6z6a
python run_collaborative_6z6a_demo.py
```

---

## 3. 3D PyMOL Multi-Track Visualization

```bash
pymol visualize_collaborative_6z6a_pymol.pml
```

#### What You Will See in PyMOL:
* **Gray Cartoon & Marine Sticks:** Keap1 Kelch $\beta$-propeller scaffold and the active-site Arginine triad (`Arg-415`, `Arg-483`, `Arg-380`, `Tyr-525`).
* **Forest Green Sticks:** Crystal reference ligand pose.
* **Warm Pink Spheres:** Shared Metadynamics repulsive basins deposited during the run (the "tabu" decoy traps).
* **Multi-Color Multi-Track Movie:** 
  * Red: Island 1
  * Cyan: Island 2
  * Green: Island 3
  * Slate: Island 4
* **Gold Sticks:** Final converged crystallographic complex.
