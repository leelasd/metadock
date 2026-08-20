# PDB 6Z6A: Macrocycle Inverse Kinematics (IK) Benchmark

This benchmark demonstrates **Damped Least Squares (DLS) Inverse Kinematics** on **PDB 6Z6A** (human Keap1 Kelch domain in complex with the 16-membered macrocyclic inhibitor **Q9E**).

---

## 1. System Details

* **Target Macromolecule:** Human Keap1 Kelch domain ([`receptor.pdb`](receptor.pdb))
* **Ligand:** Q9E ([`q9e_macrocycle.sdf`](q9e_macrocycle.sdf)), a synthetic 16-membered macrocycle mimicking the Nrf2 ETGE motif
* **Pocket Centroid:** `(-21.46, 22.44, -24.18)` ([`cavity.prm`](cavity.prm))

---

## 2. Inverse Kinematics (IK) Ring Closure

* **Closure Error Across 60 Frames:** **$0.000098\text{ \AA}$** (Maximum deviation $< 0.0001\text{ \AA}$).
* **Mechanism:** Perturbs driver dihedrals and solves $\Delta \boldsymbol{\theta} = \mathbf{J}^T (\mathbf{J} \mathbf{J}^T + \lambda^2 \mathbf{I})^{-1} \mathbf{e}(\boldsymbol{\theta})$ to keep the ring closed at every frame.

---

## 3. Running the Demo & PyMOL Visualization

```bash
cd test_examples/macrocycle_6z6a

# Run IK breathing generator
python run_6z6a_ik_demo.py

# Open 3D movie in PyMOL
pymol visualize_6z6a_pymol.pml
```
