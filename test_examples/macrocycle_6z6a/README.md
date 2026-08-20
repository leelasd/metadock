# PDB 6Z6A: Macrocycle Inverse Kinematics (IK) & Two-Tier Sampling Benchmark

This benchmark demonstrates **Closed-Loop Inverse Kinematics (IK)** and **Decoupled Two-Tier Kinematics** on **PDB 6Z6A** (human Keap1 Kelch domain in complex with the 16-membered synthetic macrocycle **Q9E**).

---

## 1. System Details

* **Target Macromolecule:** Human Keap1 Kelch domain ([`receptor.pdb`](receptor.pdb)), a 6-bladed $\beta$-propeller protein-protein interaction (PPI) target.
* **Macrocycle Ligand:** Q9E ([`q9e_crystal_pose.sdf`](q9e_crystal_pose.sdf)), a 64-atom synthetic 16-membered macrocycle mimicking the Nrf2 $\text{ETGE}$ motif.
* **Active-Site Pocket:** Centered at `(-21.46, 22.44, -24.18)` ([`cavity.prm`](cavity.prm)), defined by the key **Arginine Triad** (`Arg-415`, `Arg-483`, `Arg-380`), `Tyr-334`, and `Ser-602`.

---

## 2. Decoupled Two-Tier Macrocycle Kinematics

Macrocycles possess two fundamentally distinct types of bonds:

```
┌─────────────────────────────────┬─────────────────────────────────────────┬─────────────────────────────────────────┐
│ Structural Tier                 │ Degrees of Freedom                      │ Mathematical Formulation                │
├─────────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┤
│ **Tier 1: Macrocyclic Ring**    │ **10 Endocyclic Joints**                │ **Inverse Kinematics (IK / DLS):**      │
│ (Backbone Scaffold)             │ (Closed 16-membered ring loop)          │ Solves closed-loop constraints so the   │
│                                 │                                         │ ring flexes with **0.000000 Å gap**.    │
├─────────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┤
│ **Tier 2: Exocyclic Side Arms** │ **9 Rotatable Joints**                  │ **Forward Kinematics (FK):**            │
│ (Functional Substituents)       │ (Carboxylate, Amide, Aromatics, and all │ Unconstrained 1D rotations that orient  │
│                                 │  30 attached Hydrogens)                 │ functional groups to engage pocket args.│
└─────────────────────────────────┴─────────────────────────────────────────┴─────────────────────────────────────────┘
```

### The Analytical Damped Least Squares (DLS) Loop Closure
To close the macrocyclic ring gap $\Delta \mathbf{r} = \mathbf{r}_{\text{tip}} - \mathbf{r}_{\text{anchor}}$:

$$\Delta \boldsymbol{\theta} = \mathbf{J}^T \left( \mathbf{J} \mathbf{J}^T + \lambda^2 \mathbf{I} \right)^{-1} (-\Delta \mathbf{r})$$

Where $\mathbf{J}_j = \hat{\mathbf{u}}_j \times (\mathbf{r}_{\text{tip}} - \mathbf{p}_j)$ is the Geometric Jacobian matrix.

* Across all 120 frames: Maximum ring closure deviation is **$0.000098\text{ \AA}$** ($< 0.0001\text{ \AA}$).
* All 30 hydrogens and side chains move in rigid lockstep with their parent ring carbons (maximum bond length distortion is **$0.000197\text{ \AA}$**).

---

## 3. Files in This Directory

* [`receptor.pdb`](receptor.pdb): Cleaned Keap1 Kelch domain.
* [`cavity.prm`](cavity.prm): Active-site pocket definition.
* [`q9e_crystal_pose.sdf`](q9e_crystal_pose.sdf): Reference X-ray crystal macrocycle seated inside Keap1 pocket (Score: $-211.14\text{ kcal/mol}$).
* [`macrocycle_docked_min.sdf`](macrocycle_docked_min.sdf): L-BFGS minimized docked macrocycle (Score: $-223.54\text{ kcal/mol}$).
* [`run_6z6a_ik_demo.py`](run_6z6a_ik_demo.py): Script generating single-tier ring breathing trajectory.
* [`run_two_tier_demo.py`](run_two_tier_demo.py): Script generating 3-phase decoupled Two-Tier movie.
* [`visualize_6z6a_pymol.pml`](visualize_6z6a_pymol.pml): PyMOL script for in-pocket IK movie.
* [`visualize_two_tier_pymol.pml`](visualize_two_tier_pymol.pml): PyMOL script highlighting Ring (Magenta) vs. Side Chains (Yellow).

---

## 4. Running the Demonstrations & PyMOL Playback

### Option A: Two-Tier Decoupled Kinematics (Ring IK vs. Side-Chain FK)
```bash
cd test_examples/macrocycle_6z6a
python run_two_tier_demo.py
pymol visualize_two_tier_pymol.pml
```
* **Phase 1 (Frames 1–40):** Macrocyclic Ring Breathing (Magenta ring flexes, yellow side chains held rigid).
* **Phase 2 (Frames 41–80):** Exocyclic Side-Chain Articulation (Yellow arms rotate, magenta ring held rigid).
* **Phase 3 (Frames 81–120):** Coupled Two-Tier Docking into Keap1 Kelch binding pocket.

### Option B: Pure In-Pocket Macrocycle Flexing
```bash
python run_6z6a_ik_demo.py
pymol visualize_6z6a_pymol.pml
```

---

## 5. Keap1 Receptor Side-Chain Kinematics ($\chi_1\text{--}\chi_4$)

We also provide a full receptor side-chain kinematic exploration movie for Keap1:

* **Active-Site Residues Articulated:** **23 amino acids** within $10.0\text{ \AA}$ (52 total $\chi$ joint hinges).
  * **Arginine Triad & Network:** `Arg-415`, `Arg-483`, `Arg-380`, `Arg-336`, `Arg-601` (4 $\chi$ joints each)
  * **Aromatic Gating:** `Tyr-334`, `Tyr-572`, `Tyr-525`, `Phe-335`, `Phe-577` (2 $\chi$ joints each)
  * **Polar Network:** `Ser-602`, `Ser-555`, `Ser-363`, `Ser-338`, `Gln-530`, `Gln-337`, `Asn-382`, `Asn-414`
* **Backbone Preservation:** **$0.000000\text{ \AA}$** deviation across all 60 frames.

### Run & View Keap1 Side-Chain Kinematics:
```bash
python run_keap1_sidechain_kinematics.py
pymol visualize_keap1_sidechains_pymol.pml
```
