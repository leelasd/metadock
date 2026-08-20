# PDB 6Z6A: Automated Bridged Two-Stage Docking Pipeline (Bulk Solvent → Stage 2 Induced-Fit)

This benchmark demonstrates the **Automated Bridged Two-Stage Pipeline** that connects **Global Bulk Solvent Ingress (Stage 1)** to **In-Pocket Kinematic Induced-Fit Refinement (Stage 2)** on **PDB 6Z6A** (Keap1 + Q9E Macrocycle).

---

## 1. How the Automated Bridge Works

```
                              THE AUTOMATED TWO-STAGE BRIDGE
                              
   STAGE 1: Global Swarm Ingress (19D)               THE BRIDGE GATE               STAGE 2: Kinematic Induced Fit (50D)
   
   • Starts: 23.9 Å in bulk solvent                  Gate Condition:               • Unlocks 14 Pocket Residues (31 χ joints)
   • Rigid receptor (0 χ DOF)                  ┌─────────────────────────┐         • Unlocks Two-Tier Macrocycle IK/FK
   • 40 Walkers explore 24 Å box ────────────► │ COM Distance ≤ 5.0 Å    │ ──────► • Multi-angle rotational sweep
   • Guides via (ζ_depth, Q_contacts)          │ (Cavity Entry Confirmed)│         • OpenMM GPU L-BFGS Minimizer
                                               └─────────────────────────┘         • Final Complex: -223.5 kcal/mol well
```

---

## 2. Benchmark Trajectory Results

* **Starting State (Bulk Solvent):** $\text{RMSD} = \mathbf{23.90\text{ \AA}}$ (Red Sticks in PyMOL)
* **Post-Stage 1 Ingress:** $\text{RMSD} = \mathbf{8.96\text{ \AA}}$ (Cavity centroid entry: $\Delta \mathbf{r}_{\text{COM}} = 5.3\text{ \AA}$)
* **Post-Stage 2 Induced Fit:** $\text{RMSD} = \mathbf{5.88\text{ \AA}}$ (Cyan Sticks clasping with `Arg-415` and `Arg-483`)
* **Stage Transition Plot:** Saved to [`bridged_docking_stage_transition.png`](bridged_docking_stage_transition.png).

---

## 3. Running the Demo & PyMOL Visualization

```bash
cd test_examples/bridged_two_stage_docking_6z6a

# Run Bridged Two-Stage Pipeline
python run_bridged_two_stage_demo.py

# Open 3D multi-track movie in PyMOL
pymol visualize_bridged_docking_pymol.pml
```

#### What You Will See in PyMOL:
* **Red Sticks:** Starting unaligned pose out in solvent ($23.9\text{ \AA}$ away).
* **Green Sticks:** Reference co-crystal structure.
* **Magenta Sticks (Multi-Track Movie):** The swarm flying from bulk solvent (Stage 1, Coral in plot) and transitioning into in-pocket side-chain clasping (Stage 2, Green in plot).
* **Cyan Sticks:** The final converged complex locking into the catalytic Arginine triad (`Arg-415`, `Arg-483`, `Arg-380`).
