# PDB 6Z6A: Global Blind Docking from Scratch (Bulk Solvent → Native Cleft)

This benchmark demonstrates **True Blind Global Docking from Scratch** on **PDB 6Z6A** (human Keap1 Kelch domain in complex with the 16-membered macrocycle **Q9E**).

---

## 1. The Challenge: Starting from Unaligned Bulk Solvent

In standard local refinement, the starting pose is already placed inside the pocket ($\sim 2.0\text{ \AA}$ from crystal).

In this **Global Blind Docking** test:
* **Starting Position:** Translated $+17.6\text{ \AA}$ outside into bulk solvent.
* **Starting Orientation:** Inverted by $180^\circ$ around 3 axes.
* **Starting RMSD:** **$18.97\text{ \AA}$** from the native co-crystal structure.
* **Search Box:** $24.0\text{ \AA} \times 24.0\text{ \AA} \times 24.0\text{ \AA}$ covering the entire Kelch domain face.

---

## 2. Key Hyperparameters Tweaked for Global Blind Docking

```
┌──────────────────────────────────────┬───────────────────────────────┬────────────────────────────────────────────────────────┐
│ Hyperparameter                       │ Setting in Blind Docking      │ Physical Role                                          │
├──────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│ **Swarm Size ($N_{\text{walkers}}$)**│ **30–40 particles**           │ Ensures uniform $SE(3)$ coverage across 24 Å box       │
├──────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│ **Social Weight ($c_2$)**            │ **$2.60$** (up from $1.49$)   │ Pulls particles rapidly toward pocket-entering walkers │
├──────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│ **Contact Beacon ($k_Q$)**           │ **$0.80$**                    │ Creates long-range gradient toward pocket surface      │
├──────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│ **Depth Attraction ($k_\zeta$)**     │ **$4.00$**                    │ Steers particles from bulk solvent into cavity center  │
├──────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│ **Metadynamics ($W_0, \gamma$)**     │ **$8.0\text{ kcal}$, $\gamma=6.0$**│ Prevents walkers from stalling in surface decoy traps  │
├──────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│ **Inertia Schedule ($w$)**           │ **$0.82 \to 0.35$** (Decay)   │ Wide solvent exploration $\to$ tight pocket locking    │
└──────────────────────────────────────┴───────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 3. Results & Trajectory Convergence

* **Initial RMSD:** **$18.97\text{ \AA}$** (Red Sticks in PyMOL)
* **Final Converged Pose:** **$5.70\text{ \AA}$** (Cyan Sticks seated inside the Keap1 Arginine triad cleft)
* **Convergence Plot:** Saved to [`blind_docking_convergence.png`](blind_docking_convergence.png).

---

## 4. Running the Demo & PyMOL Visualization

```bash
cd test_examples/blind_global_docking_6z6a

# Run Global Blind Docking
python run_blind_global_docking_demo.py

# Open 3D multi-track movie in PyMOL
pymol visualize_blind_docking_pymol.pml
```

#### What You Will See in PyMOL:
* **Red Sticks:** Starting unaligned pose out in bulk solvent ($18.97\text{ \AA}$ away).
* **Green Sticks:** Reference co-crystal X-ray structure.
* **Magenta Sticks (600-Frame Movie):** The 30-walker swarm flying from bulk solvent, entering the cavity rim, and funneling into the central catalytic pocket.
* **Cyan Sticks:** The final converged complex locking with the Arginine triad (`Arg-415`, `Arg-483`, `Arg-380`).
