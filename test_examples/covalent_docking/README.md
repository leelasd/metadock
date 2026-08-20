# Covalent Docking Example in `openmm-dock`

This folder contains a complete demonstration of **GPU-accelerated covalent docking** in `openmm-dock`, targeting a reactive nucleophilic residue in a macromolecular cavity.

---

## 1. System Components

* **Receptor:** [`receptor.mol2`](receptor.mol2) containing target reactive residue **CYS33**.
* **Cavity Definition:** [`cavity.prm`](cavity.prm) centering the sampling pocket on the CYS33 nucleophilic sphere.
* **Ligand:** [`covalent_ligand.sdf`](covalent_ligand.sdf), a 3D-embedded kinase inhibitor possessing an electrophilic **acrylamide ($\alpha,\beta$-unsaturated carbonyl)** warhead.

---

## 2. Chemical Mechanism & Restraint Potential

The electrophilic $\beta$-carbon (`[C:1]=[C:2]-[C:3](=[O:4])`) is automatically identified via SMARTS perception. During docking, `openmm-dock` applies:

1. **Harmonic Bond Force ($k = 2,000,000\text{ kJ/(mol nm}^2)$):**
   $$E_{\text{bond}}(r) = \frac{1}{2} k_{\text{bond}} (r - r_0)^2, \quad r_0 = 1.82\text{ \AA}$$
2. **Harmonic Valence Angle Force ($k = 5,000\text{ kJ/(mol rad}^2)$):**
   $$E_{\text{angle}}(\theta) = \frac{1}{2} k_{\text{angle}} (\theta - \theta_0)^2, \quad \theta_0 = 104.5^\circ \ (C_\beta\text{--}S_\gamma\text{--}C_\beta)$$

---

## 3. Running the Covalent Example

```bash
# Execute the full automated demo script:
bash run_covalent_docking.sh

# Or run individual subcommands:
# A. Local Covalent L-BFGS Minimization
omm-dock minimize -r cavity.prm -i covalent_ligand.sdf -o openmm_covalent_min_out.sdf --covalent-res CYS33

# B. Covalent Monte Carlo Basin-Hopping with 3D Trajectory Export
omm-dock mc -r cavity.prm -i covalent_ligand.sdf -o openmm_covalent_mc_best.sdf -traj openmm_covalent_trajectory.sdf --covalent-res CYS33 -s 50
```
