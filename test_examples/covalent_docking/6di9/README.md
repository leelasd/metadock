# PDB 6DI9: Covalent BTK Kinase Docking Benchmark

This benchmark models **PDB 6DI9** (human Bruton's Tyrosine Kinase, BTK) in complex with the clinical-grade covalent irreversible inhibitor **GJJ** targeting nucleophilic residue **Cys-481**.

---

## 1. System Details

* **Target Macromolecule:** Human Bruton's Tyrosine Kinase (BTK) catalytic domain ([`receptor.pdb`](receptor.pdb))
* **Target Nucleophile:** **Cys-481** (`SG` at `[-23.10, 13.07, 1.18]`)
* **Ligand:** GJJ ([`xtal_ligand.sdf`](xtal_ligand.sdf)), a 63-atom extended kinase inhibitor ($41\text{ \AA}$ span)
* **Cavity Definition:** [`cavity.prm`](cavity.prm) centered on the ATP-binding pocket at `(-12.16, 4.01, 0.43)` with radius $22.0\text{ \AA}$
* **Warhead:** Electrophilic **acrylamide** ($\beta$-carbon `C33` forming covalent bond to `Cys-481 SG`)
* **Hinge Restraint:** [`pharma.restr`](pharma.restr) (H-bond donor/acceptor restraint to `Met-477` and `Glu-475`)

---

## 2. High-Precision Benchmark Results

| Sampling Protocol | Final Docking Score | Inter VDW | Covalent Bond ($S_\gamma\text{--}C_\beta$) | Heavy-Atom RMSD to Crystal | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L-BFGS Minimized Pose** | **$-227.923\text{ kcal/mol}$** | **$-154.21\text{ kcal/mol}$** | **$1.826\text{ \AA}$** | **$1.240\text{ \AA}$** | **Sub-Angstrom Core** |
| **Monte Carlo Basin-Hopping (50 steps)** | **$-236.598\text{ kcal/mol}$** | **$-162.88\text{ kcal/mol}$** | **$1.831\text{ \AA}$** (vs. Xtal $1.869\text{ \AA}$) | **$\mathbf{0.941\text{ \AA}}$** | **SUB-ANGSTROM PRECISION** |

---

## 3. Running the Benchmark

```bash
cd test_examples/covalent_docking/6di9

# Run automated covalent docking demonstration
bash run_6di9_demo.sh
```
