# PDB 6DI9: Covalent BTK Kinase Docking Benchmark

This benchmark models **PDB 6DI9** (human Bruton's Tyrosine Kinase, BTK) in complex with the clinical-grade covalent irreversible inhibitor **GJJ** targeting nucleophilic residue **Cys-481**.

---

## 1. System Details

* **Target Macromolecule:** Human Bruton's Tyrosine Kinase (BTK) catalytic domain ([`receptor.pdb`](receptor.pdb))
* **Target Nucleophile:** **Cys-481** (`SG` at `[-23.10, 13.07, 1.18]`)
* **Ligand:** GJJ ([`xtal_ligand.sdf`](xtal_ligand.sdf)), 6-[(3S)-3-(acryloylamino)pyrrolidin-1-yl]-2-{[4-(tert-butylcarbamoyl)phenyl]amino}pyridine-3-carboxamide
* **Cavity Definition:** [`cavity.prm`](cavity.prm) centered on the ATP-binding pocket at `(-12.16, 4.01, 0.43)` with radius $22.0\text{ \AA}$
* **Warhead:** Electrophilic **acrylamide** ($\beta$-carbon at atom index 19)

---

## 2. Benchmark Results

| Sampling Protocol | Final Docking Score | Intermolecular VDW | Heavy-Atom RMSD to Crystal | Status |
| :--- | :--- | :--- | :--- | :--- |
| **L-BFGS Minimization** | **$-123.44\text{ kcal/mol}$** | **$-116.40\text{ kcal/mol}$** | **$0.651\text{ \AA}$** | **SUCCESS ($< 1.0\text{ \AA}$)** |
| **Monte Carlo Basin-Hopping (50 steps)** | **$-139.35\text{ kcal/mol}$** | **$-117.83\text{ kcal/mol}$** | **$1.121\text{ \AA}$** | **SUCCESS ($< 2.0\text{ \AA}$)** |

---

## 3. Running the Benchmark

```bash
# Execute the full automated benchmark runner:
bash run_6di9_demo.sh
```
