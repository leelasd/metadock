# PDB 6DI9: Covalent BTK Kinase Docking Example

This benchmark models **PDB 6DI9** (Bruton's Tyrosine Kinase, BTK) in complex with the clinical-grade covalent irreversible inhibitor **GJJ** targeting **Cys-481**.

---

## 1. System Details

* **Target Macromolecule:** Human Bruton's Tyrosine Kinase (BTK) kinase domain ([`receptor.pdb`](receptor.pdb))
* **Reactive Target Residue:** **Cys-481** (`SG` at `[-23.10, 13.07, 1.18]`)
* **Ligand:** GJJ ([`query_ligand.sdf`](query_ligand.sdf)), 6-[(3S)-3-(acryloylamino)pyrrolidin-1-yl]-2-{[4-(tert-butylcarbamoyl)phenyl]amino}pyridine-3-carboxamide
* **Warhead:** Electrophilic **acrylamide** ($\beta$-carbon at atom index 19)

---

## 2. Running the 6DI9 Covalent Docking Protocol

```bash
# 1. Covalent L-BFGS Minimization
omm-dock minimize -r cavity.prm -i query_ligand.sdf -o openmm_6di9_min_out.sdf --covalent-res CYS481

# 2. Covalent Monte Carlo Basin-Hopping (50 steps) + 3D Movie Export
omm-dock mc -r cavity.prm -i query_ligand.sdf -o openmm_6di9_mc_out.sdf -traj openmm_6di9_mc_trajectory.sdf --covalent-res CYS481 -s 50
```
