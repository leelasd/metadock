# Flexible Active-Site Water Example

Demonstrates `openmm-dock`'s native handling of explicit, flexible active-site waters during minimization, simulated-annealing docking, and Monte Carlo basin-hopping. Files originate from rDock's own reference test set (target: PDB 1UYM) but are read and docked entirely natively here — no rDock/rxDock binary is required. `prepare_protein.py` (OpenMM/ParmEd) produced `receptor.mol2` from `output.pdb`, exactly as in the `tethered/` example.

## Files

| File | Description |
|------|-------------|
| `run_solvent_demo.sh` | Native demo: minimize / dock / MC, all with explicit flexible waters |
| `receptor.mol2` | Prepared receptor structure |
| `cavity.prm` | Cavity definition |
| `lig.sdf` | Ligand pose |
| `test_waters.pdb` | Explicit active-site water molecules (flexible during docking) |
| `1UYM.pdb`, `1UYM.fasta.gz`, `1UYM.seq` | Source PDB target and sequence |

## Running the native demo

```bash
bash run_solvent_demo.sh
```

Produces `openmm_solvent_min_out.sdf`, `openmm_solvent_dock_out.sdf`, and `openmm_mc_solvent_out.sdf`.
