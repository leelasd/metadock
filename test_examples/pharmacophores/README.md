# Pharmacophore Restraint Example

Demonstrates `openmm-dock`'s native pharmacophore-restraint-guided docking, comparing constrained vs. unconstrained Monte Carlo basin-hopping. Files originate from rDock's own reference test set but are read and docked entirely natively here — no rDock/rxDock binary is required. `prepare_protein.py` (OpenMM/ParmEd) produced `receptor.mol2` from `output.pdb`, exactly as in the `tethered/` example.

## Files

| File | Description |
|------|-------------|
| `run_pharmacophores_demo.sh` | Native demo: minimize with restraints, dock with restraints, MC with vs. without restraints |
| `receptor.mol2` | Prepared receptor structure |
| `cavity.prm` | Cavity definition |
| `xtal-lig.sd` | Crystal ligand pose |
| `pharma.restr` | Pharmacophore restraint definition (aromatic + H-bond acceptor features) |
| `optional_pharma.restr` | Alternate restraint set with optional (non-mandatory) features |
| `pharma.xyz` | Pharmacophore feature coordinates |

## Running the native demo

```bash
bash run_pharmacophores_demo.sh
```

Produces `openmm_pharma_dock_out.sdf` (restrained), `openmm_mc_pharma_out.sdf` (restrained MC), and
`openmm_mc_unconstrained_out.sdf` (unrestrained MC, for comparison).
