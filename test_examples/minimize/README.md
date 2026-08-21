# Minimization Example

Demonstrates `openmm-dock`'s native local L-BFGS minimization of a ligand pose in a receptor cavity. Files originate from rDock's own reference test set (`minimise.prm` was rDock's original minimization protocol) but are read and minimized entirely natively here — no rDock/rxDock binary is required.

## Files

| File | Description |
|------|-------------|
| `run_minimize_demo.sh` | Native demo: local minimization |
| `receptor.mol2` | Receptor structure |
| `cavity.prm` | Cavity definition |
| `ii.sd` | Ligand pose to minimize |

## Running the native demo

```bash
bash run_minimize_demo.sh
```

Produces `openmm_min_out.sdf`.
