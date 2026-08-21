# RNA Docking Example

This example demonstrates docking a small molecule to an RNA target with `openmm-dock`'s own engine. RNA docking needs no special handling — the setup is identical to protein docking; just provide an RNA receptor in MOL2 format. No external rDock/rxDock binary or Docker image is required; `1nem_rdock.prm` uses rDock's cavity-file *format* (read natively by `CavityDefinition.from_prm_file`), which is why the file naming follows that convention.

## Target: 1NEM

PDB ID **1NEM** is an RNA aptamer structure. The example files are in the `1nem/` subdirectory.

## Files (`1nem/`)

| File | Description |
|------|-------------|
| `run_rna_docking_demo.sh` | **Native** end-to-end demo: score → minimize → MC dock → RMSD validation |
| `1nem_rdock.mol2` | RNA receptor prepared in MOL2 format |
| `1nem_lig.sd` | Reference ligand (also used for cavity definition) |
| `1nem_rdock.prm` | Cavity definition (rDock file format, read natively) |

## Running the native demo

```bash
cd 1nem
bash run_rna_docking_demo.sh
```

This runs `omm-dock score` / `minimize` / `mc` against the RNA receptor + reference ligand, then
`omm-dock stats` to report heavy-atom RMSD vs. `1nem_lig.sd`. The MC step passes
`--flex-radius 3.0` to mirror the original rDock `RECEPTOR_FLEX 3.0` setting in
`1nem_rdock.prm` (that field is rDock-specific and isn't read automatically by
`CavityDefinition.from_prm_file`, so it's passed explicitly as a CLI flag instead).

## Notes

- The cavity is defined using the reference ligand method (`REF_MOL`), RADIUS 4.0 Å around `1nem_lig.sd`.
- Outputs: `openmm_rna_score_out.sdf`, `openmm_rna_min_out.sdf`, `openmm_rna_dock_out.sdf`.
