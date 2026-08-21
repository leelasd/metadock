# Tethered Docking

Tethered docking constrains ligand atoms to a reference pose using Maximum Common Substructure (MCS). Atoms shared between the query ligand and a reference molecule are held near their reference coordinates during docking; non-matching atoms dock freely.

`openmm-dock` implements this natively (`find_tethered_atoms_mcs` + `DockingEngine.dock_simulated_annealing(..., tether_constraints=...)`, exposed as the `omm-dock tether` CLI command) — no external rDock/rxDock binary or Docker image is required to run tethered docking itself. `cavity.prm` uses rDock's cavity-file *format* (read natively by `CavityDefinition.from_prm_file`), which is why the receptor/cavity files below are shared with the original rDock-based setup.

## Files

| File | Description |
|------|-------------|
| `run_tethered_demo.sh` | **Native** end-to-end demo: `omm-dock tether` + RMSD validation |
| `tetheredMinimization.py` | Standalone/offline MCS-tethering utility (same MCS logic `omm-dock tether` runs internally) — useful if you want the `TETHERED ATOMS`-tagged SDF as its own artifact before docking |
| `prepare_protein.py` | Parameterizes a raw receptor PDB with OpenMM/ParmEd to produce `receptor.mol2` (and `.prmtop`/`.inpcrd`/`.gro`, unused by the native docking path) |
| `receptor.mol2` | Prepared receptor structure (input to `DockingEngine`) |
| `receptor.pdb` | Receptor PDB before parameterization (input to `prepare_protein.py`) |
| `xtal-lig.sd` | Crystal ligand (reference pose for MCS tethering and RMSD validation) |
| `query_ligands.sdf` | Example query ligands to dock |
| `cavity.prm` | Cavity definition (rDock file format, read natively) |

## Running the native demo

```bash
bash run_tethered_demo.sh
```

This runs `omm-dock tether -r cavity.prm -ref xtal-lig.sd -i query_ligands.sdf -o openmm_tethered_dock_out.sdf`,
then `omm-dock stats` to report heavy-atom RMSD of the tethered core against `xtal-lig.sd`.

## (Optional) Prepare your own tethered ligands offline

Replace `query_ligands.sdf` with your own SDF file of ligands to dock, or run the MCS-tethering step standalone first to inspect which ligands get tethered vs. dock freely:

```bash
python tetheredMinimization.py xtal-lig.sd query_ligands.sdf outputtethered.sdf outputnontethered.sdf
```

This produces two output files:
- `outputtethered.sdf` — ligands with a `TETHERED ATOMS` property set (atom indices of the MCS match)
- `outputnontethered.sdf` — ligands with no sufficient MCS match (will dock freely)

The `ratioThreshold` in the script (default `0.20`) controls the minimum fraction of the reference molecule that must match for tethering to apply. (`omm-dock tether` applies the same MCS logic and threshold internally per-ligand, so this step is optional — it's for inspecting the tethering decision before committing to a full docking run.)
