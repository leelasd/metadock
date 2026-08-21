# PDB 9Z1L: KIT V654A Kinase + BLU-654 — Full-Suite Benchmark

**Target**: KIT V654A mutant kinase domain (359-residue construct, 312
resolved) in complex with **BLU-654** (ligand code `A1CZZ`), a potent,
selective inhibitor developed for imatinib-resistant GIST (Moine et al.,
*J. Med. Chem.* 2026). 1.543 Å resolution. [RCSB entry](https://www.rcsb.org/3d-view/9Z1L).

Every input file in this directory was generated directly from the raw PDB
download — see [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full build
log (fetching, ligand bond-order reconstruction, a missing terminal atom,
and two real bugs found and fixed along the way). This directory exercises
**every major search/scoring methodology** in `openmm-dock` on one real,
previously-unseen system: blind docking, pharmacophore-restrained docking,
explicit-water docking, forward kinematics, kinematic PSO, and
metadynamics.

BLU-654 has **no macrocyclic ring** (three separate small aromatic rings on
rotatable linkers) — a genuinely different ligand topology from every other
`test_examples/` benchmark (all macrocycles or covalent inhibitors so far),
making this a real-world test of the kinematic engines' generalization to
ordinary small molecules.

## Files

| File | Description |
|---|---|
| `extract_from_pdb.py` | Fetches nothing itself (run after `curl`-ing the raw files, see DOCUMENTATION.md) — splits the raw mmCIF into receptor/ligand/waters |
| `make_pharma_restr.py` | Generates `pharma.restr` from the crystal ligand's pharmacophore features |
| `prepare_protein.py` | Parameterizes `receptor.pdb` → `receptor.mol2` (OpenMM/ParmEd, adds missing hydrogens) |
| `9z1l.cif` | Raw mmCIF download (legacy PDB format unavailable — 5-char ligand code) |
| `A1CZZ_ccd.cif`, `A1CZZ_ideal.sdf` | Chemical Component Dictionary reference files for the ligand |
| `receptor.pdb` / `receptor.mol2` | Protein-only receptor (raw / force-field-parameterized) |
| `a1czz_crystal_pose.sdf` | Ligand crystal pose, correct bond orders, explicit hydrogens |
| `active_site_waters.pdb` | 11 crystallographic waters within 5 Å of the ligand |
| `cavity.prm` | Pocket definition (center = ligand crystal COM, 15 Å radius) |
| `pharma.restr` | 3 aromatic-ring + 2 donor-nitrogen pharmacophore points |

## Running the demos

Regenerate everything from scratch (optional — all outputs are already committed):

```bash
python extract_from_pdb.py      # mmCIF -> receptor.pdb, ligand SDF, active_site_waters.pdb
python prepare_protein.py       # receptor.pdb -> receptor.mol2
python make_pharma_restr.py     # -> pharma.restr
```

Then any of the six method demos, independently:

```bash
python run_kinematics_demo.py             # Forward kinematics: 240-frame joint sweep, 10 rotatable bonds
python run_pso_demo.py                    # Kinematic PSO: -316.20 kcal/mol, 0.50 A RMSD to crystal
python run_metadynamics_demo.py           # Metadynamics pose-strength assay (see below)
python run_pharmacophore_docking_demo.py  # Pharmacophore-restrained vs. free docking
python run_solvent_docking_demo.py        # Dry pocket vs. explicit flexible waters
python run_blind_docking_demo.py          # Global blind docking from bulk solvent (largest, ~600 frames)
```

Each writes its own `visualize_*_pymol.pml` — open with `pymol visualize_<name>_pymol.pml` to watch the trajectory as a movie.

## Results summary

| Method | Result |
|---|---|
| **Forward kinematics** | 240-frame sweep across all 10 rotatable joints, exactly 0.000000 Å bond distortion throughout |
| **Kinematic PSO** | Converged to **-316.20 kcal/mol**, **0.50 Å** RMSD to the crystal pose (20 particles × 20 iterations) |
| **Metadynamics (pose strength)** | 31 repulsive hills deposited; the native pose survives **13 steps** of active repulsion before its physical score turns unfavorable (>0 kcal/mol) — a strong, well-defined binding basin |
| **Pharmacophore docking** | Best RMSD to crystal improves from **9.53 Å** (no restraints) to **3.57 Å** (with restraints) — 10 simulated-annealing runs each |
| **Solvent docking** | Dry pocket: -351.87 kcal/mol / 0.79 Å RMSD. With 11 explicit flexible waters: -357.65 kcal/mol / 0.75 Å RMSD (minimized crystal pose + waters: -294.20 kcal/mol / 0.01 Å) |
| **Blind docking** | Global search from bulk solvent (30 particles × 20 iterations, 600 frames): **18.69 Å → 11.94 Å** RMSD, final score -132.08 kcal/mol. Real progress toward the pocket, not sub-angstrom — a genuinely blind search from complete bulk solvent on a non-macrocyclic ligand is a much harder problem than the macrocycle benchmarks elsewhere in this repo, where the ring-closure constraint itself prunes most of the search space |

Crystal-pose reference score (`DockingEngine.score()`, no search): **-282.05 kcal/mol**.

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the complete build process, including two real bugs this system's construction surfaced and fixed:
`omm-dock`'s `tether` CLI (unrelated, fixed in an earlier pass) and `pharmacophore.py`'s donor-atom detection (`GetTotalNumHs()` silently finding zero donors on any explicit-hydrogen molecule — fixed with `includeNeighbors=True`).
