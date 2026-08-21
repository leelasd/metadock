# PDB 9Z1L: End-to-End Build Log

This document records, step by step, exactly how this directory was built —
from fetching the raw structure off RCSB through every fix, bug, and design
decision made along the way — so the process is fully reproducible and
auditable. It complements [`README.md`](README.md) (which documents *how to
run* the finished demos); this file documents *how they were made*.

**Target**: PDB [9Z1L](https://www.rcsb.org/3d-view/9Z1L) — the KIT V654A
mutant kinase domain in complex with **BLU-654** (CCD ligand code `A1CZZ`), a
potent, selective inhibitor designed for imatinib-resistant GIST (Moine et
al., *J. Med. Chem.* 2026, DOI `10.1021/acs.jmedchem.5c03554`). Chosen because
it's a real, very recent (2026), high-resolution (1.543 Å) single-chain
kinase structure with a small-molecule ligand, crystallographic waters near
the binding site, and no macrocycle — a genuinely different chemotype from
every other benchmark in `test_examples/` (all Keap1+macrocycle or covalent
kinase systems so far), making it a good stress test of the codebase's
*generalization* work (adaptive ring-driver count: 0/1/4) rather than a
repeat of an already-proven case.

---

## 1. Fetching the structure

RCSB's REST API (`https://data.rcsb.org/rest/v1/core/entry/9Z1L`) was queried
first to identify the real ligand before downloading any coordinates:

- `polymer_entity/9Z1L/1` → single protein chain, "Mast/stem cell growth
  factor receptor" (KIT), *Homo sapiens*, 359-residue sample sequence.
- `nonpolymer_entity/9Z1L/2` → `MRD`, (4R)-2-methylpentane-2,4-diol — a
  cryoprotectant, not a real ligand. Excluded.
- `nonpolymer_entity/9Z1L/3` → **`A1CZZ`** — the real ligand (BLU-654).

**Legacy PDB format is not available for this entry** —
`https://files.rcsb.org/download/9Z1L.pdb` returns a 404. This is because
`A1CZZ` is a 5-character "extended CCD" code (RCSB has used 5-character
codes since ~2023 once the 3-character alphabet was exhausted), which the
legacy PDB format's 3-character `HETATM` residue-name field cannot
represent. **mmCIF is the only download format** for this entry:

```bash
curl -s "https://files.rcsb.org/download/9Z1L.cif" -o 9z1l.cif
```

Two more files were fetched for ligand reconstruction (see §3):

```bash
curl -s "https://files.rcsb.org/ligands/download/A1CZZ_ideal.sdf" -o A1CZZ_ideal.sdf   # not used in the end -- see below
curl -s "https://files.rcsb.org/ligands/download/A1CZZ.cif"       -o A1CZZ_ccd.cif      # used
```

## 2. Parsing the mmCIF and splitting receptor / ligand / waters

`OpenMM`'s `PDBxFile` parses `9z1l.cif` directly (RDKit's `MolFromPDBFile`
cannot read mmCIF at all, and doesn't perceive elements correctly on HETATM
records here either). The topology has 4 chains:

| Chain | Contents |
|---|---|
| A | 312 protein residues (TYR 1 → HIS 312) |
| B | 1× `MRD` (cryoprotectant — dropped) |
| C | 1× `A1CZZ` (the ligand, 25 heavy atoms) |
| D | 475× `HOH` (crystallographic waters) |

`extract_from_pdb.py` splits this into `receptor.pdb` (chain A only) and
picks up the ligand and water atoms separately.

## 3. Reconstructing the ligand: two failed approaches, one that worked

The crystal structure gives correct **coordinates** for the ligand's 25
heavy atoms but **no bond orders** (mmCIF/PDB coordinate files never encode
chemistry, and this resolution doesn't resolve hydrogens). Getting a
chemically valid ligand SDF required combining crystal coordinates with a
separately-sourced bond-order reference — this took two attempts:

**Attempt 1 — distance-based connectivity + `AssignBondOrdersFromTemplate`
against the CCD "ideal" SDF.** The standard RDKit recipe: perceive
single-bond connectivity from 3D distances (`rdDetermineBonds.
DetermineConnectivity`), then map correct bond orders from a reference
template via substructure match. This **failed**: RDKit reported
`"WARNING: More than one matching pattern found - picking one"` and silently
picked a wrong atom correspondence — the resulting molecule had nonsensical
valences (a carbon with total valence 1, when carbon must be 4). BLU-654 has
enough local symmetry (the pyrazole and pyridine rings) that graph
isomorphism against a same-size, same-composition template is ambiguous.

**Attempt 1b — same idea, but the immediate blocker was H-count mismatch.**
Before even reaching the symmetry problem: the crystal ligand has 25 heavy
atoms, but `A1CZZ_ideal.sdf` has 46 atoms (25 heavy + 21 H). Matching
directly failed with `"No matching found"` until the template was stripped
to heavy atoms first (`Chem.RemoveHs`). Even after fixing that, the
ambiguous-symmetry problem above remained.

**Attempt 2 — build connectivity from the CCD *definition* file by atom
name (what actually worked).** `A1CZZ.cif` (the Chemical Component
Dictionary's own definition of this ligand, distinct from the crystal
structure and distinct from the "ideal" SDF) contains a `_chem_comp_bond`
loop listing every bond **by atom name** (`C1-N1 SING`, `N1-C2 SING`, `C2-N2
DOUB Y`, ...) — and the crystal structure's atom names (`C1`, `N1`, `C2`,
...) come from that exact same CCD entry, so they correspond **exactly**, no
graph matching needed. `extract_from_pdb.py`'s `parse_ccd_bonds()` parses
this loop with a regex, builds an `RWMol` with bonds assigned by name
lookup (skipping bonds to hydrogen, since none are resolved in the crystal
structure), attaches crystal coordinates by the same name lookup, then
`Chem.AddHs(mol, addCoords=True)` adds hydrogens geometrically. Result: 46
atoms / 48 bonds — exactly matching the ideal SDF's atom/bond count, and a
`Chem.MolToSmiles` round-trip (`CNc1nccc(Nc2cc(OC(C)C)c(-c3cnn(C)c3)cn2)n1`)
matches the RCSB-reported IUPAC name
(*N²-methyl-N⁴-{5-(1-methyl-1H-pyrazol-4-yl)-4-[(propan-2-yl)oxy]pyridin-2-yl}pyrimidine-2,4-diamine*)
piece for piece — the pyrimidine-2,4-diamine core, the N²-methyl, and the
pyridine with its isopropoxy and 1-methylpyrazol-4-yl substituents are all
there. `A1CZZ_ideal.sdf` ended up unused in the final script but is kept in
the directory since it's a useful independent reference.

**A second, smaller bug surfaced during this**: after building the
heavy-atom-only molecule and calling `Chem.AddHs`, no hydrogens were
actually added (`lig_mol.GetNumAtoms()` stayed at 25). Cause: atoms built
via `rdDetermineBonds.DetermineConnectivity` in the earlier (abandoned)
attempt carried a `noImplicit=True` flag copied through by
`AssignBondOrdersFromTemplate`, which freezes an atom's implicit-H count at
0 regardless of its real valence. Fixed in the working path by not going
through that code path at all (bonds are built directly from the CCD
definition, atoms never pass through `DetermineConnectivity`).

## 4. The missing C-terminal OXT atom

`prepare_protein.py` (see §5) failed with:

```
ValueError: No template found for residue 311 (HIS). The atoms and bonds in
the residue match HID, but the set of externally bonded atoms is missing 1 C
atom. Is the chain missing a terminal capping group?
```

Diagnosis: the crystal structure never resolved the C-terminal carboxylate
oxygen (`OXT`) on the last residue (HIS 312). Without it, AMBER-style
residue templates treat that residue as *internal* (expecting a peptide
bond to a next residue that doesn't exist) rather than terminal, and
`ForceField.createSystem()` (used internally by `Modeller.addHydrogens`)
can't match it to any template. `Modeller.addHydrogens` only adds missing
**hydrogens** — it does not add missing **heavy atoms** like OXT.

Fixed in `extract_from_pdb.py` by adding OXT geometrically before writing
`receptor.pdb`: reflect the existing carbonyl oxygen (`O`) across the
`CA–C` axis, in the `CA`/`C`/`O` plane, at the same `C–O` bond length — the
standard construction for a missing terminal carboxylate oxygen (produces
a roughly symmetric, chemically reasonable carboxylate). Getting this into
`PDBFile.writeFile()` correctly required care with OpenMM's `Quantity`
unit-wrapping: `Modeller.getPositions()` returns one `Quantity` wrapping a
list of unitless `Vec3`s (nanometers), not a list of individually-wrapped
`Quantity` objects — appending a differently-shaped `Quantity` broke
`writeModel`'s internal `np.isnan()` call with a `TypeError`. Fixed by
stripping units once (`.value_in_unit(unit.nanometer)`), doing the
geometry in plain `numpy`/`Vec3`, and re-wrapping the whole list once at
the end.

## 5. Receptor parameterization (`prepare_protein.py`)

Follows the same ParmEd/OpenMM pattern as `tethered/`, `solvent/`, and
`pharmacophores/`'s `prepare_protein.py` (`amber10.xml` → ParmEd
`Structure` → `receptor.mol2`), with one addition those don't need: an
explicit `Modeller.addHydrogens(forcefield, pH=7.0)` step. Those other
examples' `output.pdb` inputs already had hydrogens added upstream before
being checked into the repo; `receptor.pdb` here comes straight from the
raw crystal structure and has none. Output: `receptor.mol2`, 4982 atoms
(2496 heavy + hydrogens), verified to load via `openmm_dock.core.
Mol2Parser` and to score the crystal pose at **-282.05 kcal/mol** via
`DockingEngine.score()` — a strongly favorable score, consistent with a
real, optimized, sub-2-Å-resolution inhibitor-kinase co-crystal structure.

## 6. Active-site waters and pocket definition

`extract_from_pdb.py` keeps any of the 475 crystallographic waters within
5 Å of any ligand heavy atom — **11 waters** survive this filter — written
to `active_site_waters.pdb` for the explicit-water docking demo.

`cavity.prm` centers the search sphere at the ligand's crystal
center-of-mass, `(16.92, -31.66, 18.54)`, with a 15 Å radius (matching the
convention used by every other single-target `cavity.prm` in this repo).

## 7. Pharmacophore restraints, and a real bug this surfaced

`make_pharma_restr.py` generates `pharma.restr` from the crystal ligand
using `openmm_dock.pharmacophore.find_ligand_pharma_features` — the same
feature detector `DockingEngine(pharma_restr_path=...)` uses internally, so
the restraints are guaranteed self-consistent with the engine's own idea of
a pharmacophore feature. It picked up the 3 aromatic ring centroids fine
(pyrimidine, pyridine, pyrazole) but **initially found zero donor atoms**,
despite BLU-654 clearly having two aniline N–H groups (the diaminopyrimidine
hinge-binding motif this whole chemotype is built around).

**Root cause (a real bug in `openmm_dock/pharmacophore.py`, fixed as part
of this work)**: donor detection used `atom.GetTotalNumHs() > 0`. RDKit's
`GetTotalNumHs()` reports the *packed* implicit/explicit-H **count
property** on an atom — it does **not** count actual explicit hydrogen
**atoms** bonded as graph neighbors. Once a molecule has been through
`Chem.AddHs()` (which `extract_from_pdb.py`'s ligand, and apparently many
other ligand SDFs already in this repo, have), each heavy atom's hydrogens
exist as separate neighbor atoms in the graph, and `GetTotalNumHs()`
silently returns 0 for all of them — the function was quietly finding zero
donors on any explicit-hydrogen molecule, repo-wide, not just here. Fixed
by passing `includeNeighbors=True`, which makes RDKit also count explicit H
neighbor atoms: `atom.GetTotalNumHs(includeNeighbors=True) > 0`. Verified
against this ligand (`N` atoms 1 and 7 now correctly report 1 H each) and
confirmed with the full `pytest` suite (37/37 still pass — no other test
happened to have a molecule that exercised this path with a nonzero
expected donor count).

Final `pharma.restr`: 3 `Aro` points (ring centroids) + 2 `Don` points (the
aniline N–H nitrogens), 5 points total.

## 8. Verifying the kinematics generalization on a non-macrocyclic ligand

Before writing the six demo scripts, `UnifiedKinematicPSOEngine` was
sanity-checked directly against this ligand, since BLU-654 has **no
macrocyclic ring** (three small separate aromatic rings connected by
rotatable bonds) — a genuinely different topology from every other ligand
this engine has been exercised against in this repo (all macrocycles).
This exact scenario is what an earlier session's generalization work
(adaptive `num_ring_drivers` ∈ {0, 1, 4} depending on ring size, instead of
always assuming ≥4 macrocycle ring joints) was for:

```
[*] Macrocycle IK Engine initialized: 46 total atoms | 6-membered ring
[*] Identified 0 rotatable joint hinges with full atomic subtree propagation
[*] Two-Tier Macrocycle Engine Ready: 0 Ring IK Joints | 10 Exocyclic FK Joints
...
Total Coupled Degrees of Freedom: 45
crystal-pose coupled score: -245.39 kcal/mol
```

`num_ring_drivers=0` (correctly recognizing this isn't a macrocycle),
`num_exo=10` (the real rotatable-bond count), and the coupled energy
evaluates cleanly — confirming the generalization holds on a real,
previously-untested small-molecule system, not just the synthetic case it
was written for.

## 9. The six demo scripts

Each demonstrates one of this repo's search/scoring methodologies on the
9Z1L system; see [`README.md`](README.md) for how to run them and what
their outputs mean. Built by directly adapting the closest existing
example in `test_examples/` (`blind_global_docking_6z6a`,
`kinematics_workflow`, `kinematic_pso_demo`,
`macrocycle_metadynamics_6z6a`) to this system's paths/pocket center, plus
two new ones (pharmacophore and solvent comparison demos) modeled on the
native `omm-dock` CLI patterns already used in `pharmacophores/` and
`solvent/`.

All six were run at full scale (not reduced/smoke-test parameters) and
verified to complete successfully:

| Demo | Result |
|---|---|
| `run_kinematics_demo.py` | 240-frame joint sweep, 10 rotatable joints, 0.000000 Å bond distortion |
| `run_pso_demo.py` | Kin-PSO converged to **-316.20 kcal/mol**, **0.50 Å RMSD** to crystal |
| `run_metadynamics_demo.py` | 31 hills deposited; pose survived 13 steps of active repulsion before the physical score turned unfavorable (a strong/deep native basin) |
| `run_pharmacophore_docking_demo.py` | Best RMSD improved from **9.53 Å** (no restraints) to **3.57 Å** (with restraints) |
| `run_solvent_docking_demo.py` | see README for dry-vs-wet pocket comparison |
| `run_blind_docking_demo.py` | 30-particle / 20-iteration (600-frame) global search from bulk solvent: 18.69 Å → 11.94 Å RMSD, -132.08 kcal/mol final score |

The metadynamics pose-strength metric went through one iteration during
testing: the first version checked "step bias first exceeds a fixed
threshold," which always trivially fired at step 1 (a repulsive basin is
deposited at the native pose itself *before* the exploration loop starts,
so bias there is high from the very first logged step by construction —
the threshold was never actually testing anything). Replaced with "how
many steps of active repulsion before the *raw physical score* itself
turns unfavorable (> 0 kcal/mol)" — a metric that's actually sensitive to
how deep the native basin is, not to when the first hill happens to be
logged.

## 10. Verification

Full `pytest` suite run after the `pharmacophore.py` fix: **37/37 passed**
(`tests/`, ~336s). All six demo scripts run to completion at full scale
with no reduced parameters.
