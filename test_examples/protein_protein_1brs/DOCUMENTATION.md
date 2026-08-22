# Protein-Protein Docking: LightDock Study & GSO Implementation

This document records how this directory was built: researching LightDock
(github.com/lightdock/lightdock), re-implementing its core algorithm
(Glowworm Swarm Optimization) natively in `openmm_dock`, and running a real,
same-scale, same-system head-to-head against the actual LightDock package on
a real PDB benchmark complex. The result was a loss for our implementation
(30.72 Å vs. LightDock's 12.63 Å best RMSD-to-native) — this is reported
honestly below, along with the concrete, diagnosed reasons, not smoothed
over.

---

## 1. Why protein-protein, why LightDock, why 1BRS

Every prior demo in this repo (`test_examples/9z1l/`, etc.) is small-molecule
ligand docking. This was a deliberate scope jump into a structurally
different problem — two macromolecules, no ligand torsion tree, rigid-body
SE(3) search only — requested to see whether `openmm_dock`'s existing
infrastructure (SE(3) kinematics, real OpenMM nonbonded scoring) could be
extended to it by studying how an established protein-protein docking tool
actually works.

**LightDock** (Jiménez-García et al., *Bioinformatics* 2018/2020, *Nat
Commun* 2020) was chosen as the reference: a modern, actively maintained,
pip-installable (`pip install lightdock`) protein-protein/peptide/DNA docking
framework built on **Glowworm Swarm Optimization (GSO)** (Krishnanand &
Ghose, *Swarm Intelligence* 2009), pluggable scoring functions (DFIRE, PISA,
MJ3H, vdw, ...), and residue-restraint support.

**1BRS** (barnase/barstar, Guillet et al. 1993) was chosen as the test
complex: a small (108 + 87 residue), extremely well-characterized rigid
protein-protein docking benchmark (the "hello world" of the field — used in
ZDOCK, ClusPro, and most other benchmark papers). Chain **A** (barnase) was
used as the fixed receptor, chain **D** (barstar) as the mobile partner —
the standard A-D pairing.

```bash
curl -s -o 1brs.pdb "https://files.rcsb.org/download/1BRS.pdb"
python extract_chains.py   # -> barnase_receptor.pdb, barstar_ligand.pdb, native_complex_AD.pdb
```

`extract_chains.py` keeps only `ATOM` records (no waters/HETATM) for each
chain, plus a combined `native_complex_AD.pdb` (both chains, still in their
original bound crystal frame) used later as the RMSD ground truth.

---

## 2. Licensing note

LightDock is **GPLv3**. This repository has no LICENSE file (i.e., no
declared license at all). Copying LightDock's actual code in would force
copyleft on anything it touches. To avoid that entirely, `openmm_dock`'s GSO
implementation (`openmm_dock/glowworm_swarm.py`) was written from
**understanding the published algorithm and reading their source for
comprehension**, not by porting or copying their code — the same approach
already used earlier in this session for AutoDock Vina/smina
(`dock_monte_carlo_minimization` in `engine.py`).

---

## 3. How LightDock's GSO actually works

Read directly from `github.com/lightdock/lightdock`'s source
(`lightdock/gso/glowworm.py`, `lightdock/gso/algorithm.py`):

- Each **glowworm** is a rigid-body pose of the mobile partner: 3D
  translation + quaternion orientation (optionally + ANM normal-mode
  amplitudes for backbone flexibility — not implemented here, see §7), plus
  a scalar **luciferin** brightness value.
- Per simulation step:
  1. Evaluate every glowworm's score; update luciferin:
     `luciferin = (1-rho)*luciferin + gamma*score`.
  2. Each glowworm looks only at **neighbors within its own local
     vision_range** that are brighter, and probabilistically moves toward
     one (roulette-wheel selection weighted by luciferin difference).
  3. Vision range self-adapts: `vision_range += beta*(max_neighbors -
     n_neighbors_found)`, clamped to `[0, max_vision_range]`.
- Because movement only considers **local** neighbors (not a single global
  best, unlike standard PSO), the swarm naturally splits across multiple
  simultaneous local optima — useful when several binding modes are
  plausible.
- LightDock's genuinely **blind global coverage** comes from swarm
  **initialization**, not the optimizer itself: many independent swarms
  (their defaults: up to several hundred, ~100-200 glowworms each) are
  distributed evenly over the receptor's true **SASA surface** before
  running GSO independently on each.
- Default scoring is a statistical potential (DFIRE by default), not real
  physics.

---

## 4. `openmm_dock/glowworm_swarm.py` — what was built

| Piece | What it does |
|---|---|
| `GSOParameters` | rho, gamma, beta, luciferin/vision-range defaults, step sizes |
| `Glowworm` | id, trans (Å), quat (xyzw), luciferin, vision_range, energy |
| `generate_surface_swarm_centers` | Fibonacci-sphere points around the receptor's bounding sphere, offset by the mobile partner's own radius + a contact gap — a **simplification** of LightDock's true SASA-surface placement (see §7) |
| `build_protein_protein_system` | Builds a minimal OpenMM `System`: receptor atoms mass=0 (fixed), mobile-partner atoms real mass, scored via `scoring.create_combined_search_force` — the *exact same* rDock-style physics force already used for every small-molecule demo in this repo, reused unchanged. Generic per-atom `(charge, sigma, epsilon, donor, acceptor, hyd, is_lig)` parameters mean it doesn't care whether the "ligand" side is a small molecule or a whole protein chain. |
| `make_energy_fn` | `(trans, quat) -> kcal/mol` closure over a prebuilt `Context` |
| `GlowwormSwarmOptimizer` | `initialize_swarm` + `run` implementing the luciferin-update / local-neighborhood-movement loop described in §3 |

### Bugs found and fixed while building this

1. **`system.addForce(nb_force)` crashed** — `create_combined_search_force`
   returns an `RDockNonbondedForces` wrapper (bundling several
   `CustomNonbondedForce` objects), not a raw `mm.Force`. Fixed by iterating
   `nb_force.forces`.
2. **Energy at the true native bound pose was ~2.3 million kJ/mol** — with
   no bonded-pair exclusions, every covalent bond within each protein
   (~0.15 nm separation) was scored by the same steep short-range VDW/REPUL
   terms used for genuine inter-chain clashes. Fixed with
   `_distance_bonded_pairs` (simple distance-based bond perception, cutoff
   1.7 Å) generating exclusions for both partners. This is a **rigid-body
   MVP without ANM flexibility, so it's worth noting explicitly**: since
   neither partner's internal conformation ever changes under pure SE(3)
   sampling, each partner's intra-molecular self-energy is actually an
   identical additive constant across every candidate pose regardless of
   exclusions — it could never have changed which pose ranked best. The fix
   was still worth making because a multi-million-kJ/mol constant buries
   the real (order-1000 kJ/mol) inter-chain signal in a meaningless
   absolute number.
3. **LightDock 0.9.4 itself doesn't run on Python 3.12** —
   `configparser.readfp` was removed in 3.12 (deprecated since 3.2).
   `lightdock/gso/parameters.py` was patched in-place in this project's
   `.venv` (`readfp` → `read_file`, a direct drop-in replacement with
   identical semantics) purely to get their own tool running for a fair
   comparison — not a change to any file in this repository.
4. **DFIRE `KeyError: 'ARGOXT'`** — LightDock's DFIRE atom-type table
   doesn't have an entry for terminal-oxygen (OXT) atoms on arginine.
   Fixed by using `lightdock3_setup.py`'s own `--noxt` flag (removes OXT
   atoms), which is exactly what it's there for.

---

## 5. Running real LightDock (for comparison)

```bash
pip install lightdock freesasa   # -> lightdock 0.9.4, freesasa 2.2.1, prody 2.6.1, biopython 1.88

cd lightdock_run
lightdock3_setup.py barnase_receptor.pdb barstar_ligand.pdb -s 20 -g 20 --seed_points 42 --noxt
lightdock3.py setup.json 50 -s dfire -c 4
lgd_rank.py 20 50
lgd_generate_conformations.py barnase_receptor.pdb barstar_ligand.pdb gso_50.out 20 --setup ../setup.json   # (run inside each swarm_N/ dir)
```

`lightdock3_setup.py`'s own log is worth noting: it evaluated **602
candidate SASA-derived surface points**, filtered to 572 after an interior-
point check, before down-selecting to the requested 20 swarms — real,
computed molecular-surface coverage, not an approximation.

`lgd_rank.py`'s `rank_by_scoring.list` RMSD column reads `-1.000` for every
entry (no reference structure was registered during this ad hoc setup), so
RMSD-to-native was computed independently for both tools using the same
method (§6) rather than trusting either tool's internal number.

---

## 6. RMSD methodology (frame-offset correction)

LightDock's setup step recenters the **entire system** by subtracting the
receptor's own centroid — verified empirically: the processed receptor's
first atom coordinates equal the original crystal coordinates minus the
receptor centroid, exactly. `compare_rmsd.py` adds that same offset back to
LightDock's generated ligand (chain D) coordinates before comparing
directly, atom-for-atom (no Kabsch alignment needed — same atom count/order,
same underlying source file), against the untouched native crystal
coordinates in `native_complex_AD.pdb`. Our own GSO demo never leaves the
original crystal frame, so no correction is needed there.

---

## 7. Results — real, both directions

Same real structures (1BRS chains A/D), same scale (**20 swarms × 20
glowworms × 50 steps** — a small, fast comparison run, far below either
tool's recommended production scale of ~400 swarms), same RMSD method:

| | Best RMSD to native | Top-ranked-by-score RMSD |
|---|---|---|
| **Real LightDock** (DFIRE) | **12.63 Å** (swarm 6) | 39.4 Å (a *different*, worse-RMSD pose ranked #1 by score) |
| **Our GSO** (OpenMM physics) | 30.72 Å (best anywhere in final swarm) | 49.97 Å (10 top-ranked glowworms all converged to one identical pose) |

Real LightDock won this comparison. Diagnosed reasons, not excuses:

1. **Our energy function is untuned for protein-protein interfaces.**
   `create_combined_search_force`'s cutoffs, softening, and dielectric
   parameters were tuned against small-molecule-in-pocket contacts; used
   as-is here with zero reparametrization for the different contact
   geometry/area of a protein-protein interface.
2. **A real premature-convergence bug**: 10 different starting glowworms
   collapsed to the *exact same* final energy and RMSD — meaning the
   effective search diversity was smaller than the nominal 400 independent
   trials. Likely cause: the movement update in `GlowwormSwarmOptimizer.run`
   is applied **sequentially within a step** (each glowworm's neighbor
   search sees some neighbors already moved this step, others not yet),
   not the synchronous "evaluate all, then move all" LightDock actually
   uses — a simplification made for implementation speed that plausibly
   causes over-fast cascading convergence onto one basin.
3. **Cruder swarm placement.** Ours: a Fibonacci sphere around the
   receptor's bounding sphere. LightDock's: true SASA-derived surface
   points (602 candidates evaluated, filtered to 572, before down-selecting
   20) — meaningfully better coverage of the actual accessible molecular
   surface, especially for a non-spherical receptor like barnase.

What *did* work correctly, confirmed independently before the comparison
run: real OpenMM protein-protein scoring correctly favors the native bound
pose over a separated one (2436.6 vs. 2902.1 kJ/mol, after fixing the
bonded-exclusion bug); genuine all-around-receptor blind coverage (initial
swarm RMSD-to-native spanned 26.7–75.4 Å, median 55.9 Å — not seeded near
the answer); and the GSO mechanics themselves (luciferin update, adaptive
vision range, local-neighborhood movement) ran without error across all 50
steps. The infrastructure is real and working; it is currently a weaker
*searcher* than a decade-mature, purpose-tuned tool, which is the expected
outcome for a same-session build, not a surprising one.

---

## 8. Known limitations / what would need work to close the gap

- **No ANM normal-mode flexibility** for either partner (LightDock's `-anm`
  flag) — both partners are treated as fully rigid.
- **Synchronous update**: fix the sequential-move-order issue in
  `GlowwormSwarmOptimizer.run` identified in §7.2 — evaluate/update
  luciferin for the whole swarm first, snapshot positions, then move
  everyone against that snapshot.
- **True surface-based swarm placement**: replace the Fibonacci-sphere
  approximation with a real SASA calculation (the `freesasa` package is
  already installed in this environment).
- **Interface-specific scoring weights**: re-tune `ScoreWeights` (or add a
  dedicated protein-protein weight profile) against a proper benchmark set
  rather than reusing the small-molecule defaults unchanged.
- **Clustering/reranking**: LightDock's real pipeline clusters nearby poses
  (`lgd_cluster_bsas.py`) and reports cluster representatives, not raw
  per-glowworm ranks — our demo reports raw glowworms only.

## Files in this directory

| File | Purpose |
|---|---|
| `1brs.pdb` | Raw RCSB download (all 6 chains) |
| `extract_chains.py` | Splits out chain A/D + builds `native_complex_AD.pdb` |
| `barnase_receptor.pdb`, `barstar_ligand.pdb` | Extracted receptor/ligand chains |
| `native_complex_AD.pdb` | Ground-truth bound complex (untouched crystal frame) |
| `run_gso_docking_demo.py` | Our own GSO protein-protein docking demo |
| `compare_rmsd.py` | Frame-offset-corrected RMSD calculator (works for either tool's output) |
| `lightdock_run/` | Real LightDock setup + simulation working directory |
| `lightdock_sim.log` | Real LightDock's simulation stdout/stderr |
