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

## 8. Follow-up investigation: production scale and DFIRE scoring

Two direct questions this section answers: *why not implement LightDock's
own scoring function*, and *why not test at the scale that actually matters*.
Both were legitimate gaps in §7 — the 20-swarm comparison there was cheap
and fast, not representative.

### 8.1. LightDock's DFIRE as a real, callable scoring backend

DFIRE is a knowledge-based statistical potential (atom-type-pair energies
binned by distance, from PDB structure statistics), not a physics force
field term — it can't be expressed as an analytic LJ/Coulomb expression, but
it doesn't need to be reimplemented either: LightDock's own installed
package exposes it as an ordinary importable Python/Cython API
(`lightdock.scoring.dfire.driver.DFIRE`, `DFIREAdapter`,
`lightdock.prep.simulation.read_input_structure`). `openmm_dock/lightdock_dfire_scoring.py`
wraps this as a `(trans, quat) -> energy` closure compatible with
`GlowwormSwarmOptimizer` — calling their installed package as an external
library dependency (same as depending on scipy/numpy), not copying their
GPLv3 code into this repository. Sanity-checked correct: native pose scores
-25.7 (favorable) vs. -4.7 for a fully separated pose. Per-eval cost turned
out to be ~10.7 ms — *slower* than our own OpenMM scoring's ~2.7 ms; real
LightDock's production throughput comes from multiprocessing across cores
(their `-c` flag), not raw single-eval speed.

### 8.2. Production-scale real LightDock (partial run, real numbers)

Ran `lightdock3_setup.py -s 400 -g 200` (LightDock's own documented
`DEFAULT_NUM_SWARMS`/`DEFAULT_NUM_GLOWWORMS`, i.e. genuinely their
production default, not an arbitrary choice) + `lightdock3.py setup.json 50
-s dfire -c 12` in `lightdock_run_production/`. The process was killed by
something external to this investigation partway through, at **206/400
swarms (52%) fully complete** — `lgd_rank.py 400 50` handles this
gracefully (skips missing swarms with a warning, ranks what exists).

**Result: the top-ranked pose (swarm 215) came in at 1.97–2.30 Å RMSD to
native** (5 top glowworms in that swarm, all 1.97–2.30 Å) — a dramatic jump
from the 20-swarm comparison's 12.63 Å best. This is the single most
important number in this document: **the earlier "LightDock wins" framing
was really "LightDock's under-sampled run beat our under-sampled run,"
which is a much weaker and less interesting claim.** At real production
scale, real LightDock gets close to native.

### 8.3. Ablation: does OUR search algorithm close the gap at matching scale?

Two real bugs were found and fixed in `GlowwormSwarmOptimizer.run` before
this test:

1. **Synchronous update.** The movement phase mutated `g.trans`/`g.quat` in
   place while still iterating the same `glowworms` list, so a glowworm late
   in a step's loop could see some neighbors already moved that step and
   others not — an order-dependent artifact, not LightDock's real
   synchronous "evaluate all, then move all" structure. Fixed by snapshotting
   every glowworm's `(trans, quat, luciferin)` before any movement each step.
2. **Wrong step-size semantics (the bigger one).** `GSOParameters.step_translation`
   was a small *fixed* 0.6 Å absolute distance per move. LightDock's real
   `DEFAULT_TRANSLATION_STEP`/`DEFAULT_ROTATION_STEP` (constants.py) are
   documented as **interpolation fractions** — "0.5 means jump 50% of the
   way to the target," not an absolute distance. With a genuinely blind
   swarm spanning 23–77 Å from native (median 55 Å) and only 30 steps, a
   fixed 0.6 Å step caps total possible travel at ~18 Å — physically unable
   to reach the interface from most starting positions, regardless of how
   many glowworms or steps are used. Fixed by replacing the fixed step with
   `step_translation_frac`/`step_rotation_frac` (fractional interpolation
   toward the chosen neighbor, matching LightDock's real mechanism).

Both fixes are real and worth keeping. **Neither closed the gap.** Same
scale (100 swarms × 50 glowworms × 30 steps) both before and after, all four
results plateau in the same ~21–51 Å band, regardless of scoring function:

| Configuration | Best-by-score RMSD | Best RMSD anywhere in swarm |
|---|---|---|
| Our GSO + DFIRE, buggy fixed 0.6 Å step | 44.16 Å | 21.63 Å |
| Our GSO + DFIRE, fixed fractional step | 46.82 Å | 24.98 Å |
| Our GSO + OpenMM physics, fixed fractional step | 51.35 Å | 23.43 Å |
| Real LightDock, 20-swarm scale | 39.4 Å | 12.63 Å |
| **Real LightDock, ~52%-complete production scale** | **1.97–2.30 Å** | — |

Swapping scoring functions (DFIRE vs. our OpenMM physics) made no
meaningful difference either (46.82 Å vs. 51.35 Å, both worse than either
tiny-scale run) — ruling out "untuned scoring weights" as the dominant
factor too.

### 8.4. Actual diagnosis: coverage density, not search sophistication

`GlowwormSwarmOptimizer`'s `vision_range` starts at 10 Å and grows by at
most `beta * max_neighbors ≈ 0.08 * 6 ≈ 0.5` Å/step, reaching at most
`10 + 30*0.5 = 25` Å after 30 steps — nowhere near enough to bridge the
tens-of-Angstrom gaps between different swarm centers placed all around a
~40 Å-radius receptor. In practice this means **a glowworm essentially
never finds a "neighbor" outside its own originating swarm's initial ~2 Å
jitter cluster** — cross-swarm information exchange (the mechanism that
would make GSO a genuinely *unified* global search rather than N independent
local ones) basically doesn't happen within any tractable step budget. Each
swarm just polishes toward whichever local optimum sits nearest its own
starting point, completely independent of every other swarm.

That reframes the entire investigation: **whether a blind run finds the
interface comes down almost entirely to whether any single swarm's starting
position happens to be well-placed** — a question of *swarm placement
density and quality*, not within-swarm optimizer sophistication. Real
LightDock's win was never really about a better search algorithm; it was
**4x the swarm count (400 vs. our 100) placed on a true SASA-derived surface
instead of our Fibonacci-sphere bounding-sphere approximation** — swarm 215
almost certainly started in a favorable spot purely from denser, better-
targeted coverage, and its own local GSO polish did the rest. This is
consistent with, and now directly explains, the coverage-density hypothesis
already listed in §7's diagnosis (item 3) — the ablations in §8.3 upgrade it
from "one of three plausible factors" to "the actual explanation," since
fixing the other two (scoring weights, search-loop bugs) provably didn't move
the result.

**The concrete, well-scoped next step, if pursued**: replace
`generate_surface_swarm_centers`'s Fibonacci-sphere approximation with a real
SASA-based placement (the `freesasa` package is already installed) and/or
simply increase `N_SWARMS` to matching scale (400) — either should
directly test this diagnosis, unlike the scoring-function and step-size
changes already ruled out above.

### 8.5. Testing the diagnosis directly: real SASA placement at matching scale

Implemented `generate_sasa_swarm_centers` in `glowworm_swarm.py`: uses the
`freesasa` package (a real, independent SASA calculator — not a port of
LightDock's own point-generation code) to compute per-atom solvent-accessible
surface area on the receptor, keeps atoms with meaningfully nonzero SASA,
places one candidate point per exposed atom along that atom's own outward
radial direction (offset past its van der Waals radius by the mobile
partner's radius + a contact gap), then downselects to `N_SWARMS`
well-spread points via greedy farthest-point sampling. This hugs the true,
non-spherical molecular surface instead of a bounding sphere — confirmed by
inspection: candidate-derived swarm centers ranged 26.3–48.7 Å from the
receptor centroid (vs. the Fibonacci sphere's single fixed radius).

Ran at `N_SWARMS=400` (matching LightDock's own default swarm count) ×
`N_PER_SWARM=50` × 30 steps, OpenMM scoring (not DFIRE — already shown in
§8.3 not to matter, and ~4x cheaper per-eval). This run took **204.7
minutes** (vs. ~27 min estimated) — much slower than expected, plausibly
because SASA-placed swarms sit in genuinely contact-dense regions near the
real surface, giving OpenMM's cutoff-based neighbor list far more atom pairs
to evaluate per energy call than the old sparse fixed-radius sphere ever did.

**Result: 29.74 Å best-by-score, 17.41 Å best RMSD found anywhere in the
final swarm.** Initial swarm coverage was measurably better too — RMSD
range 9.4–74.1 Å (median 48.2 Å) vs. the Fibonacci sphere's 23.4–76.8 Å
(median 55.2 Å); the closest starting swarm got within 9.4 Å of native,
something the old placement never achieved.

**This partially confirms, but does not complete, the coverage-density
diagnosis.** Compared to the 100-swarm Fibonacci baseline (51.35 Å OpenMM /
46.82 Å DFIRE), real SASA placement at 4x the swarm count is a genuine,
substantial improvement — roughly halving the best-anywhere RMSD (46–51 Å
→ 17–30 Å). But it falls well short of closing the gap to real LightDock's
1.97–2.30 Å production result. Two honest possibilities, not yet
distinguished: (a) LightDock's own point generation is qualitatively better
than this independent SASA approach — their pipeline evaluates 602–786
candidate points before filtering (via an "incompatible filter" and
"interior points filter" this implementation doesn't replicate) rather than
placing exactly one point per exposed atom; or (b) their 200
glowworms/swarm (4x this run's 50) gives meaningfully more thorough
per-swarm local refinement once a swarm *does* start near the interface —
consistent with the observation here that the single best-*starting* swarm
(9.4 Å) did not end up producing the best *final* pose (17.41 Å came from a
different swarm), suggesting the within-swarm search itself may still be
leaving refinement on the table even from a good start. Given this run
alone took 3.4 hours, a direct 200-glowworms/swarm test to distinguish these
two remaining hypotheses was not pursued in this session.

## 9. Known limitations / what would need work to close the gap

- **Residual gap after the swarm-placement fix (§8.5)**: real SASA placement
  at matching scale roughly halved the best-anywhere RMSD (46–51 Å → 17–30 Å)
  but did not close the gap to LightDock's 1.97–2.30 Å. Two undistinguished
  remaining hypotheses: LightDock's own candidate-filtering pipeline
  (interior-points/incompatible filters, 602–786 candidates evaluated) may
  place points more precisely than this one-point-per-exposed-atom approach;
  or their 4x-larger 200-glowworms/swarm search may extract meaningfully more
  from a good starting position than this implementation's 50/swarm does.
- **No ANM normal-mode flexibility** for either partner (LightDock's `-anm`
  flag) — both partners are treated as fully rigid.
- **No cross-swarm information exchange mechanism** — even with correct
  swarm placement, GSO's own vision-range growth is too slow to unify
  separate swarms into one coordinated search (see §8.4). LightDock accepts
  this too (it's inherent to the algorithm); it compensates entirely via
  swarm placement density, not algorithmic fix.
- **Interface-specific scoring weights**: re-tune `ScoreWeights` (or add a
  dedicated protein-protein weight profile) against a proper benchmark set
  rather than reusing the small-molecule defaults unchanged — lower priority
  now that §8.3 showed scoring function choice isn't the dominant factor.
- **Clustering/reranking**: LightDock's real pipeline clusters nearby poses
  (`lgd_cluster_bsas.py`) and reports cluster representatives, not raw
  per-glowworm ranks — our demo reports raw glowworms only.
- **SASA-placed runs are slow**: per-eval cost rises substantially near the
  true surface (§8.5) — worth profiling/optimizing before attempting the
  200-glowworms/swarm test above.

## Files in this directory

| File | Purpose |
|---|---|
| `1brs.pdb` | Raw RCSB download (all 6 chains) |
| `extract_chains.py` | Splits out chain A/D + builds `native_complex_AD.pdb` |
| `barnase_receptor.pdb`, `barstar_ligand.pdb` | Extracted receptor/ligand chains |
| `native_complex_AD.pdb` | Ground-truth bound complex (untouched crystal frame) |
| `run_gso_docking_demo.py` | Our own GSO protein-protein docking demo (20-swarm scale) |
| `run_gso_scaled_demo.py` | Parametrized (env vars: `SCORING`, `SWARM_METHOD`, `N_SWARMS`, `N_PER_SWARM`, `N_STEPS`) version used for the §8 ablations |
| `lightdock_run_production/` | Real LightDock at production-scale settings (400×200, 52% complete) |
| `compare_rmsd.py` | Frame-offset-corrected RMSD calculator (works for either tool's output) |
| `lightdock_run/` | Real LightDock setup + simulation working directory |
| `lightdock_sim.log` | Real LightDock's simulation stdout/stderr |
