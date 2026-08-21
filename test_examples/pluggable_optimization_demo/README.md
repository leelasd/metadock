# Pluggable Scoring + Bayesian Optimization + Gradient Kinematic Polish

This example demonstrates three new capabilities added to `openmm_dock` after a
comparative review against [OpenDock](https://github.com/guyuehuo/opendock)
(Hu et al., *Bioinformatics* 2024) — see [`docs/suggested_improvements.md`](../../docs/suggested_improvements.md)
for the earlier PSO/Metadynamics/Forward-Kinematics comparison this one follows up on.

OpenDock's headline strengths that this codebase didn't previously have an
equivalent for: (1) a pluggable scoring-function interface that composes a
base score with weighted correction terms (its `OnionNet-SFCT` ML correction
is the flagship example), (2) a `BayesianOptimizerSampler`, and (3)
gradient-based minimizers (Adam/SGD/LBFGS) operating directly on the reduced
pose/torsion parameterization rather than only in full-Cartesian space. This
example wires up the equivalents added here — `openmm_dock/scoring_function.py`,
`openmm_dock/bayesian_optimizer.py`, and `openmm_dock/gradient_minimizer.py` —
on the real Keap1 + Q9E macrocycle system (PDB 6Z6A).

## What this demo does *not* claim

This is a demonstration of the three new pieces of machinery working
correctly together, **not** a new full-strength docking engine — it searches
only the 6D rigid-body pose (translation + rotation), holding ring/exocyclic
torsions and receptor side chains fixed, using a deliberately simple
two-term composite score. It reliably improves a perturbed pose by orders of
magnitude in score, but shouldn't be expected to redock to sub-angstrom
accuracy the way the full kinematic PSO/metadynamics engines elsewhere in
this repo do — those remain the tools to reach for when accuracy matters.

## 1. Pluggable scoring (`scoring_function.py`)

```python
physical = OpenMMPhysicalScore(engine.context, engine.engine._full_positions_from_coords,
                                lig_start=engine.lig_start, weight=1.0)
depth_beacon = PocketDepthBeacon(cv_calc=GeneralizedCVEngine(pocket_center=POCKET_CENTER), weight=2.0)
combo = CompositeScoringFunction([physical, depth_beacon])
```

`OpenMMPhysicalScore` wraps this codebase's existing OpenMM `Context` +
`_full_positions_from_coords` pattern as a `BaseScoringFunction`.
`PocketDepthBeacon` (defined in `run_pluggable_optimization_demo.py`) is a
~10-line custom scorer following the pattern documented in
`scoring_function.py`'s module docstring — the same shape a future ML
rescoring term (an OnionNet-SFCT-style correction) would take. Both are
summed by `CompositeScoringFunction`, each carrying its own `.weight`, and
`combo.breakdown(coords)` reports the per-term contributions.

## 2. Bayesian optimization (`bayesian_optimizer.py`)

A self-contained Gaussian-Process (Matern-5/2 kernel, numpy/scipy only — no
scikit-learn dependency) + Expected-Improvement acquisition searches the 6D
pose space around a randomly perturbed starting pose. It typically needs far
fewer objective evaluations than a PSO/GA swarm for a search this
low-dimensional (this run: 8 initial + 15 sequential = 23 evaluations).

## 3. Gradient-based kinematic polish (`gradient_minimizer.py`)

The best pose Bayesian optimization finds is locally polished with
`lbfgs_minimize`, which drives `scipy.optimize.minimize(method="L-BFGS-B")`
using central finite-difference gradients of the *same* composite scoring
function — directly in the 6D kinematic space, not OpenMM's own
Cartesian-space `LocalEnergyMinimizer`.

## Running it

```bash
python run_pluggable_optimization_demo.py
```

Typical output (see the script for exact numbers, which vary slightly with
the perturbation seed):

```
Crystal pose composite score : 158.20
Perturbed start composite score: 1048967.79  (trans off by 3.00 A)

[*] Running Bayesian optimization over 6D rigid-body pose space...
    Bayesian opt best score: 134.99  (23 objective evaluations)

[*] Polishing best Bayesian-opt pose with L-BFGS-B (finite-difference gradients)...
    Polished score: 21.33  (converged=False, 1625 evaluations)
```

Saves the final pose to `best_pose.sdf` with `COMPOSITE_SCORE` and
`RMSD_TO_XTAL_A` properties.
