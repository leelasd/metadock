"""
Phase 4 of the grid-scoring plan: empirical comparison of pairwise vs.
AutoDock-style grid-based nonbonded scoring (openmm_dock.gridding /
scoring.create_grid_search_force) in dock_simulated_annealing's inner search
loop, on the same "blind SA" test case used earlier in this project (cavity
centered on the crystal ligand's own center of mass, radius 10 A -- pocket
location known, pose unknown, per the project's own definition of "blind").

Three things are measured, deliberately kept separate:
  1. The one-time grid-cache build cost (paid once per DockingEngine
     instance, amortized across every subsequent dock_* call -- see
     DockingEngine._ensure_grid_cache).
  2. Wall-clock at *equal* candidate-evaluation counts (same n_runs /
     anneal_steps / steps_per_temp) for pairwise vs. grid, to isolate
     "grid is faster per candidate" from "a bigger budget helps."
  3. Whether the per-candidate speedup measured in (2), reinvested as a
     larger search budget under the grid backend, gets blind pose recovery
     under the project's standing 2.0 A bar.
"""
import time
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock import DockingEngine, CavityDefinition
from openmm_dock.core import SDFParser

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "test_examples" / "score"

rec_path = EXAMPLES_DIR / "receptor.mol2"
lig_path = EXAMPLES_DIR / "xtal-lig.sd"

ref_mol = SDFParser.load_molecules(lig_path)[0]
conf_r = ref_mol.GetConformer()
heavy = [a.GetIdx() for a in ref_mol.GetAtoms() if a.GetAtomicNum() > 1]
p_ref = np.array([conf_r.GetAtomPosition(i) for i in heavy])
all_coords = np.array([conf_r.GetAtomPosition(i) for i in range(ref_mol.GetNumAtoms())])
com = all_coords.mean(axis=0)

cavity = CavityDefinition(center=com, radius=10.0, min_coords=com - 10.0, max_coords=com + 10.0, name="blind_com10")


def rmsd(mol: Chem.Mol) -> float:
    conf = mol.GetConformer()
    p = np.array([conf.GetAtomPosition(i) for i in heavy])
    return float(np.sqrt(np.mean(np.sum((p - p_ref) ** 2, axis=1))))


def best_of(results):
    return min(results, key=lambda r: rmsd(r.mol))


print("=" * 90)
print("  PHASE 4: GRID vs PAIRWISE SCORING -- BLIND SA (cavity = crystal-ligand COM, radius 10 A)")
print("=" * 90)

SA_KWARGS = dict(n_runs=10, anneal_steps=10, steps_per_temp=100, seed=7)

# --------------------------------------------------------------------------
# 1. One-time grid-cache build cost (measured directly, on its own engine).
# --------------------------------------------------------------------------
engine_grid = DockingEngine(receptor_path=rec_path, cavity=cavity, platform_name="CPU")
mol = SDFParser.load_molecules(lig_path)[0]
lig_sys = SDFParser.mol_to_system(mol)
required_types = {a.element.upper() for a in lig_sys.atoms}

t0 = time.perf_counter()
engine_grid._ensure_grid_cache(required_types)
t_cache = time.perf_counter() - t0
print(f"\n[1] One-time grid cache build ({sorted(required_types)}): {t_cache:.1f}s (amortized over every later dock_* call)")

# --------------------------------------------------------------------------
# 2. Equal-budget comparison: pairwise vs grid, same SA_KWARGS.
# --------------------------------------------------------------------------
print(f"\n[2] Equal-budget comparison ({SA_KWARGS})")

engine_pairwise = DockingEngine(receptor_path=rec_path, cavity=cavity, platform_name="CPU")
engine_pairwise._get_grid_search_forces = lambda lig_sys: None  # force pairwise fallback for this baseline

t0 = time.perf_counter()
results_pw = engine_pairwise.dock_simulated_annealing(mol, **SA_KWARGS)
t_pw = time.perf_counter() - t0
best_pw = best_of(results_pw)
print(f"  Pairwise: {t_pw:.1f}s | best RMSD = {rmsd(best_pw.mol):.2f} A | best score = {best_pw.score:.2f} kcal/mol")

t0 = time.perf_counter()
results_grid = engine_grid.dock_simulated_annealing(mol, **SA_KWARGS)
t_grid = time.perf_counter() - t0
best_grid = best_of(results_grid)
speedup = t_pw / t_grid if t_grid > 0 else float("inf")
print(f"  Grid:     {t_grid:.1f}s | best RMSD = {rmsd(best_grid.mol):.2f} A | best score = {best_grid.score:.2f} kcal/mol "
      f"(cache already warm on this engine)")
print(f"  --> per-candidate speedup at equal budget: {speedup:.2f}x")

# --------------------------------------------------------------------------
# 3. Reinvest the speedup as a bigger grid-backed budget; does it recover
#    the pose under the project's 2.0 A bar?
# --------------------------------------------------------------------------
scale = max(1, int(round(speedup)))
big_kwargs = dict(SA_KWARGS)
big_kwargs["n_runs"] = SA_KWARGS["n_runs"] * scale
big_kwargs["seed"] = 11
print(f"\n[3] Reinvested budget under grid scoring: n_runs {SA_KWARGS['n_runs']} -> {big_kwargs['n_runs']} ({scale}x, from measured speedup)")

t0 = time.perf_counter()
results_big = engine_grid.dock_simulated_annealing(mol, **big_kwargs)
t_big = time.perf_counter() - t0
best_big = best_of(results_big)
under_bar = rmsd(best_big.mol) < 2.0
print(f"  Grid (bigger budget): {t_big:.1f}s | best RMSD = {rmsd(best_big.mol):.2f} A | best score = {best_big.score:.2f} kcal/mol "
      f"| <2.0 A recovered: {under_bar}")

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"{'Stage':<45} | {'Wall time':<10} | {'Best RMSD (A)':<15}")
print("-" * 90)
print(f"{'Grid cache build (one-time)':<45} | {t_cache:<9.1f}s | {'n/a':<15}")
print(f"{'Pairwise, equal budget':<45} | {t_pw:<9.1f}s | {rmsd(best_pw.mol):<15.2f}")
print(f"{'Grid, equal budget':<45} | {t_grid:<9.1f}s | {rmsd(best_grid.mol):<15.2f}")
print(f"{'Grid, ' + str(scale) + 'x budget':<45} | {t_big:<9.1f}s | {rmsd(best_big.mol):<15.2f}")
print("=" * 90)
