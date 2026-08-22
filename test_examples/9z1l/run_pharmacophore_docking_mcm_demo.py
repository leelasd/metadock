"""
Monte-Carlo-with-Minimization (MCM / basin-hopping) Pharmacophore Docking on
PDB 9Z1L -- learning directly from AutoDock Vina/smina's actual search loop
(cloned from github.com/mwojcikowski/smina and read src/lib/monte_carlo.cpp,
mutate.cpp, quasi_newton.cpp, bfgs.h).

The mechanistic difference from every earlier script in this directory:
dock_simulated_annealing takes many small-Gaussian-jitter Metropolis steps
and only periodically (every lamarck_interval moves) locally minimizes, so
most visited states are raw, unminimized proposal energies. Vina/smina's
monte_carlo::single_run does the opposite every single step: ONE coarse
single-DOF jump (mutate_conf) -> immediate local BFGS minimization
(quasi_newton, analytic gradients in their C++) -> Metropolis-accept the
MINIMIZED energy. Every state the Markov chain actually compares is already
a converged local-basin minimum -- the search explores BASINS, not noisy
raw conformations, which is the likely reason Vina-family tools reliably
land closer to native poses.

openmm_dock.engine.dock_monte_carlo_minimization (new this session)
replicates that recipe: mutate_chromosome_vina_style (picks ONE of
{translation, rotation, torsion_i} and applies one full-amplitude move, like
smina's mutate_conf) + gradient_minimizer.lbfgs_minimize (finite-difference
L-BFGS-B, the numerical counterpart to smina's analytic-gradient BFGS) in
place of dock_simulated_annealing's small-jitter-then-periodic-polish loop.
No separate fine-refinement stage is needed here (unlike the SA-based
scripts): every MCM run already ends with its own full "authentic" local
minimize, so the returned pose is already basin-converged.
"""
import os
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from openmm_dock.cavity import CavityDefinition
from openmm_dock.engine import DockingEngine
from openmm_dock.core import SDFParser

DEMO_DIR = Path(__file__).resolve().parent
N_CONFORMERS = 25
CAVITY_RADIUS = float(os.environ.get("CAVITY_RADIUS", "15.0"))
FLEX_RADIUS = os.environ.get("FLEX_RADIUS", "6.0")
FLEX_RADIUS = float(FLEX_RADIUS) if FLEX_RADIUS else None

print("=" * 95)
print("   OPENMM-DOCK: MONTE-CARLO-WITH-MINIMIZATION (VINA/SMINA-STYLE) PHARMACOPHORE DOCKING (9Z1L)")
print(f"   [Settings: cavity_radius={CAVITY_RADIUS} Å, flexible_radius={FLEX_RADIUS}]")
print("=" * 95)

cavity = CavityDefinition.from_prm_file(DEMO_DIR / "cavity.prm")
if CAVITY_RADIUS != cavity.radius:
    cavity = CavityDefinition(
        center=cavity.center, radius=CAVITY_RADIUS,
        min_coords=cavity.center - CAVITY_RADIUS, max_coords=cavity.center + CAVITY_RADIUS,
        name=cavity.name,
    )
xtal_mol = SDFParser.load_molecules(DEMO_DIR / "a1czz_crystal_pose.sdf")[0]
xtal_coords = xtal_mol.GetConformer().GetPositions()
xtal_center = xtal_coords.mean(axis=0)


def rmsd_to_xtal(mol: Chem.Mol) -> float:
    coords = mol.GetConformer().GetPositions()
    return float(np.sqrt(np.mean(np.sum((coords - xtal_coords) ** 2, axis=1))))


def generate_diverse_conformers(mol: Chem.Mol, n_confs: int, center: np.ndarray, seed: int = 42) -> list:
    mol_work = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol_work, numConfs=n_confs, params=params)
    confs = []
    for cid in cids:
        conf = mol_work.GetConformer(cid)
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_work.GetNumAtoms())])
        coords = coords - coords.mean(axis=0) + center
        c_mol = Chem.Mol(mol)
        c_conf = c_mol.GetConformer()
        for i in range(c_mol.GetNumAtoms()):
            c_conf.SetAtomPosition(i, Point3D(*[float(x) for x in coords[i]]))
        confs.append(c_mol)
    if not confs:
        confs = [Chem.Mol(mol)]
    return confs


def select_diverse_top_k(results: list, k: int = 5, min_pairwise_rmsd: float = 1.5) -> list:
    def pose_dist(a, b):
        ca = a.mol.GetConformer().GetPositions()
        cb = b.mol.GetConformer().GetPositions()
        return float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1))))

    ranked = sorted(results, key=lambda r: r.score)
    selected = [ranked[0]]
    for cand in ranked[1:]:
        if len(selected) >= k:
            break
        if all(pose_dist(cand, s) >= min_pairwise_rmsd for s in selected):
            selected.append(cand)
    if len(selected) < k:
        selected_ids = {id(s) for s in selected}
        for cand in ranked:
            if len(selected) >= k:
                break
            if id(cand) not in selected_ids:
                selected.append(cand)
                selected_ids.add(id(cand))
    return selected


print(f"\n[1] Generating {N_CONFORMERS} diverse ETKDGv3 conformers, recentered to the pocket...")
diverse_conformers = generate_diverse_conformers(xtal_mol, n_confs=N_CONFORMERS, center=xtal_center, seed=7)
print(f"    Generated {len(diverse_conformers)} conformers.")

print(f"\n[2] Monte-Carlo-with-Minimization docking (Vina/smina-style basin-hopping), "
      f"1 MCM run x {len(diverse_conformers)} conformers...")
engine = DockingEngine(
    receptor_path=DEMO_DIR / "receptor.mol2", cavity=cavity, pharma_restr_path=DEMO_DIR / "pharma.restr",
    flexible_radius=FLEX_RADIUS,
)

all_results = []
for c_idx, conf_mol in enumerate(diverse_conformers):
    conf_results = engine.dock_monte_carlo_minimization(
        conf_mol, n_runs=1, num_steps=10, lbfgs_maxiter=6, lbfgs_maxiter_final=20,
        seed=1000 + c_idx,
    )
    all_results.extend(conf_results)
    r = conf_results[0]
    print(f"    Conformer {c_idx + 1}/{len(diverse_conformers)}: Score = {r.score:.3f} "
          f"(Restraint = {r.scores['SCORE.RESTR.PHARMA']:.2f}) | RMSD to Xtal = {rmsd_to_xtal(r.mol):.2f} Å")

all_results.sort(key=lambda r: r.score)
print(f"\n[*] All {len(all_results)} MCM runs ranked by score (top 15):")
for rank, r in enumerate(all_results[:15]):
    print(f"    #{rank + 1}: Score = {r.score:.3f} (Restraint = {r.scores['SCORE.RESTR.PHARMA']:.2f}) "
          f"| RMSD to Xtal = {rmsd_to_xtal(r.mol):.2f} Å")

best = all_results[0]
diverse_top5 = select_diverse_top_k(all_results, k=5, min_pairwise_rmsd=1.5)

print("\n" + "=" * 80)
print("FINAL RESULT: Monte-Carlo-with-Minimization (Vina/smina-style) Pharmacophore Docking")
print(f"  Best-scoring pose : {rmsd_to_xtal(best.mol):.2f} Å RMSD to crystal (Score {best.score:.2f})")
print("  Top-5 diverse candidates:")
for i, c in enumerate(diverse_top5):
    print(f"    #{i + 1}: Score = {c.score:.2f} (Restraint = {c.scores['SCORE.RESTR.PHARMA']:.2f}) "
          f"| RMSD to Xtal = {rmsd_to_xtal(c.mol):.2f} Å")
print("=" * 80)

tag = f"cav{CAVITY_RADIUS:g}_flex{FLEX_RADIUS if FLEX_RADIUS else 'none'}"
top5_name = f"pharma_dock_mcm_top5_{tag}_out.sdf"
best_name = f"pharma_dock_mcm_best_{tag}_out.sdf"
w = Chem.SDWriter(str(DEMO_DIR / top5_name))
for r in diverse_top5:
    w.write(r.mol)
w.close()
w_best = Chem.SDWriter(str(DEMO_DIR / best_name))
w_best.write(best.mol)
w_best.close()
print(f"\n[✓] Saved {top5_name}, {best_name}")
