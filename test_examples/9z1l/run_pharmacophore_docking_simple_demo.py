"""
Simple Diverse-Conformer + Local-SA Pharmacophore Docking on PDB 9Z1L.

Companion strategy pivot to run_blind_docking_simple_demo.py -- see that
file's docstring for the full rDock comparison rationale. The prior 4-stage
pharmacophore script (run_pharmacophore_docking_demo.py) staged soft-VDW
search -> best-of-5-BY-SCORE selection -> full-VDW polish -> fine refine,
and diagnostics showed each of those stages was making RMSD *worse*: the
score's local minimum was anti-correlated with RMSD in this region, and the
top-5-by-score cutoff was silently discarding the actual closest poses
(ranked #6/#7 by score) before they ever reached polishing.

This script drops all of that staging. dock_simulated_annealing's GUIDED
branch (active whenever pharma_points are set) already aligns each input
conformer to the pharmacophore restraints once via align_ligand_to_pharmacophore,
then anneals with restraints active AND full VDW strength for the entire run
(never softened), including its own built-in periodic Lamarckian local
minimization (lamarck_interval/lamarck_iterations) -- there is no separate
polish stage that could introduce a second, different local optimum. The
only extension: run it across many diverse ETKDGv3 conformers instead of
one, so different starting torsion basins get a fair, independent, full-
strength SA run each -- and just report the real top-5 by score directly
from that pool, with no additional funneling/re-scoring step to distort it.
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
N_CONFORMERS = 30
N_RUNS_PER_CONFORMER = 2
# Receptor-flexibility / search-space knobs, overridable via env vars so the
# same script can be re-run with different settings without editing code:
# CAVITY_RADIUS: the flat-bottom cavity restraint's radius from pocket center
#   (Angstroms) -- zero force *inside* this radius, so 15.0 (the original
#   value) is a 30 A-diameter free-roam zone with no gradient pulling stray
#   poses back, even though the pocket center is already known input (same
#   information rDock's REF_MOL cavity mapping uses).
# FLEX_RADIUS: if set, receptor atoms within this many Angstroms of the
#   pocket center get real mass and can move (backbone tethered by a strong
#   harmonic restraint, sidechains free) -- see DockingEngine(flexible_radius=).
#   None (default) = fully rigid receptor, which is what every result so far
#   this session used.
CAVITY_RADIUS = float(os.environ.get("CAVITY_RADIUS", "15.0"))
FLEX_RADIUS = os.environ.get("FLEX_RADIUS")
FLEX_RADIUS = float(FLEX_RADIUS) if FLEX_RADIUS else None

print("=" * 95)
print("   OPENMM-DOCK: SIMPLE DIVERSE-CONFORMER + LOCAL-SA PHARMACOPHORE DOCKING (PDB 9Z1L)")
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
    """ETKDGv3 diverse conformer ensemble, recentered to the pocket so the
    pharmacophore alignment step (which is rigid-body, MCS/feature-based)
    starts from a sane neighborhood rather than wherever RDKit embedded it."""
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

print(f"\n[2] Pharmacophore-restrained Simulated Annealing, FULL VDW strength throughout "
      f"(no soft-VDW/polish staging), {N_RUNS_PER_CONFORMER} runs x {len(diverse_conformers)} conformers "
      f"= {N_RUNS_PER_CONFORMER * len(diverse_conformers)} total SA runs...")
engine = DockingEngine(
    receptor_path=DEMO_DIR / "receptor.mol2", cavity=cavity, pharma_restr_path=DEMO_DIR / "pharma.restr",
    flexible_radius=FLEX_RADIUS,
)

all_results = []
for c_idx, conf_mol in enumerate(diverse_conformers):
    conf_results = engine.dock_simulated_annealing(
        conf_mol, n_runs=N_RUNS_PER_CONFORMER,
        t_high=400.0, t_low=2.0, anneal_steps=15, steps_per_temp=150,
        seed=1000 + c_idx,
    )
    all_results.extend(conf_results)

all_results.sort(key=lambda r: r.score)
print(f"\n[*] All {len(all_results)} SA runs ranked by score (top 15):")
for rank, r in enumerate(all_results[:15]):
    print(f"    #{rank + 1}: Score = {r.score:.3f} (Restraint = {r.scores['SCORE.RESTR.PHARMA']:.2f}) "
          f"| RMSD to Xtal = {rmsd_to_xtal(r.mol):.2f} Å")

best = all_results[0]
# Widened to top-15, same reasoning as the blind-docking companion script:
# a top-5 cutoff risks silently dropping the actual closest poses when
# score and RMSD aren't perfectly correlated in this region.
diverse_candidates = select_diverse_top_k(all_results, k=15, min_pairwise_rmsd=1.5)
print(f"\n[3] Fine-grained local refinement (small moves, low-temperature-only) of "
      f"{len(diverse_candidates)} diverse candidates from the pool above...")
fine_results = []
for cand_idx, cand in enumerate(diverse_candidates):
    cand_fine = engine.dock_simulated_annealing(
        cand.mol,
        n_runs=3,
        t_high=40.0,
        t_low=1.0,
        anneal_steps=15,
        steps_per_temp=200,
        trans_sigma=0.15,
        rot_sigma=3.0,
        torsion_sigma=5.0,
        seed=2000 + cand_idx,
    )
    fine_results.extend(cand_fine)
fine_results.sort(key=lambda r: r.score)
print(f"    All {len(fine_results)} fine-refined runs ranked by score (top 10):")
for rank, r in enumerate(fine_results[:10]):
    print(f"    #{rank + 1}: Score = {r.score:.3f} (Restraint = {r.scores['SCORE.RESTR.PHARMA']:.2f}) "
          f"| RMSD to Xtal = {rmsd_to_xtal(r.mol):.2f} Å")

diverse_top5 = select_diverse_top_k(fine_results + diverse_candidates, k=5, min_pairwise_rmsd=1.0)
diverse_top5.sort(key=lambda r: r.score)
best = diverse_top5[0]

print("\n" + "=" * 80)
print("FINAL RESULT: Simple Diverse-Conformer + Local-SA Pharmacophore Docking")
print(f"  Best-scoring pose : {rmsd_to_xtal(best.mol):.2f} Å RMSD to crystal (Score {best.score:.2f})")
print("  Top-5 diverse candidates:")
for i, c in enumerate(diverse_top5):
    print(f"    #{i + 1}: Score = {c.score:.2f} (Restraint = {c.scores['SCORE.RESTR.PHARMA']:.2f}) "
          f"| RMSD to Xtal = {rmsd_to_xtal(c.mol):.2f} Å")
print("=" * 80)

tag = f"cav{CAVITY_RADIUS:g}_flex{FLEX_RADIUS if FLEX_RADIUS else 'none'}"
top5_name = f"pharma_dock_simple_top5_{tag}_out.sdf"
best_name = f"pharma_dock_simple_best_{tag}_out.sdf"
w = Chem.SDWriter(str(DEMO_DIR / top5_name))
for r in diverse_top5:
    w.write(r.mol)
w.close()
w_best = Chem.SDWriter(str(DEMO_DIR / best_name))
w_best.write(best.mol)
w_best.close()
print(f"\n[✓] Saved {top5_name}, {best_name}")
