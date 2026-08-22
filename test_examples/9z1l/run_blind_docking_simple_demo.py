"""
Simple Diverse-Conformer + Local-SA Blind Docking on PDB 9Z1L (KIT V654A + BLU-654/A1CZZ).

Strategy pivot from run_blind_docking_demo.py's custom multi-phase swarm-
metadynamics/propeller-search engine, motivated by inspecting rDock's own
"blind" docking examples (github.com/leelasd/rxdock-deepdive-examples):
every rDock example defines its cavity via RbtLigandSiteMapper + REF_MOL (a
known reference ligand) with only a 6-15 A radius shell around it -- rDock,
like AutoDock/Vina/Glide, never searches an entire protein surface from
scratch either. The pocket LOCATION is always a given input, not something
the optimizer discovers; only the ORIENTATION and CONFORMATION inside that
pocket are actually searched.

This script mirrors that: the same cavity.prm pocket center used everywhere
else in this example (16.92, -31.66, 18.54), combined with the plain,
already-proven-elsewhere DockingEngine.dock_simulated_annealing in its
UNGUIDED mode (no pharmacophore/tether guidance) -- which already does
exactly "start centered at the pocket with a small jitter, fully random
global orientation, then anneal" (see decode_chromosome: the ligand's
absolute input coordinates are irrelevant, every chromosome is decoded
relative to cavity.center). The only extension here is running that same
proven unguided SA across many independent ETKDGv3 conformers instead of
just one starting conformation, so the search tries many different starting
torsion basins, not just one.
"""
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

print("=" * 95)
print("   OPENMM-DOCK: SIMPLE DIVERSE-CONFORMER + LOCAL-SA BLIND DOCKING (PDB 9Z1L)")
print("=" * 95)

cavity = CavityDefinition.from_prm_file(DEMO_DIR / "cavity.prm")
xtal_mol = SDFParser.load_molecules(DEMO_DIR / "a1czz_crystal_pose.sdf")[0]
xtal_coords = xtal_mol.GetConformer().GetPositions()


def rmsd_to_xtal(mol: Chem.Mol) -> float:
    coords = mol.GetConformer().GetPositions()
    return float(np.sqrt(np.mean(np.sum((coords - xtal_coords) ** 2, axis=1))))


def generate_diverse_conformers(mol: Chem.Mol, n_confs: int, seed: int = 42) -> list:
    """ETKDGv3 diverse conformer ensemble -- genuinely different starting
    torsion angles/3D shapes. No recentering needed: dock_simulated_annealing's
    unguided branch always decodes relative to cavity.center regardless of the
    input conformer's absolute position (see decode_chromosome)."""
    mol_work = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.numThreads = 0
    cids = AllChem.EmbedMultipleConfs(mol_work, numConfs=n_confs, params=params)
    confs = []
    for cid in cids:
        conf = mol_work.GetConformer(cid)
        c_mol = Chem.Mol(mol)
        c_conf = c_mol.GetConformer()
        for i in range(c_mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            c_conf.SetAtomPosition(i, Point3D(p.x, p.y, p.z))
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


print(f"\n[1] Generating {N_CONFORMERS} diverse ETKDGv3 conformers (no pocket alignment needed --")
print("    dock_simulated_annealing always places the ligand at the cavity center)...")
diverse_conformers = generate_diverse_conformers(xtal_mol, n_confs=N_CONFORMERS, seed=7)
print(f"    Generated {len(diverse_conformers)} conformers.")

print(f"\n[2] Unguided cavity-restrained Simulated Annealing docking "
      f"({N_RUNS_PER_CONFORMER} runs x {len(diverse_conformers)} conformers "
      f"= {N_RUNS_PER_CONFORMER * len(diverse_conformers)} total SA runs)...")
engine = DockingEngine(receptor_path=DEMO_DIR / "receptor.mol2", cavity=cavity)

all_results = []
for c_idx, conf_mol in enumerate(diverse_conformers):
    conf_results = engine.dock_simulated_annealing(
        conf_mol, n_runs=N_RUNS_PER_CONFORMER,
        t_high=800.0, t_low=5.0, anneal_steps=15, steps_per_temp=150,
        seed=1000 + c_idx,
    )
    all_results.extend(conf_results)

all_results.sort(key=lambda r: r.score)
print(f"\n[*] All {len(all_results)} SA runs ranked by score (top 15):")
for rank, r in enumerate(all_results[:15]):
    print(f"    #{rank + 1}: Score = {r.score:.3f} kcal/mol | RMSD to Xtal = {rmsd_to_xtal(r.mol):.2f} Å")

best = all_results[0]
# Widened to top-15 (not top-5): the pharmacophore version of this same
# pipeline showed a top-5-by-score cutoff silently drops the actual closest
# poses when the score's local minimum is anti-correlated with RMSD in this
# region -- which stage [2]'s own ranking above already shows here too
# (rank #4 at 3.76 A vs. rank #1's 8.70 A).
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
    print(f"    #{rank + 1}: Score = {r.score:.3f} | RMSD to Xtal = {rmsd_to_xtal(r.mol):.2f} Å")

diverse_top5 = select_diverse_top_k(fine_results + diverse_candidates, k=5, min_pairwise_rmsd=1.0)
diverse_top5.sort(key=lambda r: r.score)
best = diverse_top5[0]

print("\n" + "=" * 80)
print("FINAL RESULT: Simple Diverse-Conformer + Local-SA Blind Docking")
print(f"  Best-scoring pose : {rmsd_to_xtal(best.mol):.2f} Å RMSD to crystal (Score {best.score:.2f})")
print("  Top-5 diverse candidates:")
for i, c in enumerate(diverse_top5):
    print(f"    #{i + 1}: Score = {c.score:.2f} | RMSD to Xtal = {rmsd_to_xtal(c.mol):.2f} Å")
print("=" * 80)

w = Chem.SDWriter(str(DEMO_DIR / "blind_docking_simple_top5_out.sdf"))
for r in diverse_top5:
    w.write(r.mol)
w.close()
w_best = Chem.SDWriter(str(DEMO_DIR / "blind_docking_simple_best_out.sdf"))
w_best.write(best.mol)
w_best.close()
print("\n[✓] Saved blind_docking_simple_top5_out.sdf, blind_docking_simple_best_out.sdf")
