#!/usr/bin/env python
"""
Pluggable Scoring + Bayesian Optimization + Gradient-Based Kinematic Polish Demo.

Demonstrates the three OpenDock-inspired additions to openmm_dock together on
the real Keap1 + Q9E macrocycle system (PDB 6Z6A):

1. scoring_function.py  - a pluggable BaseScoringFunction interface. The real
   OpenMM physical energy is wrapped as OpenMMPhysicalScore, then composed
   with a custom PocketDepthBeacon scorer via CompositeScoringFunction --
   exactly the "physical score + weighted correction term" pattern OpenDock
   uses for its OnionNet-SFCT ML correction, just with a simple geometric
   beacon here instead of an ML model.
2. bayesian_optimizer.py - a Gaussian-Process Bayesian optimizer explores the
   6D rigid-body pose space (translation + rotation) around a perturbed
   starting pose, using far fewer objective evaluations than a PSO/GA swarm
   would need for the same search.
3. gradient_minimizer.py - the best pose Bayesian optimization finds is then
   locally polished with L-BFGS-B over finite-difference gradients of the
   SAME composite scoring function, directly in the reduced kinematic space
   (no OpenMM-Cartesian-space minimization involved).

Run: python run_pluggable_optimization_demo.py
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
from rdkit import Chem
from rdkit.Geometry import Point3D

from openmm_dock.unified_kinematic_pso import UnifiedKinematicPSOEngine
from openmm_dock.generalized_cv import GeneralizedCVEngine
from openmm_dock.scoring_function import BaseScoringFunction, OpenMMPhysicalScore, CompositeScoringFunction
from openmm_dock.bayesian_optimizer import BayesianKinematicOptimizer
from openmm_dock.gradient_minimizer import lbfgs_minimize

ROOT = Path(__file__).resolve().parent
RECEPTOR_PATH = ROOT.parent / "macrocycle_6z6a" / "receptor.pdb"
LIGAND_PATH = ROOT.parent / "macrocycle_6z6a" / "q9e_crystal_pose.sdf"
POCKET_CENTER = np.array([-21.46, 22.44, -24.18])


class PocketDepthBeacon(BaseScoringFunction):
    """
    Custom scorer example (see scoring_function.py's module docstring for the
    "how to add a custom scorer" pattern this follows): penalizes ligand
    center-of-mass distance from the pocket centroid, on top of the physical
    score.
    """
    name = "pocket_depth_beacon"

    def __init__(self, cv_calc: GeneralizedCVEngine, weight: float = 1.0):
        super().__init__(weight=weight)
        self.cv_calc = cv_calc

    def score(self, lig_coords, rec_coords=None):
        zeta_depth, _ = self.cv_calc.compute_pocket_depth(lig_coords)
        return zeta_depth


def main():
    print("=" * 90)
    print("  PLUGGABLE SCORING + BAYESIAN OPTIMIZATION + GRADIENT KINEMATIC POLISH DEMO")
    print("=" * 90)

    xtal_mol = Chem.SDMolSupplier(str(LIGAND_PATH), removeHs=False)[0]
    conf = xtal_mol.GetConformer()
    xtal_coords = np.array([conf.GetAtomPosition(i) for i in range(xtal_mol.GetNumAtoms())])

    engine = UnifiedKinematicPSOEngine(
        receptor_pdb_path=RECEPTOR_PATH,
        pocket_center=POCKET_CENTER,
        ligand_mol=xtal_mol,
        flex_radius=0.0  # rigid receptor: keeps this demo to a 6D pose search
    )

    # --- 1. Compose the scoring function ---
    physical = OpenMMPhysicalScore(engine.context, engine.engine._full_positions_from_coords,
                                    lig_start=engine.lig_start, weight=1.0)
    depth_beacon = PocketDepthBeacon(
        cv_calc=GeneralizedCVEngine(pocket_center=POCKET_CENTER), weight=2.0
    )
    combo = CompositeScoringFunction([physical, depth_beacon])

    zero_ring = np.zeros(engine.num_ring_drivers)
    zero_exo = np.zeros(engine.num_exo)
    zero_chi = np.zeros(engine.num_rec_chi)

    def kinematics_to_coords(theta6: np.ndarray) -> np.ndarray:
        trans, rot_vec = theta6[:3], theta6[3:6]
        _, lig_coords, _ = engine.evaluate_coupled_state(trans, rot_vec, zero_ring, zero_exo, zero_chi)
        return lig_coords

    def objective(theta6: np.ndarray) -> float:
        return combo.score(kinematics_to_coords(theta6))

    # --- Perturb the crystal pose so there is real optimization work to do ---
    rng = np.random.default_rng(7)
    perturb_trans = rng.uniform(-3.0, 3.0, size=3)
    perturb_rot = rng.uniform(-0.6, 0.6, size=3)
    x0 = np.concatenate([perturb_trans, perturb_rot])

    start_score = objective(x0)
    xtal_score = objective(np.zeros(6))
    print(f"\nCrystal pose composite score : {xtal_score:.2f}")
    print(f"Perturbed start composite score: {start_score:.2f}  (trans off by {np.linalg.norm(perturb_trans):.2f} A)")

    # --- 2. Bayesian optimization: coarse global search over the 6D pose ---
    print("\n[*] Running Bayesian optimization over 6D rigid-body pose space...")
    bo = BayesianKinematicOptimizer(
        objective_fn=objective,
        lower_bounds=np.array([-6.0, -6.0, -6.0, -np.pi, -np.pi, -np.pi]),
        upper_bounds=np.array([6.0, 6.0, 6.0, np.pi, np.pi, np.pi]),
        n_initial=8,
        n_iterations=15,
        n_candidates=400,
        random_seed=7,
    )
    bo_result = bo.optimize()
    print(f"    Bayesian opt best score: {bo_result.fun:.2f}  ({bo_result.n_evals} objective evaluations)")

    # --- 3. Gradient-based local polish via L-BFGS-B on the same objective ---
    print("\n[*] Polishing best Bayesian-opt pose with L-BFGS-B (finite-difference gradients)...")
    polish = lbfgs_minimize(
        objective, bo_result.x,
        step_size=np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]),
        max_iterations=40
    )
    print(f"    Polished score: {polish.fun:.2f}  (converged={polish.converged}, {polish.n_evals} evaluations)")

    final_coords = kinematics_to_coords(polish.x)
    final_rmsd = float(np.sqrt(np.mean(np.sum((final_coords - xtal_coords) ** 2, axis=1))))
    breakdown = combo.breakdown(final_coords)

    print("\n" + "=" * 90)
    print("RESULTS")
    print(f"  Start composite score    : {start_score:.2f}")
    print(f"  Bayesian-opt score       : {bo_result.fun:.2f}")
    print(f"  Final (polished) score   : {polish.fun:.2f}")
    print(f"  Score breakdown          : {breakdown}")
    print(f"  Final RMSD to crystal    : {final_rmsd:.2f} A")
    print("=" * 90)

    out_mol = Chem.Mol(xtal_mol)
    out_conf = out_mol.GetConformer()
    for i in range(out_mol.GetNumAtoms()):
        out_conf.SetAtomPosition(i, Point3D(*[float(c) for c in final_coords[i]]))
    out_mol.SetProp("_Name", "Q9E_PluggableOptimizationDemo_BestPose")
    out_mol.SetProp("COMPOSITE_SCORE", f"{polish.fun:.3f}")
    out_mol.SetProp("RMSD_TO_XTAL_A", f"{final_rmsd:.3f}")
    out_path = ROOT / "best_pose.sdf"
    writer = Chem.SDWriter(str(out_path))
    writer.write(out_mol)
    writer.close()
    print(f"\n[OK] Saved best pose to {out_path}")


if __name__ == "__main__":
    main()
