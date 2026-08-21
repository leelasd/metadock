"""
Unit tests for the pluggable scoring-function interface in openmm_dock.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem
import openmm as mm
from openmm import unit

from openmm_dock.engine import DockingEngine
from openmm_dock.scoring_function import (
    BaseScoringFunction,
    OpenMMPhysicalScore,
    CompositeScoringFunction,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "test_examples"


def _build_macrocycle_context():
    mol = Chem.SDMolSupplier(
        str(EXAMPLES_DIR / "macrocycle_6z6a" / "q9e_crystal_pose.sdf"), removeHs=False
    )[0]
    engine = DockingEngine(receptor_path=EXAMPLES_DIR / "macrocycle_6z6a" / "receptor.pdb")
    system, _, lig_start, lig_n = engine._build_system(mol)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = mm.Context(system, integrator)

    conf = mol.GetConformer()
    lig_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    return engine, context, lig_start, lig_coords


def test_openmm_physical_score_matches_direct_computation():
    engine, context, lig_start, lig_coords = _build_macrocycle_context()

    scorer = OpenMMPhysicalScore(context, engine._full_positions_from_coords, lig_start=lig_start)
    scored = scorer.score(lig_coords)

    full_pos = engine._full_positions_from_coords(lig_coords)
    context.setPositions(full_pos)
    direct = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    assert abs(scored - direct) < 1e-6
    assert scorer.name == "openmm_physical"


def test_base_scoring_function_is_callable():
    engine, context, lig_start, lig_coords = _build_macrocycle_context()
    scorer = OpenMMPhysicalScore(context, engine._full_positions_from_coords, lig_start=lig_start)

    assert scorer(lig_coords) == scorer.score(lig_coords)


class _ToyCorrectionScore(BaseScoringFunction):
    """Fixed-value scorer used only to test composition/weighting."""
    name = "toy"

    def score(self, lig_coords, rec_coords=None):
        return 10.0


def test_composite_scoring_function_sums_weighted_terms():
    engine, context, lig_start, lig_coords = _build_macrocycle_context()

    physical = OpenMMPhysicalScore(context, engine._full_positions_from_coords, lig_start=lig_start, weight=1.0)
    toy = _ToyCorrectionScore(weight=0.5)
    combo = CompositeScoringFunction([physical, toy])

    physical_score = physical.score(lig_coords)
    total = combo.score(lig_coords)

    assert abs(total - (physical_score + 5.0)) < 1e-6


def test_composite_scoring_function_breakdown():
    engine, context, lig_start, lig_coords = _build_macrocycle_context()

    physical = OpenMMPhysicalScore(context, engine._full_positions_from_coords, lig_start=lig_start, weight=1.0)
    toy = _ToyCorrectionScore(weight=0.5)
    combo = CompositeScoringFunction([physical, toy])

    breakdown = combo.breakdown(lig_coords)

    assert set(breakdown.keys()) == {"openmm_physical", "toy"}
    assert breakdown["toy"] == 5.0
    assert abs(breakdown["openmm_physical"] - physical.score(lig_coords)) < 1e-6
