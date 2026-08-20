"""
Correctness tests for openmm_dock.gridding (Phase 1 of grid-based scoring).

These validate the precomputed grid *values* directly against the same
pairwise formulas used in scoring.py's create_combined_search_force --
independent of how the grids get interpolated by OpenMM later (Phase 2) --
plus the boundary-penalty force's own correctness.
"""
from pathlib import Path
import math
import time

import numpy as np
import pytest
import openmm as mm
from openmm import unit

from openmm_dock.core import Mol2Parser
from openmm_dock.cavity import CavityDefinition
from openmm_dock.gridding import (
    GridBox,
    compute_potential_grids,
    create_boundary_penalty_force,
    STANDARD_VDW_ELEMENTS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "test_examples"

CUTOFF_NM = 1.2
SOFT_DELTA_NM = 0.05
DIELECTRIC_SLOPE = 2.0
REPUL_DISTANCE_NM = 0.24
REPUL_K = 20000.0


def _exact_pairwise_at_point(receptor, point_nm: np.ndarray, probe_sigma: float, probe_eps: float) -> dict:
    """
    Direct (unvectorized, deliberately independent implementation style from
    gridding.py's windowed accumulation) pairwise sum at a single Cartesian
    point, replicating scoring.py's exact formulas. Used as ground truth.
    """
    e_vdw = 0.0
    e_elec = 0.0
    e_hbdon = 0.0
    e_hbacc = 0.0
    e_hyd = 0.0
    e_repul = 0.0

    for a in receptor.atoms:
        a_nm = a.coord * 0.1
        r2 = float(np.sum((point_nm - a_nm) ** 2))
        if r2 > CUTOFF_NM * CUTOFF_NM:
            continue
        r_eff = math.sqrt(r2 + SOFT_DELTA_NM ** 2)

        sig_comb = 0.5 * (probe_sigma + a.sigma)
        eps_comb = math.sqrt(probe_eps * a.epsilon)
        e_vdw += 4.0 * eps_comb * ((sig_comb / r_eff) ** 8 - (sig_comb / r_eff) ** 4)

        e_elec += 138.935456 * a.charge / (DIELECTRIC_SLOPE * r_eff ** 2)

        if a.is_acceptor or a.is_donor:
            hb = -12.0 * math.exp(-((r_eff - 0.28) ** 2) / 0.02)
            if a.is_acceptor:
                e_hbdon += hb
            if a.is_donor:
                e_hbacc += hb
            if r_eff < REPUL_DISTANCE_NM:
                e_repul += REPUL_K * (REPUL_DISTANCE_NM - r_eff) ** 2

        is_hyd = a.element.upper() in ["C", "CL", "BR", "I", "F"] and not a.is_polar
        if is_hyd:
            e_hyd += -3.0 * math.exp(-((r_eff - 0.38) ** 2) / 0.04)

    return {"vdw": e_vdw, "elec": e_elec, "hbdon": e_hbdon, "hbacc": e_hbacc, "hyd": e_hyd, "repul": e_repul}


@pytest.fixture(scope="module")
def score_system():
    rec_path = EXAMPLES_DIR / "score" / "receptor.mol2"
    prm_path = EXAMPLES_DIR / "score" / "cavity.prm"
    receptor = Mol2Parser.parse(rec_path)
    cavity = CavityDefinition.from_prm_file(prm_path)
    return receptor, cavity


def test_grid_box_from_cavity_covers_expected_bounds(score_system):
    _, cavity = score_system
    box = GridBox.from_cavity(cavity, ligand_margin_ang=6.0, spacing_ang=0.375)

    xmin, xmax, ymin, ymax, zmin, zmax = box.bounds_nm
    half_extent_nm = (cavity.radius + 6.0) * 0.1
    center_nm = cavity.center * 0.1

    assert xmax - xmin >= 2 * half_extent_nm - box.spacing_nm
    assert xmin <= center_nm[0] - half_extent_nm + box.spacing_nm
    assert xmax >= center_nm[0] + half_extent_nm - box.spacing_nm
    # Box should be (very close to) cubic and spacing should match the request.
    assert box.spacing_nm == pytest.approx(0.0375, abs=1e-9)
    assert box.shape[0] == box.shape[1] == box.shape[2]


def test_grid_values_match_exact_pairwise_formula_at_grid_points(score_system):
    """
    The critical correctness gate: pick several exact grid-index positions
    (no interpolation involved) and confirm compute_potential_grids's value
    there matches an independently-implemented direct pairwise sum using the
    same formulas as scoring.py.
    """
    receptor, cavity = score_system
    # A small, cheap box (coarse spacing) just for this correctness check --
    # fine resolution isn't needed to validate the accumulation math itself.
    box = GridBox.from_cavity(cavity, ligand_margin_ang=2.0, spacing_ang=1.0)
    probe_types = ["C", "N", "O"]

    t0 = time.time()
    grids = compute_potential_grids(receptor, box, vdw_probe_types=probe_types)
    elapsed = time.time() - t0
    print(f"\n[test_gridding] compute_potential_grids (coarse box) took {elapsed:.2f}s")

    from openmm_dock.gridding import _resolve_probe_params
    probe_params = _resolve_probe_params(probe_types)

    x_axis, y_axis, z_axis = box.axis_coords_nm(0), box.axis_coords_nm(1), box.axis_coords_nm(2)
    nx, ny, nz = box.shape

    rng = np.random.default_rng(0)
    sample_indices = [
        (nx // 2, ny // 2, nz // 2),
        (nx // 2 + 1, ny // 2 - 1, nz // 2 + 2),
    ]
    for _ in range(3):
        sample_indices.append((
            int(rng.integers(1, nx - 1)),
            int(rng.integers(1, ny - 1)),
            int(rng.integers(1, nz - 1)),
        ))

    for (i, j, k) in sample_indices:
        point_nm = np.array([x_axis[i], y_axis[j], z_axis[k]])
        for t in probe_types:
            sig_p, eps_p = probe_params[t]
            expected = _exact_pairwise_at_point(receptor, point_nm, sig_p, eps_p)

            assert grids[f"vdw_{t}"][i, j, k] == pytest.approx(expected["vdw"], rel=1e-6, abs=1e-6)
            assert grids["elec"][i, j, k] == pytest.approx(expected["elec"], rel=1e-6, abs=1e-6)
            assert grids["hbdon"][i, j, k] == pytest.approx(expected["hbdon"], rel=1e-6, abs=1e-6)
            assert grids["hbacc"][i, j, k] == pytest.approx(expected["hbacc"], rel=1e-6, abs=1e-6)
            assert grids["hyd"][i, j, k] == pytest.approx(expected["hyd"], rel=1e-6, abs=1e-6)
            assert grids["repul"][i, j, k] == pytest.approx(expected["repul"], rel=1e-6, abs=1e-6)


def test_boundary_penalty_zero_inside_linear_outside(score_system):
    _, cavity = score_system
    box = GridBox.from_cavity(cavity, ligand_margin_ang=2.0, spacing_ang=1.0)
    xmin, xmax, ymin, ymax, zmin, zmax = box.bounds_nm
    center_nm = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])

    system = mm.System()
    system.addParticle(1.0)
    force = create_boundary_penalty_force(box, ligand_particle_indices=[0], slope=1e6)
    system.addForce(force)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = mm.Context(system, integrator)

    def energy_at(pos_nm: np.ndarray) -> float:
        context.setPositions([tuple(pos_nm)] * unit.nanometers)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # Inside the box: exactly zero.
    assert energy_at(center_nm) == pytest.approx(0.0, abs=1e-9)
    assert energy_at(np.array([xmin, ymin, zmin])) == pytest.approx(0.0, abs=1e-6)

    # Outside the box: linear in the miss distance along one axis.
    miss = 0.5  # nm
    e = energy_at(np.array([xmax + miss, center_nm[1], center_nm[2]]))
    assert e == pytest.approx(1e6 * miss, rel=1e-6)

    # Larger miss -> proportionally larger penalty.
    e2 = energy_at(np.array([xmax + 2 * miss, center_nm[1], center_nm[2]]))
    assert e2 == pytest.approx(2 * e, rel=1e-6)

    del context, integrator
