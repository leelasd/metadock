"""
AutoDock-style precomputed receptor potential grids.

Precomputes, once per receptor+cavity combination, a set of 3D potential
grids matching the intermolecular terms in scoring.py's
create_combined_search_force. During search (GA/SA inner loop), a ligand
atom's receptor-interaction energy becomes an O(1) grid interpolation
instead of an O(N_receptor) pairwise sum -- the structural reason this
codebase's search budgets have been stuck at population<=40,
generations<=30 all session (every candidate pose otherwise costs a real
OpenMM force evaluation against the full receptor).

Grounded in the real AutoDock4/AutoGrid and AutoDock-Vina source (not
textbook descriptions of "docking uses grids"):

- 0.375 A default spacing matches both tools' real defaults (AutoGrid's
  hardcoded default and Vina's CLI --spacing default).
- The boundary penalty (create_boundary_penalty_force) mirrors Vina's
  grid.cpp clamp-and-linear-penalty behavior for atoms that leave the grid
  box, since OpenMM's Continuous3DFunction is defined as exactly zero
  outside its box -- an undefined region, not a physically meaningful "no
  interaction" signal. AutoDock4 uses a similar idea (a penalty in place of
  any grid lookup once an atom exits the box) via a quadratic-from-center
  penalty; Vina's clamp+linear form is smoother (continuous at the
  boundary) and is what's implemented here.
- Grid math stays entirely in nanometers (OpenMM's native position unit),
  not Angstroms, so grid bounds line up exactly with the positions a
  CustomCompoundBondForce queries at runtime in Phase 2.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import openmm as mm

from .core import DockAtom, MolecularSystem
from .cavity import CavityDefinition

# Standard ligand element types this module can grid. Matches DockAtom's
# effective VDW typing in core.py: SDFParser.mol_to_system always assigns
# ligand atoms sybyl_type=f"{element}.3", and VDW_PARAMS resolves that to
# one of exactly these ~10 distinct (sigma, epsilon) pairs for organic
# ligands -- a small, fixed, enumerable set, matching AutoDock's own
# probe-type count.
STANDARD_VDW_ELEMENTS: List[str] = ["H", "C", "N", "O", "F", "P", "S", "CL", "BR", "I"]


@dataclass
class GridBox:
    """
    3D lattice for grid-based potential precomputation, in nanometers
    (OpenMM's native position unit) -- not Angstroms, unlike CavityDefinition
    -- so grid bounds line up exactly with the positions (also always nm)
    a CustomCompoundBondForce queries at runtime.
    """
    origin_nm: np.ndarray       # (3,) nm -- position of grid index (0,0,0)
    spacing_nm: float           # nm
    shape: Tuple[int, int, int]  # (xsize, ysize, zsize)

    @classmethod
    def from_cavity(
        cls,
        cavity: CavityDefinition,
        ligand_margin_ang: float = 6.0,
        spacing_ang: float = 0.375,
    ) -> "GridBox":
        """
        Box covers the cavity sphere plus a margin for the ligand's own
        extent -- the cavity restraint bounds the ligand *centroid* to
        within cavity.radius of cavity.center, so individual atoms can
        still extend up to roughly the ligand's own size beyond that.
        """
        half_extent_ang = cavity.radius + ligand_margin_ang
        spacing_nm = spacing_ang * 0.1
        n = int(np.ceil(2.0 * half_extent_ang * 0.1 / spacing_nm)) + 1
        center_nm = cavity.center * 0.1
        origin_nm = center_nm - (n - 1) / 2.0 * spacing_nm
        return cls(origin_nm=origin_nm, spacing_nm=spacing_nm, shape=(n, n, n))

    def axis_coords_nm(self, axis: int) -> np.ndarray:
        return self.origin_nm[axis] + np.arange(self.shape[axis]) * self.spacing_nm

    @property
    def bounds_nm(self) -> Tuple[float, float, float, float, float, float]:
        xs, ys, zs = self.axis_coords_nm(0), self.axis_coords_nm(1), self.axis_coords_nm(2)
        return float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]), float(zs[0]), float(zs[-1])


def _index_window(axis: np.ndarray, center_nm: float, cutoff_nm: float) -> Tuple[int, int]:
    """Index range [i0, i1) covering [center-cutoff, center+cutoff], clipped to the axis."""
    spacing = axis[1] - axis[0] if len(axis) > 1 else 1.0
    i0 = int(np.floor((center_nm - cutoff_nm - axis[0]) / spacing))
    i1 = int(np.ceil((center_nm + cutoff_nm - axis[0]) / spacing)) + 1
    return max(i0, 0), min(i1, len(axis))


def _resolve_probe_params(vdw_probe_types: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    (sigma_nm, epsilon_kJ/mol) for each requested probe element, resolved via
    a real DockAtom (same sybyl_type=f"{element}.3" convention ligand atoms
    get) rather than a duplicated hardcoded table, so grids always match
    core.py's live VDW_PARAMS lookup even if it's edited later.
    """
    params = {}
    for t in vdw_probe_types:
        probe = DockAtom(idx=-1, name="probe", element=t, sybyl_type=f"{t}.3", charge=0.0, coord=np.zeros(3))
        params[t] = (probe.sigma, probe.epsilon)
    return params


def _smoothed_vdw_curve(
    sig_comb: float,
    eps_comb: float,
    soft_delta_nm: float,
    r_max_nm: float,
    r_smooth_nm: float = 0.008,
    n_samples: int = 4000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AutoGrid-style smoothing, done faithfully (see compute_potential_grids'
    docstring for why smoothing is needed at all): a windowed *minimum*
    filter applied along the 1-D radial effective-distance axis for one
    specific (probe, receptor-type) combining pair -- exactly what AutoGrid
    does to its `e_vdW_Hb[i][ia][r]` table before building the 3-D grid from
    it. An isotropic 3-D spatial min-filter is a much cruder stand-in for
    this (it smears a pair's attractive well across a 3-D neighborhood in
    every direction, not just along the line to that one atom -- empirically
    this overshoots by ~30% even at the smallest possible window). Operating
    in 1-D r-space first and only then using the result as a lookup table
    keeps the smoothing exactly where AutoGrid puts it: along the distance
    axis of one atom's own contribution.

    r_smooth_nm=0.008 (0.08 A) is *calibrated to our own potential*, not
    copied from AutoGrid's literal 0.5 A default -- our soft-core 4-8 term
    is considerably steeper than AutoDock4's 12-6/12-10 forms, so AutoGrid's
    window is far too wide here: at 0.05 nm it swings the energy at r=0.35nm
    (right where real docked contacts sit) from 0.0 to -0.37 kJ/mol, an
    artificial bonus, and empirically produces a *worse* net error (-50%)
    than no smoothing at all (+17%). Swept against the true pairwise energy
    on a real docked pose: 0.05nm -> -50%, 0.02nm -> -19%, 0.01nm -> -4.2%,
    0.008nm -> -1.0%, 0.005nm -> +6.1%, 0nm (unsmoothed) -> +17%. 0.008 nm
    is the empirically-tightest point found in that sweep.

    IMPORTANT, found later (engine.py's dock_genetic_algorithm docstring has
    the full story): re-sweeping r_smooth_nm on a genuinely *clashing* pose
    (not the relaxed one above) showed the *energy* error at 0.008nm is still
    good (4.75%, actually the best of the values tried -- 0.1875nm/0.375nm/
    0.5nm/0.75nm all landed at >100% error, so AutoGrid's own window-vs-
    spacing ratio does not generalize to this steeper potential and widening
    r_smooth_nm is not a fix for anything). But even at that good 4.75%
    energy accuracy, the *gradient* OpenMM's Continuous3DFunction cubic
    spline derives from this table was wildly wrong at the same pose (mean
    per-atom force error ~36,500 kJ/mol/nm) -- a small value-level
    interpolation error at a steep point implies a large slope error, and no
    r_smooth_nm value fixes that, because it's a property of interpolating a
    r^-8 function on a fixed lattice, not of this particular smoothing
    choice. The actual fix lives at the call site: don't run a
    gradient-following minimizer against this grid when a candidate pose is
    still badly clashing (see minimize_clash_ceiling_kj in engine.py) --
    trust this table's *energy* values (they're good) without trusting its
    *derivative* in that regime.

    Returns (r_samples, e_samples) suitable for np.interp lookup by r_eff.
    """
    r_samples = np.linspace(soft_delta_nm, r_max_nm, n_samples)
    e_samples = 4.0 * eps_comb * ((sig_comb / r_samples) ** 8 - (sig_comb / r_samples) ** 4)

    dr = r_samples[1] - r_samples[0]
    window = max(1, int(round(r_smooth_nm / dr)))
    if window >= 3:
        from scipy.ndimage import minimum_filter1d
        if window % 2 == 0:
            window += 1
        e_samples = minimum_filter1d(e_samples, size=window, mode="nearest")

    return r_samples, e_samples


def compute_potential_grids(
    receptor: MolecularSystem,
    box: GridBox,
    vdw_probe_types: List[str] = STANDARD_VDW_ELEMENTS,
    cutoff_nm: float = 1.2,
    soft_delta_nm: float = 0.05,
    dielectric_slope: float = 2.0,
    repul_distance_nm: float = 0.24,
    repul_k: float = 20000.0,
    compute_vdw: bool = True,
    compute_shared: bool = True,
    smooth_vdw: bool = True,
    r_smooth_nm: float = 0.008,
) -> Dict[str, np.ndarray]:
    """
    compute_vdw / compute_shared let a caller request only the VDW channels
    or only the shared (elec/hbdon/hbacc/hyd/repul) channels from a given
    call. VDW is smoothed (smooth_vdw=True, matching AutoGrid's own default
    behavior) via _smoothed_vdw_curve: without it, the 4-8 soft-core VDW
    term's r^-8 falloff is far too steep for cubic-spline interpolation to
    represent accurately at a reasonable grid spacing (empirically: ~17%
    error in the combined VDW energy at 0.375 A spacing, vs <0.1% for every
    other term at the same spacing). The smoothed curves are cached per
    distinct (probe type, receptor sigma, receptor epsilon) combining pair
    -- receptor atoms draw from a small, finite set of VDW_PARAMS entries,
    so this cache is tiny (tens of curves) even for a receptor with
    thousands of atoms.

    The AutoGrid-equivalent: for each grid channel, accumulate receptor atom
    contributions using atom-centric windowed accumulation. For each
    receptor atom, compute its local index window in the grid (position +/-
    cutoff_nm), vectorize the atom's contribution formula (identical math to
    the matching term in scoring.create_combined_search_force) over that
    local window with numpy, and accumulate into the shared grid array's
    corresponding slice.

    This deliberately differs from Vina's spatial-bucket-grid neighbor
    search (szv_grid) -- that design suits a tight C++ point-query loop;
    atom-centric windowed accumulation suits numpy's strength of vectorizing
    one large array operation per Python-loop iteration (looping over the
    receptor's few thousand atoms, not the grid's few hundred thousand
    points), with comparable asymptotic cost.

    Returns a dict of grid name -> ndarray of shape box.shape:
      "vdw_<ELEMENT>" for each requested probe type (attractive+repulsive
        soft-core 4-8 LJ a probe of that type would feel).
      "elec": per-unit-ligand-charge electrostatic field (multiply by the
        ligand atom's own charge at lookup time).
      "hbdon": contact H-bond field from receptor acceptors (for use by
        ligand donor atoms).
      "hbacc": contact H-bond field from receptor donors (for use by ligand
        acceptor atoms).
      "hyd": hydrophobic desolvation field from receptor hydrophobic atoms.
      "repul": short-range polar-clash repulsion field from receptor
        donor/acceptor atoms.
    """
    x_axis = box.axis_coords_nm(0)
    y_axis = box.axis_coords_nm(1)
    z_axis = box.axis_coords_nm(2)

    active_vdw_types = list(vdw_probe_types) if compute_vdw else []
    grids: Dict[str, np.ndarray] = {f"vdw_{t}": np.zeros(box.shape) for t in active_vdw_types}
    if compute_shared:
        grids.update({
            "elec": np.zeros(box.shape),
            "hbdon": np.zeros(box.shape),
            "hbacc": np.zeros(box.shape),
            "hyd": np.zeros(box.shape),
            "repul": np.zeros(box.shape),
        })

    probe_params = _resolve_probe_params(active_vdw_types)
    cutoff_sq = cutoff_nm * cutoff_nm
    r_max_nm = math.sqrt(cutoff_sq + soft_delta_nm ** 2)
    curve_cache: Dict[Tuple[str, float, float], Tuple[np.ndarray, np.ndarray]] = {}

    # Pre-filter to receptor atoms whose cutoff-padded position actually
    # overlaps the box -- a receptor typically has thousands of atoms, most
    # of them nowhere near a cavity-sized box, and _index_window's own
    # per-axis clipping would empty-skip them anyway (see the i0>=i1 guard
    # below). Doing that reject/accept check once, vectorized over the whole
    # receptor, instead of per-atom in the main Python loop, is a pure
    # speed-up with the identical accept set -- not an approximation.
    xmin, xmax, ymin, ymax, zmin, zmax = box.bounds_nm
    coords_nm = receptor.coordinates * 0.1
    in_range = (
        (coords_nm[:, 0] >= xmin - cutoff_nm) & (coords_nm[:, 0] <= xmax + cutoff_nm) &
        (coords_nm[:, 1] >= ymin - cutoff_nm) & (coords_nm[:, 1] <= ymax + cutoff_nm) &
        (coords_nm[:, 2] >= zmin - cutoff_nm) & (coords_nm[:, 2] <= zmax + cutoff_nm)
    )
    relevant_atoms = [a for a, keep in zip(receptor.atoms, in_range) if keep]

    for a in relevant_atoms:
        a_coord_nm = a.coord * 0.1

        i0, i1 = _index_window(x_axis, a_coord_nm[0], cutoff_nm)
        j0, j1 = _index_window(y_axis, a_coord_nm[1], cutoff_nm)
        k0, k1 = _index_window(z_axis, a_coord_nm[2], cutoff_nm)
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue

        gx = x_axis[i0:i1][:, None, None]
        gy = y_axis[j0:j1][None, :, None]
        gz = z_axis[k0:k1][None, None, :]
        dx = gx - a_coord_nm[0]
        dy = gy - a_coord_nm[1]
        dz = gz - a_coord_nm[2]
        r2 = dx * dx + dy * dy + dz * dz
        within = r2 <= cutoff_sq
        r_eff = np.sqrt(r2 + soft_delta_nm ** 2)

        a_sig, a_eps = a.sigma, a.epsilon
        a_is_hyd = a.element.upper() in ["C", "CL", "BR", "I", "F"] and not a.is_polar

        for t in active_vdw_types:
            sig_p, eps_p = probe_params[t]
            sig_comb = 0.5 * (sig_p + a_sig)
            eps_comb = math.sqrt(eps_p * a_eps)

            if smooth_vdw:
                cache_key = (t, round(a_sig, 6), round(a_eps, 6))
                curve = curve_cache.get(cache_key)
                if curve is None:
                    curve = _smoothed_vdw_curve(sig_comb, eps_comb, soft_delta_nm, r_max_nm, r_smooth_nm)
                    curve_cache[cache_key] = curve
                r_samples, e_samples = curve
                e_vdw = np.interp(r_eff, r_samples, e_samples)
            else:
                e_vdw = 4.0 * eps_comb * ((sig_comb / r_eff) ** 8 - (sig_comb / r_eff) ** 4)

            grids[f"vdw_{t}"][i0:i1, j0:j1, k0:k1] += np.where(within, e_vdw, 0.0)

        if not compute_shared:
            continue

        e_elec = 138.935456 * a.charge / (dielectric_slope * r_eff ** 2)
        grids["elec"][i0:i1, j0:j1, k0:k1] += np.where(within, e_elec, 0.0)

        if a.is_acceptor or a.is_donor:
            e_hb = -12.0 * np.exp(-((r_eff - 0.28) ** 2) / 0.02)
            if a.is_acceptor:
                grids["hbdon"][i0:i1, j0:j1, k0:k1] += np.where(within, e_hb, 0.0)
            if a.is_donor:
                grids["hbacc"][i0:i1, j0:j1, k0:k1] += np.where(within, e_hb, 0.0)
            e_repul = np.where(r_eff < repul_distance_nm, repul_k * (repul_distance_nm - r_eff) ** 2, 0.0)
            grids["repul"][i0:i1, j0:j1, k0:k1] += np.where(within, e_repul, 0.0)

        if a_is_hyd:
            e_hyd = -3.0 * np.exp(-((r_eff - 0.38) ** 2) / 0.04)
            grids["hyd"][i0:i1, j0:j1, k0:k1] += np.where(within, e_hyd, 0.0)

    return grids


def create_boundary_penalty_force(
    box: GridBox,
    ligand_particle_indices: List[int],
    slope: float = 1e6,
) -> mm.CustomExternalForce:
    """
    Vina-style linear boundary penalty (see module docstring): zero inside
    the grid box, growing linearly with the distance a particle has strayed
    outside it. Composed alongside the grid-lookup forces (Phase 2) so a
    ligand atom that wanders outside the box -- where Continuous3DFunction
    returns exactly zero, an undefined region, not "no interaction" -- is
    still correctly pushed back toward it.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = box.bounds_nm
    expr = (
        "slope * (dx + dy + dz);"
        "dx = max(0, max(xmin_p - x, x - xmax_p));"
        "dy = max(0, max(ymin_p - y, y - ymax_p));"
        "dz = max(0, max(zmin_p - z, z - zmax_p))"
    )
    force = mm.CustomExternalForce(expr)
    force.addGlobalParameter("slope", slope)
    force.addGlobalParameter("xmin_p", xmin)
    force.addGlobalParameter("xmax_p", xmax)
    force.addGlobalParameter("ymin_p", ymin)
    force.addGlobalParameter("ymax_p", ymax)
    force.addGlobalParameter("zmin_p", zmin)
    force.addGlobalParameter("zmax_p", zmax)
    force.setName("GridBoundaryPenaltyForce")

    for idx in ligand_particle_indices:
        force.addParticle(idx, [])

    return force
