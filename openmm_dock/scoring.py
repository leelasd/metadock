"""
rDock-inspired scoring functions implemented using OpenMM Custom Forces.

Each rDock score component (VDW, POLAR, REPUL, HYD, and the INTER/INTRA split)
is computed by its own CustomNonbondedForce in its own OpenMM force group, so
that decomposed energies read back from the Context are genuine physical
quantities rather than a fixed-fraction split of one combined term.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import openmm as mm
from openmm import unit

from rdkit import Chem

from .core import MolecularSystem
from .gridding import GridBox, create_boundary_penalty_force, STANDARD_VDW_ELEMENTS

# Force group assignments for clean energy decomposition.
GROUP_VDW_INTER = 0
GROUP_VALENCE = 1
GROUP_CAVITY = 2
GROUP_PHARMA = 3
GROUP_TETHER = 4
GROUP_SOLVENT = 5
GROUP_POLAR_INTER = 6
GROUP_REPUL = 7
GROUP_HYD = 8
GROUP_VDW_INTRA = 9
GROUP_POLAR_INTRA = 10

# Back-compat alias: historically the whole nonbonded blob lived in group 0.
GROUP_NONBONDED = GROUP_VDW_INTER

_PARTICLE_PARAMS = ["q", "sig", "eps", "is_don", "is_acc", "is_hyd", "is_lig"]


@dataclass
class ScoreWeights:
    vdw: float = 1.0
    polar: float = 1.0
    hbond: float = 1.0
    hydrophobic: float = 0.5
    repul: float = 1.0
    const: float = 0.37     # kcal/mol per active-site water (rDock SOLVENT_PENALTY)
    rot: float = 1.0        # kcal/mol per rotatable bond (rDock RbtRotSF)
    cavity: float = 1.0
    pharma: float = 2.0
    tether: float = 5.0
    intra: float = 0.5


class RDockNonbondedForces:
    """
    Bundles the set of CustomNonbondedForce objects that together make up the
    rDock-style nonbonded scoring function. Particles and exclusions are added
    once through this wrapper and forwarded to every underlying force so all
    terms stay in sync over the same atom set.
    """

    def __init__(self, forces: List[mm.CustomNonbondedForce]):
        self._forces = forces

    @property
    def forces(self) -> List[mm.CustomNonbondedForce]:
        return self._forces

    def addParticle(self, params: List[float]) -> int:
        idx = None
        for f in self._forces:
            idx = f.addParticle(params)
        return idx

    def addExclusion(self, i1: int, i2: int) -> None:
        for f in self._forces:
            f.addExclusion(i1, i2)


def _new_force(expr: str, group: int, name: str, cutoff_nm: float) -> mm.CustomNonbondedForce:
    force = mm.CustomNonbondedForce(expr)
    for p in _PARTICLE_PARAMS:
        force.addPerParticleParameter(p)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(cutoff_nm * unit.nanometers)
    force.setForceGroup(group)
    force.setName(name)
    return force


def create_rdock_nonbonded_forces(
    weights: ScoreWeights,
    cutoff_distance_nm: float = 1.2,
    soft_delta_nm: float = 0.05,
    dielectric_slope: float = 2.0,
    repul_distance_nm: float = 0.24,
    repul_k: float = 20000.0,
) -> RDockNonbondedForces:
    """
    Creates the separate rDock-style nonbonded terms:
    - VDW (inter / intra): soft-core 4-8 Lennard-Jones.
    - POLAR (inter / intra): distance-dependent-dielectric screened electrostatics
      plus a contact hydrogen-bonding bonus (folds rDock's directional H-bond
      scoring into the attractive polar term).
    - REPUL (inter only): short-range steric clash penalty specific to donor/
      acceptor atom pairs closer than the ideal H-bond distance — the OpenMM
      analogue of rDock's RbtPolarIdxSF(ATTR=FALSE) repulsive polar term.
    - HYD (inter only): hydrophobic desolvation contact bonus.
    Each term lives in its own force group so real per-term energies can be
    read back via context.getState(getEnergy=True, groups={...}).
    """
    common_defs = (
        "r_eff = sqrt(r^2 + soft_delta^2);"
        "sig = 0.5 * (sig1 + sig2);"
        "eps = sqrt(eps1 * eps2);"
        "is_inter = (is_lig1 + is_lig2 - 2.0 * is_lig1 * is_lig2);"
        "is_intra = (is_lig1 * is_lig2);"
    )

    vdw_inter_expr = (
        "is_inter * w_vdw * E_vdw;"
        "E_vdw = 4.0 * eps * ((sig / r_eff)^8 - (sig / r_eff)^4);"
        + common_defs
    )
    vdw_intra_expr = (
        "is_intra * w_intra * E_vdw;"
        "E_vdw = 4.0 * eps * ((sig / r_eff)^8 - (sig / r_eff)^4);"
        + common_defs
    )
    polar_inter_expr = (
        "is_inter * (w_pol * E_polar + w_hb * E_hb);"
        "E_polar = 138.935456 * (q1 * q2) / (dielectric_slope * r_eff^2);"
        "E_hb = - 12.0 * is_hb_pair * exp(- (r_eff - 0.28)^2 / 0.02);"
        "is_hb_pair = (is_don1 * is_acc2 + is_don2 * is_acc1);"
        + common_defs
    )
    polar_intra_expr = (
        "is_intra * w_intra * E_polar;"
        "E_polar = 138.935456 * (q1 * q2) / (dielectric_slope * r_eff^2);"
        + common_defs
    )
    repul_expr = (
        "is_inter * w_repul * is_polar_pair * step(r_min_polar - r_eff) * k_repul * (r_min_polar - r_eff)^2;"
        "is_polar_pair = min(1.0, is_don1 + is_acc1) * min(1.0, is_don2 + is_acc2);"
        + common_defs
    )
    hyd_expr = (
        "is_inter * w_hyd * E_hyd;"
        "E_hyd = - 3.0 * is_hyd_pair * exp(- (r_eff - 0.38)^2 / 0.04);"
        "is_hyd_pair = (is_hyd1 * is_hyd2);"
        + common_defs
    )

    vdw_inter = _new_force(vdw_inter_expr, GROUP_VDW_INTER, "RDockVdwInterForce", cutoff_distance_nm)
    vdw_intra = _new_force(vdw_intra_expr, GROUP_VDW_INTRA, "RDockVdwIntraForce", cutoff_distance_nm)
    polar_inter = _new_force(polar_inter_expr, GROUP_POLAR_INTER, "RDockPolarInterForce", cutoff_distance_nm)
    polar_intra = _new_force(polar_intra_expr, GROUP_POLAR_INTRA, "RDockPolarIntraForce", cutoff_distance_nm)
    repul = _new_force(repul_expr, GROUP_REPUL, "RDockRepulForce", cutoff_distance_nm)
    hyd = _new_force(hyd_expr, GROUP_HYD, "RDockHydForce", cutoff_distance_nm)

    for f in (vdw_inter, vdw_intra, polar_inter, polar_intra, repul, hyd):
        f.addGlobalParameter("soft_delta", soft_delta_nm)

    vdw_inter.addGlobalParameter("w_vdw", weights.vdw)
    vdw_intra.addGlobalParameter("w_intra", weights.intra)
    polar_inter.addGlobalParameter("w_pol", weights.polar)
    polar_inter.addGlobalParameter("w_hb", weights.hbond)
    polar_inter.addGlobalParameter("dielectric_slope", dielectric_slope)
    polar_intra.addGlobalParameter("w_intra", weights.intra)
    polar_intra.addGlobalParameter("dielectric_slope", dielectric_slope)
    repul.addGlobalParameter("w_repul", weights.repul)
    repul.addGlobalParameter("r_min_polar", repul_distance_nm)
    repul.addGlobalParameter("k_repul", repul_k)
    hyd.addGlobalParameter("w_hyd", weights.hydrophobic)

    return RDockNonbondedForces([vdw_inter, vdw_intra, polar_inter, polar_intra, repul, hyd])


def create_combined_search_force(
    weights: ScoreWeights,
    cutoff_distance_nm: float = 1.2,
    soft_delta_nm: float = 0.05,
    dielectric_slope: float = 2.0,
    repul_distance_nm: float = 0.24,
    repul_k: float = 20000.0,
) -> RDockNonbondedForces:
    """
    Same physics as create_rdock_nonbonded_forces (VDW + POLAR + H-bond + REPUL +
    HYD, inter and intra), but summed into a *single* CustomNonbondedForce
    instead of six. Six separate forces means six separate neighbor-list builds
    against the full receptor every time positions change -- fine for one-off
    scoring, but ruinous for a GA inner loop that evaluates thousands of
    candidate poses (each a large jump, not a small perturbation, so neighbor
    lists can't be reused). Callers that need genuinely decomposed per-term
    energies (score/minimize/dock/mc, and each GA run's final reported pose)
    should use create_rdock_nonbonded_forces instead; this is for fast relative
    ranking only.
    """
    expr = (
        "w_vdw * (is_inter * E_vdw + is_intra * w_intra_r * E_vdw) + "
        "is_inter * (w_pol * E_polar + w_hb * E_hb) + is_intra * w_intra_r * E_polar + "
        "is_inter * w_repul * is_polar_pair * step(r_min_polar - r_eff) * k_repul * (r_min_polar - r_eff)^2 + "
        "is_inter * w_hyd * E_hyd;"
        "E_vdw = 4.0 * eps * ((sig / r_eff)^8 - (sig / r_eff)^4);"
        "E_polar = 138.935456 * (q1 * q2) / (dielectric_slope * r_eff^2);"
        "E_hb = - 12.0 * is_hb_pair * exp(- (r_eff - 0.28)^2 / 0.02);"
        "E_hyd = - 3.0 * is_hyd_pair * exp(- (r_eff - 0.38)^2 / 0.04);"
        "is_hb_pair = (is_don1 * is_acc2 + is_don2 * is_acc1);"
        "is_hyd_pair = (is_hyd1 * is_hyd2);"
        "is_polar_pair = min(1.0, is_don1 + is_acc1) * min(1.0, is_don2 + is_acc2);"
        "r_eff = sqrt(r^2 + soft_delta^2);"
        "sig = 0.5 * (sig1 + sig2);"
        "eps = sqrt(eps1 * eps2);"
        "is_inter = (is_lig1 + is_lig2 - 2.0 * is_lig1 * is_lig2);"
        "is_intra = (is_lig1 * is_lig2);"
    )
    force = _new_force(expr, GROUP_VDW_INTER, "RDockCombinedSearchForce", cutoff_distance_nm)
    force.addGlobalParameter("w_vdw", weights.vdw)
    force.addGlobalParameter("w_intra_r", weights.intra)
    force.addGlobalParameter("w_pol", weights.polar)
    force.addGlobalParameter("w_hb", weights.hbond)
    force.addGlobalParameter("w_repul", weights.repul)
    force.addGlobalParameter("w_hyd", weights.hydrophobic)
    force.addGlobalParameter("soft_delta", soft_delta_nm)
    force.addGlobalParameter("dielectric_slope", dielectric_slope)
    force.addGlobalParameter("r_min_polar", repul_distance_nm)
    force.addGlobalParameter("k_repul", repul_k)
    return RDockNonbondedForces([force])


class GridSearchForces:
    """
    Grid-based analogue of RDockNonbondedForces: same addParticle/
    addExclusion interface engine.py's _build_system already drives its
    per-atom loop through, so selecting the grid backend needs no change to
    that loop -- only to which factory function builds the nonbonded force
    set. Internally very different from the pairwise wrapper: ligand atoms
    get registered as one-particle "bonds" on the grid-lookup
    CustomCompoundBondForces (see create_grid_search_force), while every
    atom (ligand and receptor/water alike) still needs a slot on the
    intramolecular CustomNonbondedForce, matching how CustomNonbondedForce
    requires addParticle called once per system particle in index order.
    """

    def __init__(
        self,
        vdw_forces: Dict[str, mm.CustomCompoundBondForce],
        nonvdw_force: mm.CustomCompoundBondForce,
        intra_force: mm.CustomNonbondedForce,
        boundary_force: mm.CustomExternalForce,
        probe_params: Dict[str, Tuple[float, float]],
    ):
        self._vdw_forces = vdw_forces
        self._nonvdw_force = nonvdw_force
        self._intra_force = intra_force
        self._boundary_force = boundary_force
        self._sigma_eps_to_type = {(round(s, 6), round(e, 6)): t for t, (s, e) in probe_params.items()}
        self._next_idx = 0
        self._ligand_indices: List[int] = []
        self._interaction_group_finalized = False

    @property
    def forces(self) -> List[mm.Force]:
        if not self._interaction_group_finalized:
            lig_set = set(self._ligand_indices)
            if lig_set:
                self._intra_force.addInteractionGroup(lig_set, lig_set)
            self._interaction_group_finalized = True
        return list(self._vdw_forces.values()) + [self._nonvdw_force, self._intra_force, self._boundary_force]

    def addParticle(self, params: List[float]) -> int:
        idx = self._next_idx
        self._next_idx += 1
        q, sig, eps, is_don, is_acc, is_hyd, is_lig = params

        self._intra_force.addParticle(params)

        if is_lig >= 0.5:
            self._ligand_indices.append(idx)
            self._boundary_force.addParticle(idx, [])

            vdw_type = self._sigma_eps_to_type.get((round(sig, 6), round(eps, 6)))
            if vdw_type is not None:
                self._vdw_forces[vdw_type].addBond([idx], [])

            is_polar = 1.0 if (is_don >= 0.5 or is_acc >= 0.5) else 0.0
            self._nonvdw_force.addBond([idx], [q, is_don, is_acc, is_hyd, is_polar])

        return idx

    def addExclusion(self, i1: int, i2: int) -> None:
        self._intra_force.addExclusion(i1, i2)


def _make_tabulated_function(grid: np.ndarray, box: GridBox) -> mm.Continuous3DFunction:
    xmin, xmax, ymin, ymax, zmin, zmax = box.bounds_nm
    nx, ny, nz = box.shape
    return mm.Continuous3DFunction(
        nx, ny, nz, grid.flatten(order="F"), xmin, xmax, ymin, ymax, zmin, zmax
    )


def create_grid_search_force(
    vdw_grids: Dict[str, np.ndarray],
    vdw_box: GridBox,
    shared_grids: Dict[str, np.ndarray],
    shared_box: GridBox,
    weights: ScoreWeights,
    vdw_probe_types: Optional[List[str]] = None,
    cutoff_distance_nm: float = 1.2,
    soft_delta_nm: float = 0.05,
    dielectric_slope: float = 2.0,
    boundary_slope: float = 1e6,
) -> GridSearchForces:
    """
    Grid-interpolated analogue of create_combined_search_force: the same
    physics (VDW + screened electrostatics + contact H-bond + hydrophobic +
    short-range polar repulsion), but the intermolecular (ligand-receptor)
    part is O(1) grid interpolation per ligand atom instead of an O(N_lig x
    N_receptor) pairwise sum -- see gridding.py for how the grids in
    `vdw_grids`/`shared_grids` (from compute_potential_grids) are built and
    why. Intramolecular (ligand-ligand) terms stay exact pairwise,
    restricted to ligand-ligand pairs via addInteractionGroup, since the
    ligand is small and this is already cheap.

    vdw_grids/vdw_box and shared_grids/shared_box are deliberately separate:
    VDW's r^-8 falloff needs a much finer box/spacing to interpolate
    accurately than the smoother electrostatics/H-bond/hydrophobic/repulsion
    terms do (empirically ~17% error in combined VDW energy at 0.375 A
    spacing vs <0.1% for every other term at the same spacing -- see
    compute_potential_grids' docstring). The boundary penalty uses
    shared_box (assumed the larger/coarser of the two, i.e. the true search
    box) since it exists to bound the ligand's rigid-body position, not to
    track the finer VDW-only lattice.

    `vdw_probe_types` must match the grid keys actually present in
    `vdw_grids` (default: all of gridding.STANDARD_VDW_ELEMENTS). A ligand
    atom whose (sigma, epsilon) doesn't match any requested probe type
    contributes to the intramolecular force only, not the grid-based
    intermolecular VDW term -- callers (engine.py) should check ligand
    element coverage against STANDARD_VDW_ELEMENTS before choosing this
    backend over create_combined_search_force, rather than relying on this
    silent per-atom omission as the primary safety net.
    """
    if vdw_probe_types is None:
        vdw_probe_types = list(STANDARD_VDW_ELEMENTS)

    vdw_forces: Dict[str, mm.CustomCompoundBondForce] = {}
    for t in vdw_probe_types:
        key = f"vdw_{t}"
        if key not in vdw_grids:
            continue
        force = mm.CustomCompoundBondForce(1, "w_vdw * vdwGrid(x1,y1,z1)")
        force.addTabulatedFunction("vdwGrid", _make_tabulated_function(vdw_grids[key], vdw_box))
        force.addGlobalParameter("w_vdw", weights.vdw)
        force.setForceGroup(GROUP_VDW_INTER)
        force.setName(f"GridVdwForce_{t}")
        vdw_forces[t] = force

    nonvdw_expr = (
        "w_pol * q * elecGrid(x1,y1,z1) + "
        "w_hb * is_don * hbdonGrid(x1,y1,z1) + w_hb * is_acc * hbaccGrid(x1,y1,z1) + "
        "w_hyd * is_hyd * hydGrid(x1,y1,z1) + "
        "w_repul * is_polar * repulGrid(x1,y1,z1)"
    )
    nonvdw_force = mm.CustomCompoundBondForce(1, nonvdw_expr)
    for p in ("q", "is_don", "is_acc", "is_hyd", "is_polar"):
        nonvdw_force.addPerBondParameter(p)
    nonvdw_force.addTabulatedFunction("elecGrid", _make_tabulated_function(shared_grids["elec"], shared_box))
    nonvdw_force.addTabulatedFunction("hbdonGrid", _make_tabulated_function(shared_grids["hbdon"], shared_box))
    nonvdw_force.addTabulatedFunction("hbaccGrid", _make_tabulated_function(shared_grids["hbacc"], shared_box))
    nonvdw_force.addTabulatedFunction("hydGrid", _make_tabulated_function(shared_grids["hyd"], shared_box))
    nonvdw_force.addTabulatedFunction("repulGrid", _make_tabulated_function(shared_grids["repul"], shared_box))
    nonvdw_force.addGlobalParameter("w_pol", weights.polar)
    nonvdw_force.addGlobalParameter("w_hb", weights.hbond)
    nonvdw_force.addGlobalParameter("w_hyd", weights.hydrophobic)
    nonvdw_force.addGlobalParameter("w_repul", weights.repul)
    nonvdw_force.setForceGroup(GROUP_VDW_INTER)
    nonvdw_force.setName("GridNonVdwForce")

    intra_expr = (
        "w_intra_r * (E_vdw + E_polar);"
        "E_vdw = 4.0 * eps * ((sig / r_eff)^8 - (sig / r_eff)^4);"
        "E_polar = 138.935456 * (q1 * q2) / (dielectric_slope * r_eff^2);"
        "r_eff = sqrt(r^2 + soft_delta^2);"
        "sig = 0.5 * (sig1 + sig2);"
        "eps = sqrt(eps1 * eps2);"
    )
    intra_force = _new_force(intra_expr, GROUP_VDW_INTER, "GridIntraForce", cutoff_distance_nm)
    intra_force.addGlobalParameter("w_intra_r", weights.intra)
    intra_force.addGlobalParameter("soft_delta", soft_delta_nm)
    intra_force.addGlobalParameter("dielectric_slope", dielectric_slope)

    boundary_force = create_boundary_penalty_force(shared_box, ligand_particle_indices=[], slope=boundary_slope)
    boundary_force.setForceGroup(GROUP_VDW_INTER)

    probe_params: Dict[str, Tuple[float, float]] = {}
    for t in vdw_forces:
        from .core import DockAtom
        probe = DockAtom(idx=-1, name="probe", element=t, sybyl_type=f"{t}.3", charge=0.0, coord=np.zeros(3))
        probe_params[t] = (probe.sigma, probe.epsilon)

    return GridSearchForces(vdw_forces, nonvdw_force, intra_force, boundary_force, probe_params)


def create_gaussian_desolvation_force(
    weight: float = 1.0,
    cutoff_nm: float = 1.0,
) -> mm.CustomNonbondedForce:
    """
    Continuous Gaussian implicit desolvation potential (Salveson et al.,
    Science 2024, oligo-macs' make_grid.py) -- an ADDITIVE, opt-in
    supplement to the existing contact hydrophobic/H-bond terms in
    create_rdock_nonbonded_forces (GROUP_HYD), not a replacement: add this
    force's return value to a System alongside the existing ones (the same
    pattern create_covalent_bond_force and create_boundary_penalty_force
    already use for optional terms) if you want it. Not wired into the
    default create_rdock_nonbonded_forces/create_combined_search_force
    pipeline automatically, so every existing demo's baseline scoring is
    unaffected unless a caller explicitly opts in.

    Real formula from the reference:
        V_sol = -2.5 + 5.0*exp(-(2q)^2)
                + 2.5 * sum_j (|q_i| + |q_j| - 0.6) * exp(-(r_ij - (r0_i+r0_j))^2)

    Only the SECOND (pairwise sum) term is implemented here. The first term
    is a function of a single atom's OWN charge alone -- not a pairwise
    interaction at all, so it can't actually be expressed inside a
    CustomNonbondedForce (the originating GitHub issue's own proposed
    single-expression snippet folded both terms together, which silently
    can't work: OpenMM has no way to see "this atom's charge" without a
    second particle to pair it with inside a NonbondedForce expression). It
    would need a separate CustomExternalForce to add correctly. It's also
    scientifically irrelevant to omit for THIS codebase's use case: that
    baseline term is IDENTICAL for every candidate pose of a given ligand
    (charges don't change as the ligand moves), so it cannot affect pose
    ranking during a docking search at all -- it only matters when
    comparing the overall desolvation cost of DIFFERENT ligands (virtual
    screening), which isn't what this codebase's search engines do.

    r0 (atomic radius) is taken as HALF of each atom's existing VDW_PARAMS
    sigma (sigma is the sum-of-radii/zero-crossing LJ distance, so half of
    it approximates one atom's own radius) -- reuses this codebase's own
    existing VDW parameterization rather than the reference's separate
    r0_dict, for consistency with every other term in this module.

    The exp(-100*(r-r0_sum)^2) form (vs. the reference's exp(-(r-r0_sum)^2))
    is the SAME Gaussian, just written in OpenMM's native nanometers instead
    of Angstroms: converting r, r0_sum from nm to the reference's Angstrom
    convention multiplies their difference by 10, and squaring that
    multiplies the exponent by 100 -- not a different, weaker/stronger
    potential, the identical one in different length units.
    """
    energy_expr = (
        "w_sol * 2.5 * (abs(q1) + abs(q2) - 0.6) * exp(-100.0 * (r - r0_sum)^2);"
        "r0_sum = 0.5 * (sig1 + sig2)"
    )
    force = mm.CustomNonbondedForce(energy_expr)
    force.addPerParticleParameter("q")
    force.addPerParticleParameter("sig")
    force.addGlobalParameter("w_sol", weight)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(cutoff_nm * unit.nanometers)
    force.setForceGroup(GROUP_HYD)
    force.setName("GaussianDesolvationForce")
    return force


@dataclass
class HBondFeatures:
    """donor_pairs[i] = (heavy_donor_atom_idx, hydrogen_idx); acceptor_pairs[i]
    = (antecedent_heavy_atom_idx, acceptor_atom_idx). Both (0, 2) int arrays
    when a molecule has none of that feature type."""
    donor_pairs: np.ndarray
    acceptor_pairs: np.ndarray


def extract_ligand_hbond_features(mol: "Chem.Mol") -> HBondFeatures:
    """
    Donor/acceptor + antecedent-atom extraction for a ligand RDKit Mol.
    Uses GetTotalNumHs(includeNeighbors=True), not the bare
    GetTotalNumHs() the original feature-request issue's own proposed code
    used -- that packed-count property returns 0 once a molecule's
    hydrogens exist as explicit graph-neighbor atoms (true for any molecule
    that has been through Chem.AddHs(), which is common in this codebase's
    own pipeline), silently finding zero donors. Same bug already fixed
    once in pharmacophore.py's find_ligand_pharma_features and again in
    core.py's SDFParser.mol_to_system -- fixed directly here too rather
    than reproduced a third time.
    """
    donor_pairs: List[Tuple[int, int]] = []
    acceptor_pairs: List[Tuple[int, int]] = []
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        idx = atom.GetIdx()
        if elem in ("N", "O") and atom.GetTotalNumHs(includeNeighbors=True) > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "H":
                    donor_pairs.append((idx, nbr.GetIdx()))
        if elem in ("O", "N") and atom.GetFormalCharge() <= 0:
            heavy_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() != "H"]
            if heavy_neighbors:
                acceptor_pairs.append((heavy_neighbors[0], idx))
    return HBondFeatures(
        donor_pairs=np.array(donor_pairs, dtype=np.int64) if donor_pairs else np.zeros((0, 2), dtype=np.int64),
        acceptor_pairs=np.array(acceptor_pairs, dtype=np.int64) if acceptor_pairs else np.zeros((0, 2), dtype=np.int64),
    )


def extract_receptor_hbond_features(receptor: "MolecularSystem") -> HBondFeatures:
    """
    Same extraction against a DockAtom-based receptor MolecularSystem,
    using its is_donor/is_acceptor flags (see core.py's
    _assign_donor_flags_from_bonds / _assign_standard_residue_donor_fallback
    -- fixed this session; previously these flags were wrong for most
    standard-PDB-named atoms) plus its bond graph to find each donor's
    attached hydrogen and each acceptor's heavy antecedent atom.
    """
    adjacency: Dict[int, List[int]] = {}
    for b in receptor.bonds:
        adjacency.setdefault(b.atom1, []).append(b.atom2)
        adjacency.setdefault(b.atom2, []).append(b.atom1)

    donor_pairs: List[Tuple[int, int]] = []
    acceptor_pairs: List[Tuple[int, int]] = []
    for i, a in enumerate(receptor.atoms):
        if a.is_donor:
            for j in adjacency.get(i, []):
                if receptor.atoms[j].element.upper() == "H":
                    donor_pairs.append((i, j))
                    break
        if a.is_acceptor:
            heavy_neighbors = [j for j in adjacency.get(i, []) if receptor.atoms[j].element.upper() != "H"]
            if heavy_neighbors:
                acceptor_pairs.append((heavy_neighbors[0], i))
    return HBondFeatures(
        donor_pairs=np.array(donor_pairs, dtype=np.int64) if donor_pairs else np.zeros((0, 2), dtype=np.int64),
        acceptor_pairs=np.array(acceptor_pairs, dtype=np.int64) if acceptor_pairs else np.zeros((0, 2), dtype=np.int64),
    )


class VectorizedDirectionalHBondScorer:
    """
    Pure-numpy, OpenMM-free directional hydrogen-bond scorer for batch pose
    filtering/rescoring (companion to gridding.HostGridPrescreen -- this
    codebase's existing contact H-bond bonus in create_rdock_nonbonded_forces
    is purely distance-based, matching rDock's own RbtPolarIdxSF: a donor-H
    pointing 90 degrees away from an acceptor scores identically to a
    perfect, collinear 180-degree hydrogen bond. This class adds a genuine
    directional term as an ADDITIONAL host-side filtering/rescoring signal,
    not a replacement for the physical OpenMM score, which remains
    authoritative -- matching HostGridPrescreen's own "cheap filter, real
    OpenMM score still decides" design.

    Math verified directly against the real reference source (Salveson et
    al., Science 2024, oligo-macs' virtual_screening_scripts/vector_dock_align.py's
    hbond_inter, not just the paraphrased version in the originating GitHub
    issue -- confirmed to match): for a donor (heavy atom D, hydrogen H) and
    acceptor (antecedent atom C, acceptor atom A):

        r = |H -> A|                          (H...acceptor distance)
        e = r - 2.0 Angstrom                  (offset from the ideal distance)
        e *= 1.5 if e < 0                     (steeper penalty for too-close than too-far)
        S_dist  = clip(1 - (e / 0.8)^2, 0, None)         (distance term, parabolic bump at r=2.0 A)
        S_acc   = clip(-(C->A) . (H->A), 0, None)        (acceptor lone-pair alignment)
        S_don   = clip((D->H) . (H->A), 0, None)         (donor linearity: D-H...A colinear is best)
        E_hbond = -1 * S_dist^2 * S_acc * S_don^2

    summed over every (donor, acceptor) pair between the two partners.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @staticmethod
    def _hbond_energy(donor_coords: np.ndarray, acceptor_coords: np.ndarray) -> np.ndarray:
        """
        donor_coords: (n_poses, n_donors, 2, 3) -- [...,0,:]=heavy D, [...,1,:]=H
        acceptor_coords: (n_poses, n_acceptors, 2, 3) -- [...,0,:]=antecedent C, [...,1,:]=acceptor A
        Returns (n_poses,) summed directional H-bond energy, fully
        vectorized over BOTH the pose axis and every donor x acceptor pair
        in one call (no Python loop over poses) -- callers with one fixed
        side (e.g. a static receptor) broadcast it to a matching pose axis
        with np.broadcast_to before calling, so this method only ever needs
        to handle the one, unambiguous (P, n, 2, 3) shape.
        """
        n_poses = donor_coords.shape[0]
        if donor_coords.shape[1] == 0 or acceptor_coords.shape[1] == 0:
            return np.zeros(n_poses)

        d_heavy, d_h = donor_coords[..., 0, :], donor_coords[..., 1, :]      # (P, n_don, 3)
        a_ante, a_acc = acceptor_coords[..., 0, :], acceptor_coords[..., 1, :]  # (P, n_acc, 3)

        ray_ho = a_acc[:, None, :, :] - d_h[:, :, None, :]                   # (P, n_don, n_acc, 3): H -> A
        dist = np.linalg.norm(ray_ho, axis=-1)                              # (P, n_don, n_acc)
        ray_ho_unit = ray_ho / np.maximum(dist[..., None], 1e-6)

        offset = dist - 2.0
        offset = np.where(offset < 0, offset * 1.5, offset)
        s_dist = np.clip(1.0 - (offset / 0.8) ** 2, 0.0, None)

        ray_co = a_acc - a_ante                                             # (P, n_acc, 3): C -> A
        ray_co_unit = ray_co / np.maximum(np.linalg.norm(ray_co, axis=-1, keepdims=True), 1e-6)
        s_acc = np.clip(-np.einsum("pijk,pjk->pij", ray_ho_unit, ray_co_unit), 0.0, None)

        ray_nh = d_h - d_heavy                                              # (P, n_don, 3): D -> H
        ray_nh_unit = ray_nh / np.maximum(np.linalg.norm(ray_nh, axis=-1, keepdims=True), 1e-6)
        s_don = np.clip(np.einsum("pijk,pik->pij", ray_ho_unit, ray_nh_unit), 0.0, None)

        energy = -1.0 * (s_dist ** 2) * s_acc * (s_don ** 2)
        return energy.sum(axis=(1, 2))

    def score_batch(
        self,
        lig_coords_batch: np.ndarray,   # (n_poses, n_lig_atoms, 3)
        lig_features: HBondFeatures,
        rec_coords: np.ndarray,         # (n_rec_atoms, 3), fixed receptor
        rec_features: HBondFeatures,
    ) -> np.ndarray:
        """Ligand-donor -> receptor-acceptor plus receptor-donor ->
        ligand-acceptor, for every pose in one fully-vectorized call each
        (no per-pose Python loop) -- the receptor's own fixed donor/acceptor
        coordinates are broadcast across the pose axis via np.broadcast_to
        (a view, not a copy, so this costs no extra memory)."""
        n_poses = lig_coords_batch.shape[0]
        total = np.zeros(n_poses)

        n_rec_don, n_rec_acc = len(rec_features.donor_pairs), len(rec_features.acceptor_pairs)
        rec_donor_xyz = (
            np.broadcast_to(rec_coords[rec_features.donor_pairs], (n_poses, n_rec_don, 2, 3))
            if n_rec_don else np.zeros((n_poses, 0, 2, 3))
        )
        rec_acc_xyz = (
            np.broadcast_to(rec_coords[rec_features.acceptor_pairs], (n_poses, n_rec_acc, 2, 3))
            if n_rec_acc else np.zeros((n_poses, 0, 2, 3))
        )

        if len(lig_features.donor_pairs):
            lig_donor_xyz = lig_coords_batch[:, lig_features.donor_pairs]  # (P, n_lig_don, 2, 3)
            total += self.weight * self._hbond_energy(lig_donor_xyz, rec_acc_xyz)
        if len(lig_features.acceptor_pairs):
            lig_acc_xyz = lig_coords_batch[:, lig_features.acceptor_pairs]  # (P, n_lig_acc, 2, 3)
            total += self.weight * self._hbond_energy(rec_donor_xyz, lig_acc_xyz)

        return total
