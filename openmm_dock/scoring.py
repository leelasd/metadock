"""
rDock-inspired scoring functions implemented using OpenMM Custom Forces.

Each rDock score component (VDW, POLAR, REPUL, HYD, and the INTER/INTRA split)
is computed by its own CustomNonbondedForce in its own OpenMM force group, so
that decomposed energies read back from the Context are genuine physical
quantities rather than a fixed-fraction split of one combined term.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import openmm as mm
from openmm import unit

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
