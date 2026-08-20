"""
rDock-inspired scoring functions implemented using OpenMM Custom Forces.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import openmm as mm
from openmm import unit
from .core import DockAtom


# Force group assignments for clean energy decomposition
GROUP_NONBONDED = 0
GROUP_VALENCE = 1
GROUP_CAVITY = 2
GROUP_PHARMA = 3
GROUP_TETHER = 4
GROUP_SOLVENT = 5


@dataclass
class ScoreWeights:
    vdw: float = 1.0
    polar: float = 1.0
    hbond: float = 1.0
    hydrophobic: float = 0.5
    cavity: float = 1.0
    pharma: float = 2.0
    tether: float = 5.0
    intra: float = 0.5


def create_unified_rdock_force(
    weights: ScoreWeights,
    cutoff_distance_nm: float = 1.2,
    soft_delta_nm: float = 0.05,
    dielectric_slope: float = 2.0,
) -> mm.CustomNonbondedForce:
    """
    Creates a unified OpenMM CustomNonbondedForce implementing:
    - Intermolecular soft-core 4-8 LJ, screened electrostatics, H-bonding, hydrophobic contact
    - Intramolecular ligand nonbonded terms
    - Automatic exclusion of receptor-receptor internal energy
    """
    expr = (
        "is_inter * (w_vdw * E_vdw + w_pol * E_polar + w_hb * E_hb + w_hyd * E_hyd) + "
        "is_intra * w_intra * (E_vdw + E_polar);"
        "is_inter = (is_lig1 + is_lig2 - 2.0 * is_lig1 * is_lig2);"
        "is_intra = (is_lig1 * is_lig2);"
        "E_vdw = 4.0 * eps * ((sig / r_eff)^8 - (sig / r_eff)^4);"
        "E_polar = 138.935456 * (q1 * q2) / (dielectric_slope * r_eff^2);"
        "E_hb = - 12.0 * is_hb_pair * exp(- (r_eff - 0.28)^2 / 0.02);"
        "E_hyd = - 3.0 * is_hyd_pair * exp(- (r_eff - 0.38)^2 / 0.04);"
        "is_hb_pair = (is_don1 * is_acc2 + is_don2 * is_acc1);"
        "is_hyd_pair = (is_hyd1 * is_hyd2);"
        "r_eff = sqrt(r^2 + soft_delta^2);"
        "sig = 0.5 * (sig1 + sig2);"
        "eps = sqrt(eps1 * eps2);"
    )

    force = mm.CustomNonbondedForce(expr)
    force.addPerParticleParameter("q")        # Partial charge (e)
    force.addPerParticleParameter("sig")      # Sigma (nm)
    force.addPerParticleParameter("eps")      # Epsilon (kJ/mol)
    force.addPerParticleParameter("is_don")   # Is H-bond donor (0 or 1)
    force.addPerParticleParameter("is_acc")   # Is H-bond acceptor (0 or 1)
    force.addPerParticleParameter("is_hyd")   # Is hydrophobic (0 or 1)
    force.addPerParticleParameter("is_lig")   # Is ligand particle (0 or 1)

    # Global parameters
    force.addGlobalParameter("w_vdw", weights.vdw)
    force.addGlobalParameter("w_pol", weights.polar)
    force.addGlobalParameter("w_hb", weights.hbond)
    force.addGlobalParameter("w_hyd", weights.hydrophobic)
    force.addGlobalParameter("w_intra", weights.intra)
    force.addGlobalParameter("soft_delta", soft_delta_nm)
    force.addGlobalParameter("dielectric_slope", dielectric_slope)

    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(cutoff_distance_nm * unit.nanometers)
    force.setForceGroup(GROUP_NONBONDED)
    force.setName("RbtUnifiedScoringForce")

    return force
