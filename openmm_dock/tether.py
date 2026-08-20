"""
Tethered / Template-constrained docking using Maximum Common Substructure (MCS)
and OpenMM harmonic positional restraints.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import openmm as mm
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from .scoring import GROUP_TETHER


@dataclass
class TetherConstraint:
    ligand_atom_idx: int  # 0-indexed in ligand
    target_pos: np.ndarray  # Shape (3,) in Angstroms
    k_spring: float = 5000.0  # kJ/(mol * nm^2)


def find_tethered_atoms_mcs(
    query_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    threshold: float = 0.8,
    min_ratio: float = 0.20,
) -> Tuple[Optional[Chem.Mol], List[TetherConstraint]]:
    """
    Identifies Maximum Common Substructure between query and reference molecule,
    embeds query to match reference core, and returns aligned mol + tether constraints.
    """
    ref_clean = Chem.RemoveHs(ref_mol)
    query_clean = Chem.RemoveHs(query_mol)

    mcs_result = rdFMCS.FindMCS(
        [ref_clean, query_clean],
        threshold=threshold,
        completeRingsOnly=True,
        ringMatchesRingOnly=True,
    )

    if not mcs_result.smartsString:
        return None, []

    patt = Chem.MolFromSmarts(mcs_result.smartsString)
    if patt is None:
        return None, []

    ref_matches = ref_clean.GetSubstructMatches(patt)
    query_matches = query_clean.GetSubstructMatches(patt)

    if not ref_matches or not query_matches:
        return None, []

    ref_match = ref_matches[0]
    query_match = query_matches[0]

    match_ratio = len(query_match) / float(ref_clean.GetNumAtoms())
    if match_ratio < min_ratio:
        return None, []

    # Map query heavy atom -> ref heavy atom 3D coords
    ref_conf = ref_mol.GetConformer()
    constraints: List[TetherConstraint] = []
    
    for q_idx, r_idx in zip(query_match, ref_match):
        pos = ref_conf.GetAtomPosition(r_idx)
        constraints.append(
            TetherConstraint(
                ligand_atom_idx=q_idx,
                target_pos=np.array([pos.x, pos.y, pos.z], dtype=np.float64),
            )
        )

    # Constrained embed / align query to reference core
    try:
        core = AllChem.ReplaceSidechains(ref_clean, patt)
        if core is not None:
            core = AllChem.DeleteSubstructs(core, Chem.MolFromSmiles("*"))
            core.UpdatePropertyCache()
            new_mol = Chem.Mol(query_mol)
            AllChem.ConstrainedEmbed(new_mol, core)
            return new_mol, constraints
    except Exception:
        pass

    return query_mol, constraints


def create_tether_restraint_force(
    constraints: List[TetherConstraint],
    ligand_offset_in_system: int = 0,
    k_tether: float = 5000.0,
) -> mm.CustomExternalForce:
    """
    Creates an OpenMM CustomExternalForce applying harmonic positional restraints
    to tethered core atoms.
    """
    expr = "0.5 * k_tether * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    force = mm.CustomExternalForce(expr)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addGlobalParameter("k_tether", k_tether)
    force.setForceGroup(GROUP_TETHER)
    force.setName("TetherRestraintForce")

    for c in constraints:
        atom_idx = c.ligand_atom_idx + ligand_offset_in_system
        # Å to nm
        x0_nm = c.target_pos[0] * 0.1
        y0_nm = c.target_pos[1] * 0.1
        z0_nm = c.target_pos[2] * 0.1
        force.addParticle(atom_idx, [x0_nm, y0_nm, z0_nm])

    return force
