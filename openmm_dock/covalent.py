"""
Covalent Docking Module for openmm-dock.
Provides automated electrophilic warhead perception (Acrylamides, Haloacetamides, Aldehydes,
Nitriles, Boronic Acids, Sulfonyl Fluorides), receptor nucleophile matching (Cys, Ser, Thr, Lys, Tyr),
and harmonic bond/angle restraint formulation for GPU-accelerated covalent docking.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import openmm as mm
from openmm import unit
from rdkit import Chem

from .core import MolecularSystem, DockAtom


@dataclass
class CovalentRestraint:
    rec_nucleophile_idx: int          # 0-indexed in receptor
    rec_nucleophile_anchor_idx: int   # 0-indexed in receptor (e.g. CB of Cys)
    lig_electrophile_idx: int          # 0-indexed in ligand
    r0_nm: float = 0.182               # Equilibrium covalent bond length in nm
    k_bond: float = 2000000.0          # Spring constant in kJ/(mol nm^2)
    theta0_rad: float = 1.824          # Equilibrium valence angle in radians (~104.5 deg)
    k_angle: float = 5000.0            # Angle spring constant in kJ/(mol rad^2)
    warhead_name: str = "Acrylamide"
    target_residue: str = "CYS"


# Standard warhead SMARTS patterns and default parameters
# (smarts, electrophile_idx_in_match, target_res, r0_nm, theta0_rad, name)
WARHEAD_PATTERNS = [
    # 1. Acrylamide / Vinyl Sulfonamide (Michael Acceptors -> Cys SG)
    (
        "[C;H2,H1:1]=[C;H1,H0:2]-[C,S:3](=[O:4])",
        0,
        "CYS",
        0.182,
        1.824,  # ~104.5 deg
        "Acrylamide / Michael Acceptor",
    ),
    # 2. Haloacetamide (Alkylation -> Cys SG)
    (
        "[Cl,Br,I:1]-[CH2:2]-[C:3](=[O:4])",
        1,
        "CYS",
        0.182,
        1.824,
        "Haloacetamide",
    ),
    # 3. Aldehyde (Hemithioacetal / Hemiacetal -> Cys SG / Ser OG)
    (
        "[CH1:1]=[O:2]",
        0,
        "CYS",
        0.182,
        1.824,
        "Aldehyde",
    ),
    # 4. Nitrile / Carbonitrile -> Cys SG / Ser OG
    (
        "[C:1]#[N:2]",
        0,
        "CYS",
        0.182,
        1.824,
        "Nitrile",
    ),
    # 5. Boronic Acid -> Ser OG / Thr OG1
    (
        "[B:1](-[OH:2])(-[OH:3])",
        0,
        "SER",
        0.145,
        1.911,  # ~109.5 deg
        "Boronic Acid",
    ),
    # 6. Sulfonyl Fluoride -> Tyr OH / Lys NZ
    (
        "[S:1](=[O:2])(=[O:3])-[F:4]",
        0,
        "TYR",
        0.165,
        1.911,
        "Sulfonyl Fluoride",
    ),
]


# Target nucleophilic atoms and anchor atoms per amino acid residue
NUCLEOPHILE_MAP: Dict[str, Tuple[str, str]] = {
    "CYS": ("SG", "CB"),
    "SER": ("OG", "CB"),
    "THR": ("OG1", "CB"),
    "TYR": ("OH", "CZ"),
    "LYS": ("NZ", "CE"),
}


def detect_ligand_warhead(mol: Chem.Mol) -> Optional[Tuple[str, int, str, float, float]]:
    """
    Perceives reactive electrophilic warhead in query ligand.
    Returns: (warhead_name, electrophile_atom_idx, target_residue, r0_nm, theta0_rad) or None.
    """
    for smarts, el_match_idx, target_res, r0, theta0, name in WARHEAD_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            matches = mol.GetSubstructMatches(patt)
            if matches:
                electrophile_idx = matches[0][el_match_idx]
                return (name, electrophile_idx, target_res, r0, theta0)
    return None


def find_receptor_nucleophile(
    receptor: MolecularSystem,
    residue_spec: str | int,
) -> Tuple[int, int, str]:
    """
    Finds the 0-indexed nucleophile atom index and its anchor neighbor in the receptor.
    `residue_spec` can be an int residue index (e.g. 145) or string like 'CYS145', 'A:CYS:145'.
    Returns: (nucleophile_atom_idx, anchor_atom_idx, residue_name)
    """
    # Parse residue_spec
    res_num = None
    res_name = None
    chain_id = None

    if isinstance(residue_spec, int):
        res_num = residue_spec
    else:
        # e.g. "CYS145" or "A:145" or "145"
        parts = str(residue_spec).replace(":", " ").split()
        for p in parts:
            p_upper = p.upper()
            if p_upper in NUCLEOPHILE_MAP:
                res_name = p_upper
            elif p.isdigit() or (p[0] in "+-" and p[1:].isdigit()):
                res_num = int(p)
            elif len(p) > 3 and p[:3].upper() in NUCLEOPHILE_MAP and p[3:].isdigit():
                res_name = p[:3].upper()
                res_num = int(p[3:])
            elif len(p) == 1 and p.isalpha():
                chain_id = p

    # Search receptor atoms
    matching_atoms = []
    for idx, a in enumerate(receptor.atoms):
        if res_num is not None and a.residue_idx != res_num:
            continue
        if res_name is not None and a.residue_name.upper() != res_name:
            continue
        if chain_id is not None and a.chain != chain_id:
            continue
        matching_atoms.append((idx, a))

    if not matching_atoms:
        raise ValueError(f"Could not find residue matching '{residue_spec}' in receptor.")

    actual_res_name = matching_atoms[0][1].residue_name.upper()
    if actual_res_name not in NUCLEOPHILE_MAP:
        raise ValueError(f"Residue {actual_res_name} is not a supported nucleophile (Supported: {list(NUCLEOPHILE_MAP.keys())}).")

    nucl_name, anchor_name = NUCLEOPHILE_MAP[actual_res_name]

    nucl_idx = None
    anchor_idx = None

    for idx, a in matching_atoms:
        if a.name.upper() == nucl_name:
            nucl_idx = idx
        elif a.name.upper() == anchor_name:
            anchor_idx = idx

    if nucl_idx is None:
        # Fallback: Pick any non-backbone heavy atom in residue
        for idx, a in matching_atoms:
            if a.name.upper() not in ["N", "CA", "C", "O", "H", "HA"]:
                nucl_idx = idx
                break
        if nucl_idx is None:
            nucl_idx = matching_atoms[0][0]

    if anchor_idx is None:
        # Pick CA or first atom
        for idx, a in matching_atoms:
            if a.name.upper() == "CA":
                anchor_idx = idx
                break
        if anchor_idx is None:
            anchor_idx = nucl_idx

    return (nucl_idx, anchor_idx, actual_res_name)


def create_covalent_restraint(
    receptor: MolecularSystem,
    ligand_mol: Chem.Mol,
    covalent_res: str | int,
    warhead_type: Optional[str] = None,
    custom_r0_nm: Optional[float] = None,
) -> CovalentRestraint:
    """
    Constructs a CovalentRestraint object by resolving receptor nucleophile and ligand electrophile.
    """
    nucl_idx, anchor_idx, res_name = find_receptor_nucleophile(receptor, covalent_res)

    warhead_info = detect_ligand_warhead(ligand_mol)
    if warhead_info is not None:
        w_name, lig_el_idx, target_res, r0, theta0 = warhead_info
    else:
        # Fallback to first heavy atom in ligand
        w_name = "Generic"
        lig_el_idx = 0
        r0 = 0.182 if res_name == "CYS" else 0.145
        theta0 = 1.824

    if custom_r0_nm is not None:
        r0 = custom_r0_nm

    return CovalentRestraint(
        rec_nucleophile_idx=nucl_idx,
        rec_nucleophile_anchor_idx=anchor_idx,
        lig_electrophile_idx=lig_el_idx,
        r0_nm=r0,
        k_bond=500000.0,
        theta0_rad=theta0,
        k_angle=5000.0,
        warhead_name=w_name,
        target_residue=res_name,
    )


def prealign_ligand_for_covalent_docking(
    ligand_mol: Chem.Mol,
    receptor: MolecularSystem,
    restraint: CovalentRestraint,
) -> Chem.Mol:
    """
    AutoDock-style two-point rigid pre-alignment (Bianco et al. 2016, "Covalent
    docking using AutoDock: the two-point attractor method"): orients the
    ligand so its electrophile atom sits at the ideal attack geometry (bond
    length r0 along the anchor->nucleophile extension) and a second, adjacent
    ligand atom points back away from the pocket -- fixing both the attack
    position and the rotation around the forming bond, rather than a
    translation-only shift that leaves the rest of the ligand's orientation
    arbitrary and frequently drives it straight through the receptor.

    A single point only constrains 3 of the ligand's 6 rigid-body DOFs; this
    uses a second reference atom (any heavy-atom neighbor of the electrophile)
    to also fix the two rotational DOFs, leaving only rotation about the
    forming-bond axis free for the search to resolve.
    """
    mol_copy = Chem.Mol(ligand_mol)
    conf = mol_copy.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol_copy.GetNumAtoms())])

    nucl_pos = receptor.atoms[restraint.rec_nucleophile_idx].coord
    # Approach direction: point away from the local mass of nearby receptor
    # atoms, not just the CB->SG bond extension. A single bond vector is
    # unreliable -- side chains curl, and even on a real, experimentally
    # validated site (e.g. BTK Cys481) the naive CB->SG extension can point
    # straight at a neighboring residue's side chain. Estimating "outward"
    # from the local atom centroid is a much more robust proxy for the
    # solvent-accessible direction.
    rec_coords = receptor.coordinates
    dists_to_nucl = np.linalg.norm(rec_coords - nucl_pos, axis=1)
    local_mask = (dists_to_nucl < 10.0) & (dists_to_nucl > 0.1)
    if np.any(local_mask):
        local_centroid = rec_coords[local_mask].mean(axis=0)
        approach_dir = nucl_pos - local_centroid
    else:
        anchor_pos = receptor.atoms[restraint.rec_nucleophile_anchor_idx].coord
        approach_dir = nucl_pos - anchor_pos
    approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-12)

    el_idx = restraint.lig_electrophile_idx
    el_atom = mol_copy.GetAtomWithIdx(el_idx)
    ref_idx = next(
        (n.GetIdx() for n in el_atom.GetNeighbors() if n.GetAtomicNum() > 1),
        next((n.GetIdx() for n in el_atom.GetNeighbors()), el_idx),
    )

    target_el = nucl_pos + approach_dir * (restraint.r0_nm * 10.0)
    target_ref = target_el - approach_dir * 1.3  # ~a bond length back, away from the pocket

    lig_el_vec = coords[ref_idx] - coords[el_idx]
    lig_el_norm = np.linalg.norm(lig_el_vec)
    target_vec = target_ref - target_el

    if lig_el_norm > 1e-6 and el_idx != ref_idx:
        lig_dir = lig_el_vec / lig_el_norm
        tgt_dir = target_vec / (np.linalg.norm(target_vec) + 1e-12)
        axis = np.cross(lig_dir, tgt_dir)
        axis_norm = np.linalg.norm(axis)
        cos_angle = float(np.clip(np.dot(lig_dir, tgt_dir), -1.0, 1.0))
        if axis_norm > 1e-8:
            axis = axis / axis_norm
            angle = np.arccos(cos_angle)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        elif cos_angle < 0:
            # Antiparallel: rotate 180 deg about any axis perpendicular to lig_dir.
            perp = np.array([1.0, 0.0, 0.0]) if abs(lig_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(lig_dir, perp)
            axis = axis / np.linalg.norm(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            rot = np.eye(3) + 2.0 * (K @ K)
        else:
            rot = np.eye(3)
    else:
        rot = np.eye(3)

    rotated = (coords - coords[el_idx]) @ rot.T
    new_coords = rotated + target_el

    for i in range(mol_copy.GetNumAtoms()):
        p = new_coords[i]
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))

    return mol_copy


def create_covalent_bond_force(
    restraint: CovalentRestraint,
    nucl_idx: int,
    el_idx: int,
    capped_delta_nm: float = 0.15,
) -> mm.CustomBondForce:
    """
    Capped-force ("Huber loss"-shaped) covalent bond restraint: quadratic
    (standard stiff harmonic bond, sub-Angstrom precision) within
    `capped_delta_nm` of r0, transitioning to *linear* (bounded, constant
    force) beyond it. A plain HarmonicBondForce's restoring force grows
    without bound as the ligand starts farther from the target -- at 2-3 nm
    off (a realistic blind-docking starting distance) that force is large
    enough to yank the ligand through the receptor in a single integrator
    step before any steric/orientational resolution can happen. Capping the
    far-field force mirrors AutoDock's smooth covalent attractor potential:
    a gentle, bounded pull while far away, full stiffness once close.
    """
    expr = (
        "select(step(delta - d), 0.5 * k * d^2, k * delta * (d - 0.5 * delta));"
        "d = abs(r - r0)"
    )
    force = mm.CustomBondForce(expr)
    force.addPerBondParameter("r0")
    force.addPerBondParameter("k")
    force.addPerBondParameter("delta")
    force.addBond(nucl_idx, el_idx, [restraint.r0_nm, restraint.k_bond, capped_delta_nm])
    force.setName("CovalentAdductBond")
    return force
