"""
Shared low-level utilities used by every kinematic sampling/optimization engine
in openmm-dock (forward kinematics, macrocycle inverse kinematics, PSO/SA/GA
samplers, and metadynamics). Consolidated here because these were previously
duplicated near-verbatim across 6+ modules.
"""
from __future__ import annotations
from typing import List, Optional, Set, Tuple
import numpy as np
from rdkit import Chem


def toroidal_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Shortest angular difference a - b on the circle T^k (handles branch-cut
    wraparound at +-pi correctly, unlike a plain subtraction).
    """
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def find_downstream_atoms(
    mol: Chem.Mol,
    begin_idx: int,
    split_idx: int,
    extra_blocked_edges: Optional[Set[Tuple[int, int]]] = None,
) -> List[int]:
    """
    Breadth-first search from split_idx over mol's bond graph, returning every
    atom reachable without crossing the begin_idx<->split_idx edge (or any
    edge in extra_blocked_edges, needed when the two sides of a cut bond are
    still connected via a ring path elsewhere -- e.g. macrocycle loop closure).
    """
    blocked_edges = {(min(begin_idx, split_idx), max(begin_idx, split_idx))}
    if extra_blocked_edges:
        blocked_edges |= extra_blocked_edges

    visited: Set[int] = {split_idx}
    queue = [split_idx]
    while queue:
        curr = queue.pop(0)
        for nbr in mol.GetAtomWithIdx(curr).GetNeighbors():
            n_idx = nbr.GetIdx()
            edge = (min(curr, n_idx), max(curr, n_idx))
            if edge in blocked_edges:
                continue
            if n_idx not in visited:
                visited.add(n_idx)
                queue.append(n_idx)
    return sorted(visited)


def identify_rotatable_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """
    Finds acyclic, non-terminal, non-triple-bond-adjacent single bonds
    (the rotatable-bond definition shared by forward-kinematics torsion trees
    and the two-tier macrocycle engine's exocyclic joints), deduplicated by
    unordered atom pair.
    """
    rot_bond_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(rot_bond_smarts)
    seen: Set[Tuple[int, int]] = set()
    unique_matches: List[Tuple[int, int]] = []
    for a1, a2 in matches:
        pair = (min(a1, a2), max(a1, a2))
        if pair not in seen:
            seen.add(pair)
            unique_matches.append((a1, a2))
    return unique_matches
