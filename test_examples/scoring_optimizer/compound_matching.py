"""Morgan ECFP4 Tanimoto similarity and SAR-to-crystal assignment."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

try:
    from rdkit.Chem import rdFingerprintGenerator as _rfg
    _DEFAULT_MORGAN_GEN = _rfg.GetMorganGenerator(radius=2, fpSize=2048)
except (ImportError, AttributeError):
    _DEFAULT_MORGAN_GEN = None

ECFP4_NBITS = 2048  # 2048 bits reduces collisions vs. standard 1024; tradeoff is slightly larger fingerprints


def morgan_tanimoto(
    mol_a: Chem.Mol, mol_b: Chem.Mol, radius: int = 2, n_bits: int = ECFP4_NBITS
) -> float:
    """Morgan fingerprint (ECFP4) Tanimoto similarity.

    Args:
        mol_a: First RDKit Mol object.
        mol_b: Second RDKit Mol object.
        radius: ECFP radius (default 2 for ECFP4).
        n_bits: Fingerprint bit length (default 2048).

    Returns:
        Tanimoto similarity in [0.0, 1.0].

    Raises:
        ValueError: If mol_a or mol_b is None.
    """
    if mol_a is None or mol_b is None:
        raise ValueError("mol_a and mol_b must be valid RDKit Mol objects (not None)")

    if _DEFAULT_MORGAN_GEN is not None and radius == 2 and n_bits == ECFP4_NBITS:
        # Use cached default generator (avoids deprecation warning, faster)
        fp_a = _DEFAULT_MORGAN_GEN.GetFingerprint(mol_a)
        fp_b = _DEFAULT_MORGAN_GEN.GetFingerprint(mol_b)
    elif _DEFAULT_MORGAN_GEN is not None:
        # Non-default params: create a fresh generator
        from rdkit.Chem import rdFingerprintGenerator as _rfg
        gen = _rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_a = gen.GetFingerprint(mol_a)
        fp_b = gen.GetFingerprint(mol_b)
    else:
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)

    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def assign_to_crystal(
    sar_mol: Chem.Mol,
    crystal_mols: list[Chem.Mol],
    threshold: float = 0.6,
) -> tuple[int | None, float]:
    """Find the crystal structure most similar to sar_mol (Morgan ECFP4 Tanimoto).

    Returns (index, similarity). Returns (None, best_sim) if best_sim < threshold.

    Tie-breaking: If multiple crystal mols have identical similarity scores,
    the lowest-index crystal is returned.

    Args:
        sar_mol: SAR/query molecule. Must be a valid RDKit Mol object (not None).
        crystal_mols: List of crystal molecules. May contain None entries which are
            treated as missing data (assigned -1.0 similarity, never above threshold).
        threshold: Minimum similarity to return a match (default 0.6).

    Returns:
        Tuple of (crystal_index, best_similarity) where:
        - crystal_index is None if best_similarity < threshold
        - None entries in crystal_mols are treated as similarity -1.0 (never above threshold)

    Raises:
        ValueError: If sar_mol is None.
    """
    if sar_mol is None:
        raise ValueError("sar_mol must be a valid RDKit Mol object (not None)")

    if not crystal_mols:
        return None, 0.0

    sims = [morgan_tanimoto(sar_mol, c) if c is not None else -1.0 for c in crystal_mols]
    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
    best_sim = sims[best_idx]
    return (best_idx, best_sim) if best_sim >= threshold else (None, best_sim)
