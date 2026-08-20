"""Scoring optimizer metrics for virtual screening evaluation.

Provides AUC-ROC, Spearman correlation, RMSD, and composite objective functions
for assessing docking scoring function performance.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def rmsd_loss(pred_coords: np.ndarray, crystal_coords: np.ndarray) -> float:
    """RMSD between top-scored pose and crystal structure (Å)."""
    assert pred_coords.ndim == 2 and crystal_coords.ndim == 2, \
        "coords must be 2-D arrays of shape (n_atoms, 3)"
    diff = pred_coords - crystal_coords
    # Sum squared differences across x,y,z per atom, then average over atoms
    squared_distances = np.sum(diff**2, axis=1)
    return float(np.sqrt(np.mean(squared_distances)))


def enrichment_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC: actives=1 vs inactives=0. Negates scores (lower rDock = better).

    Returns 0.5 (random baseline) if only one class is present in labels.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, -scores))


def potency_spearman(scores: np.ndarray, pic50: np.ndarray) -> float:
    """Spearman rank correlation: lower rDock score should → higher pIC50.

    Returns 0.0 if scores are constant (undefined correlation).
    """
    if np.all(scores == scores[0]):
        return 0.0
    rho, _ = spearmanr(-scores, pic50)
    return float(rho)


def composite_objective(
    rmsd: float,
    auc: float,
    spearman: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Composite minimization target: α·rmsd + β·(1−auc) + γ·(1−spearman)."""
    return alpha * rmsd + beta * (1.0 - auc) + gamma * (1.0 - spearman)
