# tests/test_metrics.py
import numpy as np
import pytest
from scoring_optimizer.metrics import (
    rmsd_loss, enrichment_auc, potency_spearman, composite_objective,
)


def test_rmsd_zero_for_identical():
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert rmsd_loss(coords, coords) == pytest.approx(0.0)


def test_rmsd_known_value():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    assert rmsd_loss(a, b) == pytest.approx(1.0)


def test_auc_perfect_separation():
    scores = np.array([-10.0, -9.0, -1.0, -0.5])  # lower = better in rDock
    labels = np.array([1, 1, 0, 0])
    assert enrichment_auc(scores, labels) == pytest.approx(1.0)


def test_auc_random_is_near_half():
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(100)
    labels = rng.integers(0, 2, 100)
    assert 0.3 < enrichment_auc(scores, labels) < 0.7


def test_spearman_perfect_correlation():
    scores = np.array([-7.0, -6.0, -5.0, -4.0])
    pic50  = np.array([ 7.0,  6.0,  5.0,  4.0])
    assert potency_spearman(scores, pic50) == pytest.approx(1.0)


def test_composite_objective_perfect():
    assert composite_objective(0.0, 1.0, 1.0, 0.5, 0.3, 0.2) == pytest.approx(0.0)


def test_composite_objective_is_positive():
    assert composite_objective(2.0, 0.5, 0.0, 0.5, 0.3, 0.2) > 0.0


def test_auc_single_class_returns_half():
    scores = np.array([-5.0, -4.0, -3.0])
    labels = np.array([1, 1, 1])  # all actives
    assert enrichment_auc(scores, labels) == pytest.approx(0.5)


def test_spearman_constant_scores_returns_zero():
    scores = np.array([-5.0, -5.0, -5.0])
    pic50 = np.array([7.0, 6.0, 5.0])
    assert potency_spearman(scores, pic50) == pytest.approx(0.0)
