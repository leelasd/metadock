"""
Unit tests for the gradient-based kinematic-space minimizer in openmm_dock.
"""
import numpy as np

from openmm_dock.gradient_minimizer import (
    central_difference_gradient,
    adam_minimize,
    lbfgs_minimize,
)


def _quadratic_bowl(target: np.ndarray):
    def f(x):
        return float(np.sum((x - target) ** 2))
    return f


def test_central_difference_gradient_matches_analytic():
    target = np.array([2.0, -3.0, 1.5])
    f = _quadratic_bowl(target)
    x0 = np.zeros(3)

    grad = central_difference_gradient(f, x0, step=1e-4)
    analytic = 2 * (x0 - target)

    assert np.allclose(grad, analytic, atol=1e-3)


def test_central_difference_gradient_supports_per_dimension_step():
    target = np.array([1.0, 5.0])
    f = _quadratic_bowl(target)
    x0 = np.array([0.5, 0.5])

    grad = central_difference_gradient(f, x0, step=np.array([1e-3, 1e-5]))
    analytic = 2 * (x0 - target)

    assert np.allclose(grad, analytic, atol=1e-2)


def test_adam_minimize_reduces_quadratic_bowl():
    target = np.array([2.0, -3.0, 1.5])
    f = _quadratic_bowl(target)
    x0 = np.zeros(3)

    result = adam_minimize(f, x0, step_size=1e-4, lr=0.2, n_iterations=200)

    assert result.fun < f(x0)
    assert result.fun < 0.5
    assert len(result.history) > 1
    assert result.n_evals > 0


def test_lbfgs_minimize_finds_quadratic_minimum_precisely():
    target = np.array([2.0, -3.0, 1.5])
    f = _quadratic_bowl(target)
    x0 = np.zeros(3)

    result = lbfgs_minimize(f, x0, step_size=1e-4, max_iterations=100)

    assert result.converged
    assert np.allclose(result.x, target, atol=0.05)
    assert result.fun < 1e-4


def test_lbfgs_minimize_respects_bounds():
    target = np.array([10.0, 10.0])
    f = _quadratic_bowl(target)
    x0 = np.array([0.0, 0.0])
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    result = lbfgs_minimize(f, x0, step_size=1e-4, max_iterations=50, bounds=bounds)

    assert np.all(result.x >= -1.0 - 1e-6)
    assert np.all(result.x <= 1.0 + 1e-6)
