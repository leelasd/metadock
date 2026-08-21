"""
Unit tests for the Bayesian optimization sampler in openmm_dock.
"""
import numpy as np

from openmm_dock.bayesian_optimizer import (
    GaussianProcessRegressor,
    expected_improvement,
    BayesianKinematicOptimizer,
)


def test_gp_regressor_recovers_training_points():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(10, 1))
    y = X[:, 0] ** 2

    gp = GaussianProcessRegressor().fit(X, y)
    mean, std = gp.predict(X)

    assert np.allclose(mean, y, atol=0.5)
    assert np.all(std >= 0.0)


def test_gp_regressor_uncertainty_grows_away_from_data():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(6, 1))
    y = X[:, 0] ** 2

    gp = GaussianProcessRegressor().fit(X, y)
    _, std_near = gp.predict(np.array([[0.0]]))
    _, std_far = gp.predict(np.array([[10.0]]))

    assert std_far[0] > std_near[0]


def test_expected_improvement_favors_lower_mean_and_higher_uncertainty():
    mean = np.array([0.0, 0.0, -5.0])
    std = np.array([1.0, 3.0, 1.0])
    ei = expected_improvement(mean, std, best_f=1.0)

    # Same mean, higher std -> more expected improvement (candidate index 1)
    assert ei[1] > ei[0]
    # Much lower mean -> highest expected improvement (candidate index 2)
    assert ei[2] > ei[1]
    assert np.all(ei >= 0.0)


def test_bayesian_optimizer_finds_quadratic_minimum():
    target = np.array([1.0, -1.5])

    def f(x):
        return float(np.sum((x - target) ** 2))

    opt = BayesianKinematicOptimizer(
        objective_fn=f,
        lower_bounds=np.array([-5.0, -5.0]),
        upper_bounds=np.array([5.0, 5.0]),
        n_initial=6,
        n_iterations=20,
        n_candidates=300,
        random_seed=42,
    )
    result = opt.optimize()

    assert np.allclose(result.x, target, atol=0.5)
    assert result.fun < 0.25
    assert result.n_evals == 6 + 20
    assert result.X_history.shape == (26, 2)
    assert result.y_history.shape == (26,)


def test_bayesian_optimizer_rejects_mismatched_bounds():
    def f(x):
        return float(np.sum(x ** 2))

    try:
        BayesianKinematicOptimizer(
            objective_fn=f,
            lower_bounds=np.array([-1.0, -1.0]),
            upper_bounds=np.array([1.0]),
        )
        assert False, "expected ValueError for mismatched bounds"
    except ValueError:
        pass
