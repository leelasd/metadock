"""
Bayesian optimization sampler for openmm-dock's kinematic parameter spaces.

Mirrors OpenDock's BayesianOptimizerSampler
(https://github.com/guyuehuo/opendock, sampler/bayesian.py), which docks via
a Gaussian-Process surrogate over the ligand/receptor conformation vector.
This implementation is self-contained (numpy + scipy only) since this
project's dependency set is deliberately kept small (pyproject.toml:
openmm, rdkit, numpy, scipy, pandas) -- no scikit-learn dependency is added.

Useful when the scoring function is expensive relative to the dimensionality
of the search (e.g. a slow ML rescoring correction layered on top of the
OpenMM physical score via scoring_function.CompositeScoringFunction):
Bayesian optimization typically spends far fewer objective evaluations than
PSO/GA to find a good minimum in that regime, at the cost of being
effectively sequential (each evaluation informs the next) rather than
trivially parallel like a swarm.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize as scipy_minimize
from scipy.spatial.distance import cdist

ObjectiveFn = Callable[[np.ndarray], float]


class GaussianProcessRegressor:
    """
    Minimal GP regressor with a Matern-5/2 kernel. Hyperparameters
    (length_scale, signal_variance) are fit by maximizing the marginal
    log-likelihood via a handful of Nelder-Mead restarts -- deliberately
    simple rather than a general-purpose GP library, since this only needs
    to support low-dimensional (10s of dims), small-N (10s to ~100 points)
    Bayesian optimization inner loops.
    """
    def __init__(self, noise: float = 1e-6):
        self.noise = noise
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.length_scale_: float = 1.0
        self.signal_var_: float = 1.0
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    @staticmethod
    def _matern52(d: np.ndarray, length_scale: float) -> np.ndarray:
        r = np.sqrt(5.0) * d / length_scale
        return (1.0 + r + (r ** 2) / 3.0) * np.exp(-r)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float) -> np.ndarray:
        d = cdist(X1, X2, metric="euclidean")
        return signal_var * self._matern52(d, length_scale)

    def _neg_log_marginal_likelihood(self, log_params: np.ndarray) -> float:
        length_scale, signal_var = np.exp(log_params)
        K = self._kernel(self.X_, self.X_, length_scale, signal_var) + self.noise * np.eye(len(self.X_))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_))
        nll = 0.5 * self.y_.dot(alpha) + np.sum(np.log(np.diag(L))) + 0.5 * len(self.y_) * np.log(2 * np.pi)
        return float(nll)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        self.X_ = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) + 1e-8
        self.y_ = (y - self.y_mean_) / self.y_std_

        best_nll = np.inf
        best_params = np.log([1.0, 1.0])
        rng = np.random.default_rng(0)
        for _ in range(4):
            x0 = np.log([rng.uniform(0.3, 3.0), rng.uniform(0.3, 3.0)])
            res = scipy_minimize(self._neg_log_marginal_likelihood, x0, method="Nelder-Mead")
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        self.length_scale_, self.signal_var_ = np.exp(best_params)

        K = self._kernel(self.X_, self.X_, self.length_scale_, self.signal_var_) + self.noise * np.eye(len(self.X_))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, self.y_))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std), both in the original y units."""
        X = np.asarray(X, dtype=np.float64)
        K_s = self._kernel(X, self.X_, self.length_scale_, self.signal_var_)
        mean = K_s.dot(self._alpha)
        v = np.linalg.solve(self._L, K_s.T)
        K_ss_diag = self.signal_var_ * np.ones(len(X))
        var = np.maximum(K_ss_diag - np.sum(v ** 2, axis=0), 1e-12)
        std = np.sqrt(var)
        return mean * self.y_std_ + self.y_mean_, std * self.y_std_


def expected_improvement(mean: np.ndarray, std: np.ndarray, best_f: float, xi: float = 0.01) -> np.ndarray:
    """EI acquisition for MINIMIZATION: reward candidates predicted to beat
    the current best by at least xi, weighted by how confident the GP is."""
    std = np.maximum(std, 1e-9)
    imp = best_f - mean - xi
    z = imp / std
    ei = imp * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0.0)


@dataclass
class BayesianOptResult:
    x: np.ndarray
    fun: float
    X_history: np.ndarray
    y_history: np.ndarray
    n_evals: int


class BayesianKinematicOptimizer:
    """
    Bayesian optimization over a black-box kinematic-space objective (any of
    this codebase's evaluate_kinematics/evaluate_coupled_state-style
    functions, or a scoring_function.BaseScoringFunction, wrapped as a plain
    Callable[[np.ndarray], float]).
    """
    def __init__(
        self,
        objective_fn: ObjectiveFn,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        n_initial: int = 8,
        n_iterations: int = 25,
        n_candidates: int = 500,
        xi: float = 0.01,
        random_seed: Optional[int] = None
    ):
        self.objective_fn = objective_fn
        self.lb = np.asarray(lower_bounds, dtype=np.float64)
        self.ub = np.asarray(upper_bounds, dtype=np.float64)
        if len(self.lb) != len(self.ub):
            raise ValueError("lower_bounds and upper_bounds must have the same length")
        self.dim = len(self.lb)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.xi = xi
        self.rng = np.random.default_rng(random_seed)

    def _random_points(self, n: int) -> np.ndarray:
        u = self.rng.uniform(size=(n, self.dim))
        return self.lb + u * (self.ub - self.lb)

    def optimize(self) -> BayesianOptResult:
        # 1. Initial design: random sampling within bounds
        X = self._random_points(self.n_initial)
        y = np.array([self.objective_fn(x) for x in X])
        n_evals = self.n_initial

        gp = GaussianProcessRegressor()

        # 2. Sequential GP-fit + Expected-Improvement acquisition
        for _ in range(self.n_iterations):
            gp.fit(X, y)

            candidates = self._random_points(self.n_candidates)
            mean, std = gp.predict(candidates)
            best_f = float(np.min(y))
            ei = expected_improvement(mean, std, best_f, xi=self.xi)
            next_x = candidates[int(np.argmax(ei))]

            next_y = self.objective_fn(next_x)
            n_evals += 1
            X = np.vstack([X, next_x])
            y = np.append(y, next_y)

        best_idx = int(np.argmin(y))
        return BayesianOptResult(x=X[best_idx], fun=float(y[best_idx]), X_history=X, y_history=y, n_evals=n_evals)
