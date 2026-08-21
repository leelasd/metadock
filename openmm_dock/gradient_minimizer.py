"""
Gradient-based minimization over reduced kinematic parameter spaces for
openmm-dock.

The docking engines here (UnifiedKinematicPSOEngine, CollaborativeKinematicMetaDEngine,
GlobalBlindDockingEngine) already expose black-box energy evaluators over a
small kinematic parameter vector (6 rigid-body DOF + a handful of ring /
exocyclic / chi dihedrals -- typically 10-60 dimensions), but only ever
optimize it with derivative-free swarm/MC search, then hand off to OpenMM's
own Cartesian-space L-BFGS for final polish.

This module adds a genuine gradient-based optimizer that works directly in
the reduced kinematic space itself, using central finite-difference
gradients of the (cheap, low-dimensional) objective. This is the numerical
counterpart to what OpenDock offers via PyTorch autograd + Adam/SGD/LBFGS
(sampler/minimizer.py) -- it does not require reimplementing the OpenMM
energy function in a differentiable framework; the existing black-box
objective (any evaluate_kinematics/evaluate_coupled_state-style callable, or
a BaseScoringFunction from scoring_function.py) is called as-is.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union
import numpy as np
from scipy.optimize import minimize as scipy_minimize

ObjectiveFn = Callable[[np.ndarray], float]


def central_difference_gradient(
    fn: ObjectiveFn,
    x: np.ndarray,
    step: Union[float, np.ndarray] = 1e-4
) -> np.ndarray:
    """
    Central finite-difference gradient of a scalar black-box objective.
    `step` may be a scalar (uniform) or a per-dimension array -- e.g. a
    larger step for translations in Angstrom, a smaller one for angles in
    radians, matching the very different natural scales of a kinematic
    parameter vector.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    step_vec = np.full(n, step, dtype=np.float64) if np.isscalar(step) else np.asarray(step, dtype=np.float64)
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = step_vec[i]
        f_plus = fn(x + dx)
        f_minus = fn(x - dx)
        grad[i] = (f_plus - f_minus) / (2.0 * step_vec[i])
    return grad


@dataclass
class MinimizerResult:
    x: np.ndarray
    fun: float
    history: List[float] = field(default_factory=list)
    n_evals: int = 0
    converged: bool = False


def adam_minimize(
    fn: ObjectiveFn,
    x0: np.ndarray,
    step_size: Union[float, np.ndarray] = 1e-4,
    lr: float = 0.05,
    n_iterations: int = 100,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_clip: Optional[float] = 5.0,
    tol: float = 1e-5,
    patience: int = 8
) -> MinimizerResult:
    """
    Adam gradient descent directly over a black-box kinematic-space
    objective, using central finite-difference gradients. Mirrors OpenDock's
    Adam minimizer option but works with any callable objective from this
    codebase rather than requiring PyTorch autograd.

    grad_clip guards against the occasional huge gradient spike near a steric
    clash (unminimized OpenMM energies can be very steep there); set to None
    to disable.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    n = len(x)
    m = np.zeros(n)
    v = np.zeros(n)
    history: List[float] = []
    n_evals = 0
    best_f = fn(x)
    n_evals += 1
    history.append(best_f)
    stall_count = 0
    converged = False

    for t in range(1, n_iterations + 1):
        grad = central_difference_gradient(fn, x, step=step_size)
        n_evals += 2 * n
        if grad_clip is not None:
            gnorm = np.linalg.norm(grad)
            if gnorm > grad_clip:
                grad = grad * (grad_clip / gnorm)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

        f_val = fn(x)
        n_evals += 1
        history.append(f_val)

        if best_f - f_val > tol:
            best_f = f_val
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= patience:
                converged = True
                break

    return MinimizerResult(x=x, fun=history[-1], history=history, n_evals=n_evals, converged=converged)


def lbfgs_minimize(
    fn: ObjectiveFn,
    x0: np.ndarray,
    step_size: Union[float, np.ndarray] = 1e-4,
    max_iterations: int = 100,
    bounds: Optional[List[tuple]] = None
) -> MinimizerResult:
    """
    L-BFGS-B minimization over a black-box kinematic-space objective, using
    central finite-difference gradients (via scipy.optimize.minimize). This
    is the second-order counterpart to adam_minimize -- typically converges
    in fewer objective evaluations for a smooth, low-noise objective.
    """
    history: List[float] = []
    n_evals = 0

    def _fn_and_grad(x):
        nonlocal n_evals
        f = fn(x)
        n_evals += 1
        history.append(f)
        grad = central_difference_gradient(fn, x, step=step_size)
        n_evals += 2 * len(x)
        return f, grad

    res = scipy_minimize(
        _fn_and_grad, x0, jac=True, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": max_iterations}
    )
    return MinimizerResult(x=res.x, fun=float(res.fun), history=history, n_evals=n_evals, converged=bool(res.success))
