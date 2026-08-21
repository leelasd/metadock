"""
Pluggable scoring-function interface for openmm-dock.

Mirrors OpenDock's BaseScoringFunction extension pattern
(https://github.com/guyuehuo/opendock, scorer/scoring_function.py and
scorer/README.md): a common `.score(coords)` interface that lets a
correction term (e.g. a future ML rescoring layer, in the style of
OpenDock's OnionNet-SFCT weighted correction) be composed with the existing
OpenMM physical energy without touching the sampling engines themselves.

How to add a custom scorer
---------------------------
Subclass BaseScoringFunction and implement `score(lig_coords, rec_coords=None)`,
returning a scalar where LOWER is better (matching the convention used
throughout this codebase's OpenMM-based engines):

    class ContactCountScore(BaseScoringFunction):
        name = "contact_count"

        def __init__(self, pocket_coords, cutoff=5.0, **kwargs):
            super().__init__(**kwargs)
            self.pocket_coords = pocket_coords
            self.cutoff = cutoff

        def score(self, lig_coords, rec_coords=None):
            d = np.linalg.norm(
                lig_coords[:, None, :] - self.pocket_coords[None, :, :], axis=-1
            )
            return -float(np.sum(d <= self.cutoff))  # more contacts = lower (better) score

Then combine it with the physical score, each carrying its own weight:

    combo = CompositeScoringFunction([
        OpenMMPhysicalScore(context, engine._full_positions_from_coords, lig_start, weight=1.0),
        ContactCountScore(pocket_coords, weight=0.8),
    ])
    total = combo.score(lig_coords, rec_coords)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import openmm as mm
from openmm import unit


class BaseScoringFunction(ABC):
    """Common interface every scoring function in this codebase can implement."""
    name: str = "base"

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def score(self, lig_coords: np.ndarray, rec_coords: Optional[np.ndarray] = None) -> float:
        """Returns a scalar score for the given pose. Lower is better."""
        raise NotImplementedError

    def __call__(self, lig_coords: np.ndarray, rec_coords: Optional[np.ndarray] = None) -> float:
        return self.score(lig_coords, rec_coords)


class OpenMMPhysicalScore(BaseScoringFunction):
    """
    Wraps an existing OpenMM Context + full-position builder (the pattern
    used throughout engine.py, unified_kinematic_pso.py,
    collaborative_kinematic_metadynamics.py, ...) as a BaseScoringFunction,
    so the real physical energy this codebase already computes can be
    composed with other scorers via CompositeScoringFunction.
    """
    name = "openmm_physical"

    def __init__(
        self,
        context: mm.Context,
        full_positions_from_coords,
        lig_start: int = 0,
        weight: float = 1.0
    ):
        super().__init__(weight=weight)
        self.context = context
        self._full_positions_from_coords = full_positions_from_coords
        self.lig_start = lig_start

    def score(self, lig_coords: np.ndarray, rec_coords: Optional[np.ndarray] = None) -> float:
        full_pos = self._full_positions_from_coords(lig_coords)
        if rec_coords is not None:
            for idx in range(min(len(rec_coords), self.lig_start)):
                full_pos[idx] = mm.Vec3(*[float(c) for c in rec_coords[idx]]) * unit.angstroms
        self.context.setPositions(full_pos)
        state = self.context.getState(getEnergy=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))


class CompositeScoringFunction(BaseScoringFunction):
    """
    Sums a list of weighted BaseScoringFunctions into one scalar score -- the
    same pattern OpenDock uses for its OnionNet-SFCT correction term (a
    weighted addition on top of a base score, suggested weight 0.8 there).
    Each sub-scorer's own `.weight` is used, so weighting is set once, in one
    place, on the scorer itself.
    """
    name = "composite"

    def __init__(self, scorers: List[BaseScoringFunction]):
        super().__init__(weight=1.0)
        self.scorers = scorers

    def score(self, lig_coords: np.ndarray, rec_coords: Optional[np.ndarray] = None) -> float:
        return float(sum(s.weight * s.score(lig_coords, rec_coords) for s in self.scorers))

    def breakdown(self, lig_coords: np.ndarray, rec_coords: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Per-term weighted scores, useful for debugging which term dominates."""
        return {s.name: s.weight * s.score(lig_coords, rec_coords) for s in self.scorers}
