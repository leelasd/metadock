"""
Glowworm Swarm Optimization (GSO) for rigid-body protein-protein docking,
implementing the algorithm behind LightDock (github.com/lightdock/lightdock,
GPLv3 -- this module is an independent re-implementation from reading their
published algorithm description and source for understanding, not a port of
their GPLv3 code, to keep this repository's own licensing unencumbered).

Reference: Krishnanand, K.N. and Ghose, D. (2009). "Glowworm swarm
optimization for simultaneous capture of multiple local optima of
multimodal functions." Swarm Intelligence, 3(2), 87-124.

Algorithm (per LightDock's gso/algorithm.py, gso/glowworm.py):
  1. Each glowworm carries a rigid-body pose (translation + quaternion
     orientation) of the MOBILE partner relative to a fixed receptor, plus a
     scalar "luciferin" brightness value.
  2. Each step: evaluate every glowworm's score, update luciferin
     (`luciferin = (1-rho)*luciferin + gamma*brightness`), then each
     glowworm looks only at neighbors within its own local vision_range that
     are brighter, and probabilistically moves toward one (roulette-wheel
     selection weighted by luciferin difference). Vision range self-adapts
     to target a neighbor count (max_neighbors).
  3. Because movement is LOCAL (only neighbors within vision_range, not a
     single global best), the swarm naturally splits across multiple
     simultaneous local optima -- LightDock's key advantage over a
     single-global-best optimizer (standard PSO) for docking, where several
     distinct binding modes can be plausible.
  4. LightDock's genuinely BLIND global coverage comes from swarm
     INITIALIZATION, not the optimizer: many independent swarms are seeded
     evenly around the receptor's entire surface before running GSO on each
     -- see generate_surface_swarm_centers.

Scoring here uses openmm-dock's own real OpenMM CustomNonbondedForce physics
(scoring.create_combined_search_force) rather than LightDock's default
statistical potentials (DFIRE, PISA, etc.) -- the same physics-over-
statistical-potential choice already made throughout this codebase for
small-molecule docking.

Simplification vs. full LightDock: this MVP is rigid-body only (no ANM
normal-mode flexibility for either partner).

Bonded-pair exclusions still matter for the ABSOLUTE energy scale even
though internal conformation never changes: without them, every covalent
bond within each partner (a ~0.15nm separation) is scored by the same
steep-at-short-range VDW/REPUL terms used for genuine inter-chain clashes,
adding a huge (~1e6 kJ/mol for a ~1500-atom pair of proteins) constant that,
while it truly is constant and so cannot change which pose ranks best,
buries the real inter-chain signal in a meaningless absolute number and is
not something worth tolerating when it's cheap to fix. _distance_bonded_pairs
below does simple distance-based bond perception (any pair under a cutoff)
to generate exclusions for both partners, receptor included -- unlike
engine.py's ligand torsion tree, which needs true graph bonds for kinematics,
here it is purely a nonbonded-scoring cleanup.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
import openmm as mm
from openmm import unit

from .core import MolecularSystem
from .scoring import ScoreWeights, create_combined_search_force

PoseEnergyFn = Callable[[np.ndarray, np.ndarray], float]  # (trans_A, quat_xyzw) -> energy


@dataclass
class GSOParameters:
    rho: float = 0.4                    # luciferin decay
    gamma: float = 0.6                  # luciferin gain from current score
    beta: float = 0.08                  # vision-range adaptation rate
    initial_luciferin: float = 5.0
    initial_vision_range: float = 10.0  # Angstroms (translation-distance neighbor metric)
    max_vision_range: float = 40.0      # Angstroms
    max_neighbors: int = 6
    step_translation: float = 0.6       # Angstroms per move
    step_rotation_deg: float = 8.0      # degrees per move


@dataclass
class Glowworm:
    glowworm_id: int
    trans: np.ndarray             # (3,) Angstroms, world-frame translation of mobile partner's centroid
    quat: np.ndarray              # (4,) [x, y, z, w]
    luciferin: float
    vision_range: float
    score: float = 0.0
    energy: float = 0.0


def generate_surface_swarm_centers(
    receptor_coords: np.ndarray,
    n_swarms: int,
    ligand_radius: float,
    surface_offset: float = 3.0,
) -> np.ndarray:
    """
    Distributes n_swarms points evenly over a sphere enclosing the receptor,
    offset outward by the mobile partner's own radius plus a contact gap --
    approximating LightDock's receptor-surface swarm placement (they use
    true SASA-derived surface points; this uses a Fibonacci sphere around
    the receptor's bounding sphere, a documented simplification that still
    gives genuinely blind, uniform all-around coverage without requiring a
    SASA calculation).
    """
    center = receptor_coords.mean(axis=0)
    receptor_radius = float(np.max(np.linalg.norm(receptor_coords - center, axis=1)))
    swarm_radius = receptor_radius + ligand_radius + surface_offset

    indices = np.arange(0, n_swarms, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n_swarms)
    golden_angle = np.pi * (1 + 5 ** 0.5)
    theta = golden_angle * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    unit_points = np.stack([x, y, z], axis=1)
    return center + unit_points * swarm_radius


def _distance_bonded_pairs(coords: np.ndarray, cutoff: float = 1.7) -> List[Tuple[int, int]]:
    """Simple distance-based bond perception: any atom pair closer than
    cutoff Angstroms is treated as bonded, for nonbonded-exclusion purposes
    only (see module docstring). O(n^2) but proteins here are small enough
    (<1000 atoms) that this is negligible next to the OpenMM system build."""
    pairs: List[Tuple[int, int]] = []
    n = len(coords)
    for i in range(n):
        d2 = np.sum((coords[i + 1:] - coords[i]) ** 2, axis=1)
        close = np.nonzero(d2 < cutoff ** 2)[0]
        for c in close:
            pairs.append((i, i + 1 + int(c)))
    return pairs


def build_protein_protein_system(
    receptor_sys: MolecularSystem,
    ligand_sys: MolecularSystem,
    weights: Optional[ScoreWeights] = None,
    platform_name: Optional[str] = None,
) -> Tuple[mm.System, mm.Context, mm.Integrator, int, int, np.ndarray]:
    """
    Builds a minimal OpenMM system for rigid-body protein-protein scoring:
    receptor atoms fixed (mass=0), mobile-partner atoms free (real mass, no
    internal bonded forces -- see module docstring for why that's safe for
    pure rigid-body sampling). Reuses scoring.create_combined_search_force
    exactly as engine.py's DockingEngine._build_system does for receptor-
    ligand docking; this is a generic function of per-atom (charge, sigma,
    epsilon, donor, acceptor, hyd, is_lig) parameters and doesn't care
    whether the "ligand" side is a small molecule or a whole protein chain.

    Returns (system, context, integrator, rec_n, lig_n, ligand_local_coords)
    where ligand_local_coords is the mobile partner's own coordinates
    recentered on its own centroid (the rigid body to be translated/rotated).
    """
    weights = weights or ScoreWeights()
    system = mm.System()

    rec_n = len(receptor_sys.atoms)
    lig_n = len(ligand_sys.atoms)

    for a in receptor_sys.atoms:
        system.addParticle(0.0 * unit.dalton)
    for a in ligand_sys.atoms:
        el = a.element.upper()
        m = 12.011 if el == "C" else (1.008 if el == "H" else (15.999 if el == "O" else (14.007 if el == "N" else 32.06)))
        system.addParticle(m * unit.dalton)

    nb_force = create_combined_search_force(weights)
    all_atoms = list(receptor_sys.atoms) + list(ligand_sys.atoms)
    for i, a in enumerate(all_atoms):
        is_lig = 1.0 if i >= rec_n else 0.0
        is_hyd = 1.0 if a.element.upper() in ["C", "CL", "BR", "I", "F"] and not a.is_polar else 0.0
        nb_force.addParticle([
            a.charge, a.sigma, a.epsilon,
            1.0 if a.is_donor else 0.0,
            1.0 if a.is_acceptor else 0.0,
            is_hyd, is_lig,
        ])
    for i, j in _distance_bonded_pairs(receptor_sys.coordinates):
        nb_force.addExclusion(i, j)
    for i, j in _distance_bonded_pairs(ligand_sys.coordinates):
        nb_force.addExclusion(rec_n + i, rec_n + j)

    for f in nb_force.forces:
        system.addForce(f)

    platform = mm.Platform.getPlatformByName(platform_name) if platform_name else None
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = mm.Context(system, integrator, platform) if platform else mm.Context(system, integrator)

    ligand_coords = ligand_sys.coordinates
    ligand_local = ligand_coords - ligand_coords.mean(axis=0)
    receptor_coords = receptor_sys.coordinates

    # Fixed receptor positions never change; set once up front, then only
    # the mobile-partner slice is updated per pose evaluation.
    full_pos0 = np.vstack([receptor_coords, ligand_coords]) * 0.1  # Angstrom -> nm
    context.setPositions(full_pos0)

    return system, context, integrator, rec_n, lig_n, ligand_local


def make_energy_fn(context: mm.Context, rec_n: int, ligand_local: np.ndarray) -> PoseEnergyFn:
    """Returns a (trans, quat) -> energy_kcal_per_mol closure over a prebuilt context."""
    def energy_fn(trans: np.ndarray, quat: np.ndarray) -> float:
        rot = ScipyRotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
        world_coords = ligand_local.dot(rot.T) + trans
        pos_nm = world_coords * 0.1
        context.setPositions(np.vstack([
            np.array(context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer))[:rec_n],
            pos_nm,
        ]))
        state = context.getState(getEnergy=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) * 0.239006)
    return energy_fn


class GlowwormSwarmOptimizer:
    """Runs the GSO update loop (luciferin update + local-neighborhood movement) over a swarm."""

    def __init__(self, energy_fn: PoseEnergyFn, params: GSOParameters):
        self.energy_fn = energy_fn
        self.params = params

    def initialize_swarm(
        self,
        swarm_centers: np.ndarray,
        n_per_swarm: int,
        rng: np.random.Generator,
        jitter: float = 2.0,
    ) -> List[Glowworm]:
        glowworms: List[Glowworm] = []
        gid = 0
        p = self.params
        for center in swarm_centers:
            for _ in range(n_per_swarm):
                trans = center + rng.normal(0.0, jitter, size=3)
                quat = ScipyRotation.random(random_state=rng).as_quat()
                glowworms.append(Glowworm(
                    glowworm_id=gid, trans=trans, quat=quat,
                    luciferin=p.initial_luciferin, vision_range=p.initial_vision_range,
                ))
                gid += 1
        return glowworms

    def run(self, glowworms: List[Glowworm], n_steps: int, rng: np.random.Generator) -> List[Glowworm]:
        p = self.params
        for _step in range(n_steps):
            for g in glowworms:
                g.energy = self.energy_fn(g.trans, g.quat)
                brightness = -g.energy
                g.luciferin = (1.0 - p.rho) * g.luciferin + p.gamma * brightness

            for g in glowworms:
                vr2 = g.vision_range ** 2
                neighbors = [
                    o for o in glowworms
                    if o.glowworm_id != g.glowworm_id
                    and o.luciferin > g.luciferin
                    and float(np.sum((o.trans - g.trans) ** 2)) < vr2
                ]
                if neighbors:
                    diffs = np.array([o.luciferin - g.luciferin for o in neighbors])
                    probs = diffs / diffs.sum()
                    chosen = neighbors[rng.choice(len(neighbors), p=probs)]

                    direction = chosen.trans - g.trans
                    dist = np.linalg.norm(direction)
                    if dist > 1e-9:
                        g.trans = g.trans + (direction / dist) * min(p.step_translation, dist)

                    key_rot, _ = ScipyRotation.align_vectors(
                        ScipyRotation.from_quat(chosen.quat).apply(np.eye(3)),
                        ScipyRotation.from_quat(g.quat).apply(np.eye(3)),
                    )
                    slerp_frac = min(1.0, p.step_rotation_deg / max(1e-6, key_rot.magnitude() * 180.0 / np.pi))
                    g.quat = (ScipyRotation.from_rotvec(key_rot.as_rotvec() * slerp_frac) * ScipyRotation.from_quat(g.quat)).as_quat()

                    n_found = len(neighbors)
                else:
                    n_found = 0

                g.vision_range = float(np.clip(
                    g.vision_range + p.beta * (p.max_neighbors - n_found),
                    0.0, p.max_vision_range,
                ))

        glowworms.sort(key=lambda gw: gw.energy)
        return glowworms
