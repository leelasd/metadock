# scoring_optimizer/pose_parser.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

SCORE_FIELDS = [
    "SCORE",
    "SCORE.INTER.VDW",
    "SCORE.INTER.POLAR",
    "SCORE.INTER.REPUL",
    "SCORE.INTER.CONST",
    "SCORE.INTER.ROT",
    "SCORE.RESTR",
    "SCORE.RESTR.CAVITY",
]

SCORE_FIELDS_SOLVENT = [
    "SCORE.SYSTEM.VDW",
    "SCORE.SYSTEM.POLAR",
]

ALL_SCORE_FIELDS = SCORE_FIELDS + SCORE_FIELDS_SOLVENT


@dataclass
class Pose:
    name: str
    scores: dict[str, float] = field(default_factory=dict)
    coords: np.ndarray | None = None  # shape (n_atoms, 3), heavy atoms only


def _parse_mol_block_coords(mol_block_lines: list[str]) -> np.ndarray | None:
    """Extract heavy-atom xyz from a V2000 mol block (lines before 'M  END')."""
    try:
        counts_line = mol_block_lines[3]
        n_atoms = int(counts_line[:3].strip())
        coords = []
        for i in range(4, 4 + n_atoms):
            parts = mol_block_lines[i].split()
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(coords)
    except (IndexError, ValueError):
        return None


def parse_poses(sdf_path: Path) -> list[Pose]:
    """Parse rDock output SDF and return one Pose per record."""
    poses: list[Pose] = []
    mol_block_lines: list[str] = []
    current_scores: dict[str, float] = {}
    current_name: str | None = None
    current_field: str | None = None
    in_mol_block = True

    with open(sdf_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line == "$$$$":
                coords = _parse_mol_block_coords(mol_block_lines) if mol_block_lines else None
                if current_name is not None:
                    poses.append(Pose(
                        name=current_name,
                        scores=dict(current_scores),
                        coords=coords,
                    ))
                mol_block_lines = []
                current_scores = {}
                current_name = None
                current_field = None
                in_mol_block = True
                continue

            if in_mol_block:
                mol_block_lines.append(line)
                if line.startswith("M  END"):
                    in_mol_block = False
                continue

            if line.startswith(">  <"):
                current_field = line.rstrip()[4:].rstrip(">").strip()
            elif current_field == "Name":
                current_name = line.strip()
                current_field = None
            elif current_field in ALL_SCORE_FIELDS:
                try:
                    current_scores[current_field] = float(line.strip())
                except ValueError:
                    pass
                current_field = None

    return poses


def group_by_compound(poses: list[Pose]) -> dict[str, list[Pose]]:
    """Group poses by compound name."""
    groups: dict[str, list[Pose]] = {}
    for pose in poses:
        groups.setdefault(pose.name, []).append(pose)
    return groups


def top_pose(poses: list[Pose]) -> Pose:
    """Return the pose with the lowest (best) SCORE."""
    if not poses:
        raise ValueError("top_pose() called with an empty pose list")
    return min(poses, key=lambda p: p.scores.get("SCORE", float("inf")))
