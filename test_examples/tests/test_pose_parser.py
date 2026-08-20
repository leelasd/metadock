# tests/test_pose_parser.py
import numpy as np
import pytest
from scoring_optimizer.pose_parser import parse_poses, group_by_compound, top_pose, SCORE_FIELDS, SCORE_FIELDS_SOLVENT


def test_parse_returns_pose_objects(mini_poses_sdf):
    poses = parse_poses(mini_poses_sdf)
    assert len(poses) == 15  # 5 compounds × 3 poses


def test_pose_has_name(mini_poses_sdf):
    poses = parse_poses(mini_poses_sdf)
    assert poses[0].name == "CPD001"


def test_pose_has_all_score_fields(mini_poses_sdf):
    poses = parse_poses(mini_poses_sdf)
    for field in SCORE_FIELDS:
        assert field in poses[0].scores, f"Missing field: {field}"


def test_pose_scores_are_floats(mini_poses_sdf):
    poses = parse_poses(mini_poses_sdf)
    for field, val in poses[0].scores.items():
        assert isinstance(val, float), f"{field} is not float: {val!r}"


def test_pose_has_3d_coords(mini_poses_sdf):
    poses = parse_poses(mini_poses_sdf)
    assert poses[0].coords is not None
    assert poses[0].coords.shape[0] > 0
    assert poses[0].coords.shape[1] == 3  # (n_atoms, 3)


def test_poses_grouped_by_compound(mini_poses_sdf):
    groups = group_by_compound(parse_poses(mini_poses_sdf))
    assert set(groups.keys()) == {"CPD001", "CPD002", "CPD003", "CPD005", "CPD006"}
    assert all(len(v) == 3 for v in groups.values())


def test_top_pose_is_lowest_score(mini_poses_sdf):
    groups = group_by_compound(parse_poses(mini_poses_sdf))
    for name, poses in groups.items():
        best = top_pose(poses)
        assert best.scores["SCORE"] == min(p.scores["SCORE"] for p in poses)


def test_solvent_fields_present_in_solvent_fixture(mini_poses_sdf):
    poses = parse_poses(mini_poses_sdf)
    for field in SCORE_FIELDS_SOLVENT:
        assert field in poses[0].scores, f"Missing solvent field: {field}"
