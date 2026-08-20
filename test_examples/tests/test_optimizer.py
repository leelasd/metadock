# tests/test_optimizer.py
import numpy as np
import pytest
import pandas as pd
from pathlib import Path
from scoring_optimizer.optimizer import (
    build_train_holdout_split,
    run_optimization,
    TIER1_PARAMS,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_tier1_params_has_nine_entries():
    assert len(TIER1_PARAMS) == 9


def test_train_holdout_split_sizes(mini_sar_df):
    train, holdout = build_train_holdout_split(mini_sar_df)
    assert len(train) + len(holdout) == len(mini_sar_df)
    assert len(holdout) >= 1


def test_train_holdout_no_overlap(mini_sar_df):
    train, holdout = build_train_holdout_split(mini_sar_df)
    assert set(train.index).isdisjoint(set(holdout.index))


def test_holdout_is_most_recent(mini_sar_df):
    train, holdout = build_train_holdout_split(mini_sar_df)
    # All training dates must be <= all holdout dates
    assert train["assay_date"].max() <= holdout["assay_date"].min()


def test_run_optimization_returns_required_keys(mini_poses_sdf, mini_sar_df):
    result = run_optimization(
        poses_sdf=mini_poses_sdf,
        sar_df=mini_sar_df,
        crystal_coords_map={},
        alpha=0.0, beta=0.5, gamma=0.5,
        n_trials=10,
    )
    for key in ("weights", "holdout_auc", "holdout_spearman", "best_objective"):
        assert key in result


def test_run_optimization_weights_in_bounds(mini_poses_sdf, mini_sar_df):
    result = run_optimization(
        poses_sdf=mini_poses_sdf,
        sar_df=mini_sar_df,
        crystal_coords_map={},
        alpha=0.0, beta=0.5, gamma=0.5,
        n_trials=10,
    )
    for name, (lo, hi) in TIER1_PARAMS.items():
        assert lo <= result["weights"][name] <= hi, f"{name} out of bounds"


def test_run_optimization_holdout_metrics_not_nan(mini_poses_sdf, mini_sar_df):
    result = run_optimization(
        poses_sdf=mini_poses_sdf,
        sar_df=mini_sar_df,
        crystal_coords_map={},
        alpha=0.0, beta=0.5, gamma=0.5,
        n_trials=10,
    )
    import math
    # CPD005 (active) and CPD006 (inactive) are in holdout and have poses
    assert not math.isnan(result["holdout_auc"]), "holdout_auc should not be nan"
    assert not math.isnan(result["holdout_spearman"]), "holdout_spearman should not be nan"


def test_run_optimization_with_crystal_coords(mini_poses_sdf, mini_sar_df):
    # Provide fake crystal coords for CPD001 — shape must match pose coords
    from scoring_optimizer.pose_parser import parse_poses, group_by_compound, top_pose
    poses = parse_poses(mini_poses_sdf)
    groups = group_by_compound(poses)
    crystal_shape = groups["CPD001"][0].coords
    crystal_coords_map = {"CPD001": crystal_shape}  # same coords -> RMSD = 0
    result = run_optimization(
        poses_sdf=mini_poses_sdf,
        sar_df=mini_sar_df,
        crystal_coords_map=crystal_coords_map,
        alpha=0.3, beta=0.4, gamma=0.3,
        n_trials=10,
    )
    assert "weights" in result
