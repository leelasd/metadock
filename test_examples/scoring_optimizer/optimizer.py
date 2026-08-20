# scoring_optimizer/optimizer.py
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from .pose_parser import parse_poses, group_by_compound, Pose
from .metrics import enrichment_auc, potency_spearman, rmsd_loss, composite_objective

optuna.logging.set_verbosity(optuna.logging.WARNING)

_logger = logging.getLogger(__name__)

TIER1_PARAMS: dict[str, tuple[float, float]] = {
    "vdw_weight":     (0.1, 3.0),
    "polar_weight":   (0.5, 8.0),
    "repul_weight":   (1.0, 10.0),
    "const_weight":   (1.0, 10.0),
    "rot_weight":     (0.1, 3.0),
    "pharma_weight":  (0.5, 5.0),
    "cavity_weight":  (0.5, 3.0),
    "sys_vdw_weight": (0.1, 3.0),
    "sys_pol_weight": (0.1, 3.0),
}

# Maps weight parameter names to rDock SDF score fields
FIELD_MAP = {
    "vdw_weight":     "SCORE.INTER.VDW",
    "polar_weight":   "SCORE.INTER.POLAR",
    "repul_weight":   "SCORE.INTER.REPUL",
    "const_weight":   "SCORE.INTER.CONST",
    "rot_weight":     "SCORE.INTER.ROT",
    "pharma_weight":  "SCORE.RESTR",
    "cavity_weight":  "SCORE.RESTR.CAVITY",
    "sys_vdw_weight": "SCORE.SYSTEM.VDW",
    "sys_pol_weight": "SCORE.SYSTEM.POLAR",
}


def build_train_holdout_split(
    sar_df: pd.DataFrame,
    date_col: str = "assay_date",
    holdout_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split SAR data by assay date — most recent holdout_frac is holdout."""
    sorted_df = sar_df.sort_values(date_col).reset_index(drop=True)
    split = max(1, int(len(sorted_df) * (1.0 - holdout_frac)))
    return sorted_df.iloc[:split].copy(), sorted_df.iloc[split:].copy()


def _reweighted_score(pose: Pose, weights: dict) -> float:
    """Compute weighted sum of score components for a single pose."""
    return sum(
        weights[w] * pose.scores.get(FIELD_MAP[w], 0.0)
        for w in TIER1_PARAMS
    )


def _top_pose_per_compound(
    poses_by_compound: dict[str, list[Pose]],
    weights: dict,
) -> dict[str, Pose]:
    """For each compound, find the pose with the lowest reweighted score."""
    return {
        name: min(poses, key=lambda p: _reweighted_score(p, weights))
        for name, poses in poses_by_compound.items()
    }


def run_optimization(
    poses_sdf: Path,
    sar_df: pd.DataFrame,
    crystal_coords_map: dict[str, np.ndarray],  # compound name -> crystal heavy-atom coords
    alpha: float,
    beta: float,
    gamma: float,
    n_trials: int = 500,
) -> dict:
    """
    Run Optuna Tier 1 weight optimization.

    crystal_coords_map: for each compound with a crystal structure, provide the
    heavy-atom coordinates of the co-crystal ligand as an (n_atoms, 3) array.
    These are used to compute RMSD_loss — the RMSD between the top-scored pose
    (under the current trial weights) and the crystal reference.

    Returns dict with: weights, holdout_auc, holdout_spearman, best_objective.
    holdout_auc and holdout_spearman may be nan if the holdout SAR rows have no
    corresponding docked poses.
    """
    all_poses = parse_poses(poses_sdf)
    poses_by_compound = group_by_compound(all_poses)

    train_df, holdout_df = build_train_holdout_split(sar_df)

    n_total = len(sar_df)
    n_crystal = len(crystal_coords_map)
    alpha_eff = alpha * (n_crystal / n_total) if n_total > 0 else 0.0

    def objective(trial: optuna.Trial) -> float:
        weights = {
            name: trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in TIER1_PARAMS.items()
        }

        top_poses = _top_pose_per_compound(poses_by_compound, weights)

        # RMSD loss: recompute per trial — different weights -> different top pose
        rmsd_values = []
        for name, crystal_coords in crystal_coords_map.items():
            if name in top_poses and top_poses[name].coords is not None:
                try:
                    r = rmsd_loss(top_poses[name].coords, crystal_coords)
                    rmsd_values.append(r)
                except (ValueError, AssertionError):
                    pass  # shape mismatch (AssertionError from rmsd_loss 2-D assert) or bad data
        mean_rmsd = float(np.mean(rmsd_values)) if rmsd_values else 0.0

        # Enrichment and potency on training split
        train_rows = train_df[train_df["name"].isin(top_poses)]
        if train_rows.empty:
            return float("inf")

        train_scores = np.array([
            _reweighted_score(top_poses[n], weights)
            for n in train_rows["name"]
        ])
        train_labels = train_rows["active"].to_numpy()
        train_pic50  = train_rows["pic50"].to_numpy()

        if train_labels.sum() == 0 or (1 - train_labels).sum() == 0:
            return float("inf")

        auc = enrichment_auc(train_scores, train_labels)
        rho = potency_spearman(train_scores, train_pic50)

        return composite_objective(mean_rmsd, auc, rho, alpha_eff, beta, gamma)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_weights = study.best_params

    # Evaluate on holdout
    top_poses = _top_pose_per_compound(poses_by_compound, best_weights)
    holdout_rows = holdout_df[holdout_df["name"].isin(top_poses)]

    holdout_auc = float("nan")
    holdout_spearman = float("nan")

    if holdout_rows.empty:
        _logger.warning(
            "Holdout split has no compounds with docked poses — "
            "holdout_auc and holdout_spearman will be nan"
        )

    if not holdout_rows.empty:
        h_scores = np.array([_reweighted_score(top_poses[n], best_weights) for n in holdout_rows["name"]])
        h_labels = holdout_rows["active"].to_numpy()
        h_pic50  = holdout_rows["pic50"].to_numpy()
        if h_labels.sum() > 0 and (1 - h_labels).sum() > 0:
            holdout_auc = enrichment_auc(h_scores, h_labels)
        holdout_spearman = potency_spearman(h_scores, h_pic50)

    return {
        "weights":           best_weights,
        "holdout_auc":       holdout_auc,
        "holdout_spearman":  holdout_spearman,
        "best_objective":    study.best_value,
    }
