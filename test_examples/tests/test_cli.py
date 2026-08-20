# tests/test_cli.py
import json
import pytest
from click.testing import CliRunner
from scoring_optimizer.cli import main


def test_cli_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run-optimizer" in result.output


def test_run_optimizer_writes_output_files(tmp_path, mini_poses_sdf, mini_sar_csv):
    result = CliRunner().invoke(main, [
        "run-optimizer",
        "--poses", str(mini_poses_sdf),
        "--sar",   str(mini_sar_csv),
        "--output-dir", str(tmp_path),
        "--n-trials", "10",
        "--alpha", "0.0", "--beta", "0.5", "--gamma", "0.5",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "RbtInterIdxSF.prm").exists()
    assert (tmp_path / "cavity.prm").exists()
    assert (tmp_path / "metrics.json").exists()


def test_metrics_json_has_required_keys(tmp_path, mini_poses_sdf, mini_sar_csv):
    CliRunner().invoke(main, [
        "run-optimizer",
        "--poses", str(mini_poses_sdf),
        "--sar",   str(mini_sar_csv),
        "--output-dir", str(tmp_path),
        "--n-trials", "10",
        "--alpha", "0.0", "--beta", "0.5", "--gamma", "0.5",
    ])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for key in ("holdout_auc", "holdout_spearman", "best_objective", "weights",
                "n_compounds", "n_crystal_matched", "n_trials"):
        assert key in metrics, f"Missing metrics.json key: {key}"


def test_metrics_json_sys_weights_present(tmp_path, mini_poses_sdf, mini_sar_csv):
    """sys_vdw_weight and sys_pol_weight are optimized but not written to .prm files."""
    CliRunner().invoke(main, [
        "run-optimizer",
        "--poses", str(mini_poses_sdf),
        "--sar",   str(mini_sar_csv),
        "--output-dir", str(tmp_path),
        "--n-trials", "10",
        "--alpha", "0.0", "--beta", "0.5", "--gamma", "0.5",
    ])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert "sys_vdw_weight" in metrics["weights"]
    assert "sys_pol_weight" in metrics["weights"]
