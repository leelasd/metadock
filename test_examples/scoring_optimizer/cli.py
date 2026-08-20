# scoring_optimizer/cli.py
from __future__ import annotations
import json
import math
from pathlib import Path
import click
import pandas as pd
from .optimizer import run_optimization
from .config_writer import write_inter_sf_prm, write_cavity_prm


def _json_safe(v):
    """Convert NaN/inf to None for JSON serialization."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


@click.group()
def main():
    """rDock project-specific scoring optimizer."""


@main.command("run-optimizer")
@click.option("--poses",        required=True,  type=click.Path(exists=True, path_type=Path))
@click.option("--sar",          required=True,  type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir",   required=True,  type=click.Path(path_type=Path))
@click.option("--n-trials",     default=500,    show_default=True, type=int)
@click.option("--alpha",        default=0.5,    show_default=True, type=float)
@click.option("--beta",         default=0.3,    show_default=True, type=float)
@click.option("--gamma",        default=0.2,    show_default=True, type=float)
@click.option("--receptor",     default="receptor.mol2", show_default=True)
@click.option("--ref-mol",      default="xtal-lig.sd",  show_default=True)
@click.option("--pharma-restr", default="pharma.restr", show_default=True)
def run_optimizer_cmd(
    poses, sar, output_dir, n_trials,
    alpha, beta, gamma,
    receptor, ref_mol, pharma_restr,
):
    """Optimize Tier 1 rDock weights from docked poses and SAR data.

    SAR CSV must have columns: smiles, name, pic50, active, assay_date (ISO 8601 dates).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sar_df = pd.read_csv(sar, parse_dates=["assay_date"])
    click.echo(f"Loaded {len(sar_df)} SAR compounds")

    result = run_optimization(
        poses_sdf=poses,
        sar_df=sar_df,
        crystal_coords_map={},   # populated by crystal_processing in Plan 2 (AWS)
        alpha=alpha, beta=beta, gamma=gamma,
        n_trials=n_trials,
    )

    write_inter_sf_prm(result["weights"], output_dir / "RbtInterIdxSF.prm")
    write_cavity_prm(
        result["weights"],
        output_dir / "cavity.prm",
        title="Optimized",
        receptor_file=receptor,
        ref_mol=ref_mol,
        pharma_restr_file=pharma_restr,
        waters=[],
    )

    metrics = {
        "holdout_auc":       _json_safe(result["holdout_auc"]),
        "holdout_spearman":  _json_safe(result["holdout_spearman"]),
        "best_objective":    _json_safe(result["best_objective"]),
        "weights":           result["weights"],
        "n_compounds":       len(sar_df),
        "n_crystal_matched": 0,   # populated when crystal_coords_map is non-empty
        "n_trials":          n_trials,
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    auc_str = "nan" if _json_safe(result['holdout_auc']) is None else f"{result['holdout_auc']:.3f}"
    rho_str = "nan" if _json_safe(result['holdout_spearman']) is None else f"{result['holdout_spearman']:.3f}"
    click.echo(f"Holdout AUC:      {auc_str}")
    click.echo(f"Holdout Spearman: {rho_str}")
    click.echo(f"Config written to {output_dir}/")
