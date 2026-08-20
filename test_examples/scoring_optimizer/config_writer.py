# scoring_optimizer/config_writer.py
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def write_inter_sf_prm(weights: dict, output_path: Path) -> None:
    """Write RbtInterIdxSF.prm with optimized Tier 1 weights.

    Note: sys_vdw_weight and sys_pol_weight (SCORE.SYSTEM.*) are used in the
    rescoring formula but are not written here — SCORE.SYSTEM.* terms are not
    independently configurable in RbtInterIdxSF.prm.
    """
    tmpl = _ENV.get_template("RbtInterIdxSF.prm.j2")
    output_path.write_text(tmpl.render(**weights))


def write_cavity_prm(
    weights: dict,
    output_path: Path,
    title: str,
    receptor_file: str,
    ref_mol: str,
    pharma_restr_file: str,
    waters: list,
) -> None:
    """Write cavity.prm with optimized PHARMA/CAVITY weights and optional SOLVENT section."""
    tmpl = _ENV.get_template("cavity.prm.j2")
    output_path.write_text(tmpl.render(
        title=title,
        receptor_file=receptor_file,
        ref_mol=ref_mol,
        pharma_restr_file=pharma_restr_file,
        cavity_weight=weights["cavity_weight"],
        pharma_weight=weights["pharma_weight"],
        waters=waters,
    ))
