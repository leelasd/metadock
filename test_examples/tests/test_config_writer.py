# tests/test_config_writer.py
import numpy as np
import pytest
from pathlib import Path
from scoring_optimizer.config_writer import write_inter_sf_prm, write_cavity_prm

WEIGHTS = {
    "vdw_weight":     1.2,
    "polar_weight":   4.0,
    "repul_weight":   5.5,
    "const_weight":   5.0,
    "rot_weight":     0.9,
    "pharma_weight":  2.5,
    "cavity_weight":  1.1,
    "sys_vdw_weight": 0.8,
    "sys_pol_weight": 0.6,
}


def test_inter_sf_prm_created(tmp_path):
    out = tmp_path / "RbtInterIdxSF.prm"
    write_inter_sf_prm(WEIGHTS, out)
    assert out.exists()


def test_inter_sf_prm_contains_vdw_weight(tmp_path):
    out = tmp_path / "RbtInterIdxSF.prm"
    write_inter_sf_prm(WEIGHTS, out)
    assert "1.2000" in out.read_text()


def test_inter_sf_prm_has_required_sections(tmp_path):
    out = tmp_path / "RbtInterIdxSF.prm"
    write_inter_sf_prm(WEIGHTS, out)
    content = out.read_text()
    for section in ("CONST", "ROT", "POLAR", "REPUL", "VDW"):
        assert f"SECTION {section}" in content


def test_cavity_prm_created(tmp_path):
    out = tmp_path / "cavity.prm"
    write_cavity_prm(WEIGHTS, out, title="TEST", receptor_file="receptor.mol2",
                     ref_mol="xtal-lig.sd", pharma_restr_file="pharma.restr", waters=[])
    assert out.exists()


def test_cavity_prm_pharma_weight(tmp_path):
    out = tmp_path / "cavity.prm"
    write_cavity_prm(WEIGHTS, out, title="TEST", receptor_file="receptor.mol2",
                     ref_mol="xtal-lig.sd", pharma_restr_file="pharma.restr", waters=[])
    assert "2.5000" in out.read_text()


def test_cavity_prm_solvent_section_when_waters(tmp_path):
    out = tmp_path / "cavity.prm"
    write_cavity_prm(WEIGHTS, out, title="TEST", receptor_file="receptor.mol2",
                     ref_mol="xtal-lig.sd", pharma_restr_file="pharma.restr",
                     waters=[np.array([1.0, 2.0, 3.0])])
    assert "SECTION SOLVENT" in out.read_text()


def test_cavity_prm_no_solvent_when_no_waters(tmp_path):
    out = tmp_path / "cavity.prm"
    write_cavity_prm(WEIGHTS, out, title="TEST", receptor_file="receptor.mol2",
                     ref_mol="xtal-lig.sd", pharma_restr_file="pharma.restr", waters=[])
    assert "SECTION SOLVENT" not in out.read_text()
