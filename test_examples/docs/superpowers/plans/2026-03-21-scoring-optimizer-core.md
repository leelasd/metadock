# Scoring Optimizer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python package that reads rDock docked poses + SAR data + crystal structures and produces optimized rDock `.prm` config files using Optuna Bayesian optimization.

**Architecture:** Seven focused modules (pose parsing → compound matching → metrics → crystal processing → config writing → optimizer → CLI) wired together by a Click CLI. All modules are independently testable with small fixture files. No AWS dependency — S3 paths are passed as local paths during testing and swapped for boto3 calls in Plan 2 (AWS pipeline).

**Tech Stack:** Python 3.11+, RDKit, Optuna, NumPy, Pandas, scikit-learn, SciPy, Jinja2, Click, pytest

---

## Implementation Notes

**SYSTEM weights (`sys_vdw_weight`, `sys_pol_weight`):** `SCORE.SYSTEM.*` fields in rDock output SDF represent water–protein and water–ligand interactions. They use the same underlying scoring functions as the INTER terms but are not independently configurable in a writable `.prm` file (`RbtTargetSF.prm` only holds a DIHEDRAL section). These weights are included in the Optuna rescoring formula (they improve post-hoc pose ranking) but are **not written to output `.prm` files**. They take effect during rescoring, not during the next docking search.

**RMSD computation:** The optimizer re-ranks poses per trial (different weights → different top pose per compound). RMSD must be recomputed for each trial using the actual coordinates of the trial's top-ranked pose vs the crystal ligand coords. The `Pose` dataclass therefore stores 3D heavy-atom coordinates parsed from the SDF mol block.

---

## File Map

```
scoring_optimizer/
  __init__.py
  pose_parser.py          Parse rDock output SDF → Pose dataclasses with SCORE.* fields + 3D coords
  compound_matching.py    Morgan Tanimoto: assign SAR compounds to crystal structures
  metrics.py              RMSD loss, AUC-ROC enrichment, Spearman potency correlation
  crystal_processing.py   Parse PDB crystals → waters, pharmacophore features, pharma.restr
  config_writer.py        Write optimized RbtInterIdxSF.prm and cavity.prm from Jinja2 templates
  optimizer.py            Optuna objective + train/holdout split + run_optimization()
  cli.py                  Click CLI: run-optimizer command
  templates/
    RbtInterIdxSF.prm.j2  Template for intermolecular scoring function file
    cavity.prm.j2         Template for cavity + PHARMA + SOLVENT sections

tests/
  conftest.py             Shared fixtures: mini SDF poses, mini SAR CSV, mini crystal PDB
  fixtures/
    mini_poses.sdf        3 compounds × 3 poses each, with all SCORE.* fields + mol blocks
    mini_sar.csv          10 rows: smiles, name, pic50, active, assay_date
    mini_crystal.pdb      Tiny receptor PDB with 3 crystallographic waters
    mini_ligand.sdf       Co-crystal ligand (pyridine) in Kekulé SDF form
  test_pose_parser.py
  test_compound_matching.py
  test_metrics.py
  test_crystal_processing.py
  test_config_writer.py
  test_optimizer.py
  test_cli.py

scripts/
  make_fixtures.py        One-off script to generate test fixture SDF/PDB files

pyproject.toml            Package config, dependencies, pytest settings
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `scoring_optimizer/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/mini_sar.csv`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "scoring-optimizer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "rdkit>=2023.9",
    "optuna>=3.5",
    "numpy>=1.26",
    "pandas>=2.1",
    "scikit-learn>=1.4",
    "scipy>=1.12",
    "jinja2>=3.1",
    "click>=8.1",
]

[project.scripts]
scoring-optimizer = "scoring_optimizer.cli:main"

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `scoring_optimizer/__init__.py`**

```python
# scoring_optimizer/__init__.py
```

- [ ] **Step 3: Create `tests/fixtures/mini_sar.csv`**

```csv
smiles,name,pic50,active,assay_date
c1ccc2ncccc2c1,CPD001,7.2,1,2025-01-10
c1ccncc1,CPD002,5.1,0,2025-01-10
c1ccc(N)cc1,CPD003,6.8,1,2025-01-15
c1ccc(O)cc1,CPD004,4.9,0,2025-01-15
c1cccc2ccccc12,CPD005,7.5,1,2025-02-01
c1ccc(Cl)cc1,CPD006,5.3,0,2025-02-01
c1ccc(F)cc1,CPD007,6.1,1,2025-02-10
c1cncc1,CPD008,4.7,0,2025-02-10
c1ccc2[nH]ccc2c1,CPD009,7.8,1,2025-03-01
c1ccc(C)cc1,CPD010,5.0,0,2025-03-01
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
# tests/conftest.py
from pathlib import Path
import pytest
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_poses_sdf():
    return FIXTURES / "mini_poses.sdf"


@pytest.fixture
def mini_sar_csv():
    return FIXTURES / "mini_sar.csv"


@pytest.fixture
def mini_sar_df():
    return pd.read_csv(FIXTURES / "mini_sar.csv", parse_dates=["assay_date"])
```

- [ ] **Step 5: Generate fixture SDF and PDB files**

Create `scripts/make_fixtures.py` and run it once:

```python
# scripts/make_fixtures.py
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

Path("tests/fixtures").mkdir(parents=True, exist_ok=True)
Path("scripts").mkdir(exist_ok=True)

# --- mini_poses.sdf: 3 compounds x 3 poses, SCORE.* fields + real 3D mol blocks ---
compounds = [
    ("CPD001", "c1ccc2ncccc2c1"),
    ("CPD002", "c1ccncc1"),
    ("CPD003", "c1ccc(N)cc1"),
]

lines = []
for name, smi in compounds:
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    for rank in range(1, 4):
        score = -10.0 + rank * 2.0  # rank 1 = best (lowest) score
        mol_block = Chem.MolToMolBlock(mol)
        lines.append(mol_block)
        lines.append(f">  <Name>\n{name}\n\n")
        lines.append(f">  <SCORE>\n{score:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.VDW>\n{-3.0 + rank:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.POLAR>\n{-2.0 + rank * 0.5:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.REPUL>\n{0.5 + rank * 0.1:.3f}\n\n")
        lines.append(f">  <SCORE.INTER.CONST>\n5.400\n\n")
        lines.append(f">  <SCORE.INTER.ROT>\n{1.0 + rank * 0.2:.3f}\n\n")
        lines.append(f">  <SCORE.RESTR>\n{0.0 if rank == 1 else 2.0:.3f}\n\n")
        lines.append(f">  <SCORE.RESTR.CAVITY>\n0.000\n\n")
        lines.append(f">  <SCORE.SYSTEM.VDW>\n{-1.0 + rank * 0.3:.3f}\n\n")
        lines.append(f">  <SCORE.SYSTEM.POLAR>\n{-0.5 + rank * 0.1:.3f}\n\n")
        lines.append("$$$$\n")

Path("tests/fixtures/mini_poses.sdf").write_text("".join(lines))
print("Wrote mini_poses.sdf")

# --- mini_crystal.pdb: ligand at origin, 3 waters within 5Å, 1 water far away ---
pdb = """\
HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A   1       1.400   0.000   0.000  1.00  0.00           C
HETATM    3  N1  LIG A   1       0.700   1.212   0.000  1.00  0.00           N
HETATM    4  O   HOH A 101       2.000   0.500   0.000  1.00  0.00           O
HETATM    5  O   HOH A 102      -1.500   0.500   0.000  1.00  0.00           O
HETATM    6  O   HOH A 103       0.500   3.000   0.000  0.50  0.00           O
HETATM    7  O   HOH A 104       5.500   5.500   5.500  1.00  0.00           O
END
"""
Path("tests/fixtures/mini_crystal.pdb").write_text(pdb)
print("Wrote mini_crystal.pdb")

# --- mini_ligand.sdf: pyridine in Kekulé form, centred near origin ---
mol = Chem.MolFromSmiles("c1ccncc1")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
mol = Chem.RemoveHs(mol)
Chem.Kekulize(mol, clearAromaticFlags=True)
w = Chem.SDWriter("tests/fixtures/mini_ligand.sdf")
w.SetKekulize(True)
w.write(mol)
w.close()
print("Wrote mini_ligand.sdf")
```

Run: `python scripts/make_fixtures.py`

Verify outputs exist:
```bash
ls tests/fixtures/
```

Expected: `mini_crystal.pdb  mini_ligand.sdf  mini_poses.sdf  mini_sar.csv`

- [ ] **Step 6: Install package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add scoring_optimizer/ tests/ pyproject.toml scripts/
git commit -m "feat: scaffold scoring-optimizer package with fixtures"
```

---

## Task 2: Pose Parser

**Files:**
- Create: `scoring_optimizer/pose_parser.py`
- Create: `tests/test_pose_parser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pose_parser.py
import numpy as np
import pytest
from scoring_optimizer.pose_parser import parse_poses, group_by_compound, top_pose, SCORE_FIELDS


def test_parse_returns_pose_objects(mini_poses_sdf):
    poses = parse_poses(mini_poses_sdf)
    assert len(poses) == 9  # 3 compounds × 3 poses


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
    assert poses[0].coords.shape[1] == 3  # (n_atoms, 3)


def test_poses_grouped_by_compound(mini_poses_sdf):
    groups = group_by_compound(parse_poses(mini_poses_sdf))
    assert set(groups.keys()) == {"CPD001", "CPD002", "CPD003"}
    assert all(len(v) == 3 for v in groups.values())


def test_top_pose_is_lowest_score(mini_poses_sdf):
    groups = group_by_compound(parse_poses(mini_poses_sdf))
    for name, poses in groups.items():
        best = top_pose(poses)
        assert best.scores["SCORE"] == min(p.scores["SCORE"] for p in poses)
```

- [ ] **Step 2: Run to verify all fail**

```bash
pytest tests/test_pose_parser.py -v
```

Expected: `ImportError` — module does not exist yet.

- [ ] **Step 3: Implement `pose_parser.py`**

```python
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
    "SCORE.SYSTEM.VDW",
    "SCORE.SYSTEM.POLAR",
]


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

    with open(sdf_path) as f:
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
                current_field = line[4:].rstrip(">").strip()
            elif current_field == "Name":
                current_name = line.strip()
                current_field = None
            elif current_field in SCORE_FIELDS:
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
    return min(poses, key=lambda p: p.scores.get("SCORE", float("inf")))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pose_parser.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring_optimizer/pose_parser.py tests/test_pose_parser.py
git commit -m "feat: add pose parser for rDock SDF output with 3D coordinate extraction"
```

---

## Task 3: Compound Matching

**Files:**
- Create: `scoring_optimizer/compound_matching.py`
- Create: `tests/test_compound_matching.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compound_matching.py
import pytest
from rdkit import Chem
from scoring_optimizer.compound_matching import morgan_tanimoto, assign_to_crystal


def _mol(smi):
    return Chem.MolFromSmiles(smi)


def test_identical_molecules_score_one():
    mol = _mol("c1ccccc1")
    assert morgan_tanimoto(mol, mol) == pytest.approx(1.0)


def test_dissimilar_molecules_score_low():
    assert morgan_tanimoto(_mol("c1ccccc1"), _mol("CCC")) < 0.3


def test_assign_returns_none_below_threshold():
    query = _mol("CCC")
    crystals = [_mol("c1ccccc1"), _mol("c1ccncc1")]
    idx, sim = assign_to_crystal(query, crystals, threshold=0.6)
    assert idx is None
    assert sim < 0.6


def test_assign_picks_exact_match():
    query = _mol("c1ccncc1")  # pyridine
    crystals = [_mol("CCC"), _mol("c1ccncc1"), _mol("c1ccccc1")]
    idx, sim = assign_to_crystal(query, crystals, threshold=0.6)
    assert idx == 1
    assert sim == pytest.approx(1.0)


def test_assign_returns_best_when_multiple_match():
    query = _mol("c1ccc(N)cc1")   # aniline
    # phenol is more similar to aniline than propane
    crystals = [_mol("CCC"), _mol("c1ccc(O)cc1")]
    idx, sim = assign_to_crystal(query, crystals, threshold=0.3)
    assert idx == 1  # phenol, not propane
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_compound_matching.py -v
```

- [ ] **Step 3: Implement `compound_matching.py`**

```python
# scoring_optimizer/compound_matching.py
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def morgan_tanimoto(mol_a: Chem.Mol, mol_b: Chem.Mol, radius: int = 2) -> float:
    """Morgan fingerprint (ECFP4) Tanimoto similarity."""
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def assign_to_crystal(
    sar_mol: Chem.Mol,
    crystal_mols: list[Chem.Mol],
    threshold: float = 0.6,
) -> tuple[int | None, float]:
    """
    Find the crystal structure most similar to sar_mol (Morgan ECFP4 Tanimoto).

    Returns (index, similarity). Returns (None, best_sim) if best_sim < threshold.
    """
    if not crystal_mols:
        return None, 0.0
    sims = [morgan_tanimoto(sar_mol, c) for c in crystal_mols]
    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
    best_sim = sims[best_idx]
    return (best_idx, best_sim) if best_sim >= threshold else (None, best_sim)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_compound_matching.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring_optimizer/compound_matching.py tests/test_compound_matching.py
git commit -m "feat: add Morgan ECFP4 Tanimoto compound-crystal matching"
```

---

## Task 4: Metrics

**Files:**
- Create: `scoring_optimizer/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from scoring_optimizer.metrics import (
    rmsd_loss, enrichment_auc, potency_spearman, composite_objective,
)


def test_rmsd_zero_for_identical():
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert rmsd_loss(coords, coords) == pytest.approx(0.0)


def test_rmsd_known_value():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    assert rmsd_loss(a, b) == pytest.approx(1.0)


def test_auc_perfect_separation():
    scores = np.array([-10.0, -9.0, -1.0, -0.5])  # lower = better in rDock
    labels = np.array([1, 1, 0, 0])
    assert enrichment_auc(scores, labels) == pytest.approx(1.0)


def test_auc_random_is_near_half():
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(100)
    labels = rng.integers(0, 2, 100)
    assert 0.3 < enrichment_auc(scores, labels) < 0.7


def test_spearman_perfect_correlation():
    scores = np.array([-7.0, -6.0, -5.0, -4.0])
    pic50  = np.array([ 7.0,  6.0,  5.0,  4.0])
    assert potency_spearman(scores, pic50) == pytest.approx(1.0)


def test_composite_objective_perfect():
    assert composite_objective(0.0, 1.0, 1.0, 0.5, 0.3, 0.2) == pytest.approx(0.0)


def test_composite_objective_is_positive():
    assert composite_objective(2.0, 0.5, 0.0, 0.5, 0.3, 0.2) > 0.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement `metrics.py`**

```python
# scoring_optimizer/metrics.py
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def rmsd_loss(pred_coords: np.ndarray, crystal_coords: np.ndarray) -> float:
    """RMSD between top-scored pose and crystal structure (Å)."""
    diff = pred_coords - crystal_coords
    return float(np.sqrt(np.mean(diff**2)))


def enrichment_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC: actives=1 vs inactives=0. Negates scores (lower rDock = better)."""
    return float(roc_auc_score(labels, -scores))


def potency_spearman(scores: np.ndarray, pic50: np.ndarray) -> float:
    """Spearman rank correlation: lower rDock score should → higher pIC50."""
    rho, _ = spearmanr(-scores, pic50)
    return float(rho)


def composite_objective(
    rmsd: float,
    auc: float,
    spearman: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Composite minimization target: α·rmsd + β·(1−auc) + γ·(1−spearman)."""
    return alpha * rmsd + beta * (1.0 - auc) + gamma * (1.0 - spearman)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring_optimizer/metrics.py tests/test_metrics.py
git commit -m "feat: add RMSD, AUC-ROC, and Spearman metrics"
```

---

## Task 5: Crystal Processing

**Files:**
- Create: `scoring_optimizer/crystal_processing.py`
- Create: `tests/test_crystal_processing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_crystal_processing.py
import numpy as np
import pytest
from pathlib import Path
from rdkit import Chem
from scoring_optimizer.crystal_processing import (
    find_binding_waters,
    pharmacophore_features,
    write_pharma_restr,
    crystal_ligand_coords,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_find_waters_within_cutoff():
    waters = find_binding_waters(
        FIXTURES / "mini_crystal.pdb",
        FIXTURES / "mini_ligand.sdf",
        cutoff_angstrom=5.0,
        min_occupancy=0.5,
    )
    # HOH 101 (1.0 occ, near), HOH 102 (1.0 occ, near), HOH 103 (0.5 occ, near)
    # HOH 104 is >5Å away — excluded
    assert len(waters) == 3
    for w in waters:
        assert w.shape == (3,)


def test_occupancy_filter():
    waters = find_binding_waters(
        FIXTURES / "mini_crystal.pdb",
        FIXTURES / "mini_ligand.sdf",
        cutoff_angstrom=5.0,
        min_occupancy=0.6,  # HOH 103 has occupancy=0.5 → excluded
    )
    assert len(waters) == 2


def test_pharmacophore_features_aromatic_rings():
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    assert len(feats["aro_centers"]) >= 1


def test_pharmacophore_features_includes_aromatic_acceptors():
    # pyridine N is aromatic — must be included as acceptor
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    assert len(feats["acceptors"]) >= 1


def test_write_pharma_restr_line_count(tmp_path):
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    out = tmp_path / "test.restr"
    write_pharma_restr(feats, out, n_aro=1, n_acc=1, tolerance=1.0)
    lines = [l for l in out.read_text().strip().splitlines() if l.strip()]
    assert len(lines) == 2


def test_write_pharma_restr_format(tmp_path):
    mol = Chem.MolFromMolFile(str(FIXTURES / "mini_ligand.sdf"), removeHs=False)
    feats = pharmacophore_features(mol)
    out = tmp_path / "test.restr"
    write_pharma_restr(feats, out, n_aro=1, n_acc=1, tolerance=1.0)
    for line in out.read_text().strip().splitlines():
        parts = line.split()
        assert len(parts) == 5
        float(parts[0]); float(parts[1]); float(parts[2]); float(parts[3])
        assert parts[4] in ("Aro", "Acc", "Don", "Hyd")


def test_crystal_ligand_coords_shape():
    coords = crystal_ligand_coords(FIXTURES / "mini_ligand.sdf")
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert len(coords) > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_crystal_processing.py -v
```

- [ ] **Step 3: Implement `crystal_processing.py`**

```python
# scoring_optimizer/crystal_processing.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def find_binding_waters(
    receptor_pdb: Path,
    ligand_sdf: Path,
    cutoff_angstrom: float = 5.0,
    min_occupancy: float = 0.5,
) -> list[np.ndarray]:
    """Return xyz of crystallographic waters within cutoff_angstrom of the ligand."""
    ligand_mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False)
    conf = ligand_mol.GetConformer()
    lig_coords = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(ligand_mol.GetNumAtoms())
    ])

    waters = []
    with open(receptor_pdb) as f:
        for line in f:
            if not (line.startswith("HETATM") or line.startswith("ATOM")):
                continue
            if line[17:20].strip() != "HOH" or line[12:16].strip() != "O":
                continue
            try:
                occupancy = float(line[54:60].strip())
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if occupancy < min_occupancy:
                continue
            if np.linalg.norm(lig_coords - xyz, axis=1).min() <= cutoff_angstrom:
                waters.append(xyz)
    return waters


def pharmacophore_features(mol: Chem.Mol) -> dict:
    """
    Extract pharmacophore features from a 3D ligand molecule.

    Acceptors include both non-aromatic and aromatic N/O (e.g. pyridine N).
    """
    conf = mol.GetConformer()

    def pos(idx) -> np.ndarray:
        p = conf.GetAtomPosition(idx)
        return np.array([p.x, p.y, p.z])

    aro_centers = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            aro_centers.append(np.mean([pos(a) for a in ring], axis=0))

    acceptors, donors = [], []
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum in (7, 8):
            acceptors.append(pos(atom.GetIdx()))
        if anum in (7, 8) and atom.GetTotalNumHs() > 0:
            donors.append(pos(atom.GetIdx()))

    return {"aro_centers": aro_centers, "acceptors": acceptors, "donors": donors}


def write_pharma_restr(
    features: dict,
    output_path: Path,
    n_aro: int = 2,
    n_acc: int = 2,
    tolerance: float = 1.0,
) -> None:
    """Write a pharma.restr file from pharmacophore features."""
    lines = []
    for c in features["aro_centers"][:n_aro]:
        lines.append(f"{c[0]:7.2f} {c[1]:7.2f} {c[2]:7.2f} {tolerance:.2f} Aro")
    for c in features["acceptors"][:n_acc]:
        lines.append(f"{c[0]:7.2f} {c[1]:7.2f} {c[2]:7.2f} {tolerance:.2f} Acc")
    output_path.write_text("\n".join(lines) + "\n")


def crystal_ligand_coords(ligand_sdf: Path) -> np.ndarray:
    """Return heavy-atom 3D coordinates of the co-crystal ligand as (n, 3) array."""
    mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=True)
    conf = mol.GetConformer()
    return np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_crystal_processing.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring_optimizer/crystal_processing.py tests/test_crystal_processing.py
git commit -m "feat: add crystal structure processing (waters, pharmacophore, pharma.restr)"
```

---

## Task 6: Config Writer

**Files:**
- Create: `scoring_optimizer/templates/RbtInterIdxSF.prm.j2`
- Create: `scoring_optimizer/templates/cavity.prm.j2`
- Create: `scoring_optimizer/config_writer.py`
- Create: `tests/test_config_writer.py`

- [ ] **Step 1: Create `scoring_optimizer/templates/RbtInterIdxSF.prm.j2`**

```jinja
{# scoring_optimizer/templates/RbtInterIdxSF.prm.j2 #}
RBT_PARAMETER_FILE_V1.00
TITLE Optimized intermolecular scoring function

SECTION CONST
    SCORING_FUNCTION    RbtConstSF
    SOLVENT_PENALTY     0.37
    WEIGHT              {{ const_weight | round(4) }}
END_SECTION

SECTION ROT
    SCORING_FUNCTION    RbtRotSF
    WEIGHT              {{ rot_weight | round(4) }}
END_SECTION

SECTION SETUP_POLAR
    SCORING_FUNCTION    RbtSetupPolarSF
    RADIUS              5.0
    NORM                25
    POWER               0.5
    CHGFACTOR           0.5
    GUANFACTOR          0.5
END_SECTION

SECTION POLAR
    SCORING_FUNCTION    RbtPolarIdxSF
    WEIGHT              {{ polar_weight | round(4) }}
    R12FACTOR           1.0
    R12INCR             0.05
    DR12MIN             0.25
    DR12MAX             0.6
    A1                  180.0
    DA1MIN              30.0
    DA1MAX              80.0
    A2                  180.0
    DA2MIN              60.0
    DA2MAX              100.0
    INCMETAL            TRUE
    INCHBD              TRUE
    INCHBA              TRUE
    INCGUAN             TRUE
    GUAN_PLANE          TRUE
    ABS_DR12            TRUE
    GRIDSTEP            0.5
    RANGE               5.31
    INCR                3.36
    ATTR                TRUE
    LP_OSP2             TRUE
    LP_PHI              45
    LP_DPHIMIN          15
    LP_DPHIMAX          30
    LP_DTHETAMIN        20
    LP_DTHETAMAX        60
END_SECTION

SECTION REPUL
    SCORING_FUNCTION    RbtPolarIdxSF
    WEIGHT              {{ repul_weight | round(4) }}
    R12FACTOR           1.0
    R12INCR             0.6
    DR12MIN             0.25
    DR12MAX             1.1
    A1                  180.0
    DA1MIN              30.0
    DA1MAX              60.0
    A2                  180.0
    DA2MIN              30.0
    DA2MAX              60.0
    INCMETAL            TRUE
    INCHBD              TRUE
    INCHBA              TRUE
    INCGUAN             TRUE
    GUAN_PLANE          FALSE
    ABS_DR12            FALSE
    GRIDSTEP            0.5
    RANGE               5.32
    INCR                3.51
    ATTR                FALSE
    LP_OSP2             FALSE
END_SECTION

SECTION VDW
    SCORING_FUNCTION    RbtVdwIdxSF
    WEIGHT              {{ vdw_weight | round(4) }}
    USE_4_8             FALSE
    USE_TRIPOS          FALSE
    RMAX                1.5
    ECUT                120.0
    E0                  1.5
    FAST_SOLVENT        TRUE
END_SECTION
```

- [ ] **Step 2: Create `scoring_optimizer/templates/cavity.prm.j2`**

```jinja
{# scoring_optimizer/templates/cavity.prm.j2 #}
RBT_PARAMETER_FILE_V1.00
TITLE {{ title }}

RECEPTOR_FILE {{ receptor_file }}
RECEPTOR_FLEX 3.0

SECTION MAPPER
    SITE_MAPPER     RbtLigandSiteMapper
    REF_MOL         {{ ref_mol }}
    RADIUS          6.0
    SMALL_SPHERE    1.0
    MIN_VOLUME      100
    MAX_CAVITIES    1
    VOL_INCR        0.0
    GRIDSTEP        0.5
END_SECTION

SECTION CAVITY
    SCORING_FUNCTION    RbtCavityGridSF
    WEIGHT              {{ cavity_weight | round(4) }}
END_SECTION

SECTION PHARMA
    SCORING_FUNCTION    RbtPharmaSF
    WEIGHT              {{ pharma_weight | round(4) }}
    CONSTRAINTS_FILE    {{ pharma_restr_file }}
END_SECTION
{% if waters %}

SECTION SOLVENT
{% for w in waters %}
    FILE            water_{{ loop.index }}.pdb
{% endfor %}
    TRANS_MODE      TETHERED
    ROT_MODE        TETHERED
    MAX_TRANS       1.0
    MAX_ROT         30.0
    OCCUPANCY       0.5
END_SECTION
{% endif %}
```

- [ ] **Step 3: Write failing tests**

```python
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
```

- [ ] **Step 4: Run to verify they fail**

```bash
pytest tests/test_config_writer.py -v
```

- [ ] **Step 5: Implement `config_writer.py`**

```python
# scoring_optimizer/config_writer.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
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
    tmpl = _env().get_template("RbtInterIdxSF.prm.j2")
    output_path.write_text(tmpl.render(**weights))


def write_cavity_prm(
    weights: dict,
    output_path: Path,
    title: str,
    receptor_file: str,
    ref_mol: str,
    pharma_restr_file: str,
    waters: list[np.ndarray],
) -> None:
    """Write cavity.prm with optimized PHARMA/CAVITY weights and optional SOLVENT section."""
    tmpl = _env().get_template("cavity.prm.j2")
    output_path.write_text(tmpl.render(
        title=title,
        receptor_file=receptor_file,
        ref_mol=ref_mol,
        pharma_restr_file=pharma_restr_file,
        cavity_weight=weights["cavity_weight"],
        pharma_weight=weights["pharma_weight"],
        waters=waters,
    ))
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_config_writer.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add scoring_optimizer/config_writer.py scoring_optimizer/templates/ \
        tests/test_config_writer.py
git commit -m "feat: add config writer with Jinja2 templates for .prm files"
```

---

## Task 7: Optuna Optimizer

**Files:**
- Create: `scoring_optimizer/optimizer.py`
- Create: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests**

```python
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
    # All training dates must be ≤ all holdout dates
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


def test_run_optimization_with_crystal_coords(mini_poses_sdf, mini_sar_df):
    # Provide fake crystal coords for CPD001 — shape must match pose coords
    from scoring_optimizer.pose_parser import parse_poses, group_by_compound, top_pose
    poses = parse_poses(mini_poses_sdf)
    groups = group_by_compound(poses)
    crystal_shape = groups["CPD001"][0].coords
    crystal_coords_map = {"CPD001": crystal_shape}  # same coords → RMSD = 0
    result = run_optimization(
        poses_sdf=mini_poses_sdf,
        sar_df=mini_sar_df,
        crystal_coords_map=crystal_coords_map,
        alpha=0.3, beta=0.4, gamma=0.3,
        n_trials=10,
    )
    assert "weights" in result
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_optimizer.py -v
```

- [ ] **Step 3: Implement `optimizer.py`**

```python
# scoring_optimizer/optimizer.py
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from .pose_parser import parse_poses, group_by_compound, Pose
from .metrics import enrichment_auc, potency_spearman, rmsd_loss, composite_objective

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    crystal_coords_map: dict[str, np.ndarray],  # compound name → crystal heavy-atom coords
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

        # RMSD loss: recompute per trial — different weights → different top pose
        rmsd_values = []
        for name, crystal_coords in crystal_coords_map.items():
            if name in top_poses and top_poses[name].coords is not None:
                try:
                    r = rmsd_loss(top_poses[name].coords, crystal_coords)
                    rmsd_values.append(r)
                except ValueError:
                    pass  # shape mismatch — skip
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_optimizer.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring_optimizer/optimizer.py tests/test_optimizer.py
git commit -m "feat: add Optuna Tier 1 optimizer with per-trial RMSD recomputation"
```

---

## Task 8: CLI

**Files:**
- Create: `scoring_optimizer/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement `cli.py`**

```python
# scoring_optimizer/cli.py
from __future__ import annotations
import json
from pathlib import Path
import click
import pandas as pd
from .optimizer import run_optimization
from .config_writer import write_inter_sf_prm, write_cavity_prm


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
    """Optimize Tier 1 rDock weights from docked poses and SAR data."""
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
        "holdout_auc":       result["holdout_auc"],
        "holdout_spearman":  result["holdout_spearman"],
        "best_objective":    result["best_objective"],
        "weights":           result["weights"],
        "n_compounds":       len(sar_df),
        "n_crystal_matched": 0,   # populated when crystal_coords_map is non-empty
        "n_trials":          n_trials,
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    click.echo(f"Holdout AUC:      {result['holdout_auc']:.3f}")
    click.echo(f"Holdout Spearman: {result['holdout_spearman']:.3f}")
    click.echo(f"Config written to {output_dir}/")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring_optimizer/cli.py tests/test_cli.py
git commit -m "feat: add Click CLI run-optimizer command with metrics.json output"
```

---

## Task 9: Full Suite + End-to-End Smoke Test

**Files:** No new files.

- [ ] **Step 1: Run full test suite with coverage**

```bash
pytest tests/ -v --cov=scoring_optimizer --cov-report=term-missing
```

Expected: all tests PASS, coverage ≥ 85% per module.

- [ ] **Step 2: Fix any failures or gaps before proceeding**

If any test fails, fix the underlying code. If coverage on a module is below 85%, add a test for the uncovered branch.

- [ ] **Step 3: End-to-end smoke test**

The pharmacophore example's `output.sd` contains real rDock poses for `LGF_361_2QD9`. Create a minimal SAR CSV that uses that compound name:

```bash
cat > /tmp/pharma_sar.csv << 'EOF'
smiles,name,pic50,active,assay_date
c1ccc2ncccc2c1,LGF_361_2QD9,7.2,1,2025-01-10
c1ccncc1,CPD_INACTIVE,4.5,0,2025-01-10
c1ccc(N)cc1,CPD_ACT2,6.8,1,2025-02-01
c1ccc(O)cc1,CPD_INACT2,4.9,0,2025-02-01
c1cccc2ccccc12,CPD_ACT3,7.5,1,2025-03-01
c1ccc(Cl)cc1,CPD_INACT3,5.3,0,2025-03-01
EOF
```

Run the optimizer against the real pharmacophore example poses:

```bash
scoring-optimizer run-optimizer \
  --poses pharmacophores/output.sd \
  --sar /tmp/pharma_sar.csv \
  --output-dir /tmp/optimized_config \
  --n-trials 50 \
  --alpha 0.0 --beta 0.5 --gamma 0.5 \
  --receptor receptor.mol2 \
  --ref-mol xtal-lig.sd \
  --pharma-restr pharma.restr
```

Expected output (exact numbers will vary):
```
Loaded 6 SAR compounds
Holdout AUC:      X.XXX
Holdout Spearman: X.XXX
Config written to /tmp/optimized_config/
```

Inspect the generated files:
```bash
cat /tmp/optimized_config/metrics.json
head -20 /tmp/optimized_config/RbtInterIdxSF.prm
```

Both must exist and contain finite numeric values. The `.prm` file must be valid rDock syntax (all SECTION/END_SECTION blocks present).

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "test: full suite passes, end-to-end smoke test verified"
git push
```

---

## Notes for Plan 2 (AWS Pipeline)

Plan 2 wires this package into production:

- **SageMaker container:** `Dockerfile.sagemaker` that installs this package; `entry.py` reads S3 input paths from env vars (`SM_CHANNEL_POSES`, `SM_CHANNEL_SAR`), calls `run_optimization()`, writes output back to `SM_MODEL_DIR`
- **Crystal processing step:** Batch job that calls `find_binding_waters()` and `write_pharma_restr()` across all crystal PDBs; passes `crystal_coords_map` to the SageMaker job via S3
- **Step Functions:** orchestrates Batch (crystal) → Batch (rbdock) → SageMaker (optimizer) → Lambda (versioning)
- **Lambda versioning:** reads `metrics.json` from SageMaker output, applies promotion gate (ΔAUC > 0.02, ΔSpearman > 0.03), updates `current.json`
- **DynamoDB:** cumulative compound delta counter, reset after each successful training run
