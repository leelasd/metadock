# OpenMM Docking Suite (`openmm-dock`)

A GPU-accelerated molecular docking, scoring, and pose minimization framework implemented entirely within **OpenMM** via its Python API.

This package ports the scoring philosophy and features of [rDock (RiboDock)](https://github.com/CBDD/rDock) into OpenMM custom forces and integrators, supporting proteins, nucleic acids (RNA/DNA), explicit flexible solvent, pharmacophore constraints, and template-based tethered docking.

Detailed architectural deep-dive: See [docs/TECHNICAL_DEEPDIVE.md](docs/TECHNICAL_DEEPDIVE.md).

---

## 💡 Why Isn't Docking a Native Out-of-the-Box Feature in OpenMM?

While molecular docking **can** be built in OpenMM (as demonstrated here), OpenMM was designed from the ground up as a **molecular dynamics (MD) engine**, which differs from classical docking tools in several fundamental ways:

| Concept | Molecular Dynamics (OpenMM) | Molecular Docking (rDock / Vina) |
| :--- | :--- | :--- |
| **Coordinate Space** | **$3N$ Cartesian Coordinates** ($x, y, z$ for all $N$ atoms; e.g. $\sim 100$ DOFs) | **Internal Torsion Space** ($6$ rigid-body DOFs $+ M$ rotatable bonds; e.g. $10\text{--}15$ DOFs) |
| **Search Engine** | Continuous differential equations ($F = -\nabla V = m\ddot{r}$) using Verlet / Langevin integrators | Discrete stochastic jumps (Metropolis Monte Carlo, Genetic Algorithms, Simplex) |
| **Energy Function** | Physical Hamiltonians (12-6 LJ, $1/r$ Coulomb, explicit water/PME) | Empirical scoring functions (soft-core 4-8 LJ, screened $\epsilon(r)$, directional H-bonds) |
| **Receptor Potential** | All-atom pairwise interactions $O(N_{\text{rec}} \times N_{\text{lig}})$ | Precalculated 3D potential grids ($O(N_{\text{lig}})$ trilinear lookups) |
| **Host-Device Latency** | Integrates millions of continuous steps on GPU without CPU sync | Millions of tiny discrete pose evaluations in host memory |

### The "Python Loop" Latency Trap
When docking is naively written as a Python `for` loop calling `context.setPositions()` and `context.getState(getEnergy=True)` for 100,000 Monte Carlo steps, CPU $\leftrightarrow$ GPU memory synchronization takes $\sim 0.1\text{--}0.5\text{ ms}$ per step ($20\text{--}60\text{ seconds}$ total). 

**Our Solution:** Run **GPU-accelerated Simulated Annealing MD (SAMD)** directly on the OpenMM Context using `CustomNonbondedForce` and `LangevinMiddleIntegrator`, dropping execution time down to seconds while exploring rugged energy landscapes seamlessly!

---

## 🚀 Key Features

* **rDock Scoring Function via separate OpenMM `CustomNonbondedForce` terms:**
  * Soft-core 4-8 / 6-9 Lennard-Jones to prevent infinite singularities during search.
  * Distance-dependent dielectric screened electrostatics ($\epsilon(r) = D \cdot r$).
  * Contact hydrogen bonding, short-range polar-clash repulsion, and hydrophobic desolvation terms.
  * Intra-molecular ligand nonbonded strain.
  * Each term (`SCORE.INTER.VDW`/`POLAR`/`REPUL`/`HYD`/`CONST`/`ROT`) is a genuinely independent energy read from its own OpenMM force group — not a fixed fraction of a combined value — matching rDock's own per-term SDF output fields.
* **Ligand Structural Integrity & Aromatic Ring Planarity:**
  * Harmonic bond stretching ($k_b = 500{,}000\text{ kJ/(mol nm}^2)$) and valence angle bending ($k_\theta = 2000\text{ kJ/(mol rad}^2)$).
  * **Cross-Ring Structural Triangulation:** Eliminates ring buckling/tacoing; locks aromatic rings planar ($\text{deviation} < 0.03\text{ \AA}$).
  * **Improper Dihedrals:** Keeps exocyclic substituents coplanar with aromatic systems.
* **Cavity Definition & Restraints:**
  * Automated cavity detection from reference ligands or explicit coordinate spheres.
  * Flat-bottom harmonic potential via OpenMM `CustomExternalForce`.
* **Pharmacophore Restraints:**
  * Automated RDKit detection of aromatic rings, H-bond donors, acceptors, and hydrophobic centers.
  * Flat-bottom positional penalties from `pharma.restr` files.
* **Flexible Active-Site Waters (Solvent Docking):**
  * Support for explicit crystallographic waters tethered to their positions, capable of rotating and adjusting to ligand interactions.
* **Tethered / Template Docking:**
  * Automated Maximum Common Substructure (MCS) core alignment and harmonic position restraints.
* **RNA / DNA Docking:**
  * Native parsing and parameterization for nucleic acid aptamers (e.g. 1NEM).
* **Genetic Algorithm Local Refinement (`omm-dock ga`):**
  * Population-based GA over rigid-body + torsional DOFs (rDock's own default search engine), refining around the input ligand pose (crystal, pharmacophore-aligned, or tether-aligned) rather than blind global search — blind GA search over the full cavity did not reliably recover the binding pose in a practical budget even with Lamarckian local-minimization fitness, a known hard problem for population-based global search at this scale. Recovers the crystal pose to <0.7 Å heavy-atom RMSD from a perturbed start in ~35s/run.
  * Lamarckian/Baldwinian fitness: each candidate gets a short local minimization before scoring, using a single combined-term search force to keep the O(population × generations) inner loop fast; the fittest individual of each run is then fully minimized and scored with the real decomposed terms above.

---

## 📦 Installation

```bash
git clone https://github.com/leelasd/rxdock-deepdive-examples test_examples
pip install -e .
```

Dependencies: `openmm>=8.0`, `rdkit`, `numpy`, `scipy`, `pandas`, `pytest`.

---

## 🧪 Validated Use Cases from `rxdock-deepdive-examples`

The test suite validates all 6 deep-dive examples:

### 1. Scoring Pose Without Movement (`score/`)
```bash
omm-dock score -r test_examples/score/cavity.prm -i test_examples/score/ii.sd -o scored.sdf
```

### 2. Local L-BFGS Pose Minimization (`minimize/`)
```bash
omm-dock minimize -r test_examples/minimize/cavity.prm -i test_examples/minimize/ii.sd -o minimized.sdf
```

### 3. Solvent / Explicit Water Docking (`solvent/`)
```bash
omm-dock minimize -r test_examples/solvent/cavity.prm -w test_examples/solvent/test_waters.pdb -i test_examples/solvent/lig.sdf -o solvent_min.sdf
omm-dock dock -r test_examples/solvent/cavity.prm -w test_examples/solvent/test_waters.pdb -i test_examples/solvent/lig.sdf -o solvent_dock.sdf -n 5
```

### 4. Pharmacophore-Constrained Docking (`pharmacophores/`)
```bash
omm-dock dock -r test_examples/pharmacophores/cavity.prm -p test_examples/pharmacophores/pharma.restr -i test_examples/pharmacophores/xtal-lig.sd -o pharma_dock.sdf -n 5
```

### 5. Template-Constrained / Tethered Docking (`tethered/`)
```bash
omm-dock tether -r test_examples/tethered/cavity.prm -ref test_examples/tethered/xtal-lig.sd -i test_examples/tethered/query_ligands.sdf -o tethered_dock.sdf -n 5
```

### 6. RNA Aptamer Docking (`rna_docking_example/1nem`)
```bash
omm-dock dock -r test_examples/rna_docking_example/1nem/1nem_rdock.prm -i test_examples/rna_docking_example/1nem/1nem_lig.sd -o rna_dock.sdf -n 5
```

---

## 🐍 Python API Example

```python
from rdkit import Chem
from openmm_dock import DockingEngine, CavityDefinition, SDFParser

# 1. Load cavity and receptor
cavity = CavityDefinition.from_prm_file("test_examples/score/cavity.prm")
engine = DockingEngine(receptor_path="test_examples/score/receptor.mol2", cavity=cavity)

# 2. Load ligand
ligand = SDFParser.load_molecules("test_examples/score/ii.sd")[0]

# 3. Score pose
scores = engine.score(ligand)
print("Score Breakdown:", scores)

# 4. Minimize pose
min_result = engine.minimize(ligand)
print("Minimized Score:", min_result.score)

# 5. Run GPU Simulated Annealing Docking
docked_poses = engine.dock_simulated_annealing(ligand, n_runs=10)
print(f"Top Docked Pose Score: {docked_poses[0].score:.3f}")
```

---

## 🔬 Running Test Suite

```bash
pytest -v tests/
```
Output:
```
tests/test_docking_suite.py::test_core_parsers PASSED                    [ 12%]
tests/test_docking_suite.py::test_cavity_definition PASSED               [ 25%]
tests/test_docking_suite.py::test_use_case_score PASSED                  [ 37%]
tests/test_docking_suite.py::test_use_case_minimize PASSED               [ 50%]
tests/test_docking_suite.py::test_use_case_solvent PASSED                [ 62%]
tests/test_docking_suite.py::test_use_case_pharmacophores PASSED         [ 75%]
tests/test_docking_suite.py::test_use_case_tethered PASSED               [ 87%]
tests/test_docking_suite.py::test_use_case_rna_docking PASSED            [100%]
============================== 8 passed in 5.93s ===============================
```
