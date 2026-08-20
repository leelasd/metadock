# Modernize rxdock-deepdive-examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the rxdock-deepdive-examples repo to fix deprecated Python imports, slim down the Docker image, add CI, and fill documentation gaps — all in one branch + PR.

**Architecture:** All changes are independent file edits (no new modules, no refactors). Python files get import-only changes. Docker gets a base image swap and cleanup. CI is a new YAML file. READMEs are new Markdown files.

**Tech Stack:** Python 3 / OpenMM 8.x / ParmEd 4.x, Docker (debian:bookworm-slim), GitHub Actions, rDock/rxDock

**Spec:** `docs/superpowers/specs/2026-03-21-modernize-rxdock-examples-design.md`

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `pharmacophores/prepare_protein.py` | Fix simtk → openmm imports |
| Modify | `solvent/prepare_protein.py` | Fix simtk → openmm imports |
| Modify | `tethered/prepare_protein.py` | Fix simtk → openmm imports |
| Modify | `Dockerfile` | Switch to debian:bookworm-slim, clean up duplicate ENV PATH |
| Create | `.dockerignore` | Exclude non-build files from Docker context |
| Create | `.github/workflows/docker-build.yml` | CI smoke test: build + run |
| Create | `tethered/README.md` | Usage docs for tetheredMinimization.py |
| Create | `rna_docking_example/README.md` | Usage docs for 1NEM RNA docking example |
| Modify | `README.md` | Add per-example sections with Docker run commands |

---

## Task 1: Create feature branch

**Files:** (git only)

- [ ] **Step 1: Create and switch to branch**

```bash
git checkout -b update/modernize-components
```

Expected: `Switched to a new branch 'update/modernize-components'`

---

## Task 2: Fix deprecated simtk.openmm imports

**Files:**
- Modify: `pharmacophores/prepare_protein.py`
- Modify: `solvent/prepare_protein.py`
- Modify: `tethered/prepare_protein.py`

All three files have the same broken imports on lines 1-7. The fix is identical for all three — collapse 6 import lines to 3.

Background: `simtk.openmm` was the old namespace. OpenMM 7.6 added the new `openmm` top-level package and deprecated `simtk`. OpenMM 8.0 **removed** `simtk` entirely, so these scripts break on any modern OpenMM install.

- [ ] **Step 1: Verify current state of all three files**

```bash
head -8 pharmacophores/prepare_protein.py solvent/prepare_protein.py tethered/prepare_protein.py
```

Expected: all show `from simtk.openmm import app` on line 2.

- [ ] **Step 2: Fix `pharmacophores/prepare_protein.py`**

Replace lines 2-7 (the import block):

```python
# REMOVE these lines:
from simtk.openmm import app
from simtk.openmm.app import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
#from simtk.unit import *

# REPLACE WITH:
from openmm import app, unit
from openmm.app import *
```

Final file should start:
```python
import parmed
from openmm import app, unit
from openmm.app import *
receptor_pdbfile = PDBFile('output.pdb')
omm_forcefield = app.ForceField('amber10.xml')
```

- [ ] **Step 3: Fix `solvent/prepare_protein.py`**

Same replacement. This file also has `receptor_structure.save('receptor.gro', overwrite=True)` at line 21 — do NOT touch it.

Final file should start:
```python
import parmed
from openmm import app, unit
from openmm.app import *
receptor_pdbfile = PDBFile('output.pdb')
```

- [ ] **Step 4: Fix `tethered/prepare_protein.py`**

Same replacement. Also keeps the `.gro` save line.

- [ ] **Step 5: Verify all three files look correct**

```bash
head -5 pharmacophores/prepare_protein.py solvent/prepare_protein.py tethered/prepare_protein.py
```

Expected: all three start with `import parmed` then `from openmm import app, unit` then `from openmm.app import *`.

- [ ] **Step 6: Commit**

```bash
git add pharmacophores/prepare_protein.py solvent/prepare_protein.py tethered/prepare_protein.py
git commit -m "fix: update prepare_protein.py to use modern openmm imports (simtk removed in OpenMM 8.0)"
```

---

## Task 3: Slim down Dockerfile and add .dockerignore

**Files:**
- Modify: `Dockerfile`
- Create: `.dockerignore`

Background: The current Dockerfile uses `ubuntu:22.04` (full, ~77MB compressed). `debian:bookworm-slim` is ~30MB compressed and has glibc 2.36 (backward compatible with the Ubuntu 20.04 binary's glibc 2.31). The current Dockerfile also has a redundant `ENV PATH=/app/bin:$PATH` line (line 6) that duplicates line 9 via `$RBT_ROOT`.

- [ ] **Step 1: Rewrite the Dockerfile**

```dockerfile
# Base: debian:bookworm-slim (~30MB) — lighter than ubuntu:22.04.
# The rDock binary was built for Ubuntu 20.04 (glibc 2.31);
# Debian bookworm (glibc 2.36) is fully backward compatible.
#
# Obtain the tarball from:
# https://github.com/CBDD/rDock/releases
FROM debian:bookworm-slim
LABEL description="rxdock/rdock molecular docking examples" \
      maintainer="rxdock-deepdive-examples"
RUN apt-get update \
    && apt-get install -y --no-install-recommends make g++ libpopt-dev libpopt0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY rdock-legacy-24.04.204_ubuntu-20.04_g++_x86_64.tar.gz /app/
WORKDIR /app/
RUN tar -zxvf rdock-legacy-24.04.204_ubuntu-20.04_g++_x86_64.tar.gz \
    && rm rdock-legacy-24.04.204_ubuntu-20.04_g++_x86_64.tar.gz
ENV RBT_ROOT=/app
ENV PATH=${RBT_ROOT}/bin:${PATH}
ENV LD_LIBRARY_PATH=${RBT_ROOT}/lib:${LD_LIBRARY_PATH}
```

Note: `FROM` must be first — `LABEL` comes immediately after. The original duplicate `ENV PATH=/app/bin:$PATH` line is removed; `PATH` is now set once via `${RBT_ROOT}`. `LD_LIBRARY_PATH` now also uses `${RBT_ROOT}` instead of the hardcoded `/app/lib` from the original.

Note on layer size: The tarball is deleted in the same `RUN` layer as extraction, which minimizes the extraction layer. However, the `COPY` layer before it still holds the tarball. For maximum compression use `--squash` or BuildKit cache mounts; for the purposes of this repo a single-layer delete is sufficient.

- [ ] **Step 2: Create `.dockerignore`**

```
.git
docs/
**/*.md
**/*.py
**/*.pdb
**/*.sd
**/*.sdf
**/*.mol2
**/*.mol
**/*.prm
**/*.as
**/*.grd
**/*.fasta.gz
**/*.seq
**/*.inpcrd
**/*.prmtop
**/*.gro
**/*.csv
**/*.xyz
**/*.restr
```

Important: `*.tar.gz` is NOT excluded — the `COPY` instruction requires it.

- [ ] **Step 3: Verify .dockerignore does not exclude the tarball**

```bash
grep "tar.gz" .dockerignore
```

Expected: no output (no match means tarball is not excluded).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "chore: switch to debian:bookworm-slim, add .dockerignore, clean up Dockerfile"
```

---

## Task 4: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/docker-build.yml`

This CI workflow runs on every push and PR to `main`. It does two things:
1. Builds the Docker image (catches `apt-get` or extraction failures)
2. Runs `rbdock -h` inside the image (catches runtime library mismatches — a build-only test would miss a missing `libpopt.so` at runtime)

Note: The tarball must exist in the repo for this CI to work. If the repo does not ship the tarball (it is a large binary), the workflow will need to download it first. Add a download step with a placeholder comment so the implementer knows where to insert the actual download URL/command.

- [ ] **Step 1: Create workflow directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Create `.github/workflows/docker-build.yml`**

```yaml
name: Docker Build

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      # If the tarball is not committed to the repo, add a download step here.
      # Example: wget -O rdock-legacy-24.04.204_ubuntu-20.04_g++_x86_64.tar.gz <URL>

      - name: Build Docker image
        run: |
          docker build --platform linux/amd64 -t rxdock:latest .

      - name: Smoke test — verify rbdock binary runs
        run: |
          docker run --rm rxdock:latest rbdock -h
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docker-build.yml
git commit -m "ci: add GitHub Actions workflow for Docker build smoke test"
```

---

## Task 5: Add tethered/README.md

**Files:**
- Create: `tethered/README.md`

- [ ] **Step 1: Create `tethered/README.md`**

```markdown
# Tethered Docking

Tethered docking constrains ligand atoms to a reference pose using Maximum Common Substructure (MCS). Atoms shared between the query ligand and a reference molecule are held near their reference coordinates during docking; non-matching atoms dock freely.

## Files

| File | Description |
|------|-------------|
| `tetheredMinimization.py` | Prepares ligands for tethered docking by finding MCS with a reference and writing TETHERED ATOMS property |
| `receptor.mol2` | Prepared receptor structure |
| `xtal-lig.sd` | Crystal ligand (reference pose) |
| `cavity.prm` | rDock cavity definition |
| `cavity.as` | Pre-computed cavity grid |

## Prerequisites

- Python 3 with RDKit installed
- rDock/rxDock binary (`rbdock`, `rbcavity`) — use the Docker image from the repo root

## Step 1: Prepare tethered ligands

Replace `query_ligands.sdf` with your own SDF file of ligands to dock. `xtal-lig.sd` (included in this directory) is used as the reference pose.

```bash
python tetheredMinimization.py xtal-lig.sd query_ligands.sdf outputtethered.sdf outputnontethered.sdf
```

This produces two output files:
- `outputtethered.sdf` — ligands with a `TETHERED ATOMS` property set (atom indices of the MCS match)
- `outputnontethered.sdf` — ligands with no sufficient MCS match (will dock freely)

The `ratioThreshold` in the script (default `0.20`) controls the minimum fraction of the reference molecule that must match for tethering to apply.

## Step 2: Run tethered docking

```bash
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r cavity.prm -p dock.prm -i outputtethered.sdf -o output_tethered -T 1 -n 10
```

`dock.prm` is a standard rDock protocol file located at `$RBT_ROOT/data/scripts/dock.prm` inside the Docker image.

## Step 3: (Optional) Cavity definition

If you need to regenerate the cavity grid:

```bash
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbcavity -was -d -r cavity.prm
```
```

- [ ] **Step 2: Commit**

```bash
git add tethered/README.md
git commit -m "docs: add tethered docking README"
```

---

## Task 6: Add rna_docking_example/README.md

**Files:**
- Create: `rna_docking_example/README.md`

- [ ] **Step 1: Create `rna_docking_example/README.md`**

```markdown
# RNA Docking Example

This example demonstrates docking a small molecule to an RNA target using rDock/rxDock. RNA docking is supported natively — the setup is identical to protein docking; just provide an RNA receptor in MOL2 format.

## Target: 1NEM

PDB ID **1NEM** is an RNA aptamer structure. The example files are in the `1nem/` subdirectory.

## Files (`1nem/`)

| File | Description |
|------|-------------|
| `1nem_rdock.mol2` | RNA receptor prepared in MOL2 format |
| `1nem_lig.sd` | Reference ligand for cavity definition |
| `1nem_rdock.prm` | rDock parameter file (cavity definition) |
| `1nem_rdock.as` | Pre-computed cavity grid (regenerated by rbcavity) |
| `1nem_docking_out.sd` | Example docking output |

## Running the Example

All commands should be run from inside the `1nem/` directory.

### Step 1: Define the docking cavity

```bash
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbcavity -was -d -r 1nem_rdock.prm
```

This writes `1nem_rdock.as` (the cavity grid). Running `-was` will overwrite the existing `.as` file — this is expected.

### Step 2: Run docking

```bash
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r 1nem_rdock.prm -p dock.prm -i 1nem_lig.sd -o 1nem_docking_out -T 1 -n 10
```

`dock.prm` is the standard rDock docking protocol, located at `$RBT_ROOT/data/scripts/dock.prm` inside the Docker image. It is not bundled in this repo.

Output is written to `1nem_docking_out.sd`.

## Notes

- The cavity is defined using the reference ligand method (`RbtLigandSiteMapper`), RADIUS 4.0 Å around `1nem_lig.sd`
- `RECEPTOR_FLEX 3.0` allows flexible receptor atoms within 3 Å of the binding site
```

- [ ] **Step 2: Commit**

```bash
git add rna_docking_example/README.md
git commit -m "docs: add RNA docking example README"
```

---

## Task 7: Expand main README.md

**Files:**
- Modify: `README.md`

The current README only documents the `rbcavity` Docker command and nothing else. Add a section per example.

- [ ] **Step 1: Read the current README.md**

Confirm current content ends after the `rbcavity` command (line ~27).

- [ ] **Step 2: Append example sections to README.md**

Note: The existing README (line 27) includes `-u $(id -u ${USER}):$(id -g ${USER})` in the `rbcavity` command. The new example commands below intentionally omit it for brevity — on Linux hosts this will produce root-owned output files. If that matters for your environment, add `-u $(id -u):$(id -g)` to each `docker run` command.

Add after the existing content:

```markdown
## Examples

All examples use the Docker image. Build it first:

```bash
docker build --platform linux/amd64 -t rxdock:latest .
```

---

### Minimize

Minimize a docked ligand pose using the rDock minimization protocol.

```bash
cd minimize/
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r cavity.prm -p minimise.prm -i ii.sd -o output -T 1
```

---

### Score

Score a set of ligand poses against a receptor without moving them.

```bash
cd score/
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r cavity.prm -p score.prm -i ii.sd -o output -T 2 -n 1
```

---

### Pharmacophores

Dock with pharmacophore constraints. The `pharma.restr` file defines required interaction points.

```bash
cd pharmacophores/
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbcavity -was -d -r cavity.prm
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r cavity.prm -p dock.prm -i xtal-lig.sd -o output -T 1 -n 10 -s 42
```

---

### Solvent (Explicit Waters)

Dock with explicit, tethered water molecules occupying the binding site. Waters are treated as flexible solvent — they can translate and rotate within limits defined in `cavity.prm`.

> `lig.sdf` is included in the `solvent/` directory. `minimise.prm` is a standard rDock protocol bundled at `$RBT_ROOT/data/scripts/minimise.prm` inside the Docker image.

```bash
cd solvent/
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbcavity -was -d -r cavity.prm
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r cavity.prm -p minimise.prm -i lig.sdf -o output -T 1
```

---

### Tethered Docking

Dock ligands with atoms constrained to a reference pose via Maximum Common Substructure. See [`tethered/README.md`](tethered/README.md) for full details.

```bash
cd tethered/
python tetheredMinimization.py xtal-lig.sd query_ligands.sdf outputtethered.sdf outputnontethered.sdf
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r cavity.prm -p dock.prm -i outputtethered.sdf -o output_tethered -T 1 -n 10
```

---

### RNA Docking

Dock a small molecule to an RNA aptamer (PDB: 1NEM). See [`rna_docking_example/README.md`](rna_docking_example/README.md) for full details.

```bash
cd rna_docking_example/1nem/
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbcavity -was -d -r 1nem_rdock.prm
docker run --rm -v ${PWD}:/results -w /results rxdock:latest \
  rbdock -r 1nem_rdock.prm -p dock.prm -i 1nem_lig.sd -o 1nem_docking_out -T 1 -n 10
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: expand README with per-example usage sections"
```

---

## Task 8: Open the PR

- [ ] **Step 1: Push branch to remote**

```bash
git push -u origin update/modernize-components
```

- [ ] **Step 2: Open PR**

```bash
gh pr create \
  --title "Modernize repo: fix OpenMM imports, slim Docker image, add CI and docs" \
  --body "$(cat <<'EOF'
## Summary
- Fix deprecated `simtk.openmm` imports in all `prepare_protein.py` files (broken on OpenMM 8.0+)
- Switch Dockerfile base from `ubuntu:22.04` to `debian:bookworm-slim` (~30MB vs ~77MB), add `.dockerignore`, clean up duplicate `ENV PATH`
- Add GitHub Actions CI: builds image and smoke-tests `rbdock -h` on every push/PR
- Add missing READMEs for `tethered/` and `rna_docking_example/`
- Expand main README with Docker run commands for all 6 example types

## Test plan
- [ ] CI passes: `docker build` succeeds and `rbdock -h` exits 0
- [ ] `head -5 pharmacophores/prepare_protein.py` shows `from openmm import app, unit`
- [ ] `.dockerignore` does not contain `tar.gz`
- [ ] `tethered/README.md` and `rna_docking_example/README.md` exist
- [ ] Main README has sections for all 6 examples

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Confirm PR URL is returned and share with user**
