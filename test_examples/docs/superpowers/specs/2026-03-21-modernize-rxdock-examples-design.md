# Design: Modernize rxdock-deepdive-examples

**Date:** 2026-03-21
**Status:** Approved

## Overview

Modernize the `rxdock-deepdive-examples` repository by fixing deprecated Python imports, improving the Dockerfile, adding CI/CD, and filling documentation gaps. All changes land in a single branch `update/modernize-components` with one PR to `main`.

## Branch & PR Strategy

- Branch: `update/modernize-components`
- One PR to `main` covering all changes
- Each logical change in its own commit for reviewability

## Change Set

### 1. Fix deprecated `simtk.openmm` imports

**Files:** `pharmacophores/prepare_protein.py`, `solvent/prepare_protein.py`, `tethered/prepare_protein.py`

The `simtk` namespace was deprecated in OpenMM 7.6 and removed in OpenMM 8.x. All three files must be updated. Each file has a duplicate `from simtk.openmm.app import *` on lines 3-4 — both lines should be collapsed to one.

Before (lines 1-7 in each file):
```python
import parmed
from simtk.openmm import app
from simtk.openmm.app import *
from simtk.openmm.app import *   # duplicate line — remove
from simtk.openmm import *
from simtk import unit
#from simtk.unit import *
```

After:
```python
import parmed
from openmm import app, unit
from openmm.app import *
```

Notes:
- `from openmm.app import *` is what makes bare `PDBFile(...)` on line 8 resolve — this wildcard is intentional and must be kept.
- `unit` lives at `openmm.unit` but is also re-exported from the top-level `openmm` package (>= 7.6), so `from openmm import unit` is safe.
- `unit` is imported but never used in the script body — it is kept for API completeness and future use.
- `from simtk.openmm import *` (line 5 of the originals) is intentionally dropped. It exported core OpenMM classes (`System`, `Context`, etc.) but none of those names appear bare in these script bodies — all objects are obtained via method calls. Safe to remove.
- `solvent/prepare_protein.py` and `tethered/prepare_protein.py` have an extra `receptor_structure.save('receptor.gro', ...)` line that `pharmacophores/prepare_protein.py` does not. The import changes apply identically to all three; this one-line functional difference is preserved as-is.
- `parmed.openmm.load_topology(topology, system, xyz)` remains the correct API through ParmEd 4.x with OpenMM 8.x — no changes needed to the parmed call itself.
- Recommend pinning `parmed >= 4.0` and `openmm >= 8.0` in any environment setup docs. Current stable versions: OpenMM 8.1.x, ParmEd 4.2.x.

### 2. Dockerfile: switch to slim base image

The original Dockerfile uses `ubuntu:22.04` (full). Replace with `debian:bookworm-slim` to reduce image bloat. The rDock binary tarball (`rdock-legacy-24.04.204_ubuntu-20.04_g++_x86_64.tar.gz`) was built against Ubuntu 20.04 (glibc 2.31); `debian:bookworm-slim` ships glibc 2.36 which is backward compatible.

Changes:
- Base image: `ubuntu:22.04` → `debian:bookworm-slim`
- Keep `libpopt-dev` and `libpopt0` (available in Debian bookworm)
- Add `apt-get clean && rm -rf /var/lib/apt/lists/*` after install to shrink layer
- Remove the duplicate `ENV PATH` line in the current Dockerfile (lines 6 and 9 both set `PATH` to `/app/bin` — keep only the `RBT_ROOT`-based one)
- Add `.dockerignore` (see below) — **do not exclude `*.tar.gz`** since the `COPY` instruction requires the tarball in build context
- Add inline comment pointing to CBDD/rDock GitHub releases for the tarball download
- Add `LABEL` metadata (description, maintainer)

**.dockerignore patterns** (must NOT exclude `*.tar.gz`):
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

### 3. GitHub Actions CI

Add `.github/workflows/docker-build.yml`:
- Triggers on push and pull_request to `main`
- Runs `docker build` as smoke test
- Also runs `docker run rxdock:latest rbdock -h` to confirm the binary executes (catches runtime library mismatches that build-only tests miss)
- Uses `ubuntu-latest` runner

### 4. Add `tethered/README.md`

Document:
- Purpose: tethered docking/minimization using MCS-based atom constraints
- Prerequisites: RDKit
- Usage (matching the script's own help text exactly):
  ```
  python tetheredMinimization.py reference.sdf ligand.sdf outputtethered.sdf outputnontethered.sdf
  ```
- How to run rbdock with the tethered output

### 5. Add `rna_docking_example/README.md`

Document:
- Purpose: RNA docking using the 1NEM structure
- Files are under `rna_docking_example/1nem/`: `1nem_rdock.mol2`, `1nem_lig.sd`, `1nem_rdock.prm`
- Commands (run from inside `rna_docking_example/1nem/`):
  ```
  rbcavity -was -d -r 1nem_rdock.prm
  rbdock -r 1nem_rdock.prm -p dock.prm -i 1nem_lig.sd -o 1nem_docking_out -T 1 -n 10
  ```
- Note: `dock.prm` is a standard rDock protocol file located at `$RBT_ROOT/data/scripts/dock.prm` — it is not bundled in the repo. The README must make clear this requires the rDock binary environment (via Docker) to be configured with `$RBT_ROOT` set.
- Note: running `rbcavity -was` will regenerate/overwrite the existing `1nem_rdock.as` cavity file — this is expected behaviour.

### 6. Expand main `README.md`

Add a section per example (minimize, score, pharmacophores, solvent, tethered, RNA) with:
- Brief description of what the example demonstrates
- Docker run command to execute it

## Testing Approach

- **Python imports**: Verified by inspection. The wildcard `from openmm.app import *` is what resolves bare names like `PDBFile` and `ForceField` — this is intentional. Runtime correctness depends on compatible `parmed` and `openmm` versions in the user's environment.
- **Dockerfile**: CI smoke-tests `docker build` and `docker run rbdock -h` on every push to catch both build and runtime failures.
- **READMEs**: Manually reviewed against existing `.prm` files and script source for accuracy.

## Out of Scope

- Updating the rDock binary itself
- Adding Python unit tests (no test framework exists in the repo)
- Pinning a Python environment / requirements.txt (no Python packaging in repo)
- Migrating from tarball-based Dockerfile to a source build
