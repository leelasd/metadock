# rdock-examples
Different use cases for using rDock

Download the latest rDock release tarball from the CBDD GitHub releases page and place it in the repo root before building the Docker image:
https://github.com/CBDD/rDock/releases

## Building Docker Image

```bash
docker build --platform linux/amd64 --pull --rm -f "Dockerfile" -t rxdock:latest "."
```

For running different executables 

```bash
 docker run -it rxdock:latest rbcavity -h
```

```bash
 docker run -it rxdock:latest rbdock -h
```

## Docking in 3 Steps 


```bash
docker run --rm -v ${PWD}:/results -w /results -u $(id -u ${USER}):$(id -g ${USER}) rxdock:latest rbcavity -was -d -r cavity.prm
```

## Examples

All examples use the Docker image. Build it first:

```bash
docker build --platform linux/amd64 -t rxdock:latest .
```

---

### Minimize

Minimize a docked ligand pose using the rDock minimization protocol.

> `minimise.prm`, `score.prm`, and `dock.prm` are standard rDock protocol files bundled at `$RBT_ROOT/data/scripts/` inside the Docker image — they are not in this repo.

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

---

## Known Issues & Troubleshooting

### Pharmacophore Constraints: Wrong Coordinates

**Symptom:** `RBT_LIGAND_ERROR: The ligand has only N aromatic ring(s) (M required)`

This error means the mandatory constraints in `pharma.restr` require pharmacophore features the ligand does not have. This can happen for two reasons:

1. **Constraint file copied from a different system.** Each `pharma.restr` is receptor- and ligand-specific — coordinates must match the actual binding site geometry. Use RDKit to extract real pharmacophore feature coordinates from your reference ligand:

   ```python
   from rdkit import Chem
   from rdkit.Chem import AllChem

   mol = Chem.MolFromMolFile('xtal-lig.sd', removeHs=False)
   # Aromatic ring centers
   for ring in mol.GetRingInfo().AtomRings():
       if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
           coords = [mol.GetConformer().GetAtomPosition(a) for a in ring]
           cx = sum(p.x for p in coords) / len(coords)
           cy = sum(p.y for p in coords) / len(coords)
           cz = sum(p.z for p in coords) / len(coords)
           print(f"Aro: {cx:.2f} {cy:.2f} {cz:.2f}")
   # Acceptor/donor atoms (N, O positions)
   for atom in mol.GetAtoms():
       if atom.GetAtomicNum() in (7, 8):
           p = mol.GetConformer().GetAtomPosition(atom.GetIdx())
           print(f"{atom.GetSymbol()}{atom.GetIdx()}: {p.x:.2f} {p.y:.2f} {p.z:.2f}")
   ```

   `pharma.restr` format: `x y z tolerance type` where type is one of `Aro`, `Acc`, `Don`, `Hyd`.

2. **SDF file uses MDL aromatic bond type 4.** rDock's mol file parser does not handle bond type 4 correctly — it warns `"X makes too many bonds"` and may miscount aromatic rings. Convert to Kekulé form first:

   ```python
   from rdkit import Chem

   mol = Chem.MolFromMolFile('xtal-lig.sd', removeHs=False)
   Chem.Kekulize(mol, clearAromaticFlags=True)
   writer = Chem.SDWriter('xtal-lig-kekule.sd')
   writer.SetKekulize(True)
   writer.write(mol)
   writer.close()
   ```

   After conversion all bonds will be type 1 (single) or 2 (double) and rDock will correctly identify all aromatic rings.

---

### Python API: `simtk.openmm` ImportError

**Symptom:** `ModuleNotFoundError: No module named 'simtk'`

OpenMM 8.0 removed the legacy `simtk` namespace. Replace all `simtk` imports:

```python
# Old (OpenMM < 8.0)
from simtk.openmm import app
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit

# New (OpenMM >= 8.0)
from openmm import app, unit
from openmm.app import *
```

The wildcard `from openmm.app import *` is intentional — it exposes bare names like `PDBFile`, `ForceField`, and `Simulation` that the scripts use without qualification.

---

### MDL Aromatic Bond Type 4 in Any Input SDF

**Symptom:** `WARNING: X makes too many bonds` followed by wrong ring counts or `Terminate with this ligand` without scoring.

rDock's mol file parser does not handle MDL bond type 4 (aromatic notation) correctly — it can miscount rings, misassign valences, or skip the ligand entirely. This affects any SDF exported from tools like PyMOL or RDKit when aromaticity is written in the aromatic bond notation rather than the Kekulé alternating-bond form.

**Fix — kekulize any input SDF before passing it to rDock:**

```python
from rdkit import Chem

mol = Chem.MolFromMolFile('input.sd', removeHs=False)
Chem.Kekulize(mol, clearAromaticFlags=True)
writer = Chem.SDWriter('input-kekule.sd')
writer.SetKekulize(True)
writer.write(mol)
writer.close()
```

This was required for `pharmacophores/xtal-lig.sd`, `score/ii.sd`, and `tethered/query_ligands.sdf`.

---

### Tethered Docking: Missing `query_ligands.sdf`

`tetheredMinimization.py` expects a user-supplied `query_ligands.sdf` containing the ligands to dock. A sample input (`ligand_htt2_fixed.mol`) is included in the `tethered/` directory. Convert it to a kekulized SDF before use:

```python
from rdkit import Chem

mol = Chem.MolFromMolFile('tethered/ligand_htt2_fixed.mol', removeHs=False)
Chem.Kekulize(mol, clearAromaticFlags=True)
writer = Chem.SDWriter('tethered/query_ligands.sdf')
writer.SetKekulize(True)
writer.write(mol)
writer.close()
```

Then run `tetheredMinimization.py` as described in `tethered/README.md`.

---

### Docker: Base Image and rDock Tarball

The Dockerfile uses `debian:bookworm-slim` with the **debian-11** rDock build from the CBDD/rDock releases page. Do not substitute a `ubuntu-20.04` or `ubuntu-22.04` tarball — the debian-11 build is the correct native match and avoids `libpopt` and `GLIBC` version mismatches.

Download the tarball before building:

```bash
wget https://github.com/CBDD/rDock/releases/download/v24.04.204-legacy/rdock-legacy-24.04.204_debian-11_g%2B%2B_x86_64.tar.gz
docker build --platform linux/amd64 -t rxdock:latest .
```
