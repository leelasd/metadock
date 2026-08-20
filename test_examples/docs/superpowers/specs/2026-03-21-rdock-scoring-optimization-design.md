# rDock Project-Specific Scoring Optimization

**Date:** 2026-03-21
**Status:** Approved

## Goals

Improve rDock scoring for a drug discovery project using crystal structures and SAR data. Priorities in order:

1. **Pose selection** — correct binding mode ranks #1 among docking poses
2. **Virtual screening enrichment** — actives rank above inactives
3. **Potency prediction** — score correlates with measured pIC50/IC50

## Approach

**Option A: rDock Weight Reoptimization**

rDock writes each scoring term separately into output SDF files (`SCORE.INTER.VDW`, `SCORE.INTER.POLAR`, etc.). The optimizer reads these pre-computed components and finds the weight combination that best satisfies the composite objective — **no re-docking needed during optimization**. Each Optuna trial is a matrix multiply over stored score components (~1ms/trial).

Crystal structures serve two roles:
- Seed pharmacophore constraints (`pharma.restr`) per chemical series
- Supply conserved binding-site waters as tethered solvent

## Data Requirements

| Input | Minimum | Source |
|---|---|---|
| Crystal structures | 5+ (ideally 10–20) | PDB or in-house |
| SAR compounds with pIC50 | 50+ actives | Assay database |
| Inactive compounds | 50+ | Assay database / decoys |

## Parameters: Three Tiers

Parameters are split across three tiers by cost of change and scope of effect.

### Tier 1 — Post-hoc reweightable (no re-docking)

These correspond directly to SDF fields written by rDock. Changing their weights during Optuna optimization only requires re-computing the weighted sum over stored pose scores.

| Parameter | File | SDF Field | Default | Search Range |
|---|---|---|---|---|
| `WEIGHT` (VDW) | `RbtInterIdxSF.prm` | `SCORE.INTER.VDW` | 1.0 | 0.1 – 3.0 |
| `WEIGHT` (POLAR) | `RbtInterIdxSF.prm` | `SCORE.INTER.POLAR` | 3.4 | 0.5 – 8.0 |
| `WEIGHT` (REPUL) | `RbtInterIdxSF.prm` | `SCORE.INTER.REPUL` | 5.0 | 1.0 – 10.0 |
| `WEIGHT` (CONST) | `RbtInterIdxSF.prm` | `SCORE.INTER.CONST` | 5.4 | 1.0 – 10.0 |
| `WEIGHT` (ROT) | `RbtInterIdxSF.prm` | `SCORE.INTER.ROT` | 1.0 | 0.1 – 3.0 |
| `WEIGHT` (PHARMA) | `cavity.prm` | `SCORE.RESTR` | 2.0 | 0.5 – 5.0 |
| `WEIGHT` (CAVITY) | `cavity.prm` | `SCORE.RESTR.CAVITY` | 1.0 | 0.5 – 3.0 |
| `WEIGHT` (SYSTEM.VDW) | `RbtInterIdxSF.prm` | `SCORE.SYSTEM.VDW` | 1.0 | 0.1 – 3.0 |
| `WEIGHT` (SYSTEM.POLAR) | `RbtInterIdxSF.prm` | `SCORE.SYSTEM.POLAR` | 1.0 | 0.1 – 3.0 |

`SCORE.SYSTEM.*` captures water–protein and water–ligand interaction energy, giving the optimizer signal on how much explicit solvent contributions should count.

### Tier 2 — Requires re-docking

These parameters control the docking search space and cannot be reweighted post-hoc. They are tuned in a separate, less frequent outer loop (e.g., once per project phase) triggered manually.

| Parameter | File | Default | Controls |
|---|---|---|---|
| `SOLVENT_PENALTY` | `RbtInterIdxSF.prm` (CONST section) | 0.37 | Cost of binding each explicit water |
| `MAX_TRANS` | `cavity.prm` (SOLVENT section) | 1.0 Å | Crystal water tether radius |
| `OCCUPANCY` | `cavity.prm` (SOLVENT section) | 0.5 | Min crystallographic occupancy to include a water |

### Tier 3 — Atom-type parameters (requires re-docking + expert review)

These live in `$RBT_HOME/sf/` override files and change the fundamental physical model applied to every atom of a given type across the entire receptor and ligand. They are the deepest level of customization available without modifying rDock source code. Changes require full re-docking of the SAR set and should be driven by explicit structural or SAR evidence.

**Mechanism:** rDock checks `$RBT_HOME/sf/` before `$RBT_ROOT/sf/`. Placing a modified file in the project's `sf/` directory and setting `RBT_HOME` in the Docker container is sufficient — no recompilation needed.

```bash
# Project layout
project/
  sf/
    Tripos52_vdw.prm      # overrides /app/data/sf/Tripos52_vdw.prm
    RbtIonicAtoms.prm     # overrides /app/data/sf/RbtIonicAtoms.prm

# Pass RBT_HOME at runtime
docker run --rm -v ${PWD}:/work -w /work \
  -e RBT_HOME=/work \
  rxdock:latest rbdock -r cavity.prm -p dock.prm -i ligands.sdf -o output -n 10
```

#### `Tripos52_vdw.prm` — per-atom-type physical parameters

Each Tripos atom type section defines:

| Field | Meaning | Effect on scoring |
|---|---|---|
| `R` | vdW radius (Å) | Controls steric fit and shape complementarity |
| `K` | vdW well depth (kcal/mol) | Controls strength of dispersion attraction |
| `IP` | Ionization potential | London dispersion term |
| `POL` | Polarizability | London dispersion term |
| `isHBA TRUE` | H-bond acceptor flag | Atom participates in `SCORE.INTER.POLAR` |
| `isHBD TRUE` | H-bond donor flag | Atom participates in `SCORE.INTER.POLAR` |

#### `RbtIonicAtoms.prm` — per-residue formal charges

Defines partial charges on specific atoms in named residues. These feed the electrostatic component of `SCORE.INTER.POLAR`. Includes standard charged residues (LYS, ARG, ASP, GLU, HIS) and metal ions (ZN, MG, CA, FE).

#### Project-specific modifications — when and what to change

| Target scenario | File | Modification |
|---|---|---|
| Cysteine-targeting series | `Tripos52_vdw.prm` | Increase `R` and `K` for `S.3` to make the pocket more attractive |
| Halogen bond donors | `Tripos52_vdw.prm` | Add `isHBA TRUE` to `Cl`/`Br`/`I` sections |
| Flat amines not recognized as donors | `Tripos52_vdw.prm` | Add `isHBD TRUE` to `N.pl3` |
| Metalloprotein (e.g., ZN coordination) | `RbtIonicAtoms.prm` | Tune ZN charge or add custom metal residue section |
| Non-standard protonation states | `RbtIonicAtoms.prm` | Add/remove charges for specific residue + atom name pairs |
| Non-standard amino acids / covalent adducts | `RbtIonicAtoms.prm` + `Tripos52_vdw.prm` | Add new SECTION for the modified residue |

#### Tier 3 workflow

Tier 3 changes are made manually by a computational chemist with structural justification. After modifying the parameter files:

1. Re-dock the full SAR set with the new `RBT_HOME` override
2. Run the Tier 1 optimizer on the new pose library
3. Compare holdout AUC-ROC and Spearman ρ against the Tier 3 baseline
4. Commit the modified `sf/` files to the project repo alongside the resulting `metrics.json`

Versioned `sf/` overrides are stored alongside configs in S3:

```
s3://project/configs/
    v3/
        RbtInterIdxSF.prm
        cavity.prm
        sf/
            Tripos52_vdw.prm     ← only present if Tier 3 was changed
            RbtIonicAtoms.prm    ← only present if Tier 3 was changed
        metrics.json
```

If no `sf/` subdirectory is present in a config version, the container defaults (`$RBT_ROOT/sf/`) are used unchanged.

## Objective Function

```
objective = α·RMSD_loss + β·(1 - AUC_ROC) + γ·(1 - Spearman_ρ)
```

| Term | Weight | Measured on |
|---|---|---|
| `RMSD_loss` | α | Crystal-matched SAR compounds only (see Compound–Crystal Matching below) |
| `1 - AUC_ROC` | β | Full SAR set (active/inactive labels) |
| `1 - Spearman_ρ` | γ | Full SAR set (pIC50 values) |

Default α/β/γ values and their project-phase rationale:

| Phase | α | β | γ | Rationale |
|---|---|---|---|---|
| Lead ID | 0.5 | 0.4 | 0.1 | Enrichment dominates; few potency data |
| Lead Opt | 0.3 | 0.3 | 0.4 | Potency correlation becomes primary signal |
| Custom | configurable | configurable | configurable | Set in pipeline config |

α is scaled by the crystal-matched fraction: `α_eff = α × (n_crystal_matched / n_total)`. This prevents RMSD_loss from dominating when only a small fraction of SAR compounds have crystal counterparts.

### Compound–Crystal Matching

Each SAR compound is assigned to the crystal structure with the highest MCS Tanimoto similarity (RDKit, Morgan radius 2) to the co-crystal ligand. Only SAR compounds with Tanimoto ≥ 0.6 to their best-matching crystal ligand contribute to `RMSD_loss`. The matched subset size and threshold are recorded in `metrics.json` for each run.

### Validation Split

The SAR set is partitioned by assay date:
- **Train (80%)**: compounds assayed before the split date
- **Holdout (20%)**: most recently assayed compounds

Optuna optimizes on the training split. The promotion gate evaluates AUC-ROC and Spearman ρ on the **holdout split only**, preventing overfitting across the 9-parameter search.

## AWS Pipeline Architecture

```
New SAR data uploaded to S3
         │
         ▼
[EventBridge trigger]
         │
         ▼
[Step Functions workflow]
    │
    ├─ [Guard] Delta counter check (DynamoDB)
    │       - Count new compounds accumulated since last training run
    │       - Abort if delta < 10 (prevents spurious retraining from small uploads)
    │       - Increment counter on each upload; reset to 0 after training completes
    │
    ├─ [1] Crystal Structure Processing (AWS Batch)
    │       - Parse PDB/mol2 crystal structures
    │       - Extract conserved waters within 5Å of ligand (present in ≥50% of structures)
    │       - Assign SAR compounds to crystal series (MCS Tanimoto ≥ 0.6)
    │       - Generate pharma.restr per series (2–4 constraints: Aro + Acc/Don)
    │       - Output: pharma.restr files + water coordinates → S3
    │
    ├─ [2] Docking (AWS Batch, pinned rxdock ECR image)
    │       - Dock all new SAR compounds (n_poses=20 per compound)
    │       - Write full score component breakdown to output SDF
    │       - Tag each output SDF with container image digest in S3 metadata
    │       - Output: posed SDF files with SCORE.* fields → S3
    │
    ├─ [3] Weight Optimization — Tier 1 (SageMaker job)
    │       - Load poses + score components + pIC50 labels + crystal RMSDs
    │       - Apply train/holdout split (80/20 by assay date)
    │       - Run Optuna TPE, 500 trials, optimize on training split
    │       - Evaluate composite objective on holdout split
    │       - Output: optimized RbtInterIdxSF.prm + cavity.prm → S3
    │
    └─ [4] Config Versioning (Lambda)
            - Write new .prm files to s3://project/configs/v{N}/
            - Write metrics.json (holdout AUC, Spearman ρ, mean RMSD, n_compounds,
              n_crystal_matched, container_digest, α/β/γ used)
            - Promote by updating s3://project/configs/current.json pointer file
              ONLY IF holdout ΔAUC > 0.02 AND ΔSpearman > 0.03 vs previous version
```

### AWS Services

| Service | Role |
|---|---|
| S3 | Data store: crystal structures, SAR CSVs, docked poses, versioned configs |
| EventBridge | Trigger pipeline on new SAR data in S3 |
| Step Functions | Pipeline orchestration + delta guard |
| DynamoDB | Cumulative new-compound counter across partial uploads |
| AWS Batch | Docking jobs (CPU, pinned rxdock ECR image) |
| SageMaker Training | Optuna weight optimization (CPU, ~5 min per run) |
| Lambda | Config versioning, pointer promotion logic |
| ECR | Container registry for rxdock — images tagged by digest, never `latest` |

## Crystal Structure Processing Detail

For each crystal structure:
1. Extract co-crystal ligand and binding site residues
2. Find all crystallographic waters within 5Å of ligand
3. Flag waters conserved in ≥50% of crystal structures as tethered solvent
4. Run RDKit pharmacophore analysis (aromatic ring centers, acceptors, donors)
5. Write `pharma.restr` with 2–4 mandatory constraints (Aro + Acc/Don) per series
6. Write `cavity.prm` SOLVENT section with conserved water coordinates

Series assignment: MCS-based scaffold clustering using RDKit Murcko scaffolds. SAR compounds sharing the same Murcko scaffold as a crystal ligand (Tanimoto ≥ 0.6) are assigned to that crystal's pharmacophore and water set. Compounds matching no crystal scaffold receive a global pharmacophore (most common constraints across all series) or no pharmacophore if the global set is too restrictive.

## Config Versioning

Each optimization run produces a versioned config set:

```
s3://project/configs/
    v1/
        RbtInterIdxSF.prm
        cavity.prm
        metrics.json
    v2/
        RbtInterIdxSF.prm
        cavity.prm
        metrics.json
    current.json   ← { "version": "v2", "promoted_at": "2026-03-21T..." }
```

`current.json` is updated by the Lambda step on promotion. Docking jobs read their config by fetching `current.json` first, then loading the referenced version. No S3 symlinks are used.

## Tier 2 and Tier 3 Retuning (Manual, Per Project Phase)

**Tier 2** (`SOLVENT_PENALTY`, `MAX_TRANS`, `OCCUPANCY`) and **Tier 3** (atom-type vdW/polar/charge parameters) both require re-docking. These are tuned in a separate outer loop:
- Triggered manually (not by EventBridge)
- Tier 2: grid search or short Optuna study over the 3 solvent parameters
- Tier 3: single change per structural hypothesis, evaluated by a computational chemist
- For each candidate config: re-dock the full SAR set, then run the Tier 1 optimizer on the new pose library
- Produces a new baseline docked-pose library stored in S3
- Recommended cadence: once at lead ID entry, once at lead optimization entry
- Tier 3 `sf/` overrides are committed to the project repo and stored under the config version in S3

## Container Reproducibility

All docking jobs are run with a pinned ECR image digest, never `rxdock:latest`. The digest is stored in S3 metadata alongside each docked SDF batch. The SageMaker optimization job refuses to mix pose batches from different container digests in a single training run to prevent score distribution shift.

## Promotion Gate

A new config version is promoted to `current` only when all of the following hold on the **holdout** split:
- `ΔAUC-ROC > 0.02` vs the current production version
- `ΔSpearman ρ > 0.03` vs the current production version
- `n_holdout_compounds ≥ 20` (sufficient holdout size for reliable metrics)

If promotion fails, the new version is stored in S3 for inspection but the `current.json` pointer is not updated.

## Out of Scope

- Modifying rDock C++ source
- GPU-based deep learning rescoring
- Ligand preparation (handled upstream)
- ADMET filtering (handled downstream)
- Tier 2 and Tier 3 parameter automation (manual trigger only)
