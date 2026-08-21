# Scoring Example

Demonstrates `openmm-dock`'s native scoring: evaluating a ligand pose against a receptor cavity without moving it, with an optional flexible-pocket variant and a Genetic Algorithm local-refinement comparison. This directory's files originate from rDock's own reference test set (`score.prm` was rDock's original scoring protocol) but are read and scored entirely natively here — no rDock/rxDock binary is required.

## Files

| File | Description |
|------|-------------|
| `run_score_demo.sh` | Native end-to-end demo: score → flexible-pocket score → GA refinement |
| `receptor.mol2` | Receptor structure |
| `cavity.prm` | Cavity definition |
| `ii.sd` | Ligand pose to score |

## Running the native demo

```bash
bash run_score_demo.sh
```

Produces `openmm_score_out.sdf` (rigid score of the pose as-is), `openmm_score_flex_out.sdf`
(pose after a 5 Å flexible-pocket minimization — `score` itself evaluates a fixed pose and
has no `--flex-radius` option, so flexibility is demonstrated via `minimize` instead), and
`openmm_ga_out.sdf` (GA-refined poses).
