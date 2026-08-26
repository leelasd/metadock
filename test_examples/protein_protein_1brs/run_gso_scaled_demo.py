"""
Scaled-up GSO protein-protein docking demo, parametrized by scoring backend
(SCORING=openmm or SCORING=dfire env var) and scale, so the same script can
run the "isolate search algorithm" (DFIRE, same scoring as real LightDock)
and "isolate scoring function" (OpenMM physics) experiments at matching,
much larger scale than the original 20x20 toy comparison -- see
DOCUMENTATION.md for why both were needed: the original comparison used a
tiny 20-swarm/20-glowworm/50-step scale for both tools, which under-sampled
the receptor surface badly (real LightDock's own log showed 602-786
candidate SASA points before down-selecting to just 20-400 swarms).
"""
import os
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from openmm_dock.core import PDBParser
from openmm_dock.glowworm_swarm import (
    build_protein_protein_system, make_energy_fn, generate_surface_swarm_centers,
    generate_sasa_swarm_centers, GSOParameters, GlowwormSwarmOptimizer,
)

DEMO_DIR = Path(__file__).resolve().parent
SCORING = os.environ.get("SCORING", "openmm")
SWARM_METHOD = os.environ.get("SWARM_METHOD", "fibonacci")  # "fibonacci" or "sasa"
N_SWARMS = int(os.environ.get("N_SWARMS", "100"))
N_PER_SWARM = int(os.environ.get("N_PER_SWARM", "50"))
N_STEPS = int(os.environ.get("N_STEPS", "30"))

print("=" * 90)
print(f"   OPENMM-DOCK: SCALED GSO PROTEIN-PROTEIN DOCKING (1BRS) -- scoring={SCORING}, swarms={SWARM_METHOD}")
print(f"   [{N_SWARMS} swarms x {N_PER_SWARM} glowworms = {N_SWARMS * N_PER_SWARM} total, {N_STEPS} steps]")
print("=" * 90)

receptor = PDBParser.parse(DEMO_DIR / "barnase_receptor.pdb")
ligand_native = PDBParser.parse(DEMO_DIR / "barstar_ligand.pdb")
native_ligand_coords = ligand_native.coordinates


def rmsd_to_native(trans: np.ndarray, quat: np.ndarray, ligand_local: np.ndarray) -> float:
    rot = ScipyRotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
    world = ligand_local.dot(rot.T) + trans
    return float(np.sqrt(np.mean(np.sum((world - native_ligand_coords) ** 2, axis=1))))


print(f"\n[1] Building {SCORING} scoring backend...")
if SCORING == "dfire":
    from openmm_dock.lightdock_dfire_scoring import build_dfire_energy_fn
    energy_fn, ligand_local = build_dfire_energy_fn(
        str(DEMO_DIR / "barnase_receptor.pdb"), str(DEMO_DIR / "barstar_ligand.pdb"),
    )
else:
    system, context, integrator, rec_n, lig_n, ligand_local = build_protein_protein_system(receptor, ligand_native)
    energy_fn = make_energy_fn(context, rec_n, ligand_local)

print(f"\n[2] Distributing swarms over the receptor's surface (method={SWARM_METHOD})...")
ligand_radius = float(np.max(np.linalg.norm(native_ligand_coords - native_ligand_coords.mean(axis=0), axis=1)))
if SWARM_METHOD == "sasa":
    swarm_centers = generate_sasa_swarm_centers(
        str(DEMO_DIR / "barnase_receptor.pdb"), n_swarms=N_SWARMS, ligand_radius=ligand_radius,
    )
else:
    swarm_centers = generate_surface_swarm_centers(receptor.coordinates, n_swarms=N_SWARMS, ligand_radius=ligand_radius)

print(f"\n[3] Running GSO ({N_STEPS} steps, {N_SWARMS * N_PER_SWARM} total glowworms)...")
params = GSOParameters()
optimizer = GlowwormSwarmOptimizer(energy_fn, params)
rng = np.random.default_rng(42)
swarm = optimizer.initialize_swarm(swarm_centers, n_per_swarm=N_PER_SWARM, rng=rng, jitter=2.0)

init_rmsds = [rmsd_to_native(g.trans, g.quat, ligand_local) for g in swarm]
print(f"    Initial swarm RMSD-to-native range: {min(init_rmsds):.1f}-{max(init_rmsds):.1f} Å "
      f"(median {np.median(init_rmsds):.1f} Å)")

t0 = time.time()
result = optimizer.run(swarm, n_steps=N_STEPS, rng=rng)
elapsed = time.time() - t0
print(f"    Run completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

print("\n[*] Top 10 glowworms by final energy:")
for i, g in enumerate(result[:10]):
    r = rmsd_to_native(g.trans, g.quat, ligand_local)
    print(f"    #{i + 1}: Energy = {g.energy:.2f} | RMSD to native = {r:.2f} Å")

all_rmsds = [rmsd_to_native(g.trans, g.quat, ligand_local) for g in result]
best = result[0]
best_rmsd = rmsd_to_native(best.trans, best.quat, ligand_local)

print("\n" + "=" * 80)
print(f"FINAL RESULT: Scaled GSO Protein-Protein Docking (scoring={SCORING})")
print(f"  Best-scoring pose : {best_rmsd:.2f} Å RMSD to native (Energy {best.energy:.2f})")
print(f"  Best RMSD found anywhere in final swarm : {min(all_rmsds):.2f} Å")
print(f"  Top-10-by-score RMSD range : {min(rmsd_to_native(g.trans, g.quat, ligand_local) for g in result[:10]):.2f}"
      f"-{max(rmsd_to_native(g.trans, g.quat, ligand_local) for g in result[:10]):.2f} Å")
print("=" * 80)
