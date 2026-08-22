"""
Glowworm Swarm Optimization protein-protein docking on the barnase (chain A,
fixed receptor) / barstar (chain D, mobile partner) pair from PDB 1BRS --
openmm_dock.glowworm_swarm's own re-implementation of the LightDock
algorithm (see that module's docstring), scored with real OpenMM physics
instead of a statistical potential.

Run at matching scale to the real-LightDock comparison run in this same
directory (lightdock_run/): 20 swarms x 20 glowworms, 50 steps.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from openmm_dock.core import PDBParser
from openmm_dock.glowworm_swarm import (
    build_protein_protein_system, make_energy_fn, generate_surface_swarm_centers,
    GSOParameters, GlowwormSwarmOptimizer,
)

DEMO_DIR = Path(__file__).resolve().parent
N_SWARMS = 20
N_PER_SWARM = 20
N_STEPS = 50

print("=" * 90)
print("   OPENMM-DOCK: GLOWWORM SWARM OPTIMIZATION PROTEIN-PROTEIN DOCKING (1BRS Barnase-Barstar)")
print(f"   [{N_SWARMS} swarms x {N_PER_SWARM} glowworms, {N_STEPS} steps -- matching the real-LightDock comparison run]")
print("=" * 90)

receptor = PDBParser.parse(DEMO_DIR / "barnase_receptor.pdb")
ligand_native = PDBParser.parse(DEMO_DIR / "barstar_ligand.pdb")
native_ligand_coords = ligand_native.coordinates


def rmsd_to_native(trans: np.ndarray, quat: np.ndarray, ligand_local: np.ndarray) -> float:
    rot = ScipyRotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
    world = ligand_local.dot(rot.T) + trans
    return float(np.sqrt(np.mean(np.sum((world - native_ligand_coords) ** 2, axis=1))))


print("\n[1] Building OpenMM protein-protein rigid-body scoring system...")
system, context, integrator, rec_n, lig_n, ligand_local = build_protein_protein_system(receptor, ligand_native)
energy_fn = make_energy_fn(context, rec_n, ligand_local)
print(f"    Receptor: {rec_n} atoms (fixed) | Mobile partner: {lig_n} atoms")

print("\n[2] Distributing swarms evenly over the receptor's surface (blind global coverage)...")
ligand_radius = float(np.max(np.linalg.norm(native_ligand_coords - native_ligand_coords.mean(axis=0), axis=1)))
swarm_centers = generate_surface_swarm_centers(receptor.coordinates, n_swarms=N_SWARMS, ligand_radius=ligand_radius)
print(f"    {N_SWARMS} swarm centers generated around the receptor.")

print(f"\n[3] Running GSO ({N_STEPS} steps, {N_SWARMS * N_PER_SWARM} total glowworms)...")
params = GSOParameters()
optimizer = GlowwormSwarmOptimizer(energy_fn, params)
rng = np.random.default_rng(42)
swarm = optimizer.initialize_swarm(swarm_centers, n_per_swarm=N_PER_SWARM, rng=rng, jitter=2.0)

init_rmsds = [rmsd_to_native(g.trans, g.quat, ligand_local) for g in swarm]
print(f"    Initial swarm RMSD-to-native range: {min(init_rmsds):.1f}-{max(init_rmsds):.1f} Å "
      f"(median {np.median(init_rmsds):.1f} Å) -- genuinely blind start.")

result = optimizer.run(swarm, n_steps=N_STEPS, rng=rng)

print("\n[*] Top 10 glowworms by final energy:")
for i, g in enumerate(result[:10]):
    r = rmsd_to_native(g.trans, g.quat, ligand_local)
    print(f"    #{i + 1}: Energy = {g.energy:.2f} kcal/mol | RMSD to native = {r:.2f} Å")

best = result[0]
best_rmsd = rmsd_to_native(best.trans, best.quat, ligand_local)
all_rmsds = [rmsd_to_native(g.trans, g.quat, ligand_local) for g in result]

print("\n" + "=" * 80)
print("FINAL RESULT: OpenMM-Dock GSO Protein-Protein Docking")
print(f"  Best-scoring pose : {best_rmsd:.2f} Å RMSD to native (Energy {best.energy:.2f} kcal/mol)")
print(f"  Best RMSD found anywhere in final swarm : {min(all_rmsds):.2f} Å")
print("=" * 80)
