"""
Demonstration of Kinematic Particle Swarm Optimization (Kin-PSO).
Runs a 20-particle swarm on PDB 6DI9 and exports the full multi-particle
convergence trajectory viewable in PyMOL.
"""
import shutil
from pathlib import Path
import numpy as np
from rdkit import Chem

from openmm_dock import DockingEngine, CavityDefinition
from openmm_dock.kinematics import (
    LigandKinematicTree,
    KinematicDockingEngine,
    KinematicParticleSwarmOptimizer
)

DEMO_DIR = Path(__file__).resolve().parent
DIR_6DI9 = DEMO_DIR.parent / "covalent_docking" / "6di9"

# Copy 6DI9 assets
shutil.copy(DIR_6DI9 / "receptor.pdb", DEMO_DIR / "receptor.pdb")
shutil.copy(DIR_6DI9 / "cavity.prm", DEMO_DIR / "cavity.prm")
shutil.copy(DIR_6DI9 / "xtal_ligand.sdf", DEMO_DIR / "xtal_ligand.sdf")

print("=" * 80)
print("     OPENMM-DOCK: KINEMATIC PARTICLE SWARM OPTIMIZATION (KIN-PSO)")
print("=" * 80)

lig_path = DEMO_DIR / "xtal_ligand.sdf"
lig_mol = Chem.SDMolSupplier(str(lig_path), removeHs=False)[0]

cavity = CavityDefinition.from_prm_file(DEMO_DIR / "cavity.prm")
engine = DockingEngine(receptor_path=DEMO_DIR / "receptor.pdb", cavity=cavity, covalent_res="CYS481")
kin_engine = KinematicDockingEngine(engine, lig_mol, covalent_res="CYS481")
pso = KinematicParticleSwarmOptimizer(kin_engine)

print("[*] Launching Kin-PSO: 20 Particles x 20 Iterations (400 Swarm Poses)...")
best_mol, best_score, swarm_frames = pso.run_pso(
    n_particles=20,
    n_iterations=20,
    ref_mol=lig_mol
)

print(f"\n[✓] Kin-PSO Optimization Converged to Global Best Score: {best_score:.3f} kcal/mol")

# Save Best Pose & Full Swarm Movie
out_best = DEMO_DIR / "pso_best_pose.sdf"
writer = Chem.SDWriter(str(out_best))
writer.write(best_mol)
writer.close()

out_traj = DEMO_DIR / "pso_swarm_trajectory.sdf"
writer_traj = Chem.SDWriter(str(out_traj))
for f in swarm_frames:
    writer_traj.write(f)
writer_traj.close()

print(f"[✓] Saved {len(swarm_frames)}-frame multi-particle swarm movie to {out_traj.name}")

# Generate PyMOL Script
pml_content = f"""# PyMOL Script for Kinematic Particle Swarm Optimization Movie
# Run directly in PyMOL: pymol visualize_pso_pymol.pml

reinitialize
load receptor.pdb, receptor
load pso_swarm_trajectory.sdf, pso_swarm
load pso_best_pose.sdf, best_docked_pose

hide everything, receptor
show cartoon, receptor
color slate, receptor
show surface, receptor
set transparency, 0.65, receptor

# Pocket residue highlight
select pocket, receptor within 6.0 of best_docked_pose
show sticks, pocket
color gray80, pocket
select cys481, (resn CYS and resi 481)
show sticks, cys481
color yellow, cys481

# Style Swarm Trajectory
hide everything, pso_swarm
show sticks, pso_swarm
color cyan, pso_swarm
set stick_radius, 0.18, pso_swarm

# Style Best Pose
show sticks, best_docked_pose
color green, best_docked_pose
set stick_radius, 0.28, best_docked_pose

zoom best_docked_pose, 7.0
set movie_fps, 30
mplay

print "================================================================="
print "  Loaded 400-frame Kinematic Particle Swarm (Kin-PSO) Movie!"
print "  Cyan: 20 Swarm Particles evolving through iterations."
print "  Green: Final Converged Global Best Pose."
print "  Press Play (bottom right) or Spacebar to watch swarm collapse."
print "================================================================="
"""
(DEMO_DIR / "visualize_pso_pymol.pml").write_text(pml_content)
print(f"[✓] PyMOL visualization script written: {DEMO_DIR / 'visualize_pso_pymol.pml'}")
