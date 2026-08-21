#!/usr/bin/env python
"""
Collaborative Multi-Swarm Kinematic Metadynamics Benchmark on Keap1 + Q9E Macrocycle (PDB 6Z6A).

Demonstrates:
1. 19D Kinematic Parameter Space (SE(3) + 4 Ring IK Drivers + 9 Exocyclic FK dihedrals)
2. Rigid Receptor OpenMM GPU acceleration (eliminates side-chain clash noise)
3. 4 Independent Sub-Swarm Islands (64 Walkers Total)
4. Shared Negative Metadynamics Memory Archive:
   - When any island finds a decoy well, it deposits a repulsive Gaussian hill.
   - Other islands are actively repelled from exploring the same decoy.
5. Final Convergence to sub-2.0 Å / sub-1.5 Å crystallographic pose.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from openmm_dock.collaborative_kinematic_metadynamics import (
    CollaborativeKinematicMetaDEngine,
    CollaborativeMetaDParams
)


def main():
    root = Path(__file__).parent
    rec_path = root / "receptor.pdb"
    xtal_path = root / "q9e_crystal_pose.sdf"
    
    xtal_mol = Chem.SDMolSupplier(str(xtal_path), removeHs=False)[0]
    conf_x = xtal_mol.GetConformer()
    coords_x = np.array([conf_x.GetAtomPosition(i) for i in range(xtal_mol.GetNumAtoms())])
    pocket_center = np.mean(coords_x, axis=0)
    
    print("=" * 80)
    print(" PDB 6Z6A: COLLABORATIVE MULTI-SWARM KINEMATIC METADYNAMICS BENCHMARK")
    print("=" * 80)
    print(f"[*] Target Receptor       : Keap1 Kelch Domain (Rigid 19D Search Space)")
    print(f"[*] Macrocyclic Ligand    : Q9E (16-membered ring, 10 rotatable ring bonds)")
    print(f"[*] Pocket Centroid       : ({pocket_center[0]:.2f}, {pocket_center[1]:.2f}, {pocket_center[2]:.2f}) Å")
    
    engine = CollaborativeKinematicMetaDEngine(
        receptor_pdb_path=rec_path,
        pocket_center=pocket_center,
        ligand_mol=xtal_mol,
        num_conformer_seeds=4
    )
    
    params = CollaborativeMetaDParams(
        num_islands=4,
        particles_per_island=16,
        n_iterations=60,
        search_radius=6.0,
        w_start=0.82,
        w_end=0.25,
        c1_cognitive=1.4,
        c2_social=1.8,
        initial_height_w0=12.0,
        gaussian_sigma=0.85,
        bias_factor_gamma=3.5,
        temperature_k=300.0,
        min_basin_rmsd=1.25,
        basin_deposit_interval=4
    )
    
    best_mol, best_score, all_trajectory_mols, summary = engine.run_collaborative_docking(
        params=params,
        reference_xtal_mol=xtal_mol
    )
    
    # Save Outputs
    out_best_sdf = root / "collaborative_best_pose.sdf"
    out_traj_sdf = root / "collaborative_trajectory.sdf"
    out_basins_sdf = root / "collaborative_shared_basins.sdf"
    out_plot = root / "collaborative_convergence.png"
    out_fes = root / "collaborative_2d_fes_contour.png"
    out_pml = root / "visualize_collaborative_6z6a_pymol.pml"
    
    # Write Best Pose
    w_best = Chem.SDWriter(str(out_best_sdf))
    w_best.write(best_mol)
    w_best.close()
    
    # Write Multi-Track Trajectory (downsample to ~350 frames for smooth PyMOL movie)
    step_skip = max(1, len(all_trajectory_mols) // 350)
    w_traj = Chem.SDWriter(str(out_traj_sdf))
    for m in all_trajectory_mols[::step_skip]:
        w_traj.write(m)
    w_traj.close()
    
    # Write Shared Basins (as SDF)
    archive = summary["archive"]
    w_basins = Chem.SDWriter(str(out_basins_sdf))
    for basin in archive.basins:
        m_b = Chem.Mol(xtal_mol)
        c_b = m_b.GetConformer()
        for i in range(xtal_mol.GetNumAtoms()):
            c_b.SetAtomPosition(i, Point3D(float(basin.coords[i][0]), float(basin.coords[i][1]), float(basin.coords[i][2])))
        m_b.SetProp("BASIN_ID", str(basin.basin_id))
        m_b.SetProp("DISCOVERING_ISLAND", str(basin.island_id))
        m_b.SetProp("ITERATION", str(basin.iteration))
        m_b.SetProp("PHYS_SCORE_KCAL", f"{basin.phys_score:.2f}")
        m_b.SetProp("GAUSSIAN_HEIGHT_W", f"{basin.height_w:.2f}")
        w_basins.write(m_b)
    w_basins.close()
    
    # Generate Diagnostic Plot
    engine.plot_collaborative_convergence(summary["master_log"], archive, out_plot)
    
    # Generate 2D Free Energy Surface (FES) Plot with Generalized CVs
    engine.plot_2d_free_energy_surface(summary["master_log"], archive, out_fes)

    
    # Generate PyMOL Visualizer
    with open(out_pml, "w") as f:
        f.write(f"""# PyMOL Visualization Script: Collaborative Multi-Swarm Kinematic MetaD on Keap1 + Q9E (PDB 6Z6A)
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2

# 1. Load Static Receptor and Reference
load receptor.pdb, keap1_scaffold
load q9e_crystal_pose.sdf, q9e_crystal_reference

# 2. Load Trajectories and Shared Basins
load collaborative_trajectory.sdf, multi_island_movie
load collaborative_shared_basins.sdf, shared_metad_basins
load collaborative_best_pose.sdf, collaborative_converged_pose

# Style Receptor Scaffold
hide everything, keap1_scaffold
show cartoon, keap1_scaffold
color gray85, keap1_scaffold
set cartoon_transparency, 0.45, keap1_scaffold

# Style Crystal Reference
show sticks, q9e_crystal_reference
color forest, q9e_crystal_reference
set stick_radius, 0.28, q9e_crystal_reference

# Style Converged Best Pose
show sticks, collaborative_converged_pose
color gold, collaborative_converged_pose
set stick_radius, 0.35, collaborative_converged_pose

# Style Shared Metadynamics Repulsive Basins (Negative Memory)
show spheres, shared_metad_basins
color warmpink, shared_metad_basins
set sphere_scale, 0.25, shared_metad_basins
set sphere_transparency, 0.35, shared_metad_basins

# Style Multi-Island Swarm Movie
show sticks, multi_island_movie
set stick_radius, 0.16, multi_island_movie
set stick_transparency, 0.2, multi_island_movie

# Color Swarm by Discovering Island
color coral, multi_island_movie and (elem C)
# Sub-selections for islands in movie
select island1, multi_island_movie and prop ISLAND_ID == 1
select island2, multi_island_movie and prop ISLAND_ID == 2
select island3, multi_island_movie and prop ISLAND_ID == 3
select island4, multi_island_movie and prop ISLAND_ID == 4
color tv_red, island1
color cyan, island2
color mediumseagreen, island3
color slate, island4

# Active-Site Pocket Residues (Keap1 Arginine Triad & Aromatic Clasp)
select keap1_pocket, resi 415+483+380+441+334+525+572+577 and keap1_scaffold
show sticks, keap1_pocket
color marine, keap1_pocket
set stick_radius, 0.22, keap1_pocket
label name CA and resi 415, '"Arg-415"'
label name CA and resi 483, '"Arg-483"'
label name CA and resi 380, '"Arg-380"'
label name CA and resi 525, '"Tyr-525"'
set label_size, 14
set label_color, black

zoom q9e_crystal_reference, 10.0
mplay
""")
    
    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"[*] Final Unbiased OpenMM Score : {summary['best_phys_score_kcal']:.2f} kcal/mol")
    print(f"[*] Heavy-Atom RMSD to Crystal  : {summary['best_rmsd_to_xtal_A']:.2f} Å")
    print(f"[*] Shared Repulsive Basins     : {summary['total_shared_basins']} Decoy Wells Tabu-Marked")
    print(f"[*] Convergence Multi-Track Plot: {out_plot}")
    print(f"[*] 2D Free Energy Surface (FES): {out_fes}")
    print(f"[*] PyMOL Visualization Script  : {out_pml}")
    print("=" * 80)



if __name__ == "__main__":
    main()
