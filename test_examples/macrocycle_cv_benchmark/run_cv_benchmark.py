#!/usr/bin/env python
"""
Macrocycle Kinematic Metadynamics Collective Variable (CV) Benchmark.

Compares enhanced conformational sampling across:
1. Unbiased Kinematic Exploration (Baseline)
2. Radius of Gyration (Rg) Metadynamics
3. Shape Anisotropy / PMI (kappa^2) Metadynamics
4. Ring Puckering Amplitude (Q_puck) Metadynamics
5. Coupled Dual-CV (Rg + kappa^2) Metadynamics

Evaluates:
- Discovery rate of the crystallographic bioactive envelope (< 1.0 Å RMSD)
- Conformational phase space coverage (PMI triangular plot)
- Barrier crossing efficiency & ring breathing dynamics
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Geometry import Point3D

from macrocycle_cv_engine import MacrocycleKinematicSampler, MacrocycleCVCalculator


def plot_pmi_triangle(results_dict: dict, crystal_cv, out_png: Path):
    """
    Renders the Triangular Principal Moments of Inertia (PMI) Shape Plot (NPR1 vs NPR2).
    Bounding vertices: Rod (0, 1), Disc (0.5, 0.5), Sphere (1, 1).
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Draw PMI Bounding Triangle
    # Line 1: (0, 1) to (0.5, 0.5) -> NPR2 = 1 - NPR1
    x_rod_disc = np.linspace(0.0, 0.5, 100)
    y_rod_disc = 1.0 - x_rod_disc
    ax.plot(x_rod_disc, y_rod_disc, "k-", lw=1.5)
    
    # Line 2: (0.5, 0.5) to (1.0, 1.0) -> NPR2 = NPR1
    x_disc_sphere = np.linspace(0.5, 1.0, 100)
    y_disc_sphere = x_disc_sphere
    ax.plot(x_disc_sphere, y_disc_sphere, "k-", lw=1.5)
    
    # Line 3: (0.0, 1.0) to (1.0, 1.0) -> NPR2 = 1.0
    x_rod_sphere = np.linspace(0.0, 1.0, 100)
    y_rod_sphere = np.ones_like(x_rod_sphere)
    ax.plot(x_rod_sphere, y_rod_sphere, "k-", lw=1.5)
    
    # Fill Triangle Background
    ax.fill_between([0.0, 0.5, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0], color="#F8F9FA", alpha=0.8)
    
    # Annotate Vertices
    ax.text(0.0, 1.02, "Rod / Linear\n(NPR1=0, NPR2=1)", fontsize=11, fontweight="bold", ha="center", va="bottom", color="#8B0000")
    ax.text(0.5, 0.46, "Disc / Oblate\n(NPR1=0.5, NPR2=0.5)", fontsize=11, fontweight="bold", ha="center", va="top", color="#00008B")
    ax.text(1.0, 1.02, "Sphere / Globular\n(NPR1=1, NPR2=1)", fontsize=11, fontweight="bold", ha="center", va="bottom", color="#006400")
    
    colors = {
        "unbiased": "#7F7F7F",
        "rg": "#E64B35",
        "kappa_sq": "#4DBBD5",
        "qpuck": "#00A087",
        "coupled_rg_kappa": "#8491B4"
    }
    labels = {
        "unbiased": "Unbiased Baseline",
        "rg": "Rg Metadynamics",
        "kappa_sq": "PMI / kappa^2 Metadynamics",
        "qpuck": "Ring Pucker Metadynamics",
        "coupled_rg_kappa": "Coupled (Rg + kappa^2) MetaD"
    }
    
    for mode, res in results_dict.items():
        npr1s = [row["npr1"] for row in res["exp_log"]]
        npr2s = [row["npr2"] for row in res["exp_log"]]
        step_skip = max(1, len(npr1s) // 250)
        ax.scatter(
            npr1s[::step_skip],
            npr2s[::step_skip],
            c=colors[mode],
            alpha=0.55,
            s=22,
            edgecolor="black",
            linewidth=0.3,
            label=labels[mode]
        )
        
    # Plot Crystal Pose Marker
    ax.plot(
        crystal_cv.npr1,
        crystal_cv.npr2,
        marker="*",
        color="yellow",
        markersize=22,
        markeredgecolor="black",
        markeredgewidth=1.5,
        label=f"Crystal Bioactive Pose ({crystal_cv.npr1:.2f}, {crystal_cv.npr2:.2f})"
    )
    
    ax.set_title("Triangular PMI Macrocycle Shape Space Exploration (NPR1 vs NPR2)", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Normalized Principal Ratio 1: $I_1 / I_3$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Principal Ratio 2: $I_2 / I_3$", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.42, 1.10)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved Triangular PMI Plot to {out_png}")


def plot_comparative_metrics(results_dict: dict, crystal_cv, out_png: Path):
    """
    Renders 4-panel comparative benchmark metrics:
    A. RMSD Evolution to Crystal Pose
    B. Radius of Gyration Distribution
    C. Relative Shape Anisotropy (kappa^2) Distribution
    D. Ring Puckering Amplitude Distribution
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    colors = {
        "unbiased": "#7F7F7F",
        "rg": "#E64B35",
        "kappa_sq": "#4DBBD5",
        "qpuck": "#00A087",
        "coupled_rg_kappa": "#3C5488"
    }
    labels = {
        "unbiased": "Unbiased Baseline",
        "rg": "Rg Metadynamics",
        "kappa_sq": "PMI (kappa^2) MetaD",
        "qpuck": "Ring Pucker MetaD",
        "coupled_rg_kappa": "Coupled (Rg + kappa^2)"
    }
    
    # --- Panel A: Macrocyclic Ring Backbone RMSD Evolution (< 1.0 Å Milestone) ---
    for mode, res in results_dict.items():
        iters = [row["iteration"] for row in res["exp_log"]]
        ring_rmsds = [row["rmsd_ring_xtal"] for row in res["exp_log"]]
        
        # Calculate running minimum Ring RMSD per iteration
        unique_iters = sorted(list(set(iters)))
        running_min = []
        cur_min = float("inf")
        for it in unique_iters:
            it_rmsds = [ring_rmsds[i] for i in range(len(iters)) if iters[i] == it]
            cur_min = min(cur_min, min(it_rmsds))
            running_min.append(cur_min)
            
        ax1.plot(unique_iters, running_min, lw=2.2, color=colors[mode], label=f"{labels[mode]} (Min Ring: {res['min_ring_rmsd_A']:.2f} Å)")
        
    ax1.axhline(1.0, color="green", linestyle="--", lw=2.0, label=r"Sub-1.0 Å Bioactive Ring Threshold")
    ax1.set_title(r"A. Macrocycle Backbone Ring Convergence (< 1.0 Å Bioactive Envelope)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Metadynamics Iteration", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Ring Backbone Core RMSD to Crystal Pose (Å)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    
    # --- Panel B: Radius of Gyration Distribution ---
    for mode, res in results_dict.items():
        rgs = [row["r_g_total"] for row in res["exp_log"]]
        ax2.hist(rgs, bins=35, density=True, alpha=0.35, color=colors[mode], label=labels[mode])
        
    ax2.axvline(crystal_cv.r_g_total, color="gold", linestyle="-", lw=2.5, label=f"Crystal Rg ({crystal_cv.r_g_total:.2f} Å)")
    ax2.set_title("B. Radius of Gyration ($R_g$) Exploration", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Radius of Gyration $R_g$ (Å)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    # --- Panel C: Shape Anisotropy (kappa^2) Distribution ---
    for mode, res in results_dict.items():
        kappas = [row["kappa_sq"] for row in res["exp_log"]]
        ax3.hist(kappas, bins=35, density=True, alpha=0.35, color=colors[mode], label=labels[mode])
        
    ax3.axvline(crystal_cv.kappa_sq, color="gold", linestyle="-", lw=2.5, label=f"Crystal $\\kappa^2$ ({crystal_cv.kappa_sq:.3f})")
    ax3.set_title(r"C. Relative Shape Anisotropy ($\kappa^2$) Exploration", fontsize=12, fontweight="bold")
    ax3.set_xlabel(r"Relative Shape Anisotropy $\kappa^2$ (0=Sphere, 1=Rod)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.3)
    
    # --- Panel D: Ring Puckering Amplitude Distribution ---
    for mode, res in results_dict.items():
        qpucks = [row["q_puck"] for row in res["exp_log"]]
        ax4.hist(qpucks, bins=35, density=True, alpha=0.35, color=colors[mode], label=labels[mode])
        
    ax4.axvline(crystal_cv.q_puck, color="gold", linestyle="-", lw=2.5, label=f"Crystal $Q_{{puck}}$ ({crystal_cv.q_puck:.2f} Å)")
    ax4.set_title(r"D. Macrocyclic Ring Puckering Amplitude ($Q_{puck}$)", fontsize=12, fontweight="bold")
    ax4.set_xlabel(r"Ring Puckering Amplitude $Q_{puck}$ (Å)", fontsize=11, fontweight="bold")

    ax4.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved Comparative Metrics Plot to {out_png}")


def main():
    root = Path(__file__).parent
    xtal_path = root / "q9e_crystal_pose.sdf"
    
    xtal_mol = Chem.SDMolSupplier(str(xtal_path), removeHs=False)[0]
    calc = MacrocycleCVCalculator(xtal_mol)
    conf_x = xtal_mol.GetConformer()
    coords_x = np.array([conf_x.GetAtomPosition(i) for i in range(xtal_mol.GetNumAtoms())])
    crystal_cv = calc.compute_all_cvs(coords_x)
    
    print("=" * 80)
    print(" MACROCYCLE KINEMATIC METADYNAMICS COLLECTIVE VARIABLE (CV) BENCHMARK")
    print("=" * 80)
    print(f"[*] Target Macrocycle     : Q9E (16-Membered Ring, 64 Atoms)")
    print(f"[*] Crystal Properties    : Rg = {crystal_cv.r_g_total:.3f} Å | kappa^2 = {crystal_cv.kappa_sq:.3f} | Q_puck = {crystal_cv.q_puck:.3f} Å")
    print(f"[*] Normalized PMIs       : (NPR1 = {crystal_cv.npr1:.3f}, NPR2 = {crystal_cv.npr2:.3f}) [Oblate/Globular Region]")
    print("=" * 80)
    
    sampler = MacrocycleKinematicSampler(ligand_mol=xtal_mol, reference_xtal_mol=xtal_mol)
    
    # Run 5 Experimental CV Modes
    modes = ["unbiased", "rg", "kappa_sq", "qpuck", "coupled_rg_kappa"]
    results = {}
    
    for mode in modes:
        res = sampler.run_metadynamics_cv_experiment(
            cv_mode=mode,
            n_iterations=80,
            n_particles=16,
            w0_height=8.0,
            gamma_well_temper=4.0
        )
        results[mode] = res
        
        # Save individual best conformer SDF with full property tags
        if res["min_rmsd_coords"] is not None:
            m_best = Chem.Mol(xtal_mol)
            c_b = m_best.GetConformer()
            for i in range(xtal_mol.GetNumAtoms()):
                c_b.SetAtomPosition(i, Point3D(float(res["min_rmsd_coords"][i][0]), float(res["min_rmsd_coords"][i][1]), float(res["min_rmsd_coords"][i][2])))
            
            m_best.SetProp("_Name", f"Q9E_{mode.upper()}_Conformer")
            m_best.SetProp("CV_MODE", mode)
            m_best.SetProp("TOTAL_64_ATOM_RMSD_A", f"{res['min_rmsd_to_crystal_A']:.3f}")
            m_best.SetProp("RING_CORE_RMSD_A", f"{res['min_ring_rmsd_A']:.3f}")
            m_best.SetProp("SUB_1A_RING_ITERATION", str(res["sub_1_ring_iter"]))
            
            cv_metrics = calc.compute_all_cvs(res["min_rmsd_coords"])
            m_best.SetProp("RG_TOTAL_A", f"{cv_metrics.r_g_total:.3f}")
            m_best.SetProp("RG_RING_A", f"{cv_metrics.r_g_ring:.3f}")
            m_best.SetProp("KAPPA_SQ", f"{cv_metrics.kappa_sq:.4f}")
            m_best.SetProp("NPR1", f"{cv_metrics.npr1:.3f}")
            m_best.SetProp("NPR2", f"{cv_metrics.npr2:.3f}")
            m_best.SetProp("Q_PUCK_A", f"{cv_metrics.q_puck:.3f}")
            
            w_sdf = Chem.SDWriter(str(root / f"best_conformer_{mode}.sdf"))
            w_sdf.write(m_best)
            w_sdf.close()

    # Save Consolidated Multi-Conformer Ensemble SDF
    ensemble_path = root / "all_modes_comparison_ensemble.sdf"
    w_ens = Chem.SDWriter(str(ensemble_path))
    
    # 1. Write Reference Crystal Pose
    xtal_export = Chem.Mol(xtal_mol)
    xtal_export.SetProp("_Name", "Q9E_XRAY_CRYSTAL_REFERENCE")
    xtal_export.SetProp("CV_MODE", "CRYSTAL_EXPERIMENT")
    xtal_export.SetProp("TOTAL_64_ATOM_RMSD_A", "0.000")
    xtal_export.SetProp("RING_CORE_RMSD_A", "0.000")
    xtal_export.SetProp("RG_TOTAL_A", f"{crystal_cv.r_g_total:.3f}")
    xtal_export.SetProp("KAPPA_SQ", f"{crystal_cv.kappa_sq:.4f}")
    w_ens.write(xtal_export)
    
    # 2. Write each best mode conformer
    for mode in modes:
        res = results[mode]
        if res["min_rmsd_coords"] is not None:
            m_conf = Chem.Mol(xtal_mol)
            c_b = m_conf.GetConformer()
            for i in range(xtal_mol.GetNumAtoms()):
                c_b.SetAtomPosition(i, Point3D(float(res["min_rmsd_coords"][i][0]), float(res["min_rmsd_coords"][i][1]), float(res["min_rmsd_coords"][i][2])))
            m_conf.SetProp("_Name", f"Q9E_{mode.upper()}_BestPose")
            m_conf.SetProp("CV_MODE", mode)
            m_conf.SetProp("TOTAL_64_ATOM_RMSD_A", f"{res['min_rmsd_to_crystal_A']:.3f}")
            m_conf.SetProp("RING_CORE_RMSD_A", f"{res['min_ring_rmsd_A']:.3f}")
            m_conf.SetProp("SUB_1A_RING_ITERATION", str(res["sub_1_ring_iter"]))
            cv_m = calc.compute_all_cvs(res["min_rmsd_coords"])
            m_conf.SetProp("RG_TOTAL_A", f"{cv_m.r_g_total:.3f}")
            m_conf.SetProp("KAPPA_SQ", f"{cv_m.kappa_sq:.4f}")
            w_ens.write(m_conf)
    w_ens.close()
    print(f"[✓] Saved Multi-Conformer Ensemble SDF to: {ensemble_path}")

    # Generate Comparison Plots
    out_pmi = root / "pmi_triangular_comparison.png"
    out_metrics = root / "cv_benchmark_comparison.png"

    
    plot_pmi_triangle(results, crystal_cv, out_pmi)
    plot_comparative_metrics(results, crystal_cv, out_metrics)
    
    # Generate PyMOL Visualizer
    out_pml = root / "visualize_cv_benchmark_pymol.pml"
    with open(out_pml, "w") as f:
        f.write(f"""# PyMOL Script: Macrocycle CV Benchmark Comparison
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2

load q9e_crystal_pose.sdf, crystal_reference
show sticks, crystal_reference
color forest, crystal_reference
set stick_radius, 0.35, crystal_reference

load best_conformer_unbiased.sdf, conf_unbiased
show sticks, conf_unbiased
color gray60, conf_unbiased
set stick_radius, 0.22, conf_unbiased

load best_conformer_rg.sdf, conf_rg
show sticks, conf_rg
color tv_red, conf_rg
set stick_radius, 0.22, conf_rg

load best_conformer_kappa_sq.sdf, conf_pmi_kappa
show sticks, conf_pmi_kappa
color cyan, conf_pmi_kappa
set stick_radius, 0.22, conf_pmi_kappa

load best_conformer_qpuck.sdf, conf_qpuck
show sticks, conf_qpuck
color mediumseagreen, conf_qpuck
set stick_radius, 0.22, conf_qpuck

load best_conformer_coupled_rg_kappa.sdf, conf_coupled_rg_kappa
show sticks, conf_coupled_rg_kappa
color gold, conf_coupled_rg_kappa
set stick_radius, 0.28, conf_coupled_rg_kappa

zoom crystal_reference, 4.0
""")

    print("\n" + "=" * 95)
    print(" BENCHMARK COMPLETE: QUANTITATIVE SUMMARY (RANDOMIZED ETKDGv3 START)")
    print("=" * 95)
    print(f"{'CV Mode':<20} | {'64-Atom RMSD':<14} | {'Ring Core RMSD':<16} | {'Sub-1.0Å Ring Iter':<20} | {'Rg Span (Å)'}")
    print("-" * 95)
    for mode, res in results.items():
        sub_iter_str = f"Iter #{res['sub_1_ring_iter']}" if res['sub_1_ring_iter'] is not None else "None"
        print(f"{mode:<20} | {res['min_rmsd_to_crystal_A']:<14.2f} | {res['min_ring_rmsd_A']:<16.3f} | {sub_iter_str:<20} | {res['rg_span_A']:.2f}")
    print("=" * 95)
    print(f"[*] Triangular PMI Shape Plot      : {out_pmi}")
    print(f"[*] Comparative 4-Panel Metrics    : {out_metrics}")
    print(f"[*] PyMOL Visualizer Script        : {out_pml}")
    print("=" * 95)



if __name__ == "__main__":
    main()
