"""
Command-line interface for openmm_dock.
"""
from __future__ import annotations
import argparse
import sys
import re
from typing import Tuple, List, Optional
from pathlib import Path
import numpy as np
from rdkit import Chem

from .core import SDFParser, Mol2Parser
from .cavity import CavityDefinition
from .engine import DockingEngine
from .tether import find_tethered_atoms_mcs
from .protonation import protonate_ligand_ph
from .clustering import cluster_docked_poses


def parse_prm_receptor_and_cavity(prm_file: Path) -> Tuple[Path, CavityDefinition]:
    content = prm_file.read_text()
    rec_match = re.search(r"RECEPTOR_FILE\s+([^\s\n]+)", content, re.IGNORECASE)
    if not rec_match:
        raise ValueError(f"No RECEPTOR_FILE found in {prm_file}")
    rec_file = prm_file.parent / rec_match.group(1).strip()
    cavity = CavityDefinition.from_prm_file(prm_file)
    return rec_file, cavity


def main():
    parser = argparse.ArgumentParser(description="OpenMM Docking Suite (rDock-in-OpenMM)")
    parser.add_argument("--protonate", action="store_true", help="Automatically perceive and set physiological pH 7.4 ionization states for ligands")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: score
    score_p = subparsers.add_parser("score", help="Score ligand poses against a receptor without moving them")
    score_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    score_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    score_p.add_argument("-o", "--output", required=True, help="Output SDF file with scores")
    score_p.add_argument("--covalent-res", default=None, help="Target reactive amino acid residue for covalent docking (e.g. CYS145, CYS797, SER195)")
    score_p.add_argument("--protonate", action="store_true", help="Automatically perceive and set physiological pH 7.4 ionization states for ligands")

    # Command: minimize
    min_p = subparsers.add_parser("minimize", help="Locally minimize ligand poses in the receptor cavity")
    min_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    min_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    min_p.add_argument("-o", "--output", required=True, help="Output SDF file for minimized poses")
    min_p.add_argument("-w", "--waters", default=None, help="Optional PDB file with active-site waters")
    min_p.add_argument("--flex-radius", type=float, default=None, help="Radius (Å) around cavity center to treat receptor side chains as flexible")
    min_p.add_argument("--covalent-res", default=None, help="Target reactive amino acid residue for covalent docking (e.g. CYS145, CYS797, SER195)")
    min_p.add_argument("--protonate", action="store_true", help="Automatically perceive and set physiological pH 7.4 ionization states for ligands")

    # Command: dock
    dock_p = subparsers.add_parser("dock", help="Dock ligands using GPU Simulated Annealing MD")
    dock_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    dock_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    dock_p.add_argument("-o", "--output", required=True, help="Output SDF file for docked poses")
    dock_p.add_argument("-n", "--runs", type=int, default=10, help="Number of docking runs / poses (default: 10)")
    dock_p.add_argument("-p", "--pharma", default=None, help="Optional pharmacophore constraint file (pharma.restr)")
    dock_p.add_argument("-w", "--waters", default=None, help="Optional PDB file with active-site waters")
    dock_p.add_argument("--flex-radius", type=float, default=None, help="Radius (Å) around cavity center to treat receptor side chains as flexible")
    dock_p.add_argument("--covalent-res", default=None, help="Target reactive amino acid residue for covalent docking (e.g. CYS145, CYS797, SER195)")
    dock_p.add_argument("--protonate", action="store_true", help="Automatically perceive and set physiological pH 7.4 ionization states for ligands")

    # Command: tether
    teth_p = subparsers.add_parser("tether", help="Dock ligands with MCS template core restraints")
    teth_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    teth_p.add_argument("-ref", "--reference", required=True, help="Reference co-crystal ligand SDF")
    teth_p.add_argument("-i", "--input", required=True, help="Input query ligands SDF")
    teth_p.add_argument("-o", "--output", required=True, help="Output SDF file for tethered docked poses")
    teth_p.add_argument("--protonate", action="store_true", help="Automatically perceive and set physiological pH 7.4 ionization states for ligands")

    # Command: mc (Monte Carlo Basin-Hopping)
    mc_p = subparsers.add_parser("mc", help="Dock ligands using Metropolis Monte Carlo with Basin-Hopping Minimization")
    mc_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    mc_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    mc_p.add_argument("-o", "--output", required=True, help="Output SDF file for docked poses")
    mc_p.add_argument("-s", "--steps", type=int, default=100, help="Number of Monte Carlo steps (default: 100)")
    mc_p.add_argument("-t", "--temperature", type=float, default=300.0, help="Monte Carlo simulation temperature in K (default: 300)")
    mc_p.add_argument("-traj", "--trajectory", default=None, help="Optional output SDF path to save the complete multi-frame Monte Carlo trajectory")
    mc_p.add_argument("-p", "--pharma", default=None, help="Optional pharmacophore constraint file (pharma.restr)")
    mc_p.add_argument("-w", "--waters", default=None, help="Optional PDB file with active-site waters")
    mc_p.add_argument("--flex-radius", type=float, default=None, help="Radius (Å) around cavity center to treat receptor side chains as flexible")
    mc_p.add_argument("--covalent-res", default=None, help="Target reactive amino acid residue for covalent docking (e.g. CYS145, CYS797, SER195)")
    mc_p.add_argument("--protonate", action="store_true", help="Automatically perceive and set physiological pH 7.4 ionization states for ligands")

    # Command: ga (Genetic Algorithm -- rDock's own default search engine)
    ga_p = subparsers.add_parser("ga", help="Refine ligand poses with a Genetic Algorithm local search around the input pose (rigid-body + torsional DOFs)")
    ga_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    ga_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file (used as the starting pose to refine around)")
    ga_p.add_argument("-o", "--output", required=True, help="Output SDF file for docked poses")
    ga_p.add_argument("-n", "--runs", type=int, default=5, help="Number of independent GA runs / output poses (default: 5)")
    ga_p.add_argument("--pop-size", type=int, default=20, help="GA population size per run (default: 20)")
    ga_p.add_argument("--generations", type=int, default=15, help="Number of GA generations per run (default: 15)")
    ga_p.add_argument("--mutation-rate", type=float, default=0.2, help="Per-gene mutation probability (default: 0.2)")
    ga_p.add_argument("-p", "--pharma", default=None, help="Optional pharmacophore constraint file (pharma.restr)")
    ga_p.add_argument("-w", "--waters", default=None, help="Optional PDB file with active-site waters")
    ga_p.add_argument("--flex-radius", type=float, default=None, help="Radius (Å) around cavity center to treat receptor side chains as flexible")
    ga_p.add_argument("--covalent-res", default=None, help="Target reactive amino acid residue for covalent docking (e.g. CYS145, CYS797, SER195)")
    ga_p.add_argument("--protonate", action="store_true", help="Automatically perceive and set physiological pH 7.4 ionization states for ligands")

    # Command: stats
    stats_p = subparsers.add_parser("stats", help="Compute docking statistics (heavy-atom RMSD, valence bond & angle deviations) vs crystal reference")
    stats_p.add_argument("-ref", "--reference", required=True, help="Reference co-crystal ligand SDF / SD file")
    stats_p.add_argument("-i", "--input", required=True, help="Docked poses SDF / SD file to evaluate")

    # Command: cluster
    cluster_p = subparsers.add_parser("cluster", help="Cluster docked poses using heavy-atom RMSD (Butina) to remove redundant poses")
    cluster_p.add_argument("-i", "--input", required=True, help="Input docked poses SDF / SD file")
    cluster_p.add_argument("-o", "--output", required=True, help="Output filtered SDF file with unique cluster leaders")
    cluster_p.add_argument("--cutoff", type=float, default=1.5, help="Heavy-atom RMSD clustering cutoff in Å (default: 1.5)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "stats":
        ref_path = Path(args.reference)
        in_path = Path(args.input)
        ref_mols = SDFParser.load_molecules(ref_path)
        if not ref_mols:
            sys.exit(f"Error: could not load reference ligand {ref_path}")
        ref_mol = ref_mols[0]
        test_mols = SDFParser.load_molecules(in_path)
        if not test_mols:
            sys.exit(f"Error: could not load test poses from {in_path}")

        conf_r = ref_mol.GetConformer()
        heavy_r = [a.GetIdx() for a in ref_mol.GetAtoms() if a.GetAtomicNum() > 1]

        print(f"[*] Comparing {len(test_mols)} docked pose(s) from {in_path} against crystal reference {ref_path}")
        print(f"{'Pose':<8} | {'Score (kcal/mol)':<18} | {'Heavy RMSD (Å)':<16} | {'Max Bond Dev (Å)':<18} | {'Max Angle Dev (°)':<18} | {'Status'}")
        print("-" * 105)

        from rdkit.Chem import rdMolTransforms
        for idx, t_mol in enumerate(test_mols):
            conf_t = t_mol.GetConformer()
            heavy_t = [a.GetIdx() for a in t_mol.GetAtoms() if a.GetAtomicNum() > 1]
            score_str = t_mol.GetProp("SCORE") if t_mol.HasProp("SCORE") else "N/A"

            if len(heavy_r) == len(heavy_t):
                p_r = np.array([conf_r.GetAtomPosition(i) for i in heavy_r])
                p_t = np.array([conf_t.GetAtomPosition(i) for i in heavy_t])
                rmsd = float(np.sqrt(np.mean(np.sum((p_t - p_r) ** 2, axis=1))))
                rmsd_str = f"{rmsd:.3f}"
                status = "SUCCESS (RMSD < 2.0 Å)" if rmsd < 2.0 else "POOR"
            else:
                rmsd_str = "N/A"
                status = "Scaffold Mismatch"

            bond_diffs = []
            for b in t_mol.GetBonds():
                a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                d_r = np.linalg.norm(np.array(conf_r.GetAtomPosition(a1)) - np.array(conf_r.GetAtomPosition(a2)))
                d_t = np.linalg.norm(np.array(conf_t.GetAtomPosition(a1)) - np.array(conf_t.GetAtomPosition(a2)))
                bond_diffs.append(abs(d_t - d_r))
            max_b = max(bond_diffs) if bond_diffs else 0.0

            angle_diffs = []
            for atom in t_mol.GetAtoms():
                c = atom.GetIdx()
                nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        a1, a3 = nbrs[i], nbrs[j]
                        th_r = rdMolTransforms.GetAngleDeg(conf_r, a1, c, a3)
                        th_t = rdMolTransforms.GetAngleDeg(conf_t, a1, c, a3)
                        angle_diffs.append(abs(th_t - th_r))
            max_a = max(angle_diffs) if angle_diffs else 0.0

            print(f"#{idx+1:<7} | {score_str:<18} | {rmsd_str:<16} | {max_b:<18.4f} | {max_a:<18.2f} | {status}")
        return

    if args.command == "cluster":
        in_path = Path(args.input)
        out_path = Path(args.output)
        if not out_path.name.endswith(".sdf") and not out_path.name.endswith(".sd"):
            out_path = out_path.with_suffix(".sdf")
        mols = SDFParser.load_molecules(in_path)
        if not mols:
            sys.exit(f"Error: could not load poses from {in_path}")
        print(f"[*] Loaded {len(mols)} pose(s) from {in_path}. Clustering at RMSD cutoff = {args.cutoff:.2f} Å...")
        clustered = cluster_docked_poses(mols, rmsd_cutoff=args.cutoff)
        writer = Chem.SDWriter(str(out_path))
        for m in clustered:
            writer.write(m)
        writer.close()
        print(f"[✓] Extracted {len(clustered)} distinct binding cluster representative(s) (written to {out_path})")
        return

    prm_path = Path(args.prm)
    rec_path, cavity = parse_prm_receptor_and_cavity(prm_path)
    print(f"[*] Loaded receptor: {rec_path} | Cavity center: {cavity.center.round(2)} | Radius: {cavity.radius:.1f} Å")

    input_path = Path(args.input)
    ligands = SDFParser.load_molecules(input_path)
    print(f"[*] Loaded {len(ligands)} input ligand molecule(s) from {input_path}")

    if getattr(args, "protonate", False):
        print("[*] Perceiving and setting physiological pH 7.4 ionization states for ligands...")
        ligands = [protonate_ligand_ph(lig) for lig in ligands]

    out_path = Path(args.output)
    if not out_path.name.endswith(".sdf") and not out_path.name.endswith(".sd"):
        out_path = out_path.with_suffix(".sdf")

    writer = Chem.SDWriter(str(out_path))

    if args.command == "score":
        engine = DockingEngine(receptor_path=rec_path, cavity=cavity, covalent_res=getattr(args, "covalent_res", None))
        for i, lig in enumerate(ligands):
            scores = engine.score(lig)
            for k, v in scores.items():
                lig.SetProp(k, f"{v:.4f}")
            writer.write(lig)
            print(f"  Pose #{i+1}: Total Score = {scores['SCORE']:.3f} | Inter = {scores['SCORE.INTER']:.3f} | Cavity = {scores['SCORE.RESTR.CAVITY']:.3f}")

    elif args.command == "minimize":
        engine = DockingEngine(
            receptor_path=rec_path,
            cavity=cavity,
            waters_pdb_path=getattr(args, "waters", None),
            flexible_radius=getattr(args, "flex_radius", None),
            covalent_res=getattr(args, "covalent_res", None),
        )
        for i, lig in enumerate(ligands):
            res = engine.minimize(lig)
            writer.write(res.mol)
            print(f"  Pose #{i+1} Minimized: Initial -> Final Score = {res.score:.3f}")

    elif args.command == "dock":
        pharma = getattr(args, "pharma", None)
        engine = DockingEngine(
            receptor_path=rec_path,
            cavity=cavity,
            pharma_restr_path=pharma,
            waters_pdb_path=getattr(args, "waters", None),
            flexible_radius=getattr(args, "flex_radius", None),
            covalent_res=getattr(args, "covalent_res", None),
        )
        for lig_idx, lig in enumerate(ligands):
            print(f"[*] Docking ligand #{lig_idx+1} ({args.runs} runs)...")
            results = engine.dock_simulated_annealing(lig, n_runs=args.runs)
            for rank, r in enumerate(results):
                r.mol.SetProp("DOCK_RANK", str(rank + 1))
                writer.write(r.mol)
                print(f"  Rank #{rank+1}: Score = {r.score:.3f} (VDW: {r.scores['SCORE.INTER.VDW']:.2f}, Polar: {r.scores['SCORE.INTER.POLAR']:.2f})")

    elif args.command == "tether":
        ref_mols = SDFParser.load_molecules(args.reference)
        if not ref_mols:
            sys.exit(f"Error: could not load reference ligand {args.reference}")
        ref_mol = ref_mols[0]
        engine = DockingEngine(receptor_path=rec_path, cavity=cavity)

        for lig_idx, lig in enumerate(ligands):
            aligned_mol, constraints = find_tethered_atoms_mcs(lig, ref_mol)
            if aligned_mol is not None and constraints:
                print(f"[*] Ligand #{lig_idx+1}: Found MCS with {len(constraints)} tethered core atoms. Docking with restraints...")
                results = engine.dock_simulated_annealing(aligned_mol, tether_constraints=constraints, n_runs=args.runs)
                for rank, r in enumerate(results):
                    r.mol.SetProp("DOCK_RANK", str(rank + 1))
                    writer.write(r.mol)
                    print(f"  Tethered Rank #{rank+1}: Score = {r.score:.3f}")
            else:
                print(f"[*] Ligand #{lig_idx+1}: No significant MCS found. Minimizing freely...")
                res = engine.minimize(lig)
                writer.write(res.mol)

    elif args.command == "mc":
        pharma = getattr(args, "pharma", None)
        engine = DockingEngine(
            receptor_path=rec_path,
            cavity=cavity,
            pharma_restr_path=pharma,
            waters_pdb_path=getattr(args, "waters", None),
            flexible_radius=getattr(args, "flex_radius", None),
            covalent_res=getattr(args, "covalent_res", None),
        )
        for lig_idx, lig in enumerate(ligands):
            print(f"[*] Monte Carlo Basin-Hopping docking on ligand #{lig_idx+1} ({args.steps} steps @ {args.temperature}K)...")
            res = engine.dock_monte_carlo(lig, n_steps=args.steps, temperature_k=args.temperature)
            writer.write(res.mol)
            print(f"  Best Pose: Score = {res.score:.3f} (VDW: {res.scores['SCORE.INTER.VDW']:.2f}, Polar: {res.scores['SCORE.INTER.POLAR']:.2f})")

            if getattr(args, "trajectory", None) and res.trajectory:
                traj_path = Path(args.trajectory)
                traj_writer = Chem.SDWriter(str(traj_path))
                for frame in res.trajectory:
                    traj_writer.write(frame)
                traj_writer.close()
                print(f"[✓] Complete {len(res.trajectory)}-frame Monte Carlo trajectory written to {traj_path}")

    elif args.command == "ga":
        pharma = getattr(args, "pharma", None)
        engine = DockingEngine(
            receptor_path=rec_path,
            cavity=cavity,
            pharma_restr_path=pharma,
            waters_pdb_path=getattr(args, "waters", None),
            flexible_radius=getattr(args, "flex_radius", None),
            covalent_res=getattr(args, "covalent_res", None),
        )
        for lig_idx, lig in enumerate(ligands):
            print(
                f"[*] Genetic Algorithm docking on ligand #{lig_idx+1} "
                f"({args.runs} runs x {args.pop_size} individuals x {args.generations} generations)..."
            )
            results = engine.dock_genetic_algorithm(
                lig,
                population_size=args.pop_size,
                n_generations=args.generations,
                mutation_rate=args.mutation_rate,
                n_runs=args.runs,
            )
            for rank, r in enumerate(results):
                r.mol.SetProp("DOCK_RANK", str(rank + 1))
                writer.write(r.mol)
                print(f"  Rank #{rank+1}: Score = {r.score:.3f} (VDW: {r.scores['SCORE.INTER.VDW']:.2f}, Polar: {r.scores['SCORE.INTER.POLAR']:.2f})")

    writer.close()
    print(f"[✓] Results written to {out_path}")


if __name__ == "__main__":
    main()
