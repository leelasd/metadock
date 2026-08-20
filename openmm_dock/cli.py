"""
Command-line interface for openmm_dock.
"""
from __future__ import annotations
import argparse
import sys
import re
from typing import Tuple, List, Optional
from pathlib import Path
from rdkit import Chem

from .core import SDFParser, Mol2Parser
from .cavity import CavityDefinition
from .engine import DockingEngine
from .tether import find_tethered_atoms_mcs


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
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: score
    score_p = subparsers.add_parser("score", help="Score ligand poses against a receptor without moving them")
    score_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    score_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    score_p.add_argument("-o", "--output", required=True, help="Output SDF file with scores")

    # Command: minimize
    min_p = subparsers.add_parser("minimize", help="Locally minimize ligand poses in the receptor cavity")
    min_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    min_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    min_p.add_argument("-o", "--output", required=True, help="Output SDF file for minimized poses")
    min_p.add_argument("-w", "--waters", default=None, help="Optional PDB file with active-site waters")

    # Command: dock
    dock_p = subparsers.add_parser("dock", help="Dock ligands using GPU Simulated Annealing MD")
    dock_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    dock_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    dock_p.add_argument("-o", "--output", required=True, help="Output SDF file for docked poses")
    dock_p.add_argument("-n", "--runs", type=int, default=10, help="Number of docking runs / poses (default: 10)")
    dock_p.add_argument("-p", "--pharma", default=None, help="Optional pharmacophore constraint file (pharma.restr)")
    dock_p.add_argument("-w", "--waters", default=None, help="Optional PDB file with active-site waters")

    # Command: tether
    teth_p = subparsers.add_parser("tether", help="Dock ligands with MCS template core restraints")
    teth_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    teth_p.add_argument("-ref", "--reference", required=True, help="Reference co-crystal ligand SDF")
    teth_p.add_argument("-i", "--input", required=True, help="Input query ligands SDF")
    teth_p.add_argument("-o", "--output", required=True, help="Output SDF file for tethered docked poses")
    # Command: mc (Monte Carlo Basin-Hopping)
    mc_p = subparsers.add_parser("mc", help="Dock ligands using Metropolis Monte Carlo with Basin-Hopping Minimization")
    mc_p.add_argument("-r", "--prm", required=True, help="Path to cavity.prm parameter file")
    mc_p.add_argument("-i", "--input", required=True, help="Input SDF / SD ligand file")
    mc_p.add_argument("-o", "--output", required=True, help="Output SDF file for docked poses")
    mc_p.add_argument("-s", "--steps", type=int, default=100, help="Number of Monte Carlo steps (default: 100)")
    mc_p.add_argument("-t", "--temperature", type=float, default=300.0, help="Monte Carlo simulation temperature in K (default: 300)")
    mc_p.add_argument("-p", "--pharma", default=None, help="Optional pharmacophore constraint file (pharma.restr)")
    mc_p.add_argument("-w", "--waters", default=None, help="Optional PDB file with active-site waters")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    prm_path = Path(args.prm)
    rec_path, cavity = parse_prm_receptor_and_cavity(prm_path)
    print(f"[*] Loaded receptor: {rec_path} | Cavity center: {cavity.center.round(2)} | Radius: {cavity.radius:.1f} Å")

    input_path = Path(args.input)
    ligands = SDFParser.load_molecules(input_path)
    print(f"[*] Loaded {len(ligands)} input ligand molecule(s) from {input_path}")

    out_path = Path(args.output)
    if not out_path.name.endswith(".sdf") and not out_path.name.endswith(".sd"):
        out_path = out_path.with_suffix(".sdf")

    writer = Chem.SDWriter(str(out_path))

    if args.command == "score":
        engine = DockingEngine(receptor_path=rec_path, cavity=cavity)
        for i, lig in enumerate(ligands):
            scores = engine.score(lig)
            for k, v in scores.items():
                lig.SetProp(k, f"{v:.4f}")
            writer.write(lig)
            print(f"  Pose #{i+1}: Total Score = {scores['SCORE']:.3f} | Inter = {scores['SCORE.INTER']:.3f} | Cavity = {scores['SCORE.RESTR.CAVITY']:.3f}")

    elif args.command == "minimize":
        engine = DockingEngine(receptor_path=rec_path, cavity=cavity, waters_pdb_path=getattr(args, "waters", None))
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
        )
        for lig_idx, lig in enumerate(ligands):
            print(f"[*] Monte Carlo Basin-Hopping docking on ligand #{lig_idx+1} ({args.steps} steps @ {args.temperature}K)...")
            res = engine.dock_monte_carlo(lig, n_steps=args.steps, temperature_k=args.temperature)
            writer.write(res.mol)
            print(f"  Best Pose: Score = {res.score:.3f} (VDW: {res.scores['SCORE.INTER.VDW']:.2f}, Polar: {res.scores['SCORE.INTER.POLAR']:.2f})")

    writer.close()
    print(f"[✓] Results written to {out_path}")


if __name__ == "__main__":
    main()
