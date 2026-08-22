"""
Extracts the barnase (chain A, receptor) / barstar (chain D, mobile partner)
pair from the 1BRS crystal structure (a classic rigid-body protein-protein
docking benchmark complex) into separate PDB files, keeping only standard
ATOM records (protein heavy+H atoms if present, no waters/HETATM) so both
can be loaded independently via openmm_dock.core.PDBParser.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lines = (ROOT / "1brs.pdb").read_text().splitlines()

for chain_id, out_name in [("A", "barnase_receptor.pdb"), ("D", "barstar_ligand.pdb")]:
    kept = []
    for line in lines:
        if line.startswith("ATOM") and line[21] == chain_id:
            kept.append(line)
    kept.append("END")
    (ROOT / out_name).write_text("\n".join(kept) + "\n")
    print(f"[✓] Wrote {out_name}: {len(kept) - 1} atom records (chain {chain_id})")

# Native bound complex (both chains) for RMSD reference
native = []
for line in lines:
    if line.startswith("ATOM") and line[21] in ("A", "D"):
        native.append(line)
native.append("END")
(ROOT / "native_complex_AD.pdb").write_text("\n".join(native) + "\n")
print(f"[✓] Wrote native_complex_AD.pdb: {len(native) - 1} atom records")
