"""
Computes ligand (barstar, chain D)-only RMSD-to-native for a LightDock-
generated pose PDB. LightDock's setup recenters the whole system by
subtracting the receptor's own centroid (verified empirically: the
processed receptor's coordinates equal the original minus its centroid,
atom-for-atom) -- so the same offset is added back here before comparing
directly (same atom count/order for chain D since no atoms were removed
from the ligand side) against the untouched native crystal coordinates.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_chain_coords(pdb_path: Path, chain_id: str) -> np.ndarray:
    coords = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == chain_id:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


if __name__ == "__main__":
    native_D = read_chain_coords(ROOT / "native_complex_AD.pdb", "D")
    receptor_centroid = read_chain_coords(ROOT / "barnase_receptor.pdb", "A").mean(axis=0)

    for pdb_path in sys.argv[1:]:
        gen_D = read_chain_coords(Path(pdb_path), "D") + receptor_centroid
        if gen_D.shape != native_D.shape:
            print(f"{pdb_path}: SHAPE MISMATCH {gen_D.shape} vs native {native_D.shape}")
            continue
        print(f"{pdb_path}: RMSD = {rmsd(gen_D, native_D):.2f} Å")
