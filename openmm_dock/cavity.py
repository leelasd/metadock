"""
Cavity definition and OpenMM cavity restraint potentials.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import openmm as mm
from openmm import unit
from .core import MolecularSystem, SDFParser, Mol2Parser


@dataclass
class CavityDefinition:
    center: np.ndarray  # (3,) in Angstroms
    radius: float       # in Angstroms
    min_coords: np.ndarray  # (3,) in Angstroms
    max_coords: np.ndarray  # (3,) in Angstroms
    name: str = "Cavity"

    @classmethod
    def from_reference_ligand(
        cls,
        ref_ligand_path: Path | str,
        radius: float = 6.0,
        base_dir: Optional[Path] = None,
    ) -> CavityDefinition:
        ref_path = Path(ref_ligand_path)
        if not ref_path.is_absolute() and base_dir is not None:
            ref_path = base_dir / ref_path

        if ref_path.suffix in [".sd", ".sdf", ".mol"]:
            mols = SDFParser.load_molecules(ref_path)
            if not mols:
                raise ValueError(f"Could not load reference ligand from {ref_path}")
            sys = SDFParser.mol_to_system(mols[0])
        elif ref_path.suffix in [".mol2"]:
            sys = Mol2Parser.parse(ref_path)
        else:
            raise ValueError(f"Unsupported reference ligand format: {ref_path.suffix}")

        coords = sys.coordinates
        center = np.mean(coords, axis=0)
        # Bounding box + radius
        min_c = np.min(coords, axis=0) - radius
        max_c = np.max(coords, axis=0) + radius
        # Compute effective enclosing radius
        dists = np.linalg.norm(coords - center, axis=1)
        effective_radius = float(np.max(dists) + radius)

        return cls(
            center=center,
            radius=effective_radius,
            min_coords=min_c,
            max_coords=max_c,
            name=ref_path.stem,
        )

    @classmethod
    def from_prm_file(cls, prm_path: Path | str) -> CavityDefinition:
        prm_path = Path(prm_path)
        base_dir = prm_path.parent
        content = prm_path.read_text()

        # Check for REF_MOL
        ref_mol_match = re.search(r"REF_MOL\s+([^\s\n]+)", content, re.IGNORECASE)
        radius_match = re.search(r"RADIUS\s+([0-9.]+)", content, re.IGNORECASE)
        center_match = re.search(
            r"CENTER\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)",
            content,
            re.IGNORECASE,
        )

        radius = float(radius_match.group(1)) if radius_match else 6.0

        if ref_mol_match:
            ref_file = ref_mol_match.group(1).strip()
            return cls.from_reference_ligand(ref_file, radius=radius, base_dir=base_dir)
        elif center_match:
            cx = float(center_match.group(1))
            cy = float(center_match.group(2))
            cz = float(center_match.group(3))
            center = np.array([cx, cy, cz], dtype=np.float64)
            min_c = center - radius
            max_c = center + radius
            return cls(
                center=center,
                radius=radius,
                min_coords=min_c,
                max_coords=max_c,
                name="SphereCavity",
            )
        else:
            raise ValueError(f"Could not find REF_MOL or CENTER in {prm_path}")


def create_cavity_restraint_force(
    cavity: CavityDefinition,
    ligand_particle_indices: List[int],
    k_cavity: float = 1000.0,  # kJ/(mol * nm^2)
) -> mm.CustomExternalForce:
    """
    Creates a flat-bottom harmonic restraint force in OpenMM that confines ligand
    atoms within the cavity sphere.
    """
    # Convert Å to nm
    x0_nm = cavity.center[0] * 0.1
    y0_nm = cavity.center[1] * 0.1
    z0_nm = cavity.center[2] * 0.1
    r_cav_nm = cavity.radius * 0.1

    # Flat-bottom harmonic potential: 0 inside R_cav, k*(r - R_cav)^2 outside
    energy_expr = (
        "0.5 * k_cav * step(r_dist - r_cav) * (r_dist - r_cav)^2;"
        "r_dist = sqrt((x - x0_cav)^2 + (y - y0_cav)^2 + (z - z0_cav)^2)"
    )

    force = mm.CustomExternalForce(energy_expr)
    force.addGlobalParameter("k_cav", k_cavity)
    force.addGlobalParameter("r_cav", r_cav_nm)
    force.addGlobalParameter("x0_cav", x0_nm)
    force.addGlobalParameter("y0_cav", y0_nm)
    force.addGlobalParameter("z0_cav", z0_nm)
    force.setName("CavityRestraintForce")

    for idx in ligand_particle_indices:
        force.addParticle(idx, [])

    return force
