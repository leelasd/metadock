#!/usr/bin/env python
"""
Extracts receptor, ligand, and active-site waters from the raw 9Z1L mmCIF
(downloaded from RCSB) for use with openmm-dock.

PDB 9Z1L: KIT V654A mutant kinase domain + BLU-654 (ligand code A1CZZ), a
potent selective inhibitor for imatinib-resistant GIST (Moine et al.,
J. Med. Chem. 2026). Legacy PDB format is unavailable for this entry since
A1CZZ is a 5-character extended-CCD code; mmCIF is the only download format,
so this script does the receptor/ligand/water split manually via OpenMM's
PDBxFile + RDKit.

Ligand bond orders come from the authoritative CCD definition file
(A1CZZ_ccd.cif's _chem_comp_bond loop), matched to the crystal structure's
heavy atoms BY NAME (both come from the same wwPDB Chemical Component
Dictionary entry, so atom names -- C1, N1, C2, ... -- correspond exactly).
This is more robust than 3D-distance/graph-isomorphism bond-order guessing
against the separate "ideal" SDF, which produced an incorrect assignment
here (RDKit's AssignBondOrdersFromTemplate reported "more than one matching
pattern found" due to local symmetry and silently picked a wrong one,
giving nonsensical valences like a 1-bonded carbon).
"""
import re
from pathlib import Path
import numpy as np
from openmm.app import PDBxFile, PDBFile
from openmm import app, unit
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
CIF_PATH = ROOT / "9z1l.cif"
CCD_PATH = ROOT / "A1CZZ_ccd.cif"

WATER_CUTOFF_A = 5.0  # keep crystallographic waters within this distance of the ligand

_BOND_ORDER = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}


def parse_ccd_bonds(ccd_path: Path):
    """Parses the _chem_comp_bond loop: (atom1_name, atom2_name, bond_type, is_aromatic)."""
    text = ccd_path.read_text()
    block = re.search(r"_chem_comp_bond\.pdbx_ordinal\s*\n(.*?)\n#", text, re.DOTALL)
    if block is None:
        raise SystemExit(f"No _chem_comp_bond loop found in {ccd_path}")
    bonds = []
    for line in block.group(1).splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, a1, a2, order, aromatic = parts[:5]
        bonds.append((a1, a2, _BOND_ORDER[order], aromatic == "Y"))
    return bonds


def main():
    cif = PDBxFile(str(CIF_PATH))
    top = cif.getTopology()
    pos = cif.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    protein_atoms, ligand_atoms, water_residues = [], [], []
    for chain in top.chains():
        for res in chain.residues():
            if res.name == "A1CZZ":
                ligand_atoms.extend(res.atoms())
            elif res.name == "HOH":
                water_residues.append(res)
            elif res.name != "MRD":
                protein_atoms.extend(res.atoms())

    print(f"[*] Protein atoms: {len(protein_atoms)} | Ligand atoms: {len(ligand_atoms)} | Water residues: {len(water_residues)}")

    # 1. Write protein-only PDB (input to prepare_protein.py)
    protein_idx = {a.index for a in protein_atoms}
    modeller = app.Modeller(top, cif.getPositions())
    to_delete = [a for a in top.atoms() if a.index not in protein_idx]
    modeller.delete(to_delete)

    # The crystal structure never resolved the C-terminal carboxylate oxygen
    # (OXT) on the last residue -- without it, standard AMBER-style residue
    # templates treat that residue as non-terminal (expecting a peptide bond
    # to a next residue that doesn't exist) and createSystem() fails during
    # prepare_protein.py. Add it geometrically: reflect the existing O atom
    # across the CA-C axis, in-plane, at the same C-O bond length -- the
    # standard construction for a missing terminal carboxylate oxygen.
    from openmm import Vec3

    mod_top = modeller.getTopology()
    # Plain Vec3s in nm (no per-element Quantity wrapping) -- PDBFile.writeFile
    # expects either this form or a single Quantity wrapping the whole list,
    # not a list of individually-unit-wrapped Vec3s.
    mod_pos = list(modeller.getPositions().value_in_unit(unit.nanometer))
    last_res = list(mod_top.residues())[-1]
    atom_by_name = {a.name: a for a in last_res.atoms()}
    c_pos = np.array(mod_pos[atom_by_name["C"].index])
    ca_pos = np.array(mod_pos[atom_by_name["CA"].index])
    o_pos = np.array(mod_pos[atom_by_name["O"].index])
    v_o = o_pos - c_pos
    v_ca_hat = ca_pos - c_pos
    v_ca_hat = v_ca_hat / np.linalg.norm(v_ca_hat)
    reflected = 2 * np.dot(v_o, v_ca_hat) * v_ca_hat - v_o
    bond_len = np.linalg.norm(v_o)
    oxt_xyz = c_pos + (reflected / np.linalg.norm(reflected)) * bond_len  # still nm, same units as mod_pos

    oxt_atom = mod_top.addAtom("OXT", app.Element.getBySymbol("O"), last_res)
    mod_top.addBond(atom_by_name["C"], oxt_atom)
    mod_pos.append(Vec3(*oxt_xyz))
    print(f"[✓] Added missing C-terminal OXT atom to {last_res.name} {last_res.id}")

    with open(ROOT / "receptor.pdb", "w") as f:
        PDBFile.writeFile(mod_top, unit.Quantity(mod_pos, unit.nanometer), f)
    print(f"[✓] Wrote receptor.pdb ({mod_top.getNumAtoms()} protein atoms)")

    # 2. Ligand: build connectivity + bond orders from the CCD definition (by
    # atom name), then attach crystal 3D coordinates (also matched by name).
    lig_coords = np.array([[pos[a.index][0], pos[a.index][1], pos[a.index][2]] for a in ligand_atoms])
    lig_com = lig_coords.mean(axis=0)

    bonds = parse_ccd_bonds(CCD_PATH)
    name_to_elem = {a.name: (a.element.symbol if a.element is not None else a.name[0]) for a in ligand_atoms}
    name_to_coord = {a.name: pos[a.index] for a in ligand_atoms}

    rw = Chem.RWMol()
    name_to_idx = {}
    for name, elem in name_to_elem.items():
        idx = rw.AddAtom(Chem.Atom(elem))
        name_to_idx[name] = idx

    aromatic_atoms = set()
    for a1, a2, order, is_aromatic in bonds:
        if a1 not in name_to_idx or a2 not in name_to_idx:
            continue  # bond to a hydrogen (crystal structure has none resolved) -- skip, added back via AddHs
        rw.AddBond(name_to_idx[a1], name_to_idx[a2], order)
        if is_aromatic:
            aromatic_atoms.add(a1)
            aromatic_atoms.add(a2)
            rw.GetBondBetweenAtoms(name_to_idx[a1], name_to_idx[a2]).SetIsAromatic(True)
    for name in aromatic_atoms:
        rw.GetAtomWithIdx(name_to_idx[name]).SetIsAromatic(True)

    conf = Chem.Conformer(rw.GetNumAtoms())
    for name, idx in name_to_idx.items():
        c = name_to_coord[name]
        conf.SetAtomPosition(idx, (float(c[0]), float(c[1]), float(c[2])))
    rw.AddConformer(conf)

    lig_mol = rw.GetMol()
    Chem.SanitizeMol(lig_mol)

    # 3. Add explicit hydrogens with 3D geometry (crystal structure has none
    # resolved at this resolution), matching this repo's convention of
    # explicit-H ligand SDFs (e.g. q9e_crystal_pose.sdf).
    lig_mol = Chem.AddHs(lig_mol, addCoords=True)
    writer = Chem.SDWriter(str(ROOT / "a1czz_crystal_pose.sdf"))
    lig_mol.SetProp("_Name", "A1CZZ_9Z1L_crystal_pose")
    writer.write(lig_mol)
    writer.close()
    print(f"[✓] Wrote a1czz_crystal_pose.sdf ({lig_mol.GetNumAtoms()} atoms, {lig_mol.GetNumBonds()} bonds)")

    # 4. Active-site waters: crystallographic waters within WATER_CUTOFF_A of any ligand atom
    kept_waters = []
    for res in water_residues:
        for atom in res.atoms():
            if np.min(np.linalg.norm(lig_coords - pos[atom.index], axis=1)) < WATER_CUTOFF_A:
                kept_waters.append(res)
                break

    water_idx = {a.index for res in kept_waters for a in res.atoms()}
    modeller_w = app.Modeller(top, cif.getPositions())
    to_delete_w = [a for a in top.atoms() if a.index not in water_idx]
    modeller_w.delete(to_delete_w)
    with open(ROOT / "active_site_waters.pdb", "w") as f:
        PDBFile.writeFile(modeller_w.getTopology(), modeller_w.getPositions(), f)
    print(f"[✓] Wrote active_site_waters.pdb ({len(kept_waters)} waters within {WATER_CUTOFF_A} A of ligand)")

    print(f"[*] Ligand COM: {lig_com.round(2)} (pocket center for cavity.prm)")


if __name__ == "__main__":
    main()
