"""
Parameterizes the 9Z1L receptor (extracted from the raw mmCIF by
extract_from_pdb.py) with OpenMM/ParmEd to produce receptor.mol2, following
the same pattern as tethered/, solvent/, and pharmacophores/'s
prepare_protein.py -- except this one explicitly adds hydrogens first
(Modeller.addHydrogens), since receptor.pdb comes straight from the raw
crystal structure and has none resolved (unlike those other examples'
output.pdb, which already had hydrogens added upstream before being
checked in).
"""
import parmed
from openmm import app, unit
from openmm.app import *

receptor_pdbfile = PDBFile('receptor.pdb')
omm_forcefield = app.ForceField('amber10.xml')

modeller = app.Modeller(receptor_pdbfile.topology, receptor_pdbfile.positions)
modeller.addHydrogens(omm_forcefield, pH=7.0)

receptor_system = omm_forcefield.createSystem(modeller.topology)

receptor_structure = parmed.openmm.load_topology(
    modeller.topology, receptor_system, xyz=modeller.positions
)
receptor_structure.save('receptor.mol2', overwrite=True)
