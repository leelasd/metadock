import parmed
from openmm import app, unit
from openmm.app import *
receptor_pdbfile = PDBFile('output.pdb')
omm_forcefield = app.ForceField('amber10.xml')

# Parameterize the protein.
receptor_system = omm_forcefield.createSystem(receptor_pdbfile.topology)

# Convert the protein System into a ParmEd Structure.
receptor_structure = parmed.openmm.load_topology(receptor_pdbfile.topology,
                                                 receptor_system,
                                                 xyz=receptor_pdbfile.positions)
receptor_structure.save('receptor.prmtop', overwrite=True)
receptor_structure.save('receptor.inpcrd', overwrite=True)
receptor_structure.save('receptor.mol2', overwrite=True)
receptor_structure.save('receptor.gro', overwrite=True)
