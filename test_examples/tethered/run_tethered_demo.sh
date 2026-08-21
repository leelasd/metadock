#!/bin/bash
# Native openmm-dock tethered docking demonstration (MCS-based template restraints).
# Replaces the external rDock/rxDock + Docker workflow previously documented
# here: openmm-dock's own `tether` CLI command already finds the MCS between
# each query ligand and the reference pose (find_tethered_atoms_mcs) and
# docks with GPU simulated annealing under those tether restraints natively
# -- no rDock binary or Docker image required.

set -e
echo "================================================================================"
echo "          OPENMM-DOCK: NATIVE TETHERED (MCS-RESTRAINED) DOCKING DEMO"
echo "================================================================================"

echo "[1] Running native tethered docking (MCS core restraints vs. xtal-lig.sd)..."
omm-dock tether -r cavity.prm \
                -ref xtal-lig.sd \
                -i query_ligands.sdf \
                -o openmm_tethered_dock_out.sdf \
                -n 5

echo ""
echo "[2] Validating heavy-atom RMSD vs. the reference pose..."
omm-dock stats -ref xtal-lig.sd -i openmm_tethered_dock_out.sdf

echo ""
echo "[✓] Tethered docking completed. Output poses saved to:"
echo "    - openmm_tethered_dock_out.sdf"
