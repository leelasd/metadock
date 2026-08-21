#!/bin/bash
# Native openmm-dock explicit flexible-water docking demonstration.

set -e
echo "================================================================================"
echo "              OPENMM-DOCK: NATIVE FLEXIBLE ACTIVE-SITE WATER DEMO"
echo "================================================================================"

echo "[1] Local minimization with explicit flexible waters..."
omm-dock minimize -r cavity.prm -i lig.sdf -o openmm_solvent_min_out.sdf -w test_waters.pdb

echo ""
echo "[2] Simulated Annealing docking with explicit flexible waters..."
omm-dock dock -r cavity.prm -i lig.sdf -o openmm_solvent_dock_out.sdf -w test_waters.pdb -n 5

echo ""
echo "[3] Monte Carlo Basin-Hopping with explicit flexible waters..."
omm-dock mc -r cavity.prm -i lig.sdf -o openmm_mc_solvent_out.sdf -w test_waters.pdb -s 50

echo ""
echo "[✓] Solvent demo completed. Output poses saved to:"
echo "    - openmm_solvent_min_out.sdf"
echo "    - openmm_solvent_dock_out.sdf"
echo "    - openmm_mc_solvent_out.sdf"
