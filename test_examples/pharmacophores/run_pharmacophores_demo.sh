#!/bin/bash
# Native openmm-dock pharmacophore-restraint docking demonstration.

set -e
echo "================================================================================"
echo "              OPENMM-DOCK: NATIVE PHARMACOPHORE RESTRAINT DEMO"
echo "================================================================================"

echo "[1] Simulated Annealing docking with pharmacophore restraints..."
omm-dock dock -r cavity.prm -i xtal-lig.sd -o openmm_pharma_dock_out.sdf -p pharma.restr -n 10

echo ""
echo "[2] Monte Carlo Basin-Hopping WITH pharmacophore restraints..."
omm-dock mc -r cavity.prm -i xtal-lig.sd -o openmm_mc_pharma_out.sdf -p pharma.restr -s 50

echo ""
echo "[3] Monte Carlo Basin-Hopping WITHOUT restraints (comparison baseline)..."
omm-dock mc -r cavity.prm -i xtal-lig.sd -o openmm_mc_unconstrained_out.sdf -s 50

echo ""
echo "[✓] Pharmacophore demo completed. Output poses saved to:"
echo "    - openmm_pharma_dock_out.sdf"
echo "    - openmm_mc_pharma_out.sdf"
echo "    - openmm_mc_unconstrained_out.sdf"
