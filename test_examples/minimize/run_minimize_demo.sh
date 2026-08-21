#!/bin/bash
# Native openmm-dock minimization demonstration.

set -e
echo "================================================================================"
echo "                  OPENMM-DOCK: NATIVE MINIMIZATION DEMONSTRATION"
echo "================================================================================"

echo "[1] Running local L-BFGS minimization..."
omm-dock minimize -r cavity.prm -i ii.sd -o openmm_min_out.sdf

echo ""
echo "[✓] Minimization completed. Output saved to openmm_min_out.sdf"
