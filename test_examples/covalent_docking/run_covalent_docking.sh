#!/bin/bash
# Script demonstrating OpenMM Covalent Docking targeting Cysteine-33

set -e
echo "================================================================================"
echo "          OPENMM-DOCK: AUTOMATED COVALENT DOCKING DEMONSTRATION"
echo "================================================================================"

# 1. Covalent L-BFGS Minimization
echo "[1] Running Covalent L-BFGS Minimization targeting CYS33..."
omm-dock minimize -r cavity.prm \
                  -i covalent_ligand.sdf \
                  -o openmm_covalent_min_out.sdf \
                  --covalent-res CYS33

# 2. Covalent Monte Carlo Basin-Hopping with 3D Trajectory Export
echo ""
echo "[2] Running Covalent Monte Carlo Basin-Hopping (50 steps) + Trajectory Export..."
omm-dock mc -r cavity.prm \
            -i covalent_ligand.sdf \
            -o openmm_covalent_mc_best.sdf \
            -traj openmm_covalent_trajectory.sdf \
            --covalent-res CYS33 \
            -s 50 -t 300.0

echo ""
echo "[✓] Covalent docking completed successfully! Output poses saved to:"
echo "    - openmm_covalent_min_out.sdf"
echo "    - openmm_covalent_mc_best.sdf"
echo "    - openmm_covalent_trajectory.sdf (3D conformational movie)"
