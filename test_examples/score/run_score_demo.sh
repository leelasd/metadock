#!/bin/bash
# Native openmm-dock scoring demonstration.

set -e
echo "================================================================================"
echo "                    OPENMM-DOCK: NATIVE SCORING DEMONSTRATION"
echo "================================================================================"

echo "[1] Scoring the input pose as-is..."
omm-dock score -r cavity.prm -i ii.sd -o openmm_score_out.sdf

echo ""
echo "[2] Scoring after a 5 A flexible-pocket minimization (score isn't itself"
echo "    flexible -- it evaluates a fixed pose as-is -- so flexibility is"
echo "    demonstrated via minimize, then the relaxed pose is re-scored)..."
omm-dock minimize -r cavity.prm -i ii.sd -o openmm_score_flex_out.sdf --flex-radius 5.0

echo ""
echo "[3] Genetic Algorithm local refinement around the input pose..."
omm-dock ga -r cavity.prm -i ii.sd -o openmm_ga_out.sdf -n 5

echo ""
echo "[✓] Scoring demo completed. Output poses saved to:"
echo "    - openmm_score_out.sdf"
echo "    - openmm_score_flex_out.sdf"
echo "    - openmm_ga_out.sdf"
