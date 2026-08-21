#!/bin/bash
# Native openmm-dock RNA docking demonstration on PDB 1NEM (RNA aptamer).
# RNA receptors need no special handling: openmm-dock reads the receptor
# and cavity exactly as for a protein target -- no rDock/rxDock binary or
# Docker image is required.

set -e
echo "================================================================================"
echo "          OPENMM-DOCK: NATIVE RNA DOCKING DEMO (PDB 1NEM APTAMER)"
echo "================================================================================"

echo "[1] Scoring the reference pose as-is..."
omm-dock score -r 1nem_rdock.prm \
               -i 1nem_lig.sd \
               -o openmm_rna_score_out.sdf

echo ""
echo "[2] Local L-BFGS minimization..."
omm-dock minimize -r 1nem_rdock.prm \
                  -i 1nem_lig.sd \
                  -o openmm_rna_min_out.sdf

echo ""
echo "[3] Monte Carlo Basin-Hopping docking (100 steps @ 300K, 3 A flexible pocket)..."
# --flex-radius 3.0 mirrors the original rDock RECEPTOR_FLEX 3.0 setting in
# 1nem_rdock.prm -- that field is rDock-specific and not read automatically
# by CavityDefinition.from_prm_file, so it's passed explicitly here.
omm-dock mc -r 1nem_rdock.prm \
            -i 1nem_lig.sd \
            -o openmm_rna_dock_out.sdf \
            -s 100 -t 300.0 \
            --flex-radius 3.0

echo ""
echo "[4] Validating heavy-atom RMSD vs. the reference pose..."
omm-dock stats -ref 1nem_lig.sd -i openmm_rna_min_out.sdf
omm-dock stats -ref 1nem_lig.sd -i openmm_rna_dock_out.sdf

echo ""
echo "[✓] RNA docking completed. Output poses saved to:"
echo "    - openmm_rna_score_out.sdf"
echo "    - openmm_rna_min_out.sdf"
echo "    - openmm_rna_dock_out.sdf"
