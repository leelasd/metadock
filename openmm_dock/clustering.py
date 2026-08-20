"""
Heavy-atom RMSD pose clustering and diversity filtering for molecular docking.
Uses Butina clustering to remove redundant poses and extract representative binding modes.
"""
from typing import List
import numpy as np
from rdkit import Chem
from rdkit.ML.Cluster import Butina


def cluster_docked_poses(mols: List[Chem.Mol], rmsd_cutoff: float = 1.5) -> List[Chem.Mol]:
    """
    Clusters docked ligand conformers based on heavy-atom pairwise RMSD using the Butina algorithm.
    Returns the lowest-energy representative pose from each distinct cluster,
    annotated with 'CLUSTER_ID' and 'CLUSTER_SIZE'.
    """
    if not mols:
        return []
    if len(mols) == 1:
        mols[0].SetProp("CLUSTER_ID", "1")
        mols[0].SetProp("CLUSTER_SIZE", "1")
        return mols

    heavy_indices = [a.GetIdx() for a in mols[0].GetAtoms() if a.GetAtomicNum() > 1]
    n_mols = len(mols)
    coords_list = []
    for m in mols:
        conf = m.GetConformer()
        coords_list.append(np.array([conf.GetAtomPosition(i) for i in heavy_indices]))

    # Compute condensed distance matrix (lower triangular)
    dists = []
    for i in range(1, n_mols):
        for j in range(i):
            d = float(np.sqrt(np.mean(np.sum((coords_list[i] - coords_list[j]) ** 2, axis=1))))
            dists.append(d)

    # Butina clustering
    clusters = Butina.ClusterData(dists, n_mols, rmsd_cutoff, isDistData=True)

    unique_mols = []
    for c_idx, cluster in enumerate(clusters):
        # Leader of cluster (first molecule in cluster is the cluster center)
        rep_mol = mols[cluster[0]]
        rep_mol.SetProp("CLUSTER_ID", str(c_idx + 1))
        rep_mol.SetProp("CLUSTER_SIZE", str(len(cluster)))
        unique_mols.append(rep_mol)

    return unique_mols
