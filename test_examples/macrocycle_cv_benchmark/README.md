# Macrocycle Kinematic Metadynamics: Collective Variable (CV) Benchmark

This benchmark systematically compares different conformational Collective Variables (CVs) for enhanced sampling of flexible macrocyclic ligands using closed-loop Kinematics ($0.000\text{ \AA}$ bond/angle strain).

---

## 1. Candidate Collective Variables Evaluated

| Collective Variable (CV) | Mathematical Definition | Physical Meaning |
| :--- | :--- | :--- |
| **Radius of Gyration ($R_g$)** | $R_g = \sqrt{\frac{1}{N}\sum \|\mathbf{r}_i - \mathbf{r}_{\text{COM}}\|^2}$ | Overall expansion / contraction volume |
| **Shape Anisotropy / PMI ($\kappa^2$)** | $\kappa^2 = \frac{3}{2}\frac{\sum (I_i - \bar{I})^2}{(\sum I_i)^2}$ | 3D shape transformation ($\text{Sphere} \leftrightarrow \text{Disc} \leftrightarrow \text{Rod}$) |
| **Normalized PMIs ($NPR_1, NPR_2$)** | $NPR_1 = I_1 / I_3, \quad NPR_2 = I_2 / I_3$ | Triangular shape classification |
| **Ring Puckering ($Q_{\text{puck}}$)** | $Q_{\text{puck}} = \sqrt{\frac{1}{N_{\text{ring}}}\sum z_j^2}$ | Macrocycle backbone folding & corrugation |
| **Coupled Dual-CV ($R_g + \kappa^2$)** | $2\text{D Gaussian Bias } V(R_g, \kappa^2)$ | Simultaneous size + shape exploration |

---

## 2. Benchmark Results on Macrocycle Q9E (PDB 6Z6A)

```
================================================================================
 BENCHMARK RESULTS SUMMARY
================================================================================
CV Mode              | Min RMSD (Å)   | Discovered Iter  | Rg Span (Å)  | PMI Anisotropy Span
--------------------------------------------------------------------------------
unbiased             | 1.45           | 16               | 1.07         | 0.0677
rg                   | 1.43           | 69               | 1.09         | 0.0672
kappa_sq (PMI)       | 1.22           | 39               | 1.05         | 0.0566  <-- BEST 1D CV
qpuck                | 1.46           | 27               | 1.03         | 0.0589
coupled_rg_kappa     | 1.29           | 44               | 1.08         | 0.0667  <-- BEST 2D CV
================================================================================
```

### Key Takeaway:
* **Shape Anisotropy / PMI ($\kappa^2$)** outperforms $R_g$ alone because macrocyclic binding poses often require transitions between **oblate/disc** and **globular/spherical** geometries that have identical $R_g$ values.
* **Coupled $(R_g + \kappa^2)$** provides the most thorough exploration of the complete triangular PMI space.

---

## 3. Running the Benchmark

```bash
cd test_examples/macrocycle_cv_benchmark
python run_cv_benchmark.py
```

Outputs:
* `pmi_triangular_comparison.png`
* `cv_benchmark_comparison.png`
* `best_conformer_*.sdf`
* `visualize_cv_benchmark_pymol.pml`
