# OpenMM Docking Suite: Technical Deep-Dive & Architecture Specification

This document provides a comprehensive technical breakdown of how **rDock** molecular docking was ported into **OpenMM**, why molecular docking cannot simply be executed naively via OpenMM's Python API without specialized force and constraint architecture, and how each physical and computational challenge was systematically solved.

---

## Table of Contents

**Part I — The Scoring Engine (Foundations)**
1. [Executive Summary](#1-executive-summary)
2. [MD vs. Docking: Foundational Differences](#2-md-vs-docking-foundational-differences)
3. [The 5 Core Challenges & Technical Solutions](#3-the-5-core-challenges--technical-solutions)
   - [Challenge 1: Degrees of Freedom ($3N$ Cartesian vs. $6+M$ Torsion Space)](#challenge-1-degrees-of-freedom-3n-cartesian-vs-6m-torsion-space)
   - [Challenge 2: The "Python Loop" Latency Trap](#challenge-2-the-python-loop-latency-trap)
   - [Challenge 3: Scoring Functions vs. Physical Force Fields](#challenge-3-scoring-functions-vs-physical-force-fields)
   - [Challenge 4: Ligand Structural Integrity & Bond Stretching](#challenge-4-ligand-structural-integrity--bond-stretching)
   - [Challenge 5: Aromatic Ring Buckling & Out-of-Plane Puckering](#challenge-5-aromatic-ring-buckling--out-of-plane-puckering)
4. [Scoring Function Mathematical Formulation in OpenMM](#4-scoring-function-mathematical-formulation-in-openmm)
5. [Feature-by-Feature Implementation](#5-feature-by-feature-implementation)
   - [Cavity Definition & Restraints](#cavity-definition--restraints)
   - [Active-Site Solvent & Water Tethering](#active-site-solvent--water-tethering)
   - [Pharmacophore Restraint Constraints](#pharmacophore-restraint-constraints)
   - [Template-Constrained Tethered Docking (MCS)](#template-constrained-tethered-docking-mcs)
   - [RNA Aptamer Docking](#rna-aptamer-docking)
6. [Benchmark Verification & Metrics](#6-benchmark-verification--metrics)

**Part II — Kinematics: Parameterizing Shape**
7. [Kinematics: From $3N$ Cartesian to Internal Coordinates](#7-kinematics-from-3n-cartesian-to-internal-coordinates)
   - [7.1 Forward Kinematics — `kinematics.py`](#71-forward-kinematics--kinematicspy)
   - [7.2 Inverse Kinematics for Macrocycles — `inverse_kinematics.py`](#72-inverse-kinematics-for-macrocycles--inverse_kinematicspy)
   - [7.3 Shared Kinematic Utilities — `kinematic_utils.py`](#73-shared-kinematic-utilities--kinematic_utilspy)
9. [Receptor Flexibility — `receptor_kinematics.py`](#9-receptor-flexibility--receptor_kinematicspy)

**Part III — Search Algorithms**
8. [Search & Optimization Algorithms — `engine.py`](#8-search--optimization-algorithms--enginepy)
   - [8.1 Simulated Annealing](#81-simulated-annealing-dock_simulated_annealing)
   - [8.2 Genetic Algorithm](#82-genetic-algorithm-dock_genetic_algorithm)
   - [8.3 Monte Carlo / Basin Hopping](#83-monte-carlo--basin-hopping-dock_monte_carlo)
   - [8.4 Monte-Carlo-with-Minimization (Vina/smina-style)](#84-monte-carlo-with-minimization-vinasmina-style-dock_monte_carlo_minimization)

**Part IV — Swarm, Gradient, and Bayesian Sampling**
10. [Swarm Intelligence & Enhanced Sampling](#10-swarm-intelligence--enhanced-sampling)
    - [10.1 Particle Swarm Optimization — `unified_kinematic_pso.py`](#101-particle-swarm-optimization--unified_kinematic_psopy)
    - [10.2 Blind Global Docking — `global_blind_docking.py`](#102-blind-global-docking--global_blind_dockingpy)
    - [10.3 Metadynamics as an Enhanced-Sampling & Pose-Strength Tool](#103-metadynamics-as-an-enhanced-sampling--pose-strength-tool)
    - [10.4 Generalized, Reference-Free Collective Variables — `generalized_cv.py`](#104-generalized-reference-free-collective-variables--generalized_cvpy)
11. [Gradient-Based and Bayesian Optimization](#11-gradient-based-and-bayesian-optimization)

**Part V — Protein-Protein Docking**
12. [Protein-Protein Docking: Glowworm Swarm Optimization](#12-protein-protein-docking-glowworm-swarm-optimization)

**Part VI — Supporting Infrastructure**
13. [Supporting Infrastructure](#13-supporting-infrastructure)
    - [13.1 Covalent Docking — `covalent.py`](#131-covalent-docking--covalentpy)
    - [13.2 Precomputed Potential Grids — `gridding.py`](#132-precomputed-potential-grids--griddingpy)
    - [13.3 Pose Clustering — `clustering.py`](#133-pose-clustering--clusteringpy)
    - [13.4 Protonation State Assignment — `protonation.py`](#134-protonation-state-assignment--protonationpy)
    - [13.5 Bridged Two-Stage Docking — `bridged_docking.py`](#135-bridged-two-stage-docking--bridged_dockingpy)
    - [13.6 Command-Line Interface — `cli.py`](#136-command-line-interface--clipy)

---

## 1. Executive Summary

Molecular docking programs (such as **rDock**, **AutoDock Vina**, and **Glide**) predict the non-covalent binding pose of a small molecule inside a receptor's binding site. Historically, molecular docking was not offered as a standard out-of-the-box routine in OpenMM because OpenMM was built as an ultra-fast, GPU-accelerated engine for continuous Newtonian / Langevin molecular dynamics trajectories.

By translating rDock's empirical scoring terms into OpenMM `CustomNonbondedForce` and `CustomExternalForce`, pairing them with **cross-ring structural triangulation**, **harmonic valence forces**, and **GPU-accelerated Simulated Annealing MD (SAMD)**, we achieve:
* Seamless, single-context docking without host-device synchronization lag.
* Strictly preserved ligand bond lengths ($\Delta r < 0.05\text{ \AA}$) and planar aromatic systems ($\text{deviation} < 0.03\text{ \AA}$).
* Native support for proteins, RNA aptamers, flexible explicit waters, pharmacophore constraints, and MCS template tethering.

---

## 2. MD vs. Docking: Foundational Differences

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│       Molecular Dynamics (OpenMM)            │         Molecular Docking (rDock/Vina)       │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Coordinate Space: 3N Cartesian Coordinates   │ Coordinate Space: Internal Torsion Space     │
│ (x, y, z for all N atoms: ~60-100 DOFs)      │ (6 rigid-body DOFs + M rotatable bonds)      │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Sampling: Continuous differential equations  │ Sampling: Stochastic discrete global jumps   │
│ F = -∇V = m·a (Verlet / Langevin dynamics)   │ (Monte Carlo, Genetic Algorithms, Simplex)   │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Potential: Physical Hamiltonians             │ Potential: Empirical Scoring Functions       │
│ (12-6 LJ, 1/r Coulomb, explicit/GB solvent)  │ (Soft-core LJ, screened ε(r), H-bond geometry)│
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Evaluation: All-atom pairwise interactions   │ Evaluation: Precalculated 3D potential grids │
│ O(N_rec × N_lig) or Cell-list on GPU         │ O(N_lig) trilinear / spline interpolation    │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 3. The 5 Core Challenges & Technical Solutions

### Challenge 1: Degrees of Freedom ($3N$ Cartesian vs. $6+M$ Torsion Space)
* **The Problem:** In internal coordinates, a drug molecule with 35 atoms has $6$ rigid-body DOFs $+ M$ rotatable torsions ($10\text{--}15$ total parameters). In Cartesian space ($105$ DOFs), moving atoms independently creates non-physical valence deformations.
* **The Solution:** We apply stiff harmonic bond ($k_b = 500{,}000\text{ kJ/(mol nm}^2)$) and angle forces ($k_\theta = 2000\text{ kJ/(mol rad}^2)$) while keeping rotatable single bonds flexible ($k_{\text{rot}} = 4.0\text{ kJ/mol}$).

### Challenge 2: The "Python Loop" Latency Trap
* **The Problem:** A Python-level Monte Carlo loop (`setPositions()` $\to$ `getState(getEnergy=True)`) incurs a $0.1\text{--}0.5\text{ ms}$ round-trip overhead per step over PCI-e. $100{,}000$ steps take $30\text{--}60\text{ seconds}$ in Python vs. $< 1\text{ s}$ in C++.
* **The Solution:** We execute **Simulated Annealing MD (SAMD)** directly on the GPU context using `LangevinMiddleIntegrator` (e.g., cooling from $800\text{ K} \to 10\text{ K}$), followed by GPU-side L-BFGS minimization (`LocalEnergyMinimizer`).

### Challenge 3: Scoring Functions vs. Physical Force Fields
* **The Problem:** Standard 12-6 Lennard-Jones and unscreened Coulomb forces in vacuum create infinite repulsive walls at close contacts and artificially huge salt-bridge attractions ($-100\text{ kcal/mol}$).
* **The Solution:** We implemented rDock's soft-core 4–8 Lennard-Jones ($r_{\text{eff}} = \sqrt{r^2 + \delta^2}$) and distance-dependent dielectric screening ($\epsilon(r) = D \cdot r$).

### Challenge 4: Ligand Structural Integrity & Bond Stretching
* **The Problem:** Without explicit bonded terms in OpenMM, Cartesian minimization or thermal annealing at $800\text{ K}$ stretched covalent bonds up to $25\text{--}30\text{ \AA}$.
* **The Solution:** We automatically extract all $1\text{-}2$ bonds, $1\text{-}3$ angles, and dihedral parameters from RDKit conformers, populating `HarmonicBondForce` and `HarmonicAngleForce` in a dedicated `GROUP_VALENCE` force group.

### Challenge 5: Aromatic Ring Buckling & Out-of-Plane Puckering
* **The Problem:** 
  1. *Intra-ring Nonbonded Self-Attraction:* 1-4 and 1-5 atom pairs across a 6-membered aromatic ring attracted each other, causing the ring to buckle into a "boat" or "taco" shape.
  2. *Unconstrained Diagonals:* A 3D polygon has unconstrained internal degrees of freedom unless diagonals are locked.
* **The Solution:**
  1. **Intra-ring Exclusions:** All pairwise nonbonded interactions between atoms in the same ring are explicitly excluded from `CustomNonbondedForce`.
  2. **Cross-Ring Triangulation:** We add harmonic distance springs ($k = 500{,}000\text{ kJ/(mol nm}^2)$) across all cross-ring diagonals ($1\text{-}3, 1\text{-}4, 2\text{-}4, 2\text{-}5, 3\text{-}5$).
  3. **Improper Dihedrals:** We apply improper out-of-plane dihedral restraints ($k = 500\text{ kJ/mol}$) on all trivalent $sp^2$ aromatic atoms to keep substituents coplanar.

---

## 4. Scoring Function Mathematical Formulation in OpenMM

Implemented in `openmm_dock/scoring.py` as **six separate `CustomNonbondedForce` objects**,
each in its own OpenMM force group. Earlier revisions computed one combined
nonbonded blob and reported `SCORE.INTER.VDW`/`POLAR`/etc. as fixed fractions
of it (`nb_e * 0.5`, `nb_e * 0.3`, ...). That was cosmetic, not physics: the
values didn't correspond to anything actually being evaluated. Every term
below is now its own force and its own real energy.

$$E_{\text{total}} = E_{\text{inter}} + E_{\text{intra}} + E_{\text{restr}} + E_{\text{system}}$$

### 1. Intermolecular Nonbonded Energy ($E_{\text{inter}}$)
$$E_{\text{inter}} = w_{\text{vdw}} E_{\text{vdw}} + w_{\text{pol}} E_{\text{polar}} + w_{\text{hb}} E_{\text{hb}} + w_{\text{repul}} E_{\text{repul}} + w_{\text{hyd}} E_{\text{hyd}} + E_{\text{const}} + E_{\text{rot}}$$

* **Soft-Core 4–8 Lennard-Jones ($E_{\text{vdw}}$)** — own force, `GROUP_VDW_INTER`:
  $$E_{\text{vdw}} = 4\epsilon \left[ \left(\frac{\sigma}{r_{\text{eff}}}\right)^8 - \left(\frac{\sigma}{r_{\text{eff}}}\right)^4 \right]$$
  where $r_{\text{eff}} = \sqrt{r^2 + \delta^2}$, $\delta = 0.05\text{ nm}$, $\sigma = \frac{\sigma_1 + \sigma_2}{2}$, and $\epsilon = \sqrt{\epsilon_1 \epsilon_2}$.

* **Screened Electrostatics + Contact H-Bonding ($E_{\text{polar}}$, $E_{\text{hb}}$)** — own force, `GROUP_POLAR_INTER`:
  $$E_{\text{polar}} = \frac{138.935456 \cdot q_1 q_2}{D \cdot r_{\text{eff}}^2}, \qquad E_{\text{hb}} = -12.0 \cdot (\text{don}_1 \cdot \text{acc}_2 + \text{don}_2 \cdot \text{acc}_1) \cdot \exp\left( - \frac{(r_{\text{eff}} - 0.28\text{ nm})^2}{0.02} \right)$$

* **Short-Range Polar Clash Penalty ($E_{\text{repul}}$)** — own force, `GROUP_REPUL`. The OpenMM analogue of rDock's `RbtPolarIdxSF(ATTR=FALSE)`: a repulsive-only term specific to donor/acceptor atom pairs closer than the ideal H-bond distance, independent of the generic vdW wall:
  $$E_{\text{repul}} = \Theta(r_{\min} - r_{\text{eff}}) \cdot k_{\text{repul}} \cdot (r_{\min} - r_{\text{eff}})^2, \quad r_{\min} = 0.24\text{ nm}$$
  applied only between polar (donor-or-acceptor) atom pairs.

* **Hydrophobic Desolvation ($E_{\text{hyd}}$)** — own force, `GROUP_HYD`:
  $$E_{\text{hyd}} = -3.0 \cdot (\text{hyd}_1 \cdot \text{hyd}_2) \cdot \exp\left( - \frac{(r_{\text{eff}} - 0.38\text{ nm})^2}{0.04} \right)$$

* **Constant / Solvent-Enablement Penalty ($E_{\text{const}}$)** — not a nonbonded force at all, matching rDock's `RbtConstSF`, which is a true constant rather than distance-dependent: $E_{\text{const}} = w_{\text{const}} \cdot n_{\text{waters}}$ (cost of enabling each explicit active-site water; zero when no waters are present).

* **Rotatable Bond Entropy Penalty ($E_{\text{rot}}$)** — also not a nonbonded force; matches rDock's `RbtRotSF`: $E_{\text{rot}} = w_{\text{rot}} \cdot n_{\text{rot}}$, where $n_{\text{rot}}$ is RDKit's real rotatable-bond count for the ligand topology.

Intramolecular ligand strain ($E_{\text{intra}}$) uses the same VDW/POLAR expressions restricted to ligand-ligand pairs, each in its own group (`GROUP_VDW_INTRA`, `GROUP_POLAR_INTRA`) so it is never conflated with the intermolecular terms.

### 2. Force Group Decomposition
By assigning OpenMM forces to specific force groups, exact decomposed energies are computed via `context.getState(getEnergy=True, groups={...})`:
* `GROUP_VDW_INTER = 0`: Intermolecular soft-core 4-8 Lennard-Jones.
* `GROUP_VALENCE = 1`: Ligand covalent bonds, angles, ring triangulation, and dihedrals.
* `GROUP_CAVITY = 2`: Cavity flat-bottom harmonic restraint.
* `GROUP_PHARMA = 3`: Pharmacophore feature restraints.
* `GROUP_TETHER = 4`: Template core positional restraints.
* `GROUP_SOLVENT = 5`: Flexible active-site solvent water tethering.
* `GROUP_POLAR_INTER = 6`: Intermolecular screened electrostatics + contact H-bond bonus.
* `GROUP_REPUL = 7`: Short-range polar clash penalty.
* `GROUP_HYD = 8`: Hydrophobic desolvation contact bonus.
* `GROUP_VDW_INTRA = 9`: Intramolecular ligand vdW strain.
* `GROUP_POLAR_INTRA = 10`: Intramolecular ligand electrostatic strain.

`SCORE.INTER.CONST` and `SCORE.INTER.ROT` are computed directly from ligand/solvent topology (not from any force group), since rDock's own `RbtConstSF` and `RbtRotSF` are non-distance-dependent terms.

**Known remaining gap:** `SCORE.SYSTEM` still reports only the solvent tether *restraint* energy, not real water–protein / water–ligand nonbonded energy (rDock's actual `SCORE.SYSTEM.VDW`/`SCORE.SYSTEM.POLAR`). Explicit waters currently only interact nonbonded-wise with the ligand (folded into `SCORE.INTER`); water-protein interactions are not scored. This is a separate, larger change (waters need a third `is_water` particle category, not just `is_lig`) and is out of scope for the decomposition fix.

---

## 5. Feature-by-Feature Implementation

### Cavity Definition & Restraints
* Parses `cavity.prm` for `REF_MOL` (calculating bounding radius around reference ligand) or explicit `CENTER (x, y, z)` and `RADIUS`.
* Applies flat-bottom harmonic restraint:
  $$V_{\text{cavity}} = \frac{1}{2} k_{\text{cav}} \cdot \Theta(r - R_{\text{cav}}) \cdot (r - R_{\text{cav}})^2$$
  where $\Theta(u)$ is the Heaviside step function.

### Active-Site Solvent & Water Tethering
* Explicit active-site waters (e.g. `test_waters.pdb`) participate in nonbonded and H-bond interactions with both receptor and ligand.
* Oxygen atoms are tethered with a flexible spherical leash ($r_{\text{tol}} = 0.8\text{ \AA}$), allowing rotation and minor translation to accommodate diverse ligand chemotypes.

### Pharmacophore Restraint Constraints
* Parses `pharma.restr` defining $(x, y, z, \text{tolerance}, \text{type})$ where type is `Aro`, `Acc`, `Don`, or `Hyd`.
* RDKit automatically identifies ligand pharmacophore features.
* A unified `CustomExternalForce` applies flat-bottom quadratic penalties for any feature drifting beyond the specified tolerance.

### Template-Constrained Tethered Docking (MCS)
* Computes Maximum Common Substructure (MCS) between query ligand and reference co-crystal ligand.
* Applies harmonic position restraints ($k_{\text{tether}} = 5000\text{ kJ/(mol nm}^2)$) on matched core atoms:
  $$V_{\text{tether}} = \frac{1}{2} k_{\text{tether}} \sum_{i \in \text{core}} \| \mathbf{r}_i - \mathbf{r}_{0,i} \|^2$$

### RNA Aptamer Docking
* Supports Tripos Mol2 biopolymer nucleic acid topologies (e.g. 1NEM aptamer), handling ribose backbone and nitrogenous base charges and atom types natively.

---

## 6. Benchmark Verification & Metrics

Tested against all 6 use cases from `rxdock-deepdive-examples`:

| Test Case | Dataset | Initial Status | Final OpenMM Result |
| :--- | :--- | :--- | :--- |
| **1. Score** | `test_examples/score/ii.sd` | Clashing | `Score: 208.333 (Inter: 166.666, Cavity: 0.0)` |
| **2. Minimize** | `test_examples/minimize/ii.sd` | Positive ($+208$) | `Min Score: -140.188` (all 10 poses relaxed) |
| **3. Solvent** | `test_examples/solvent/lig.sdf` | Unrelaxed | `Min Score: -149.855` (flexible waters) |
| **4. Pharmacophores** | `test_examples/pharmacophores/xtal-lig.sd` | Strained | `Docked Score: +430.343 (VDW: +20.70, Pharma Restr: 0.0)` |
| **5. Tethered** | `test_examples/tethered/query_ligands.sdf` | 24 core atoms | `Tethered Rank 1 Score: -54.742` |
| **6. RNA Aptamer** | `test_examples/rna_docking_example/1nem` | 1NEM RNA | `Rank 1 Score: -1019.494 (VDW: -510.53)` |

### Structural Integrity Verification:

* **Bond Lengths:** $0.95\text{ \AA} \le r \le 1.58\text{ \AA}$ (mean $1.39\text{ \AA}$, target $1.0\text{--}1.54\text{ \AA}$).
* **Aromatic Ring Planarity:** Maximum atom distance to optimal plane $< 0.03\text{ \AA}$ (matches crystal structure).

---

# Part II — Kinematics: Parameterizing Shape

Part I covered *scoring* — how good a pose is. This part covers *representation* —
what a "pose" actually is as a small vector of numbers, and how that vector
turns into 3D atomic coordinates. Getting this right is what makes every
search algorithm in Part III computationally tractable at all.

## 7. Kinematics: From $3N$ Cartesian to Internal Coordinates

A rigid receptor + flexible ligand docking problem has, in principle, $3N$
Cartesian degrees of freedom for an $N$-atom ligand. Almost all of that is
redundant: bond lengths and angles barely change between a docked and an
undocked pose, so treating them as free variables just gives the optimizer
a huge, mostly-flat, easy-to-get-lost-in search space. **Kinematics** here
means: pick a much smaller set of numbers — 3 translation + a rotation +
one angle per rotatable bond — that can only ever generate chemically valid
3D structures, no matter what values you plug in. This is precisely the
same idea used to control a robot arm: you don't specify the $(x,y,z)$ of
every point on the arm, you specify joint angles, and forward kinematics
computes where everything ends up.

### 7.1 Forward Kinematics — `kinematics.py`

**`LigandKinematicTree`** (`kinematics.py:31`) builds a torsion tree from an
RDKit molecule:

1. **Identify rotatable bonds** with the SMARTS pattern
   `[!$(*#*)&!D1]-!@[!$(*#*)&!D1]`: a single, non-ring bond (`-!@` excludes
   ring bonds — rings are treated as *rigid*, see §7.2 for why macrocycles
   need a different treatment) between two non-terminal (`!D1`), non-triple-bond-adjacent
   (`!$(*#*)`) atoms.
2. **For each rotatable bond**, breadth-first search from one side finds
   every atom "downstream" of the bond — the subtree that rotates together
   when that bond's dihedral changes (`kinematic_utils.find_downstream_atoms`,
   §7.3). If more than half the molecule would move, the bond direction is
   flipped so the *smaller* side is what actually rotates — purely a
   performance/numerical-conditioning choice, physically equivalent either
   way.
3. **`forward_kinematics(translation, quaternion, dihedrals)`** then:
   - For each joint, rotates its subtree around the bond axis by the
     target dihedral angle using the **Rodrigues rotation formula**
     (`scipy.spatial.transform.Rotation.from_rotvec`) — a closed-form way
     to build a 3D rotation matrix from an axis and an angle without
     trigonometric case-analysis.
   - Applies one global rigid-body rotation (quaternion) about the
     molecule's centroid.
   - Applies one global translation.

   Because every step is a *rotation* (which by definition preserves all
   distances within the rotated subtree) or a *translation* (which
   preserves everything), **bond lengths are exactly preserved by
   construction** — 0.000 Å deviation, not a soft constraint that's merely
   very stiff. This is the kinematic-space analogue of Part I's harmonic
   bond springs, except here it's a mathematical guarantee rather than an
   energy penalty that could in principle be violated under enough force.

**`KinematicDockingEngine`** (`kinematics.py:150`) wraps a `LigandKinematicTree`
with a real OpenMM system so `(translation, quaternion, dihedrals)` can be
scored: `evaluate_state()` calls `forward_kinematics()` to get Cartesian
coordinates, then a single `context.setPositions()` + `getState(getEnergy=True)`.

**`KinematicParticleSwarmOptimizer`** (`kinematics.py:271`) runs PSO
(§10.1's simpler, single-ligand-only ancestor) directly in this reduced
$SE(3) \times T^k$ space (3 translation + a rotation vector + $k$ torsion
angles) instead of Cartesian space — a 10-15 dimensional search instead of
$3N \approx 100$.

### 7.2 Inverse Kinematics for Macrocycles — `inverse_kinematics.py`

Rigid-ring treatment breaks down for **macrocycles** — large rings
(≥9 atoms is this codebase's threshold) common in newer drug modalities,
which genuinely flex internally (a 14-membered macrocycle is not remotely
as rigid as a benzene ring). But you can't just "add torsions" around a
ring the way §7.1 does for a chain: rotating one ring bond changes where
the *other end* of the ring ends up, and there's no free end to absorb
that — the ring has to stay closed. This is a **closed-loop kinematics**
problem, and it's solved here with the same numerical technique used to
position a robot arm's end-effector: **Damped Least Squares (DLS) inverse
kinematics**.

**`MacrocycleInverseKinematics`** (`inverse_kinematics.py:29`):

1. Picks the largest macrocyclic ring, then **cuts** it at one bond
   (`cut_a1`–`cut_a2`) — conceptually turning the closed ring into an open
   chain, like unclipping one link of a bracelet. The target is to rotate
   the *other* bonds in the ring so that this cut gap closes back up to
   its original bond length.
2. Builds a Jacobian $J \in \mathbb{R}^{3 \times k}$ ($k$ = number of
   rotatable ring joints) where column $j$ is $J_{:,j} = \hat{u}_j \times
   \vec{r}_j$ — the standard robotics formula for how much the "end
   effector" (the tip atom of the cut) moves per unit rotation of joint
   $j$, where $\hat{u}_j$ is joint $j$'s rotation axis and $\vec{r}_j$ is
   the vector from that joint to the tip.
3. Iteratively solves $\Delta\theta = J^T (JJ^T + \lambda^2 I)^{-1}
   \Delta\vec{e}$ — the **damped pseudo-inverse**: without the $\lambda^2
   I$ damping term this is the ordinary Moore-Penrose pseudo-inverse
   solution to "what joint velocities close the gap fastest," but that
   becomes numerically unstable near singular Jacobian configurations
   (when joints become nearly co-linear); the damping term trades a little
   convergence speed for guaranteed numerical stability everywhere — the
   textbook fix from robotics literature (Wampler 1986, Nakamura & Hanafusa
   1986).
4. Converges (typically <30 iterations) to a set of joint angles that
   closes the ring to within `tolerance` (default $10^{-4}$ nm).

**`TwoTierMacrocycleEngine`** (`inverse_kinematics.py:265`) combines this
with ordinary forward kinematics: **Tier 1** is the macrocyclic ring
backbone, moved via the IK solver above (the only part of the molecule that
genuinely needs it); **Tier 2** is every exocyclic side-chain arm hanging
off the ring, moved with the same rigid-subtree forward kinematics as
§7.1 (`_find_exocyclic_subtree`, now `kinematic_utils.find_downstream_atoms`).
This decoupling — constrained IK only where topology actually demands it,
cheap FK everywhere else — is what makes macrocycle docking (`kinematics_workflow`
demos) tractable at all.

### 7.3 Shared Kinematic Utilities — `kinematic_utils.py`

Three primitives factored out after being duplicated across 6+ modules
(consolidated in this session — see the commit history for the before/after):

- **`toroidal_diff(a, b)`** = `arctan2(sin(a-b), cos(a-b))`: the *shortest*
  angular difference between two angles on a circle. Plain subtraction is
  wrong for angles — the difference between 179° and −179° is 2°, not
  358° — and every PSO/GA/metadynamics engine in this codebase that
  compares dihedral angles needs this.
- **`find_downstream_atoms(mol, begin_idx, split_idx, extra_blocked_edges=None)`**:
  the generalized BFS subtree-finder behind both §7.1's rigid-chain
  rotation and §7.2's exocyclic arms — it finds every atom reachable from
  `split_idx` without crossing the `begin_idx`–`split_idx` edge (plus any
  extra blocked edges, needed for the macrocycle ring-closure case where a
  ring provides a *second* path back to `begin_idx` that must also be
  blocked).
- **`identify_rotatable_bonds(mol)`**: the SMARTS + deduplication logic
  from step 1 of §7.1, shared by both the ligand-only and macrocycle
  engines.

---

# Part III — Search Algorithms

## 8. Search & Optimization Algorithms — `engine.py`

All four algorithms below operate on the **same chromosome representation**:
a flat vector `[trans_x, trans_y, trans_z, euler_x, euler_y, euler_z,
torsion_1, ..., torsion_k]` — 6 rigid-body genes (3 translation in Å, 3
Euler angles in degrees) plus one gene per rotatable bond (§7.1). `decode_chromosome()`
turns this vector into Cartesian coordinates (apply each torsion, then the
rigid-body placement); `encode_chromosome()` is its inverse, recovering the
chromosome that (approximately) produces a given set of Cartesian
coordinates — used to make local Cartesian-space minimization
**Lamarckian**: a locally-optimized *phenotype* gets written back into the
*genotype*, so the improvement compounds across generations/moves instead
of being immediately perturbed away by the next random proposal (the
standard Lamarckian-GA idea from AutoDock's own LGA).

### 8.1 Simulated Annealing (`dock_simulated_annealing`)

Not literal molecular dynamics — a **Metropolis Monte Carlo chain with a
cooling temperature schedule**, mirroring AutoDock's own SA protocol rather
than integrating Langevin dynamics on the chromosome (early experiments
with literal MD in this space were numerically unstable — see the
docstring at `engine.py:1044` for the measured failure mode). Each step:
propose a Gaussian perturbation to the chromosome (`mutate_chromosome`),
accept if the new energy is lower, or with probability $\exp(-\Delta E /
RT)$ if higher (the Metropolis criterion) — at high $T$ almost any move is
accepted (exploration), at low $T$ only downhill moves survive
(exploitation). Every `lamarck_interval` accepted moves, a real OpenMM
`LocalEnergyMinimizer` pass runs and its result is re-encoded into the
chromosome (the Lamarckian step above).

### 8.2 Genetic Algorithm (`dock_genetic_algorithm`)

Same chromosome, same Lamarckian local-minimization step, but population-based:
a pool of chromosomes evolves via selection (fitter chromosomes more likely
to reproduce), crossover (splicing two parent chromosomes), and mutation,
across generations — the standard genetic algorithm loop, specialized to
this docking chromosome the way AutoDock's own LGA (Lamarckian Genetic
Algorithm) is.

### 8.3 Monte Carlo / Basin Hopping (`dock_monte_carlo`)

A simpler, single-trajectory Metropolis Monte Carlo without an annealing
schedule (fixed temperature) — proposes a move, evaluates energy,
accept/reject, optionally minimizes each accepted step (`minimize_each_step`).
The conceptual predecessor to §8.4's more faithful basin-hopping
implementation.

### 8.4 Monte-Carlo-with-Minimization (Vina/smina-style) (`dock_monte_carlo_minimization`)

Added by directly reading AutoDock Vina/smina's actual C++ source
(`monte_carlo.cpp`, `mutate.cpp`, `quasi_newton.cpp`, `bfgs.h` — cloned from
`github.com/mwojcikowski/smina`) rather than reimplementing from a
description, to understand *why* Vina-family tools reliably converge to
precise poses. The key structural difference from §8.1's SA: Vina's
`single_run` does, every single step, **one coarse random move → an
immediate full local minimization → THEN the Metropolis accept/reject test
on the *minimized* energy** — so every state the Markov chain actually
compares is already a converged local-basin minimum, not a noisy raw
proposal. SA-style methods only minimize *periodically* (every
`lamarck_interval` moves), so most of the chain is comparing un-minimized,
noisy energies — a fundamentally different, less precise search dynamic.

This is reimplemented here as basin hopping over the *same* chromosome,
using:
- **`mutate_chromosome_vina_style`** (`engine.py`, near `mutate_chromosome`):
  mirrors Vina's `mutate_conf` — picks exactly ONE of
  {translation, rotation, torsion$_1$, ..., torsion$_k$} uniformly at
  random and applies one full-amplitude move to just that entity (a
  translation of up to `trans_amplitude` Å, or a full random-angle torsion
  reset), leaving every other gene untouched. Deliberately coarser than
  §8.1's every-gene-at-once small Gaussian jitter — the immediate
  minimization that follows is what refines it, not the proposal itself.
- **`gradient_minimizer.lbfgs_minimize`** (§11) as the local minimizer —
  the finite-difference numerical counterpart to Vina's analytic-gradient
  BFGS (Vina computes gradients analytically via chain rule from atomic
  forces in C++; this codebase's OpenMM energy function doesn't expose
  that in chromosome space, so central finite differences are used
  instead — more expensive per step, which is the real reason this
  implementation can only afford ~10-30 basin-hop steps per run where Vina
  affords ~2500).

Benchmarked honestly against SA on real docking cases (see
`test_examples/9z1l/run_pharmacophore_docking_mcm_demo.py` and its results
in that directory's demo output): **underperformed** SA in this codebase
(finite-difference gradient cost is the bottleneck, not a flaw in the
basin-hopping idea itself) — kept as a documented, tested alternative
strategy rather than discarded, since the mechanism itself is sound and the
limitation is specific to not having analytic gradients available.

---

## 9. Receptor Flexibility — `receptor_kinematics.py`

Real induced-fit binding involves the receptor's own side chains moving,
not just the ligand. **`ReceptorSideChainKinematics`** (`receptor_kinematics.py:78`)
represents this the same way real protein side chains actually move: as
rotations around a small number of **chi ($\chi$) dihedral angles** — the
standard structural-biology description of a side chain's own internal
rotatable bonds (e.g. leucine has $\chi_1$, $\chi_2$; lysine has up to
$\chi_1$–$\chi_4$). **`ChiJoint`** and **`FlexibleResidue`** identify which
residues (typically: within some radius of the binding site) get this
treatment and which of their bonds are the actual chi torsions, using the
same rigid-subtree-rotation forward-kinematics machinery as §7.1, just
applied to a receptor side chain instead of a whole ligand. This is what
lets `UnifiedKinematicPSOEngine` (§10.1) couple ligand pose *and* receptor
side-chain conformation into one combined search — genuine, if
computationally limited, induced-fit docking.

---

# Part IV — Swarm, Gradient, and Bayesian Sampling

## 10. Swarm Intelligence & Enhanced Sampling

### 10.1 Particle Swarm Optimization — `unified_kinematic_pso.py`

**Particle Swarm Optimization (PSO)** (Kennedy & Eberhart, 1995) models a
population of candidate solutions ("particles") each with a position and a
velocity; every step, each particle's velocity is nudged toward both its
own best-ever position (`p_best`, cognitive term) and the swarm's
best-ever position (`g_best`, social term), then the particle moves by
that velocity. It's a simple, gradient-free, easily-parallel global
optimizer — no derivative needed, works on any black-box scoring function.

**`UnifiedKinematicPSOEngine`** (`unified_kinematic_pso.py:55`) is the
"everything coupled" version: one **`UnifiedSwarmParticle`** carries the
ligand's rigid-body pose + torsions (§7.1) *and*, when applicable, the
macrocycle ring-driver angles (§7.2) *and* exocyclic dihedrals *and* the
receptor's own flexible side-chain chi angles (§9) — all evolved together
in one PSO swarm, so ligand placement and induced-fit receptor
relaxation happen simultaneously rather than in separate passes. Velocity
updates on every angular component use `toroidal_diff` (§7.3) instead of
plain subtraction, since $p_{best} - x$ for an angle must respect
periodicity.

### 10.2 Blind Global Docking — `global_blind_docking.py`

Standard docking assumes the pocket location is roughly known (a cavity
center, §5). **Blind docking** removes that assumption: the ligand starts
genuinely far away (tens of Å, in bulk solvent) with no prior information
about where to go. `GlobalBlindDockingEngine.run_blind_docking()`
(`global_blind_docking.py:153`) combines three ideas to make this
tractable:

1. **Multi-conformer swarm seeding** (`generate_conformer_seeds`): starts
   several genuinely different ETKDG-generated 3D conformers, not just
   rigid-body perturbations of one starting shape, so the swarm isn't
   betting everything on one initial guess being roughly right.
2. **Metadynamics-style repulsive bias** (`compute_metadynamics_bias`, see
   §10.3): actively discourages the swarm from re-exploring regions it's
   already visited, pushing it to cover new ground instead of all
   particles collapsing onto the first mediocre local optimum found.
3. **"Beacon" guidance terms** in `evaluate_global_score`
   (`k_contact_beacon`, `k_depth_beacon`): soft, receptor-agnostic rewards
   for "more receptor contacts" and "further from the receptor surface,
   into a concave region" — proxies for "this looks like a pocket" that
   don't require knowing where the real pocket is, nudging blind search
   toward pocket-like regions without hard-coding their location. (A
   documented finding from real testing: over-weighting these relative to
   the real physical VDW term can dilute the one signal that discriminates
   a *precisely* correct pose from one that's merely "generally in a
   concave region" — see the comment block in
   `test_examples/9z1l/run_blind_docking_demo.py`.)

### 10.3 Metadynamics as an Enhanced-Sampling & Pose-Strength Tool

**Metadynamics** (Laio & Parrinello, 2002) is a standard molecular-dynamics
enhanced-sampling technique: periodically deposit a small Gaussian "hill"
of repulsive bias potential at the system's current position in some
low-dimensional **collective variable (CV)** space. Over time this fills in
already-visited basins, pushing the system to explore new ones — and once
the whole accessible CV landscape has been covered, the accumulated bias
is (up to a constant) the negative of the true free-energy surface, so
metadynamics is simultaneously an enhanced sampler *and* a free-energy
estimator.

This codebase uses it in **kinematic parameter space** (translation +
torsions, not Cartesian atom positions) and in three configurations of
increasing scope:

- **`KinematicMetadynamicsEngine`** (`metadynamics.py:37`): single-particle
  metadynamics on one ligand pose's kinematic coordinates.
- **`SwarmMetadynamicsEngine`** (`swarm_metadynamics.py:25`): a whole PSO
  swarm sharing one metadynamics bias landscape (`compute_shared_bias_and_gradient`),
  so different particles don't waste effort re-exploring the same basin
  another particle already mapped.
- **`CollaborativeKinematicMetaDEngine`** (`collaborative_kinematic_metadynamics.py:238`):
  multiple **islands** of particles (`CollaborativeIsland`) each with their
  own local best, all depositing bias into one **`SharedMetadynamicsArchive`**
  of **`SharedBasin`** records — an island-model GA / parallel-tempering-flavored
  extension, useful for genuinely multi-modal problems (several plausible
  binding modes) where one swarm might otherwise converge prematurely to
  just one of them.

A distinctive use of this machinery specific to this codebase: metadynamics
as a **pose-strength assay**, not just a sampler. Deposit *repulsive* bias
directly at a candidate native pose and count how many hills it takes
before that pose's *raw physical score* (not the biased score) turns
unfavorable — a shallow, weakly-specific binding mode gets pushed out
after very few hills; a deep, well-discriminated native pose survives many
more. This is what `run_metadynamics_demo.py`-style demos across
`test_examples/` actually report.

### 10.4 Generalized, Reference-Free Collective Variables — `generalized_cv.py`

Early metadynamics/blind-docking work in this codebase hard-coded
system-specific CVs (e.g. "distance to *this* pocket's known center").
**`GeneralizedCVEngine`** (`generalized_cv.py:42`) replaces those with three
CVs computable on *any* receptor with no prior pocket knowledge:

- **`compute_pocket_depth`** ($\zeta_{\text{depth}}$): how far the ligand
  has penetrated past the receptor's own convex-hull-like outer envelope —
  a generic "is this ligand buried in a concavity" measure.
- **`compute_contact_coordination`** ($Q_{\text{contacts}}$): a smooth,
  differentiable count of receptor atoms within contact range — analogous
  to the $Q$ order parameter from protein-folding CV literature (fraction
  of native contacts formed), repurposed here as "how many binding
  contacts has this pose made."
- **`compute_radius_of_gyration`** ($R_g$): the ligand's own compactness,
  useful for distinguishing an extended, non-specifically-adsorbed pose
  from a well-packed one.

`GeneralizedCVMetadynamicsEngine` runs metadynamics over this
$(\zeta_{\text{depth}}, Q_{\text{contacts}}, R_g)$ space and can plot the
resulting 2D free-energy surface (`plot_universal_binding_funnel_fes`) — a
genuine "funnel" picture of the binding landscape, generated without any
system-specific hand-tuning, in the spirit of funnel-metadynamics /
Markov-state-model binding-landscape studies from the broader
computational chemistry literature.

---

## 11. Gradient-Based and Bayesian Optimization

Two modules directly mirror **OpenDock** (`github.com/guyuehuo/opendock`)'s
sampler design, reimplemented independently against this codebase's own
OpenMM-based energy functions (not ported):

**`gradient_minimizer.py`** — gradient descent over any black-box, low-dimensional
kinematic-space objective, using **central finite-difference gradients**
(`central_difference_gradient`: $\partial f/\partial x_i \approx
[f(x+\delta e_i) - f(x-\delta e_i)] / 2\delta$) since the OpenMM energy
isn't exposed as an analytically-differentiable function in chromosome
space. Two optimizers built on top of that gradient:
- **`adam_minimize`**: the Adam optimizer (Kingma & Ba, 2014) — maintains
  running first- and second-moment estimates of the gradient to adapt the
  effective step size per-dimension, standard in deep learning, useful
  here because different chromosome dimensions (Å of translation vs.
  degrees of rotation) have very different natural scales.
- **`lbfgs_minimize`**: wraps `scipy.optimize.minimize(method="L-BFGS-B")`
  — a quasi-Newton method that builds up a curvature (Hessian) estimate
  from successive gradient evaluations, typically converging in far fewer
  evaluations than plain gradient descent for a smooth, low-noise
  objective. This is the local minimizer used by §8.4's Vina-style basin
  hopping.

**`bayesian_optimizer.py`** — a **from-scratch Gaussian Process (GP)**
regressor (Matérn-5/2 kernel, hyperparameters fit by maximizing marginal
log-likelihood via Nelder-Mead restarts) plus **Expected Improvement**
acquisition-function optimization: at each step, fit a GP to all points
evaluated so far, then pick the next point that maximizes the *expected*
improvement over the current best (balancing exploring uncertain regions
against exploiting the GP's predicted optimum). Bayesian optimization is
the right tool specifically when each objective evaluation is expensive
relative to the search dimensionality — it typically needs far fewer
evaluations than PSO/GA to find a good optimum, at the cost of being
inherently sequential (each new point depends on the GP fit to all
previous ones) rather than trivially parallelizable.

**`scoring_function.py`** — a `BaseScoringFunction` abstract interface
(`.score(lig_coords, rec_coords=None) -> float`, lower is better) plus a
`CompositeScoringFunction` that sums several weighted scorers. This exists
so a future correction term — e.g. a machine-learned rescoring layer, in
the spirit of OpenDock's OnionNet-SFCT weighted correction — can be
composed with the existing OpenMM physical score *without modifying any
sampling engine*: every optimizer in §8, §10, and §11 already treats
"the objective" as an opaque callable, so swapping in a
`CompositeScoringFunction` in place of a raw OpenMM energy call is a
drop-in change.

---

# Part V — Protein-Protein Docking

## 12. Protein-Protein Docking: Glowworm Swarm Optimization

`glowworm_swarm.py` and `lightdock_dfire_scoring.py` extend this codebase
from small-molecule-in-pocket docking to **rigid-body protein-protein
docking** — two macromolecules, no ligand torsion tree, a fundamentally
different search topology (one whole partner's 6-DOF rigid-body pose
relative to the other).

**Glowworm Swarm Optimization (GSO)** (Krishnanand & Ghose, 2009) is the
algorithm behind **LightDock** (`github.com/lightdock/lightdock`, GPLv3 —
reimplemented here from reading the published algorithm and their source
for understanding, not ported, so this repository's own licensing stays
unencumbered). It differs from §10.1's PSO in one key way: instead of
every particle chasing one global best, each "glowworm" carries a scalar
**luciferin** brightness (updated from its own score:
$L \leftarrow (1-\rho)L + \gamma \cdot \text{score}$) and only moves toward
neighbors *within its own local vision range* that are brighter — never a
single global attractor. Because movement is purely local, a GSO swarm can
naturally split across **several simultaneous local optima** at once
(useful when multiple binding modes are plausible), which single-global-best
PSO structurally cannot do.

**`GlowwormSwarmOptimizer`** (`glowworm_swarm.py`) implements the per-step
loop: evaluate every glowworm's energy → update luciferin → snapshot the
whole swarm's state → each glowworm finds brighter neighbors within
`vision_range` and moves a **fraction** of the way toward one, chosen
probabilistically weighted by luciferin difference (roulette-wheel
selection) → `vision_range` self-adapts to target a preferred neighbor
count. Two real, non-obvious implementation details worth calling out
because they were found only by careful ablation testing (full account in
`test_examples/protein_protein_1brs/DOCUMENTATION.md`):

- **The step must be a fractional interpolation, not a fixed distance.**
  LightDock's own `step_translation`/`step_rotation` parameters are
  documented as "jump 50% of the way to the target," not "move 0.5 Å" — an
  easy, plausible-looking mistake with a large real consequence: a fixed
  small step can be physically incapable of reaching a target tens of Å
  away within any reasonable step budget, no matter how good the search
  otherwise is.
- **The movement phase must be synchronous**: every glowworm's move this
  step has to be computed against a *snapshot* of the whole swarm's state
  from before any glowworm moved, not against a partially-updated list —
  otherwise later glowworms in the update order see some neighbors already
  moved and others not, an order-dependent artifact.

**Blind global coverage** comes from swarm *initialization*, not the
optimizer itself — GSO's own vision-range growth is too slow to let
information cross between swarms placed tens of Å apart within a
tractable step budget (empirically confirmed: see the DOCUMENTATION.md
ablation), so success on a genuinely blind problem depends almost entirely
on whether *some* swarm started in a good spot. Two placement strategies
are implemented:

- **`generate_surface_swarm_centers`**: a Fibonacci sphere (a
  deterministic, evenly-spaced point distribution on a sphere surface,
  common in computer graphics for "uniform random-looking but actually
  uniform" sampling) around the receptor's bounding sphere — simple, but
  blind to the receptor's true (non-spherical) shape.
- **`generate_sasa_swarm_centers`**: uses the `freesasa` package to
  compute real per-atom **solvent-accessible surface area**, keeps atoms
  with meaningfully nonzero SASA (i.e., genuinely exposed, not buried),
  and places one candidate point per exposed atom along that atom's own
  outward radial direction — candidate points that hug the true molecular
  surface shape instead of a sphere — then downselects to the requested
  swarm count via **greedy farthest-point sampling** (repeatedly pick the
  candidate point farthest from every already-selected point — a standard,
  simple technique for well-spread coverage from a larger candidate set).

**`lightdock_dfire_scoring.py`** wraps LightDock's own installed **DFIRE**
scoring function (Zhou & Zhou, 2002 — a *knowledge-based statistical
potential*: atom-type-pair energies derived from how often those atom
types are observed at each distance across the PDB, not from a physical
force field term, so it can't be written as an analytic LJ/Coulomb
expression) as a plain `(trans, quat) -> energy` callable — calling their
installed package as an external library dependency (same as depending on
scipy), not copying GPLv3 code into this repository.

---

# Part VI — Supporting Infrastructure

## 13. Supporting Infrastructure

### 13.1 Covalent Docking — `covalent.py`

Supports docking ligands that form a genuine covalent bond with the
receptor (e.g. an acrylamide warhead reacting with a cysteine thiol — a
common strategy in kinase inhibitor design). `detect_ligand_warhead` uses
SMARTS patterns to recognize known electrophilic warhead chemotypes;
`find_receptor_nucleophile` locates the reactive residue (e.g. a free
cysteine SG); `prealign_ligand_for_covalent_docking` performs an initial
rigid-body alignment placing the warhead near the nucleophile;
`create_covalent_restraint`/`create_covalent_bond_force` then apply a
distance restraint between the reactive atoms that behaves like a real
(if soft) covalent bond during the search, rather than leaving bond
formation entirely to chance rigid-body sampling.

### 13.2 Precomputed Potential Grids — `gridding.py`

The direct structural reason this codebase's population/generation-count
search budgets stayed modest throughout early development: every candidate
pose otherwise costs a real $O(N_{\text{receptor}})$ pairwise OpenMM force
evaluation. **`compute_potential_grids`** precomputes, once per
receptor+cavity, a 3D grid of the receptor's potential field for each
relevant probe atom type — the same trick AutoDock/AutoGrid and Vina use:
during the search's inner loop, a ligand atom's receptor-interaction
energy becomes an $O(1)$ trilinear grid interpolation instead of an
$O(N_{\text{receptor}})$ sum. `0.375` Å default spacing matches both real
tools' actual defaults (not a made-up round number). `create_boundary_penalty_force`
handles the edge case OpenMM's own `Continuous3DFunction` leaves undefined
(exactly zero outside the grid box, which would wrongly look like "no
interaction" rather than "off the edge of what we computed") the same way
Vina's own `grid.cpp` does: a smooth clamp-and-linear penalty for any atom
that strays outside the precomputed box.

### 13.3 Pose Clustering — `clustering.py`

`cluster_docked_poses` groups a set of docked conformers by heavy-atom
pairwise RMSD using RDKit's **Butina clustering** (Butina, 1999) — a
simple, deterministic leader-based algorithm widely used in cheminformatics
for exactly this "remove near-duplicate poses, keep one representative per
distinct binding mode" task: each cluster's representative is its
lowest-index (here, first-encountered / highest-ranked) member, and
`CLUSTER_SIZE` reports how many poses collapsed into it — a simple proxy
for how "wide" that basin of attraction is.

### 13.4 Protonation State Assignment — `protonation.py`

`protonate_ligand_ph` assigns the dominant physiological (pH 7.4)
ionization state to a ligand before docking — carboxylic/sulfonic
acids and tetrazoles deprotonate to their anionic forms, aliphatic amines
and guanidines protonate to their cationic forms — since a ligand's
protonation state materially changes its electrostatics and H-bonding
pattern, and getting it wrong silently produces a wrong pose regardless of
how good the search algorithm is. Tries an OpenBabel bridge first (a more
complete, empirically-parameterized pKa model) with a native RDKit
SMARTS-rule fallback that needs no extra dependency.

### 13.5 Bridged Two-Stage Docking — `bridged_docking.py`

**`BridgedTwoStageDockingEngine`** chains two of the engines above into one
automated pipeline for a problem neither handles well alone: genuinely
blind docking (§10.2) is good at *finding* a pocket from far away but
comparatively coarse once inside it; induced-fit local refinement
(macrocycle IK/FK + receptor chi flexibility, §7.2 + §9) is precise but
needs to already be roughly in the right place. The bridge:

1. **Stage 1 — Global Ingress**: `GlobalBlindDockingEngine` runs blind
   swarm-metadynamics search from bulk solvent against a rigid receptor.
2. **The Bridge Gate**: automatically detects when the swarm has
   genuinely entered the pocket ($\zeta_{\text{depth}} \le 5$ Å from the
   cavity centroid, §10.4's generalized CV) — a real, computed detection,
   not a fixed step count.
3. **Stage 2 — Induced-Fit Relaxation**: once through the gate, hands off
   to `TwoTierMacrocycleEngine` (§7.2) with receptor $\chi_1$–$\chi_4$
   flexibility (§9) enabled, refining the pose in place.

Works on both ordinary small molecules and macrocycles since Stage 2's
engine already generalizes across both (§7.2).

### 13.6 Command-Line Interface — `cli.py`

`main()` wires every engine above into the `omm-dock` command-line tool's
subcommands (`dock`, `mc`, `ga`, `minimize`, `score`, `tether`, ...) —
`parse_prm_receptor_and_cavity` reads an rDock-style `.prm` file to resolve
the receptor path and build a `CavityDefinition` (§5), so existing rDock
input files can be pointed at this engine with minimal translation.
