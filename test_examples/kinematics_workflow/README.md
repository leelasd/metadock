# Molecular Forward Kinematics Workflow Demonstration

This example demonstrates the **Robotic Forward Kinematics Tree Engine** in `openmm-dock`, parameterizing the ligand as a tree of rigid fragments connected by rotatable joint hinges $(SE(3) \times \mathbb{T}^k)$.

---

## 1. What This Example Demonstrates

1. **Dimensionality Reduction:** 
   * Reduces the Cartesian search space ($3N = 99$ coordinates) to strictly **$14$ joint dihedral angles** + **$6$ rigid-body degrees of freedom**.
2. **Zero Chemical Distortion:** 
   * Evaluated across **336 continuous frames**: Maximum bond length distortion is **$0.000000\text{ \AA}$** by mathematical definition.
3. **GPU-Accelerated Energy Evaluation:** 
   * Every kinematic frame is sent to the OpenMM GPU Context, evaluating nonbonded VDW, electrostatics, desolvation, and covalent forces in real time.

---

## 2. Running the Demonstration

```bash
cd test_examples/kinematics_workflow
python run_kinematics_demo.py
```

---

## 3. 3D Visual Inspection in PyMOL

Open the pre-configured PyMOL session script:

```bash
pymol visualize_pymol.pml
```

* **What you will see:**
  * The BTK kinase domain displayed in transparent cartoon and surface.
  * The target nucleophile **Cys-481** highlighted in yellow with the covalent linkage.
  * A 336-frame smooth movie sweeping through each internal rotatable bond of the inhibitor, showing the robotic articulation of the molecule inside the catalytic cleft without any bond stretching or distortion.
