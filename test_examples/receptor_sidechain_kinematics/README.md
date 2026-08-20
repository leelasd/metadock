# Receptor Side-Chain Kinematics ($\chi_1, \chi_2, \chi_3, \chi_4$ Articulation) Benchmark

This benchmark demonstrates **Robotic Receptor Side-Chain Kinematics** on **PDB 6DI9** (human Bruton's Tyrosine Kinase, BTK), parameterizing active-site amino acids as robotic kinematic arms rooted at the $C_\alpha$ backbone.

---

## 1. The Physics & Kinematics of Protein Side Chains

Instead of running heavy, expensive Cartesian MD on thousands of receptor atoms (which can distort secondary structure $\alpha$-helices and $\beta$-sheets), we parameterize active-site amino acids strictly by their **rotamer torsional joint angles**:

```
      Receptor Pocket Kinematics (e.g. Catalytic Lysine):
      
      [ Cα Backbone ] ─── (χ₁) ─── [ Cβ ] ─── (χ₂) ─── [ Cγ ] ─── (χ₃) ─── [ Cδ ] ─── (χ₄) ─── [ Nζ Head ]
       (Permanently                (N-CA-CB-CG)        (CA-CB-CG-CD)       (CB-CG-CD-CE)        (CG-CD-CE-NZ)
        100% Rigid)
```

### Standard IUPAC Chi Dihedral Angle Definitions:
* **$\chi_1$ (All Amino Acids except Gly/Ala):** $N - C_\alpha - C_\beta - C_\gamma / O_\gamma / S_\gamma$
* **$\chi_2$ (Leu, Ile, Met, Phe, Tyr, Trp, His, Asp, Asn, Glu, Gln, Lys, Arg):** $C_\alpha - C_\beta - C_\gamma - C_\delta / S_\delta / O_\delta$
* **$\chi_3$ (Met, Glu, Gln, Lys, Arg):** $C_\beta - C_\gamma - C_\delta - C_\epsilon / O_\epsilon / N_\epsilon$
* **$\chi_4$ (Lys, Arg):** $C_\gamma - C_\delta - C_\epsilon - N_\zeta / N_\eta$

---

## 2. Benchmark Validation Results

* **Active-Site Residues Articulated:** **15 amino acids** within $8.0\text{ \AA}$ of pocket centroid (31 total $\chi$ joint hinges).
  * **Hinge Gatekeeper:** `Met-477` ($\chi_1, \chi_2, \chi_3$)
  * **Hinge Acceptor:** `Glu-475` ($\chi_1, \chi_2, \chi_3$)
  * **Catalytic Lysine:** `Lys-536` ($\chi_1, \chi_2, \chi_3, \chi_4$)
  * **Aromatic Gating:** `Tyr-461`, `Tyr-476` ($\chi_1, \chi_2$)
  * **Hydrophobic Pocket Walls:** `Leu-457`, `Leu-528`, `Val-427`, `Val-458`, `Val-529`
* **Protein Backbone Distortion:** **$0.000000\text{ \AA}$** across all 60 frames (100% preserved crystal backbone).
* **Kinematic Side-Chain Adaptation:** Side chains flex smoothly to relieve steric clashes and optimize hydrogen bonds.

---

## 3. Files in This Directory

* [`receptor.pdb`](receptor.pdb): Cleaned BTK catalytic domain.
* [`cavity.prm`](cavity.prm): Active-site pocket definition.
* [`xtal_ligand.sdf`](xtal_ligand.sdf): Reference X-ray crystal inhibitor GJJ.
* [`receptor_sidechain_movie.pdb`](receptor_sidechain_movie.pdb): 60-frame multi-model PDB trajectory of active-site side chains flexing.
* [`run_sidechain_kinematics_demo.py`](run_sidechain_kinematics_demo.py): Automated generator for receptor kinematics.
* [`visualize_sidechains_pymol.pml`](visualize_sidechains_pymol.pml): PyMOL script for live 3D side-chain rotamer animation.

---

## 4. Running the Demo & PyMOL Visualization

```bash
cd test_examples/receptor_sidechain_kinematics

# Run the side-chain kinematics generator
python run_sidechain_kinematics_demo.py

# Open 3D multi-model movie in PyMOL
pymol visualize_sidechains_pymol.pml
```

#### What You Will See in PyMOL:
* **Slate Cartoon:** Protein Backbone (100% Rigid, $0.000\text{ \AA}$ distortion).
* **Orange Sticks:** Hinge Residues (`Glu-475`, `Met-477`) breathing with hydrogen bonds.
* **Yellow Sticks:** Catalytic Lysine (`Lys-536`, 4 Chi Joints).
* **Green Sticks:** Aromatic Gating residues (`Tyr-461`, `Tyr-476`).
* **Cyan Sticks:** Docked Kinase Inhibitor GJJ.
