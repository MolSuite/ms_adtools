# ms_adtools benchmark

Measured 2026-11-08 and 2026-11-10 on one machine: Python 3.12, RDKit 2026.09.1, numpy 2.5.4,
gemmi 0.7.6. meeko 0.8.0, the reference tool for AutoDock preparation, is measured alongside on
the same inputs. Reproduce with `python benchmarks/bench.py`; each row runs in a fresh
subprocess, so RSS is a clean-interpreter number.

Ligands: 200 Enamine_D10k molecules, hydrogens added, ETKDG + MMFF.
Receptor: `4uwh.pdb` (4905 atoms, `BENCH_PDB=.../4uwh.pdb`), read from the PDB file by both
(meeko with `allow_bad_res=True`, `default_altloc="A"`).

## Time and memory

All single-threaded.

| measurement | wall s | RSS growth MB | peak RSS MB |
|---|---|---|---|
| `from rdkit import Chem` alone | 0.03 | 30 | 42 |
| import ms_adtools, ligand path | 0.12 | 44 | 56 |
| import ms_adtools, receptor path | 0.12 | 59 | 71 |
| import meeko | 1.31 | 231 | 243 |
| 200 ligands, ms_adtools | 0.29 | 2 | 72 |
| 200 ligands, meeko | 0.43 | 1 | 268 |
| receptor, ms_adtools | 0.16 | 12 | 83 |
| receptor, meeko (`Polymer.from_pdb_file`) | 2.6 | 110 | 354 |

Opening macrocycles takes the 200 ligands to 0.31 s and hydration to 0.36 s. The number that
mattered for us is the per-worker footprint: a worker that only prepares receptors peaks at
about 80 MB.

## Output

`prepare_ligand` follows meeko's ligand conventions; over the same 200 ligands, in each mode,
its PDBQT matches meeko's:

| | rigid rings | open macrocycles | hydrated |
|---|---|---|---|
| same atoms written | 200/200 | 200/200 | 200/200 |
| same atom types | 200/200 | 189/200 | 200/200 |
| max charge difference | 0.000 | 0.000 | 0.000 |
| same TORSDOF | 200/200 | 200/200 | 200/200 |
| same number of BRANCHes | 200/200 | 200/200 | 200/200 |
| same W / G / CG counts | 200/200 | 200/200 | 200/200 |
| pose back to the input SMILES | 200/200 | 200/200 | 200/200 |

The 11 differences with open macrocycles are the choice of which C–C bond of the ring is cut
when several cuts give an equally shallow tree: the same number of breaks and the same TORSDOF,
one bond over.

The receptor follows AutoDockTools' conventions: bonds by distance (`DetermineConnectivity`),
bond orders and formal charges from the hydrogens the structure carries, Gasteiger over the
whole receptor. Against `prepare_receptor4.py` on `1iep_receptorH.pdb` it writes the same 2702
atoms with the same types, charges within 0.020 on average (ADT charges residue by residue);
`prepare_flexreceptor` writes ADT's rigid file byte for byte and the same torsion tree.

## Scope

What each covers, for choosing between them:

| | ms_adtools | meeko |
|---|---|---|
| ligand PDBQT, `REMARK SMILES` / `IDX` / `H PARENT` | yes | yes |
| pose back to an RDKit molecule | yes | yes |
| receptor PDBQT | yes, by distances and hydrogens; CCD templates for non-standard residues | yes, by residue templates |
| flexible residues | yes | yes |
| macrocycle opening, hydrated docking, flexible amides | yes | yes |
| reactive docking | yes | yes |
| covalent docking | no | yes |
| AutoDock4 DLG poses | yes | yes |
| AutoDock4Zn TZ pseudo-atoms | yes | no |
| receptor input | PDB, mmCIF, PQR, PDBQT, mol2 | PDB, PQR, mmCIF, prody selection |
| required imports | rdkit, numpy, gemmi | rdkit, numpy, scipy, gemmi, pandas, prody |
