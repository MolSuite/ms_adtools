# ms_adtools

AutoDock preparation on RDKit and gemmi: ligand PDBQT, receptor PDBQT (PDB, mmCIF, PQR, PDBQT,
mol2), the rigid/flexible pair for flexible-residue docking as AutoDockTools writes it, and
docked poses (PDBQT, DLG) back to RDKit. Requires only `numpy`, `gemmi` and `rdkit`.

```bash
pip install ms_adtools
```

## Why

Performance. A virtual screen prepares molecules in many parallel workers, so what each worker
loads matters as much as how fast it runs. meeko, the reference preparation tool for AutoDock,
brings heavy dependencies (scipy, pandas, prody): importing it alone takes about 240 MB, and
preparing a receptor peaks above 350 MB. ms_adtools follows its ligand conventions (typing,
torsion tree, `REMARK SMILES` / `IDX` / `H PARENT`), so its ligand PDBQT reads the same, with numpy, gemmi
and RDKit as the only requirements: a worker that prepares a receptor peaks at about 80 MB.

## Command line

```bash
ms_adtools ligand lig.sdf -o lig.pdbqt              # SDF, MOL, mol2, or .smi (embedded in 3D)
ms_adtools ligand library.smi -o library/           # several molecules: one PDBQT each
ms_adtools receptor 1iep.pdb -o 1iep.pdbqt          # --repairs hydrogens, --zinc, --ccd, --charges
ms_adtools flexreceptor 1iep.pdbqt -r "A:239,315"   # writes 1iep_rigid.pdbqt + 1iep_flex.pdbqt
ms_adtools poses docked.pdbqt -o poses.sdf          # or a .dlg; one record per pose, with its score
```

`ms_adtools <command> -h` lists every option. A molecule that fails is reported on stderr and the
rest of the library goes on (exit code 1 if any failed).

## Selecting residues

`-r` / `residues=` takes chains and residue numbers, comma- or space-separated:

| selection | means |
|---|---|
| `A:2-10,B:20` | residues 2 to 10 of chain A, 20 of chain B |
| `A:2,4,6` | three residues of chain A (a chain holds until the next one) |
| `A:ARG239,THR315` | with names, which must then match |
| `100` / `100A` | any chain; `100` takes its insertion codes, `100A` only that one |

Residues with nothing to rotate (Gly, Ala) and Pro stay rigid, with a warning.

## Python API

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from ms_adtools import prepare_ligand, prepare_receptor, prepare_flexreceptor, pdbqt_to_rdkit

mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
AllChem.EmbedMolecule(mol)
pdbqt = prepare_ligand(mol)                         # str, with REMARK SMILES / IDX / H PARENT
receptor = prepare_receptor("1iep.pdb")             # str
rigid, flex = prepare_flexreceptor("1iep.pdbqt", "A:239,315")
poses = pdbqt_to_rdkit(docked_text)                 # one conformer per MODEL, `score` on each
```

| function | returns |
|---|---|
| `prepare_ligand(mol, open_macrocycles=False, hydrate=False, flexible_amides=False, ...)` | ligand PDBQT text |
| `prepare_reactive_ligand(mol, smarts, idx=0, ...)` | ligand PDBQT with the reactive atom typed |
| `prepare_receptor(path, repairs="", charges=None, cleanup=..., zinc=False, ccd=False)` | receptor PDBQT text |
| `prepare_flexreceptor(path, residues, reactive=())` | `(rigid, flex)` PDBQT texts |
| `pdbqt_to_rdkit(text)` | RDKit molecule, one conformer per pose |
| `dlg_to_pdbqt(text, cluster_leads=True)` | PDBQT poses of an AutoDock4/AutoDock-GPU DLG |
| `flexres_models(text, rigid=None)` | docked flexible residues as a multi-model PDB |
| `add_zinc_pseudo_atoms(pdbqt)` | receptor PDBQT with AutoDock4Zn TZ atoms |

`ms_adtools.reactive.reactive_config` writes the config of reactive docking;
`ms_adtools.zinc.PARAMETERS` is the `AD4Zn.dat` that autogrid4 needs for AutoDock4Zn maps.
Each function's docstring has the details.

## Speed and memory

One CPU, Python 3.12, RDKit 2026.09; each row is a fresh process.

| | wall | peak RSS |
|---|---|---|
| import | 0.12 s | 71 MB |
| 200 ligands (Enamine, 3D) | 0.29 s (1.5 ms each) | 72 MB |
| receptor (4uwh, 4905 atoms) | 0.16 s | 83 MB |

On `1iep_receptorH.pdb` the receptor has AutoDockTools' atoms and types (charges within 0.02 on
average), and `prepare_flexreceptor` writes its rigid file byte for byte. `benchmarks/` has the
script and the full measurements.

## License

MIT (`LICENSE`). The PDBQT conventions are AutoDockTools' (MGLTools, M. Sanner, TSRI) and the Forli
lab's; cite them with AutoDock. `src/ms_adtools/zinc/` carries AutoDock Vina's
`zinc_pseudo.py` (Apache-2.0) and the unmodified `AD4Zn.dat` parameter file (GPL-2.0-or-later
data), each with its own license (see its `NOTICE`).
