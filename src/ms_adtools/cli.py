"""ms_adtools command line: ligand, receptor, flexreceptor, poses."""
import argparse
import sys
import warnings
from pathlib import Path


def _ligands(path):
    """(name, 3D mol) of each molecule in an SDF/MOL/mol2 file, or of each SMILES line of a .smi."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    suffix = path.suffix.lower()
    if suffix in (".smi", ".smiles", ".txt"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            smiles, _, name = line.strip().partition(" ")
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) != 0:
                    mol = None
                else:
                    AllChem.MMFFOptimizeMolecule(mol)
            yield name.strip() or f"ligand_{n}", mol
    elif suffix == ".mol2":
        yield path.stem, Chem.MolFromMol2File(str(path), removeHs=False)
    else:
        for n, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), 1):
            yield (mol.GetProp("_Name").strip() if mol is not None and mol.HasProp("_Name") else "") \
                or f"ligand_{n}", mol


def _ligand(a):
    from ms_adtools import prepare_ligand
    mols = list(_ligands(a.input))
    single = len(mols) == 1
    out = Path(a.output or (a.input.with_suffix(".pdbqt") if single else a.input.with_suffix("")))
    if not single:
        out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, mol in mols:
        try:
            if mol is None:
                raise ValueError("RDKit could not read it")
            text = prepare_ligand(mol, open_macrocycles=a.macrocycles, hydrate=a.hydrate,
                                  flexible_amides=a.flexible_amides)
        except Exception as exc:  # one bad molecule does not stop a library
            failed += 1
            print(f"{name}: {exc}", file=sys.stderr)
            continue
        (out if single else out / f"{name}.pdbqt").write_text(text)
    return 1 if failed else 0


def _receptor(a):
    from ms_adtools import prepare_receptor
    text = prepare_receptor(a.input, repairs=a.repairs, charges=a.charges, cleanup=a.cleanup,
                            zinc="auto" if a.zinc else False, ccd=a.ccd)
    Path(a.output or a.input.with_suffix(".pdbqt")).write_text(text)
    return 0


def _flexreceptor(a):
    from ms_adtools import prepare_flexreceptor
    rigid, flex = prepare_flexreceptor(a.input, a.residues, reactive=a.reactive or ())
    stem = a.output or str(a.input.with_suffix(""))
    Path(stem + "_rigid.pdbqt").write_text(rigid)
    Path(stem + "_flex.pdbqt").write_text(flex)
    return 0


def _poses(a):
    from rdkit import Chem
    from ms_adtools import dlg_to_pdbqt, pdbqt_to_rdkit
    text = a.input.read_text()
    mol = pdbqt_to_rdkit(dlg_to_pdbqt(text) if a.input.suffix.lower() == ".dlg" else text)
    with Chem.SDWriter(str(a.output or a.input.with_suffix(".sdf"))) as writer:
        for conf in mol.GetConformers():
            for name, value in conf.GetPropsAsDict().items():
                mol.SetProp(name, str(value))
            writer.write(mol, confId=conf.GetId())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ms_adtools", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ligand", help="ligand PDBQT from an SDF, MOL, mol2 or SMILES (.smi) file")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", help="PDBQT file; a directory when the input holds several molecules")
    p.add_argument("--macrocycles", action="store_true", help="open macrocycles (Vina scoring only)")
    p.add_argument("--hydrate", action="store_true", help="add the W atoms of hydrated docking")
    p.add_argument("--flexible-amides", action="store_true", help="let amide bonds rotate")
    p.set_defaults(run=_ligand)

    p = sub.add_parser("receptor", help="receptor PDBQT from a PDB, mmCIF, PQR, PDBQT or mol2 file")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output")
    p.add_argument("--repairs", default="", help='"hydrogens": add them to residues that have none')
    p.add_argument("--charges", choices=("gasteiger", "keep"), help="default: keep a PQR/mol2's, else Gasteiger")
    p.add_argument("--cleanup", default="nphs_lps_waters_nonstdres", help="as prepare_receptor4.py -U")
    p.add_argument("--zinc", action="store_true", help="add AutoDock4Zn TZ pseudo-atoms")
    p.add_argument("--ccd", action="store_true", help="type non-standard residues from their CCD component")
    p.set_defaults(run=_receptor)

    p = sub.add_parser("flexreceptor", help="rigid and flexible PDBQT pair for flexible-residue docking")
    p.add_argument("input", type=Path)
    p.add_argument("-r", "--residues", required=True, help='e.g. "A:2-10,B:20", "A:239,315", "A:ARG239"')
    p.add_argument("--reactive", nargs="+", help='reactive residues, e.g. "A:SER203" "A:HIS57=NE2"')
    p.add_argument("-o", "--output", help="prefix: writes PREFIX_rigid.pdbqt and PREFIX_flex.pdbqt")
    p.set_defaults(run=_flexreceptor)

    p = sub.add_parser("poses", help="docked poses (PDBQT or DLG) to an SDF, one record per pose")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output")
    p.set_defaults(run=_poses)

    a = parser.parse_args(argv)
    warnings.formatwarning = lambda message, *_a, **_k: f"warning: {message}\n"
    try:
        return a.run(a)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ms_adtools {a.command}: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
