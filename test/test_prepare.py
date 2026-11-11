from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from ms_adtools import pdbqt_to_rdkit, prepare_flexreceptor, prepare_ligand, prepare_receptor

DATA = Path(__file__).parent / "data"
SMILES = ["CC(=O)Nc1ccc(O)cc1", "c1ccc2c(c1)nc(n2CC(=O)N3CCOCC3)Cl", "CN1CCN(CC1)c1ccc(cc1)C(=O)[O-]"]


def _mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    return mol


@pytest.mark.parametrize("smiles", SMILES)
def test_ligand_roundtrip(smiles):
    mol = _mol(smiles)
    pdbqt = prepare_ligand(mol)
    assert pdbqt.startswith("REMARK SMILES ")
    assert pdbqt.rstrip().splitlines()[-1].startswith("TORSDOF")
    back = pdbqt_to_rdkit(pdbqt)
    assert Chem.MolToSmiles(Chem.RemoveHs(back)) == Chem.MolToSmiles(Chem.RemoveHs(mol))
    assert rdMolAlign.CalcRMS(Chem.RemoveHs(back), Chem.RemoveHs(mol)) < 1e-3


def test_ligand_salt_keeps_largest_fragment():
    mol = _mol("CC(=O)Nc1ccc(O)cc1")
    salt = Chem.AddHs(Chem.MolFromSmiles("[Na+].[Cl-]"))
    AllChem.EmbedMolecule(salt, randomSeed=1)
    merged = Chem.CombineMols(mol, salt)
    assert prepare_ligand(merged).count("ATOM") == prepare_ligand(mol).count("ATOM")


def _atoms(pdbqt):
    return [l for l in pdbqt.splitlines() if l.startswith(("ATOM", "HETATM"))]


def test_receptor_matches_reference():
    # ADT's prepare_receptor4.py output: the same atoms and types, Gasteiger charges close to its own
    # (RDKit charges the whole receptor at once, ADT residue by residue)
    new, ref = _atoms(prepare_receptor(DATA / "1iep_receptorH.pdb")), _atoms((DATA / "1iep.pdbqt").read_text())
    assert [l[12:27] + l[77:] for l in new] == [l[12:27] + l[77:] for l in ref]
    q = np.array([[float(a[70:76]), float(b[70:76])] for a, b in zip(new, ref)])
    assert abs(q[:, 0] - q[:, 1]).mean() < 0.03


def _as_pqr(atoms):
    return "".join("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f %7.4f %6.4f\n"
                   % (n, l[12:16].strip(), l[17:20], l[21], int(l[22:26]), *map(float, (l[30:38], l[38:46], l[46:54])),
                      0.01 * (n % 7 - 3), 1.5) for n, l in enumerate(atoms, 1))


def test_receptor_from_pqr_and_mol2(tmp_path):
    pdb = DATA / "1iep_receptorH.pdb"
    lines = [l for l in pdb.read_text().splitlines() if l.startswith("ATOM")]
    ref = _atoms(prepare_receptor(pdb))
    pqr = tmp_path / "r.pqr"
    pqr.write_text(_as_pqr(lines))
    kept = _atoms(prepare_receptor(pqr))  # the PQR's own charges, the non-polar H's merged into C
    assert [l[77:] for l in kept] == [l[77:] for l in ref]
    assert abs(sum(float(l[70:76]) for l in kept) - sum(0.01 * (n % 7 - 3) for n in range(1, len(lines) + 1))) < 0.05
    assert [l[70:] for l in _atoms(prepare_receptor(pqr, charges="gasteiger"))] == [l[70:] for l in ref]
    mol2 = tmp_path / "r.mol2"  # no BOND section: bonds from distances, the zero charges are no charges
    mol2.write_text("@<TRIPOS>MOLECULE\nr\n%d 0\n@<TRIPOS>ATOM\n" % len(lines) + "".join(
        "%d %s %s %s %s %s 1 %s%d 0.0\n" % (n, l[12:16].strip(), l[30:38], l[38:46], l[46:54], l[76:78].strip().title(),
                                          l[17:20], int(l[22:26])) for n, l in enumerate(lines, 1)))
    assert [l[77:] for l in _atoms(prepare_receptor(mol2))] == [l[77:] for l in ref]


def test_flexreceptor_splits_atoms():
    rigid, flex = prepare_flexreceptor(DATA / "1iep.pdbqt", "A:ILE360_THR315")
    assert flex.count("BEGIN_RES") == 2
    whole = (DATA / "1iep.pdbqt").read_text()
    count = lambda t: sum(l.startswith("ATOM") for l in t.splitlines())
    assert count(rigid) + count(flex) == count(whole)


def test_receptor_stray_atom_keeps_peptide_bonds():
    # 1iep_receptorH.pdb has SER A438 HB2 on line 2; stock ADT moved SER438 to the chain start
    pdbqt = prepare_receptor(DATA / "1iep_receptorH.pdb")
    atom = {(l[17:26], l[12:16].strip()): l for l in pdbqt.splitlines() if l.startswith("ATOM")}
    assert atom["SER A 438", "N"].split()[-1] == "N"  # not NA: bonded to MET437 C
    assert float(atom["PRO A 439", "N"][70:76]) < 0
    residues = [l[22:26] for l in pdbqt.splitlines() if l.startswith("ATOM")]
    assert residues.index(" 438") > residues.index(" 437")


def test_ligand_hydrate_adds_waters():
    pdbqt = prepare_ligand(_mol("CC(=O)Nc1ccc(O)cc1"), hydrate=True)
    # N-H, carbonyl O (two), hydroxyl O (two) and its H: meeko's HydrateMoleculeLegacy
    assert sum(l[77:].strip() == "W" for l in pdbqt.splitlines()) == 6
    assert all(" WAT " in l for l in pdbqt.splitlines() if l[77:].strip() == "W")


def test_ligand_open_macrocycles_cuts_the_ring():
    macro = "C1CCCCCCCCCCC1"
    closed = prepare_ligand(_mol(macro))
    opened = prepare_ligand(_mol(macro), open_macrocycles=True)
    assert "CG0" not in closed and closed.rstrip().endswith("TORSDOF 0")
    assert sum(l[77:].strip() == "CG0" for l in opened.splitlines()) == 2
    assert sum(l[77:].strip() == "G0" for l in opened.splitlines()) == 2
    assert opened.rstrip().endswith("TORSDOF 0")  # the closed molecule has no free torsion


def test_zinc_pseudo_atom():
    from ms_adtools.zinc import PARAMETERS, add_zinc_pseudo_atoms

    out = add_zinc_pseudo_atoms((DATA / "zn_site.pdbqt").read_text())
    tz = [l for l in out.splitlines() if l[77:].strip() == "TZ"]
    assert len(tz) == 1 and [l[30:54] for l in tz] == ["  19.113 132.964   3.689"]
    zn = [l for l in out.splitlines() if l[77:].strip().upper() == "ZN"]
    assert [float(l[70:76]) for l in zn] == [0.0]  # the zinc charge goes into the force field
    assert PARAMETERS.read_text().startswith("#")


def test_receptor_zinc_option_needs_no_second_pass():
    # the same TZ atom, straight out of prepare_receptor, with no intermediate file
    plain = prepare_receptor(DATA / "zn_site.pdbqt")
    assert not [l for l in plain.splitlines() if l[77:].strip() == "TZ"]
    for zinc in ("auto", True):
        out = prepare_receptor(DATA / "zn_site.pdbqt", zinc=zinc)
        assert [l[30:54] for l in out.splitlines() if l[77:].strip() == "TZ"] == \
               ["  19.113 132.964   3.689"]
    assert prepare_receptor(DATA / "1iep_receptorH.pdb", zinc="auto").count("TZ") == 0
    with pytest.raises(ValueError):
        prepare_receptor(DATA / "1iep_receptorH.pdb", zinc=True)


def test_flexreceptor_tree_as_adt_writes_it():
    with pytest.warns(UserWarning, match="PRO A 227 has no rotatable side chain"):
        rigid, flex = prepare_flexreceptor(DATA / "1iep.pdbqt", "A:ARG239_TYR232_PRO227")
    assert [l[:9] for l in flex.splitlines() if l.startswith("BEGIN_RES")] == ["BEGIN_RES"] * 2
    assert "BEGIN_RES ARG A 239\nREMARK  4 active torsions:" in flex  # in the order given
    assert "REMARK    7  A    between atoms: CZ   and  OH  " in flex  # torsions numbered across residues
    assert "REMARK       I    between atoms: NE   and  CZ  " in flex  # guanidinium C-N: listed, not a branch
    tree = [l.split()[0] for l in flex.split("BEGIN_RES TYR")[1].splitlines() if l.startswith(("ROOT", "BRANCH", "END"))]
    assert tree == ["ROOT", "ENDROOT", "BRANCH", "BRANCH", "BRANCH", "ENDBRANCH", "ENDBRANCH", "ENDBRANCH", "END_RES"]
    assert [l[12:16].strip() for l in _atoms(rigid) if l[17:26] == "ARG A 239"] == ["N", "HN", "C", "O"]
    assert [int(l[6:11]) for l in _atoms(rigid)] == list(range(1, len(_atoms(rigid)) + 1))


def _types_by_symbol(pdbqt):
    return [l[77:].strip() for l in pdbqt.splitlines() if l.startswith("ATOM")]


def test_ligand_custom_types_and_rigid_bonds():
    mol = _mol("COc1ccc(CCO)cc1")
    assert "OS" not in _types_by_symbol(prepare_ligand(mol))
    custom = prepare_ligand(mol, add_atom_types=[{"smarts": "[#8X2;!$([#8][#1])]", "atype": "OS"}])
    assert _types_by_symbol(custom).count("OS") == 1
    torsdof = lambda t: int(t.rstrip().split()[-1])
    assert torsdof(prepare_ligand(mol, rigidify_bonds=[("[c]-[CH2]", (0, 1))])) == torsdof(prepare_ligand(mol)) - 1


def test_reactive_ligand_orders():
    from ms_adtools.ligand import prepare_reactive_ligand
    acrylamide = _mol("C=CC(=O)Nc1ccccc1")
    (pdbqt,) = prepare_reactive_ligand(acrylamide, "[CH2]=[CH]-[C]=[O]")
    # the terminal CH2 reacts (1), its CH neighbour is 2, the carbonyl C 3
    assert sorted(t for t in _types_by_symbol(pdbqt) if t[-1].isdigit()) == ["C1", "C2", "C3"]
    with pytest.raises(ValueError):
        prepare_reactive_ligand(acrylamide, "[#16]")


def test_reactive_flexreceptor_and_config():
    from ms_adtools.reactive import reactive_config
    rigid, flex = prepare_flexreceptor(DATA / "1iep.pdbqt", "A:THR315", reactive=["A:CYS305"])
    assert flex.count("BEGIN_RES") == 2
    react = {l[12:16].strip(): l[77:].strip() for l in flex.splitlines() if l.startswith("ATOM") and l[77].isdigit()}
    assert react == {"SG": "1S4", "CB": "1C2", "CA": "1C3"}
    cfg = reactive_config(flex, "1iep_rigid").splitlines()
    assert cfg[1] == "fld 1iep_rigid.maps.fld"
    assert "intnbp_r_eps 1.800000 2.500000  13   7  1S4   C1" in cfg  # the 1-2 pair: 13-7 potential


def test_flexible_docking_output():
    from ms_adtools.poses import flexres_models
    # Vina 1.2 with A:THR315_LYS271 flexible; the residues number their atoms from 1 again
    out = (DATA / "1iep_flex_vina_out.pdbqt").read_text()
    mol = pdbqt_to_rdkit(out)
    assert mol.GetNumConformers() == 3 and mol.GetConformer(0).GetDoubleProp("score") == -10.972
    model1 = out.split("ENDMDL")[0].split("BEGIN_RES")[0]
    lig = np.array([[float(l[c:c + 8]) for c in (30, 38, 46)] for l in model1.splitlines() if l.startswith("ATOM")])
    xyz = mol.GetConformer(0).GetPositions()
    assert all(np.linalg.norm(xyz - p, axis=1).min() < 1e-3 for p in lig)  # no flexres atom took a ligand serial
    flex_only = flexres_models(out)
    assert flex_only.count("MODEL") == 3 and flex_only.count(" THR A 315 ") == 3 * 5  # CA CB CG2 OG1 HG1
    rigid, flex = prepare_flexreceptor(DATA / "1iep.pdbqt", "A:THR315_LYS271")
    whole = flexres_models(out, rigid).split("ENDMDL")[0].splitlines()
    count = lambda t: sum(l.startswith("ATOM") for l in t.splitlines())
    atoms = [l for l in whole if l.startswith("ATOM")]
    assert len(atoms) == count(rigid) + count(flex)
    thr = [l[12:16].strip() for l in atoms if l[17:26] == "THR A 315"]
    assert thr[:4] == ["N", "HN", "C", "O"] and "OG1" in thr  # the side chain follows its own backbone


def test_dlg_cluster_leads():
    from ms_adtools.poses import dlg_to_pdbqt
    pdbqt = prepare_ligand(_mol("CC(=O)Nc1ccc(O)cc1"))
    run = lambda n, e: ["DOCKED: MODEL %8d" % n, "DOCKED: USER    Run = %d" % n,
                        "DOCKED: USER    Estimated Free Energy of Binding    =  %.2f kcal/mol" % e] + [
        "DOCKED: " + l for l in pdbqt.splitlines()] + ["DOCKED: ENDMDL"]
    table = ["   1      1      2      -7.10      0.00      1.00           RANKING",
             "   1      2      3      -6.90      1.20      1.10           RANKING",
             "   2      1      1      -5.00      0.00      4.00           RANKING"]
    dlg = "\n".join(run(1, -5.0) + run(2, -7.1) + run(3, -6.9) + ["", "\tRMSD TABLE"] + table) + "\n"
    leads = pdbqt_to_rdkit(dlg_to_pdbqt(dlg))
    got = [(c.GetIntProp("run"), c.GetDoubleProp("score"), c.GetIntProp("cluster_size")) for c in leads.GetConformers()]
    assert got == [(2, -7.1, 2), (1, -5.0, 1)]
    assert pdbqt_to_rdkit(dlg_to_pdbqt(dlg, cluster_leads=False)).GetNumConformers() == 3


def test_ccd_template_types_and_ionizes():
    plain = prepare_receptor(str(DATA / "3pg.pdb"), repairs="checkhydrogens")
    ccd = prepare_receptor(str(DATA / "3pg.pdb"), repairs="checkhydrogens", ccd=DATA / "ccd")
    atoms = lambda t: [l for l in t.splitlines() if l.startswith(("ATOM", "HETATM"))]
    assert len({l[70:76] for l in atoms(plain)}) > 5  # its oxo bonds found: Gasteiger has parameters for all
    assert [l[:70] for l in atoms(plain)] == [l[:70] for l in atoms(ccd)]  # same atoms, only types/charges
    assert round(sum(float(l[70:76]) for l in atoms(ccd))) == -3  # carboxylate + phosphate ionized
    with pytest.warns(UserWarning, match="too few atoms"):  # names that do not match keep their values
        from ms_adtools.ccd import apply_ccd
        renamed = "\n".join(l[:12] + " X%-2d" % n + l[16:] for n, l in enumerate(atoms(plain)))
        assert apply_ccd(renamed, DATA / "ccd").splitlines() == renamed.splitlines()


def test_receptor_keeps_first_altloc_only():
    pdbqt = prepare_receptor(str(DATA / "altloc.pdb"), repairs="checkhydrogens")
    cys = [l for l in pdbqt.splitlines() if l.startswith("ATOM") and l[17:26] == "CYS A 733"]
    assert [l[12:16].strip() for l in cys].count("N") == 1  # ADT took altloc B as a second residue N
    assert [l[12:16].strip() for l in cys if l[77:].strip() == "HD"] == ["HN", "HG"]  # ADT gave it three N-terminal H


def test_ccd_from_the_mmcif_own_tables_and_metal_charge():
    from ms_adtools.ccd import template
    import gemmi
    pdbqt = prepare_receptor(str(DATA / "3pg_chem_comp.cif"), repairs="checkhydrogens", ccd=DATA / "nowhere")
    assert round(sum(float(l[70:76]) for l in pdbqt.splitlines() if l.startswith(("ATOM", "HETATM")))) == -3
    hec = gemmi.cif.read_string("""data_HEC
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
FE FE ? NA N -1 C1 C 0 C2 C 0 H1 H 0 H2 H 0 H3 H 0 H4 H 0 H5 H 0 H6 H 0
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
FE NA SING NA C1 SING NA C2 SING C1 H1 SING C1 H2 SING C1 H3 SING C2 H4 SING C2 H5 SING C2 H6 SING
""").sole_block()
    mol, names, metals = template(hec)
    assert "FE" not in names and metals == {"FE": 1}  # the N-1 bonded to the iron gave it +1


def test_residue_selection_reads_ranges_lists_and_names():
    rows = [l for l in (DATA / "1iep.pdbqt").read_text().splitlines() if l.startswith("ATOM")]
    from ms_adtools.receptor import select_residues

    def names(spec):
        return [" ".join(k[1].split()) for k in select_residues(rows, spec)]
    assert names("A:311-313") == ["PHE A 311", "TYR A 312", "ILE A 313"]
    assert names("A:239,315 ILE360") == ["ARG A 239", "THR A 315", "ILE A 360"]  # chain carries over
    assert names("A:ILE360_THR315") == ["ILE A 360", "THR A 315"]  # the old form still reads
    for bad in ("A:THR360", "Z:5", "A:x"):
        with pytest.raises(ValueError):
            select_residues(rows, bad)


def test_cli_writes_ligand_receptor_pair_and_poses(tmp_path):
    from ms_adtools.cli import main

    (tmp_path / "lib.smi").write_text("CC(=O)Nc1ccc(O)cc1 para\nCCO etoh\n")
    assert main(["ligand", str(tmp_path / "lib.smi")]) == 0
    assert sorted(p.name for p in (tmp_path / "lib").iterdir()) == ["etoh.pdbqt", "para.pdbqt"]
    assert main(["flexreceptor", str(DATA / "1iep.pdbqt"), "-r", "A:239,315", "-o", str(tmp_path / "x")]) == 0
    assert (tmp_path / "x_flex.pdbqt").read_text().count("BEGIN_RES") == 2
    assert main(["poses", str(DATA / "1iep_flex_vina_out.pdbqt"), "-o", str(tmp_path / "p.sdf")]) == 0
    assert (tmp_path / "p.sdf").read_text().count("$$$$") == 3
