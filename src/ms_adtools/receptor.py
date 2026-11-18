"""Receptor PDBQT, and the rigid/flexible pair for flexible-residue docking, as text.

PDB and mmCIF are read by gemmi; PQR, PDBQT and mol2 line by line. Bonds come from distances,
bond orders and formal charges from the hydrogens the structure carries, and the Gasteiger
charges from RDKit over the whole receptor, so each residue's charges feel its neighbours. A
file that brings its own charges (PQR, PDBQT, mol2) keeps them unless Gasteiger is asked for.
Non-polar hydrogens are merged into their carbon; the AD4 types are ``ms_adtools.ligand``'s.
"""
import gzip
import re
import warnings
from collections import deque
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdPartialCharges

from ms_adtools.ccd import _METALS, STANDARD
from ms_adtools.ligand import _ad4_type

_GEMMI = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".ent.gz", ".cif.gz", ".mmcif.gz")
_WATERS = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP3", "SOL"})
_AD_ELEMENT = {"A": "C", "HD": "H", "HS": "H", "NA": "N", "NS": "N", "OA": "O", "OS": "O", "SA": "S",
               "CL": "Cl", "BR": "Br"}
_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "Se": 2, "B": 3, "P": 3, "F": 1, "Cl": 1, "Br": 1, "I": 1}
_HYPER = {"S": 6, "P": 5}  # may take double bonds to terminal O/S/N beyond their plain valence
# Kekulé double bonds of the standard residues, for those read without hydrogens
_DOUBLE = {"ARG": [("CZ", "NH2")], "ASN": [("CG", "OD1")], "ASP": [("CG", "OD1")], "GLN": [("CD", "OE1")],
           "GLU": [("CD", "OE1")], "HIS": [("CG", "CD2"), ("ND1", "CE1")],
           "PHE": [("CG", "CD1"), ("CE1", "CZ"), ("CD2", "CE2")], "TYR": [("CG", "CD1"), ("CE1", "CZ"), ("CD2", "CE2")],
           "TRP": [("CG", "CD1"), ("CD2", "CE2"), ("CE3", "CZ3"), ("CZ2", "CH2")]}
_BACKBONE = frozenset({"N", "C", "O", "OXT"})


def _atom(rec, name, resname, chain, resseq, icode, xyz, el, occ=1.0, b=0.0, q=None):
    return {"rec": rec, "name": name, "resname": resname, "chain": chain, "resseq": resseq, "icode": icode,
            "xyz": [float(v) for v in xyz], "el": "H" if el in ("D", "T") else el, "occ": occ, "b": b, "q": q}


def _read_gemmi(path):
    """Atoms of the first model, first alternate conformation only (ADT took every altloc as
    atoms of one residue). Residues go where most of their atoms are in the file, so one stray
    atom at the top of a PDB does not move its residue to the chain start."""
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    atoms = []
    for chain in st[0]:
        median = lambda r: sorted(a.serial for a in r)[len(r) // 2]
        for r in sorted(chain, key=median):
            for a in r:
                atoms.append(_atom("HETATM" if r.het_flag == "H" else "ATOM", a.name, r.name, chain.name,
                                   r.seqid.num, r.seqid.icode.strip(), a.pos.tolist(), a.element.name, a.occ, a.b_iso))
    return atoms, None


def _read_pdbqt(text):
    atoms = []
    for l in text.splitlines():
        if l.startswith(("ATOM", "HETATM")) and l[16] in " A" and l[77:79].strip() != "TZ":
            t = l[77:79].strip()
            atoms.append(_atom(l[:6].strip(), l[12:16].strip(), l[17:20].strip(), l[21].strip(), int(l[22:26]),
                               l[26].strip(), (l[30:38], l[38:46], l[46:54]), _AD_ELEMENT.get(t, t),
                               float(l[54:60] or 1), float(l[60:66] or 0), float(l[70:76])))
            atoms[-1]["t"] = t
    return atoms, None


def _read_pqr(text):
    atoms = []
    for l in text.splitlines():
        if l.startswith(("ATOM", "HETATM")):
            f = l.split()
            rec, name, resname = f[0], f[2], f[3]
            chain, f = (f[4], f[5:]) if len(f) == 11 else ("", f[4:])
            num, icode = re.fullmatch(r"(-?\d+)(\D?)", f[0]).groups()
            el = name.title() if name == resname and len(name) <= 2 else name.lstrip("0123456789")[0]
            atoms.append(_atom(rec, name, resname, chain, int(num), icode, f[1:4], el, q=float(f[4])))
    return atoms, None


def _read_mol2(text):
    atoms, bonds, ids, section = [], [], {}, None
    for l in text.splitlines():
        if l.startswith("@<TRIPOS>"):
            section = l[9:].strip()
            continue
        f = l.split()
        if section == "ATOM" and len(f) >= 6:
            el = f[5].split(".")[0]
            if el in ("Du", "LP", "Lp"):
                continue
            resname, num = re.fullmatch(r"(\D*?)(-?\d*)", f[7] if len(f) > 7 else "UNL1").groups()
            ids[f[0]] = len(atoms)
            atoms.append(_atom("ATOM" if resname in STANDARD else "HETATM", f[1], resname[:3] or "UNL", "",
                               int(num or 1), "", f[2:5], el, q=float(f[8]) if len(f) > 8 else None))
        elif section == "BOND" and len(f) >= 4 and f[1] in ids and f[2] in ids:
            bonds.append((ids[f[1]], ids[f[2]]))
    if all(not a["q"] for a in atoms):  # an all-zero charge column is no charge at all
        for a in atoms:
            a["q"] = None
    return atoms, bonds or None  # no BOND section: bonds from distances


def read_receptor(path):
    """(atoms, bonds or None) of a receptor file: PDB, mmCIF, PQR, PDBQT or mol2, gzip allowed."""
    name = str(path).lower()
    if name.endswith(_GEMMI):
        return _read_gemmi(path)
    text = (gzip.open if name.endswith(".gz") else open)(path, "rt").read()
    for ext, read in ((".pdbqt", _read_pdbqt), (".pqr", _read_pqr), (".mol2", _read_mol2)):
        if name.removesuffix(".gz").endswith(ext):
            return read(text)
    raise ValueError(f"unknown receptor format: {path}")


def _residue(a):
    return a["chain"], a["resseq"], a["icode"], a["resname"]


def _bonded(atoms, bonds):
    """RWMol of the atoms, bonded as in the file or by distance; metals stay unbonded (their
    links are not covalent for the charges) and a hydrogen keeps only its nearest heavy atom."""
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        atom = Chem.Atom(a["el"])
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, a["xyz"])
    mol.AddConformer(conf)
    if bonds is None:
        rdDetermineBonds.DetermineConnectivity(mol)
    else:
        for i, j in bonds:
            mol.AddBond(i, j, Chem.BondType.SINGLE)
    xyz = np.array([a["xyz"] for a in atoms])
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.GetAtomicNum() in _METALS:
            for n in [n.GetIdx() for n in atom.GetNeighbors()]:
                mol.RemoveBond(i, n)
        elif atom.GetAtomicNum() == 1 and atom.GetDegree() > 1:
            heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1] or \
                    [n.GetIdx() for n in atom.GetNeighbors()]
            keep = min(heavy, key=lambda n: np.linalg.norm(xyz[n] - xyz[i]))
            for n in [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != keep]:
                mol.RemoveBond(i, n)
    return mol


def _matching(adj):
    """A maximum set of bonds, no atom in two, among ``adj`` {atom: [atoms]}: per connected part,
    most constrained atom first. ponytail: exhaustive search capped per part, ample for residues."""
    pairs, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        part, todo = [], [start]
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                part.append(n)
                todo += adj[n]
        best, budget = [], [20000]

        def search(free, chosen):
            nonlocal best
            if len(chosen) > len(best):
                best = list(chosen)
            budget[0] -= 1
            if not free or budget[0] < 0 or len(chosen) + len(free) // 2 <= len(best):
                return
            n = min(free, key=lambda x: sum(m in free for m in adj[x]))
            free.discard(n)
            for m in adj[n]:
                if m in free:
                    free.discard(m)
                    chosen.append((n, m))
                    search(free, chosen)
                    chosen.pop()
                    free.add(m)
            search(free, chosen)
            free.add(n)

        search(set(part), [])
        pairs += best
    return pairs


def _perceive(mol, hydrogenated):
    """Bond orders and formal charges of the atoms in ``hydrogenated`` (residues that carry their
    hydrogens): the valence an atom still lacks pairs up into double bonds; a carbon left over is
    the C=N+ of a guanidinium/amidinium/imidazolium, an O/N left over is an anion and an S
    left over keeps an implicit hydrogen (a cysteine written without HG is a thiol, as for ADT)."""
    lack = {}
    for atom in mol.GetAtoms():
        i, s, deg = atom.GetIdx(), atom.GetSymbol(), atom.GetDegree()
        if i not in hydrogenated or s not in _VALENCE:
            continue
        if deg > _HYPER.get(s, _VALENCE[s]):  # an onium: R4N+, R3O+
            atom.SetFormalCharge(deg - _VALENCE[s])
        elif deg < _VALENCE[s]:
            lack[i] = _VALENCE[s] - deg

    def double(i, j):
        b = mol.GetBondBetweenAtoms(i, j)
        b.SetBondType(Chem.BondType.DOUBLE if b.GetBondType() == Chem.BondType.SINGLE else Chem.BondType.TRIPLE)
        for k in (i, j):
            lack[k] = lack.get(k, 0) - 1
            if not lack[k]:
                del lack[k]

    for _ in range(2):  # the second round makes the triple bonds
        adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in lack] for i in lack}
        for i, j in _matching(adj):
            double(i, j)
    for atom in mol.GetAtoms():  # S=O, P=O: a hypervalent atom takes its terminal O/S/N
        i, s = atom.GetIdx(), atom.GetSymbol()
        if i in hydrogenated and s in _HYPER:
            for n in atom.GetNeighbors():
                valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                if n.GetIdx() in lack and n.GetDegree() == 1 and valence + 1 <= _HYPER[s]:
                    lack[i] = lack.get(i, 0) + 1
                    double(i, n.GetIdx())
    for i, d in list(lack.items()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == "C":
            onium = [n for n in atom.GetNeighbors() if n.GetSymbol() == "N" and n.GetDegree() == 3
                     and n.GetFormalCharge() == 0 and n.GetIdx() not in lack]
            if onium:  # ponytail: the N with most hydrogens takes the + (NH2 of a guanidinium)
                n = max(onium, key=lambda n: sum(m.GetAtomicNum() == 1 for m in n.GetNeighbors()))
                double(i, n.GetIdx())
                n.SetFormalCharge(1)
            else:
                atom.SetNumRadicalElectrons(d)  # a missing atom; Gasteiger takes it as is
        elif atom.GetSymbol() == "S":
            atom.SetNoImplicit(False)
        else:
            atom.SetFormalCharge(-d)


def _standard_orders(mol, atoms, residues):
    """Double bonds of the standard residues read without hydrogens, by atom name."""
    index = {(_residue(atoms[i]), atoms[i]["name"]): i for i in residues}
    for i in residues:
        if atoms[i]["name"] == "C":
            o = index.get((_residue(atoms[i]), "O"))
            if o is not None and mol.GetBondBetweenAtoms(i, o):
                mol.GetBondBetweenAtoms(i, o).SetBondType(Chem.BondType.DOUBLE)
    for key in {_residue(atoms[i]) for i in residues}:
        for a, b in _DOUBLE.get(key[3], ()):
            i, j = index.get((key, a)), index.get((key, b))
            if i is not None and j is not None and mol.GetBondBetweenAtoms(i, j):
                mol.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)


def _oxo_orders(mol, residues):
    """Double bonds of non-standard residues read without hydrogens: the shortest terminal O (or S)
    of each P, trigonal C and S (two for S) is an oxo, so phosphates, sulfates, carboxylates
    and carbonyls charge as such; everything else stays single, its hydrogens implicit.
    ponytail: no aromaticity or C=C from geometry; ccd=True has the real bond orders."""
    pos = mol.GetConformer().GetPositions()
    for i in residues:
        atom = mol.GetAtomWithIdx(i)
        s, deg = atom.GetSymbol(), atom.GetDegree()
        if not (s in ("P", "S") and deg >= 3 or s == "C" and deg == 3):
            continue
        ends = sorted((n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() in ("O", "S") and n.GetDegree() == 1),
                      key=lambda j: np.linalg.norm(pos[i] - pos[j]))
        for j in ends[:2 if s == "S" and deg == 4 else 1]:
            mol.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)


def _chemistry(atoms, bonds, add_hydrogens):
    """Sanitized RDKit molecule of the receptor; atoms appended for added hydrogens."""
    mol = _bonded(atoms, bonds)
    with_h = {_residue(atoms[a.GetIdx()]) for a in mol.GetAtoms() if a.GetAtomicNum() == 1}
    hydrogenated = {i for i, a in enumerate(atoms) if _residue(a) in with_h}
    _perceive(mol, hydrogenated)
    bare = [i for i in range(len(atoms)) if i not in hydrogenated]
    _standard_orders(mol, atoms, [i for i in bare if atoms[i]["resname"] in STANDARD])
    _oxo_orders(mol, [i for i in bare if atoms[i]["resname"] not in STANDARD])
    for i in bare:
        if mol.GetAtomWithIdx(i).GetAtomicNum() not in _METALS:
            mol.GetAtomWithIdx(i).SetNoImplicit(False)  # the hydrogens it lacks are implicit
    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    standard = [i for i in bare if atoms[i]["resname"] in STANDARD]  # a cofactor's protonation is no guess
    if add_hydrogens and standard:
        mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=standard)
    return mol


def _gasteiger(mol):
    rdPartialCharges.ComputeGasteigerCharges(mol, throwOnParamFailure=False)
    # _GasteigerHCharge is set on every atom, but only means something on one with implicit hydrogens
    q = np.array([a.GetDoubleProp("_GasteigerCharge") + (a.GetDoubleProp("_GasteigerHCharge") if a.GetNumImplicitHs() else 0)
                  for a in mol.GetAtoms()])
    bad = ~np.isfinite(q)
    nonmetal = sorted({mol.GetAtomWithIdx(int(i)).GetSymbol() for i in np.flatnonzero(bad)
                       if mol.GetAtomWithIdx(int(i)).GetAtomicNum() not in _METALS})
    if nonmetal:
        warnings.warn(f"no Gasteiger parameters for {', '.join(nonmetal)}: charge 0")
    q[bad] = 0.0  # metals at 0, as ADT; ccd_metal_charges gives a component's metal its own
    return q


def _cleanup(atoms, cleanup, delete_single_nonstd_residues):
    if "waters" in cleanup:
        atoms = [a for a in atoms if a["resname"] not in _WATERS]
    if "lps" in cleanup:
        atoms = [a for a in atoms if not a["name"].upper().startswith("LP")]
    if delete_single_nonstd_residues:
        atoms = [a for a in atoms if a["resname"] in STANDARD]
    elif "nonstdres" in cleanup:  # chains with no standard residue go, unless no chain has one
        keep = {a["chain"] for a in atoms if a["resname"] in STANDARD and a["resname"] not in _WATERS}
        atoms = [a for a in atoms if a["chain"] in keep] if keep else atoms
    return atoms


def _line(serial, a, q, t):
    name = a["name"] if len(a["name"]) == 4 or len(a["el"]) == 2 else " " + a["name"]
    return ("%-6s%5d %-4s %3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f    %6.3f %-2s"
            % (a["rec"], serial % 100000, name, a["resname"], a["chain"][:1], a["resseq"], a["icode"][:1],
               *a["xyz"], a["occ"], a["b"], q, t))


def prepare_receptor(path, repairs="", charges=None, cleanup="nphs_lps_waters_nonstdres",
                     delete_single_nonstd_residues=False, zinc=False, ccd=False, ccd_metal_charges=False, ion_charges=False):
    """PDBQT text of a receptor file (PDB, mmCIF, PQR, PDBQT, mol2), with prepare_receptor4.py's options.

    ``charges``: None keeps the charges a PQR, PDBQT or mol2 brings and computes Gasteiger's for
    the rest; "gasteiger" always computes them; "keep" demands the file's. ``repairs`` with
    "hydrogens" adds the hydrogens of residues that have none ("checkhydrogens": only when the
    whole receptor has none); the structure's protonation is otherwise taken as it is.
    ``cleanup`` tokens: "nphs" merges non-polar hydrogens, "lps" drops lone pairs, "waters"
    waters, "nonstdres" the chains with no standard residue.

    ``zinc="auto"`` also places the AutoDock4Zn TZ pseudo-atoms (and zeroes the zinc charges)
    when the receptor has a zinc that can bind a ligand tetrahedrally, in the same pass, with
    no intermediate file; ``zinc=True`` demands at least one. It is off by default because a
    TZ atom only means something to autogrid4 with ``ms_adtools.zinc.PARAMETERS`` as the
    parameter file: the grid maps are where the force field changes, so that step is the
    caller's (the GPF needs the parameter file and the AD4Zn pair terms).

    ``ccd=True`` types and charges non-standard residues (cofactors, modified residues) from their
    CCD component, by bond orders, fetched once into ~/.molsuite/cache/ccd; ``ccd="dir"`` reads
    <RESNAME>.cif templates from a directory instead. An mmCIF receptor that carries its own
    chem_comp tables (PDBe updated mmCIF) is its own template. ``ccd_metal_charges`` also charges
    a component's metal (heme Fe +2). See ``ms_adtools.ccd``.

    ``ion_charges`` gives bare metal ions (MG, MN, CA, FE, FE2, CU, CU1) their formal charge
    instead of ADT's 0; zinc stays at 0 (AutoDock4Zn). Only the electrostatics change: a point
    charge has no coordination geometry. See ``ms_adtools.ccd.ION_CHARGES``."""
    atoms, bonds = read_receptor(path)
    if bonds is not None:  # ponytail: the cleanup drops atoms, so a mol2's own bonds give way to distances
        bonds = None if any(a["resname"] in _WATERS for a in atoms) or "lps" in cleanup else bonds
    atoms = _cleanup(atoms, cleanup, delete_single_nonstd_residues)
    if not atoms:
        raise ValueError(f"no receptor atoms left in {Path(str(path)).name}")
    own = all(a["q"] is not None for a in atoms)
    if charges == "keep" and not own:
        raise ValueError(f"{Path(str(path)).name} carries no charges to keep")
    if "t" in atoms[0]:  # a PDBQT: its non-polar hydrogens are merged, so its chemistry cannot be read again
        if charges == "gasteiger":
            raise ValueError("a PDBQT cannot be charged again (its non-polar hydrogens are merged): use the PDB")
        return _finish("\n".join(_line(n + 1, a, a["q"], a["t"]) for n, a in enumerate(atoms)) + "\n",
                       path, zinc, ccd, ccd_metal_charges, ion_charges)
    add_h = "hydrogens" in repairs and not ("checkhydrogens" in repairs and any(a["el"] == "H" for a in atoms))
    mol = _chemistry(atoms, bonds, add_h)
    q = np.array([a["q"] for a in atoms] + [0.0] * (mol.GetNumAtoms() - len(atoms))) \
        if own and charges != "gasteiger" else _gasteiger(mol)
    rows, added = list(atoms), {}
    for atom in list(mol.GetAtoms())[len(atoms):]:  # added hydrogens: named as ADT names them, written after their atom
        parent = atom.GetNeighbors()[0].GetIdx()
        added.setdefault(parent, []).append(atom.GetIdx())
        rows.append(dict(atoms[parent], name="HN" if atoms[parent]["name"] == "N" else "H" + atoms[parent]["name"][1:],
                         el="H", xyz=list(mol.GetConformer().GetAtomPosition(atom.GetIdx())), occ=1.0, b=0.0))
    for hs in added.values():
        if len(hs) > 1:
            for n, k in enumerate(hs, 1):
                rows[k]["name"] = (rows[k]["name"] + str(n))[-4:]
    merge = "nphs" in cleanup
    out, after = [], {}
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.GetAtomicNum() == 1 and atom.GetDegree() and atom.GetNeighbors()[0].GetSymbol() == "C" and merge:
            q[atom.GetNeighbors()[0].GetIdx()] += q[i]
            q[i] = np.nan
        elif i >= len(atoms):
            after.setdefault(atom.GetNeighbors()[0].GetIdx(), []).append(i)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if i >= len(atoms) or np.isnan(q[i]):
            continue
        for k in [i] + after.get(i, []):
            a = mol.GetAtomWithIdx(k)
            t = _ad4_type(a) or a.GetSymbol()
            out.append(_line(len(out) + 1, rows[k], q[k], t))
    return _finish("\n".join(out) + "\n", path, zinc, ccd, ccd_metal_charges, ion_charges)


def _finish(pdbqt, path, zinc, ccd, ccd_metal_charges, ion_charges):
    if ccd:
        from ms_adtools.ccd import apply_ccd
        pdbqt = apply_ccd(pdbqt, None if ccd is True else ccd, structure=path, metal_charges=ccd_metal_charges)
    if not zinc:
        return pdbqt
    from ms_adtools.zinc import add_zinc_pseudo_atoms  # only now, it is the rarer path
    tz = add_zinc_pseudo_atoms(pdbqt)
    if zinc is not True or any(l[77:].strip() == "TZ" for l in tz.splitlines()):
        return tz
    raise ValueError("no zinc in this receptor can take a ligand: no TZ pseudo-atom was placed")


def _bonds(rows):
    """Neighbour sets of a residue's PDBQT lines, from covalent radii; a hydrogen keeps its nearest atom."""
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in rows])
    el = [_AD_ELEMENT.get(l[77:79].strip(), l[77:79].strip()) for l in rows]
    table = Chem.GetPeriodicTable()
    r = np.array([table.GetRcovalent(e) for e in el])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    bond = (d < 1.25 * (r[:, None] + r[None])) & (d > 0.1)
    for i, e in enumerate(el):
        if e == "H" and bond[i].any():
            keep = np.argmin(np.where(bond[i], d[i], np.inf))
            bond[i] = bond[:, i] = False
            bond[i, keep] = bond[keep, i] = True
    return xyz, el, [set(np.flatnonzero(b).tolist()) for b in bond]


def _flexres(rows, serial, torsion):
    """(BEGIN_RES..END_RES lines, side-chain row indices, last torsion number) of one residue, as
    ADT writes it: CA alone is the ROOT, each side-chain bond out of a ring with atoms beyond it
    is a BRANCH, except a C-N bond to a planar carbon (amide, guanidinium), listed inactive.
    None when the residue has nothing to rotate or its side chain closes on the backbone (Pro)."""
    xyz, el, nbr = _bonds(rows)
    names = [l[12:16].strip() for l in rows]
    if "CA" not in names:
        return None
    ca = names.index("CA")
    rigid = {i for i, n in enumerate(names) if n in _BACKBONE}
    rigid |= {i for i in range(len(rows)) if el[i] == "H" and nbr[i] & rigid}
    side = set(range(len(rows))) - rigid
    if any(nbr[i] & rigid for i in side - {ca}):  # proline
        return None

    def planar_c(i):
        u = [(xyz[j] - xyz[i]) / np.linalg.norm(xyz[j] - xyz[i]) for j in nbr[i]]
        return el[i] == "C" and len(u) == 3 and \
            sum(np.degrees(np.arccos(np.clip(u[a] @ u[b], -1, 1))) for a, b in ((0, 1), (0, 2), (1, 2))) > 350

    active, inactive = set(), []
    for i in sorted(side):
        for j in sorted(nbr[i] & side):
            if i > j:
                continue
            ring = j in _component(nbr, side, i, (i, j))
            if ring or len(nbr[i]) < 2 or len(nbr[j]) < 2:  # a terminal atom turns nothing
                continue
            if el[i] + el[j] in ("CN", "NC") and planar_c(i if el[i] == "C" else j):
                inactive.append((i, j))
            else:
                active.add(frozenset((i, j)))
    if not active:
        return None
    cut = {tuple(sorted(b)) for b in active}
    body, remarks, ids = [], [], {}

    def write(start, parent):
        nonlocal torsion
        group = sorted(_component(nbr, side, start, *cut), key=lambda k: (k != start, k))
        for k in group:
            ids[k] = serial + len(ids)
            body.append(rows[k][:6] + "%5d" % ids[k] + rows[k][11:])
        if parent is None:
            body.append("ENDROOT")
        for k in group:
            for m in sorted(nbr[k] & side - set(group) - set(ids)):
                torsion += 1
                remarks.append("REMARK %4d  A    between atoms: %-4s and  %-4s" % (torsion, names[k], names[m]))
                head = "BRANCH %3d %3d" % (ids[k], serial + len(ids))
                body.append(head)
                write(m, k)
                body.append("END" + head)

    body.append("ROOT")
    write(ca, None)
    remarks += ["REMARK       I    between atoms: %-4s and  %-4s" % (names[i], names[j]) for i, j in inactive]
    label = "%s %s %s" % (rows[ca][17:20], rows[ca][21], rows[ca][22:27].strip())
    head = ["BEGIN_RES " + label, "REMARK %2d active torsions:" % len(active),
            "REMARK  status: ('A' for Active; 'I' for Inactive)"]
    return head + remarks + body + ["END_RES " + label], sorted(side), torsion


def _component(nbr, allowed, start, *cut):
    """Atoms of ``allowed`` reached from ``start`` without crossing the ``cut`` bonds (index pairs)."""
    cut = {tuple(sorted(b)) for b in cut}
    seen, todo = {start}, [start]
    while todo:
        n = todo.pop()
        for m in nbr[n] & allowed:
            if m not in seen and tuple(sorted((n, m))) not in cut:
                seen.add(m)
                todo.append(m)
    return seen


_TOKEN = re.compile(r"(?:(\w):)?([A-Za-z]{1,4})?(-?\d+)([A-Za-z]?)(?:-(-?\d+))?")


def select_residues(rows, spec, where="the receptor"):
    """(chain, PDB columns 18-27) of the residues a selection names, in its order.

    ``spec`` is comma- (or space-) separated: ``A:2-10,B:20``, ``A:2,4,6``, ``A:ARG239,THR315``.
    A chain applies until the next one names another; with none, any chain matches. A name, when
    given, must match too. A number takes its insertion codes (100, 100A); ``100A`` only that one."""
    keys, chain = [], ""
    for token in filter(None, re.split(r"[,\s_]+", spec)):  # "_": the old A:ILE360_THR315 form
        got = _TOKEN.fullmatch(token)
        if got is None:
            raise ValueError(f"cannot read residue {token!r} (e.g. A:2-10,B:20 or A:ARG239)")
        chain = got.group(1) or chain
        name, lo, icode, hi = got.group(2), int(got.group(3)), got.group(4), got.group(5)
        hi = int(hi) if hi else lo
        found = [(l[21], l[17:27]) for l in rows
                 if (not chain or l[21] == chain) and lo <= int(l[22:26]) <= hi
                 and (not name or l[17:20].strip() == name.upper()) and (not icode or l[26] == icode)]
        if not found:
            shown = token if got.group(1) or not chain else f"{chain}:{token}"
            raise ValueError(f"residue {shown} not found in {where}")
        keys += list(dict.fromkeys(found))
    return keys


def prepare_flexreceptor(path, residues=(), reactive=()):
    """(rigid PDBQT, flexible PDBQT) of a receptor, as prepare_flexreceptor4.py writes them.

    ``path`` is a receptor PDBQT, or any file ``prepare_receptor`` reads (prepared first, with
    its defaults). ``residues`` is a selection, ``"A:2-10,B:20"``, ``"A:239,315"`` or
    ``"A:ARG239,THR315"`` (see ``select_residues``), or a list of them; written in the order given. A residue with nothing to rotate (Gly, Ala)
    or a ring on its backbone (Pro) stays rigid, with a warning. ``reactive`` residues
    ("A:SER203", "A:HIS57=NE2") are flexible too and get the reactive types of AutoDock-GPU
    reactive docking; ``ms_adtools.reactive.reactive_config`` writes its config."""
    p = Path(str(path))
    text = p.read_text() if p.suffix.lower() == ".pdbqt" else prepare_receptor(p)
    rows = [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))]
    residues = [residues] if isinstance(residues, str) else list(residues)
    reactive = [reactive] if isinstance(reactive, str) else list(reactive)
    wanted = list(dict.fromkeys(k for spec in residues + [r.partition("=")[0] for r in reactive]
                                for k in select_residues(rows, spec, p.name)))  # each item its own chain
    flex, gone, serial, torsion = [], set(), 1, 0
    for key in wanted:
        at = [n for n, l in enumerate(rows) if (l[21], l[17:27]) == key]
        res = _flexres([rows[n] for n in at], serial, torsion)
        if res is None:
            warnings.warn(f"{' '.join(key[1].split())} has no rotatable side chain: kept rigid")
            continue
        block, side, torsion = res
        flex += block
        serial += len(side)
        gone |= {at[k] for k in side}
    rigid = [l for n, l in enumerate(rows) if n not in gone]
    rigid = "\n".join(l[:6] + "%5d" % (n % 100000) + l[11:] for n, l in enumerate(rigid, 1)) + "\n"
    flex = "\n".join(flex) + "\n" if flex else ""
    if not reactive:
        return rigid, flex
    from ms_adtools.reactive import reactive_flexres
    return rigid, reactive_flexres(flex, reactive)
