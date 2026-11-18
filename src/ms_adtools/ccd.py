"""Types and charges of non-standard receptor residues from their CCD component.

prepare_receptor guesses a bare cofactor's or modified residue's chemistry; the Chemical Component
Dictionary has its bond orders and formal charges, keyed by the residue name. The component's
types and Gasteiger charges are computed as for a ligand and given to the residue's atoms by
name. Where meeko rejects a residue that does not match its template, the match here is lenient:
an acid hydrogen the structure lacks (phosphate, carboxylic, sulfonic) ionizes its acid, other
missing hydrogens fold their charge into their heavy atom (as merged non-polar ones do), a
hydrogen named otherwise takes a missing one of its heavy atom, one the template has no place
for is made inert (charge 0, type H), atoms the template does not know keep their values, and a
residue that cannot be matched keeps them all.
"""
import os
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem, rdBase

URL = "https://files.rcsb.org/ligands/download/{}.cif"
STANDARD = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "HID HIE HIP HSD HSE HSP CYX CYM ASH GLH LYN DA DC DG DT DU A C G U HOH WAT".split())
_BOND = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE,
         "QUAD": Chem.BondType.QUADRUPLE, "AROM": Chem.BondType.AROMATIC}
_METALS = frozenset(range(3, 5)) | frozenset(range(11, 14)) | frozenset(range(19, 32)) \
    | frozenset(range(37, 51)) | frozenset(range(55, 85))
_offline, _missing = False, set()  # per process: offline, one timeout; a name the RCSB lacks, one ask


def cache_dir():
    return Path(os.environ.get("MOLSUITE_HOME") or Path.home() / ".molsuite") / "cache" / "ccd"


def component_file(resname, where=None):
    """CIF of the CCD component ``resname``: ``where/<RESNAME>.cif`` if given (own or offline
    templates), else the cache, fetched once from the RCSB (components never change). None if absent."""
    global _offline
    code = resname.strip().upper()
    if where is not None:
        path = Path(where) / f"{code}.cif"
        return path if path.exists() else None
    path = cache_dir() / f"{code}.cif"
    if path.exists() or not code.isalnum():
        return path if path.exists() else None
    if _offline or code in _missing or os.environ.get("MOLSUITE_OFFLINE"):
        return None
    try:
        req = urllib.request.Request(URL.format(code), headers={"User-Agent": "ms_adtools"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = r.read()
    except urllib.error.HTTPError:
        _missing.add(code)
        return None
    except (urllib.error.URLError, OSError, ValueError):
        _offline = True
        return None
    if not data.startswith(b"data_"):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")  # atomic: parallel preparations fetch alike
    part.write_bytes(data)
    os.replace(part, path)
    return path


def _charge_from_valence(mol):
    for a in mol.GetAtoms():
        if a.GetSymbol() in ("N", "O"):
            a.SetFormalCharge(int(sum(b.GetBondTypeAsDouble() for b in a.GetBonds())) - {"N": 3, "O": 2}[a.GetSymbol()])


def template(block, code=None):
    """(RDKit molecule with every atom, hydrogens too, atom names, {metal name: charge}) of the
    chem_comp tables of a CIF ``block``: a CCD component, or the component ``code`` of an entry
    mmCIF that carries them. Metals are left out (Gasteiger has no parameters for them); their
    charge is the one their bonded atoms lost to them in the drawing (HEC: two N-1 -> Fe +2)."""
    s = gemmi.cif.as_string
    table = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "?charge", "?comp_id"])
    rows = [r for r in table if code is None or not r.has(3) or s(r[3]) == code]
    if not rows:
        raise ValueError(f"no chem_comp atoms for {code}")
    charge = lambda r: int(s(r[2])) if r.has(2) and s(r[2]) not in ("?", ".", "") else 0
    atoms = [(s(r[0]), s(r[1]).capitalize(), charge(r)) for r in rows]
    pt = Chem.GetPeriodicTable()
    metal = {a[0] for a in atoms if pt.GetAtomicNumber(a[1]) in _METALS}
    mol, index, links = Chem.RWMol(), {}, {}
    for name, element, q in atoms:
        if name not in metal:
            atom = Chem.Atom(element)
            atom.SetFormalCharge(q)
            atom.SetNoImplicit(True)  # the component lists every hydrogen
            index[name] = mol.AddAtom(atom)
    for r in block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order", "?comp_id"]):
        if code is not None and r.has(3) and s(r[3]) != code:
            continue
        x, y = s(r[0]), s(r[1])
        if x in index and y in index:
            mol.AddBond(index[x], index[y], _BOND.get(s(r[2]).upper(), Chem.BondType.SINGLE))
        elif {x, y} & metal and {x, y} - metal:
            links.setdefault(*({x, y} & metal), []).append(index.get(*({x, y} - metal)))
    mol = mol.GetMol()
    if not table.has_column(2):  # an entry mmCIF's tables have no formal charges
        _charge_from_valence(mol)
    quiet = rdBase.BlockLogs()  # noqa: F841 - the retry below answers RDKit's complaint
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # ponytail: inconsistent charges (HEC's O-1 that keeps its H): N/O charges from valence, aromatic bonds aside
        _charge_from_valence(mol)
        Chem.SanitizeMol(mol)
    metals = {m: -sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in links.get(m, ()) if i is not None)
              for m in metal}
    return mol, [a[0] for a in atoms if a[0] not in metal], metals


def smiles_template(smiles, names, metals):
    """template()'s triple from a ``templates.TEMPLATES`` entry: SMILES with its heavy atoms named."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # hydrogens go last, unnamed: _apply matches them by their atom
    return mol, list(names) + [None] * (mol.GetNumAtoms() - len(names)), dict(metals)


def _component(resname, where, own, comps):
    """Template of a residue: a file in ``where``, else the built-in pH-7 one, else the receptor
    mmCIF's own tables, else the CCD (cached); None if there is none."""
    from ms_adtools.templates import TEMPLATES
    path = component_file(resname, where) if where is not None else None
    if path is None and resname in TEMPLATES:
        return smiles_template(*TEMPLATES[resname])
    if path is None and resname in comps:
        return template(own, resname)
    path = path or (component_file(resname) if where is None else None)
    return template(gemmi.cif.read(str(path)).sole_block()) if path else None


#: Formal charges of bare metal ions, by residue name (FE is Fe(III), CU1 Cu(I)). Zinc is not
#: here: AutoDock4Zn was calibrated with it at 0 and its TZ pseudo-atoms carrying the binding.
ION_CHARGES = {"MG": 2, "MN": 2, "CA": 2, "FE": 3, "FE2": 2, "CU": 2, "CU1": 1}


def charge_ions(pdbqt):
    """Receptor PDBQT with each one-atom residue of ``ION_CHARGES`` given its formal charge.
    ponytail: a point charge, no coordination geometry; the protein's charges were computed without it."""
    lines = pdbqt.splitlines()
    rows = {}
    for n, l in enumerate(lines):
        if l.startswith(("ATOM", "HETATM")):
            rows.setdefault((l[21], l[22:27], l[17:20].strip()), []).append(n)
    for (_, _, resname), ns in rows.items():
        if len(ns) == 1 and resname in ION_CHARGES:
            l = lines[ns[0]]
            lines[ns[0]] = "%s%6.3f %-2s" % (l[:70], ION_CHARGES[resname], l[77:].strip())
    return "\n".join(lines) + "\n"


_ACID_H = Chem.MolFromSmarts("[#1]-[OX2]-[P,$([#6,#16]=O)]")  # H of a phosphate, carboxylic or sulfonic acid


def _apply(lines, mol, names, metals, metal_charges=False):
    """The residue's PDBQT ``lines`` retyped and recharged from the template; None if they do not match."""
    from ms_adtools.ligand import _charges, _types
    at = {n: i for i, n in enumerate(names)}
    name = lambda l: l[12:16].strip()
    is_h = lambda l: l[77:].strip() in ("H", "HD", "HS")
    hydrogens = lambda i: [x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors() if x.GetAtomicNum() == 1]
    heavy = [n for n, l in enumerate(lines) if not is_h(l)]
    matched = {n: at[name(lines[n])] for n in heavy
               if name(lines[n]) in at and mol.GetAtomWithIdx(at[name(lines[n])]).GetAtomicNum() > 1}
    if not heavy or len(matched) * 2 < len(heavy):
        return None
    xyz = np.array([[float(l[c:c + 8]) for c in (30, 38, 46)] for l in lines])
    hs = [n for n, l in enumerate(lines) if is_h(l)]
    parent = {n: matched.get(min(heavy, key=lambda h: np.linalg.norm(xyz[h] - xyz[n]))) for n in hs}
    taken = set()
    for n in hs:  # named as in the template and on the same atom
        i = at.get(name(lines[n]))
        if parent[n] is not None and i in hydrogens(parent[n]):
            matched[n] = i
            taken.add(i)
    extra = []
    for n in hs:  # named otherwise: a missing H of its atom; one the template has no room for is extra
        if n in matched or parent[n] is None:
            continue
        free = [i for i in hydrogens(parent[n]) if i not in taken]
        if free:
            matched[n] = free[0]
            taken.add(free[0])
        else:
            extra.append(n)
    present = set(matched.values())
    # an acid H the structure lacks on an atom it has: the acid is ionized (the CCD draws it neutral)
    drop = sorted({h for h, o, _ in mol.GetSubstructMatches(_ACID_H) if h not in taken and o in present}, reverse=True)
    rw = Chem.RWMol(mol)
    for h in drop:
        o = rw.GetAtomWithIdx(h).GetNeighbors()[0]
        o.SetFormalCharge(o.GetFormalCharge() - 1)
        rw.RemoveAtom(h)
    ionized = rw.GetMol()
    Chem.SanitizeMol(ionized)
    new = {i: k for k, i in enumerate(i for i in range(mol.GetNumAtoms()) if i not in drop)}
    types, charges = _types(ionized), _charges(ionized)
    q = {n: charges[new[i]] for n, i in matched.items()}
    owner = {i: n for n, i in matched.items()}
    for i in new:  # other template H the structure lacks: merged into their atom
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() == 1 and i not in taken and a.GetDegree():
            p = owner.get(a.GetNeighbors()[0].GetIdx())
            if p is not None:
                q[p] += charges[new[i]]
    out = list(lines)
    for n, i in matched.items():
        out[n] = "%s%6.3f %-2s" % (lines[n][:70], q[n], types[new[i]])
    if metal_charges:
        for n, l in enumerate(lines):
            if name(l) in metals:
                out[n] = "%s%6.3f %-2s" % (l[:70], metals[name(l)], l[77:].strip())
    for n in extra:  # ponytail: an H the chemistry has no place for (an aromatic N protonated by guess) is made inert, not deleted
        out[n] = "%s%6.3f %-2s" % (lines[n][:70], 0.0, "H")
    return out


def apply_ccd(pdbqt, where=None, structure=None, metal_charges=False):
    """Receptor PDBQT with each non-standard residue typed and charged from its chemical component:
    ``where/<RESNAME>.cif``, else the built-in pH-7 template, else the chem_comp tables of
    ``structure`` (the receptor's mmCIF), else the CCD (cached; not when ``where`` is given);
    residues without a usable component keep their values, with a warning when it cannot be
    applied. A residue whose net charge comes out fractional (atoms missing) is warned about too.
    ``metal_charges`` also gives a metal of the component (heme Fe) the charge its bonds took."""
    lines = pdbqt.splitlines()
    groups = {}
    for n, l in enumerate(lines):
        if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() not in STANDARD:
            groups.setdefault((l[21], l[22:27], l[17:20].strip()), []).append(n)
    own, comps = None, set()
    if structure is not None and str(structure).lower().endswith((".cif", ".mmcif", ".cif.gz")):
        own = gemmi.cif.read(str(structure)).sole_block()
        comps = set(own.find_values("_chem_comp_bond.comp_id")) & set(own.find_values("_chem_comp_atom.comp_id"))
    cache = {}
    for (chain, num, resname), rows in groups.items():
        if len(rows) == 1:  # ions: the single atom is right as it is
            continue
        where_ = f"{resname} {chain.strip()}{num.strip()}"
        if resname not in cache:
            try:
                cache[resname] = _component(resname, where, own, comps)
            except Exception as e:  # ponytail: an unusable component (odd valences) keeps the receptor's chemistry
                warnings.warn(f"CCD {resname}: {e}; the receptor's types and charges kept")
                cache[resname] = None
        if cache[resname] is None:
            continue
        try:
            got = _apply([lines[n] for n in rows], *cache[resname], metal_charges=metal_charges)
        except ValueError as e:  # an element AD4 has no type for (selenomethionine)
            warnings.warn(f"{where_}: {e}; the receptor's types and charges kept")
            continue
        if got is None:
            warnings.warn(f"{where_}: too few atoms named as in its CCD component; the receptor's kept")
            continue
        net = sum(float(l[70:76]) for l in got)
        if abs(net - round(net)) > 0.05:
            warnings.warn(f"{where_}: net charge {net:+.2f}, not a whole number: atoms of its component "
                          "are missing from the structure, so its charges are off by the remainder")
        for n, l in zip(rows, got):
            lines[n] = l
    return "\n".join(lines) + "\n"
