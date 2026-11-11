"""RDKit molecule -> AutoDock ligand PDBQT.

The chemistry is RDKit's: rings, aromaticity, Gasteiger charges. Atom types, rotatable bonds,
the root, macrocycle opening and hydration follow meeko's rules, so a ligand docks as a
meeko-prepared one; the torsion tree and the lines are those AutoDock reads. The PDBQT
carries meeko's REMARK SMILES / SMILES IDX / H PARENT, so a pose goes back to RDKit with
``ms_adtools.poses.pdbqt_to_rdkit`` (or meeko's ``RDKitMolCreate``).
"""
from itertools import count

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

# typed as their own element
_AS_ELEMENT = {"B", "F", "Mg", "Si", "P", "Cl", "Ca", "Mn", "Fe", "Zn", "Br", "I"}


def _lone_pair_taken(n):
    """A nitrogen that does not accept a hydrogen bond: a cation, or a three-connected amine
    whose lone pair conjugates (an aromatic or sp2-carbon neighbour, or an N=X neighbour)."""
    if n.GetFormalCharge() == 1:
        return True
    if n.GetTotalDegree() != 3 or n.GetTotalValence() != 3:
        return False
    return any(a.GetIsAromatic()
               or (a.GetAtomicNum() == 6 and a.GetTotalDegree() == 3 and a.GetTotalValence() == 4)
               or (a.GetAtomicNum() == 7 and a.GetTotalDegree() == 2
                   and any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds()))
               for a in n.GetNeighbors())


def _ad4_type(atom):
    """AutoDock4 type of one atom, None when AutoDock has none for its element."""
    s = atom.GetSymbol()
    if s == "H":
        return "HD" if any(a.GetAtomicNum() in (7, 8, 9, 15, 16) for a in atom.GetNeighbors()) else "H"
    if s == "C":
        return "A" if atom.GetIsAromatic() else "C"
    if s == "N":
        return "N" if _lone_pair_taken(atom) else "NA"
    if s == "O":
        return "OA"
    if s == "S":  # a thioether/thiol S accepts; aromatic or oxidised S does not
        return "SA" if not atom.GetIsAromatic() and atom.GetTotalDegree() == 2 else "S"
    return s if s in _AS_ELEMENT else None


_AMIDE = Chem.MolFromSmarts("[NX3]-[CX3]=[O,N,S]")  # amidines too
_TERTIARY_AMIDE = Chem.MolFromSmarts("[NX3]([!#1])([!#1])-[CX3]=[O,N,S]")
_ALKYNE = Chem.MolFromSmarts("[*]-[*]#[*]-[*]")
_TERMINAL_ALKYNE = Chem.MolFromSmarts("[*]-[*]#[*X1]")
MACROCYCLE_SIZES = (7, 33)
MAX_BREAKS = 4
HB_LENGTH = 3.0  # water to acceptor; a donor H gets its water at HB_LENGTH - 1


def _key(i, j):
    return (i, j) if i < j else (j, i)


def _types(mol, extra=()):
    """AD4 types; ``extra`` are meeko's add_atom_types entries {"smarts", "atype", "IDX": [0-based]}."""
    types = [_ad4_type(a) for a in mol.GetAtoms()]
    for patt, t, idx in [(Chem.MolFromSmarts(e["smarts"]), e["atype"], e.get("IDX", (0,))) for e in extra]:
        if patt is None:
            raise ValueError(f"bad SMARTS for atom type {t}")
        for m in mol.GetSubstructMatches(patt, uniquify=False):
            for i in idx:
                types[m[i]] = t
    missing = sorted({mol.GetAtomWithIdx(i).GetSymbol() for i, t in enumerate(types) if t is None})
    if missing:
        raise ValueError(f"no AutoDock type for element(s) {', '.join(missing)}")
    return types


def _charges(mol):
    rdPartialCharges.ComputeGasteigerCharges(mol)
    q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()])
    if not np.isfinite(q).all():
        bad = sorted({mol.GetAtomWithIdx(int(i)).GetSymbol() for i in np.flatnonzero(~np.isfinite(q))})
        raise ValueError(f"no Gasteiger charge for {', '.join(bad)}")
    return q


def _rotatable(mol, flexible_amides, rigid=()):
    """Single bonds that may rotate, before rings are taken out (meeko's BondTyperLegacy)."""
    rot = {_key(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
           if b.GetBondType() == Chem.BondType.SINGLE}
    if not flexible_amides:
        amides = {_key(m[0], m[1]) for m in mol.GetSubstructMatches(_AMIDE)}
        rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        for n, r1, r2, c in (m[:4] for m in mol.GetSubstructMatches(_TERTIARY_AMIDE)):
            if rank[r1] != rank[r2]:  # meeko lets a tertiary amide with two different R rotate
                amides.discard(_key(n, c))
        rot -= amides
    for i, j, k, m in mol.GetSubstructMatches(_ALKYNE):
        rot -= {_key(i, j), _key(k, m)}
        rot.add(_key(j, k))
    rot -= {_key(m[0], m[1]) for m in mol.GetSubstructMatches(_TERMINAL_ALKYNE)}
    for smarts, (i, j) in rigid:  # (SMARTS, (i, j)): bond between match atoms i and j, 0-based
        rot -= {_key(m[i], m[j]) for m in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=False)}
    return rot


def _ring_bonds(mol, broken=()):
    """Bonds still in a ring once ``broken`` are cut."""
    m = Chem.RWMol(mol)
    for i, j in broken:
        m.RemoveBond(i, j)
    Chem.FastFindRings(m)
    ri = m.GetRingInfo()
    return {_key(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds() if ri.NumBondRings(b.GetIdx())}


def _model(mol, keep, rot, broken, glue):
    """Rigid bodies of the written atoms: (members per body, {(body, body): (atom, atom)}, root)."""
    rot = rot - _ring_bonds(mol, broken)
    parent = {i: i for i in keep}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = [_key(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edges = [e for e in edges if e[0] in parent and e[1] in parent and e not in broken]
    for i, j in edges:
        if (i, j) not in rot:
            parent[find(i)] = find(j)
    body_of = {i: find(i) for i in keep}
    members = {}
    for i in sorted(keep):
        members.setdefault(body_of[i], []).append(i)
    links = {}
    for i, j in edges:
        if (i, j) in rot:
            a, b = body_of[i], body_of[j]
            links[a, b], links[b, a] = (i, j), (j, i)
    graph = {b: [] for b in members}
    for a, b in links:
        graph[a].append(b)
    # root: peel leaves until one or two bodies are left, then the larger (meeko)
    g = {b: set(n) for b, n in graph.items()}
    while len(g) > 2:
        leaves = [b for b, n in g.items() if len(n) <= 1]
        for leaf in leaves:
            for n in g.pop(leaf):
                g[n].discard(leaf)
    root = max(sorted(g), key=lambda b: len(members[b]))
    # a bond to a lone terminal atom rotates nothing: merge that atom into its neighbour body
    for b in [b for b in members if len(members[b]) == 1 and len(graph[b]) == 1 and b != root
              and members[b][0] not in glue]:
        host = graph[b][0]
        members[host].append(members.pop(b)[0])
        graph[host].remove(b)
        del graph[b], links[b, host], links[host, b]
    return members, graph, links, root


def _weighted_depth(members, graph, node, glue, depth=0, seen=None):
    """meeko's flexibility score: deep, heavy branches cost more."""
    seen = seen if seen is not None else set()
    seen.add(node)
    here = depth ** 2 * (len(members[node]) - 1 + sum(i in glue for i in members[node]))
    best = here
    for n in graph[node]:
        if n not in seen:
            best = max(best, here + _weighted_depth(members, graph, n, glue, depth + 1, seen))
    return best


def _break_macrocycle(mol, types, rot, sizes=MACROCYCLE_SIZES, allow_A=False):
    """Bonds to cut so every ring of ``sizes`` opens: the most cuts, each atom in one cut, C-C (or with
    ``allow_A`` also aromatic-carbon) single bonds outside small rings, then the shallowest tree
    (meeko's FlexMacrocycle)."""
    ri = mol.GetRingInfo()
    rings = [set(_key(mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx())
                 for b in ring) for ring in ri.BondRings()]
    small = set().union(*[r for r in rings if len(r) < sizes[0]])
    big = [r for r in rings if sizes[0] <= len(r) <= sizes[1]]
    cut = ("C", "A") if allow_A else ("C",)
    candidates = sorted({b for r in big for b in r
                         if b in rot and b not in small and types[b[0]] in cut and types[b[1]] in cut})
    combos = set()

    def grow(cut):
        if len(cut) == MAX_BREAKS:
            return
        used = {a for b in cut for a in b}
        intact = [r for r in big if not r & set(cut)]
        for b in candidates:
            if b[0] not in used and b[1] not in used and any(b in r for r in intact):
                combos.add(tuple(sorted(cut + [b])))
                if len(intact) > 1:
                    grow(cut + [b])
    grow([])
    if not combos:
        return ()
    most = max(map(len, combos))
    return [c for c in sorted(combos) if len(c) == most]


def _water_sites(mol, types, xyz):
    """(anchor atom, position) of the W pseudo-atoms for hydrated docking (meeko's
    HydrateMoleculeLegacy: acceptors and polar Hs, by neighbour count)."""
    unit = lambda v: v / np.linalg.norm(v)
    out = []
    for a in mol.GetAtoms():
        t, i = types[a.GetIdx()], a.GetIdx()
        nb = [n.GetIdx() for n in a.GetNeighbors()]
        o = xyz[i]
        if t not in ("HD", "OA", "SA", "NA"):
            continue
        if t == "HD" or (t == "NA" and len(nb) == 1):
            sites = [o + unit(o - xyz[nb[0]]) * (HB_LENGTH - 1.0)]
        elif t == "NA" and len(nb) in (2, 3):
            ends = xyz[nb] if len(nb) == 2 else o + np.array([unit(xyz[n] - o) for n in nb])
            sites = [o + unit(o - ends.mean(axis=0)) * HB_LENGTH]
        elif t in ("OA", "SA") and len(nb) == 1:  # sp2: two waters in the plane, +-120 deg
            far = [n.GetIdx() for n in mol.GetAtomWithIdx(nb[0]).GetNeighbors() if n.GetIdx() != i]
            if not far:
                continue
            axis = o + unit(np.cross(o - xyz[nb[0]], xyz[far[0]] - xyz[nb[0]]))
            sites = [_rotate(xyz[nb[0]], o, axis, ang) for ang in np.radians((-120, 120))]
            sites = [o + unit(s - o) * HB_LENGTH for s in sites]
        elif t in ("OA", "SA") and len(nb) == 2:  # sp3: two waters, +-60 deg about the lone-pair plane
            v1, v2 = (o + unit(xyz[n] - o) for n in nb)
            axis, start = o + unit(v2 - v1), o + unit(o - (v1 + v2) / 2)
            sites = [o + unit(_rotate(start, o, axis, ang) - o) * HB_LENGTH for ang in np.radians((-60, 60))]
        else:
            raise ValueError(f"cannot place waters on atom {i} ({t}) with {len(nb)} neighbours")
        out += [(i, s) for s in sites]
    return out


def _rotate(p, p1, p2, angle):
    """``p`` rotated by ``angle`` about the axis p1 -> p2."""
    x, y, z = (p2 - p1) / np.linalg.norm(p2 - p1)
    c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    r = np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                  [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                  [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])
    return (p - p1) @ r + p1


def _line(serial, name, xyz, charge, atype):
    name = name if len(name) == 4 or (len(name) > 1 and name[1].islower()) else " " + name
    return ("ATOM  %5d %-4s UNL     1    %8.3f%8.3f%8.3f  1.00  0.00    %6.3f %-2s"
            % (serial, name, *xyz, charge, atype))


def _remark_lines(prefix, pairs):
    """meeko's break_long_remark_lines: pairs appended while the line stays under 79 chars."""
    out = [prefix]
    for s in (" %d %d" % p for p in pairs):
        if len(out[-1]) + len(s) < 79:
            out[-1] += s
        else:
            out.append(prefix + s)
    return out


def _smiles_remarks(mol, written):
    """REMARK SMILES / SMILES IDX / H PARENT for ``written`` [(atom index, serial)], as meeko writes them."""
    for a in mol.GetAtoms():
        a.SetIntProp("_adt_i", a.GetIdx())
    noh = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(noh)
    order = list(noh.GetProp("_smilesAtomOutputOrder", autoConvert=True))
    pos = {noh.GetAtomWithIdx(k).GetIntProp("_adt_i"): n + 1 for n, k in enumerate(order)}
    idx, h_parent = [], []
    for i, serial in written:
        if i in pos:
            idx.append((pos[i], serial))
        else:  # a polar H RemoveHs dropped
            h_parent.append((pos[mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()], serial))
    return (["REMARK SMILES " + smiles] + _remark_lines("REMARK SMILES IDX", idx)
            + _remark_lines("REMARK H PARENT", h_parent))


def _whole(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:  # ponytail: salts/solvents dropped, largest fragment docked (as meeko demands one)
        mol = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    mol = Chem.Mol(mol)
    if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)
    return mol


def prepare_ligand(mol, conf_id=-1, open_macrocycles=False, hydrate=False, flexible_amides=False,
                   smiles_remarks=True, add_atom_types=(), rigidify_bonds=(), macrocycle_sizes=MACROCYCLE_SIZES,
                   macrocycle_allow_A=False, reactive_atom=None):
    """PDBQT text of an RDKit molecule with a 3D conformer.

    Missing hydrogens are added with coordinates; of a salt only the largest fragment is kept.
    Non-polar hydrogens are merged into their carbon. ``open_macrocycles`` cuts each ring of
    ``macrocycle_sizes`` (7-33) into CG/G closure atoms, which only Vina's own scoring functions
    close again; leave it off for any other program. ``hydrate`` adds the W atoms of AutoDock4
    hydrated docking.

    ``add_atom_types`` are meeko's entries, applied over the standard types:
    [{"smarts": "[#7]-[#1]", "atype": "N", "IDX": [0]}] (IDX 0-based, default [0]).
    ``rigidify_bonds`` freezes bonds: [("[C]=[C]-[C]=[C]", (1, 2))], atoms of the match, 0-based.
    ``reactive_atom`` (an atom index of ``mol``, a single fragment) gives it and its neighbours the reactive types of
    AutoDock-GPU reactive docking; ``prepare_reactive_ligand`` finds it by SMARTS.

    Not ported from meeko: double_bond_penalty (unused there), keep_chorded/equivalent_rings
    (exhaustive ring search) and tethering (covalent docking; reactive docking is here instead)."""
    mol = _whole(mol)
    xyz = mol.GetConformer(conf_id).GetPositions()
    types = _types(mol, add_atom_types)
    charge = _charges(mol)
    keep = [i for i, t in enumerate(types) if t != "H"]
    for i, t in enumerate(types):
        if t == "H":  # non-polar H: its charge goes to the atom it is bonded to
            charge[mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()] += charge[i]

    rot = _rotatable(mol, flexible_amides, rigidify_bonds)
    waters = _water_sites(mol, types, xyz) if hydrate else []
    wet = {i for i, _ in waters}  # meeko keeps an atom carrying a W as its own body, never merged
    broken, glue = (), set()
    combos = _break_macrocycle(mol, types, rot, macrocycle_sizes, macrocycle_allow_A) if open_macrocycles else ()
    if combos:
        def score(cut):
            members, graph, _, root = _model(mol, keep, rot, set(cut), {a for b in cut for a in b} | wet)
            return _weighted_depth(members, graph, root, {a for b in cut for a in b})
        broken = min(combos, key=score)  # first of the shallowest, as meeko
        glue = {a for b in broken for a in b}
    members, graph, links, root = _model(mol, keep, rot, set(broken), glue | wet)
    torsdof = len(members) - 1
    if broken:  # TORSDOF counts the torsions of the closed molecule
        ring = _ring_bonds(mol)
        torsdof = sum(_key(*links[k]) not in ring for k in links) // 2

    # pseudo-atoms ride with the body of their anchor: (name, type, xyz, charge)
    pseudo = {}
    names = [a.GetSymbol() for a in mol.GetAtoms()]
    for n, (i, j) in enumerate(broken):
        types[i] = types[j] = f"CG{n}"
        pseudo.setdefault(i, []).append(("G", f"G{n}", xyz[j]))
        pseudo.setdefault(j, []).append(("G", f"G{n}", xyz[i]))
    for i, w in waters:
        pseudo.setdefault(i, []).append(("WAT", "W", w))
    if reactive_atom is not None:
        from ms_adtools.reactive import retype
        nb = lambda i: [n.GetIdx() for n in mol.GetAtomWithIdx(int(i)).GetNeighbors()]
        merged = {i for i, t in enumerate(types) if t == "H" or t.startswith("CG")}  # CG keeps its closure type
        types = retype(types, nb, reactive_atom, merged)

    lines, serial, written, next_serial = [], {}, [], count(1).__next__

    def write(body, first=None):
        tail = []
        for i in ([first] if first is not None else []) + sorted(i for i in members[body] if i != first):
            serial[i] = next_serial()
            lines.append(_line(serial[i], names[i], xyz[i], charge[i], types[i]))
            written.append((i, serial[i]))
            tail += pseudo.get(i, [])
        for name, t, p in tail:
            lines.append(_line(next_serial(), name, p, 0.0, t))

    def walk(body, seen):
        seen.add(body)
        for child in sorted(graph[body], key=lambda b: links[body, b]):
            if child in seen:
                continue
            i, j = links[body, child]
            at = len(lines)
            lines.append(None)  # BRANCH needs the serial j gets below
            write(child, j)
            lines[at] = "BRANCH %3d %3d" % (serial[i], serial[j])
            walk(child, seen)
            lines.append("ENDBRANCH %3d %3d" % (serial[i], serial[j]))

    lines.append("ROOT")
    write(root)
    lines.append("ENDROOT")
    walk(root, set())
    lines.append("TORSDOF %d" % torsdof)
    if smiles_remarks:
        lines = _smiles_remarks(mol, written) + lines
    return "\n".join(lines) + "\n"


def prepare_reactive_ligand(mol, smarts, idx=0, **kw):
    """One reactive PDBQT (``prepare_ligand(reactive_atom=...)``) per atom at position ``idx``
    (0-based) of a ``smarts`` match, for AutoDock-GPU reactive docking."""
    mol = _whole(mol)  # the match indices must be those prepare_ligand sees
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"bad reactive SMARTS {smarts}")
    # ponytail: one PDBQT per distinct atom, where meeko repeats one per (ununiquified) match
    atoms = list(dict.fromkeys(m[idx] for m in mol.GetSubstructMatches(patt, uniquify=False)))
    if not atoms:
        raise ValueError(f"reactive SMARTS {smarts} does not match")
    return [prepare_ligand(mol, reactive_atom=a, **kw) for a in atoms]
