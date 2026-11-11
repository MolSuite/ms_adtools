"""Reactive docking with AutoDock-GPU: reactive atom types and their pair potentials.

The reactive atom gets order 1, its bonded neighbours order 2 and theirs order 3; each order
has its own type name (C -> C1/C2/C3, OA -> O4/O5/O6, ...), the names meeko uses. A reactive
receptor atom and a reactive ligand atom (orders 1 + 1) attract with a 13-7 potential; the
1-3 and 1-4 pairs across them get softened 12-6 terms, so the bond can form.
"""
import numpy as np

# Rii, epsii of the AD4 base types that have reactive variants
FF = {"HD": (2.00, 0.020), "C": (4.00, 0.150), "A": (4.00, 0.150), "N": (3.50, 0.160),
      "NA": (3.50, 0.160), "OA": (3.20, 0.200), "OS": (3.20, 0.200), "F": (3.09, 0.080),
      "P": (4.20, 0.200), "SA": (4.00, 0.200), "S": (4.00, 0.200), "Cl": (4.09, 0.276),
      "CL": (4.09, 0.276), "Br": (4.33, 0.389), "BR": (4.33, 0.389), "I": (4.72, 0.550),
      "Si": (4.10, 0.200), "B": (3.84, 0.155), "W": (0.00, 0.000)}
LIGAND_BASE_TYPES = ("HD", "C", "A", "N", "NA", "OA", "F", "P", "SA", "S", "Cl", "Br", "I", "Si", "B")
RECEPTOR_ATOMS = {"SER": "OG", "LYS": "NZ", "TYR": "OH", "CYS": "SG", "HIE": "NE2", "HID": "ND1",
                  "GLU": "OE2", "THR": "OG1", "MET": "SD"}

NAME, BASE = {}, {}  # (base type, order) -> reactive type; reactive type -> (base type, order)
for _order in (1, 2, 3):
    for _t in FF:  # one letter: type + order; two: first letter + order + 3, or + 6 when taken
        _n = f"{_t}{_order}" if len(_t) == 1 else next(
            n for n in (f"{_t[0]}{_order + 3}", f"{_t[0]}{_order + 6}") if n not in BASE)
        NAME[_t, _order], BASE[_n] = _n, (_t, _order)


def base_and_order(atype):
    """(base type, order) of a reactive type, a receptor residue prefix ("1O4") allowed; else None."""
    return BASE.get(atype[1:] if len(atype) > 1 and atype[0].isdigit() else atype)


def retype(types, neighbors, atom, skip=()):
    """``types`` (a list) with ``atom`` made reactive: order 1 for it, 2 for its neighbours, 3 for
    theirs, a later order overwriting an earlier one (as meeko does, a 3-ring ends at 3).
    ``neighbors(i)`` gives the bonded atoms; ``skip`` atoms (merged hydrogens) keep their type."""
    def name(i, order):
        if (types[i], order) not in NAME:
            raise ValueError(f"atom type {types[i]} has no reactive variant")
        return NAME[types[i], order]
    out = list(types)
    out[atom] = name(atom, 1)
    for j in neighbors(atom):
        if j not in skip:
            out[j] = name(j, 2)
        for k in neighbors(j):
            if k != atom and k not in skip:
                out[k] = name(k, 3)
    return out


def reactive_flexres(flex, residues):
    """A flexible-residue PDBQT with reactive types on ``residues``: "A:SER203" (or "A:203"), or "A:HIS57=NE2"
    for an atom other than the default (``RECEPTOR_ATOMS``; HIS takes the N carrying H). The
    types of residue n (1-8) are prefixed "n", so each reactive residue has its own maps."""
    if len(residues) > 8:
        raise ValueError(f"at most 8 reactive residues, got {len(residues)}")
    lines = flex.splitlines()
    for prefix, spec in enumerate(residues, 1):
        res, _, atom_name = spec.partition("=")
        chain, _, res = res.rpartition(":")
        at = [n for n, l in enumerate(lines) if l.startswith("ATOM") and res in (l[17:20].strip() + l[22:26].strip(), l[22:26].strip())
              and (not chain or l[21] == chain)]
        if not at:
            raise ValueError(f"reactive residue {spec} is not among the flexible residues")
        names = [lines[n][12:16].strip() for n in at]
        resname = lines[at[0]][17:20]
        if not atom_name:
            atom_name = "NE2" if resname == "HIS" and "HE2" in names else "ND1" if resname == "HIS" \
                else RECEPTOR_ATOMS.get(resname)
        if atom_name not in names:
            raise ValueError(f"no reactive atom {atom_name} in {spec}; name it as {spec}=ATOM")
        xyz = np.array([[float(lines[n][c:c + 8]) for c in (30, 38, 46)] for n in at])
        hyd = np.array([lines[n][77:79].strip() in ("H", "HD") for n in at])
        d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        # ponytail: bonds by distance (H 1.3 A, heavy 2.0 A); a residue's side chain has no closer non-bonded pair
        bonded = (d > 0) & (d < np.where(hyd[:, None] | hyd[None], 1.3, 2.0))
        types = [lines[n][77:79].strip() for n in at]
        new = retype(types, lambda i: np.flatnonzero(bonded[i]), names.index(atom_name))
        for n, old, t in zip(at, types, new):
            if t != old:
                lines[n] = lines[n][:77] + f"{prefix}{t}"
    return "\n".join(lines) + "\n"


def reactive_config(flex, map_prefix, eps12=2.5, r12=1.8, r13_scaling=0.5, r14_scaling=0.5):
    """AutoDock-GPU configuration for reactive docking with the flexible residues ``flex``
    (from ``reactive_flexres``): ligand types, the maps (``map_prefix`` names the autogrid output,
    each reactive type reading the map of its base type) and the intnbp_r_eps pair terms.
    Run as ``autodock_gpu --import_dpf <this file> --flexres <flex> -L <ligand>``."""
    lig = [NAME[t, o] for o in (1, 2, 3) for t in LIGAND_BASE_TYPES]
    rec = sorted({l[77:].strip() for l in flex.splitlines() if l.startswith(("ATOM", "HETATM"))
                  and base_and_order(l[77:].strip())})
    if not rec:
        raise ValueError("no reactive types in the flexible residues; run reactive_flexres first")
    derived = {}
    for t in lig + rec:
        derived.setdefault(base_and_order(t)[0], []).append(t)
    out = ["ligand_types " + " ".join(x for b, ts in derived.items() for x in [b] + ts),
           f"fld {map_prefix}.maps.fld"]
    out += [f"map {map_prefix}.{b}.map" for b, ts in derived.items() for _ in [b] + ts]
    scaling = {3: r13_scaling, 4: r14_scaling}
    for t1 in lig:
        (b1, o1) = base_and_order(t1)
        for t2 in rec:
            (b2, o2) = base_and_order(t2)
            order = o1 + o2
            if order == 2:
                r, eps, n, m = r12, eps12, 13, 7
            elif order in scaling and ("HD" in (b1, b2) or "F" in (b1, b2)):
                r, eps, n, m = 0.01, 0.001, 12, 6
            elif order in scaling:
                r = (FF[b1][0] + FF[b2][0]) / 2 * scaling[order]
                eps, n, m = np.sqrt(FF[b1][1] * FF[b2][1]) * 0.1662, 12, 6
            else:
                continue
            out.append("intnbp_r_eps %8.6f %8.6f %3d %3d %4s %4s" % (r, eps, n, m, *sorted((t1, t2))))
    return "\n".join(out) + "\n"
