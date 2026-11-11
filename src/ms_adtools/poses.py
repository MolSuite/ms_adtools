"""Docking output -> RDKit molecule (from the REMARK SMILES lines ``ligand.prepare_ligand`` writes),
AutoDock4 / AutoDock-GPU DLG -> PDBQT poses, and the docked flexible residues as a structure."""
import re

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

# conformer properties read from each MODEL: Vina remarks, or the USER lines of a DLG pose
_PROPS = [("score", re.compile(r"REMARK VINA RESULT:\s+(\S+)")),
          ("score", re.compile(r"USER\s+Estimated Free Energy of Binding\s+=\s+(\S+)")),
          ("intermolecular", re.compile(r"USER\s+\(1\) Final Intermolecular Energy\s+=\s+(\S+)")),
          ("internal", re.compile(r"USER\s+\(2\) Final Total Internal Energy\s+=\s+(\S+)")),
          ("run", re.compile(r"USER\s+Run = (\d+)")),
          ("cluster_rank", re.compile(r"USER\s+Cluster Rank = (\d+)")),
          ("cluster_size", re.compile(r"USER\s+Cluster Size = (\d+)"))]


def _pairs(lines, prefix):
    nums = [int(t) for l in lines if l.startswith(prefix) for t in l[len(prefix):].split()]
    return list(zip(nums[0::2], nums[1::2]))


def pdbqt_to_rdkit(text):
    """One molecule with explicit hydrogens and one conformer per MODEL of a PDBQT
    (Vina, QuickVina, GNINA output, ``dlg_to_pdbqt`` of a DLG, or a prepared ligand).

    Each conformer has the docked heavy atoms and polar hydrogens; the other hydrogens are
    placed by RDKit. The energy (``REMARK VINA RESULT`` or a DLG's free energy of binding) is the
    conformer property ``score``; a DLG pose also has intermolecular, internal, run,
    cluster_rank and cluster_size. Flexible residues are left out: see ``flexres_models``."""
    lines = text.splitlines()
    smiles = next((l.split()[2] for l in lines if l.startswith("REMARK SMILES ")
                   and not l.startswith("REMARK SMILES IDX")), None)
    if smiles is None:
        raise ValueError("no REMARK SMILES in the PDBQT; it was not prepared with SMILES remarks")
    mol = Chem.MolFromSmiles(smiles)  # atom i is SMILES position i + 1
    first = lines[:next((i for i, l in enumerate(lines) if l.startswith("ENDMDL")), len(lines))]
    heavy = [(s - 1, serial) for s, serial in _pairs(first, "REMARK SMILES IDX")]
    polar = [(s - 1, serial) for s, serial in _pairs(first, "REMARK H PARENT")]

    models, cur, in_res = [], None, False
    for l in lines:
        if l.startswith("MODEL") or cur is None:
            cur = {"xyz": {}, "props": {}}
            models.append(cur)
        if l.startswith(("BEGIN_RES", "END_RES")):  # a flexible residue numbers its atoms from 1 again
            in_res = l.startswith("BEGIN_RES")
        elif l.startswith(("ATOM", "HETATM")) and not in_res:
            cur["xyz"][int(l[6:11])] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        elif l.startswith(("REMARK", "USER")):
            for name, rx in _PROPS:
                if name not in cur["props"] and (got := rx.match(l)):
                    cur["props"][name] = got.group(1)
    models = [m for m in models if m["xyz"]]

    molh = Chem.AddHs(mol)
    h_of = {a: [n.GetIdx() for n in molh.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() == 1]
            for a, _ in polar}
    for m in models:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for a, serial in heavy:
            conf.SetAtomPosition(a, Point3D(*m["xyz"][serial]))
        one = Chem.Mol(mol)
        one.AddConformer(conf)
        one = Chem.AddHs(one, addCoords=True)
        taken = {a: 0 for a in h_of}
        for a, serial in polar:  # ponytail: Hs of one parent filled in written order, not matched by geometry
            one.GetConformer().SetAtomPosition(h_of[a][taken[a]], Point3D(*m["xyz"][serial]))
            taken[a] += 1
        c = Chem.Conformer(one.GetConformer())
        for name, value in m["props"].items():
            c.SetIntProp(name, int(value)) if value.isdigit() else c.SetDoubleProp(name, float(value))
        molh.AddConformer(c, assignId=True)
    return molh


def dlg_to_pdbqt(text, cluster_leads=True):
    """PDBQT poses of an AutoDock4 / AutoDock-GPU DLG, as ranked by its clustering: the best pose of
    each cluster (``cluster_leads``) or every run, best cluster first. Each MODEL keeps its USER
    energy lines and gets ``USER    Cluster Rank = n`` / ``USER    Cluster Size = n``."""
    poses, cur = {}, None
    for l in text.splitlines():
        if not l.startswith("DOCKED: "):
            continue
        l = l[8:]
        if l.startswith("MODEL"):
            cur = [l]
            poses[int(l.split()[1])] = cur
        elif cur is not None:
            cur.append(l)
            if (got := re.match(r"USER\s+Run = (\d+)", l)) and int(got.group(1)) not in poses:
                poses[int(got.group(1))] = poses.pop(int(cur[0].split()[1]))
    # RMSD table rows: rank, sub-rank, run, energy, cluster RMSD, reference RMSD, "RANKING"
    table = [tuple(map(int, l.split()[:3])) for l in text.splitlines() if l.rstrip().endswith("RANKING")]
    if not table:  # ponytail: no clustering written (a cut run): every pose its own cluster, file order
        table = [(n, 1, run) for n, run in enumerate(poses, 1)]
    size = {}
    for rank, _, _ in table:
        size[rank] = size.get(rank, 0) + 1
    out = []
    for rank, sub, run in table:
        if run in poses and (sub == 1 or not cluster_leads):
            model = poses[run]
            out += model[:1] + [f"USER    Cluster Rank = {rank}", f"USER    Cluster Size = {size[rank]}"] + model[1:]
    if not out:
        raise ValueError("no DOCKED poses in the DLG")
    return "\n".join(out) + "\n"


def _element(atype):
    """Element of an AutoDock type: OA -> O, A -> C, Zn -> Zn; reactive types by their base."""
    from ms_adtools.reactive import base_and_order
    t = (base_and_order(atype) or (atype,))[0]
    return "C" if t == "A" else t if len(t) == 1 or t[1].islower() else t[0]


def _pdb(l, serial):
    return "%-6s%5d%s  1.00  0.00          %2s" % (l[:6], serial % 100000, l[11:54], _element(l[77:].strip()))


def flexres_models(text, rigid=None):
    """Multi-model PDB of the flexible residues docked with each pose of ``text`` (Vina output, or
    ``dlg_to_pdbqt`` of a DLG); MODEL n goes with conformer n of ``pdbqt_to_rdkit(text)``.

    With ``rigid`` (the rigid receptor PDBQT text) every model is the whole receptor, each docked
    side chain written after the rest of its residue, ready for the viewer and interactions."""
    models, cur, in_res = [], None, False
    for l in text.splitlines():
        if l.startswith("MODEL") or cur is None:
            cur = []
            models.append(cur)
        if l.startswith(("BEGIN_RES", "END_RES")):
            in_res = l.startswith("BEGIN_RES")
        elif in_res and l.startswith(("ATOM", "HETATM")):
            cur.append(l)
    models = [m for m in models if m]
    if not models:
        raise ValueError("no flexible residues (BEGIN_RES ... END_RES) in the docking output")
    res = lambda l: (l[21], l[22:27], l[17:20])  # chain, number + insertion code, name
    base = [l for l in (rigid or "").splitlines() if l.startswith(("ATOM", "HETATM"))]
    last = {res(l): n for n, l in enumerate(base)}
    out = []
    for n, flex in enumerate(models, 1):
        after = {}
        for l in flex:
            after.setdefault(last.get(res(l), len(base) - 1), []).append(l)
        lines = after.get(-1, [])
        for i, l in enumerate(base):
            lines += [l] + after.get(i, [])
        out += [f"MODEL {n:8d}"] + [_pdb(l, s) for s, l in enumerate(lines, 1)] + ["ENDMDL"]
    return "\n".join(out + ["END"]) + "\n"


if __name__ == "__main__":  # self-check: a prepared ligand comes back with its own coordinates
    from rdkit.Chem import AllChem, rdMolAlign
    from ms_adtools.ligand import prepare_ligand

    ref = Chem.AddHs(Chem.MolFromSmiles("NC(=O)c1ccc(O)cc1CC[NH3+]"))
    AllChem.EmbedMolecule(ref, randomSeed=7)
    back = pdbqt_to_rdkit(prepare_ligand(ref))
    assert Chem.MolToSmiles(Chem.RemoveHs(back)) == Chem.MolToSmiles(Chem.RemoveHs(ref))
    assert rdMolAlign.CalcRMS(Chem.RemoveHs(back), Chem.RemoveHs(ref)) < 1e-3
    pol = [a.GetIdx() for a in back.GetAtoms() if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetAtomicNum() != 6]
    assert np.allclose(sorted(map(tuple, back.GetConformer().GetPositions()[pol].round(2))),
                       sorted(map(tuple, ref.GetConformer().GetPositions()[[a.GetIdx() for a in ref.GetAtoms()
                              if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetAtomicNum() != 6]].round(2))), atol=2e-3)
    print("ok")
