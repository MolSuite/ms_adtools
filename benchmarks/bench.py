"""ms_adtools benchmark, meeko as the reference: import cost, ligand and receptor preparation time and memory.

Each measurement runs in its own subprocess, so the reported RSS is that of a fresh
interpreter that only imported what the measurement needs.

    python bench.py            # all measurements, as a markdown table
    python bench.py import_meeko   # one measurement, as JSON
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIGANDS = os.environ.get("BENCH_SDF", "/tmp/adt/ligs/set200.sdf")
RECEPTOR = os.environ.get("BENCH_PDB", os.path.join(HERE, "..", "test", "data", "1iep_receptorH.pdb"))


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _timed(fn):
    """(wall seconds, cpu seconds, RSS growth MB, peak RSS MB) of fn()."""
    base = rss_mb()
    t, c = time.perf_counter(), time.process_time()
    fn()
    wall, cpu = time.perf_counter() - t, time.process_time() - c
    return wall, cpu, rss_mb() - base, rss_mb()


def import_adtools_receptor():
    def go():
        from ms_adtools import prepare_receptor  # noqa: F401  (numpy only)
    return _timed(go)


def import_adtools_ligand():
    def go():
        from ms_adtools import prepare_ligand  # noqa: F401  (pulls RDKit in)
    return _timed(go)


def import_rdkit():
    def go():
        from rdkit import Chem  # noqa: F401  (the floor both ligand paths pay)
    return _timed(go)


def import_meeko():
    def go():
        import meeko  # noqa: F401
    return _timed(go)


def _mols():
    from rdkit import Chem
    return [m for m in Chem.SDMolSupplier(LIGANDS, removeHs=False) if m is not None]


def ligands_adtools():
    from ms_adtools import prepare_ligand
    mols = _mols()
    out = []
    return (*_timed(lambda: out.extend(prepare_ligand(m) for m in mols)), len(mols))


def ligands_meeko():
    from rdkit import Chem
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    # meeko refuses a molecule with more than one fragment, so the salts are split here,
    # as ms_adtools does on its own
    mols = [max(Chem.GetMolFrags(m, asMols=True), key=lambda f: f.GetNumHeavyAtoms()) for m in _mols()]
    prep = MoleculePreparation()
    out = []

    def go():
        for m in mols:
            for setup in prep.prepare(m):
                out.append(PDBQTWriterLegacy.write_string(setup)[0])
    return (*_timed(go), len(mols))


def receptor_adtools():
    from ms_adtools import prepare_receptor
    return _timed(lambda: prepare_receptor(RECEPTOR))


def receptor_meeko():
    from meeko import MoleculePreparation, Polymer, ResidueChemTemplates

    def go():
        Polymer.from_pdb_file(RECEPTOR, ResidueChemTemplates.create_from_defaults(),
                              MoleculePreparation(), blunt_ends=[(0, 0)], allow_bad_res=True,
                              default_altloc="A")
    return _timed(go)


MEASUREMENTS = {k: v for k, v in list(globals().items())
                if k.startswith(("import_", "ligands_", "receptor_"))}


def main():
    if len(sys.argv) > 1:
        print(json.dumps(MEASUREMENTS[sys.argv[1]]()))
        return
    rows = {}
    for name in MEASUREMENTS:
        p = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True)
        last = p.stdout.strip().splitlines()[-1:]  # the tools print notes of their own
        rows[name] = json.loads(last[0]) if p.returncode == 0 and last else p.stderr.strip().splitlines()[-1:]
    print("| measurement | wall s | cpu s | RSS growth MB | peak RSS MB | n |")
    print("|---|---|---|---|---|---|")
    for name, r in rows.items():
        if isinstance(r, list) and len(r) >= 4:
            n = r[4] if len(r) > 4 else ""
            print(f"| {name} | {r[0]:.2f} | {r[1]:.2f} | {r[2]:.0f} | {r[3]:.0f} | {n} |")
        else:
            print(f"| {name} | failed: {r} | | | | |")


if __name__ == "__main__":
    main()
