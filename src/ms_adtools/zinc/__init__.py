"""AutoDock4Zn receptor preparation: TZ pseudo-atoms and the AD4Zn parameter file.

``zinc_pseudo.py`` (Apache-2.0, see LICENSE) and ``AD4Zn.dat`` (GPL-2.0-or-later data, see
GPL-2.0) come from AutoDock Vina, see NOTICE; only the entry points are adapted so the work
happens in process.
"""
from pathlib import Path

from .zinc_pseudo import add_pseudo_atoms

PARAMETERS = Path(__file__).with_name("AD4Zn.dat")

__all__ = ["add_zinc_pseudo_atoms", "PARAMETERS"]


def add_zinc_pseudo_atoms(pdbqt):
    """Receptor PDBQT text with a TZ pseudo-atom next to every zinc that can bind a ligand
    tetrahedrally, and the zinc charges zeroed. Dock with ``PARAMETERS`` as the AutoDock4
    parameter file."""
    lines, _ = add_pseudo_atoms(pdbqt.splitlines(keepends=True))
    return "".join(lines)
