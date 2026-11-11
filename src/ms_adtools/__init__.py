"""AutoDock preparation (ligand, receptor, flexible residues, poses) on RDKit and gemmi."""
import importlib

__all__ = ["prepare_ligand", "prepare_reactive_ligand", "prepare_receptor", "prepare_flexreceptor",
           "pdbqt_to_rdkit", "dlg_to_pdbqt", "flexres_models", "add_zinc_pseudo_atoms"]
_WHERE = {"prepare_ligand": "ligand", "prepare_reactive_ligand": "ligand", "pdbqt_to_rdkit": "poses",
          "dlg_to_pdbqt": "poses", "flexres_models": "poses",
          "prepare_receptor": "receptor", "prepare_flexreceptor": "receptor",
          "add_zinc_pseudo_atoms": "zinc"}


def __getattr__(name):  # load a submodule on first use: a pose reader does not import the receptor path
    if name in _WHERE:
        return getattr(importlib.import_module("ms_adtools." + _WHERE[name]), name)
    raise AttributeError(name)
