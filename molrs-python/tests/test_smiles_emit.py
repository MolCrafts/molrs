"""Python surface for SMILES emit (``molrs.io.write_smiles_str``) and the
SMARTS pattern of an atom's environment
(``molrs.perceive.SmartsPattern.from_environment``)."""

from __future__ import annotations

import molrs
import pytest


def test_write_smiles_str_round_trip():
    mol = molrs.io.read_smiles_str("CCO")
    s = molrs.io.write_smiles_str(mol, canonical=True)
    assert isinstance(s, str) and s
    assert molrs.io.read_smiles_str(s).n_atoms == 3


def test_write_smiles_str_is_the_one_spelling():
    # Writing SMILES text is `molrs.io.write_smiles_str`; neither an IR method
    # nor a module-level `write_smiles` writes it.
    assert not hasattr(molrs.io, "write_smiles")
    assert not hasattr(molrs.io.smiles.SmilesIr, "write_smiles")
    assert not hasattr(molrs.io.smiles.SmilesIr, "write_smarts")
    mol = molrs.io.read_smiles_str("c1ccccc1")
    s = molrs.io.write_smiles_str(mol, canonical=True)
    assert s
    molrs.io.smiles.SmilesIr(s)


def test_environment_pattern_matches():
    mol = molrs.io.smiles.SmilesIr("CCO").to_atomistic()
    # first heavy atom handle from atoms iteration
    atoms = list(mol.atoms) if hasattr(mol, "atoms") else []
    if atoms:
        center = atoms[0].handle if hasattr(atoms[0], "handle") else int(atoms[0])
    else:
        # fallback: structural handles via canonical_order
        center = mol.canonical_order()[0]
    from molrs.perceive import SmartsPattern

    env = SmartsPattern.from_environment(mol, center, reach=1, atomic_number=True)
    s = str(env)
    assert isinstance(s, str) and s
    assert not hasattr(molrs.io, "write_smarts")
    assert not hasattr(molrs.io, "write_local_smarts")
    assert env.has_match(mol)
    assert SmartsPattern(s).has_match(mol)


def test_atomistic_has_no_to_smiles():
    mol = molrs.io.smiles.SmilesIr("CCO").to_atomistic()
    assert not hasattr(mol, "to_smiles")
    assert not hasattr(mol, "from_smiles")
    assert not hasattr(mol, "to_smarts")
    assert not hasattr(type(mol), "to_smiles")


def test_bad_aromatic_flag():
    mol = molrs.io.smiles.SmilesIr("CCO").to_atomistic()
    with pytest.raises((ValueError, TypeError)):
        molrs.io.smiles.SmilesIr.from_atomistic(mol, aromatic="nope")


def test_bad_neighbor_style():
    mol = molrs.io.smiles.SmilesIr("CCO").to_atomistic()
    center = mol.canonical_order()[0]
    with pytest.raises((ValueError, TypeError)):
        molrs.perceive.SmartsPattern.from_environment(mol, center, neighbor_style="x")
