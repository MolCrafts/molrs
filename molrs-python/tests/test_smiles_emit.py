"""Python surface for SMILES / local-SMARTS emit (smiles-emit-04)."""

from __future__ import annotations

import molrs
import pytest


def test_write_smiles_round_trip():
    mol = molrs.io.SmilesIR("CCO").to_atomistic()
    ir2 = molrs.io.SmilesIR.from_atomistic(mol, canonical=True)
    s = ir2.write_smiles()
    assert isinstance(s, str) and s
    molrs.io.SmilesIR(s)  # re-parse


def test_write_smiles_module():
    mol = molrs.io.SmilesIR("c1ccccc1").to_atomistic()
    s = molrs.io.write_smiles(mol, canonical=True)
    assert s
    molrs.io.SmilesIR(s)


def test_write_smarts_matches():
    mol = molrs.io.SmilesIR("CCO").to_atomistic()
    # first heavy atom handle from atoms iteration
    atoms = list(mol.atoms) if hasattr(mol, "atoms") else []
    if atoms:
        center = atoms[0].handle if hasattr(atoms[0], "handle") else int(atoms[0])
    else:
        # fallback: structural handles via canonical_order
        center = mol.canonical_order()[0]
    s = molrs.io.write_smarts(mol, center, reach=1, atomic_number=True)
    assert isinstance(s, str) and s
    # alias still works
    assert molrs.io.write_local_smarts(mol, center, reach=1, atomic_number=True) == s
    from molrs.perceive import SmartsPattern

    pat = SmartsPattern(s)
    assert pat.has_match(mol)


def test_atomistic_has_no_to_smiles():
    mol = molrs.io.SmilesIR("CCO").to_atomistic()
    assert not hasattr(mol, "to_smiles")
    assert not hasattr(mol, "from_smiles")
    assert not hasattr(mol, "to_smarts")
    assert not hasattr(type(mol), "to_smiles")


def test_bad_aromatic_flag():
    mol = molrs.io.SmilesIR("CCO").to_atomistic()
    with pytest.raises((ValueError, TypeError)):
        molrs.io.SmilesIR.from_atomistic(mol, aromatic="nope")


def test_bad_neighbor_style():
    mol = molrs.io.SmilesIR("CCO").to_atomistic()
    center = mol.canonical_order()[0]
    with pytest.raises((ValueError, TypeError)):
        molrs.io.write_smarts(mol, center, neighbor_style="x")
