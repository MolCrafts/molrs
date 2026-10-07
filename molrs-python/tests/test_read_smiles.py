"""``molrs.io.read_smiles_str``: one SMILES molecule in, an ``Atomistic`` out —
connectivity only."""

from __future__ import annotations

import molrs
import pytest


def test_reads_one_molecule_without_adding_hydrogens() -> None:
    mol = molrs.io.read_smiles_str("CCO")
    assert isinstance(mol, molrs.core.Atomistic)
    assert mol.n_atoms == 3
    assert mol.n_bonds == 2


def test_matches_the_ir_conversion() -> None:
    expected = molrs.io.smiles.SmilesIr("c1ccccc1").to_atomistic()
    assert molrs.io.read_smiles_str("c1ccccc1").n_atoms == expected.n_atoms


def test_a_set_of_molecules_is_refused_naming_components() -> None:
    with pytest.raises(ValueError, match=r"SmilesIr\(s\)\.components\(\)") as excinfo:
        molrs.io.read_smiles_str("CCO.O")
    assert isinstance(excinfo.value, molrs.io.smiles.SmilesError)
    assert excinfo.value.kind == "MultipleComponents"
    assert "2" in str(excinfo.value)


def test_invalid_smiles_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        molrs.io.read_smiles_str("C(")
