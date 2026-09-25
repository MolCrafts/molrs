"""Seam tests for ForceField special-bonds accessors."""

from __future__ import annotations

import numpy as np
import pytest

import molrs


def test_special_bonds_default_shape_and_dtype():
    ff = molrs.ff.ForceField("x")
    lj = ff.special_bonds_lj
    coul = ff.special_bonds_coul
    assert lj.shape == (3,)
    assert coul.shape == (3,)
    assert lj.dtype == np.float64
    np.testing.assert_array_equal(lj, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(coul, [0.0, 0.0, 1.0])


def test_special_bonds_copy_semantics():
    ff = molrs.ff.ForceField("x")
    a = ff.special_bonds_lj
    b = ff.special_bonds_lj
    assert a is not b
    a[2] = 0.25
    np.testing.assert_array_equal(ff.special_bonds_lj, [0.0, 0.0, 1.0])


def test_set_special_bonds_round_trip():
    ff = molrs.ff.ForceField("x")
    ff.set_special_bonds([1.0, 1.0, 0.5], [0.0, 0.0, 1.0 / 1.2])
    np.testing.assert_allclose(ff.special_bonds_lj, [1.0, 1.0, 0.5])
    np.testing.assert_allclose(ff.special_bonds_coul, [0.0, 0.0, 1.0 / 1.2])


def test_set_special_bonds_wrong_length_raises():
    ff = molrs.ff.ForceField("x")
    with pytest.raises(ValueError):
        ff.set_special_bonds([0.5], [0.0, 0.0, 0.8333])


def test_from_raw_forwards_special_bonds():
    raw = molrs.ff.ForceField("src")
    raw.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 1.0 / 1.2])
    wrapped = molrs.ff.ForceField._from_raw(raw)
    np.testing.assert_allclose(wrapped.special_bonds_lj, [0.0, 0.0, 0.5])
    np.testing.assert_allclose(wrapped.special_bonds_coul, [0.0, 0.0, 1.0 / 1.2])


def test_declared_special_bonds_is_none_on_an_undeclared_forcefield():
    ff = molrs.ff.ForceField("x")
    assert ff.declared_special_bonds() is None


def test_set_special_bonds_declares_them():
    ff = molrs.ff.ForceField("x")
    ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.75])
    assert ff.declared_special_bonds() == ((0.0, 0.0, 0.5), (0.0, 0.0, 0.75))


def test_merge_keeps_special_bonds_lj():
    other = molrs.ff.ForceField("src")
    other.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.75])
    ff = molrs.ff.ForceField("dst")
    ff.merge(other)
    np.testing.assert_array_equal(ff.special_bonds_lj, [0.0, 0.0, 0.5])
    np.testing.assert_array_equal(ff.special_bonds_coul, [0.0, 0.0, 0.75])
