"""Tests for the ``molrs.ff.ForceField`` programmatic builder PyO3 surface.

The Rust builder semantics (idempotent ``def_*style``, type naming, ``def_type``
arity rules) are covered by unit tests in ``molrs-ff/src/forcefield/mod.rs``.
These tests verify the binding is wired end-to-end: a force field built entirely
from Python via ``def_*style`` / ``def_*type`` / ``def_type`` reads back through
``style_names`` / ``types`` / ``style_params`` and compiles into evaluable
``Potentials`` that produce the closed-form energy.

This is the P0-B enabler that lets molpy drop its parallel Python ForceField
hierarchy and inherit ``molrs.ff.ForceField`` directly.
"""

import numpy as np
import pytest

import molrs


def test_empty_forcefield_constructs():
    ff = molrs.ff.ForceField("scratch")
    assert ff.name == "scratch"
    assert ff.style_names() == []


def test_def_bondtype_creates_style_and_type():
    ff = molrs.ff.ForceField("bond-only")
    ff.def_bondtype("harmonic", "CT", "CT", {"k": 300.0, "r0": 1.5})
    assert ff.style_names() == ["bond:harmonic"]
    ((name, params),) = ff.types("bond", "harmonic")
    assert name == "CT-CT"
    assert params == {"k": 300.0, "r0": 1.5}


def test_def_pairstyle_carries_style_level_params():
    ff = molrs.ff.ForceField("lj")
    ff.def_pairstyle("lj/cut", {"cutoff": 10.0})
    ff.def_pairtype("lj/cut", "CT", None, {"epsilon": 0.066, "sigma": 3.5})
    assert ff.style_params("pair", "lj/cut") == {"cutoff": 10.0}
    ((name, params),) = ff.types("pair", "lj/cut")
    assert name == "CT"  # self-pair name is just the single atom type
    assert params == {"epsilon": 0.066, "sigma": 3.5}


def test_def_style_is_idempotent():
    ff = molrs.ff.ForceField("dup")
    ff.def_bondtype("harmonic", "A", "B", {"k": 1.0, "r0": 1.0})
    ff.def_bondtype("harmonic", "A", "C", {"k": 2.0, "r0": 1.1})
    # one style, two types
    assert ff.style_names() == ["bond:harmonic"]
    assert len(ff.types("bond", "harmonic")) == 2


def test_unified_def_type_matches_typed_methods():
    ff = molrs.ff.ForceField("unified")
    ff.def_type("bond", "harmonic", "CT-CT", {"k": 300.0, "r0": 1.5})
    ff.def_type("angle", "harmonic", "CT-CT-CT", {"k": 40.0, "theta0": 1.9})
    assert set(ff.style_names()) == {"bond:harmonic", "angle:harmonic"}
    assert ff.types("angle", "harmonic")[0][0] == "CT-CT-CT"


@pytest.mark.parametrize(
    "category,name",
    [("bond", "A-B-C"), ("angle", "A-B"), ("dihedral", "A-B-C")],
)
def test_def_type_arity_raises_not_panics(category, name):
    ff = molrs.ff.ForceField("guard")
    with pytest.raises(ValueError):
        ff.def_type(category, "s", name, {"k": 1.0})


def test_kspace_is_not_a_category():
    from molrs._lib import ForceField as RsForceField

    assert not hasattr(RsForceField, "def_kspacestyle")
    ff = molrs.ff.ForceField("guard")
    with pytest.raises(ValueError, match="unknown"):
        ff.def_type("kspace", "pme", "X", {"k": 1.0})


def test_types_on_missing_style_raises():
    ff = molrs.ff.ForceField("empty")
    with pytest.raises(ValueError):
        ff.types("bond", "nope")


def test_builder_compiles_and_evaluates():
    """Built FF -> to_potentials -> calc_energy gives the closed-form value."""
    ff = molrs.ff.ForceField("bond-only")
    ff.def_bondtype("harmonic", "CT", "CT", {"k": 300.0, "r0": 1.5})

    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, 2.0]))
    atoms.insert("y", np.array([0.0, 0.0]))
    atoms.insert("z", np.array([0.0, 0.0]))
    frame["atoms"] = atoms
    bonds = molrs.Block()
    bonds.insert("atomi", np.array([0], dtype=np.uint32))
    bonds.insert("atomj", np.array([1], dtype=np.uint32))
    bonds.insert("type", np.array(["CT-CT"], dtype=str))
    frame["bonds"] = bonds

    pots = ff.to_potentials(frame)
    energy = pots.calc_energy(molrs.ff.extract_coords(frame))
    # r = 2.0, dr = 0.5 -> E = 0.5 * 300 * 0.25 = 37.5
    assert abs(energy - 37.5) < 1e-9
