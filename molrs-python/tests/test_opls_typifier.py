"""Python-binding coverage for the native OPLS-AA typifier."""

import math

import numpy as np
import pytest

import molrs


def _ethane() -> "molrs.Atomistic":
    """Ethane (C2H6) with explicit hydrogens and a plausible geometry."""
    mol = molrs.Atomistic()
    c1 = mol.add_atom("C", 0.0, 0.0, 0.0)
    c2 = mol.add_atom("C", 1.54, 0.0, 0.0)
    hpos = [
        (c1, (-0.36, 1.03, 0.0)),
        (c1, (-0.36, -0.51, 0.89)),
        (c1, (-0.36, -0.51, -0.89)),
        (c2, (1.90, 1.03, 0.0)),
        (c2, (1.90, -0.51, 0.89)),
        (c2, (1.90, -0.51, -0.89)),
    ]
    for c, (x, y, z) in hpos:
        h = mol.add_atom("H", x, y, z)
        mol.add_bond(c, h)
    mol.add_bond(c1, c2)
    return mol


def test_opls_typifier_is_exposed():
    """OPLSAATypifier exists and constructs from embedded OPLS-AA."""
    assert "OPLSAATypifier" in dir(molrs)
    assert molrs.typifier.OPLSAATypifier is molrs.OPLSAATypifier
    typifier = molrs.OPLSAATypifier()
    assert typifier is not None


def test_typify_assigns_atom_types():
    """typify() returns a typed Atomistic graph."""
    typifier = molrs.OPLSAATypifier()
    typed = typifier.typify(_ethane())
    assert isinstance(typed, molrs.Atomistic)
    frame = typed.to_frame()
    atoms = frame["atoms"]
    assert atoms.nrows == 8
    types = atoms["type"]
    # Every atom typed (no empty / null type label).
    assert all(str(t) != "" for t in types)


def test_typify_and_build():
    """typify() adds bonded blocks; build() yields finite energy."""
    typifier = molrs.OPLSAATypifier()
    mol = _ethane()
    typed = typifier.typify(mol)
    frame = typed.to_frame()
    assert frame["bonds"].nrows == 7
    assert frame["angles"].nrows == 12
    assert frame["dihedrals"].nrows == 9

    pots = typifier.build(mol)
    coords = molrs.extract_coords(frame)
    energy, forces = pots.calc_energy_forces(coords)
    assert math.isfinite(energy)
    assert np.isfinite(np.asarray(forces)).all()


def test_xml_source_constructs():
    """The constructor accepts OPLS-AA XML text."""
    # The embedded canonical set is also reachable via the reader; round-trip a
    # minimal well-formed OPLS-AA forcefield document.
    xml = (
        "<ForceField><AtomTypes>"
        '<Type name="opls_135" class="CT" element="C" mass="12.011"/>'
        "</AtomTypes></ForceField>"
    )
    typifier = molrs.OPLSAATypifier(xml)
    assert typifier is not None


def test_invalid_xml_raises_not_panics():
    """Malformed input raises a Python exception rather than aborting."""
    with pytest.raises((ValueError, RuntimeError)):
        molrs.OPLSAATypifier("<not valid xml <<<")


def test_oplsaa_rejects_coarse_grain():
    typifier = molrs.OPLSAATypifier()
    cg = molrs.CoarseGrain()
    with pytest.raises(TypeError):
        typifier.typify(cg)
