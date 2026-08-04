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
    assert "OPLSAATypifier" in molrs.ff.typifier.__all__
    assert molrs.ff.OPLSAATypifier is molrs.ff.typifier.OPLSAATypifier
    typifier = molrs.ff.typifier.OPLSAATypifier()
    assert isinstance(typifier, molrs.ff.Typifier)


def test_typify_assigns_atom_types():
    """typify() returns a typed Atomistic graph."""
    typifier = molrs.ff.typifier.OPLSAATypifier()
    typed = typifier.typify(_ethane())
    assert isinstance(typed, molrs.Atomistic)
    frame = typed.to_frame()
    atoms = frame["atoms"]
    assert atoms.nrows == 8
    types = atoms["type"]
    # Every atom typed (no empty / null type label).
    assert all(str(t) != "" for t in types)


def test_typify_and_compose_potentials():
    """typify() adds bonded blocks; compose path yields finite energy (no build())."""
    typifier = molrs.ff.typifier.OPLSAATypifier()
    mol = _ethane()
    typed = typifier.typify(mol)
    frame = typed.to_frame()
    assert frame["bonds"].nrows == 7
    assert frame["angles"].nrows == 12
    assert frame["dihedrals"].nrows == 9

    pairs = molrs.ff.intramolecular_pairs(frame)
    frame["pairs"] = pairs
    pots = typifier.forcefield().to_potentials(frame)
    coords = molrs.ff.extract_coords(frame)
    energy, forces = pots.calc_energy_forces(coords)
    assert math.isfinite(energy)
    assert np.isfinite(np.asarray(forces)).all()


def test_opls_has_no_build_facade():
    """0.12: OPLS matches MMFF — no typifier.build()."""
    assert not hasattr(molrs.ff.typifier.OPLSAATypifier(), "build")


def test_xml_source_constructs():
    """The constructor accepts OPLS-AA XML text."""
    # The embedded canonical set is also reachable via the reader; round-trip a
    # minimal well-formed OPLS-AA forcefield document.
    xml = (
        "<ForceField><AtomTypes>"
        '<Type name="opls_135" class="CT" element="C" mass="12.011"/>'
        "</AtomTypes></ForceField>"
    )
    typifier = molrs.ff.typifier.OPLSAATypifier(xml)
    assert typifier is not None


def test_invalid_xml_raises_not_panics():
    """Malformed input raises a Python exception rather than aborting."""
    with pytest.raises((ValueError, RuntimeError)):
        molrs.ff.typifier.OPLSAATypifier("<not valid xml <<<")


def test_oplsaa_rejects_coarse_grain():
    typifier = molrs.ff.typifier.OPLSAATypifier()
    cg = molrs.CoarseGrain()
    with pytest.raises(TypeError):
        typifier.typify(cg)
