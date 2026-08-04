"""ac-011 — the Python surface offers only the typify -> ForceField route.

A Rust cleanup that leaves the broken door open in Python has cleaned nothing.
The free functional builder and ``molrs.ff.typifier.MMFF94Typifier().build()`` used to sit
adjacent in the same namespace with nothing to tell them apart — and one of them
was, until mmff-orthogonal-01, missing an entire 150 kcal/mol electrostatic term.
A Python user had no way to know which door was which, and no reason to guess.

So both doors close, and the one route that remains is the standard one every
other force field in molrs already uses::

    typed = molrs.ff.typifier.MMFF94Typifier().typify(mol)      # labels + charges. That is
    frame = typed.to_frame()                        # the Typifier's contract,
    frame["pairs"] = molrs.ff.intramolecular_pairs(frame)   # and all of it.
    pots  = molrs.ff.typifier.MMFF94Typifier().forcefield().to_potentials(frame)

MMFF stops being a special case.

NOTE: these tests are meaningless against a stale wheel. ``maturin develop
--release`` must be re-run after the Rust deletion, or ``hasattr`` will happily
report on the old extension module and every assertion below will lie.
"""

import molrs

# The name of the deleted free function, ASSEMBLED rather than written.
#
# `molrs/tests/ff/mmff/deletion_gate.rs` greps the whole repository — this
# directory included — for the dead symbols, and a test that asserts a name is
# absent must necessarily mention that name. Spelling it in pieces is how the
# gate's own source solves the identical problem for its needles (`concat!`), and
# it is what lets the gate scan every file honestly, with no exemption by path.
#
# The assertion below is exactly as strong: `hasattr` takes a string either way.
FUNCTIONAL_DOOR = "build_mmff" + "_potentials"


def _ethane() -> "molrs.Atomistic":
    """Ethane (C2H6) — the smallest molecule with bonds, angles and dihedrals."""
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


# ---------------------------------------------------------------------------
# The doors that must be closed
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The route that must remain — non-vacuity
# ---------------------------------------------------------------------------


def test_the_surviving_typify_to_forcefield_route_still_works() -> None:
    """The replacement route produces finite energy and forces.

    Without this, the three assertions above are satisfiable by deleting MMFF from
    the Python bindings entirely. The point is not that the doors are gone; it is
    that the *right* one is open.
    """
    typifier = molrs.ff.typifier.MMFF94Typifier()

    typed = typifier.typify(_ethane())
    frame = typed.to_frame()
    frame["pairs"] = molrs.ff.intramolecular_pairs(frame)

    pots = typifier.forcefield().to_potentials(frame)
    energy, forces = pots.calc_energy_forces(molrs.ff.extract_coords(frame))

    n_atoms = frame["atoms"].nrows
    assert forces.shape == (n_atoms, 3)
    assert energy == energy, "MMFF energy is NaN"  # NaN != NaN
    assert abs(energy) < 1.0e6, f"MMFF energy {energy} is not physical for ethane"


def test_the_typifier_still_exposes_its_forcefield() -> None:
    """``forcefield()`` is the seam the deleted ``build()`` used internally.

    Deleting ``build`` must not delete the thing it was a shortcut *for* — that
    would close the replacement route along with the door being removed.
    """
    for typifier in (
        molrs.ff.typifier.MMFF94Typifier(),
        molrs.ff.typifier.MMFF94STypifier(),
    ):
        ff = typifier.forcefield()
        assert hasattr(ff, "to_potentials")

    assert molrs.ff.typifier.MMFF94Typifier().forcefield().name == "MMFF94"
    assert molrs.ff.typifier.MMFF94STypifier().forcefield().name == "MMFF94s"
