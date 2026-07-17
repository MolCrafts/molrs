"""Python-binding coverage for MMFF graph typification."""

import molrs


def _ethane() -> "molrs.Atomistic":
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


def test_mmff_typify_returns_typed_atomistic_topology():
    typifier = molrs.MMFF94Typifier()
    typed = typifier.typify(_ethane())
    assert isinstance(typed, molrs.Atomistic)

    frame = typed.to_frame()
    assert frame["atoms"].nrows == 8
    assert frame["bonds"].nrows == 7
    assert frame["angles"].nrows == 12
    assert frame["dihedrals"].nrows == 9
    assert len(frame["atoms"]["charge"]) == 8
