import molrs


def test_atomistic_replicate_is_native_and_non_mutating():
    molecule = molrs.Atomistic()
    a = molecule.def_atom(element="C")
    b = molecule.def_atom(element="H")
    molecule.def_bond(a, b)
    output = molecule.replicate(3)
    assert len(output.atoms) == 6
    assert len(output.bonds) == 3
    assert len(molecule.atoms) == 2


def test_coarse_grain_replicate_is_native_and_non_mutating():
    molecule = molrs.CoarseGrain()
    molecule.def_bead(type="A")
    output = molecule.replicate(4)
    assert len(output.beads) == 4
    assert len(molecule.beads) == 1
