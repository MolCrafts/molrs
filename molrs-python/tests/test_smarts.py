"""Python-binding coverage for the native SMARTS matcher + graph-edit conveniences.

Exposes the validated Rust ``SmartsPattern`` (``molrs/src/core/chem/smarts``) to
Python with map-keyed results, plus the graph-edit conveniences
(``remove_atom`` / ``remove_bond`` / ``set_bond_order`` / ``copy``) on
``Atomistic``. The matching algorithm itself is exercised in Rust; these tests
only assert the PyO3 surface and Daylight atom-map semantics.
"""

import pytest

import molrs


def _methanol() -> "molrs.Atomistic":
    """Methanol (CH3-OH) with explicit hydrogens.

    Connectivity: C-O, O-H(hydroxyl), plus three C-H. The C-O-H path is the
    single ``[C][O][H]`` embedding.
    """
    mol = molrs.Atomistic()
    c = mol.add_atom("C", 0.0, 0.0, 0.0)
    o = mol.add_atom("O", 1.4, 0.0, 0.0)
    ho = mol.add_atom("H", 2.0, 0.0, 0.0)
    mol.add_bond(c, o)
    mol.add_bond(o, ho)
    for pos in [(-0.4, 1.0, 0.0), (-0.4, -0.5, 0.9), (-0.4, -0.5, -0.9)]:
        h = mol.add_atom("H", *pos)
        mol.add_bond(c, h)
    return mol, c, o, ho


def _ethane() -> "molrs.Atomistic":
    """Ethane (C2H6) with explicit hydrogens (no O — for non-match tests)."""
    mol = molrs.Atomistic()
    c1 = mol.add_atom("C", 0.0, 0.0, 0.0)
    c2 = mol.add_atom("C", 1.54, 0.0, 0.0)
    for c, (x, y, z) in [
        (c1, (-0.36, 1.03, 0.0)),
        (c1, (-0.36, -0.51, 0.89)),
        (c1, (-0.36, -0.51, -0.89)),
        (c2, (1.90, 1.03, 0.0)),
        (c2, (1.90, -0.51, 0.89)),
        (c2, (1.90, -0.51, -0.89)),
    ]:
        h = mol.add_atom("H", x, y, z)
        mol.add_bond(c, h)
    mol.add_bond(c1, c2)
    return mol, c1, c2


def _methylamine() -> "molrs.Atomistic":
    """Methylamine (CH3-NH2) with explicit hydrogens; N carries 2 explicit H."""
    mol = molrs.Atomistic()
    c = mol.add_atom("C", 0.0, 0.0, 0.0)
    n = mol.add_atom("N", 1.47, 0.0, 0.0)
    mol.add_bond(c, n)
    for pos in [(1.9, 0.9, 0.0), (1.9, -0.9, 0.0)]:  # 2 H on N
        h = mol.add_atom("H", *pos)
        mol.add_bond(n, h)
    for pos in [(-0.4, 1.0, 0.0), (-0.4, -0.5, 0.9), (-0.4, -0.5, -0.9)]:  # 3 H on C
        h = mol.add_atom("H", *pos)
        mol.add_bond(c, h)
    return mol, n


# ---------------------------------------------------------------------------
# ac-001: SmartsPattern exposed; atom-map capture correct
# ---------------------------------------------------------------------------


def test_smarts_pattern_is_exposed():
    """molrs.SmartsPattern exists and parses a query."""
    assert "SmartsPattern" in dir(molrs)
    pat = molrs.SmartsPattern("[C:1][O:2][H:3]")
    assert pat is not None


def test_find_matches_mapped_captures_map_numbers():
    """find_matches_mapped returns list[dict[int,int]] keyed by atom-map number."""
    mol, c, o, ho = _methanol()
    pat = molrs.SmartsPattern("[C:1][O:2][H:3]")
    matches = pat.find_matches_mapped(mol)
    assert isinstance(matches, list)
    assert len(matches) == 1
    m = matches[0]
    assert set(m.keys()) == {1, 2, 3}
    # Handles map to the expected atoms / elements.
    assert m[1] == c
    assert m[2] == o
    assert m[3] == ho
    assert mol.get(m[1], "element") == "C"
    assert mol.get(m[2], "element") == "O"
    assert mol.get(m[3], "element") == "H"


def test_find_matches_mapped_empty_on_non_match():
    """A molecule lacking the group yields an empty match list."""
    mol, _c1, _c2 = _ethane()
    pat = molrs.SmartsPattern("[C:1][O:2][H:3]")
    assert pat.find_matches_mapped(mol) == []
    assert pat.has_match(mol) is False


def test_primary_amine_matches_NH2():
    """[N;H2:1] matches a primary amine nitrogen (2 explicit H)."""
    mol, n = _methylamine()
    pat = molrs.SmartsPattern("[N;H2:1]")
    matches = pat.find_matches_mapped(mol)
    assert len(matches) == 1
    assert matches[0] == {1: n}
    # Ethane (no N) does not match.
    ethane, _c1, _c2 = _ethane()
    assert pat.find_matches_mapped(ethane) == []


# ---------------------------------------------------------------------------
# ac-002: Daylight atom-map semantics — map adds no match constraint
# ---------------------------------------------------------------------------


def test_map_number_adds_no_constraint():
    """find_matches('[C:1]') and find_matches('[C]') return the same atom set."""
    mol, c1, c2 = _ethane()
    mapped = molrs.SmartsPattern("[C:1]").find_matches(mol)
    plain = molrs.SmartsPattern("[C]").find_matches(mol)
    # Each match is a single-atom list; compare the flat atom sets.
    assert {m[0] for m in mapped} == {m[0] for m in plain} == {c1, c2}


def test_map_label_and_num_query_atoms():
    """map_label / num_query_atoms expose the parsed query metadata."""
    pat = molrs.SmartsPattern("[C:1][O:2][H:3]")
    assert pat.num_query_atoms == 3
    assert pat.map_label(0) == 1
    assert pat.map_label(1) == 2
    assert pat.map_label(2) == 3

    plain = molrs.SmartsPattern("[C]")
    assert plain.num_query_atoms == 1
    assert plain.map_label(0) is None


# ---------------------------------------------------------------------------
# ac-003: PyAtomistic graph-edit conveniences
# ---------------------------------------------------------------------------


def test_remove_atom_cascades():
    """remove_atom drops the atom and its incident bonds."""
    mol, c, o, ho = _methanol()
    n_bonds_before = mol.n_relations("bonds")
    assert mol.n_atoms == 6
    mol.remove_atom(o)  # O is bonded to C and to H → 2 incident bonds cascade
    assert mol.n_atoms == 5
    assert mol.n_relations("bonds") == n_bonds_before - 2
    assert mol.has_entity(o) is False


def test_remove_bond():
    """remove_bond removes just the bond, leaving atoms intact."""
    mol, c, o, ho = _methanol()
    # Bond handle from the C-O bond via incident relations.
    (bond_handle, _other) = mol.incident_relations(c, "bonds")[0]
    n_atoms_before = mol.n_atoms
    n_bonds_before = mol.n_relations("bonds")
    mol.remove_bond(bond_handle)
    assert mol.n_relations("bonds") == n_bonds_before - 1
    assert mol.n_atoms == n_atoms_before  # atoms untouched


def test_set_bond_order():
    """set_bond_order updates the bond 'order' property."""
    mol, c, o, ho = _methanol()
    (bond_handle, _other) = mol.incident_relations(c, "bonds")[0]
    mol.set_bond_order(bond_handle, 2.0)
    assert mol.get_relation_prop("bonds", bond_handle, "order") == pytest.approx(2.0)


def test_copy_is_independent():
    """copy() yields an independent graph — mutating the copy leaves the original."""
    mol, c, o, ho = _methanol()
    n_atoms = mol.n_atoms
    n_bonds = mol.n_relations("bonds")
    dup = mol.copy()
    assert dup.n_atoms == n_atoms
    dup.remove_atom(c)
    # Original untouched.
    assert mol.n_atoms == n_atoms
    assert mol.n_relations("bonds") == n_bonds
    assert dup.n_atoms == n_atoms - 1
