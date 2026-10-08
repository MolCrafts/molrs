"""Contract tests for the ECS-shaped Python binding (molgraph-ecs-02-pybind).

The core is an ECS *world*: entities are stable opaque handles, components live
in aligned columns, and topology is kind-tagged relations. Rigid-body moves
(translate, rotate, scale) are methods of the leaves and return the leaf; chemical
perception has owners (`molrs.perceive.assign_*` / `perceive_rings`,
`molrs.ff.charge.*`, `molrs.io.smiles.SmilesIr`) and is reached through them, never
through a method on the graph classes. Leaves (`Atomistic`/`CoarseGrain`) hold a
core leaf from construction and subclass `MolGraph`; they are never *converted*
from a `MolGraph`.
"""

import molrs
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Stable handles                                                              #
# --------------------------------------------------------------------------- #


def test_handles_are_stable_opaque_ints():
    g = molrs.core.MolGraph()
    handles = [g.spawn() for _ in range(3)]
    assert all(isinstance(h, int) for h in handles)
    assert len(set(handles)) == 3  # distinct


def test_despawn_middle_keeps_others_valid_no_reindex():
    g = molrs.core.MolGraph()
    e0, e1, e2 = g.spawn(), g.spawn(), g.spawn()
    g.set(e0, "x", 0.0)
    g.set(e2, "x", 2.0)

    g.despawn(e1)

    # The surviving handles still resolve to *their own* data — no positional
    # reindexing shifted e2 into e1's slot.
    assert g.has_entity(e0) and g.has_entity(e2)
    assert not g.has_entity(e1)
    assert g.get(e0, "x") == 0.0
    assert g.get(e2, "x") == 2.0
    assert sorted(g.entities()) == sorted([e0, e2])


def test_stale_handle_raises():
    g = molrs.core.MolGraph()
    e = g.spawn()
    g.despawn(e)
    with pytest.raises(ValueError):
        g.set(e, "x", 1.0)


def test_relation_endpoints_survive_despawn_no_reindex():
    g = molrs.core.MolGraph()
    g.register_kind("link", 2)
    a, b, c = g.spawn(), g.spawn(), g.spawn()
    r = g.add_relation("link", [a, c])

    g.despawn(b)  # remove a non-endpoint entity

    # The relation's endpoints still resolve to a and c — handles weren't
    # shifted by the swap-remove.
    assert g.relation_nodes("link", r) == [a, c]
    assert g.get(a, "x") is None  # a still a valid (unset) entity
    assert g.has_entity(a) and g.has_entity(c)


# --------------------------------------------------------------------------- #
# Zero-copy component columns                                                 #
# --------------------------------------------------------------------------- #


def test_column_is_zero_copy_view_write_through():
    a = molrs.core.Atomistic()
    h0 = a.add_atom("C", 1.0, 2.0, 3.0)
    a.add_atom("O", 4.0, 5.0, 6.0)

    col = a.column(molrs.core.keys.X)
    assert isinstance(col, np.ndarray)
    assert col.tolist() == [1.0, 4.0]

    # Mutating the view writes through to the world.
    col[0] = 9.0
    assert a.get(h0, molrs.core.keys.X) == 9.0


def test_validity_mask_reflects_set_components():
    a = molrs.core.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    a.add_atom("O", 0.0, 0.0, 0.0)
    a.set(h0, molrs.core.keys.CHARGE, -0.5)

    v = a.validity(molrs.core.keys.CHARGE)
    assert v.dtype == np.bool_
    assert v.tolist() == [True, False]


def test_column_with_a_hole_is_a_key_error_not_a_zero_fill():
    a = molrs.core.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    a.add_atom("O", 0.0, 0.0, 0.0)
    a.set(h0, molrs.core.keys.CHARGE, -0.5)

    with pytest.raises(KeyError, match="1 of 2"):
        a.column(molrs.core.keys.CHARGE)
    with pytest.raises(KeyError):
        a.column("never_set")


def test_column_comes_back_in_the_component_type():
    a = molrs.core.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    h1 = a.add_atom("O", 0.0, 0.0, 0.0)
    for h, n, flag in ((h0, 3, True), (h1, -1, False)):
        a.set(h, "n", n)
        a.set(h, "flag", flag)

    assert a.column(molrs.core.keys.ELEMENT).tolist() == ["C", "O"]
    ints = a.column("n")
    assert ints.dtype == np.int32 and ints.tolist() == [3, -1]
    flags = a.column("flag")
    assert flags.dtype == np.bool_ and flags.tolist() == [True, False]
    # Non-f64 columns are copies: writing them does not reach the world.
    ints[0] = 99
    assert a.get(h0, "n") == 3


def test_columns_lists_every_registered_component():
    a = molrs.core.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    a.set(h0, molrs.core.keys.CHARGE, -0.5)  # partial columns are listed too

    cols = a.columns()
    assert isinstance(cols, list)
    assert {molrs.core.keys.X.key, molrs.core.keys.ELEMENT.key, molrs.core.keys.CHARGE.key} <= set(
        cols
    )
    assert molrs.core.Atomistic().columns() == []


def test_get_missing_component_returns_none_and_type_conflict_raises():
    g = molrs.core.MolGraph()
    e = g.spawn()
    assert g.get(e, molrs.core.keys.X) is None  # absent
    g.set(e, molrs.core.keys.CHARGE, 1.0)
    with pytest.raises(ValueError):
        g.set(e, molrs.core.keys.CHARGE, "not-a-number")  # type conflict


# --------------------------------------------------------------------------- #
# Rigid-body moves are leaf methods; perception is owned by a type           #
# --------------------------------------------------------------------------- #


def test_translate_rotate_and_scale_are_methods_of_the_two_leaves():
    for cls in (molrs.core.Atomistic, molrs.core.CoarseGrain):
        assert callable(cls.translate)
        assert callable(cls.rotate)
        assert callable(cls.scale)


LEAVES = [molrs.core.Atomistic, molrs.core.CoarseGrain]


def _one_node(cls):
    mol = cls()
    h = mol.spawn()
    for key, value in zip(("x", "y", "z"), (1.0, 0.0, 0.0)):
        mol.set(h, key, value)
    return mol, h


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_translate_returns_the_leaf_itself(cls):
    mol, h = _one_node(cls)
    assert mol.translate([1.0, 0.0, 0.0]) is mol
    assert mol.get(h, "x") == 2.0


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_rotate_returns_the_leaf_itself(cls):
    mol, h = _one_node(cls)
    assert mol.rotate([0.0, 0.0, 1.0], np.pi / 2) is mol
    assert mol.get(h, "y") == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_scale_returns_the_leaf_itself(cls):
    mol, h = _one_node(cls)
    assert mol.scale([2.0, 2.0, 2.0]) is mol
    assert mol.get(h, "x") == 2.0


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_rigid_body_moves_chain(cls):
    mol, h = _one_node(cls)
    chained = (
        mol.translate([1.0, 0.0, 0.0])
        .rotate([0.0, 0.0, 1.0], np.pi)
        .scale([0.5, 0.5, 0.5])
    )
    assert chained is mol
    assert mol.get(h, "x") == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize(
    "axis",
    [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]],
    ids=["zero", "nan", "inf"],
)
@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_rotate_about_a_degenerate_axis_is_a_value_error(cls, axis):
    mol, h = _one_node(cls)
    with pytest.raises(ValueError):
        mol.rotate(axis, 1.0)
    assert (mol.get(h, "x"), mol.get(h, "y"), mol.get(h, "z")) == (1.0, 0.0, 0.0)


def test_find_rings_system():
    bz = molrs.perceive.add_hydrogens(
        molrs.io.smiles.SmilesIr("C1=CC=CC=C1").to_atomistic()
    )
    rings = molrs.perceive.perceive_rings(bz).rings()
    assert len(rings) == 1
    assert len(rings[0]) == 6  # six-membered ring


def test_gasteiger_charges_system():
    eth = molrs.perceive.add_hydrogens(
        molrs.io.smiles.SmilesIr("CO").to_atomistic()
    )
    charges = np.asarray(molrs.ff.charge.GasteigerModel().assign(eth))
    # One charge per atom, hydrogens included, and neutral methanol sums to ~0.
    assert charges.shape == (len(eth.entities()),)
    assert charges.sum() == pytest.approx(0.0, abs=1e-6)


def test_translate_operates_on_leaf_own_graph_not_empty_base():
    a = molrs.core.Atomistic()
    h = a.add_atom("C", 1.0, 0.0, 0.0)
    a.translate([10.0, 0.0, 0.0])
    assert a.get(h, molrs.core.keys.X) == 11.0


def test_perceive_aromaticity_pipeline():
    # Aromaticity perception needs explicit hydrogens (pi-electron counting).
    bz = molrs.perceive.add_hydrogens(molrs.io.smiles.SmilesIr("C1=CC=CC=C1").to_atomistic())
    bz = molrs.perceive.assign_aromaticity(bz)
    aromatic = [h for h in bz.entities() if bz.get(h, "is_aromatic")]
    assert len(aromatic) == 6


# --------------------------------------------------------------------------- #
# Leaves subclass MolGraph; hold a core leaf; never converted                #
# --------------------------------------------------------------------------- #


def test_leaf_is_a_graph():
    assert issubclass(molrs.core.Atomistic, molrs.core.MolGraph)
    assert issubclass(molrs.core.CoarseGrain, molrs.core.MolGraph)


def test_leaf_generic_api_uses_its_own_graph():
    a = molrs.core.Atomistic()
    a.add_atom("C", 0.0, 0.0, 0.0)
    a.add_atom("O", 0.0, 0.0, 0.0)
    # The generic ECS API reflects the leaf's own atoms, not an empty base.
    assert len(a.entities()) == 2
    assert a.n_nodes == 2


def test_leaf_frame_round_trip():
    a = molrs.core.Atomistic()
    h1 = a.add_atom("C", 0.0, 0.0, 0.0)
    h2 = a.add_atom("O", 1.2, 0.0, 0.0)
    a.add_bond(h1, h2)

    frame = a.to_frame()
    a2 = molrs.core.Atomistic.from_frame(frame)
    assert a2.n_atoms == 2
    assert a2.n_relations("bonds") == 1
    assert a2.n_bonds == 1


def test_find_rotatable_unknown_bond_policy():
    # Butane skeleton; the middle bond's class is cleared to "unknown" (0, 0).
    mol = molrs.core.Atomistic()
    c = [mol.add_atom("C", float(i), 0.0, 0.0) for i in range(4)]
    bonds = [mol.add_bond(c[i], c[i + 1]) for i in range(3)]
    mol.set_bond_class(bonds[1], 0, 0)

    def middle_flag(out):
        # A bond handle names a relation: `get` would read the node that
        # happens to share its slot.
        return out.get_relation_prop("bonds", bonds[1], "is_rotatable")
    assert middle_flag(molrs.perceive.assign_rotatable_bonds(mol)) == 0
    assert middle_flag(molrs.perceive.assign_rotatable_bonds(mol, unknown_bond="single")) == 1
    with pytest.raises(ValueError):
        molrs.perceive.assign_rotatable_bonds(mol, unknown_bond="guess")


# --------------------------------------------------------------------------- #
# adopt — zero-copy move                                                      #
# --------------------------------------------------------------------------- #


def test_adopt_moves_storage_and_empties_source():
    src = molrs.core.MolGraph()
    s0 = src.spawn()
    src.set(s0, "x", 7.0)

    dst = molrs.core.MolGraph()
    dst.adopt(src)

    assert dst.has_entity(s0)
    assert dst.get(s0, "x") == 7.0
    assert len(src.entities()) == 0  # source emptied


def test_adopt_on_leaf_moves_its_own_store():
    # adopt must move the *leaf's* backing store, not an empty base graph, so
    # adopt on Atomistic/CoarseGrain carries the atoms over.
    src = molrs.io.smiles.SmilesIr("CCO").to_atomistic()
    n = src.n_atoms
    assert n == 3

    dst = molrs.core.Atomistic()
    dst.adopt(src)

    assert dst.n_atoms == n
    assert src.n_atoms == 0  # source emptied


# --------------------------------------------------------------------------- #
# keys convention                                                            #
# --------------------------------------------------------------------------- #


def test_keys_convention_exposed():
    assert isinstance(molrs.core.keys.X, molrs.core.keys.Key)
    assert molrs.core.keys.X.key == "x"
    assert molrs.core.keys.ELEMENT.key == "element"
    assert molrs.core.keys.CHARGE.key == "charge"
    # Key equals its string form for convenient comparisons.
    assert molrs.core.keys.X == "x"
    assert [k.key for k in molrs.core.keys.COORDS] == ["x", "y", "z"]
    by_str = molrs.core.schema.column("atomic_number")
    by_key = molrs.core.schema.column(molrs.core.keys.ATOMIC_NUMBER)
    assert by_str is not None and by_key is not None
    assert by_key.key == by_str.key
    assert by_key.dtype == by_str.dtype


def test_column_spec_exposes_dimension_and_the_unit_derived_from_it():
    # SchemaDocument: `x` has dimension "length", whose real-preset unit is
    # "angstrom" (molrs/src/core/schema/document.rs).
    x = molrs.core.schema.column("x")
    assert x is not None
    assert x.dimension == "length"
    assert x.unit == "angstrom"


# --------------------------------------------------------------------------- #
# relation enumeration + geometry systems (P0-C)                              #
# --------------------------------------------------------------------------- #


def test_relation_ids_enumerates_handles():
    """Authoritative enumeration — replaces probing opaque handle ranges."""
    mol = molrs.core.Atomistic()
    a, b, c = mol.spawn(), mol.spawn(), mol.spawn()
    mol.register_kind("bond", 2)
    rh1 = mol.add_relation("bond", [a, b])
    rh2 = mol.add_relation("bond", [b, c])
    assert set(mol.relation_ids("bond")) == {rh1, rh2}
    assert mol.n_relations("bond") == 2


def test_relation_ids_empty_for_registered_kind():
    mol = molrs.core.Atomistic()
    mol.register_kind("angle", 3)
    assert mol.relation_ids("angle") == []


def test_relation_ids_unregistered_kind_raises():
    mol = molrs.core.Atomistic()
    with pytest.raises(ValueError):
        mol.relation_ids("nope")


def test_scale_about_center():
    mol = molrs.core.Atomistic()
    handles = [mol.spawn() for _ in range(3)]
    for i, h in enumerate(handles):
        mol.set(h, "x", float(i))
        mol.set(h, "y", 0.0)
        mol.set(h, "z", 0.0)
    mol.scale([2.0, 2.0, 2.0], [1.0, 0.0, 0.0])
    assert [mol.get(h, "x") for h in handles] == [-1.0, 1.0, 3.0]


def test_scale_uniform_about_origin():
    mol = molrs.core.Atomistic()
    h = mol.spawn()
    mol.set(h, "x", 1.0)
    mol.set(h, "y", 2.0)
    mol.set(h, "z", 3.0)
    mol.scale([0.5, 0.5, 0.5])
    assert (mol.get(h, "x"), mol.get(h, "y"), mol.get(h, "z")) == (0.5, 1.0, 1.5)


# --------------------------------------------------------------------------- #
# center — a query on each leaf (backmap-primitives-07)                       #
# --------------------------------------------------------------------------- #
#
# Seam only: the mass-weighted centre R = sum(m_i r_i) / sum(m_i) and every
# refusal are proven by molrs/src/op/geometry.rs. These tests check
# the call shape per leaf, the ndarray that crosses back, and the error mapping
# (every CenterError is a ValueError naming the int handle, amendment 1).


def _one_weighted_node(cls, position, mass):
    """A leaf holding one node at ``position`` (Å) with ``mass`` (g/mol)."""
    mol = cls()
    h = mol.spawn()
    for key, value in zip(("x", "y", "z"), position):
        mol.set(h, key, value)
    mol.set(h, "mass", mass)
    return mol, h


def _center(mol, handles):
    # Atomistic centres all its own nodes; CoarseGrain centres
    # the bead group it is given.
    if isinstance(mol, molrs.core.CoarseGrain):
        return mol.center(handles)
    return mol.center()


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_center_returns_a_float64_triple_of_the_leafs_own_nodes(cls):
    mol, h = _one_weighted_node(cls, (1.0, 2.0, 3.0), 1.0)

    center = _center(mol, [h])

    assert isinstance(center, np.ndarray)
    assert center.dtype == np.float64
    assert center.shape == (3,)
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0], rtol=0, atol=1e-12)


def test_center_of_an_unknown_bead_is_a_value_error_naming_the_handle():
    cg, live = _one_weighted_node(molrs.core.CoarseGrain, (0.0, 0.0, 0.0), 1.0)
    stale = cg.spawn()
    cg.despawn(stale)

    with pytest.raises(ValueError) as excinfo:
        cg.center([live, stale])

    message = str(excinfo.value)
    assert str(stale) in message
    assert "NodeId(" not in message


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_center_with_a_non_finite_mass_is_a_value_error_naming_the_handle(cls):
    mol, h = _one_weighted_node(cls, (1.0, 2.0, 3.0), float("nan"))

    with pytest.raises(ValueError) as excinfo:
        _center(mol, [h])

    message = str(excinfo.value)
    assert str(h) in message
    assert "NodeId(" not in message


def test_to_frame_keeps_only_the_requested_atom_fields():
    mol = molrs.core.Atomistic()
    mol.def_atom(element="O", x=0.0, y=0.0, z=0.0, charge=-0.8)
    frame = mol.to_frame(atom_fields=["element", "x"])
    assert set(frame["atoms"].keys()) == {"element", "x"}
    with pytest.raises(ValueError, match="'mass'"):
        mol.to_frame(atom_fields=["x", "mass"])


def test_perception_is_free_functions():
    # Two verbs: `perceive_*` reports a side table, `assign_*` writes a clone.
    mol = molrs.io.smiles.SmilesIr("C1CC1C").to_atomistic()
    info = molrs.perceive.perceive_rings(mol)
    assert info.ring_sizes() == [3]
    with pytest.raises(TypeError):
        molrs.perceive.RingSet(mol)
    flagged = molrs.perceive.assign_rings(mol)
    in_ring = sorted(flagged.get(h, "is_in_ring") for h in flagged.entities())
    assert in_ring == [0, 1, 1, 1]
    assert not any(mol.has(h, "is_in_ring") for h in mol.entities())
