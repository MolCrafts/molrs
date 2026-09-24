"""Contract tests for the ECS-shaped Python binding (molgraph-ecs-02-pybind).

The core is an ECS *world*: entities are stable opaque handles, components live
in aligned columns, and topology is kind-tagged relations. Rigid-body moves
(translate, rotate, scale) are methods of the leaves and return the leaf; chemical
perception has owners (`molrs.perceive.Perceive` / `RingInfo`,
`molrs.ff.charge.*`, `molrs.io.SmilesIR`) and is reached through them, never
through a method on the graph classes. Leaves (`Atomistic`/`CoarseGrain`) hold a
core leaf from construction and subclass `Graph`; they are never *converted*
from a `MolGraph`.
"""

import numpy as np
import pytest

import molrs


# --------------------------------------------------------------------------- #
# Stable handles                                                              #
# --------------------------------------------------------------------------- #


def test_handles_are_stable_opaque_ints():
    g = molrs.Graph()
    handles = [g.spawn() for _ in range(3)]
    assert all(isinstance(h, int) for h in handles)
    assert len(set(handles)) == 3  # distinct


def test_despawn_middle_keeps_others_valid_no_reindex():
    g = molrs.Graph()
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
    g = molrs.Graph()
    e = g.spawn()
    g.despawn(e)
    with pytest.raises(Exception):
        g.set(e, "x", 1.0)


def test_relation_endpoints_survive_despawn_no_reindex():
    g = molrs.Graph()
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
    a = molrs.Atomistic()
    h0 = a.add_atom("C", 1.0, 2.0, 3.0)
    a.add_atom("O", 4.0, 5.0, 6.0)

    col = a.column(molrs.keys.X)
    assert isinstance(col, np.ndarray)
    assert col.tolist() == [1.0, 4.0]

    # Mutating the view writes through to the world.
    col[0] = 9.0
    assert a.get(h0, molrs.keys.X) == 9.0


def test_validity_mask_reflects_set_components():
    a = molrs.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    a.add_atom("O", 0.0, 0.0, 0.0)
    a.set(h0, molrs.keys.CHARGE, -0.5)

    v = a.validity(molrs.keys.CHARGE)
    assert v.dtype == np.bool_
    assert v.tolist() == [True, False]


def test_column_with_a_hole_is_a_key_error_not_a_zero_fill():
    a = molrs.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    a.add_atom("O", 0.0, 0.0, 0.0)
    a.set(h0, molrs.keys.CHARGE, -0.5)

    with pytest.raises(KeyError, match="1 of 2"):
        a.column(molrs.keys.CHARGE)
    with pytest.raises(KeyError):
        a.column("never_set")


def test_column_comes_back_in_the_component_type():
    a = molrs.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    h1 = a.add_atom("O", 0.0, 0.0, 0.0)
    for h, n, flag in ((h0, 3, True), (h1, -1, False)):
        a.set(h, "n", n)
        a.set(h, "flag", flag)

    assert a.column(molrs.keys.ELEMENT).tolist() == ["C", "O"]
    ints = a.column("n")
    assert ints.dtype == np.int32 and ints.tolist() == [3, -1]
    flags = a.column("flag")
    assert flags.dtype == np.bool_ and flags.tolist() == [True, False]
    # Non-f64 columns are copies: writing them does not reach the world.
    ints[0] = 99
    assert a.get(h0, "n") == 3


def test_columns_lists_every_registered_component():
    a = molrs.Atomistic()
    h0 = a.add_atom("C", 0.0, 0.0, 0.0)
    a.set(h0, molrs.keys.CHARGE, -0.5)  # partial columns are listed too

    cols = a.columns()
    assert isinstance(cols, list)
    assert {molrs.keys.X.key, molrs.keys.ELEMENT.key, molrs.keys.CHARGE.key} <= set(cols)
    assert molrs.Atomistic().columns() == []


def test_get_missing_component_returns_none_and_type_conflict_raises():
    g = molrs.Graph()
    e = g.spawn()
    assert g.get(e, molrs.keys.X) is None  # absent
    g.set(e, molrs.keys.CHARGE, 1.0)
    with pytest.raises(Exception):
        g.set(e, molrs.keys.CHARGE, "not-a-number")  # type conflict


# --------------------------------------------------------------------------- #
# Rigid-body moves are leaf methods; perception is owned by a type           #
# --------------------------------------------------------------------------- #


def test_translate_rotate_and_scale_are_methods_of_the_three_leaves():
    for cls in (molrs.Atomistic, molrs.CoarseGrain, molrs.Fragment):
        assert callable(getattr(cls, "translate"))
        assert callable(getattr(cls, "rotate"))
        assert callable(getattr(cls, "scale"))
    assert not hasattr(molrs, "translate")
    assert not hasattr(molrs, "rotate")
    assert not hasattr(molrs, "align_direction")
    assert not hasattr(molrs, "scale")


LEAVES = [molrs.Atomistic, molrs.CoarseGrain, molrs.Fragment]


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
    chained = mol.translate([1.0, 0.0, 0.0]).rotate([0.0, 0.0, 1.0], np.pi).scale(
        [0.5, 0.5, 0.5]
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


def test_placer_and_orienter_are_base_classes():
    # The variant is the class, never a flag: the motion is a Placer subclass
    # and the facing rule is an Orienter subclass.
    assert issubclass(molrs.TracePlacer, molrs.Placer)
    assert issubclass(molrs.LineOrienter, molrs.Orienter)
    assert issubclass(molrs.TangOrienter, molrs.Orienter)
    assert callable(molrs.TracePlacer().place)
    assert callable(molrs.LineOrienter().orient)
    assert callable(molrs.LineOrienter().direction)


def test_a_placer_subclass_may_take_its_own_constructor_arguments():
    class P(molrs.Placer):
        def __init__(self, cg):
            super().__init__()
            self.cg = cg

    cg = object()
    placer = P(cg)
    assert placer.cg is cg
    assert isinstance(placer, molrs.Placer)


def test_the_placer_base_place_is_not_implemented():
    with pytest.raises(NotImplementedError):
        molrs.Placer().place(molrs.Atomistic(), [])


def test_the_orienter_set_is_closed_at_class_creation():
    # The facing rule is a native class; a Python subclass could never be
    # applied, so it must be refused when it is defined, not when it is used.
    with pytest.raises(TypeError):

        class Custom(molrs.Orienter):
            pass


@pytest.mark.parametrize("orienter_type", ["LineOrienter", "TangOrienter"])
def test_an_overflowing_site_axis_has_no_direction_and_never_panics(orienter_type):
    orienter = getattr(molrs, orienter_type)()
    try:
        result = orienter.direction([1e200, 0.0, 0.0])
    except BaseException as error:  # PanicException derives from BaseException
        pytest.fail(f"{orienter_type}.direction raised {type(error).__name__}: {error}")
    assert result is None, f"{orienter_type}.direction gave {result}"


def test_trace_placer_place_maps_an_unreachable_fragment_to_value_error():
    # Four two-atom fragments; the bonds join 1-2 and 3-4 separately, so
    # fragment 3 has no path from the root and the native PlaceError surfaces.
    mol = molrs.Atomistic()
    fragments = []
    for k in range(1, 5):
        head = mol.add_atom("C", 4.0 * k, 0.0, 0.0)
        tail = mol.add_atom("O", 4.0 * k + 1.4, 0.0, 0.0)
        mol.add_bond(head, tail)
        for h in (head, tail):
            mol.set(h, "res_id", k)
        fragments.append((head, tail))
    bonds = [(fragments[0][1], fragments[1][0]), (fragments[2][1], fragments[3][0])]
    with pytest.raises(ValueError, match="fragment 3"):
        molrs.TracePlacer().place(mol, bonds)


def test_a_place_error_names_a_node_by_its_python_handle():
    mol = molrs.Atomistic()
    grouped = mol.def_atom(element="O", x=0.0, y=0.0, z=0.0, res_id=1)
    stray = mol.def_atom(element="C", x=1.4, y=0.0, z=0.0)
    with pytest.raises(ValueError) as caught:
        molrs.TracePlacer().place(mol, [(grouped.handle, stray.handle)])
    message = str(caught.value)
    assert str(stray.handle) in message
    assert "NodeId(" not in message


# --------------------------------------------------------------------------- #
# SiteMap: node views and handles are interchangeable                          #
# --------------------------------------------------------------------------- #


def _carbons(n):
    mol = molrs.Atomistic()
    atoms = [mol.def_atom(element="C", x=1.5 * k, y=0.0, z=0.0) for k in range(n)]
    return mol, atoms


def test_site_map_label_accepts_a_node_view():
    mol, (atom,) = _carbons(1)
    molrs.SiteMap(mol).label(atom, "a")
    assert mol.get(atom.handle, "site") == "a"


def test_site_map_label_atoms_accepts_node_views():
    mol, (atom,) = _carbons(1)
    marked = molrs.SiteMap(mol).label_atoms([atom], "b")
    assert marked == [atom.handle]
    assert mol.get(atom.handle, "site") == "b"


def test_site_map_every_nth_accepts_node_views():
    mol, atoms = _carbons(4)
    marked = molrs.SiteMap(mol).every_nth(atoms, 2, "s")
    assert marked == [atoms[0].handle, atoms[2].handle]


def test_site_map_clear_accepts_node_views():
    mol, atoms = _carbons(2)
    sites = molrs.SiteMap(mol)
    sites.label_atoms([a.handle for a in atoms], "a", "b")
    sites.clear([atoms[0]])
    assert mol.get(atoms[0].handle, "site") in (None, "")
    assert mol.get(atoms[1].handle, "site") == "b"


def test_site_map_label_elements_error_names_the_element():
    mol, _ = _carbons(1)
    with pytest.raises(ValueError, match="Cl"):
        molrs.SiteMap(mol).label_elements("Cl", "x")


def test_a_site_error_names_a_node_by_its_python_handle():
    mol, (atom,) = _carbons(1)
    with pytest.raises(ValueError) as caught:
        molrs.SiteMap(mol).every_nth([atom.handle], 1, "s", leaving="h")
    message = str(caught.value)
    assert str(atom.handle) in message
    assert "NodeId(" not in message


def test_find_rings_system():
    bz = molrs.perceive.Perceive().find_hydrogens(
        molrs.io.SmilesIR("C1=CC=CC=C1").to_atomistic()
    )
    rings = molrs.perceive.RingInfo(bz).rings()
    assert len(rings) == 1
    assert len(rings[0]) == 6  # six-membered ring


def test_gasteiger_charges_system():
    eth = molrs.perceive.Perceive().find_hydrogens(
        molrs.io.SmilesIR("CO").to_atomistic()
    )
    charges = np.asarray(molrs.ff.charge.GasteigerModel().assign(eth))
    # One charge per atom, hydrogens included, and neutral methanol sums to ~0.
    assert charges.shape == (len(eth.entities()),)
    assert charges.sum() == pytest.approx(0.0, abs=1e-6)


def test_translate_operates_on_leaf_own_graph_not_empty_base():
    a = molrs.Atomistic()
    h = a.add_atom("C", 1.0, 0.0, 0.0)
    a.translate([10.0, 0.0, 0.0])
    assert a.get(h, molrs.keys.X) == 11.0


def test_generic_graph_has_no_translate():
    assert not hasattr(molrs.Graph, "translate")


def test_perceive_aromaticity_pipeline():
    # Aromaticity perception needs explicit hydrogens (pi-electron counting).
    perceive = molrs.perceive.Perceive()
    bz = perceive.find_hydrogens(molrs.io.SmilesIR("C1=CC=CC=C1").to_atomistic())
    bz = perceive.find_aromaticity(bz)
    aromatic = [h for h in bz.entities() if bz.get(h, "is_aromatic")]
    assert len(aromatic) == 6


# --------------------------------------------------------------------------- #
# Leaves subclass Graph; hold a core leaf; never converted                   #
# --------------------------------------------------------------------------- #


def test_leaf_is_subclass_and_instantiable():
    assert issubclass(molrs.Atomistic, molrs.Graph)
    assert issubclass(molrs.CoarseGrain, molrs.Graph)

    class S(molrs.Atomistic):
        pass

    s = S()  # `subclass` fixes the historical TypeError
    assert isinstance(s, molrs.Atomistic)
    assert isinstance(s, molrs.Graph)


def test_leaf_generic_api_uses_its_own_graph():
    a = molrs.Atomistic()
    a.add_atom("C", 0.0, 0.0, 0.0)
    a.add_atom("O", 0.0, 0.0, 0.0)
    # The generic ECS API reflects the leaf's own atoms, not an empty base.
    assert len(a.entities()) == 2
    assert a.n_nodes == 2


def test_leaf_frame_round_trip():
    a = molrs.Atomistic()
    h1 = a.add_atom("C", 0.0, 0.0, 0.0)
    h2 = a.add_atom("O", 1.2, 0.0, 0.0)
    a.add_bond(h1, h2)

    frame = a.to_frame()
    a2 = molrs.Atomistic.from_frame(frame)
    assert a2.n_atoms == 2
    assert a2.n_relations("bonds") == 1


# --------------------------------------------------------------------------- #
# adopt — zero-copy move                                                      #
# --------------------------------------------------------------------------- #


def test_adopt_moves_storage_and_empties_source():
    src = molrs.Graph()
    s0 = src.spawn()
    src.set(s0, "x", 7.0)

    dst = molrs.Graph()
    dst.adopt(src)

    assert dst.has_entity(s0)
    assert dst.get(s0, "x") == 7.0
    assert len(src.entities()) == 0  # source emptied


def test_adopt_on_leaf_moves_its_own_store():
    # adopt must move the *leaf's* backing store, not an empty base graph —
    # regression for adopt being a no-op on Atomistic/CoarseGrain.
    src = molrs.io.SmilesIR("CCO").to_atomistic()
    n = src.n_atoms
    assert n == 3

    dst = molrs.Atomistic()
    dst.adopt(src)

    assert dst.n_atoms == n
    assert src.n_atoms == 0  # source emptied


# --------------------------------------------------------------------------- #
# keys convention                                                            #
# --------------------------------------------------------------------------- #


def test_keys_convention_exposed():
    assert isinstance(molrs.keys.X, molrs.keys.Key)
    assert molrs.keys.X.key == "x"
    assert molrs.keys.ELEMENT.key == "element"
    assert molrs.keys.CHARGE.key == "charge"
    # Key equals its string form for convenient comparisons.
    assert molrs.keys.X == "x"
    assert [k.key for k in molrs.keys.COORDS] == ["x", "y", "z"]
    by_str = molrs.schema.column("atomic_number")
    by_key = molrs.schema.column(molrs.keys.ATOMIC_NUMBER)
    assert by_str is not None and by_key is not None
    assert by_key.key == by_str.key
    assert by_key.dtype == by_str.dtype


# --------------------------------------------------------------------------- #
# relation enumeration + geometry systems (P0-C)                              #
# --------------------------------------------------------------------------- #


def test_relation_ids_enumerates_handles():
    """Authoritative enumeration — replaces probing opaque handle ranges."""
    mol = molrs.Atomistic()
    a, b, c = mol.spawn(), mol.spawn(), mol.spawn()
    mol.register_kind("bond", 2)
    rh1 = mol.add_relation("bond", [a, b])
    rh2 = mol.add_relation("bond", [b, c])
    assert set(mol.relation_ids("bond")) == {rh1, rh2}
    assert mol.n_relations("bond") == 2


def test_relation_ids_empty_for_registered_kind():
    mol = molrs.Atomistic()
    mol.register_kind("angle", 3)
    assert mol.relation_ids("angle") == []


def test_relation_ids_unregistered_kind_raises():
    mol = molrs.Atomistic()
    with pytest.raises(ValueError):
        mol.relation_ids("nope")


def test_scale_about_center():
    mol = molrs.Atomistic()
    handles = [mol.spawn() for _ in range(3)]
    for i, h in enumerate(handles):
        mol.set(h, "x", float(i))
        mol.set(h, "y", 0.0)
        mol.set(h, "z", 0.0)
    mol.scale([2.0, 2.0, 2.0], [1.0, 0.0, 0.0])
    assert [mol.get(h, "x") for h in handles] == [-1.0, 1.0, 3.0]


def test_scale_uniform_about_origin():
    mol = molrs.Atomistic()
    h = mol.spawn()
    mol.set(h, "x", 1.0)
    mol.set(h, "y", 2.0)
    mol.set(h, "z", 3.0)
    mol.scale([0.5, 0.5, 0.5])
    assert (mol.get(h, "x"), mol.get(h, "y"), mol.get(h, "z")) == (0.5, 1.0, 1.5)
