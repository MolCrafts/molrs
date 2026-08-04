"""Contract tests for the ECS-shaped Python binding (molgraph-ecs-02-pybind).

The core is an ECS *world*: entities are stable opaque handles, components live
in aligned columns, and topology is kind-tagged relations. Rigid-body moves are
module-level free functions because they have no owning type; chemical
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


def test_get_missing_component_returns_none_and_type_conflict_raises():
    g = molrs.Graph()
    e = g.spawn()
    assert g.get(e, molrs.keys.X) is None  # absent
    g.set(e, molrs.keys.CHARGE, 1.0)
    with pytest.raises(Exception):
        g.set(e, molrs.keys.CHARGE, "not-a-number")  # type conflict


# --------------------------------------------------------------------------- #
# Rigid-body moves are free functions; perception is owned by a type          #
# --------------------------------------------------------------------------- #


def test_rigid_moves_are_module_level_free_functions():
    # translate/rotate act on any Graph and carry no state, so they have no
    # owning type and live on molrs::core as free functions.
    for name in ("translate", "rotate", "scale", "align_direction"):
        assert callable(getattr(molrs, name))
        assert not hasattr(molrs.Atomistic, name)
        assert not hasattr(molrs.Graph, name)


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
    molrs.translate(a, [10.0, 0.0, 0.0])
    assert a.get(h, molrs.keys.X) == 11.0


def test_translate_on_generic_graph():
    g = molrs.Graph()
    e = g.spawn()
    for k in molrs.keys.COORDS:
        g.set(e, k, 0.0)
    molrs.translate(g, [5.0, -1.0, 2.0])
    assert (g.get(e, "x"), g.get(e, "y"), g.get(e, "z")) == (5.0, -1.0, 2.0)


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
    assert molrs.keys.X == "x"
    assert molrs.keys.ELEMENT == "element"
    assert molrs.keys.CHARGE == "charge"
    assert list(molrs.keys.COORDS) == ["x", "y", "z"]


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
    molrs.scale(mol, [2.0, 2.0, 2.0], [1.0, 0.0, 0.0])
    assert [mol.get(h, "x") for h in handles] == [-1.0, 1.0, 3.0]


def test_scale_uniform_about_origin():
    mol = molrs.Atomistic()
    h = mol.spawn()
    mol.set(h, "x", 1.0)
    mol.set(h, "y", 2.0)
    mol.set(h, "z", 3.0)
    molrs.scale(mol, [0.5, 0.5, 0.5])
    assert (mol.get(h, "x"), mol.get(h, "y"), mol.get(h, "z")) == (0.5, 1.0, 1.5)
