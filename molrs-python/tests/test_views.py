import pytest

import molrs


def test_public_graph_leaf_owns_the_view_api() -> None:
    graph = molrs.Atomistic()

    assert isinstance(graph, molrs.GraphViews)
    assert graph.nodes == []
    assert graph.links.all() == []
    assert callable(graph.def_atom)


def test_native_graph_out_paths_keep_the_public_view_type() -> None:
    from_smiles = molrs.parse_smiles("CO").to_atomistic()
    from_copy = from_smiles.copy()
    from_frame = molrs.Atomistic.from_frame(from_smiles.to_frame())
    from_perception = molrs.Perceive().find_rings(from_smiles)

    for graph in (from_smiles, from_copy, from_frame, from_perception):
        assert type(graph) is molrs.Atomistic
        assert isinstance(graph, molrs.GraphViews)
        assert len(graph.atoms) == 2


def test_public_coarse_grain_factories_and_graph_out_paths() -> None:
    graph = molrs.CoarseGrain(label="cg")
    a = graph.def_bead(bead_type="P1", xyz=[0.0, 0.0, 0.0])
    b = graph.def_bead(bead_type="P1", xyz=[1.0, 0.0, 0.0])
    bond = graph.def_cgbond(a, b, order=1.0)

    assert graph.beads[0] is a
    assert graph.cgbonds[0] is bond
    for result in (graph.copy(), molrs.CoarseGrain.from_frame(graph.to_frame())):
        assert type(result) is molrs.CoarseGrain
        assert isinstance(result, molrs.GraphViews)
        assert len(result.beads) == 2


def test_factories_return_interned_live_refs() -> None:
    graph = molrs.Atomistic()
    carbon = graph.def_atom(element="C", xyz=[0.0, 0.0, 0.0])
    oxygen = graph.def_atom(element="O", xyz=[1.0, 0.0, 0.0])
    bond = graph.def_bond(carbon, oxygen, order=2.0)

    assert graph.nodes[0] is carbon
    assert graph.links.all()[0] is bond
    assert bond.itom is carbon
    assert bond.jtom is oxygen
    assert carbon["xyz"] == [0.0, 0.0, 0.0]
    assert bond["order"] == 2.0


def test_refs_have_no_detached_constructor_form() -> None:
    with pytest.raises(TypeError):
        molrs.Atom(element="C")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        molrs.Bond(object(), object())  # type: ignore[call-arg]


def test_cross_world_relation_is_rejected() -> None:
    left = molrs.Atomistic()
    right = molrs.Atomistic()
    a = left.def_atom(element="C")
    b = right.def_atom(element="C")

    with pytest.raises(ValueError, match="belong to this graph"):
        left.def_bond(a, b)


def test_removed_ref_stays_live_and_becomes_stale() -> None:
    graph = molrs.Atomistic()
    atom = graph.def_atom(element="C")
    graph._remove_node(atom)

    with pytest.raises(Exception):
        _ = atom["element"]


def test_unregistered_native_relation_kind_is_visible_and_removable() -> None:
    graph = molrs.Atomistic()
    a = graph.def_atom(element="C")
    b = graph.def_atom(element="C")
    graph.register_kind("constraints", 2)
    graph.add_relation("constraints", [a.handle, b.handle])

    links = graph.links.all()
    assert len(links) == 1
    assert type(links[0]) is molrs.RelationRef
    assert links[0].kind == "constraints"

    graph.del_atom(a)
    assert not graph.has_entity(a.handle)
    assert graph.n_relations("constraints") == 0


def test_custom_relation_view_registration_is_per_world() -> None:
    class Constraint(molrs.RelationRef[molrs.Atom]):
        _kind = "constraints"
        _arity = 2

    typed = molrs.Atomistic()
    typed.links.register_type(Constraint)
    a = typed.def_atom(element="C")
    b = typed.def_atom(element="C")
    typed.add_relation("constraints", [a.handle, b.handle])

    assert len(typed.links[Constraint]) == 1
    assert type(typed.links.all()[0]) is Constraint

    other = molrs.Atomistic()
    assert "constraints" not in other.kinds()
