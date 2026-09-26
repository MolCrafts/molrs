import numpy as np
import pytest

import molrs


def test_public_graph_leaf_owns_the_view_api() -> None:
    graph = molrs.Atomistic()

    assert isinstance(graph, molrs.GraphViews)
    assert graph.nodes == []
    assert graph.links.all() == []
    assert callable(graph.def_atom)


def test_native_graph_out_paths_keep_the_public_view_type() -> None:
    from_smiles = molrs.io.SmilesIR("CO").to_atomistic()
    from_copy = from_smiles.copy()
    from_frame = molrs.Atomistic.from_frame(from_smiles.to_frame())
    from_perception = molrs.perceive.Perceive().find_rings(from_smiles)

    for graph in (from_smiles, from_copy, from_frame, from_perception):
        assert type(graph) is molrs.Atomistic
        assert isinstance(graph, molrs.GraphViews)
        assert len(graph.atoms) == 2


def test_public_coarse_grain_factories_and_graph_out_paths() -> None:
    graph = molrs.CoarseGrain(label="cg")
    a = graph.def_bead(bead_type="P1", x=0.0, y=0.0, z=0.0)
    b = graph.def_bead(bead_type="P1", x=1.0, y=0.0, z=0.0)
    bond = graph.def_cgbond(a, b, order=1.0)

    assert graph.beads[0] is a
    assert graph.cgbonds[0] is bond
    for result in (graph.copy(), molrs.CoarseGrain.from_frame(graph.to_frame())):
        assert type(result) is molrs.CoarseGrain
        assert isinstance(result, molrs.GraphViews)
        assert len(result.beads) == 2


def test_coarse_grain_from_atom_frame_reads_atoms_as_beads_and_bonds_as_cg_bonds() -> None:
    # A LAMMPS-style atom frame: bond endpoints are 0-based `atoms` rows.
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, 1.0, 2.0], dtype=np.float64))
    atoms.insert("y", np.zeros(3, dtype=np.float64))
    atoms.insert("z", np.zeros(3, dtype=np.float64))
    atoms.insert("type", ["A", "B", "C"])
    bonds = molrs.Block()
    bonds.insert("atomi", np.array([0, 1], dtype=np.uint64))
    bonds.insert("atomj", np.array([2, 2], dtype=np.uint64))
    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds

    cg = molrs.CoarseGrain.from_atom_frame(frame, "type")

    assert type(cg) is molrs.CoarseGrain
    assert cg.n_beads == 3
    assert sorted(cg.get(h, "bead_type") for h in cg.entities()) == ["A", "B", "C"]
    assert len(cg.cgbonds) == 2


def test_coarse_grain_from_atom_frame_without_the_type_column_is_a_value_error() -> None:
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0], dtype=np.float64))
    frame = molrs.Frame()
    frame["atoms"] = atoms

    with pytest.raises(ValueError, match="type"):
        molrs.CoarseGrain.from_atom_frame(frame, "type")


def test_public_fragment_factories_and_graph_out_paths() -> None:
    graph = molrs.Fragment()
    anchor = graph.def_atom(element="O", x=0.0, y=0.0, z=0.0)
    handle = graph.def_atom(element="H", x=0.96, y=0.0, z=0.0)
    graph.def_bond(anchor, handle)
    port = graph.def_port(anchor, handle, "$")

    assert graph.atoms[0] is anchor
    assert graph.ports[0] is port
    assert port.anchor is anchor
    assert port.handle_atom is handle
    for result in (graph.copy(), molrs.Fragment.from_frame(graph.to_frame())):
        assert type(result) is molrs.Fragment
        assert isinstance(result, molrs.GraphViews)
        assert len(result.atoms) == 2
        assert len(result.ports) == 1


def test_factories_return_interned_live_refs() -> None:
    graph = molrs.Atomistic()
    carbon = graph.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    oxygen = graph.def_atom(element="O", x=1.0, y=0.0, z=0.0)
    bond = graph.def_bond(carbon, oxygen, order=2.0)

    assert graph.nodes[0] is carbon
    assert graph.links.all()[0] is bond
    assert bond.itom is carbon
    assert bond.jtom is oxygen
    assert carbon["x", "y", "z"] == [0.0, 0.0, 0.0]
    assert bond["order"] == 2.0


def test_atomistic_def_bond_stamps_both_bond_facts() -> None:
    """A Python-built bond carries the same two facts a native one does.

    ``Fragment.def_bond`` already routes through the native writer, which
    stamps ``bond_type = 1`` and ``bond_number = 1``; ``Atomistic.def_bond``
    owns the same bond kind and must not write a classless bond.
    """
    graph = molrs.Atomistic()
    carbon = graph.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    other = graph.def_atom(element="C", x=1.5, y=0.0, z=0.0)
    graph.def_bond(carbon, other)

    bond = graph.bonds[0]
    assert bond["bond_type"] == 1
    assert bond["bond_number"] == 1


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
