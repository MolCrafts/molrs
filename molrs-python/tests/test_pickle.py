import copy
import pickle

import molrs
import numpy as np
from molrs import _lib


def _unit_cube_mesh() -> "molrs.core.TriMesh":
    """Closed unit cube, 12 triangles, outward winding."""
    v = np.array(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=np.float64,
    )
    # vertex index = x + 2*y + 4*z
    faces = np.array(
        [
            [0, 4, 6],
            [0, 6, 2],  # -x
            [1, 3, 7],
            [1, 7, 5],  # +x
            [0, 1, 5],
            [0, 5, 4],  # -y
            [2, 6, 7],
            [2, 7, 3],  # +y
            [0, 2, 3],
            [0, 3, 1],  # -z
            [4, 5, 7],
            [4, 7, 6],  # +z
        ],
        dtype=np.uint32,
    )
    return molrs.core.TriMesh(v, faces)


def roundtrip(value):
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def test_storage_units_and_observables_pickle_by_logical_state() -> None:
    block = molrs.core.Block()
    block.insert("sample", np.array([1, 2], dtype=np.int16))
    block.insert("label", ["left", "right"])
    block.set_shape([1, 2])
    restored = roundtrip(block)
    assert type(restored) is molrs.core.Block
    assert restored.dtype("sample") == "i16"
    assert restored["sample"].tolist() == [1, 2]
    assert restored["label"].tolist() == ["left", "right"]
    assert restored.structural_shape == [1, 2]

    empty_rows = molrs.core.Block()
    empty_rows.resize(3)
    assert roundtrip(empty_rows).nrows == 3

    frame = molrs.core.Frame({"grid": block}, meta={"nested": {"ok": True}})
    frame.box = molrs.core.Box.cube(4.0)
    restored_frame = roundtrip(frame)
    assert type(restored_frame) is molrs.core.Frame
    assert restored_frame["grid"].dtype("sample") == "i16"
    assert restored_frame.meta["nested"] == {"ok": True}
    assert restored_frame.box.volume() == 64.0

    registry = molrs.core.UnitRegistry()
    registry.define("smoot", 1.7018, [1, 0, 0, 0, 0, 0, 0])
    registry.note = "custom registry"
    restored_registry = roundtrip(registry)
    assert restored_registry.note == "custom registry"
    assert restored_registry.smoot.factor_to(restored_registry.m) == 1.7018

    quantity = roundtrip(2.5 * registry.smoot)
    assert quantity.magnitude == 2.5
    assert str(quantity.unit) == "smoot"
    assert roundtrip(molrs.core.UnitPreset("real")).name == "real"
    assert roundtrip(molrs.core.Element("C")) == molrs.core.Element(6)

    observable = molrs.core.VectorObservable(
        "force",
        np.array([[1.0, 2.0, 3.0]]),
        "force vector",
        "kJ/mol/nm",
        ["atom", "xyz"],
        True,
        "every_step",
        "particle",
        "atoms",
    )
    restored_observable = roundtrip(observable)
    np.testing.assert_array_equal(restored_observable.data, observable.data)
    assert restored_observable.description == "force vector"
    assert restored_observable.axes == ["atom", "xyz"]
    assert restored_observable.target == "atoms"

    trajectory = molrs.core.Trajectory(
        [frame], step=np.array([7], dtype=np.int64), time=np.array([0.5])
    )
    restored_trajectory = roundtrip(trajectory)
    assert restored_trajectory.step.tolist() == [7]
    assert restored_trajectory.time.tolist() == [0.5]
    assert restored_trajectory.frames[0].box.volume() == 64.0


def test_spatial_types_preserve_queries_and_region_behavior() -> None:
    box = molrs.core.Box.cube(10.0, pbc=np.array([True, False, True]))
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]])

    restored_box = roundtrip(box)
    np.testing.assert_array_equal(restored_box.pbc, box.pbc)
    np.testing.assert_array_equal(restored_box.h, box.h)

    neighbor_list = molrs.core.NeighborList.brute_force(1.0)
    neighbor_list.build(points, box)
    restored_list = roundtrip(neighbor_list)
    restored_pairs = restored_list.neighbors(disp=False)
    assert restored_pairs.query_point_indices().tolist() == [0]
    assert restored_pairs.point_indices().tolist() == [1]
    assert restored_pairs.disp() is None

    table = roundtrip(neighbor_list.neighbors(dist_sq=False, disp=True))
    assert table.dist_sq() is None
    np.testing.assert_allclose(table.disp(), [[0.5, 0.0, 0.0]])

    query = roundtrip(molrs.core.NeighborQuery(box, points, 1.0))
    assert query.query_self().n_pairs == 1
    assert query.query(np.array([[0.25, 0.0, 0.0]])).n_pairs == 2

    skin = molrs.core.VerletSkin(
        molrs.core.NeighborList(1.2),
        1.0,
        points,
        box,
        skin=0.2,
        every=2,
    )
    assert skin.update(points + 0.01) is False
    restored_skin = roundtrip(skin)
    assert restored_skin.ago == 1
    assert restored_skin.rebuild_count == 0
    assert restored_skin.n_edges == skin.n_edges

    cube_mesh = _unit_cube_mesh()
    primitives = [
        molrs.core.Sphere(np.zeros(3), 2.0),
        molrs.core.Cuboid(np.zeros(3), np.ones(3)),
        molrs.core.Parallelepiped(np.eye(3), np.zeros(3)),
        molrs.core.HalfSpace(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5])),
        molrs.core.Cylinder(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0, 2.0),
        molrs.core.Ellipsoid(np.zeros(3), np.array([1.0, 2.0, 0.5])),
        molrs.core.Polyhedron(cube_mesh),
        molrs.core.SphereUnion(np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), 0.8),
        molrs.core.SphereUnion(
            np.array([[0.2, 0.2, 0.2]]),
            np.array([0.5]),
            box=molrs.core.Box.cube(3.0, np.zeros(3), np.array([True, True, True])),
        ),
    ]
    composed = (primitives[0] & ~primitives[1]) | (primitives[3] & primitives[7])
    restored_mesh = roundtrip(cube_mesh)
    assert isinstance(restored_mesh, molrs.core.TriMesh)
    np.testing.assert_array_equal(restored_mesh.faces(), cube_mesh.faces())
    probes = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.5, 0.0, 0.0]])
    for region in [*primitives, composed, molrs.core.Region(primitives[0])]:
        restored = roundtrip(region)
        assert type(restored) is type(region)
        np.testing.assert_array_equal(
            restored.contains(probes), region.contains(probes)
        )


def test_graphs_views_and_extraction_pickle_as_one_object_graph() -> None:
    graph = molrs.core.MolGraph()
    graph.register_kind("empty", 3)
    graph.register_kind("links", 2)
    first, removed, last = [graph.spawn() for _ in range(3)]
    graph.despawn(removed)
    graph.set(first, "name", "first")
    graph.set(last, "x", 2.0)
    relation = graph.add_relation("links", [first, last])
    graph.set_relation_prop("links", relation, "order", 2)
    restored_graph = roundtrip(graph)
    handles = restored_graph.entities()
    assert restored_graph.kind_arity("empty") == 3
    assert restored_graph.get(handles[0], "name") == "first"
    assert (
        restored_graph.relation_nodes("links", restored_graph.relation_ids("links")[0])
        == handles
    )

    molecule = molrs.core.Atomistic(label="typed refs")
    atoms = [molecule.def_atom(element="C") for _ in range(4)]
    virtual = molecule.def_virtual_site(kind=molrs.core.VirtualSite)
    drude = molecule.def_virtual_site(kind=molrs.core.DrudeParticle)
    massless = molecule.def_virtual_site(kind=molrs.core.MasslessSite)
    refs = [
        *atoms,
        virtual,
        drude,
        massless,
        molecule.def_bond(atoms[0], atoms[1]),
        molecule.def_angle(*atoms[:3]),
        molecule.def_dihedral(*atoms),
        molecule.def_improper(*atoms),
    ]
    restored_molecule, restored_refs, restored_lazy = roundtrip(
        (molecule, refs, molecule.atoms)
    )
    assert restored_molecule.props == {"label": "typed refs"}
    assert [type(ref) for ref in restored_refs] == [type(ref) for ref in refs]
    assert all(ref.world is restored_molecule for ref in restored_refs)
    assert all(ref.world is restored_molecule for ref in restored_lazy)

    coarse = molrs.core.CoarseGrain(label="cg")
    bead = coarse.def_bead(type="P", atoms=tuple(atoms[:2]))
    coarse.def_bead(type="Q")
    coarse.def_cgbond(coarse.beads[0], coarse.beads[1])
    restored_source, restored_coarse, restored_bead = roundtrip(
        (molecule, coarse, bead)
    )
    assert restored_bead is restored_coarse.beads[0]
    assert all(atom.world is restored_source for atom in restored_bead["atoms"])

    chain = molrs.core.Atomistic()
    chain_atoms = [chain.def_atom(element="C") for _ in range(3)]
    chain.def_bond(chain_atoms[0], chain_atoms[1])
    chain.def_bond(chain_atoms[1], chain_atoms[2])
    extracted = roundtrip(chain.extract_subgraph([chain_atoms[1].handle], 1))
    assert len(extracted.graph.atoms) == 3
    assert set(extracted.parent_of) == set(extracted.graph.entities())

    fragment = molrs.core.Atomistic()
    fragment_anchor = fragment.def_atom(element="O")
    fragment_handle = fragment.def_atom(element="H")
    fragment.def_bond(fragment_anchor, fragment_handle)
    fragment.def_port(fragment_anchor, fragment_handle, "$")
    fragment.set_frag_id(fragment_anchor.handle, 2)
    restored_fragment = roundtrip(fragment)
    assert type(restored_fragment) is molrs.core.Atomistic
    assert restored_fragment.n_ports == 1
    assert restored_fragment.ports[0]["port_kind"] == "$"
    assert restored_fragment.frag_id(restored_fragment.atoms[0].handle) == 2

    # A partially labelled fragment keeps its holes: an unlabelled atom comes
    # back unlabelled, never as a stated ``frag_id`` of 0.
    partial = molrs.core.Atomistic()
    labelled = partial.def_atom(element="C")
    middle = partial.def_atom(element="C")
    capping = partial.def_atom(element="H")
    partial.def_bond(labelled, middle)
    partial.def_bond(middle, capping)
    partial.set_frag_id(labelled.handle, 7)
    restored_partial = roundtrip(partial)
    assert [
        restored_partial.frag_id(atom.handle) for atom in restored_partial.atoms
    ] == [7, None, None]

    assert type(roundtrip(_lib.Atomistic())) is _lib.Atomistic
    assert type(roundtrip(_lib.CoarseGrain())) is _lib.CoarseGrain
    assert roundtrip(molrs.perceive.Reaction("[C:1]>>[C:1]")).forming_bonds == []


def test_schema_and_metadata_pickle_as_value_types() -> None:
    column = molrs.core.schema.columns[0]
    restored_column = roundtrip(column)
    assert type(restored_column) is type(column)
    assert restored_column.key == column.key
    assert restored_column.const_name == column.const_name
    assert restored_column.dtype == column.dtype
    assert restored_column.shape == column.shape
    assert restored_column.unit == column.unit
    assert restored_column.doc == column.doc
    assert restored_column.numpy_dtype == column.numpy_dtype

    block = next(spec for spec in molrs.core.schema.blocks if spec.endpoint_columns)
    restored_block = roundtrip(block)
    assert type(restored_block) is type(block)
    assert restored_block.name == block.name
    assert restored_block.row_kind == block.row_kind
    assert restored_block.endpoint_target == block.endpoint_target
    assert restored_block.endpoint_columns == block.endpoint_columns
    assert restored_block.required == block.required
    assert restored_block.optional == block.optional
    assert restored_block.open == block.open
    assert restored_block.doc == block.doc

    assert [spec.key for spec in roundtrip(list(molrs.core.schema.columns))] == [
        spec.key for spec in molrs.core.schema.columns
    ]

    text = roundtrip(molrs.core.MetaValue("string", "hello"))
    assert text.dtype == "string"
    assert text.value == "hello"
    nested = roundtrip(molrs.core.MetaValue("json", {"ok": True}))
    assert nested.dtype == "json"
    assert nested.value == {"ok": True}


def test_frame_json_meta_roundtrip_stays_json_and_frozen() -> None:
    frame = molrs.core.Frame(meta={"nested": {"ok": True, "tags": [1, 2]}})
    restored = roundtrip(frame)
    assert restored.meta.dtype("nested") == "json"
    assert isinstance(restored.meta["nested"], molrs.core.MetaDocument)
    assert isinstance(restored.meta["nested"], dict) is False
    assert restored.meta["nested"] == {"ok": True, "tags": (1, 2)}


def test_meta_value_json_payload_is_plain_and_vector_payload_is_tuple() -> None:
    vector = molrs.core.MetaValue("f64x6", [1, 2, 3, 4, 5, 6])
    assert isinstance(vector.value, tuple)
    assert vector.value == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    document = molrs.core.MetaValue("json", {"ok": True, "tags": [1, 2]})
    assert isinstance(document.value, dict)
    assert isinstance(document.value["tags"], list)
    assert document.value == {"ok": True, "tags": [1, 2]}
    assert not isinstance(document.value, molrs.core.MetaDocument)


def test_document_assigned_across_frames_keeps_json_dtype() -> None:
    src = molrs.core.Frame()
    src.meta["run"] = {"step": 1, "tags": [1, 2], "inner": {"a": [3]}}
    dst = molrs.core.Frame()
    dst.meta["copied"] = src.meta["run"]
    assert dst.meta.dtype("copied") == "json"
    assert isinstance(dst.meta["copied"], molrs.core.MetaDocument)
    assert dst.meta["copied"] == {"step": 1, "tags": (1, 2), "inner": {"a": (3,)}}
    assert isinstance(dst.meta["copied"]["tags"], tuple)
    assert isinstance(dst.meta["copied"]["inner"]["a"], tuple)


def test_a_meta_document_pickles_by_content() -> None:
    frame = molrs.core.Frame(meta={"doc": {"ok": True, "tags": [1, 2], "inner": {"a": 1}}})
    doc = frame.meta["doc"]
    assert roundtrip(doc) == {"ok": True, "tags": (1, 2), "inner": {"a": 1}}


def test_a_meta_document_deep_copies_by_content() -> None:
    frame = molrs.core.Frame(meta={"doc": {"ok": True, "tags": [1, 2], "inner": {"a": 1}}})
    doc = frame.meta["doc"]
    assert copy.deepcopy(doc) == {"ok": True, "tags": (1, 2), "inner": {"a": 1}}


def test_a_meta_dict_snapshot_with_a_document_pickles() -> None:
    frame = molrs.core.Frame(meta={"doc": {"ok": True, "tags": [1, 2]}, "step": 3})
    restored = roundtrip(dict(frame.meta))
    assert restored == {"doc": {"ok": True, "tags": (1, 2)}, "step": 3}
