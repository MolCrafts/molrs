"""Python surface for the CGsmiles IR (cgsmiles-01e-python-ir).

These are FFI-seam tests: they prove that ``molrs.io.CGSmilesIR`` imports,
constructs, hands every fact of the notation across the boundary with the
right Python type and spelling, and maps a malformed string to ``ValueError``.
They re-derive no numeric: every count asserted here is a value the Rust unit
tests in ``molrs/src/io/smiles/cgsmiles/`` already prove (F2 in
``resolve.rs`` / ``to_atomistic.rs``, F8 in ``resolve.rs`` /
``instantiate.rs``), reused only to show Python sees the same number.

Fixtures are inline strings; no third-party scientific software runs.
"""

from __future__ import annotations

import molrs
import pytest

# An OH-capped PEO trimer: one coarse level, one atomistic fragment table.
F2 = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}"

# Beads of blocks, blocks of beads, beads of atoms: two coarse levels and two
# fragment tables, the last of them atomistic.
F8 = (
    "{[#B1][#B2][#B1]}."
    "{#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}."
    "{#PEO=[>]COC[<],#PE=[>]CC[<]}"
)

# A symmetric-descriptor fixture. Only a *coarse* body exposes its bonding
# descriptors to Python (``SmilesIR`` publishes none), so the ``[$]`` under
# test is written on the intermediate table's body, not on F2's atomistic one.
F_SYM = "{[#A][#A]}.{#A=[$][#B][#B][$]}.{#B=[$]CC[$]}"

# The eight classes this binding publishes, all from ``molrs.io``.
CG_NAMES = (
    "CGSmilesIR",
    "CGGraph",
    "CGNode",
    "CGEdge",
    "CGFragmentDef",
    "ResolvedPair",
    "PairEnd",
    "BondingDescriptor",
)

# Documented value sets for the enums at the seam. Three of them cross as the
# lowercase spelling of their Rust variant; a bonding descriptor's kind crosses
# as the notation glyph instead, exactly as a stored port's ``port_kind`` does.
BOND_KINDS = frozenset(
    {
        "single",
        "double",
        "triple",
        "quadruple",
        "aromatic",
        "up",
        "down",
        "any",
        "ring",
    }
)
# The four notation glyphs: ``[$]`` symmetric, ``[<]`` left, ``[>]`` right,
# ``[!]`` shared.
DESCRIPTOR_KINDS = frozenset({"$", "<", ">", "!"})
PAIR_END_TAGS = frozenset({"sub", "body"})


# ---------------------------------------------------------------------------
# F2 — the graph surface of a single-level string
# ---------------------------------------------------------------------------


def test_f2_has_one_resolution_level() -> None:
    ir = molrs.io.CGSmilesIR(F2)
    assert len(ir.levels) == 1


def test_f2_level_has_five_nodes_and_four_edges() -> None:
    level = molrs.io.CGSmilesIR(F2).levels[0]
    assert len(level.nodes) == 5
    assert len(level.edges) == 4


def test_f2_node_names_are_the_written_bead_names() -> None:
    level = molrs.io.CGSmilesIR(F2).levels[0]
    assert [node.name for node in level.nodes] == [
        "OH",
        "PEO",
        "PEO",
        "PEO",
        "OH",
    ]


def test_f2_nodes_carry_no_charge() -> None:
    level = molrs.io.CGSmilesIR(F2).levels[0]
    assert all(node.charge is None for node in level.nodes)


def test_f2_nodes_carry_no_annotations() -> None:
    level = molrs.io.CGSmilesIR(F2).levels[0]
    assert all(node.annotations == [] for node in level.nodes)


def test_f2_base_level_nodes_have_no_parent() -> None:
    level = molrs.io.CGSmilesIR(F2).levels[0]
    assert all(node.parent is None for node in level.nodes)


def test_f2_edges_all_have_multiplicity_one() -> None:
    level = molrs.io.CGSmilesIR(F2).levels[0]
    assert [edge.multiplicity for edge in level.edges] == [1, 1, 1, 1]


def test_f2_edges_are_all_written_not_derived() -> None:
    level = molrs.io.CGSmilesIR(F2).levels[0]
    assert all(edge.derived_from is None for edge in level.edges)


def test_f2_has_one_fragment_table_naming_both_beads() -> None:
    ir = molrs.io.CGSmilesIR(F2)
    assert len(ir.fragments) == 1
    assert set(ir.fragments[0]) == {"OH", "PEO"}


def test_f2_resolves_four_pairs() -> None:
    ir = molrs.io.CGSmilesIR(F2)
    assert len(ir.pairs[0]) == 4


def test_f2_pairs_are_all_single_bonds() -> None:
    ir = molrs.io.CGSmilesIR(F2)
    assert [resolved.kind for resolved in ir.pairs[0]] == ["single"] * 4


def test_f2_pair_ends_are_body_ends() -> None:
    ir = molrs.io.CGSmilesIR(F2)
    for resolved in ir.pairs[0]:
        assert resolved.src.end == "body"
        assert resolved.dst.end == "body"


def test_f2_pair_indices_cross_as_ints() -> None:
    resolved = molrs.io.CGSmilesIR(F2).pairs[0][0]
    assert isinstance(resolved.edge, int)
    assert isinstance(resolved.bond, int)
    assert isinstance(resolved.src.index, int)
    assert isinstance(resolved.src.port, int)


# ---------------------------------------------------------------------------
# F8 — levels, parents, descriptors, fragment bodies and provenance
# ---------------------------------------------------------------------------


def test_f8_has_two_resolution_levels() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    assert len(ir.levels) == 2


def test_f8_level_zero_has_three_nodes_and_two_edges() -> None:
    level = molrs.io.CGSmilesIR(F8).levels[0]
    assert len(level.nodes) == 3
    assert len(level.edges) == 2


def test_f8_level_one_has_six_nodes() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    assert len(ir.levels[1].nodes) == 6


def test_f8_level_one_nodes_point_back_at_the_bead_they_came_from() -> None:
    level = molrs.io.CGSmilesIR(F8).levels[1]
    assert [node.parent for node in level.nodes] == [0, 0, 1, 1, 2, 2]


def test_f8_level_one_node_names_come_from_the_expanded_bodies() -> None:
    level = molrs.io.CGSmilesIR(F8).levels[1]
    assert [node.name for node in level.nodes] == [
        "PEO",
        "PEO",
        "PE",
        "PE",
        "PEO",
        "PEO",
    ]


def test_f8_level_one_has_five_edges() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    assert len(ir.levels[1].edges) == 5


def test_f8_written_edges_of_level_one_have_no_provenance() -> None:
    edges = molrs.io.CGSmilesIR(F8).levels[1].edges
    assert [edge.derived_from for edge in edges[:3]] == [None, None, None]


def test_f8_derived_edges_name_the_pair_that_induced_them() -> None:
    edges = molrs.io.CGSmilesIR(F8).levels[1].edges
    assert edges[3].derived_from == (0, 0)
    assert edges[4].derived_from == (0, 1)


def test_f8_coarse_fragment_body_is_a_cg_graph() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    assert isinstance(ir.fragments[0]["B1"].body, molrs.io.CGGraph)


def test_f8_coarse_body_opens_with_an_unlabelled_orderless_right_descriptor() -> None:
    body = molrs.io.CGSmilesIR(F8).fragments[0]["B1"].body
    descriptor = body.nodes[0].descriptors[0]
    assert (descriptor.kind, descriptor.label, descriptor.order) == (">", "", None)


def test_f8_coarse_body_closes_with_a_left_descriptor() -> None:
    body = molrs.io.CGSmilesIR(F8).fragments[0]["B1"].body
    assert body.nodes[1].descriptors[0].kind == "<"


def test_symmetric_descriptor_crosses_as_the_dollar_glyph() -> None:
    body = molrs.io.CGSmilesIR(F_SYM).fragments[0]["A"].body
    assert body.nodes[0].descriptors[0].kind == "$"


def test_f8_last_fragment_table_holds_atomistic_bodies() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    assert isinstance(ir.fragments[1]["PEO"].body, molrs.io.SmilesIR)


def test_f8_has_two_fragment_tables() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    assert len(ir.fragments) == 2


def test_f8_level_zero_pairs_reach_into_the_child_level() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    assert ir.pairs[0][0].src.end == "sub"


# ---------------------------------------------------------------------------
# Types and spellings at the boundary
# ---------------------------------------------------------------------------


def test_every_enum_crosses_as_its_documented_spelling() -> None:
    """Bond kinds and pair-end tags cross as lowercase variant names; a
    descriptor kind crosses as the glyph the notation wrote."""
    ir = molrs.io.CGSmilesIR(F8)
    seen = 0
    for level_pairs in ir.pairs:
        for resolved in level_pairs:
            assert resolved.kind in BOND_KINDS
            assert resolved.src.end in PAIR_END_TAGS
            assert resolved.dst.end in PAIR_END_TAGS
            seen += 1
    for table in ir.fragments:
        for definition in table.values():
            body = definition.body
            if not isinstance(body, molrs.io.CGGraph):
                continue
            for node in body.nodes:
                for descriptor in node.descriptors:
                    assert descriptor.kind in DESCRIPTOR_KINDS
                    assert descriptor.order is None or descriptor.order in BOND_KINDS
                    seen += 1
    assert seen > 0, "fixture produced no enum value to check"


def test_charge_crosses_as_a_float() -> None:
    node = molrs.io.CGSmilesIR("{[#A;q=-0.5]}").levels[0].nodes[0]
    assert isinstance(node.charge, float)
    assert abs(node.charge - (-0.5)) < 1e-10


def test_annotations_cross_as_a_list_of_string_pairs() -> None:
    node = molrs.io.CGSmilesIR("{[#A;q=-0.5;kind=ether]}").levels[0].nodes[0]
    assert node.annotations == [("kind", "ether")]


def test_edge_endpoints_and_multiplicity_cross_as_ints() -> None:
    edge = molrs.io.CGSmilesIR(F2).levels[0].edges[0]
    assert isinstance(edge.i, int)
    assert isinstance(edge.j, int)
    assert isinstance(edge.multiplicity, int)


def test_parent_crosses_as_an_int_on_an_expanded_level() -> None:
    node = molrs.io.CGSmilesIR(F8).levels[1].nodes[0]
    assert isinstance(node.parent, int)


def test_derived_from_crosses_as_a_two_tuple_of_ints() -> None:
    derived_from = molrs.io.CGSmilesIR(F8).levels[1].edges[3].derived_from
    assert isinstance(derived_from, tuple)
    assert len(derived_from) == 2
    assert all(isinstance(value, int) for value in derived_from)


def test_fragment_definition_carries_its_own_name() -> None:
    ir = molrs.io.CGSmilesIR(F2)
    assert ir.fragments[0]["PEO"].name == "PEO"


# ---------------------------------------------------------------------------
# One spelling per fact
# ---------------------------------------------------------------------------


def test_cg_edge_has_no_second_spelling_of_its_order_or_origin() -> None:
    edge = molrs.io.CGSmilesIR(F2).levels[0].edges[0]
    assert not hasattr(edge, "order")
    assert not hasattr(edge, "origin")


def test_cg_fragment_def_has_no_body_kind() -> None:
    definition = molrs.io.CGSmilesIR(F2).fragments[0]["PEO"]
    assert not hasattr(definition, "body_kind")


def test_cg_smiles_ir_has_no_n_levels() -> None:
    ir = molrs.io.CGSmilesIR(F2)
    assert not hasattr(ir, "n_levels")


def test_io_namespace_has_no_cgsmiles_reader() -> None:
    assert not hasattr(molrs.io, "CGSmilesReader")


def test_molrs_has_no_parse_cgsmiles_free_function() -> None:
    assert not hasattr(molrs, "parse_cgsmiles")


def test_all_eight_classes_are_public_names_of_molrs_io() -> None:
    for name in CG_NAMES:
        assert hasattr(molrs.io, name), name
        assert name in molrs.io.__all__, name


# ---------------------------------------------------------------------------
# Errors and the read-only contract
# ---------------------------------------------------------------------------


def test_unterminated_block_raises_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        molrs.io.CGSmilesIR("{[#A]")
    assert str(excinfo.value)


def test_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        molrs.io.CGSmilesIR("")
    assert str(excinfo.value)


def test_repr_echoes_the_input_string() -> None:
    assert F2 in repr(molrs.io.CGSmilesIR(F2))


def test_nested_records_reject_attribute_assignment() -> None:
    ir = molrs.io.CGSmilesIR(F8)
    level = ir.levels[1]
    definition = ir.fragments[0]["B1"]
    resolved = ir.pairs[0][0]
    samples: list[tuple[object, str, object]] = [
        (level, "nodes", []),
        (level.nodes[0], "name", "X"),
        (level.edges[0], "i", 0),
        (definition, "name", "X"),
        (definition.body.nodes[0].descriptors[0], "kind", "<"),
        (resolved, "kind", "double"),
        (resolved.src, "end", "body"),
    ]
    for record, attribute, value in samples:
        with pytest.raises(AttributeError):
            setattr(record, attribute, value)


def test_cg_node_is_not_constructible_from_python() -> None:
    with pytest.raises(TypeError):
        molrs.io.CGNode()


# ---------------------------------------------------------------------------
# Whole-string template and unit-level topology (assembly-07)
# ---------------------------------------------------------------------------

# A two-bead monomer: `[<]CC[$]` + `[$]CO[>]`, the `$` pair consumed, the `<`
# and `>` left open (the `to_template` rustdoc example in
# molrs/src/io/smiles/cgsmiles/to_template.rs).
MONOMER = "{[#A][#B]}.{#A=[<]CC[$],#B=[$]CO[>]}"


def test_to_template_returns_a_fragment_with_the_open_valences_as_ports() -> None:
    # 4 heavy atoms + one capping hydrogen per unpaired `<` / `>` = 6 atoms;
    # 3 heavy-atom bonds + 2 handle bonds = 5 bonds; 2 ports.
    template = molrs.io.CGSmilesIR(MONOMER).to_template()

    assert type(template) is molrs.Fragment
    assert template.n_atoms == 6
    assert template.n_relations("bonds") == 5
    assert template.n_ports == 2


def test_to_template_of_a_base_only_string_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        molrs.io.CGSmilesIR("{[#A][#B]}").to_template()


def test_to_frag_graph_crosses_node_names_and_edge_tuples() -> None:
    # Seam only: the mapping itself is proven by the Rust `to_frag_graph` tests.
    graph = molrs.io.CGSmilesIR(F2).to_frag_graph()

    assert isinstance(graph, molrs.FragGraph)
    assert graph.nodes and graph.edges
    assert isinstance(graph.nodes, list)
    assert all(isinstance(name, str) for name in graph.nodes)
    assert isinstance(graph.edges, list)
    assert all(
        isinstance(edge, tuple) and len(edge) == 4 and all(isinstance(v, int) for v in edge)
        for edge in graph.edges
    )


def test_to_frag_graph_of_a_base_only_string_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        molrs.io.CGSmilesIR("{[#A][#B]}").to_frag_graph()


# ---------------------------------------------------------------------------
# Public-API example (this repo's stand-in for a regressions/ script)
# ---------------------------------------------------------------------------


def test_cgsmiles_f2_public_api() -> None:
    """Read an OH-capped PEO trimer and expand it, through the public path only.

    Hand-derived in cgsmiles-01c / 01d from the notation rules: `[$]O` is one
    atom and `[$]COC[$]` is three, so 2 * 1 + 3 * 3 = 11 atoms; the three PEO
    bodies carry two bonds each and the four resolved pairs add one bond each,
    so 6 + 4 = 10 bonds. No third-party tool produced these numbers.
    """
    ir = molrs.io.CGSmilesIR(F2)
    assert len(ir.levels) == 1
    assert len(ir.pairs[0]) == 4

    mol = ir.to_atomistic()
    assert isinstance(mol, molrs.Atomistic)
    assert mol.n_atoms == 11
    assert mol.n_relations("bonds") == 10
