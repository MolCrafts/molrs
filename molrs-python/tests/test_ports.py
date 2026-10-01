"""Python surface for ports: named attachment points any graph may carry.

These are FFI-seam tests: they prove that ports reach Python on a plain
``molrs.Atomistic`` (graph types are peers, and ports are a capability of
every one), that the typed writers reach the core ones (a Python-built bond
carries its class and a Python-built port is validated), that graph-out paths
keep the public class and its ports, and that the port vocabulary crosses as
the notation glyph.

They re-derive no chemistry and no geometry: every claim about ports,
embedding and ``frag_id`` propagation is owned by the Rust unit tests in
``molrs/src/core/system/port.rs``, ``molrs/src/io/smiles/cgsmiles/`` and
``molrs/src/conformer/``, and is reused here only to show Python sees the same
answer. Fixtures are built in process; no third-party scientific software runs.
"""

from __future__ import annotations

import math

import molrs
import pytest

# An OH-capped PEO trimer: the fragment table names exactly ``OH`` and ``PEO``.
F2 = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}"


def _capped_fragment() -> tuple[molrs.Atomistic, molrs.Atom, molrs.Atom, molrs.Atom]:
    """``C–O–H`` with one unnamed ``$`` port on the capping hydrogen.

    The smallest graph that can carry a legal port: ``add_port`` requires the
    handle to be a hydrogen *bonded to* its anchor. Returns
    ``(fragment, carbon, oxygen, hydrogen)``.
    """
    fragment = molrs.Atomistic()
    carbon = fragment.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    oxygen = fragment.def_atom(element="O", x=1.43, y=0.0, z=0.0)
    hydrogen = fragment.def_atom(element="H", x=2.39, y=0.0, z=0.0)
    fragment.def_bond(carbon, oxygen)
    fragment.def_bond(oxygen, hydrogen)
    fragment.def_port(oxygen, hydrogen, "$")
    return fragment, carbon, oxygen, hydrogen


# ---------------------------------------------------------------------------
# Ports live on any graph: the kind appears on the first port
# ---------------------------------------------------------------------------


def test_the_ports_kind_is_registered_by_the_first_port() -> None:
    mol = molrs.Atomistic()
    assert "ports" not in mol.kinds()
    assert mol.n_ports == 0
    assert list(mol.ports) == []

    fragment, _carbon, _oxygen, _hydrogen = _capped_fragment()

    assert "ports" in fragment.kinds()
    assert fragment.n_ports == 1


def test_fragment_ports_are_interned_live_views() -> None:
    fragment, _carbon, oxygen, hydrogen = _capped_fragment()

    assert fragment.n_atoms == 3
    assert fragment.n_ports == 1
    assert fragment.ports[0] is fragment.ports[0]

    port = fragment.ports[0]
    assert port.anchor is oxygen
    assert port.handle_atom is hydrogen
    assert port["port_kind"] == "$"
    assert port["port_label"] == ""
    assert port["port_order"] == 1


def test_def_bond_stamps_bond_class() -> None:
    fragment = molrs.Atomistic()
    carbon = fragment.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    oxygen = fragment.def_atom(element="O", x=1.43, y=0.0, z=0.0)
    fragment.def_bond(carbon, oxygen)

    # The native writer stamps both facts; the generic relation path does not.
    assert fragment.bonds[0]["bond_type"] == 1
    assert fragment.bonds[0]["bond_number"] == 1


def test_frag_id_round_trips() -> None:
    fragment, carbon, oxygen, hydrogen = _capped_fragment()
    fragment.set_frag_id(carbon.handle, 7)
    fragment.set_frag_id(oxygen.handle, 7)

    assert fragment.frag_id(carbon.handle) == 7
    assert fragment.frag_id(hydrogen.handle) is None

    # One pass: the unlabelled degree-1 hydrogen inherits from its only bonded
    # neighbour; the labelled atoms are left alone.
    assert fragment.inherit_frag_ids() == 1
    assert fragment.frag_id(hydrogen.handle) == 7


def test_graph_out_paths_keep_the_public_fragment_type() -> None:
    fragment, carbon, oxygen, hydrogen = _capped_fragment()
    # ``to_frame`` emits the ``frag_id`` column only when every atom carries
    # one, so all three are labelled — with distinct ids, which also pins the
    # per-atom order across the round trip.
    for atom, frag_id in ((carbon, 1), (oxygen, 2), (hydrogen, 3)):
        fragment.set_frag_id(atom.handle, frag_id)

    for result in (fragment.copy(), molrs.Atomistic.from_frame(fragment.to_frame())):
        assert type(result) is molrs.Atomistic
        assert result.n_atoms == 3
        assert result.n_ports == 1
        assert result.ports[0]["port_kind"] == "$"
        assert [result.frag_id(atom.handle) for atom in result.atoms] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Geometry: the leaf's own atoms move, not the empty base graph
# ---------------------------------------------------------------------------


def test_translate_moves_fragment_atoms() -> None:
    fragment, carbon, _oxygen, _hydrogen = _capped_fragment()

    fragment.translate((1.0, 0.0, 0.0))

    assert carbon["x"] == pytest.approx(1.0, abs=1e-12)
    assert carbon["y"] == pytest.approx(0.0, abs=1e-12)
    assert carbon["z"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Embedding: leaf in, same leaf out
# ---------------------------------------------------------------------------


def test_conformer_keeps_the_ports_of_an_atomistic() -> None:
    # Three heavy atoms plus the capping hydrogen the port needs.
    fragment = molrs.Atomistic()
    head = fragment.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    ether = fragment.def_atom(element="O", x=1.43, y=0.0, z=0.0)
    tail = fragment.def_atom(element="C", x=2.86, y=0.0, z=0.0)
    hydrogen = fragment.def_atom(element="H", x=-1.09, y=0.0, z=0.0)
    fragment.def_bond(head, ether)
    fragment.def_bond(ether, tail)
    fragment.def_bond(head, hydrogen)
    fragment.def_port(head, hydrogen, "$")

    embedded, report = molrs.conformer.Conformer(speed="fast", seed=42).generate(
        fragment
    )

    assert type(embedded) is molrs.Atomistic
    assert isinstance(report, molrs.conformer.ConformerReport)
    assert embedded.n_ports == 1
    for atom in embedded.atoms:
        # Coordinates are Å; the seam claim is that they exist and are finite.
        assert all(math.isfinite(atom[key]) for key in ("x", "y", "z"))


def test_conformer_rejects_a_non_graph() -> None:
    with pytest.raises(TypeError) as excinfo:
        molrs.conformer.Conformer(speed="fast", seed=42).generate(object())

    message = str(excinfo.value)
    assert "Atomistic" in message


# ---------------------------------------------------------------------------
# The port vocabulary is the glyph, and both writers validate it
# ---------------------------------------------------------------------------


def test_def_port_rejects_an_unknown_kind() -> None:
    fragment, _carbon, oxygen, hydrogen = _capped_fragment()

    # Only possible if ``def_port`` reaches core ``add_port``: the generic
    # relation path would write ``"Z"`` into ``port_kind`` without complaint.
    with pytest.raises(ValueError):
        fragment.def_port(oxygen, hydrogen, "Z")


def test_add_port_rejects_an_unknown_kind() -> None:
    fragment, _carbon, oxygen, hydrogen = _capped_fragment()

    with pytest.raises(ValueError):
        fragment.add_port(oxygen.handle, hydrogen.handle, "Z")


# ---------------------------------------------------------------------------
# Public-API example (this repo's stand-in for a regressions/ script)
# ---------------------------------------------------------------------------


def test_templates_returns_named_ported_templates() -> None:
    """Read the two fragment bodies of an OH-capped PEO trimer.

    Hand-derived from the notation: ``#PEO=[$]COC[$]`` writes two bonding
    descriptors and ``#OH=[$]O`` writes one, so the fragments they build carry
    two and one port respectively. No third-party tool produced these numbers.
    """
    fragments = molrs.io.CGSmilesIR(F2).templates()

    assert isinstance(fragments, dict)
    assert set(fragments) == {"OH", "PEO"}
    for fragment in fragments.values():
        assert type(fragment) is molrs.Atomistic
    assert fragments["PEO"].n_ports == 2
    assert fragments["OH"].n_ports == 1


def test_from_fragment_to_template_builds_one_unit() -> None:
    """``[<]OCC[>]``: three heavy atoms plus one hydrogen handle per
    descriptor, one ``<`` and one ``>`` port — the same unit a one-entry table
    builds."""
    unit = molrs.io.SmilesIR.from_fragment("[<]OCC[>]").to_template()
    assert type(unit) is molrs.Atomistic
    assert (unit.n_atoms, unit.n_bonds, unit.n_ports) == (5, 4, 2)
    from_table = molrs.io.CGSmilesIR("{[#EO]}.{#EO=[<]OCC[>]}").templates()["EO"]
    assert (from_table.n_atoms, from_table.n_ports) == (5, 2)


def test_plain_smiles_ir_refuses_a_descriptor() -> None:
    with pytest.raises(ValueError):
        molrs.io.SmilesIR("[<]OCC[>]")


# ---------------------------------------------------------------------------
# merge / link
#
# Seam only: the handle maps, the leaving-group removal and the refusals are
# proven by ``molrs/src/core/system/port.rs`` and ``link.rs``. These tests
# check the Python shapes (a dict of int handles, an int bond handle) and that
# every refusal is a ``ValueError`` that leaves the graph as it was.
# ---------------------------------------------------------------------------


def _def_ported_monomer(fragment: molrs.Atomistic, y: float = 0.0) -> dict[str, int]:
    """Write ``H0–C0–C1–H1`` with ports ``(C0, H0, ">")`` and ``(C1, H1, "<")``
    into ``fragment`` through the native writers; return the port handles keyed
    by glyph.

    Positions in Å and masses in g/mol are hand-set: C0 (0, y, 0), C1 (1.54,
    y, 0), H0 (-1, y, 0), H1 (2.54, y, 0); C 12.011, H 1.008.
    """
    c0 = fragment.def_atom(element="C", x=0.0, y=y, z=0.0, mass=12.011)
    c1 = fragment.def_atom(element="C", x=1.54, y=y, z=0.0, mass=12.011)
    h0 = fragment.def_atom(element="H", x=-1.0, y=y, z=0.0, mass=1.008)
    h1 = fragment.def_atom(element="H", x=2.54, y=y, z=0.0, mass=1.008)
    fragment.def_bond(c0, c1)
    fragment.def_bond(c0, h0)
    fragment.def_bond(c1, h1)
    head = fragment.def_port(c0, h0, ">")
    tail = fragment.def_port(c1, h1, "<")
    return {">": head.handle, "<": tail.handle}


def _ported_monomer() -> molrs.Atomistic:
    """One ported monomer (see :func:`_def_ported_monomer`) in its own
    fragment."""
    fragment = molrs.Atomistic()
    _def_ported_monomer(fragment)
    return fragment


def _two_monomers() -> tuple[molrs.Atomistic, dict[str, int], dict[str, int]]:
    """One fragment holding two separate ported monomers, written directly
    (no ``merge``), and each monomer's port handles keyed by glyph.

    The second copy sits 10 Å along +y so the two are disjoint in space too.
    """
    fragment = molrs.Atomistic()
    first = _def_ported_monomer(fragment, y=0.0)
    second = _def_ported_monomer(fragment, y=10.0)
    return fragment, first, second


def test_merge_carries_the_ports_across_and_empties_other() -> None:
    world = _ported_monomer()
    other = _ported_monomer()

    atom_map = world.merge(other)

    assert isinstance(atom_map, dict)
    assert len(atom_map) == 4
    assert set(atom_map.values()) <= set(world.entities())
    assert other.n_atoms == 0
    assert other.n_ports == 0
    assert world.n_atoms == 8
    assert world.n_ports == 4


def test_merge_type_conflict_is_a_value_error() -> None:
    world = molrs.Atomistic()
    world.def_atom(element="C", tag=1.0)
    other = molrs.Atomistic()
    other.def_atom(element="C", tag="one")

    with pytest.raises(ValueError):
        world.merge(other)


def test_link_returns_a_bond_handle_and_consumes_both_ports() -> None:
    world, first, second = _two_monomers()
    assert world.n_atoms == 8
    assert world.n_ports == 4

    bond = world.link(first[">"], second["<"])

    assert isinstance(bond, int)
    assert bond in world.relation_ids("bonds")
    # One leaving hydrogen per port.
    assert world.n_atoms == 6
    assert world.n_ports == 2


def test_link_refusal_is_a_value_error_naming_int_handles() -> None:
    world, first, second = _two_monomers()

    # Two ">" ports are not complements, so the pair is refused.
    with pytest.raises(ValueError) as excinfo:
        world.link(first[">"], second[">"])

    message = str(excinfo.value)
    assert "RelationId(" not in message
    assert "NodeId(" not in message
    assert str(first[">"]) in message
    assert str(second[">"]) in message
    assert world.n_atoms == 8
    assert world.n_ports == 4


def test_link_of_a_stale_port_is_a_value_error_naming_int_handles() -> None:
    world, first, second = _two_monomers()
    world.link(first[">"], second["<"])
    stale = first[">"]  # consumed by the link above
    n_atoms, n_ports = world.n_atoms, world.n_ports

    with pytest.raises(ValueError) as excinfo:
        world.link(stale, second[">"])

    message = str(excinfo.value)
    assert str(stale) in message
    assert "RelationId(" not in message
    assert "NodeId(" not in message
    assert world.n_atoms == n_atoms
    assert world.n_ports == n_ports
