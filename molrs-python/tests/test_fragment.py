"""Python surface for the fragment leaf (cgsmiles-02d-python-fragment).

These are FFI-seam tests: they prove that ``molrs.Fragment`` imports and
constructs, that its three typed writers reach the core ones (so a Python-built
bond carries its class and a Python-built port is validated), that every
graph-out path hands back the shadowed public class, and that the port
vocabulary crosses as the notation glyph.

They re-derive no chemistry and no geometry: every claim about ports,
embedding and ``frag_id`` propagation is owned by the Rust unit tests in
``molrs/src/core/system/fragment.rs``, ``molrs/src/io/smiles/cgsmiles/`` and
``molrs/src/conformer/``, and is reused here only to show Python sees the same
answer. Fixtures are built in process; no third-party scientific software runs.
"""

from __future__ import annotations

import math

import molrs
import pytest
from molrs import _lib

# An OH-capped PEO trimer: the fragment table names exactly ``OH`` and ``PEO``.
F2 = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}"


def _capped_fragment() -> tuple[molrs.Fragment, molrs.Atom, molrs.Atom, molrs.Atom]:
    """``C–O–H`` with one unnamed ``$`` port on the capping hydrogen.

    The smallest graph that can carry a legal port: ``add_port`` requires the
    handle to be a hydrogen *bonded to* its anchor. Returns
    ``(fragment, carbon, oxygen, hydrogen)``.
    """
    fragment = molrs.Fragment()
    carbon = fragment.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    oxygen = fragment.def_atom(element="O", x=1.43, y=0.0, z=0.0)
    hydrogen = fragment.def_atom(element="H", x=2.39, y=0.0, z=0.0)
    fragment.def_bond(carbon, oxygen)
    fragment.def_bond(oxygen, hydrogen)
    fragment.def_port(oxygen, hydrogen, "$")
    return fragment, carbon, oxygen, hydrogen


# ---------------------------------------------------------------------------
# The leaf: one shadowed class, both relation kinds registered at construction
# ---------------------------------------------------------------------------


def test_fragment_is_a_shadowed_leaf_with_both_kinds_registered() -> None:
    fragment = molrs.Fragment()

    assert type(fragment) is molrs.Fragment
    assert isinstance(fragment, molrs.Graph)
    assert isinstance(fragment, molrs.GraphViews)
    assert set(fragment.kinds()) >= {"bonds", "ports"}
    assert _lib.Fragment.__module__ == "molrs._lib"


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
    fragment = molrs.Fragment()
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

    for result in (fragment.copy(), molrs.Fragment.from_frame(fragment.to_frame())):
        assert type(result) is molrs.Fragment
        assert isinstance(result, molrs.GraphViews)
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


def test_conformer_returns_a_fragment() -> None:
    # Three heavy atoms plus the capping hydrogen the port needs.
    fragment = molrs.Fragment()
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

    assert type(embedded) is molrs.Fragment
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
    assert "Fragment" in message


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


def test_to_fragment_returns_named_fragments() -> None:
    """Read the two fragment bodies of an OH-capped PEO trimer.

    Hand-derived from the notation: ``#PEO=[$]COC[$]`` writes two bonding
    descriptors and ``#OH=[$]O`` writes one, so the fragments they build carry
    two and one port respectively. No third-party tool produced these numbers.
    """
    fragments = molrs.io.CGSmilesIR(F2).to_fragment()

    assert isinstance(fragments, dict)
    assert set(fragments) == {"OH", "PEO"}
    for fragment in fragments.values():
        assert type(fragment) is molrs.Fragment
    assert fragments["PEO"].n_ports == 2
    assert fragments["OH"].n_ports == 1
