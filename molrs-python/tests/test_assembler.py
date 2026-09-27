"""``molrs.builder.Assembler`` / ``TracePlacer`` FFI seam: library values of
both graph types, the public ``Fragment`` out, the id columns and error
mapping. Placement and the link rule are proven by the Rust suite.

Hand counts: two copies of the 4-atom ``H0-C0-C1-H1`` monomer (ports ``>`` on
C0/H0, ``<`` on C1/H1) are 8 atoms and 4 ports; the one link between them
removes one hydrogen per joined port, leaving 6 atoms and 2 ports; the lone Li
adds 1 atom, so the world has 7 atoms and 2 ports.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs
from molrs.builder import Assembler, TracePlacer


def _monomer() -> molrs.Fragment:
    fragment = molrs.Fragment()
    c0 = fragment.def_atom(element="C", x=0.0, y=0.0, z=0.0, mass=12.011)
    c1 = fragment.def_atom(element="C", x=1.54, y=0.0, z=0.0, mass=12.011)
    h0 = fragment.def_atom(element="H", x=-1.0, y=0.0, z=0.0, mass=1.008)
    h1 = fragment.def_atom(element="H", x=2.54, y=0.0, z=0.0, mass=1.008)
    fragment.def_bond(c0, c1)
    fragment.def_bond(c0, h0)
    fragment.def_bond(c1, h1)
    fragment.def_port(c0, h0, ">")
    fragment.def_port(c1, h1, "<")
    return fragment


def _lithium() -> molrs.Atomistic:
    mol = molrs.Atomistic()
    mol.def_atom(element="Li", x=0.0, y=0.0, z=0.0, mass=6.94)
    return mol


def _traces() -> list[molrs.Trace]:
    return [
        molrs.Trace(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])),
        molrs.Trace(np.array([[40.0, 0.0, 0.0]])),
    ]


def test_assemble_links_fragments_and_places_atomistic_singles() -> None:
    assembler = Assembler({"M": _monomer(), "Li": _lithium()}, TracePlacer())

    world = assembler.assemble(_traces(), [["M", "M"], ["Li"]])

    assert type(world) is molrs.Fragment
    assert world.n_atoms == 7
    assert world.n_ports == 2
    atoms = world.to_frame()["atoms"]
    mol_id = np.asarray(atoms["mol_id"])
    element = np.asarray(atoms["element"])
    assert set(mol_id.tolist()) == {1, 2}
    assert mol_id[element == "Li"].tolist() == [2]


def test_an_unknown_name_is_a_value_error_naming_trace_and_unit() -> None:
    assembler = Assembler({"M": _monomer()}, TracePlacer())

    with pytest.raises(ValueError, match=r"unit 0 of trace 1 names 'Li'"):
        assembler.assemble(_traces(), [["M", "M"], ["Li"]])


def test_a_placer_that_is_not_a_trace_placer_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        Assembler({"M": _monomer()}, object())


def test_trace_placer_takes_no_arguments() -> None:
    TracePlacer()
    with pytest.raises(TypeError):
        TracePlacer(1.0)  # type: ignore[call-arg]
