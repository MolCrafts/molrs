"""``molrs.perceive.Perceive(graph).linear_paths()`` FFI seam: the held graph,
the handle lists out and error mapping. Path semantics are proven by the Rust
suite.
"""

from __future__ import annotations

import pytest

import molrs
from molrs.perceive import Perceive


def test_linear_paths_lists_each_chain_as_bead_handles() -> None:
    cg = molrs.CoarseGrain()
    a, b, c, d = (cg.add_bead("S", float(i), 0.0, 0.0) for i in range(4))
    cg.add_bond(a, b)
    cg.add_bond(b, c)

    assert Perceive(cg).linear_paths() == [[a, b, c], [d]]


def test_a_branch_is_a_value_error_naming_the_centre() -> None:
    cg = molrs.CoarseGrain()
    centre = cg.add_bead("S", 0.0, 0.0, 0.0)
    for i in range(3):
        cg.add_bond(centre, cg.add_bead("S", float(i + 1), 0.0, 0.0))

    with pytest.raises(ValueError) as caught:
        Perceive(cg).linear_paths()

    message = str(caught.value)
    assert f"node {centre} " in message
    assert "NodeId(" not in message


def test_linear_paths_without_a_graph_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        Perceive().linear_paths()


def test_find_rings_still_takes_its_molecule() -> None:
    mol = molrs.Atomistic()
    mol.def_atom(element="C", x=0.0, y=0.0, z=0.0)

    assert isinstance(Perceive().find_rings(mol), molrs.Atomistic)
