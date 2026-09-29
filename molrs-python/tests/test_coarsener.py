"""``molrs.perceive.Coarsener`` FFI seam: sources accepted, the graph out and
error mapping. The centre-of-mass mapping itself is proven by the Rust suite.
"""

from __future__ import annotations

import itertools

import molrs
import pytest
from molrs.perceive import Coarsener


def _bead_chain() -> tuple[molrs.CoarseGrain, list[int]]:
    """Four unit-mass beads ``b0-b1-b2-b3`` on the x axis, 1 Å apart."""
    cg = molrs.CoarseGrain()
    beads = [cg.add_bead("S", float(i), 0.0, 0.0) for i in range(4)]
    for bead in beads:
        cg.set(bead, "mass", 1.0)
    for a, b in itertools.pairwise(beads):
        cg.add_bond(a, b)
    return cg, beads


def test_coarsen_returns_a_coarsegrain_with_one_site_per_group() -> None:
    cg, (b0, b1, b2, b3) = _bead_chain()

    sites = Coarsener(cg).coarsen([[b0, b1], [b2, b3]], ["A", "B"])

    assert type(sites) is molrs.CoarseGrain
    assert sites.n_beads == 2
    assert sites.n_relations("bonds") == 1
    assert sites.bead_types(sites.entities()) == ["A", "B"]


def test_an_atomistic_source_is_accepted() -> None:
    mol = molrs.Atomistic()
    c = mol.def_atom(element="C", x=0.0, y=0.0, z=0.0, mass=12.011)
    h = mol.def_atom(element="H", x=1.09, y=0.0, z=0.0, mass=1.008)
    mol.def_bond(c, h)

    sites = Coarsener(mol).coarsen([[c.handle, h.handle]], ["CH"])

    assert type(sites) is molrs.CoarseGrain
    assert sites.n_beads == 1


def test_a_frame_source_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        Coarsener(molrs.Frame())


def test_an_overlap_is_a_value_error_naming_the_int_handle() -> None:
    cg, (b0, b1, b2, _) = _bead_chain()

    with pytest.raises(ValueError) as caught:
        Coarsener(cg).coarsen([[b0, b1], [b1, b2]], ["A", "B"])

    message = str(caught.value)
    assert str(b1) in message
    assert "NodeId(" not in message
