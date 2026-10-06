"""Python surface for ``molrs.perceive.SubgraphMatcher`` (backmap-primitives-07).

These are FFI-seam tests: they prove that the matcher is published from
``molrs.perceive``, that ``find`` hands back live target bead handles in
pattern order as ``list[list[int]]``, that no match is ``[]``, and that a
non-``CoarseGrain`` target is PyO3's ``TypeError``. Matching semantics (bead
type equality, group enumeration, no partitioning of overlaps) are proven by
the Rust unit tests of ``molrs::perceive::SubgraphMatcher``. Fixtures are
built in process.
"""

from __future__ import annotations

import itertools

import molrs
import pytest
from molrs import _lib


def _chain(*bead_types: str) -> tuple[molrs.system.CoarseGrain, list[int]]:
    """A linear bead chain of ``bead_types``, and its handles in that order."""
    cg = molrs.system.CoarseGrain()
    handles = [cg.add_bead(bead_type) for bead_type in bead_types]
    for a, b in itertools.pairwise(handles):
        cg.add_bond(a, b)
    return cg, handles


def test_subgraph_matcher_is_published_from_molrs_perceive() -> None:
    assert molrs.perceive.SubgraphMatcher is _lib.SubgraphMatcher
    assert "SubgraphMatcher" in molrs.perceive.__all__
    assert molrs.perceive.SubgraphMatcher.__module__ == "molrs.perceive"


def test_find_returns_one_group_of_live_handles_in_pattern_order() -> None:
    pattern, _ = _chain("1", "4")
    # The target is built type-4 first, so pattern order differs from the
    # target's own bead order.
    target = molrs.system.CoarseGrain()
    four = target.add_bead("4")
    one = target.add_bead("1")
    target.add_bond(four, one)

    groups = molrs.perceive.SubgraphMatcher(pattern).find(target)

    assert groups == [[one, four]]
    assert all(target.has_entity(handle) for handle in groups[0])


def test_find_without_a_match_is_an_empty_list() -> None:
    pattern, _ = _chain("1", "4")
    target, _ = _chain("1", "1")

    assert molrs.perceive.SubgraphMatcher(pattern).find(target) == []


def test_find_on_an_atomistic_target_is_a_type_error() -> None:
    pattern, _ = _chain("1", "4")
    target = molrs.system.Atomistic()
    target.add_atom("C", 0.0, 0.0, 0.0)

    with pytest.raises(TypeError):
        molrs.perceive.SubgraphMatcher(pattern).find(target)
