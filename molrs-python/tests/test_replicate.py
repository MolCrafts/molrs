"""FFI seam of ``replicate(template, rotations, translations, frag_ids)``.

``replicate(n)`` is deleted (assembly-07). The leaf method grows *this* graph
by one rigid copy of ``template`` per row of ``rotations (N,3,3)`` /
``translations (N,3)``, stamping ``frag_id = frag_ids[c]`` on copy ``c``.
Column-wise copying, relation offsets and atomicity are proven by
``MolGraph::replicate`` in ``molrs/src/core/system/molgraph.rs``; these tests
check only that the arrays cross, the graph grows, and bad shapes map to
``ValueError``. The one position asserted is a pure translation.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest

TWO_IDENTITIES = np.stack([np.eye(3), np.eye(3)])
TWO_SHIFTS = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
TWO_FRAG_IDS = np.array([7, 9], dtype=np.int32)


def _ch_template() -> molrs.system.Atomistic:
    """C at the origin bonded to H at x = 1.09."""
    template = molrs.system.Atomistic()
    c = template.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    h = template.def_atom(element="H", x=1.09, y=0.0, z=0.0)
    template.def_bond(c, h)
    return template


class TestAtomisticReplicate:
    def test_grows_the_graph_by_one_copy_per_transform_with_frag_ids_stamped(self):
        world = molrs.system.Atomistic()
        world.def_atom(element="O", x=-5.0, y=0.0, z=0.0)
        before = set(world.entities())

        world.replicate(_ch_template(), TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)

        new = [h for h in world.entities() if h not in before]
        assert world.n_atoms == 1 + 2 * 2
        assert world.n_relations("bonds") == 2
        assert sorted(world.get(h, "frag_id") for h in new) == [7, 7, 9, 9]
        # Copy 1 is the template shifted by (10, 0, 0): its carbon sits at x = 10.
        carbons = {
            world.get(h, "frag_id"): world.get(h, "x")
            for h in new
            if world.get(h, "element") == "C"
        }
        assert carbons == {7: 0.0, 9: 10.0}

    def test_the_template_is_not_mutated(self):
        template = _ch_template()
        molrs.system.Atomistic().replicate(template, TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)
        assert template.n_atoms == 2
        assert all(not template.has(h, "frag_id") for h in template.entities())

    @pytest.mark.parametrize(
        ("rotations", "translations", "frag_ids"),
        [
            (np.zeros((2, 3)), TWO_SHIFTS, TWO_FRAG_IDS),
            (TWO_IDENTITIES, np.zeros((2, 2)), TWO_FRAG_IDS),
            (TWO_IDENTITIES, TWO_SHIFTS[:1], TWO_FRAG_IDS),
            (TWO_IDENTITIES, TWO_SHIFTS, np.array([7, 9, 11], dtype=np.int32)),
        ],
        ids=[
            "rotations-2x3",
            "translations-2x2",
            "translations-count",
            "frag_ids-count",
        ],
    )
    def test_a_wrong_shape_is_a_value_error_and_leaves_the_graph_unchanged(
        self, rotations, translations, frag_ids
    ):
        world = molrs.system.Atomistic()
        with pytest.raises(ValueError):
            world.replicate(_ch_template(), rotations, translations, frag_ids)
        assert world.n_atoms == 0


class TestPortedReplicate:
    def test_copies_keep_the_template_ports(self):
        template = molrs.system.Atomistic()
        o = template.def_atom(element="O", x=0.0, y=0.0, z=0.0)
        h = template.def_atom(element="H", x=0.96, y=0.0, z=0.0)
        template.def_bond(o, h)
        template.def_port(o, h, "$")

        world = molrs.system.Atomistic()
        world.replicate(template, TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)

        assert world.n_atoms == 4
        assert world.n_ports == 2
        assert sorted(world.frag_id(a) for a in world.entities()) == [7, 7, 9, 9]


class TestCoarseGrainReplicate:
    def test_grows_by_one_bead_per_transform(self):
        template = molrs.system.CoarseGrain()
        template.def_bead(bead_type="A", x=0.0, y=0.0, z=0.0)

        world = molrs.system.CoarseGrain()
        world.replicate(template, TWO_IDENTITIES, TWO_SHIFTS, TWO_FRAG_IDS)

        assert world.n_beads == 2
        assert sorted(world.get(b, "frag_id") for b in world.entities()) == [7, 9]
