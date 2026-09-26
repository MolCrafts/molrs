"""The operator's backmap script, run through every seam it crosses.

Regression example (backmap-primitives-07, ac-012). This is the
operator-ruled exception to "no multi-stage chains under ``tests/``": one
public-API script composing the backmap primitives exactly as a user writes
it,

    to_coarsegrain pattern -> subset + from_frame target -> SubgraphMatcher.find
    -> center -> translate -> merge -> link -> to_atomistic.

The library ships only the primitives; the composition lives here. Every
number asserted is a hand count on the fixtures below, not a re-derived
geometry: the centre, the match semantics and the link chemistry are proven
by the Rust suites.

Fixtures, built by hand in process (no third-party software):

- an 8-bead head-to-head CG chain, bead types ``4 1 1 1 1 1 1 4``, bead ``i``
  at ``x = 4 i`` Å (``y = z = 0``), ``mass = 72.0`` g/mol, all ``mol_id = 1``,
  bonded ``(i, i + 1)``;
- the ported monomer ``H0-C0-C1-H1`` with ports ``(C0, H0, ">")`` and
  ``(C1, H1, "<")``, positions C0 (0, 0, 0), C1 (1.54, 0, 0), H0 (-1, 0, 0),
  H1 (2.54, 0, 0) Å, masses C 12.011 and H 1.008 g/mol.

Hand counts: the pattern ``1-1-1-4`` embeds in the chain once from each end,
so there are 2 groups; merging two 4-atom monomers gives 8 atoms and 4 ports;
one link removes one leaving hydrogen per port, leaving 6 atoms and 2 ports.
"""

from __future__ import annotations

import molrs
import numpy as np
from molrs import CoarseGrain, perceive
from molrs.io import CGSmilesIR

N_BEADS = 8


def _cg_frame() -> molrs.Frame:
    """The 8-bead head-to-head CG chain as a frame (Å, g/mol)."""
    index = np.arange(N_BEADS)
    return molrs.Frame(
        {
            "atoms": {
                "type": np.array(["4", "1", "1", "1", "1", "1", "1", "4"]),
                "x": 4.0 * index.astype(np.float64),
                "y": np.zeros(N_BEADS, dtype=np.float64),
                "z": np.zeros(N_BEADS, dtype=np.float64),
                "mass": np.full(N_BEADS, 72.0, dtype=np.float64),
                "mol_id": np.ones(N_BEADS, dtype=np.int64),
            },
            "bonds": {
                "atomi": index[:-1],
                "atomj": index[1:],
            },
        }
    )


def _ported_monomer() -> molrs.Fragment:
    """``H0-C0-C1-H1`` with ports ``(C0, H0, ">")`` and ``(C1, H1, "<")``.

    Kept beside the script, not imported from ``test_fragment.py``, so this
    regression example runs on its own.
    """
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


def test_operator_backmap_composition_crosses_every_seam() -> None:
    frame = _cg_frame()
    monomer = _ported_monomer()

    # 1. The bead-group pattern, from a CGsmiles string.
    pattern = CGSmilesIR("{[#1][#1][#1][#4]}").to_coarsegrain()
    # 2. One molecule of the CG frame, as a CoarseGrain.
    target = CoarseGrain.from_frame(frame.subset(frame["atoms", "mol_id"] == 1))
    # 3. Every embedding of the pattern, as target bead handles.
    groups = perceive.SubgraphMatcher(pattern).find(target)
    assert len(groups) == 2

    # 4. Per group: place a copy of the monomer on the group's centre and
    #    merge it into the world, keeping each copy's ports as world handles.
    world = molrs.Fragment()
    copies: list[dict[str, int]] = []
    for group in groups:
        unit = monomer.copy()
        bead_center = target.center(group)
        unit_center = unit.center()
        for center in (bead_center, unit_center):
            assert isinstance(center, np.ndarray)
            assert center.dtype == np.float64
            assert center.shape == (3,)
        # ``translate`` takes the 1-D ndarray ``center`` returns, unwrapped.
        unit.translate(bead_center - unit_center)
        own_ports = {port["port_kind"]: port.handle for port in unit.ports}

        maps = world.merge(unit)

        assert isinstance(maps, tuple)
        assert len(maps) == 2
        atom_map, port_map = maps
        assert isinstance(atom_map, dict)
        assert isinstance(port_map, dict)
        copies.append({kind: port_map[handle] for kind, handle in own_ports.items()})
    assert world.n_atoms == 8
    assert world.n_ports == 4

    # 5. Join copy 1's ">" to copy 2's "<".
    bond = world.link(copies[0][">"], copies[1]["<"])
    assert isinstance(bond, int)

    # 6. The finished world as a public Atomistic.
    atomistic = world.to_atomistic()
    assert type(atomistic) is molrs.Atomistic
    assert atomistic.n_atoms == 6
    assert world.n_ports == 2
