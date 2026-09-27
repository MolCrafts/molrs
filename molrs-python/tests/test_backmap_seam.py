"""The operator's backmap script, run through every seam it crosses.

Regression example (trace-assembly-07, ac-008). This is the operator-ruled
exception to "no multi-stage chains under ``tests/``" (notes.md 2026-09-26):
one public-API script composing the backmap classes exactly as the binding
script ``backmap_pe_pma/backmap.py`` does, steps 1 (the tuple-key coordinate
write only) and 3-6,

    atoms["x", "y", "z"] = arr -> SubgraphMatcher.find -> Coarsener.coarsen
    -> Perceive.linear_paths -> Trace(positions) / bead_types
    -> Assembler(lib, TracePlacer()).assemble
    -> ElementTypifier().typify(world.to_atomistic()).

The library ships only the classes; the composition lives here. Every number
asserted is a hand count on the fixtures below, not a re-derived geometry:
matching, coarsening, path walking, placement, linking and typing are proven
by the Rust suites.

Fixtures, built by hand in process (no third-party software):

- a CG frame: an 8-bead head-to-head chain, bead types ``4 1 1 1 1 1 1 4``,
  bead ``i`` at ``x = 4 i`` Å, ``mass = 72.0`` g/mol, bonded ``(i, i + 1)``,
  plus one type-``2`` bead at ``x = 40`` Å with ``mass = 7.0`` g/mol and no
  bond; no ``mol_id`` column;
- the library: the ported monomer ``H0-C0-C1-H1`` (a ``Fragment``, ports
  ``(C0, H0, ">")`` and ``(C1, H1, "<")``; C 12.011, H 1.008 g/mol) under
  ``"M"`` and a one-atom Li ``Atomistic`` (6.94 g/mol) under ``"Li"``.

Hand counts: ``1-1-1-4`` embeds in the chain once from each end (beads 0-3
and 4-7, disjoint) and ``2`` once, so 3 groups and 3 sites; the chain's
bond 3-4 joins the two chain sites, so the paths are one 2-site chain and one
lone site (sorted lengths ``[1, 2]``). Two monomer copies are 8 atoms and 4
ports; their one link removes one hydrogen per joined port, leaving 6 atoms
and 2 ports; the Li adds 1 atom: 7 atoms, 2 ports, ``mol_id`` 1 (the chain)
and 2 (the Li), and every atom type is an element of C, H or Li.
"""

from __future__ import annotations

import numpy as np

import molrs
from molrs import CoarseGrain
from molrs.builder import Assembler, TracePlacer
from molrs.ff.typifier import ElementTypifier
from molrs.io import CGSmilesIR
from molrs.perceive import Coarsener, Perceive, SubgraphMatcher

N_CHAIN = 8


def _cg_frame() -> molrs.Frame:
    """The CG chain plus one lone ``2`` bead, coordinates still zero."""
    n = N_CHAIN + 1
    index = np.arange(N_CHAIN)
    return molrs.Frame(
        {
            "atoms": {
                "type": np.array(["4", "1", "1", "1", "1", "1", "1", "4", "2"]),
                "x": np.zeros(n, dtype=np.float64),
                "y": np.zeros(n, dtype=np.float64),
                "z": np.zeros(n, dtype=np.float64),
                "mass": np.array([72.0] * N_CHAIN + [7.0], dtype=np.float64),
            },
            "bonds": {
                "atomi": index[:-1],
                "atomj": index[1:],
            },
        }
    )


def _coordinates() -> np.ndarray:
    """``(9, 3)``: chain bead ``i`` at ``x = 4 i`` Å, the lone bead at 40 Å."""
    xyz = np.zeros((N_CHAIN + 1, 3), dtype=np.float64)
    xyz[:N_CHAIN, 0] = 4.0 * np.arange(N_CHAIN)
    xyz[N_CHAIN, 0] = 40.0
    return xyz


def _ported_monomer() -> molrs.Fragment:
    """``H0-C0-C1-H1`` with ports ``(C0, H0, ">")`` and ``(C1, H1, "<")``.

    Kept beside the script, not imported from another test module, so this
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


def _lithium() -> molrs.Atomistic:
    mol = molrs.Atomistic()
    mol.def_atom(element="Li", x=0.0, y=0.0, z=0.0, mass=6.94)
    return mol


def test_operator_backmap_script_crosses_every_seam() -> None:
    # 1. CG input: coordinates written through the tuple key.
    frame = _cg_frame()
    atoms = frame["atoms"]
    atoms["x", "y", "z"] = _coordinates()
    cg = CoarseGrain.from_frame(frame)

    # 2. Library: name -> one molecule, a ported Fragment or a portless
    #    Atomistic.
    lib = {"M": _ported_monomer(), "Li": _lithium()}

    # 3. Rules: bead-group pattern -> molecule name; each group is one site.
    rules = {"{[#1][#1][#1][#4]}": "M", "{[#2]}": "Li"}
    groups: list[list[int]] = []
    names: list[str] = []
    for pattern, name in rules.items():
        found = SubgraphMatcher(CGSmilesIR(pattern).to_coarsegrain()).find(cg)
        groups += found
        names += [name] * len(found)
    assert len(groups) == 3
    sites = Coarsener(cg).coarsen(groups, names)
    assert type(sites) is molrs.CoarseGrain
    assert sites.n_beads == 3

    # 4. Sites -> traces: one ordered path per chain.
    paths = Perceive(sites).linear_paths()
    assert sorted(len(path) for path in paths) == [1, 2]
    traces = [molrs.Trace(sites.positions(path)) for path in paths]
    seqs = [sites.bead_types(path) for path in paths]

    # 5. Assemble: one placed copy per site, ports join each chain.
    world = Assembler(lib, TracePlacer()).assemble(traces, seqs)
    assert type(world) is molrs.Fragment
    assert world.n_atoms == 7
    assert world.n_ports == 2
    mol_id = np.asarray(world.to_frame()["atoms"]["mol_id"])
    assert set(mol_id.tolist()) == {1, 2}

    # 6. Element types for the writer.
    typed = ElementTypifier().typify(world.to_atomistic())
    assert type(typed) is molrs.Atomistic
    assert set(typed.to_frame()["atoms"]["type"]) <= {"C", "H", "Li"}
