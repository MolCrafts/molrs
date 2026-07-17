"""The `Perceive` builder, from Python (chem-perceive-13 ac-002, ac-004).

The builder's contract is one sentence — **graph in / graph out, non-mutating**:
every ``find_*`` clones the molecule, writes the perceived facts onto the clone as
atom / bond props, and returns it. The input is never touched. That is the whole
reason it exists (the free functions it wraps have four different shapes: a side
table, an in-place mutation returning a count, a graph-out transform, and maps),
and it is the property a binding is most likely to lose — a binder that handed
PyO3 a ``&mut`` and returned ``None`` would still "work" for a caller who only
looks at the output.

The molecules are the antechamber oracle's, read from the Rust suite's one copy
(see ``antechamber_oracle``). No hand-built geometries: the same 37 molecules the
Rust perception tests use.
"""

from __future__ import annotations

import pytest

import molrs
from antechamber_oracle import CASES, REPO_ROOT, Case, build, report

#: Every finder the Rust builder publishes. A binding that exposes the builder but
#: only half its methods passes a smoke test and fails the user.
FIND_METHODS = (
    "find_rings",
    "find_aromaticity",
    "find_hydrogens",
    "find_stereo",
    "find_rotatable",
    "find_bond_types",
    "find_equivalence_classes",
)


def _case(name: str) -> Case:
    return next(case for case in CASES if case.name == name)


# --------------------------------------------------------------------------- #
# Exposure                                                                     #
# --------------------------------------------------------------------------- #


def test_perceive_is_exported() -> None:
    """`molrs.Perceive` exists, and is re-exported by the package."""
    assert "Perceive" in dir(molrs)
    assert "Perceive" in molrs.__all__


def test_perceive_is_declared_in_the_type_stub() -> None:
    """The stub ships the class — `py.typed` promises a typed API, not a partial one."""
    stub = (REPO_ROOT / "molrs-python/python/molrs/molrs.pyi").read_text()
    assert "class Perceive" in stub


def test_every_find_method_is_exposed() -> None:
    perceive = molrs.Perceive()
    for method in FIND_METHODS:
        assert callable(getattr(perceive, method)), method


# --------------------------------------------------------------------------- #
# Graph in / graph out                                                         #
# --------------------------------------------------------------------------- #


def test_find_rings_returns_a_graph() -> None:
    """ac-002: `Perceive().find_rings(g)` returns a graph — not a list of rings.

    The free function `molrs.find_rings` (which stays, and stays list-shaped) is the
    side-table API. The builder is the graph-out one, and the difference is the
    point: a graph composes with the next finder, a list does not.
    """
    mol, handles = build(_case("benzene"))
    perceived = molrs.Perceive().find_rings(mol)

    assert isinstance(perceived, molrs.Atomistic)
    assert perceived is not mol
    assert perceived.n_atoms == mol.n_atoms


def test_find_rings_projects_the_ring_facts_onto_every_atom() -> None:
    """Ring props land on ring and non-ring atoms alike: acyclic ones say `0`."""
    case = _case("benzene")
    mol, handles = build(case)
    perceived = molrs.Perceive().find_rings(mol)

    carbons = [h for h, el in zip(handles, case.elements, strict=True) if el == "C"]
    hydrogens = [h for h, el in zip(handles, case.elements, strict=True) if el == "H"]

    for handle in carbons:
        assert perceived.get(handle, "is_in_ring"), "benzene carbon is in a ring"
        assert perceived.get(handle, "n_rings") == 1
    for handle in hydrogens:
        assert perceived.has(handle, "is_in_ring"), (
            "acyclic atoms are flagged, not skipped"
        )
        assert not perceived.get(handle, "is_in_ring")
        assert perceived.get(handle, "n_rings") == 0


def test_find_rings_does_not_mutate_its_input() -> None:
    """The input carries no perceived fact afterwards — the clone took them all."""
    mol, handles = build(_case("benzene"))
    before = mol.n_atoms

    molrs.Perceive().find_rings(mol)

    assert mol.n_atoms == before
    for handle in handles:
        assert not mol.has(handle, "is_in_ring")
        assert not mol.has(handle, "n_rings")


def test_find_aromaticity_does_not_rewrite_the_input_bond_orders() -> None:
    """Cloning is load-bearing here: perception rewrites aromatic bond `order` to 1.5.

    Whatever bond orders the caller handed in must still be there afterwards — a
    mutating binding would silently overwrite their structure, and stash the original
    under a `kekule_order` key they never asked for.
    """
    case = _case("toluene")
    mol, handles = build(case)
    bonds = mol.relation_ids("bonds")
    before = [mol.get_relation_prop("bonds", b, molrs.keys.ORDER) for b in bonds]

    perceived = molrs.Perceive().find_aromaticity(mol)

    after = [mol.get_relation_prop("bonds", b, molrs.keys.ORDER) for b in bonds]
    assert after == before, "find_aromaticity mutated the caller's bond orders"
    assert not any(mol.has(h, "is_aromatic") for h in handles)
    assert perceived.n_atoms == mol.n_atoms


def test_find_aromaticity_marks_the_benzene_ring() -> None:
    case = _case("benzene")
    mol, handles = build(case)
    perceived = molrs.Perceive().find_aromaticity(mol)

    carbons = [h for h, el in zip(handles, case.elements, strict=True) if el == "C"]
    assert all(perceived.get(h, "is_aromatic") for h in carbons)


def test_find_hydrogens_returns_a_filled_graph_and_leaves_the_input_bare() -> None:
    """The heavy-atom skeleton of methanol (C, O) fills to CH3OH — on the clone."""
    skeleton = molrs.parse_smiles("CO").to_atomistic()
    before = skeleton.n_atoms
    assert before == 2

    filled = molrs.Perceive().find_hydrogens(skeleton)

    assert isinstance(filled, molrs.Atomistic)
    assert filled.n_atoms == 6, "CH3OH"
    assert skeleton.n_atoms == before, "find_hydrogens mutated its input"


def test_finders_compose() -> None:
    """Graph-out means the finders chain, and earlier facts survive later stages."""
    mol, handles = build(_case("phenol"))

    perceived = molrs.Perceive().find_rings(mol)
    perceived = molrs.Perceive().find_bond_types(perceived)
    perceived = molrs.Perceive().find_rotatable(perceived)

    # The ring facts from stage 1 are still on the graph after stages 2 and 3.
    assert any(perceived.get(h, "is_in_ring") for h in handles)
    bonds = perceived.relation_ids("bonds")
    assert all(
        perceived.get_relation_prop("bonds", b, "bcc_bond_type") is not None
        for b in bonds
    )
    assert all(
        perceived.get_relation_prop("bonds", b, "is_rotatable") is not None
        for b in bonds
    )


def test_find_equivalence_classes_merges_the_methyl_hydrogens() -> None:
    """Methanol's three methyl H are one topological class — antechamber's `-eq 1`.

    The class-mean over exactly this partition is what turns sqm's conformer-split
    Mulliken charges (0.053 / 0.098 / 0.053) into the symmetric base AM1-BCC needs.
    """
    case = _case("methanol")
    mol, handles = build(case)

    perceived = molrs.Perceive().find_equivalence_classes(mol)

    classes = [perceived.get(h, "equiv_class") for h in handles]
    assert all(c is not None for c in classes)
    # Atoms 2,3,4 of the oracle's methanol are the methyl hydrogens (its own
    # `gas_charges` column gives all three the identical 0.052691).
    methyl = classes[2:5]
    assert len(set(methyl)) == 1, f"methyl H split across classes: {methyl}"


# --------------------------------------------------------------------------- #
# Against antechamber                                                          #
# --------------------------------------------------------------------------- #


def test_find_bond_types_match_antechamber_on_the_full_oracle() -> None:
    """37/37 — the perception the whole AM1-BCC pipeline is keyed on.

    Bond types are where a binding bug hides best: they are integers on *relations*,
    so a binder that read the props off the input graph instead of the returned clone,
    or that lost the relation handles in the clone, gets plausible-looking output for
    the acyclic cases and silently wrong types for every aromatic and delocalized one
    (7/8/9). Hence all 37, not a smoke test.
    """
    perceive = molrs.Perceive()
    failures: list[str] = []

    for case in CASES:
        mol, handles = build(case)
        index = {handle: i for i, handle in enumerate(handles)}
        perceived = perceive.find_bond_types(mol)

        got: dict[frozenset[int], int | None] = {}
        for bond in perceived.relation_ids("bonds"):
            i, j = perceived.relation_nodes("bonds", bond)
            got[frozenset((index[i], index[j]))] = perceived.get_relation_prop(
                "bonds", bond, "bcc_bond_type"
            )

        wrong = [
            f"{i}-{j}: got {got.get(frozenset((i, j)))} want {want}"
            for i, j, want in case.bcc_bond_types
            if got.get(frozenset((i, j))) != want
        ]
        if wrong:
            failures.append(
                f"  {case.name:22} {len(wrong)} bond(s): {', '.join(wrong)}"
            )

    report("BCC bond types via Perceive().find_bond_types", failures, len(CASES))


# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


def test_find_rings_on_an_empty_graph_is_an_empty_graph() -> None:
    empty = molrs.Atomistic()
    perceived = molrs.Perceive().find_rings(empty)
    assert isinstance(perceived, molrs.Atomistic)
    assert perceived.n_atoms == 0


def test_find_rings_rejects_a_coarse_grain_leaf() -> None:
    """Perception is all-atom. A CG leaf is a TypeError, not a wrong answer."""
    with pytest.raises(TypeError):
        molrs.Perceive().find_rings(molrs.CoarseGrain())


# --------------------------------------------------------------------------- #
# The free functions the builder does NOT replace                             #
# --------------------------------------------------------------------------- #


def test_the_free_perception_functions_survive() -> None:
    """ac-004: the existing Python API keeps working across the module rename.

    `molrs::chem::…` becoming `molrs::perceive::…` is a Rust-side path change. If it
    leaks into Python — a dropped export, a renamed free function — these break, and
    they are exactly what molpy calls today.
    """
    for name in ("perceive_aromaticity", "add_hydrogens", "find_rings"):
        assert callable(getattr(molrs, name)), name
