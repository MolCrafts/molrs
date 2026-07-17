"""`AtdTypifier` from Python — one engine, seven tables (chem-perceive-13 ac-002).

ac-002: ``molrs.AtdTypifier(parameter_set="gaff")`` types a molecule and matches
``antechamber -at gaff``. The parameter set is the whole point of the binding: there
is one rule engine and seven ``ATOMTYPE_*.DEF`` tables behind it, so a binder that
wires the Python string to the wrong table — or ignores it and always builds the same
one — produces plausible atom types that are *someone else's*.

That failure is invisible to a smoke test, and it is what the multi-set tests below
exist to catch: all seven sets, all 37 oracle molecules, and an explicit "two sets
disagree" test so a constant-table binding cannot pass by accident.

The names are antechamber's ``-at`` flags, as the acceptance contract words them. The
flag and the table it walks are spelled differently (``-at gaff`` reads
``ATOMTYPE_GFF.DEF``); ``antechamber_oracle.TYPE_COLUMNS`` is the one place that
mapping is written down.
"""

from __future__ import annotations

import pytest

import molrs
from antechamber_oracle import (
    CASES,
    REPO_ROOT,
    TYPE_COLUMNS,
    Case,
    atom_types,
    build,
    report,
)


def _case(name: str) -> Case:
    return next(case for case in CASES if case.name == name)


# --------------------------------------------------------------------------- #
# Exposure                                                                     #
# --------------------------------------------------------------------------- #


def test_atd_typifier_is_exported() -> None:
    assert "AtdTypifier" in dir(molrs)
    assert "AtdTypifier" in molrs.__all__


def test_atd_typifier_is_declared_in_the_type_stub() -> None:
    stub = (REPO_ROOT / "molrs-python/python/molrs/molrs.pyi").read_text()
    assert "class AtdTypifier" in stub


def test_atd_typifier_reports_its_parameter_set() -> None:
    """The object knows which table it walks — a caller holding one can ask."""
    typifier = molrs.AtdTypifier(parameter_set="gaff")
    assert typifier.parameter_set == "gaff"


# --------------------------------------------------------------------------- #
# The parameter set is required, and it is honoured                            #
# --------------------------------------------------------------------------- #


def test_parameter_set_is_required() -> None:
    """There is no default table.

    Seven exist and they disagree; picking one silently would be picking for the
    caller. This mirrors the Rust-side gate (`parameter_set_required`) that forbids a
    `Default` on any parameter-table-owning model.
    """
    with pytest.raises(TypeError):
        molrs.AtdTypifier()  # type: ignore[call-arg]


def test_unknown_parameter_set_raises() -> None:
    """An unknown name is an error, not a fallback to some default table."""
    with pytest.raises(ValueError):
        molrs.AtdTypifier(parameter_set="not-a-table")


def test_two_parameter_sets_type_the_same_molecule_differently() -> None:
    """The wrong-set binding cannot pass by accident.

    Benzene is `ca`/`ha` in GAFF and `C.ar`/`H` in SYBYL. If both sets came back with
    the same labels, the `parameter_set` argument would be decoration — which is
    exactly the bug the seven-column oracle below could still miss if every column
    were compared against a typifier that ignored its argument and happened to be
    built with the right table by default.
    """
    mol, handles = build(_case("benzene"))

    gaff = atom_types(molrs.AtdTypifier(parameter_set="gaff").typify(mol), handles)
    sybyl = atom_types(molrs.AtdTypifier(parameter_set="sybyl").typify(mol), handles)

    assert gaff != sybyl
    assert gaff[0] == "ca"
    assert sybyl[0] == "C.ar"


# --------------------------------------------------------------------------- #
# Against antechamber — all seven tables, all 37 molecules                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("parameter_set", sorted(TYPE_COLUMNS))
def test_atom_types_match_antechamber_on_the_full_oracle(parameter_set: str) -> None:
    """37/37 per column, seven columns — the same gate the Rust suite runs.

    The columns are not restatements of each other: they disagree exactly where the
    tables do. Asserting all seven is what makes "one engine, N tables" a claim a
    binding can fail — a per-table special case (or a column read off the wrong
    field) regresses the others.
    """
    typifier = molrs.AtdTypifier(parameter_set=parameter_set)
    failures: list[str] = []

    for case in CASES:
        mol, handles = build(case)
        typed = typifier.typify(mol)
        got = atom_types(typed, handles)
        want = case.atom_types[parameter_set]

        wrong = [
            f"{k}{case.elements[k]}: got {got[k]!r} want {want[k]}"
            for k in range(case.n_atoms)
            if got[k] != want[k]
        ]
        if wrong:
            failures.append(
                f"  {case.name:22} {len(wrong)}/{case.n_atoms} atoms wrong: {', '.join(wrong)}"
            )

    report(f"ATD atom typing (-at {parameter_set})", failures, len(CASES))


# --------------------------------------------------------------------------- #
# Shape of the call                                                            #
# --------------------------------------------------------------------------- #


def test_typify_returns_a_typed_atomistic_and_keeps_the_atom_handles() -> None:
    """Graph in / graph out, and the caller's handles still address their atoms.

    A binding that rebuilt the graph would return correct types under handles the
    caller cannot use — and every per-atom assertion in this file would then be
    reading someone else's atom.
    """
    mol, handles = build(_case("acetate"))
    typed = molrs.AtdTypifier(parameter_set="gaff").typify(mol)

    assert isinstance(typed, molrs.Atomistic)
    assert typed is not mol
    assert set(typed.entities()) >= set(handles)
    assert all(typed.get(h, molrs.keys.TYPE) is not None for h in handles)


def test_typify_does_not_write_into_the_input() -> None:
    """The caller's `type` column is theirs.

    The standard AM1-BCC workflow needs GAFF types *and* BCC charges at once; a pass
    that relabelled the input would destroy the force field it was meant to complete.
    """
    mol, handles = build(_case("acetate"))
    molrs.AtdTypifier(parameter_set="bcc").typify(mol)

    assert all(not mol.has(h, molrs.keys.TYPE) for h in handles)


def test_typify_rejects_a_coarse_grain_leaf() -> None:
    with pytest.raises(TypeError):
        molrs.AtdTypifier(parameter_set="gaff").typify(molrs.CoarseGrain())


def test_an_unmatched_atom_gets_the_tables_own_catch_all_label() -> None:
    """`DU` is a rule OF the table, not a fallback the engine invents.

    Every `ATOMTYPE_*.DEF` ends with a constraint-free fall-through row, so an atom no
    rule matches comes back labelled `DU` rather than untyped. The binding must carry
    that label through unchanged — turning it into `None`, or into a silent success, is
    where a "plausible-looking" force field comes from. Refusing `DU` is the *charge*
    model's job, and is asserted there
    (`test_charge_models.py::test_bcc_refuses_an_atom_the_table_cannot_type`).

    The witness is a **bondless sulfur**. Every GAFF sulfur rule requires at least one
    bond, so a bare S matches none of them and falls through to `DU`. A noble gas would
    NOT do: `ATOMTYPE_GFF.DEF` carries a real periodic-table block (`He`, `Li`, `Be`, …),
    so a lone helium types as `He` — matched, not unmatched.
    """
    mol = molrs.Atomistic()
    sulfur = mol.add_atom("S", 0.0, 0.0, 0.0)

    typed = molrs.AtdTypifier(parameter_set="gaff").typify(mol)

    assert typed.get(sulfur, molrs.keys.TYPE) == "DU"
