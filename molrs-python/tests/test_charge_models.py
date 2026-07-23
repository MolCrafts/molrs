"""The three charge models from Python (chem-perceive-13 ac-001, ac-003).

This is Python's first **native** AM1-BCC. Until this binding lands, molpy's only
AM1-BCC is shelling out to `antechamber -c bcc`; afterwards it can compare the two,
which is the verification that is missing today.

# What these tests are for, and what they are not

The chemistry is already pinned in Rust: `molrs/tests/ff/charge/` matches antechamber
37/37 on the `bcc`, `abcg2` and `gas` columns. What is unproven is the **binding** —
that it does not lose precision, transpose an array, drop a column, or reach for the
wrong parameter table. So every expected number here comes from the Rust suite's ONE
oracle (parsed, not copied — see `antechamber_oracle`), and the assertions are aimed
at the seam:

* **values** — all 37 molecules, all three models, at the Rust suite's own
  `CHARGE_TOL`. Catches a transposed / dropped / mis-ordered column.
* **precision** — `float64` dtype, plus two exactness assertions the antechamber
  columns cannot make (they carry six decimals): BCC increments are pairwise
  antisymmetric, so the total charge is conserved to machine precision, and methane's
  four C–H increments sum to exactly 4 x 0.0393 e. An `f32` anywhere in the binding
  lands ~1e-8 off and fails both, while sailing through a 1e-4 comparison.
* **one implementation** — the surviving free function `compute_gasteiger_charges`
  must be *bit-identical* to `GasteigerModel().assign(mol)`, because it is required to
  delegate to it rather than keep a second Gasteiger alive.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs
from antechamber_oracle import CASES, CHARGE_TOL, REPO_ROOT, Case, build, report

#: The exactness column. Charges are dimensionless and O(1); a float64 sum of ~30 of
#: them rounds at ~1e-16, while an f32 round-trip lands at ~1e-8. 1e-12 sits four
#: orders below the f32 error and four above the f64 noise, so it separates them
#: cleanly — which is what makes it a *precision* gate rather than a value gate.
EXACT_TOL = 1.0e-12


def _case(name: str) -> Case:
    return next(case for case in CASES if case.name == name)


# --------------------------------------------------------------------------- #
# ac-003 — all three models are bound, with one calling convention             #
# --------------------------------------------------------------------------- #


def test_all_three_charge_models_are_exported() -> None:
    for name in ("BccModel", "MullikenModel", "GasteigerModel"):
        assert name in dir(molrs), name
        assert name in molrs.__all__, name


def test_all_three_charge_models_are_declared_in_the_type_stub() -> None:
    stub = (REPO_ROOT / "molrs-python/python/molrs/_lib.pyi").read_text()
    for name in ("BccModel", "MullikenModel", "GasteigerModel"):
        assert f"class {name}" in stub, name


def test_the_charge_bindings_live_in_their_own_module() -> None:
    """The spec creates `molrs-python/src/ff/charge.rs`; it does not grow `ff/mod.rs`.

    A source gate, because the binding works either way and only the source can say
    where it went. `molrs-python/src/ff/` is one 44K `mod.rs` today, and three charge
    models plus a parameter-set enum is precisely the increment that turns a large file
    into an unreviewable one (house rule: many small files, 200-400 lines).
    """
    ff = REPO_ROOT / "molrs-python/src/ff"
    charge = ff / "charge.rs"
    if not charge.is_file():
        charge = ff / "charge" / "mod.rs"

    assert charge.is_file(), (
        "expected the charge-model bindings in molrs-python/src/ff/charge.rs "
        "(or src/ff/charge/mod.rs)"
    )
    source = charge.read_text()
    for name in ("BccModel", "MullikenModel", "GasteigerModel"):
        assert name in source, f"{name} is bound somewhere other than ff/charge.rs"


def _models() -> list[tuple[str, object]]:
    return [
        ("BccModel(bcc)", molrs.BccModel(parameter_set="bcc")),
        ("BccModel(abcg2)", molrs.BccModel(parameter_set="abcg2")),
        ("MullikenModel", molrs.MullikenModel()),
        ("GasteigerModel", molrs.GasteigerModel()),
    ]


def test_every_model_answers_the_same_two_calls() -> None:
    """ac-003: one calling convention — `assign(mol, qm)` and `needs_equivalencing()`.

    The Rust trait is object-safe precisely so a caller can hold any model without
    knowing which; a binding that gave Gasteiger a different method name would put the
    branch back into every caller. Gasteiger takes the `qm` argument and ignores it —
    that is the row that keeps the trait honest, and it must survive the crossing.
    """
    case = _case("methanol")
    mol, _ = build(case)
    qm = np.asarray(case.am1_charges_raw, dtype=np.float64)

    for label, model in _models():
        assert isinstance(model.needs_equivalencing(), bool), label  # type: ignore[attr-defined]
        charges = model.assign(mol, qm)  # type: ignore[attr-defined]
        assert isinstance(charges, np.ndarray), label
        assert charges.shape == (case.n_atoms,), label
        assert charges.dtype == np.float64, label


def test_needs_equivalencing_is_declared_per_model() -> None:
    """antechamber's `-eq` default is per charge METHOD: 1 for bcc/abcg2, 0 otherwise."""
    assert molrs.BccModel(parameter_set="bcc").needs_equivalencing() is True
    assert molrs.BccModel(parameter_set="abcg2").needs_equivalencing() is True
    assert molrs.MullikenModel().needs_equivalencing() is False
    assert molrs.GasteigerModel().needs_equivalencing() is False


def test_gasteiger_needs_no_qm_charges_at_all() -> None:
    """The zero-QM corner: `assign(mol)` with no second argument is the whole call."""
    case = _case("methanol")
    mol, _ = build(case)

    charges = molrs.GasteigerModel().assign(mol)

    assert isinstance(charges, np.ndarray)
    assert charges.shape == (case.n_atoms,)


# --------------------------------------------------------------------------- #
# ac-001 — native AM1-BCC is reachable, and it is antechamber's answer         #
# --------------------------------------------------------------------------- #


def test_bcc_correct_returns_one_float64_charge_per_atom() -> None:
    """`BccModel(...).correct(mol, am1)` returns an ndarray — ac-001, literally."""
    case = _case("methane")
    mol, _ = build(case)
    am1 = np.asarray(case.am1_charges, dtype=np.float64)

    charges = molrs.BccModel(parameter_set="bcc").correct(mol, am1)

    assert isinstance(charges, np.ndarray)
    assert charges.ndim == 1, "charges are a column, not an N x 1 (or 1 x N) matrix"
    assert charges.shape == (case.n_atoms,)
    assert charges.dtype == np.float64


@pytest.mark.parametrize("parameter_set", ["bcc", "abcg2"])
def test_correct_matches_antechamber_on_the_full_oracle(parameter_set: str) -> None:
    """37/37, both correction families — `-c bcc` and `-c abcg2`.

    `correct` consumes the base charges antechamber's `-eq 1` already averaged
    (`am1_charges`), which is the stage this method IS. Both families correct the SAME
    base charges, so a binding that special-cased one regresses the other column.
    """
    model = molrs.BccModel(parameter_set=parameter_set)
    failures: list[str] = []

    for case in CASES:
        mol, _ = build(case)
        am1 = np.asarray(case.am1_charges, dtype=np.float64)
        got = model.correct(mol, am1)
        want = np.asarray(case.charges(parameter_set), dtype=np.float64)

        assert got.shape == want.shape, f"{case.name}: one charge per atom"
        delta = np.abs(got - want)
        bad = np.flatnonzero(delta > CHARGE_TOL)
        if bad.size:
            detail = ", ".join(
                f"{k}{case.elements[k]}: {got[k]:+.4f} vs {want[k]:+.4f}" for k in bad
            )
            failures.append(f"  {case.name:22} max|dq|={delta.max():.4f}  {detail}")

    report(f"AM1-BCC ({parameter_set}) end-to-end", failures, len(CASES))


def test_assign_averages_the_raw_am1_charges_before_correcting() -> None:
    """`assign` is the door that does antechamber's `-eq 1`; `correct` is not.

    sqm hands back one conformer's Mulliken charges, and they are NOT symmetric —
    methanol's three methyl H come out split. `assign` declares `needs_equivalencing`
    and honours it; feeding the same raw charges to `correct` must therefore give a
    DIFFERENT (and wrong) answer. A binding that wired `assign` straight through to
    `correct` passes every other test in this file and fails this one.
    """
    case = _case("methanol")
    mol, _ = build(case)
    raw = np.asarray(case.am1_charges_raw, dtype=np.float64)
    want = np.asarray(case.bcc_charges, dtype=np.float64)
    model = molrs.BccModel(parameter_set="bcc")

    assert not np.allclose(raw, case.am1_charges, atol=CHARGE_TOL), (
        "methanol's raw sqm charges must differ from the equivalenced ones, or this "
        "test cannot tell the two paths apart"
    )

    assert np.abs(model.assign(mol, raw) - want).max() <= CHARGE_TOL
    assert np.abs(model.correct(mol, raw) - want).max() > CHARGE_TOL


def test_bcc_and_abcg2_do_not_agree_everywhere() -> None:
    """Two parameter sets, two answers — else `parameter_set` is decoration.

    BCCPARM.DAT and BCCPARM_ABCG2.DAT are different tables read against different
    ATOMTYPE tables. If the Python argument were ignored, both would return the same
    charges and the parity test above would still pass for whichever set the binding
    happened to hard-code.
    """
    bcc = molrs.BccModel(parameter_set="bcc")
    abcg2 = molrs.BccModel(parameter_set="abcg2")

    differ = 0
    for case in CASES:
        mol, _ = build(case)
        am1 = np.asarray(case.am1_charges, dtype=np.float64)
        if np.abs(bcc.correct(mol, am1) - abcg2.correct(mol, am1)).max() > CHARGE_TOL:
            differ += 1

    assert differ > 0, "BCC and ABCG2 returned identical charges on all 37 molecules"


# --------------------------------------------------------------------------- #
# Precision — what a 1e-4 comparison against a 6-decimal oracle cannot see     #
# --------------------------------------------------------------------------- #


def test_the_increments_conserve_total_charge_to_machine_precision() -> None:
    """BCC increments are pairwise antisymmetric: what goes in as a total comes out.

    Exact in f64 (~1e-16 over 30 atoms), and ~1e-8 off through an f32. This is the
    assertion that catches a binding that quietly narrows the array on the way out —
    a bug the antechamber columns cannot see, since they only carry six decimals.

    antechamber does not renormalize either: it carries the AM1 rounding residual
    straight through, so the target is `sum(am1)`, never the integer net charge.
    """
    model = molrs.BccModel(parameter_set="bcc")
    failures: list[str] = []

    for case in CASES:
        mol, _ = build(case)
        am1 = np.asarray(case.am1_charges, dtype=np.float64)
        drift = abs(float(model.correct(mol, am1).sum()) - float(am1.sum()))
        if drift > EXACT_TOL:
            failures.append(f"  {case.name:22} total charge drifted by {drift:.3e}")

    report("BCC charge conservation", failures, len(CASES))


def test_methane_increments_are_exact() -> None:
    """With a zero base, what comes back IS the table: C–H is +0.0393 e per bond.

    The same number, at the same tolerance, as the Rust doctest on `BccModel::correct`
    — so this is the one place a Python charge is checked against a *Rust-side* value
    rather than antechamber's six-decimal print, and it is why `EXACT_TOL` is 1e-12.
    """
    mol, _ = build(_case("methane"))
    charges = molrs.BccModel(parameter_set="bcc").correct(
        mol, np.zeros(5, dtype=np.float64)
    )

    assert abs(float(charges[0]) - 4.0 * 0.0393) < EXACT_TOL
    assert abs(float(charges[1]) + 0.0393) < EXACT_TOL


# --------------------------------------------------------------------------- #
# Gasteiger — and the free function that must NOT be a second implementation   #
# --------------------------------------------------------------------------- #


def test_gasteiger_matches_antechamber_on_the_full_oracle() -> None:
    """37/37 against `-c gas`. No sqm ran for this column: it is pure topology."""
    model = molrs.GasteigerModel()
    failures: list[str] = []

    for case in CASES:
        mol, _ = build(case)
        got = model.assign(mol)
        want = np.asarray(case.gas_charges, dtype=np.float64)

        assert got.shape == want.shape, f"{case.name}: one charge per atom"
        delta = np.abs(got - want)
        bad = np.flatnonzero(delta > CHARGE_TOL)
        if bad.size:
            detail = ", ".join(
                f"{k}{case.elements[k]}: {got[k]:+.6f} vs {want[k]:+.6f}" for k in bad
            )
            failures.append(f"  {case.name:22} max|dq|={delta.max():.6f}  {detail}")

    report("Gasteiger (-c gas) end-to-end", failures, len(CASES))


def test_the_free_gasteiger_function_delegates_to_the_model() -> None:
    """One Gasteiger, two doors — and the doors must agree BITWISE.

    `compute_gasteiger_charges` survives as a compat layer (molpy calls it today), but
    it is required to delegate to `GasteigerModel`. Two implementations would drift,
    and the drift would be invisible at any tolerance the oracle can express — so the
    comparison here is exact equality, not `allclose`.
    """
    model = molrs.GasteigerModel()

    for case in CASES:
        mol, handles = build(case)
        by_handle = dict(molrs.compute_gasteiger_charges(mol))
        free = np.asarray([by_handle[h] for h in handles], dtype=np.float64)

        assert np.array_equal(free, model.assign(mol)), (
            f"{case.name}: compute_gasteiger_charges and GasteigerModel disagree — "
            "there is a second Gasteiger implementation"
        )


def test_mulliken_is_a_pass_through() -> None:
    """No correction, no equivalencing: the same bits back.

    Exact equality again — Mulliken's whole content is that it hands back what it was
    given, so anything but bit-identity is a bug in the crossing (a float32 round-trip
    would show up here and nowhere else).
    """
    case = _case("acetic_acid")
    mol, _ = build(case)
    qm = np.asarray(case.am1_charges_raw, dtype=np.float64)

    assert np.array_equal(molrs.MullikenModel().assign(mol, qm), qm)


# --------------------------------------------------------------------------- #
# Refusals — no fallback values                                                #
# --------------------------------------------------------------------------- #


def test_bcc_assign_without_qm_charges_raises() -> None:
    """It does not invent a base to correct."""
    mol, _ = build(_case("methane"))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        molrs.BccModel(parameter_set="bcc").assign(mol)


def test_a_charge_array_of_the_wrong_length_raises() -> None:
    mol, _ = build(_case("methane"))  # 5 atoms
    with pytest.raises((ValueError, RuntimeError)):
        molrs.BccModel(parameter_set="bcc").correct(mol, np.zeros(4, dtype=np.float64))


def test_bcc_refuses_an_atom_the_table_cannot_type() -> None:
    """A `DU` atom has no BCC row, and no charge. It raises; it is not left alone.

    The bondless case is the sharp one: nothing looks its type up, so a model without
    this refusal would return the input charge untouched — a plausible-looking answer,
    which is exactly what molrs's no-fallback-values rule forbids.

    The witness is a **bare sulfur**, the element molrs's own `ff/charge/model.rs` names
    for this: every BCC sulfur rule requires at least one bond, so a bondless S types as
    `DU`. A lone helium would NOT work — `ATOMTYPE_BCC.DEF` has a real `He` row, so it
    types fine, has no bonds, and there is nothing left to refuse.
    """
    mol = molrs.Atomistic()
    mol.add_atom("S", 0.0, 0.0, 0.0)

    with pytest.raises((ValueError, RuntimeError)):
        molrs.BccModel(parameter_set="bcc").correct(mol, np.zeros(1, dtype=np.float64))


def test_an_unknown_parameter_set_raises() -> None:
    with pytest.raises(ValueError):
        molrs.BccModel(parameter_set="gaff")  # an ATOMTYPE table, not a BCC family


def test_the_bcc_parameter_set_is_required() -> None:
    """There is no default correction table: there is BCCPARM.DAT and BCCPARM_ABCG2.DAT.

    A model built without one constructs, type-checks, and then fails on the first bond
    it sees — the "constructible but non-functional" shape the Rust-side gate
    (`parameter_set_required`) forbids. The binding must not reintroduce it with a
    default argument.
    """
    with pytest.raises(TypeError):
        molrs.BccModel()  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Immutability                                                                 #
# --------------------------------------------------------------------------- #


def test_correct_writes_nothing_into_the_molecule() -> None:
    """Charges are the return value; the caller's `type` column stays theirs.

    The standard AM1-BCC workflow needs GAFF types AND BCC charges at once. A charge
    pass that wrote its internal BCC codes (`11`, `91`) into `type` would destroy the
    force field it was meant to complete.
    """
    case = _case("acetate")
    mol, handles = build(case)

    molrs.BccModel(parameter_set="bcc").correct(
        mol, np.asarray(case.am1_charges, dtype=np.float64)
    )

    for handle in handles:
        assert not mol.has(handle, molrs.keys.TYPE), (
            "a BCC code was written into `type`"
        )
        assert not mol.has(handle, molrs.keys.CHARGE), "a charge was written back"


def test_correct_does_not_consume_the_input_array() -> None:
    """The `am1` array crosses as data, not as a buffer to write into."""
    case = _case("acetate")
    mol, _ = build(case)
    am1 = np.asarray(case.am1_charges, dtype=np.float64)
    before = am1.copy()

    molrs.BccModel(parameter_set="bcc").correct(mol, am1)

    assert np.array_equal(am1, before)
