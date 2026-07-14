"""chem-perceive-15 — **Python reproduces Rust bit for bit.** Not "close". The same number.

What this file can and cannot prove, stated up front, because the difference is the
whole design:

**There is only ONE implementation.** The Python bindings are pyo3 wrappers that call
the same Rust functions the Rust suite calls; ``molrs-python/python/molrs/*.py`` adds no
arithmetic to the force-field path. So "Python computes the same value as Rust" is not
a *falsifiable* claim about two computations — it is a structural fact about one.

What IS falsifiable, and what the defect this criterion names actually is, is
**precision loss at the boundary**: an ``f32`` round-trip, a renormalization, a
transposed array, a dropped column. Those are all invisible to a 1e-4 comparison against
a six-decimal oracle — an ``f32`` round-trip misses charge conservation by four orders
of magnitude and sails straight through it. So this file asserts, on the boundary
itself:

1. every array on the path is ``float64`` (:func:`test_charges_are_float64`);
2. the values carry **more than 24 bits of mantissa** — i.e. they cannot have survived
   an ``f32`` round-trip (:func:`test_charges_did_not_survive_an_f32_round_trip`);
3. total charge is conserved to **1e-12** (:func:`test_charge_conservation_to_1e_12`);
4. the AM1 residual is **carried, not scrubbed** (:func:`test_the_binding_does_not_renormalize`);
5. every **bit-level** invariant the Rust suite asserts holds identically here —
   equivalence classes share bits, acetate's two oxygens share bits, a conformational
   change moves no bit, MMFF94 and MMFF94s are bit-identical without a delocalized N.

An ``f32`` anywhere on the path breaks (2), (3) and (5) at once. That is the gate.

The oracle is the Rust suite's ONE copy (``antechamber_oracle.py`` parses
``molrs/tests/ff/typifier/antechamber_oracle.rs``), so the two suites cannot drift
apart, and neither can their tolerances.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import numpy as np
import pytest

import molrs
from antechamber_oracle import CASES, Case, build

#: Charge conservation. Not a physics tolerance — an ARITHMETIC one: the BCC increments
#: are pairwise antisymmetric (they conserve exactly) and the equivalence class-mean
#: conserves to ULP, so anything above this is a lost bit, not a rounded number.
CONSERVATION_TOL: Final[float] = 1e-12

#: The MMFF fixtures the Rust suite validates against RDKit. Scanned, never listed.
MMFF_FIXTURES: Final[Path] = (
    Path(__file__).resolve().parents[2] / "molrs" / "tests" / "ff" / "mmff" / "fixtures"
)


def _bcc(case: Case) -> np.ndarray:
    """The chain: molecule -> Perceive -> AtdTypifier -> BccModel, through the bindings."""
    mol, _ = build(case)
    qm = np.asarray(case.am1_charges_raw, dtype=np.float64)
    return molrs.BccModel(parameter_set="bcc").assign(mol, qm)


# --------------------------------------------------------------------------- #
# 1 & 2 — the boundary carries f64, and can be proven to                       #
# --------------------------------------------------------------------------- #


def test_charges_are_float64() -> None:
    """Every array that crosses the boundary is float64. No narrowing, anywhere."""
    for case in CASES:
        q = _bcc(case)
        assert q.dtype == np.float64, f"{case.name}: charges came back as {q.dtype}"
        assert q.shape == (case.n_atoms,), f"{case.name}: shape {q.shape}"


def test_charges_did_not_survive_an_f32_round_trip() -> None:
    """The charges carry more than 24 bits of mantissa — so no ``f32`` touched them.

    This is the assertion with teeth, and it is a REVERSE one. A binding that narrowed
    to ``f32`` and widened back would return values that are *exactly* representable in
    ``f32`` — and every value check against a six-decimal oracle would still pass, at
    1e-4, forever. The only way to see it is to ask for the bits back.

    A float that has been through an ``f32`` is equal to its own ``float32`` cast.
    Ours must not be — not for every atom (a charge of exactly 0.0 or -0.5 is
    representable in ``f32`` legitimately), but *somewhere* in a 37-molecule,
    ~400-atom population, which is a certainty for real AM1-BCC output.
    """
    survivors = 0
    total = 0
    for case in CASES:
        q = _bcc(case)
        narrowed = q.astype(np.float32).astype(np.float64)
        survivors += int(np.count_nonzero(q != narrowed))
        total += q.size

    assert survivors > 0, (
        f"every one of {total} charges is exactly representable in float32. Either the "
        "binding narrowed to f32 and widened back — which no value check against a "
        "6-decimal oracle would ever catch — or the charges are not real AM1-BCC output."
    )
    # Not a threshold with slack: real BCC charges are dense in the f64 mantissa, so the
    # overwhelming majority must fail to be f32-representable. Half would already mean
    # something is quantising them.
    assert survivors > total // 2, (
        f"only {survivors}/{total} charges carry more than 24 bits of mantissa; a "
        "population of real f64 charges is dense there. Something is quantising them."
    )


# --------------------------------------------------------------------------- #
# 3 & 4 — conservation, and the residual that must NOT be tidied away          #
# --------------------------------------------------------------------------- #


def test_charge_conservation_to_1e_12() -> None:
    """Total charge is conserved from the raw AM1 input to the BCC output, to 1e-12.

    Two stages act, and neither may move the sum: the equivalence class-mean conserves
    it in exact arithmetic (and to ULP in f64), and the BCC increments are pairwise
    antisymmetric, so they conserve it exactly.

    1e-12 is the number that matters. An ``f32`` round-trip would miss it by about 1e-8
    — four orders of magnitude — while passing any 1e-4 check against the oracle.
    """
    failures = []
    for case in CASES:
        q = _bcc(case)
        got = float(q.sum())
        want = float(np.asarray(case.am1_charges_raw, dtype=np.float64).sum())
        if abs(got - want) > CONSERVATION_TOL:
            failures.append(
                f"  {case.name}: Σq_in = {want!r}, Σq_out = {got!r}, drift {abs(got - want):.3e}"
            )
    assert not failures, "charge is not conserved across the binding:\n" + "\n".join(failures)


def test_the_binding_does_not_renormalize() -> None:
    """The AM1 rounding residual is CARRIED, never scrubbed onto the integer net charge.

    ``am1bcc.c`` ends at the increment loop; antechamber ships the residual, so matching
    antechamber means shipping it too. A binding that "helpfully" rescaled the charges to
    sum to exactly the formal net charge would produce a tidier number that is no longer
    the oracle's answer — and it would do so invisibly, because the difference is ~1e-6.
    """
    scrubbed = []
    for case in CASES:
        q = _bcc(case)
        raw_sum = float(np.asarray(case.am1_charges_raw, dtype=np.float64).sum())
        if abs(raw_sum - case.net_charge) <= 1e-9:
            continue  # this molecule's input carries no residual — nothing to scrub
        if float(q.sum()) == float(case.net_charge):
            scrubbed.append(
                f"  {case.name}: Σq_in = {raw_sum:.9f} but Σq_out is EXACTLY {case.net_charge}"
            )
    assert not scrubbed, (
        "the binding renormalized the total charge onto the integer net charge:\n"
        + "\n".join(scrubbed)
    )


# --------------------------------------------------------------------------- #
# 5 — every BIT-level invariant the Rust suite asserts, asserted here too      #
# --------------------------------------------------------------------------- #


def test_equivalent_atoms_share_bits_not_just_values() -> None:
    """Symmetry-equivalent atoms carry the SAME BITS — a class mean is one number, shared.

    ``approx``-equal is not the assertion. A class mean is computed once and handed to
    every member, so the members are bit-identical; anything that recomputed it per atom
    (or round-tripped it through a narrower type on the way out) would give values that
    are close and not equal.

    Acetate is the witness the chain earned: its two carboxylate oxygens once differed
    by 0.2014 e, and every charge test passed anyway.
    """
    case = next(c for c in CASES if c.name == "acetate")
    q = _bcc(case)
    oxygens = [i for i, el in enumerate(case.elements) if el == "O"]
    assert len(oxygens) == 2

    a, b = q[oxygens[0]], q[oxygens[1]]
    assert a.tobytes() == b.tobytes(), (
        f"acetate's two carboxylate oxygens are equivalent by symmetry and must carry the "
        f"same BITS. Got {a!r} and {b!r} (hex {float(a).hex()} vs {float(b).hex()}). "
        "They once differed by 0.2014 e."
    )


def test_charges_do_not_move_a_bit_when_the_conformation_changes() -> None:
    """A conformational change moves no bit of any charge. That is what equivalencing is for.

    BCC reads topology — bond types, atom types, equivalence classes. If a rotamer gives
    a different answer, something on the path is reading coordinates it has no business
    reading, and the force field a user ships depends on which conformer they loaded.
    """
    theta = math.radians(40.0)
    c, s = math.cos(theta), math.sin(theta)

    for case in CASES:
        mol, handles = build(case)
        moved, moved_handles = build(case)
        # Bend the molecule: every atom past the second is rotated about x. Not a rigid
        # motion — a genuine change of internal geometry, which a geometric model could
        # not be invariant under.
        for i, h in enumerate(moved_handles):
            if i < 2:
                continue
            _, y, z = case.xyz[i]
            moved.set(h, "y", y * c - z * s)
            moved.set(h, "z", y * s + z * c)

        qm = np.asarray(case.am1_charges_raw, dtype=np.float64)
        model = molrs.BccModel(parameter_set="bcc")
        q0 = model.assign(mol, qm)
        q1 = model.assign(moved, qm)

        assert q0.tobytes() == q1.tobytes(), (
            f"{case.name}: the same molecule in a different conformation got different "
            "charges — not different by a tolerance, different in the bits."
        )


def _mmff_energy(typifier: molrs.MMFF94Typifier | molrs.MMFF94STypifier, name: str) -> float:
    """One MMFF fixture through the Python chain: typify -> Frame -> pairs -> potentials."""
    mol, coords = _load_sdf(name)
    typed = typifier.typify(mol)
    frame = typed.to_frame()
    frame["pairs"] = molrs.intramolecular_pairs(frame)
    pots = typifier.forcefield().to_potentials(frame)
    energy, _ = pots.calc_energy_forces(coords)
    return float(energy)


def _load_sdf(name: str) -> tuple[molrs.Atomistic, np.ndarray]:
    """Minimal V2000 SDF reader — the same fixtures the Rust suite validates on."""
    lines = (MMFF_FIXTURES / f"{name}.sdf").read_text().splitlines()
    n_atoms = int(lines[3][0:3])
    n_bonds = int(lines[3][3:6])

    mol = molrs.Atomistic()
    handles = []
    coords = []
    for k in range(n_atoms):
        line = lines[4 + k]
        x, y, z = float(line[0:10]), float(line[10:20]), float(line[20:30])
        coords.extend((x, y, z))
        handles.append(mol.add_atom(line[31:34].strip(), x, y, z))
    for k in range(n_bonds):
        line = lines[4 + n_atoms + k]
        i, j = int(line[0:3]) - 1, int(line[3:6]) - 1
        bid = mol.add_bond(handles[i], handles[j])
        mol.set_relation_prop("bonds", bid, molrs.keys.ORDER, float(line[6:9]))
    return mol, np.asarray(coords, dtype=np.float64)


def _mmff_fixture_names() -> list[str]:
    """Scanned from disk. A list you can write by hand is a list you can shorten by hand."""
    names = sorted(p.name[: -len(".energy.json")] for p in MMFF_FIXTURES.glob("*.energy.json"))
    assert len(names) == 11, f"the RDKit MMFF oracle is 11 molecules, found {len(names)}"
    return names


@pytest.mark.parametrize("name", _mmff_fixture_names())
def test_mmff94_and_mmff94s_are_bit_identical_without_a_delocalized_nitrogen(name: str) -> None:
    """The Rust suite's bit-identity invariant, reproduced through the bindings.

    On a molecule with no type-10 / type-40 nitrogen, MMFF94 and MMFF94s ARE the same
    force field, so the two energies must be the same BITS. If the binding narrowed
    anything anywhere, two computations that agree bit for bit in Rust would still agree
    here — but they would agree on the *wrong* bits, and the conservation and f32 gates
    above are what catch that. This one catches the variant leaking through the binding.

    The partition is COMPUTED from the typed molecule, never listed.
    """
    mol, _ = _load_sdf(name)
    typed = molrs.MMFF94Typifier().typify(mol)
    frame = typed.to_frame()
    types = [int(t) for t in frame["atoms"].view(molrs.keys.TYPE)]
    delocalized_n = any(t in (10, 40) for t in types)

    e94 = _mmff_energy(molrs.MMFF94Typifier(), name)
    e94s = _mmff_energy(molrs.MMFF94STypifier(), name)

    if delocalized_n:
        assert e94 != e94s, (
            f"{name} HAS a delocalized nitrogen — the one place MMFF94s changes a "
            "parameter — and the two variants returned the identical energy through the "
            "binding. MMFF94s is not reaching the potentials."
        )
    else:
        assert e94.hex() == e94s.hex(), (
            f"{name} has no delocalized nitrogen, so MMFF94 and MMFF94s are the SAME "
            f"force field: the energies must be the same bits. Got {e94.hex()} vs "
            f"{e94s.hex()}."
        )


@pytest.mark.parametrize("name", _mmff_fixture_names())
def test_the_python_mmff_energy_matches_the_rdkit_oracle(name: str) -> None:
    """The Python chain, end to end, against RDKit — the same 11 molecules, the same 1e-3.

    The external oracle, reached through the bindings rather than through Rust. If the
    boundary transposed an array or dropped a column, the energy is where it shows.
    """
    import json

    want = json.loads((MMFF_FIXTURES / f"{name}.energy.json").read_text())["mmff94_total_energy"]
    got = _mmff_energy(molrs.MMFF94Typifier(), name)
    assert got == pytest.approx(want, abs=1e-3), f"{name}: molrs {got}, RDKit {want}"


def test_the_chain_matches_antechamber_through_the_binding() -> None:
    """The whole charge chain, through Python, against antechamber — all 37, no subset."""
    from antechamber_oracle import CHARGE_TOL, report

    failures = []
    for case in CASES:
        q = _bcc(case)
        want = np.asarray(case.bcc_charges, dtype=np.float64)
        worst = float(np.max(np.abs(q - want)))
        if worst > CHARGE_TOL:
            failures.append(f"{case.name}: worst |Δq| = {worst:.6f}")
    report("BCC charges through the Python binding", failures, len(CASES))
