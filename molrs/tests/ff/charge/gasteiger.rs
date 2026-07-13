//! Gasteiger/PEOE against `antechamber -c gas` — the ZERO-QM corner of the 2×2.
//!
//! | model | QM input? | topology correction? | `needs_equivalencing` |
//! |---|---|---|---|
//! | Mulliken | yes | no | false |
//! | BCC / ABCG2 | yes | yes (bond increments) | true |
//! | **Gasteiger** | **no** | yes (iterative EE) | **false** |
//!
//! Every other model in this directory is handed AM1 charges someone else computed.
//! This one is handed a molecule and nothing else — `assign(&mol, None)` — and must
//! still land on an antechamber column. That is what says the `ChargeModel` trait did
//! not quietly assume "QM base charges plus a correction": if it had, this file could
//! not exist.
//!
//! # What the model is
//!
//! `chi_i = a + b*q_i + c*q_i^2`, and for each bond the charge flows from the lower-χ
//! atom (which goes positive) to the higher-χ one, normalised by the DONOR's χ⁺ and
//! damped by `(1/2)^(n+1)`:
//!
//! ```text
//! q = (chi_high - chi_low) / chi_plus[donor] * 0.5^(iter+1)
//! ```
//!
//! It is **Jacobi**: the whole χ array is built from the previous iterate before any
//! transfer is applied. And it is a **convergence loop** — antechamber's `CONVERG` is
//! 1e-5, `GASMAXITER` 500, `DAMPFACTOR` 0.5 — not the textbook's fixed six passes.
//! Measured on the reference trajectory, ALL 37 oracle molecules need more than six
//! (methane 7, methylammonium 15), which is why [`the_damped_loop_runs_past_six_sweeps`]
//! is not an exotic corner but the common case.
//!
//! # What it is not
//!
//! The `d` column of `GASPARM.DAT` is χ⁺, the **divisor**. It is not a quartic
//! coefficient, and there is no `q^3` term anywhere in the model. That claim is
//! pinned on the DATA in `params::gasteiger_columns_keep_their_semantics` (H's χ⁺ is
//! 20.02, while its `a+b+c` is 12.85 — the two differ, and only χ⁺ is right) and on
//! the SOURCE in `gasteiger_source::the_chi_plus_column_is_never_a_cubic_term`.

use molrs::ff::charge::{ChargeModel, GasteigerModel};
use molrs::ff::params::GASTEIGER_PARAMS;

use crate::typifier::antechamber_oracle::{AntechamberCase, CASES};
use crate::typifier::oracle_mol::{build_case, report};

/// The spec's gate. antechamber's mol2 carries 6 decimals; the reference trajectory
/// agrees with it to 5e-7, so anything above this is a real disagreement.
const CHARGE_TOL: f64 = 1.0e-4;

/// A change larger than the last bits of an f64 sum.
///
/// The same constant `antechamber::bcc_conserves_total_charge_through_the_corrections`
/// uses, and for the same reason — see [`peoe_conserves_the_seeded_total`].
const ULP_SCALE: f64 = 1.0e-14;

/// Sum in atom order — the order the model produced the charges in.
///
/// f64 addition is not associative, so "the total charge" is only a number once you
/// say how it was added up. Both sides of every comparison below use this function.
fn total(charges: &[f64]) -> f64 {
    charges.iter().sum()
}

/// The seed charge q⁰ of a GAS atom type: `GASPARM.DAT`'s `formal_charge` column.
///
/// Not a lookup key and not the atom's Lewis formal charge — the initial charge the
/// PEOE iteration starts from. Most rows are 0.00; the ionic types carry one
/// (`o-2` = −0.50, `n4` = +1.00, `cg` = +0.04).
fn seed_charge(atom_type: &str) -> f64 {
    GASTEIGER_PARAMS
        .iter()
        .find(|row| row.atom_type == atom_type)
        .unwrap_or_else(|| panic!("no GASPARM row for the GAS type `{atom_type}`"))
        .seed_charge
}

/// What the molecule's charges must sum to: the total of its seeds.
///
/// Read off the ORACLE's `gas_atom_types` column, i.e. antechamber's own typing, so
/// that this expectation is independent of molrs's typifier. A molrs that mis-typed
/// an atom would move the seeded total and fail here even though it conserved its own
/// (wrong) seeds perfectly.
fn seeded_total(case: &AntechamberCase) -> f64 {
    case.gas_atom_types
        .iter()
        .map(|t| seed_charge(t))
        .sum::<f64>()
}

// ── ac-001: 37/37 vs `antechamber -c gas` ────────────────────────────────────

/// Gasteiger end-to-end: 37/37 against `antechamber -c gas`, from the molecule alone.
///
/// `assign(&mol, None)` — no QM charges, because this model needs none. The oracle
/// column was produced by an antechamber run that never called sqm (the generator
/// asserts no `sqm.out` was written), so there is no AM1 base charge hiding in it:
/// every number here comes from `GASPARM.DAT`, `ATOMTYPE_GAS.DEF` and the bond graph.
#[test]
fn gasteiger_charges_match_antechamber_end_to_end() {
    let mut failures = Vec::new();
    for case in CASES {
        let (mol, _) = build_case(case);
        let got = match GasteigerModel.assign(&mol, None) {
            Ok(q) => q,
            Err(e) => {
                failures.push(format!("  {:22} ERROR {e}", case.name));
                continue;
            }
        };
        assert_eq!(
            got.len(),
            case.gas_charges.len(),
            "{}: one charge per atom",
            case.name
        );

        let mut worst = 0.0f64;
        let mut wrong = Vec::new();
        for (k, (q, want)) in got.iter().zip(case.gas_charges).enumerate() {
            let d = (q - want).abs();
            worst = worst.max(d);
            if d > CHARGE_TOL {
                wrong.push(format!("{k}{}: {q:+.6} vs {want:+.6}", case.elements[k]));
            }
        }
        if !wrong.is_empty() {
            failures.push(format!(
                "  {:22} max|dq|={worst:.6}  {}",
                case.name,
                wrong.join(", ")
            ));
        }
    }
    report("Gasteiger (-c gas) end-to-end", &failures, CASES.len());
}

/// Methanol, atom by atom, against the values read off the live binary.
///
/// The spec's reference row, spelled out where a reader can see it:
/// `0.031933, -0.399641, 0.052691, 0.052691, 0.052691, 0.209634`.
/// A 37-molecule loop that reports "max|dq|" is a good gate and a poor document; this
/// is the one case anybody debugging the model will want to read.
#[test]
fn methanol_reproduces_the_reference_row() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methanol")
        .expect("methanol in the oracle");
    let want = [0.031933, -0.399641, 0.052691, 0.052691, 0.052691, 0.209634];
    assert_eq!(
        case.gas_charges, want,
        "the oracle's methanol `-c gas` row moved; the spec's reference is stale"
    );

    let (mol, _) = build_case(case);
    let got = GasteigerModel
        .assign(&mol, None)
        .expect("gasteiger methanol");
    for (k, (q, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (q - w).abs() < CHARGE_TOL,
            "methanol atom {k}{}: {q:+.6} vs antechamber {w:+.6}",
            case.elements[k]
        );
    }
}

// ── ac-004 (behavioural half): the model is symmetric WITHOUT equivalencing ──

/// The three methyl hydrogens come out IDENTICAL — same bits, no averaging.
///
/// This is the evidence for `needs_equivalencing() == false` being *correct* rather
/// than merely *declared*. `-eq 1` exists to remove a conformer artefact of a QM
/// calculation: sqm splits these same three hydrogens 0.053 / 0.098 / 0.053 purely
/// because one of them eclipses the O–H (see
/// `model::mulliken_does_not_average_equivalent_atoms`). A topology-only model cannot
/// produce that artefact — it never sees the conformer — so it is symmetric for free,
/// and antechamber agrees: it prints 0.052691 three times.
///
/// Asserted BITWISE, because "within 1e-10" would let a model that had quietly
/// averaged the three pass. Each methyl H has exactly one bond, so it receives exactly
/// one transfer per sweep from identical inputs: identical f64 operations, identical
/// bits. Anything else means the three are not being treated as the same atom.
#[test]
fn the_methyl_hydrogens_are_identical_without_any_averaging() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methanol")
        .expect("methanol in the oracle");
    assert!(
        case.gas_charges[2] == case.gas_charges[3] && case.gas_charges[3] == case.gas_charges[4],
        "antechamber's own `-c gas` methyl H are not equal ({:?}); the premise of this \
         test — that a topology-only model is inherently symmetric — is gone",
        &case.gas_charges[2..5]
    );

    let (mol, _) = build_case(case);
    let got = GasteigerModel
        .assign(&mol, None)
        .expect("gasteiger methanol");
    assert_eq!(
        got[2].to_bits(),
        got[3].to_bits(),
        "methyl H 2 and 3 differ ({} vs {}). A purely topological model has no \
         conformer to break the symmetry, so equivalent atoms must come out equal \
         with no equivalencing pass at all",
        got[2],
        got[3]
    );
    assert_eq!(
        got[3].to_bits(),
        got[4].to_bits(),
        "methyl H 3 and 4 differ ({} vs {})",
        got[3],
        got[4]
    );
}

/// Handed QM charges it did not ask for, the model returns the same bits as before.
///
/// `assign(&mol, qm)` takes `Option<&[f64]>` for the whole trait, so a caller holding
/// a `Box<dyn ChargeModel>` will pass whatever they have — and a Gasteiger that let a
/// stray AM1 vector leak into its answer would be a QM model wearing a topological
/// one's name. Both oracle QM columns are tried, and both must be ignored COMPLETELY:
/// same bits as `None`, not "same to 1e-10".
#[test]
fn gasteiger_ignores_any_qm_charges_it_is_handed() {
    for case in CASES {
        let (mol, _) = build_case(case);
        let base = GasteigerModel
            .assign(&mol, None)
            .unwrap_or_else(|e| panic!("{}: assign(mol, None): {e}", case.name));

        for (column, qm) in [
            ("am1_charges_raw", case.am1_charges_raw),
            ("am1_charges", case.am1_charges),
            // Not charges at all: if the model reads this argument, it cannot be
            // reading it "harmlessly".
            ("zeros", &vec![0.0; case.elements.len()][..]),
        ] {
            let got = GasteigerModel
                .assign(&mol, Some(qm))
                .unwrap_or_else(|e| panic!("{}/{column}: {e}", case.name));
            for (k, (q, b)) in got.iter().zip(&base).enumerate() {
                assert_eq!(
                    q.to_bits(),
                    b.to_bits(),
                    "{}/{column}: atom {k} changed ({q} vs {b}) when QM charges were \
                     supplied. Gasteiger takes NO QM input — the argument exists for \
                     the models that do, and this one must ignore it",
                    case.name
                );
            }
        }
    }
}

// ── ac-003: a damped CONVERGENCE loop, not six fixed sweeps ──────────────────

/// The charges after exactly six damped sweeps — methylammonium, the worst case.
///
/// Provenance: a reference trajectory of antechamber's own loop (Jacobi χ from the
/// previous iterate; per-bond transfer `(χ_hi − χ_lo)/χ⁺_donor · 0.5^(n+1)`; each
/// unordered bond once), seeded from `GASPARM.DAT` and typed with the oracle's
/// `gas_atom_types`. Run to convergence it reproduces all 37 antechamber columns to
/// 4.97e-07 — which is what makes its SIXTH-sweep snapshot trustworthy as the answer
/// the textbook's fixed-6-iteration Gasteiger would have given here.
///
/// It is used for exactly one thing: to show that antechamber's column is NOT
/// reachable in six sweeps. No test compares molrs to it.
const METHYLAMMONIUM_AFTER_SIX_SWEEPS: [f64; 8] = [
    -0.046577, 0.216302, 0.077620, 0.077620, 0.077620, 0.199138, 0.199138, 0.199138,
];

/// Six sweeps CANNOT produce the oracle column — so the loop has to keep going.
///
/// A pure statement about the reference data, in the spirit of
/// `antechamber::bcc_and_abcg2_are_not_the_same_column`: it names the thing the real
/// test below would otherwise be silently assuming. If the truncated and converged
/// answers were within the 1e-4 gate of each other, `the_damped_loop_runs_past_six_sweeps`
/// would be a second copy of ac-001 and a fixed-6-iteration model would sail through it.
///
/// They are not: the gap on methylammonium's nitrogen is 0.0131 e — 131× the gate.
#[test]
fn six_sweeps_cannot_reach_the_oracle_column() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methylammonium")
        .expect("methylammonium in the oracle");

    let gap = case
        .gas_charges
        .iter()
        .zip(METHYLAMMONIUM_AFTER_SIX_SWEEPS)
        .map(|(converged, truncated)| (converged - truncated).abs())
        .fold(0.0f64, f64::max);

    assert!(
        gap > 10.0 * CHARGE_TOL,
        "the 6-sweep truncation is only {gap:.6} away from antechamber's converged \
         column — within the 1e-4 gate, a model that stopped at six iterations would \
         pass ac-001, and the convergence test below would be testing nothing"
    );
}

/// The molecule that needs FIFTEEN sweeps still lands on antechamber.
///
/// Damping is `(1/2)^(n+1)`, so a sweep's transfers are half the previous sweep's and
/// the iteration is a geometric SERIES, not a fixed-point map: where you stop IS the
/// answer. antechamber stops on `rmsd <= 1e-5` (`CONVERG`), with a 500-sweep ceiling
/// (`GASMAXITER`) it never reaches — on the reference trajectory methylammonium takes
/// 15 sweeps to get there, and every one of the 37 takes more than six (methane, the
/// smallest, takes 7).
///
/// So a `for _ in 0..6` loop is not a smaller version of this model, it is a different
/// model — one that stops 0.0131 e short on this molecule
/// ([`six_sweeps_cannot_reach_the_oracle_column`]). Hard-coding six iterations is the
/// single likeliest way to write this file wrong, and it fails right here.
#[test]
fn the_damped_loop_runs_past_six_sweeps() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methylammonium")
        .expect("methylammonium in the oracle");

    let (mol, _) = build_case(case);
    let got = GasteigerModel
        .assign(&mol, None)
        .expect("gasteiger methylammonium");

    for (k, (q, want)) in got.iter().zip(case.gas_charges).enumerate() {
        assert!(
            (q - want).abs() < CHARGE_TOL,
            "methylammonium atom {k}{}: {q:+.6} vs antechamber {want:+.6}. The 6-sweep \
             truncation would be {:+.6} — if that is what this is, the loop is stopping \
             at a hard-coded iteration count instead of converging (CONVERG 1e-5, \
             GASMAXITER 500, DAMPFACTOR 0.5).",
            case.elements[k],
            METHYLAMMONIUM_AFTER_SIX_SWEEPS[k]
        );
    }
}

// ── ac-005: PEOE conserves charge, and conserves the SEEDED total ────────────

/// The seeds are the `formal_charge` column of `GASPARM.DAT`, and a charged species
/// starts from them.
///
/// Three witnesses, all in the oracle:
///
/// * **acetate** — two `o-2` rows, seed −0.50 each, so it starts at −1.00;
/// * **methylammonium** — one `n4` row, seed +1.00;
/// * **imidazolium** — every one of its GAS types (`c3`, `na`, `c2`, `h`) has seed
///   0.00, so it starts at ZERO despite being a +1 cation.
///
/// That last one is not a bug in molrs and must not be "fixed": `ATOMTYPE_GAS.DEF`
/// has no aromatic-N⁺ type to assign, so antechamber's own `-c gas` charges for
/// imidazolium sum to 0.0 and not to +1. It is the reason the conservation law below
/// is written against the SEEDED total and not against `net_charge` — a model that
/// normalized to the formal net charge would diverge from antechamber by a whole
/// electron here, and `-nc 1` would be the flag that made it wrong.
#[test]
fn the_seed_charges_come_from_the_gasparm_formal_charge_column() {
    let seeded = |name: &str| {
        seeded_total(
            CASES
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{name} in the oracle")),
        )
    };

    assert!(
        (seeded("acetate") - -1.0).abs() < 1e-12,
        "acetate's two `o-2` seeds (−0.50 each) must sum to −1.00"
    );
    assert!(
        (seeded("methylammonium") - 1.0).abs() < 1e-12,
        "methylammonium's `n4` seed must be +1.00"
    );
    assert!(
        seeded("imidazolium").abs() < 1e-12,
        "imidazolium's GAS types carry no seed at all — if this changed, the \
         conservation law below is asserting the wrong total"
    );

    // And antechamber agrees, in its own output column: the +1 cation comes back
    // neutral, because that is what the seeds say.
    let imidazolium = CASES
        .iter()
        .find(|c| c.name == "imidazolium")
        .expect("imidazolium in the oracle");
    assert_eq!(imidazolium.net_charge, 1);
    assert!(
        total(imidazolium.gas_charges).abs() < 1.0e-5,
        "antechamber's `-c gas` imidazolium sums to {:+.6}, not 0 — the claim that \
         `-c gas` ignores `-nc` and conserves only the SEEDED total is what needs \
         re-deriving, not this assertion relaxing",
        total(imidazolium.gas_charges)
    );
}

/// The per-bond transfer is antisymmetric, so PEOE cannot move the total.
///
/// `charge_i += q; charge_j -= q`, for every bond, on every sweep. In exact arithmetic
/// the sum of the transfers is identically zero and the molecule ends where its seeds
/// started it.
///
/// **Asserted at ULP scale, not bitwise — and that is not a weakened claim, it is the
/// only true one.** The acceptance contract's word was "bitwise", and it is
/// unattainable for the same reason it was in `chem-perceive-04`'s ac-005: each atom's
/// charge is a rounded f64 accumulated over its own bonds, and no rounding of the parts
/// reproduces the sum of the un-rounded whole bit-for-bit. Measured on the reference
/// trajectory: the worst residual over the 37 molecules is 2.96e-16 (dimethylformamide)
/// and only 4 of the 37 land bit-exactly on their seeded total. Demanding bits here
/// would be demanding that IEEE-754 addition be associative.
///
/// What the 1e-14 bound DOES exclude is any real leak — a transfer applied to only one
/// end of its bond, a dropped sweep, a rescale to the net charge — every one of which
/// is a 1e-3 effect or larger, twelve orders of magnitude above this line.
#[test]
fn peoe_conserves_the_seeded_total() {
    let mut worst = 0.0f64;
    let mut worst_name = "";
    for case in CASES {
        let (mol, _) = build_case(case);
        let q = GasteigerModel
            .assign(&mol, None)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));

        let before = seeded_total(case);
        let after = total(&q);
        let drift = (after - before).abs();
        if drift > worst {
            worst = drift;
            worst_name = case.name;
        }
        assert!(
            drift < ULP_SCALE,
            "{}: PEOE moved the total charge {before:+.17} -> {after:+.17} \
             (drift {drift:e}). Every transfer is added to one atom and subtracted \
             from the other, so a drift this size means one was applied single-ended \
             — or that the charges were renormalized to the net charge afterwards.",
            case.name
        );
    }
    assert!(
        worst < ULP_SCALE,
        "worst residual {worst:e} on {worst_name} — ULP scale, not a leak"
    );
}

/// The total is never "corrected" to the integer net charge.
///
/// Guard the guard. imidazolium is a +1 cation whose seeded total is 0, so a model
/// that renormalized its charges to `net_charge` would still conserve *something*
/// perfectly and sail through the test above — while sitting a full electron away from
/// antechamber. The no-fallback-values rule and antechamber parity say the same thing
/// here: carry the seeds, do not invent the total.
#[test]
fn peoe_does_not_renormalize_to_the_formal_net_charge() {
    let case = CASES
        .iter()
        .find(|c| c.name == "imidazolium")
        .expect("imidazolium in the oracle");

    let (mol, _) = build_case(case);
    let q = GasteigerModel
        .assign(&mol, None)
        .expect("gasteiger imidazolium");
    let sum = total(&q);

    assert!(
        (sum - f64::from(case.net_charge)).abs() > 0.9,
        "imidazolium's Gasteiger charges sum to {sum:+.6}, i.e. to its formal net \
         charge of {:+}. antechamber's do not — its GAS types carry no seed, so they \
         sum to 0.0 — which means a renormalizer has been added. Delete it: it makes \
         molrs diverge from the oracle by a whole electron.",
        case.net_charge
    );
    assert!(
        sum.abs() < ULP_SCALE,
        "imidazolium must sum to its seeded total of exactly 0, got {sum:+.17}"
    );
}
