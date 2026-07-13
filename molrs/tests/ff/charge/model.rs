//! The `ChargeModel` trait — the 2×2 generality proof.
//!
//! One trait must hold three models without any of them being a special case:
//!
//! | model            | needs QM input? | topology correction? | `needs_equivalencing` |
//! |------------------|-----------------|----------------------|-----------------------|
//! | Mulliken         | yes             | no (pass-through)    | false (`-eq 0`)       |
//! | AM1-BCC          | yes             | yes (bond increments)| true  (`-eq 1`)       |
//! | ABCG2            | yes             | yes (bond increments)| true  (`-eq 1`)       |
//! | Gasteiger (08)   | **no**          | yes (iterative)      | false                 |
//!
//! If one trait carries all of them with no branch on the concrete type, it has not
//! secretly assumed "QM base charges + a correction". That is the whole claim, and
//! the test that makes it is [`every_model_lands_on_its_own_oracle_column`]: three
//! models behind ONE `&dyn ChargeModel`, three oracle columns, one loop.
//!
//! `needs_equivalencing` is a DECLARATION THE MODEL HONOURS, not advice it leaves
//! to the caller. molrs already committed to that reading in
//! `perceive::Perceive::find_equivalence_classes`, whose doc says the class-mean is
//! "an explicit step **the charge model calls**, because whether to average is a
//! property of the charge model and not of the graph". So `assign` is the full
//! model — raw QM charges in, final charges out, equivalencing applied iff the
//! model declares it needs it — while `BccModel::correct` is the pure BCC stage
//! alone, taking base charges that are already equivalenced. Both are tested; the
//! oracle carries both columns (`am1_charges_raw` and `am1_charges`) precisely so
//! that it can tell them apart.

use molrs::ff::charge::{BccModel, BccParameterSet, ChargeModel, MullikenModel};

use crate::typifier::antechamber_oracle::{AntechamberCase, CASES};
use crate::typifier::oracle_mol::{build_case, report};

/// antechamber writes charges with 4 significant decimals and the BCC increments
/// carry 4. Above 1e-4 is a real disagreement, not formatting.
const CHARGE_TOL: f64 = 1.0e-4;

/// One model under test, with the oracle column it must land on when it is handed
/// the RAW (un-equivalenced) sqm Mulliken charges.
struct ModelCase {
    name: &'static str,
    model: Box<dyn ChargeModel>,
    needs_equivalencing: bool,
    /// The final charges this model must produce from `am1_charges_raw`.
    oracle: fn(&AntechamberCase) -> &'static [f64],
}

/// Every charge model that exists today, behind the one trait.
///
/// They are `Box<dyn ChargeModel>` on purpose: a trait that cannot be made into an
/// object cannot carry a model the caller chose at runtime (which is exactly what
/// the C++ and Python bridges must do), and a trait whose users have to downcast to
/// find out what they are holding is a special case wearing a trait's clothes.
fn models() -> Vec<ModelCase> {
    vec![
        ModelCase {
            name: "AM1-BCC",
            model: Box::new(BccModel::new(BccParameterSet::Bcc).expect("build the BCC model")),
            needs_equivalencing: true,
            oracle: |c| c.bcc_charges,
        },
        ModelCase {
            name: "ABCG2",
            model: Box::new(BccModel::new(BccParameterSet::Abcg2).expect("build the ABCG2 model")),
            needs_equivalencing: true,
            oracle: |c| c.abcg2_charges,
        },
        ModelCase {
            name: "Mulliken",
            model: Box::new(MullikenModel),
            needs_equivalencing: false,
            // No equivalencing, no correction: what went in comes out.
            oracle: |c| c.am1_charges_raw,
        },
    ]
}

// ── the 2×2, declared ────────────────────────────────────────────────────────

/// The two BCC families need their AM1 charges equivalenced; Mulliken does not.
///
/// This is antechamber's own default, per method: `-eq 1` for `bcc` / `abcg2` /
/// `resp`, `-eq 0` for everything else. It is a property of the MODEL, which is why
/// it is a method on the model's trait and not a flag on the graph or an argument
/// the caller has to remember.
#[test]
fn needs_equivalencing_is_declared_per_model() {
    for case in models() {
        assert_eq!(
            case.model.needs_equivalencing(),
            case.needs_equivalencing,
            "{}: needs_equivalencing() disagrees with antechamber's `-eq` default \
             for this charge method",
            case.name
        );
    }
}

// ── the 2×2, proven ──────────────────────────────────────────────────────────

/// Three models, one trait object, three oracle columns — and no special case.
///
/// Each model is handed the SAME input (the raw sqm Mulliken charges, i.e. what an
/// AM1 backend really hands molrs) through the SAME method, and each must land on
/// its own antechamber column:
///
/// * AM1-BCC  -> `bcc_charges`    (equivalence, then BCCPARM.DAT)
/// * ABCG2    -> `abcg2_charges`  (equivalence, then BCCPARM_ABCG2.DAT)
/// * Mulliken -> `am1_charges_raw` (nothing at all)
///
/// This is the load-bearing test of the whole trait. It fails if `assign` ignores
/// `needs_equivalencing` (the 20 molecules antechamber averages would come out with
/// their sqm conformer artefacts baked in — methanol's methyl H would stay split
/// 0.053/0.098/0.053, a ~0.02 e error, two hundred times the tolerance); it fails
/// if the parameter set is ignored (the two families disagree on 33 of the 37
/// molecules, by up to 0.35 e — see
/// `antechamber::bcc_and_abcg2_are_not_the_same_column`); and it fails if Mulliken
/// is quietly given a correction.
#[test]
fn every_model_lands_on_its_own_oracle_column() {
    for ModelCase {
        name,
        model,
        oracle,
        ..
    } in models()
    {
        let mut failures = Vec::new();
        for case in CASES {
            let (mol, _) = build_case(case);
            let got = match model.assign(&mol, Some(case.am1_charges_raw)) {
                Ok(q) => q,
                Err(e) => {
                    failures.push(format!("  {:22} ERROR {e}", case.name));
                    continue;
                }
            };
            let want = oracle(case);
            assert_eq!(
                got.len(),
                want.len(),
                "{name}/{}: one charge per atom",
                case.name
            );
            let mut worst = 0.0f64;
            let mut wrong = Vec::new();
            for (k, (q, w)) in got.iter().zip(want).enumerate() {
                let d = (q - w).abs();
                worst = worst.max(d);
                if d > CHARGE_TOL {
                    wrong.push(format!("{k}{}: {q:+.4} vs {w:+.4}", case.elements[k]));
                }
            }
            if !wrong.is_empty() {
                failures.push(format!(
                    "  {:22} max|dq|={worst:.4}  {}",
                    case.name,
                    wrong.join(", ")
                ));
            }
        }
        report(
            &format!("{name} through ChargeModel::assign"),
            &failures,
            CASES.len(),
        );
    }
}

// ── Mulliken: the model with no correction at all ────────────────────────────

/// Mulliken hands back exactly the charges it was given — the same BITS.
///
/// "Within 1e-10" would not be the same claim. A pass-through model that returned
/// `q * 1.0000000001`, or that rounded, or that renormalized the total to the net
/// charge, would satisfy a tolerance and still be wrong: it would be *doing
/// something*, and the entire content of this model is that it does nothing. Bits
/// are the only honest assertion here.
///
/// It is checked against BOTH oracle charge columns because pass-through must not
/// care which one it is handed: the raw sqm Mulliken (`-eq 0`, this model's own
/// default) and the equivalenced charges (what the BCC stage consumes).
#[test]
fn mulliken_returns_the_supplied_qm_charges_bitwise() {
    let model = MullikenModel;
    for case in CASES {
        let (mol, _) = build_case(case);
        for (column, qm) in [
            ("am1_charges_raw", case.am1_charges_raw),
            ("am1_charges", case.am1_charges),
        ] {
            let got = model
                .assign(&mol, Some(qm))
                .unwrap_or_else(|e| panic!("{}/{column}: {e}", case.name));
            assert_eq!(got.len(), qm.len(), "{}/{column}", case.name);
            for (k, (q, w)) in got.iter().zip(qm).enumerate() {
                assert_eq!(
                    q.to_bits(),
                    w.to_bits(),
                    "{}/{column}: atom {k} came back changed ({q} vs {w}) — a \
                     pass-through model must not touch the charges",
                    case.name
                );
            }
        }
    }
}

/// Mulliken does not equivalence, and that is visible in the numbers.
///
/// Guard against the guard: if `MullikenModel` silently averaged over equivalence
/// classes, the bitwise test above would still pass on the 17 molecules that are
/// already symmetric out of `sqm`. Methanol is one of the 20 where it would not —
/// its three methyl hydrogens come out of `sqm` split 0.053 / 0.098 / 0.053 by a
/// pure conformer artefact, and `-eq 0` means they stay split.
#[test]
fn mulliken_does_not_average_equivalent_atoms() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methanol")
        .expect("methanol in the oracle");
    assert_eq!(
        case.am1_charges_raw[2..5],
        [0.053, 0.098, 0.053],
        "the oracle's raw methyl H moved; this test's premise is gone"
    );

    let (mol, _) = build_case(case);
    let got = MullikenModel
        .assign(&mol, Some(case.am1_charges_raw))
        .expect("mulliken pass-through");

    assert_ne!(
        got[2].to_bits(),
        got[3].to_bits(),
        "MullikenModel averaged two topologically-equivalent hydrogens, but it \
         declares needs_equivalencing() == false — the declaration and the \
         behaviour must be the same thing"
    );
}

// ── no model invents a charge it was not given ───────────────────────────────

/// A model that needs QM charges and is handed none must FAIL.
///
/// This is what `am1bcc_without_an_am1_backend_is_an_error` protected when the
/// missing input was spelled "no `AM1ChargeBackend` is configured". The input is now
/// an `Option<&[f64]>` argument rather than a trait one had to fake, but the
/// guarantee is the same one and it is the important one: absent AM1, the answer is
/// an error, never a plausible-looking charge.
///
/// (Gasteiger, in 08, is the model that answers `None` with real charges — it needs
/// no QM input at all. That it can join this trait without either of these two
/// changing is the point of the 2×2.)
#[test]
fn a_model_that_needs_qm_charges_refuses_to_invent_them() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methanol")
        .expect("methanol in the oracle");
    let (mol, _) = build_case(case);

    for ModelCase { name, model, .. } in models() {
        let err = model
            .assign(&mol, None)
            .err()
            .unwrap_or_else(|| panic!("{name}: assign(mol, None) invented charges out of nothing"));
        // The error must be about the missing input, not about a molecule that is
        // perfectly well-formed.
        let text = err.to_string();
        assert!(
            !text.contains("missing BCC correction"),
            "{name}: the molecule is correctable — the missing QM input is what \
             must be reported, got: {text}"
        );
    }
}
