//! Two structural gates on the Gasteiger model — ac-002 and ac-004.
//!
//! Both criteria are properties of what the source may CONTAIN, and a behavioural test
//! cannot see either of them:
//!
//! * **ac-002** — "no code path uses `d` as a `q^3` coefficient". A model that squared
//!   its way to the right answer through a wrong formula would still be wrong, but the
//!   `d`-as-quartic reading is such a specific and such a plausible mistake (the column
//!   sits fourth, after `a`/`b`/`c`) that the criterion words itself against the source.
//! * **ac-004** — "not special-cased anywhere in the ChargeModel plumbing". A `match`
//!   on the concrete model type is invisible from outside: the answers would come out
//!   right, and the trait's whole claim — that it hosts a zero-QM model without one —
//!   would be false anyway.
//!
//! The scanner is the one the typifier gates already use
//! (`typifier::source_gate::charge_sources`): line-based, line-comments stripped, so
//! the prose above cannot trip a gate on itself. Its bias is towards false negatives —
//! stripping text can only remove a match, never invent one.

use std::path::Path;

use crate::typifier::source_gate::{charge_sources, lines_containing_in};

/// The file the model is expected to live in (`molrs/src/ff/charge/gasteiger.rs`).
///
/// Everything else under `src/ff/charge/` is "the plumbing": the trait, the error, the
/// module root, and the other three models.
fn is_the_gasteiger_model(path: &Path) -> bool {
    path.file_name().and_then(|n| n.to_str()) == Some("gasteiger.rs")
}

/// Non-vacuity. Every gate below is a `grep` that must find nothing; a gate pointed at
/// a tree with no Gasteiger model in it finds nothing for the wrong reason.
#[test]
fn the_gate_can_see_the_gasteiger_model() {
    let sources = charge_sources();

    assert!(
        sources.iter().any(|(path, _)| is_the_gasteiger_model(path)),
        "no `gasteiger.rs` under src/ff/charge/ — the two gates below would pass \
         vacuously. The spec puts the model at `molrs/src/ff/charge/gasteiger.rs`; if \
         it has been put somewhere else, move this gate, do not delete it."
    );
    assert!(
        !lines_containing_in(&sources, "GasteigerModel").is_empty(),
        "the charge tree does not mention `GasteigerModel`"
    );
    assert!(
        !lines_containing_in(&sources, "chi_plus").is_empty(),
        "the charge tree never reads the `chi_plus` column, so it cannot be running \
         PEOE at all — the transfer is normalised by the DONOR's chi+"
    );
}

/// The `d` column is a DIVISOR. There is no cubic or quartic term in this model.
///
/// `GASPARM.DAT`'s five value columns are `a`, `b`, `c`, `d`, `formal_charge`, and the
/// obvious-and-wrong reading is `chi = a + b*q + c*q^2 + d*q^3`. It is wrong on the
/// data — for every heavy row `d == a+b+c` exactly (`c3`: 7.98+9.18+1.88 = 19.04 = d),
/// which is the signature of a cation electronegativity χ⁺ = χ(q=1), not of a quartic
/// coefficient — and it is wrong on hydrogen, whose χ⁺ is a fixed 20.02 eV where its
/// polynomial would give 12.85. `params::gasteiger_columns_keep_their_semantics` pins
/// both of those on the table. This pins that the CODE never uses it as a power series
/// term: no `q^3`, no `q^4`, anywhere in the charge tree, and χ⁺ appears under a
/// division sign.
#[test]
fn the_chi_plus_column_is_never_a_cubic_term() {
    let sources = charge_sources();

    // No cubic or quartic term, however it is spelled.
    let cubic: Vec<String> = ["powi(3", "powi(4", "powf(3", "powf(4", "q * q * q", "q*q*q"]
        .iter()
        .flat_map(|needle| lines_containing_in(&sources, needle))
        .collect();
    assert!(
        cubic.is_empty(),
        "a cubic/quartic term appears in the charge tree:\n{}\n\nThe electronegativity \
         polynomial is `chi = a + b*q + c*q^2` and stops there. The `d` column is chi+, \
         the normalisation DIVISOR — encoding it as `+ d*q^3` is the one catastrophic \
         misreading of GASPARM.DAT.",
        cubic.join("\n")
    );

    // And chi+ is used the way a divisor is used.
    let divided: Vec<String> = lines_containing_in(&sources, "chi_plus")
        .into_iter()
        .filter(|line| line.contains('/'))
        .collect();
    assert!(
        !divided.is_empty(),
        "`chi_plus` is read somewhere in the charge tree but never appears in a \
         division. The transfer is `(chi_hi - chi_lo) / chi_plus[donor] * damp` — if \
         chi+ is being multiplied into a polynomial instead, it is being used as a \
         coefficient, which is exactly what ac-002 forbids."
    );
}

/// Gasteiger is reached through the trait, never around it.
///
/// The 2×2's whole claim is that ONE trait carries a QM-free model beside two QM-based
/// ones with no branch on the concrete type. A `match` on the model, a `downcast_ref`,
/// an `if model_is_gasteiger` — any of them and the trait has stopped being the seam;
/// it has become a tag, and the "generality" is a `match` arm someone has to extend for
/// the next model.
///
/// So outside `gasteiger.rs` itself, the name `GasteigerModel` may appear only where a
/// module re-exports it. Registering the model, constructing it, listing it in a
/// `Vec<Box<dyn ChargeModel>>` — all of that is a caller's business (and the tests do
/// exactly that), not the plumbing's.
#[test]
fn the_charge_plumbing_does_not_special_case_gasteiger() {
    let plumbing: Vec<_> = charge_sources()
        .into_iter()
        .filter(|(path, _)| !is_the_gasteiger_model(path))
        .collect();

    let branches: Vec<String> = lines_containing_in(&plumbing, "GasteigerModel")
        .into_iter()
        .filter(|line| {
            // `file:line: <text>` — the gate is on the code, not the location.
            let text = line.splitn(3, ':').nth(2).unwrap_or_default().trim();
            !(text.starts_with("use ") || text.starts_with("pub use "))
        })
        .collect();
    assert!(
        branches.is_empty(),
        "the ChargeModel plumbing names `GasteigerModel` outside a re-export:\n{}\n\n\
         The model must be reached through `dyn ChargeModel` like every other one. If \
         the trait needs a branch to host a model that takes no QM input, then it HAS \
         assumed QM base charges and spec 07's abstraction does not hold.",
        branches.join("\n")
    );

    // The other way to special-case a trait object, and the harder one to spot.
    let downcasts: Vec<String> = ["downcast", "TypeId", "dyn Any", "is::<"]
        .iter()
        .flat_map(|needle| lines_containing_in(&charge_sources(), needle))
        .collect();
    assert!(
        downcasts.is_empty(),
        "the charge tree downcasts a `dyn ChargeModel`:\n{}\n\nA trait whose users have \
         to downcast to find out what they are holding is a special case wearing a \
         trait's clothes.",
        downcasts.join("\n")
    );
}
