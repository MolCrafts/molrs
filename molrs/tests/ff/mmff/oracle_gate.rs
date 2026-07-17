//! ac-001 + ac-008 — the baseline sits BEHIND the deletion, and it does not move.
//!
//! # ac-001 — prove, then delete
//!
//! The parity assertions themselves already live in `energy.rs`:
//!
//!   * `generic_path_total_energy_matches_rdkit` — 11/11 fixtures, < 1e-3 kcal/mol
//!   * `generic_path_per_style_breakdown_matches_frozen` — all 7 styles, < 1e-6
//!   * `frozen_breakdown_sums_to_the_rdkit_total` — the frozen terms reconstruct
//!     the RDKit total, so they do not rest on molrs at all
//!
//! They pass **today**, on the commit that starts this spec, with the bespoke path
//! still in the tree. That is the point: the deletion changes the label source in
//! `frame_builder`, the call site in `conformer/etkdg`, and the line count of the
//! XML. Every one of those needs a known-green baseline *behind* it, not ahead of
//! it. This file adds nothing to those assertions — it stops them being quietly
//! rewritten.
//!
//! # ac-008 — the two ways to fake it
//!
//! After the deletion, the same 11 fixtures must still match, with **no assertion
//! value edited**. There are exactly two ways to cheat that, and this file closes
//! both:
//!
//! 1. **Loosen the tolerance.** 1e-3 becomes 1e-2 and caffeine's missing
//!    electrostatics fits comfortably underneath. → the tolerance constants
//!    declared in `energy.rs` are read out of its source and pinned by VALUE.
//!
//! 2. **Move the target.** Regenerate `<name>.breakdown.json` from whatever the
//!    code now produces, and every term matches by construction. → all 33 oracle
//!    files (11 × sdf + energy.json + breakdown.json) are pinned by SHA-256.
//!
//! Neither is caught by any energy test, because in both cases every energy test
//! goes green. That is precisely what makes them worth a gate.
//!
//! The tolerances are read as *numbers*, not as source text, so reformatting
//! `1.0e-3` to `1e-3` is not a failure — changing it to `1e-2` is.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn mmff_test_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/mmff")
}

fn fixtures_dir() -> PathBuf {
    mmff_test_dir().join("fixtures")
}

// ---------------------------------------------------------------------------
// ac-008 (1) — the oracle data cannot be regenerated to fit the code
// ---------------------------------------------------------------------------

/// Every fixture file pinned by `ORACLE.sha256` still hashes to what it did.
///
/// 33 files: for each of the 11 fixtures, the geometry (`.sdf`), the RDKit total
/// (`.energy.json`) and the frozen 7-term breakdown (`.breakdown.json`).
///
/// The breakdown files were dumped from the bespoke path, which this spec
/// deletes — and they stay legitimate afterwards only because each one
/// independently reconstructs the RDKit total (`energy.rs`'s
/// `frozen_breakdown_sums_to_the_rdkit_total`). That external self-validation is
/// worth nothing if the file itself can be rewritten to whatever makes the day's
/// code pass.
#[test]
fn the_rdkit_oracle_data_is_byte_for_byte_unchanged() {
    let manifest_path = mmff_test_dir().join("ORACLE.sha256");
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));

    let pinned: Vec<(&str, &str)> = manifest
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            (
                it.next().expect("manifest row has a hash"),
                it.next().expect("manifest row has a file name"),
            )
        })
        .collect();

    assert_eq!(
        pinned.len(),
        33,
        "ORACLE.sha256 pins {} files, want 33 (11 fixtures x sdf + energy.json + \
         breakdown.json). A manifest that pins less than the whole oracle leaves a hole exactly \
         where someone would put a regenerated reference.",
        pinned.len()
    );

    let mut changed = Vec::new();
    let mut missing = Vec::new();
    for (want, name) in pinned {
        let path = fixtures_dir().join(name);
        let Ok(bytes) = std::fs::read(&path) else {
            missing.push(name.to_owned());
            continue;
        };
        let got = format!("{:x}", Sha256::digest(&bytes));
        if got != want {
            changed.push(format!("{name}\n      pinned {want}\n      found  {got}"));
        }
    }

    assert!(
        missing.is_empty(),
        "oracle fixtures have been DELETED: {missing:?}. The 11 fixtures are the deletion's \
         entire proof of no-change; removing one is removing the proof."
    );
    assert!(
        changed.is_empty(),
        "the RDKit oracle data was EDITED:\n    {}\n\n\
         ac-008: the energies must be bit-for-bit unchanged across the deletion, and \"unchanged\" \
         is measured against these files. Rewriting the reference so that the code matches it is \
         not a passing test — it is a test that has stopped testing.\n\
         If the oracle genuinely needs regenerating, that is a different spec: re-derive from \
         RDKit and re-prove that the 7 frozen terms still sum to `mmff94_total_energy`.",
        changed.join("\n    ")
    );
}

// ---------------------------------------------------------------------------
// ac-008 (2) — the tolerances cannot be loosened
// ---------------------------------------------------------------------------

/// `const NAME: f64 = <literal>;` in `energy.rs`, parsed as a number.
fn tolerance_const(src: &str, name: &str) -> f64 {
    let marker = format!("const {name}: f64 =");
    let line = src
        .lines()
        .find(|l| l.trim_start().starts_with(&marker))
        .unwrap_or_else(|| {
            panic!(
                "`tests/ff/mmff/energy.rs` no longer declares `{marker} ...`. The parity \
                 tolerances are the assertion; deleting the constant deletes the assertion."
            )
        });
    let rhs = line
        .split('=')
        .nth(1)
        .expect("a const has a right-hand side")
        .trim()
        .trim_end_matches(';')
        .trim();
    rhs.parse::<f64>()
        .unwrap_or_else(|e| panic!("cannot parse `{name} = {rhs}` as f64: {e}"))
}

/// The two tolerances that define "the energies did not change".
///
/// Read by value, so formatting is free and meaning is not.
///
///   * `ENERGY_TOL` = 1e-3 kcal/mol — the generic path against RDKit. The spec's
///     parity tolerance, unchanged since mmff-orthogonal-01 measured 3.1e-9
///     against it.
///   * `BREAKDOWN_TOL` = 1e-6 kcal/mol — each of the 7 style energies against its
///     frozen term. This is the one with teeth: caffeine's total sat within
///     tolerance of RDKit while 150 kcal/mol of electrostatics did not exist,
///     because the terms partially cancel. A per-term tolerance is the only thing
///     that sees that.
///
/// `BESPOKE_TOL` is deliberately NOT pinned: it measures the bespoke path this
/// spec deletes, and pinning it would demand the constant outlive its subject.
#[test]
fn the_parity_tolerances_are_not_loosened() {
    let path = mmff_test_dir().join("energy.rs");
    let src = std::fs::read_to_string(&path).expect("read tests/ff/mmff/energy.rs");

    let energy_tol = tolerance_const(&src, "ENERGY_TOL");
    let breakdown_tol = tolerance_const(&src, "BREAKDOWN_TOL");

    assert_eq!(
        energy_tol, 1.0e-3,
        "ENERGY_TOL is {energy_tol:e}, want 1e-3 kcal/mol. Loosening the RDKit parity tolerance \
         to make the post-deletion suite green is a failure, not a pass — the generic path \
         measured 3.1e-9 against this tolerance BEFORE the deletion, so any change large enough \
         to need a wider tolerance is a change the deletion was not supposed to make."
    );
    assert_eq!(
        breakdown_tol, 1.0e-6,
        "BREAKDOWN_TOL is {breakdown_tol:e}, want 1e-6 kcal/mol. This is the per-STYLE \
         tolerance, and it is the one that has teeth: on caffeine the TOTAL energy hid a \
         150 kcal/mol electrostatic hole behind partially-cancelling terms. Widen this and the \
         breakdown stops being able to see the thing it was written to see."
    );
}

/// The 11 fixtures are all still there, and each still has all three oracle files.
///
/// `energy.rs` scans `*.energy.json` to build its fixture list — deliberately, so
/// that a new fixture cannot be forgotten. The flip side is that a *deleted*
/// fixture also cannot be noticed: the scan simply returns ten, every test
/// iterates ten, and everything is green. This is the assertion that notices.
#[test]
fn all_eleven_fixtures_still_carry_their_full_oracle() {
    const REQUIRED: [&str; 11] = [
        "e_acetonitrile",
        "e_benzene",
        "e_big",
        "e_butane",
        "e_caffeine",
        "e_ethane",
        "e_ethylene",
        "s_acetamide",
        "s_aniline",
        "s_nmethylacetamide",
        "s_urea",
    ];

    let mut missing = Vec::new();
    for name in REQUIRED {
        for suffix in ["sdf", "energy.json", "breakdown.json"] {
            let path = fixtures_dir().join(format!("{name}.{suffix}"));
            if !path.is_file() {
                missing.push(format!("{name}.{suffix}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "oracle files missing from {}: {missing:?}\n\
         The 11 fixtures ARE the proof that the deletion changed no number. Dropping one is not \
         a smaller test suite — it is a smaller proof.",
        fixtures_dir().display()
    );
}
