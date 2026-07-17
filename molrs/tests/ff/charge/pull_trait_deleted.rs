//! ac-003 — the pull-trait backend seam is deleted, not deprecated.
//!
//! `AM1ChargeBackend` was a **pull** trait: `compute_am1_charges(&self, mol)`, as if
//! the implementor were going to compute AM1 from the molecule. Not one of the three
//! implementors in the entire workspace ever read `mol`:
//!
//! * `UnavailableAM1Backend` — always errors;
//! * `ProvidedAM1ChargeBackend` (`molrs-cxxapi`) — ignores `_mol`, returns a `Vec`
//!   Atomiverse computed;
//! * `FakeAM1Backend` (tests) — ignores `_mol`, returns a `Vec` the test wrote.
//!
//! Three constant-carriers wearing a compute trait. Production had to FAKE A BACKEND
//! to hand a `Vec<f64>` to the BCC table, which is the shape of a seam installed
//! backwards. The cure is a push API — `correct(&mol, &am1) -> Result<Vec<f64>>` —
//! and the criterion is that the old seam leaves no residue: not a deprecated alias,
//! not a `#[doc(hidden)]` re-export, nothing that a caller could still reach for.
//!
//! Scanned as source rather than exercised as behaviour because that is what the
//! criterion says (`grep -rnE ... molrs/src` -> 0 hits) and because there is no
//! behavioural test for the ABSENCE of an API: a test that named `AM1ChargeBackend`
//! to prove it was gone would not compile once it was.
//!
//! The other half of ac-003 — "a pure function taking a slice, with no backend trait
//! and no mutation of `mol`" — is enforced by the compiler, not by a scan:
//! `BccModel::correct(&self, mol: &Atomistic, am1: &[f64])` takes `mol` behind a
//! shared reference, so every test in this directory that calls it and then reads the
//! molecule back (`bcc::correct_leaves_gaff_atom_types_byte_identical`) is a
//! statically-guaranteed no-mutation proof.

use std::path::{Path, PathBuf};

/// The names the pull seam left behind. Every one must be gone from `molrs/src`.
///
/// The first three are ac-003's own list. The last two are the typifiers that only
/// existed to hold a backend (`AM1BCCTypifier<B>`, `AM1ChargeTypifier<B>`); the spec
/// deletes them in the same breath, and leaving them behind would leave the pull
/// seam behind with them — a generic over `B: AM1ChargeBackend` cannot survive the
/// trait's deletion anyway, so a hit here means the deletion was incomplete.
const DELETED: [&str; 5] = [
    "AM1ChargeBackend",
    "UnavailableAM1Backend",
    "normalize_total_charge",
    "AM1BCCTypifier",
    "AM1ChargeTypifier",
];

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` under `molrs/src`, verbatim.
///
/// Comments are NOT stripped, unlike the typifier gates: the criterion is a plain
/// `grep` over the source, and a doc-comment that still tells the reader to
/// implement `AM1ChargeBackend` is exactly as much of a live pull seam as the trait
/// would be. Prose that must discuss the deleted API belongs in the tests (here) or
/// in the spec, not in the source it was deleted from.
fn sources() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    collect(&src_dir(), &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(dir).expect("read molrs/src") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).expect("read a source file");
            out.push((path, src));
        }
    }
}

/// The scanner can see the source, or every gate here is vacuous.
#[test]
fn the_gate_can_read_the_source_tree() {
    let sources = sources();
    assert!(
        sources.len() > 50,
        "only {} .rs files found under {} — the gate is reading the wrong tree",
        sources.len(),
        src_dir().display()
    );
    let mentions_the_model = sources
        .iter()
        .any(|(_, src)| src.contains("BccModel") || src.contains("ChargeModel"));
    assert!(
        mentions_the_model,
        "the source tree mentions neither `BccModel` nor `ChargeModel` — the gate \
         below would report the pull trait as deleted from a tree that has no charge \
         models in it either"
    );
}

/// Not one hit, anywhere in `molrs/src`.
#[test]
fn the_pull_trait_leaves_no_residue_in_the_source() {
    let sources = sources();
    let mut hits = Vec::new();
    for name in DELETED {
        for (path, src) in &sources {
            for (n, line) in src.lines().enumerate() {
                if line.contains(name) {
                    hits.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "the pull-trait seam is still reachable from molrs/src:\n{}\n\nAll of \
         {DELETED:?} must be deleted — the AM1 charges are an ARGUMENT now \
         (`BccModel::correct(&mol, &am1)`), not something a backend is asked for.",
        hits.join("\n")
    );
}
