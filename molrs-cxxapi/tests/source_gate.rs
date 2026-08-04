//! Source-level gates over the bridge — chem-perceive-12 ac-003.
//!
//! The criterion is worded as a grep, because it is structural rather than
//! behavioural: `grep -rnE 'ProvidedAM1ChargeBackend|normalize_total_charge'
//! molrs-cxxapi/src` must return 0 hits. Neither is visible to a behavioural
//! test — a dead argument still lets every call succeed, and a fake charge
//! backend still returns the charges someone else computed. So the gate reads the
//! source, exactly as the contract words it, in the style of
//! `molrs/tests/ff/typifier/source_gate.rs`.
//!
//! ## Where the bridge's source actually is
//!
//! `src/bridge.rs` is **generated**: `build.rs` holds the schema as
//! `CXX_BRIDGE_SCHEMA` and rewrites `src/bridge.rs` from it on every build. A
//! gate that scanned only `src/` would therefore be reading a mirror, and one
//! that scanned only `build.rs` would miss a hand-written `normalize_total_charge`
//! in `lib.rs`. This one reads both, and reports which file it found a violation
//! in — the fix belongs in whichever it names, and for `bridge.rs` that means
//! `build.rs`.
//!
//! The scanner is deliberately crude (line-based, line-comments stripped). Its
//! bias is towards false negatives: stripping text can only remove a match, never
//! invent one, so it can miss a violation but cannot fail an innocent file. In
//! particular, the doc-comment in `lib.rs` that *explains* why the argument is
//! vestigial is not a violation, and stripping comments is what keeps it from
//! being counted as one.

use std::path::{Path, PathBuf};

/// The crate root — `molrs-cxxapi/`.
fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every Rust source the bridge is built from: `src/**/*.rs` (which includes the
/// generated `src/bridge.rs`) plus `build.rs` (which generates it).
///
/// Comments are stripped so that prose naming a forbidden symbol — including this
/// file's own reason for existing, were it ever moved into `src/` — does not trip
/// the gate.
fn bridge_sources() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    collect(&crate_dir().join("src"), &mut out);

    let build_rs = crate_dir().join("build.rs");
    let src = std::fs::read_to_string(&build_rs).expect("read build.rs");
    out.push((build_rs, strip_line_comments(&src)));

    assert!(
        out.len() > 1,
        "no .rs found under {} — the gate would pass vacuously",
        crate_dir().join("src").display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(dir).expect("read the bridge source tree") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).expect("read a bridge source file");
            out.push((path, strip_line_comments(&src)));
        }
    }
}

/// Drop everything from the first `//` of each line.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|line| match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `file:line: <text>` for every line of the bridge sources containing `needle`.
fn lines_containing(needle: &str) -> Vec<String> {
    bridge_sources()
        .iter()
        .flat_map(|(path, src)| {
            src.lines()
                .enumerate()
                .filter(|(_, line)| line.contains(needle))
                .map(|(n, line)| format!("{}:{}: {}", path.display(), n + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// ac-003 — the dead argument is gone.
///
/// `normalize_total_charge` was already a lie the impl had to refuse: passing
/// `true` returns an error, because the AM1-BCC reference algorithm does not renormalize (`am1bcc.c`
/// ends at the increment loop) and spreading the AM1 rounding residual over the
/// atoms would make molrs diverge from the reference while hiding a
/// non-converged AM1 behind a plausible answer. An argument whose only legal
/// value is `false` is not a knob; it is an ABI wart on the one signature this
/// repo shares with another.
#[test]
fn the_normalize_total_charge_argument_is_gone() {
    let hits = lines_containing("normalize_total_charge");
    assert!(
        hits.is_empty(),
        "chem-perceive-12 ac-003: `normalize_total_charge` still appears in the bridge \
         ({} hit(s)). Its only legal value is `false`; delete the argument.\n{}\n\nNOTE: \
         src/bridge.rs is GENERATED — the declaration lives in build.rs's CXX_BRIDGE_SCHEMA, \
         and editing src/bridge.rs is clobbered on the next build.",
        hits.len(),
        hits.join("\n")
    );
}

/// ac-003 — the fake AM1 backend is gone.
///
/// `ProvidedAM1ChargeBackend` dressed a `Vec<f64>` up as a solver so that a
/// molecule-in/charges-out trait could be handed charges it had not computed. The
/// trait died with chem-perceive-07 (charges are now an argument:
/// `BccModel::correct(&mol, &am1)`), and the costume dies with it.
///
/// The gate stays even though the symbol is already absent: it is one of the two
/// halves of the criterion, and a gate written only for the half that is still
/// broken would retire silently the moment someone re-introduced the other.
#[test]
fn the_fake_am1_charge_backend_is_gone() {
    let hits = lines_containing("ProvidedAM1ChargeBackend");
    assert!(
        hits.is_empty(),
        "chem-perceive-12 ac-003: `ProvidedAM1ChargeBackend` is back in the bridge ({} hit(s)). \
         AM1 charges are pushed in as a `&[f64]` argument; nothing has to impersonate a \
         solver.\n{}",
        hits.len(),
        hits.join("\n")
    );
}

/// ac-003, the tail the criterion does not name — `total_charge` is unused.
///
/// The impl binds it as `_total_charge` and never reads it: it is dead in exactly
/// the same way `normalize_total_charge` is, and it is dead for the same reason
/// (molrs carries the reference rounding residual through rather than
/// renormalizing to a target). The spec leaves its fate to the implementer, so
/// this test does not demand its removal — it demands that if it is KEPT, it is
/// kept as something the code actually reads. An argument that crosses an FFI
/// boundary into an underscore is a promise to Atomiverse that molrs does not
/// keep.
///
/// Delete the argument (and this test goes green), or use it (and it goes green).
/// Only `_total_charge` fails.
#[test]
fn total_charge_is_either_used_or_gone() {
    let hits = lines_containing("_total_charge");
    let dead: Vec<&String> = hits
        .iter()
        .filter(|line| !line.contains("normalize_total_charge"))
        .collect();

    assert!(
        dead.is_empty(),
        "`total_charge` still crosses the bridge but is bound as `_total_charge` and never \
         read ({} hit(s)). Either drop it — nothing in the AM1-BCC stage renormalizes to a \
         target — or read it.\n{}",
        dead.len(),
        dead.iter()
            .map(|line| line.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}
