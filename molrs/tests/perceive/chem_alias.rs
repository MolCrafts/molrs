//! Source gate: the old `chem` compat alias is gone, and nothing reaches for it.
//!
//! `chem-perceive-01` re-exported this layer under its former name so that the split
//! could land without breaking the binder workspaces, which import molrs by module
//! path and are **not** members of the main workspace — so `cargo check` never sees
//! them. `chem-perceive-13` is where that debt is paid: the binders migrate, and the
//! alias is deleted. ac-004 words the criterion as a grep: zero hits on the old path
//! across molrs, molrs-python, molrs-cxxapi, molrs-ffi and molrs-wasm, and the
//! re-export line removed from `molrs/src/lib.rs`.
//!
//! An alias is invisible to every behavioural test, because both names denote the
//! same module: only the source can say whether the second name still exists. Hence a
//! source gate, in the shape of the ones in `tests/ff/typifier/source_gate.rs`.
//!
//! # Two deliberate deviations from those gates
//!
//! 1. **It scans the sibling workspaces.** The typifier gates read `src/ff/…` of this
//!    crate. That scope would pass this criterion vacuously: the call sites that make
//!    the alias load-bearing live in `molrs-python/src/`, a separate workspace on a
//!    path dependency. A gate that only read this crate would go green while the
//!    binder still imported through the alias — and the binder's build would then
//!    break on the very merge that deletes it.
//!
//! 2. **It does not strip comments.** The typifier gates strip line comments so that
//!    prose mentioning a forbidden form does not trip them. Here the criterion *is* a
//!    raw grep, and prose is not innocent: the old path in rustdoc is either a stale
//!    instruction to the next reader or an intra-doc link that
//!    `RUSTDOCFLAGS='-D warnings' cargo doc` will fail on once the alias is gone.
//!
//! Neither forbidden string is spelled anywhere in this file — both are assembled at
//! run time — so the scanner cannot report itself. A false positive here would have
//! exactly one available fix (weaken the gate), which is how gates die.

use std::path::{Path, PathBuf};

/// The forbidden module path: the crate name, `::`, and the layer's former name.
fn chem_path() -> String {
    format!("molrs{}{}", "::", "chem")
}

/// The re-export line that publishes the former name from the crate root.
fn alias_line() -> String {
    format!("pub use crate::perceive as {};", "chem")
}

/// The repository root — the parent of this crate's directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the molrs crate lives one level under the repository root")
        .to_path_buf()
}

/// Every source tree that can name molrs by module path.
///
/// The three binder workspaces (`molrs-python`, `molrs-ffi`, `molrs-wasm`), the C API
/// and the CXX bridge, plus this crate's own `src` / `tests` / `examples` / `benches`
/// — the last two included because they compile under `cargo test --all-features` and
/// `cargo bench`, so a reference there is a build break waiting for the merge, not a
/// harmless leftover.
///
/// Trees that do not exist are skipped rather than asserted, so that removing a
/// binder does not fail this gate; [`scanned_sources`] separately requires the one
/// tree whose absence would make the gate meaningless.
fn scanned_roots() -> Vec<PathBuf> {
    let root = repo_root();
    [
        "molrs/src",
        "molrs/tests",
        "molrs/examples",
        "molrs/benches",
        "molrs-python/src",
        "molrs-ffi/src",
        "molrs-wasm/src",
        "molrs-capi/src",
        "molrs-cxxapi/src",
    ]
    .iter()
    .map(|rel| root.join(rel))
    .filter(|dir| dir.is_dir())
    .collect()
}

/// Every `.rs` under `dir`, as `(path, raw text)` — no comment stripping.
fn collect(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(dir).expect("read a scanned source tree") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).expect("read a scanned source file");
            out.push((path, src));
        }
    }
}

/// Every scanned `.rs`.
///
/// Panics when nothing was read, and when `molrs-python/src` — the workspace the
/// alias exists *for* — was not among the trees. A gate that reads nothing passes,
/// and this one has a specific nothing it must not read.
fn scanned_sources() -> Vec<(PathBuf, String)> {
    let roots = scanned_roots();
    let mut out = Vec::new();
    for dir in &roots {
        collect(dir, &mut out);
    }

    assert!(
        !out.is_empty(),
        "no .rs found under {roots:?} — the gate would pass vacuously"
    );
    let binder = repo_root().join("molrs-python/src");
    assert!(
        out.iter().any(|(path, _)| path.starts_with(&binder)),
        "molrs-python/src was not scanned — it is the workspace the compat alias \
         exists for, and a gate that skips it passes vacuously"
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// ac-004, first half — no source anywhere still names the old module path.
///
/// The binder is the migration this spec is named after
/// (`molrs-python/src/core/system/molgraph.rs`), but it is not the only caller: this
/// crate's own `conformer`, `ff/mmff` and `perceive/smarts` modules, its example and
/// its bench import through the alias too, and every one of them stops compiling the
/// moment `lib.rs` drops it. They are one migration, not five.
#[test]
fn no_source_names_the_chem_module_path() {
    let needle = chem_path();
    let hits: Vec<String> = scanned_sources()
        .iter()
        .flat_map(|(path, src)| {
            src.lines()
                .enumerate()
                .filter(|(_, line)| line.contains(&needle))
                .map(|(n, line)| format!("  {}:{}: {}", path.display(), n + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect();

    assert!(
        hits.is_empty(),
        "{} source line(s) still name the `{needle}` compat path; \
         migrate them to `molrs::perceive`:\n{}",
        hits.len(),
        hits.join("\n")
    );
}

/// ac-004, second half — the alias itself is gone from `lib.rs`.
///
/// Separate from the scan above so a failure says which half is open: every call site
/// can be migrated while the crate root still exports the second name (a dead alias,
/// which is precisely the debt this spec closes), whereas the reverse cannot compile.
#[test]
fn lib_rs_no_longer_exports_the_compat_alias() {
    let lib = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let src = std::fs::read_to_string(&lib).expect("read molrs/src/lib.rs");
    let alias = alias_line();

    let hit = src
        .lines()
        .enumerate()
        .find(|(_, line)| line.trim() == alias);

    assert!(
        hit.is_none(),
        "molrs/src/lib.rs:{} still exports the compat alias `{alias}` — introduced by \
         chem-perceive-01 to keep the separate molrs-python workspace building, and \
         removed by chem-perceive-13.",
        hit.map(|(n, _)| n + 1).unwrap_or(0)
    );
}

/// The module the alias pointed at is the real one, and it answers under its own
/// name.
///
/// Cheap, but it is what stops the two gates above from being satisfied by deleting
/// the perception layer instead of migrating the path to it.
#[test]
fn the_perceive_module_is_reachable_under_its_own_name() {
    let mol = molrs::Atomistic::new();
    let perceived = molrs::perceive::Perceive::new().find_rings(&mol);
    assert_eq!(perceived.n_atoms(), mol.n_atoms());
}
