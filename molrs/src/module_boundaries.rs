//! Which modules may name which: the dependency edges the crate forbids,
//! checked over the source text.
//!
//! * `ff` reads no file: the force-field data model, kernels, typifiers and
//!   parameter tables never name [`crate::io`]. Every file reader and writer —
//!   force-field files included — is `io`'s, and `io` depends on `ff`, never
//!   the reverse.
//! * `perceive` and `io` are independent: perception never reads a file and
//!   a reader never perceives. SMILES (`io`) and SMARTS (`perceive`) share
//!   their grammar through the crate-private `line_notation`, so neither
//!   needs the other.
//!
//! Only test code may cross: a `#[cfg(test)]` module file (the engine checks
//! — `equivalence_check`, `engine_codec_check`, cmap `lammps_check`, …) and a
//! file's trailing `#[cfg(test)]` block may build fixtures with the other
//! module. Each check walks every source file under its module and fails on
//! any other code line that names the forbidden one; comments (doc links
//! included) are not code.

use std::path::{Path, PathBuf};

/// Every `.rs` file under `dir`.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// A `cfg` attribute that compiles its item only under test: `#[cfg(test)]`
/// or `#[cfg(all(test, …))]`.
fn is_test_cfg(line: &str) -> bool {
    line == "#[cfg(test)]" || line.starts_with("#[cfg(all(test,")
}

/// The files of modules declared test-only (`#[cfg(test)] mod x;` in
/// `parent.rs` / `parent/mod.rs` is `parent/x.rs` or `parent/x/…`).
fn test_only_modules(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(file).unwrap();
        let dir = if file.file_name().is_some_and(|n| n == "mod.rs") {
            file.parent().unwrap().to_path_buf()
        } else {
            file.with_extension("")
        };
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        for pair in lines.windows(2) {
            if !is_test_cfg(pair[0]) {
                continue;
            }
            let decl = pair[1]
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub(super) ");
            if let Some(name) = decl.strip_prefix("mod ").and_then(|d| d.strip_suffix(';')) {
                out.push(dir.join(format!("{name}.rs")));
                out.push(dir.join(name));
            }
        }
    }
    out
}

/// The non-test code lines under `src/<module>` that name `src/<forbidden>`
/// (`crate::<forbidden>` or `molrs::<forbidden>`), as `file:line: code`.
fn crossings(module: &str, forbidden: &str) -> Vec<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(module);
    let mut files = Vec::new();
    sources(&root, &mut files);
    let test_only = test_only_modules(&files);
    let paths = [format!("crate::{forbidden}"), format!("molrs::{forbidden}")];

    let mut out = Vec::new();
    for file in &files {
        if test_only.iter().any(|t| file.starts_with(t)) {
            continue;
        }
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.trim();
            if is_test_cfg(code) {
                break; // the trailing test block
            }
            if code.starts_with("//") {
                continue;
            }
            if paths.iter().any(|p| code.contains(p.as_str())) {
                out.push(format!(
                    "{module}/{}:{}: {code}",
                    file.strip_prefix(&root).unwrap().display(),
                    n + 1
                ));
            }
        }
    }
    out
}

fn assert_no_crossing(module: &str, forbidden: &str) {
    let found = crossings(module, forbidden);
    assert!(
        found.is_empty(),
        "{module} names {forbidden} outside test code:\n{}",
        found.join("\n")
    );
}

#[test]
fn ff_names_io_only_in_test_code() {
    assert_no_crossing("ff", "io");
}

#[test]
fn perceive_names_io_only_in_test_code() {
    assert_no_crossing("perceive", "io");
}

#[test]
fn io_names_perceive_only_in_test_code() {
    assert_no_crossing("io", "perceive");
}
