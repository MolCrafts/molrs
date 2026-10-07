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
//! * `op` is the numeric base beneath `core`: it names no other molrs
//!   module. What acts on a core type (a `MolGraph`'s rigid-body transforms)
//!   is that type's, in `core`.
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

#[test]
fn op_names_no_other_module() {
    for other in [
        "core",
        "perceive",
        "io",
        "ff",
        "compute",
        "signal",
        "conformer",
        "md",
        "builder",
        "optimize",
        "stream",
        "line_notation",
    ] {
        assert_no_crossing("op", other);
    }
}

// ---------------------------------------------------------------------------
// Units: one definition per unit.
//
// Every unit conversion goes through `core::units` (`UnitFactor`,
// `UnitRegistry::factor`, `Quantity::to`), and `core::constants` holds
// physical and engine constants, never a conversion factor. These checks
// fail on a conversion-factor constant (defined or named) outside the units
// module, and on a known conversion factor written as a literal in non-test
// code.
// ---------------------------------------------------------------------------

/// Conversion-factor constants `core::constants` no longer defines.
const RETIRED_FACTOR_CONSTANTS: [&str; 9] = [
    "KJ_PER_KCAL",
    "ANGSTROM_PER_NM",
    "ANGSTROM_PER_BOHR",
    "ANGSTROM3_PER_CM3",
    "ANGSTROM_M",
    "FEMTOSECOND_S",
    "CENTIMETER_PER_METER",
    "OPENMM_COULOMB",
    "GROMACS_COULOMB",
];

/// Conversion factors as they would be written by hand: kcal ↔ kJ (and its
/// nm² / Å² product), bohr ↔ Å, hartree, eV and Faraday's kcal/mol forms.
const FACTOR_LITERALS: [&str; 9] = [
    "4.184", "418.4", "0.52917", "1.88972", "627.50", "27.211", "23.060", "96.485", "0.043364",
];

/// The `.rs` files of the crate, with their path relative to `src`.
fn crate_sources() -> Vec<(PathBuf, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&root, &mut files);
    files
        .into_iter()
        .map(|f| {
            let rel = f.strip_prefix(&root).unwrap().display().to_string();
            (f, rel)
        })
        .collect()
}

/// Whether `rel` is the units module, which defines the units, or this file.
fn defines_units(rel: &str) -> bool {
    rel.starts_with("core/units/") || rel == "core/constants.rs" || rel == "module_boundaries.rs"
}

/// Whether the word `name` occurs in `code` (not as part of a longer name).
fn names(code: &str, name: &str) -> bool {
    code.match_indices(name).any(|(i, _)| {
        let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let before = code[..i].chars().next_back().is_none_or(|c| !word(c));
        let after = code[i + name.len()..]
            .chars()
            .next()
            .is_none_or(|c| !word(c));
        before && after
    })
}

#[test]
fn no_module_names_a_retired_conversion_constant() {
    let mut found = Vec::new();
    for (file, rel) in crate_sources() {
        if rel == "module_boundaries.rs" {
            continue;
        }
        let text = std::fs::read_to_string(&file).unwrap();
        for (n, line) in text.lines().enumerate() {
            if RETIRED_FACTOR_CONSTANTS.iter().any(|c| names(line, c)) {
                found.push(format!("{rel}:{}: {}", n + 1, line.trim()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "unit conversions go through core::units (UnitFactor), not a constant:\n{}",
        found.join("\n")
    );
}

#[test]
fn no_module_defines_a_conversion_factor_constant() {
    let mut found = Vec::new();
    for (file, rel) in crate_sources() {
        if defines_units(&rel) {
            continue;
        }
        let text = std::fs::read_to_string(&file).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.trim();
            let decl = code
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub ");
            let Some(rest) = decl
                .strip_prefix("const ")
                .or_else(|| decl.strip_prefix("static "))
            else {
                continue;
            };
            let name = rest.split(':').next().unwrap_or("");
            let numeric = rest.contains(": F =") || rest.contains(": f64 =");
            if numeric && (name.contains("_PER_") || name.contains("_TO_")) {
                found.push(format!("{rel}:{}: {code}", n + 1));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a conversion factor is a `static UnitFactor`, resolved from its two units:\n{}",
        found.join("\n")
    );
}

#[test]
fn no_non_test_code_writes_a_conversion_factor_literal() {
    let mut found = Vec::new();
    let files = crate_sources();
    let paths: Vec<PathBuf> = files.iter().map(|(f, _)| f.clone()).collect();
    let test_only = test_only_modules(&paths);
    for (file, rel) in &files {
        if defines_units(rel) || test_only.iter().any(|t| file.starts_with(t)) {
            continue;
        }
        // Parameter tables transcribe engine data files verbatim.
        if rel.starts_with("ff/params/") {
            continue;
        }
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.trim();
            if is_test_cfg(code) {
                break;
            }
            if code.starts_with("//") {
                continue;
            }
            if FACTOR_LITERALS.iter().any(|lit| names(code, lit)) {
                found.push(format!("{rel}:{}: {code}", n + 1));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a hand-written unit conversion; use core::units::UnitFactor:\n{}",
        found.join("\n")
    );
}
