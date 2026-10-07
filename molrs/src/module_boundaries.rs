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
// `UnitRegistry::factor`, `Quantity::to`), each `UnitFactor` is defined once
// in `core::unit_factors`, and `core::constants` holds physical and engine
// constants, never a conversion factor. These checks fail on a
// conversion-factor constant (defined or named) outside the units module, a
// `UnitFactor` defined anywhere but `core::unit_factors`, and a conversion
// factor written by hand in non-test code — of molrs and of every binder
// crate beside it (molrs-python, molrs-wasm, molrs-capi, molrs-ffi,
// molrs-cxxapi, molrs-ext-example).
//
// Exemptions, each on purpose:
// * the units module itself (`core/units/`, `core/unit_factors.rs`) and
//   `core/constants.rs`, which define the units and hold the engines' data
//   (`PARMCHK2_PI`'s `/ 180`, `MMFF_MDYNE_A_TO_KCAL_MOL`'s `143.9325`);
// * `ff/params/`, whose tables transcribe engine data files verbatim;
// * test code (a `#[cfg(test)]` module file or a file's trailing
//   `#[cfg(test)]` block): a test may write an engine's number by hand
//   (`4.184`, `332.06371`, `1.987…e-3`) as an independent oracle the
//   registry's value is checked against.
// ---------------------------------------------------------------------------

/// Conversion-factor constants `core::constants` no longer defines.
const RETIRED_FACTOR_CONSTANTS: [&str; 12] = [
    "KJ_PER_KCAL",
    "ANGSTROM_PER_NM",
    "ANGSTROM_PER_BOHR",
    "ANGSTROM3_PER_CM3",
    "ANGSTROM_M",
    "FEMTOSECOND_S",
    "CENTIMETER_PER_METER",
    "OPENMM_COULOMB",
    "GROMACS_COULOMB",
    "BOLTZMANN_REAL",
    "KCAL_MOL_PER_MDYNE_ANGSTROM",
    "RADIANS_PER_DEGREE",
];

/// Conversion factors as they would be written by hand: kcal ↔ kJ (and its
/// nm² / Å² product), bohr ↔ Å, hartree, eV and Faraday's kcal/mol forms.
const FACTOR_LITERALS: [&str; 9] = [
    "4.184", "418.4", "0.52917", "1.88972", "627.50", "27.211", "23.060", "96.485", "0.043364",
];

/// The leading digits of a hand-written factor, matched whatever digits
/// follow: π/180 and 180/π (degrees ↔ radians is `to_radians` /
/// `to_degrees`, or the registry's `deg`), MMFF's mdyne·Å → kcal/mol
/// (`MMFF_MDYNE_A_TO_KCAL_MOL`), and k_B in kcal·mol⁻¹·K⁻¹
/// (`UnitPreset::real().boltzmann()`).
const FACTOR_PREFIXES: [&str; 4] = ["0.0174532", "57.29577", "143.9325", "0.001987"];

/// The `.rs` files of molrs (`src`) and of every binder crate present beside
/// it, each with its path relative to its crate's `src` (a binder's prefixed
/// by its crate name).
fn crate_sources() -> Vec<(PathBuf, String)> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut roots = vec![(manifest.join("src"), String::new())];
    for binder in [
        "molrs-python",
        "molrs-wasm",
        "molrs-capi",
        "molrs-ffi",
        "molrs-cxxapi",
        "molrs-ext-example",
    ] {
        let src = manifest.join("..").join(binder).join("src");
        if src.is_dir() {
            roots.push((src, format!("{binder}/")));
        }
    }
    let mut out = Vec::new();
    for (root, prefix) in roots {
        let mut files = Vec::new();
        sources(&root, &mut files);
        out.extend(files.into_iter().map(|f| {
            let rel = format!("{prefix}{}", f.strip_prefix(&root).unwrap().display());
            (f, rel)
        }));
    }
    out
}

/// Whether `rel` is the units module, which defines the units, or this file.
fn defines_units(rel: &str) -> bool {
    rel.starts_with("core/units/")
        || rel == "core/unit_factors.rs"
        || rel == "core/constants.rs"
        || rel == "module_boundaries.rs"
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

/// Whether `code` writes a numeric literal starting with `prefix`.
fn starts_a_literal(code: &str, prefix: &str) -> bool {
    code.match_indices(prefix).any(|(i, _)| {
        code[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
    })
}

/// Whether `code` writes k_B in kcal·mol⁻¹·K⁻¹ by hand as `1.987…e-3`.
fn writes_boltzmann_in_kcal(code: &str) -> bool {
    code.match_indices("1.987").any(|(i, _)| {
        let before = code[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'));
        let literal: String = code[i..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '_' | 'e' | 'E' | '-'))
            .collect();
        before && (literal.ends_with("e-3") || literal.ends_with("E-3"))
    })
}

/// Whether `code` divides or multiplies by 180 by hand (`/ 180.0`, `* 180.`,
/// `180.0 / PI`): a degree ↔ radian conversion. An expression's `(pi/180)`
/// (the IR's expression language, which converts a degree-valued parameter
/// itself) is not Rust arithmetic.
fn scales_by_180(code: &str) -> bool {
    let number = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.';
    code.match_indices("180").any(|(i, _)| {
        let head = &code[..i];
        if head.chars().next_back().is_some_and(number) {
            return false; // part of a longer number or name
        }
        if head.ends_with("(pi/") {
            return false; // an expression's own degree conversion, `(pi/180)`
        }
        // The literal's end: `180`, `180.`, `180.0`, `180.0_f64`, `180f64`.
        let tail = &code[i + 3..];
        if tail.starts_with(|c: char| c.is_ascii_digit()) {
            return false;
        }
        let tail = tail.trim_start_matches(number).trim_start();
        let head = head.trim_end();
        head.ends_with(['/', '*']) || tail.starts_with(['/', '*'])
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
            let code = code.split(" //").next().unwrap_or(code);
            if FACTOR_LITERALS.iter().any(|lit| names(code, lit))
                || FACTOR_PREFIXES.iter().any(|p| starts_a_literal(code, p))
                || writes_boltzmann_in_kcal(code)
                || scales_by_180(code)
            {
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

#[test]
fn a_unit_factor_is_defined_once_in_core_unit_factors() {
    let mut found = Vec::new();
    for (file, rel) in crate_sources() {
        if defines_units(&rel) {
            continue;
        }
        let text = std::fs::read_to_string(&file).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.trim();
            if code.starts_with("//") {
                continue;
            }
            if code.contains("UnitFactor::new(") || code.contains(": UnitFactor =") {
                found.push(format!("{rel}:{}: {code}", n + 1));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a unit factor is defined once, in core::unit_factors:\n{}",
        found.join("\n")
    );
}

#[test]
fn the_hand_written_factor_matchers_match() {
    for code in [
        "x * 0.017453292519943295",
        "let k = 143.9325 * ka;",
        "let kb = 1.987_204_258_640_83e-3;",
        "let kb = 0.0019872067;",
        "(theta0 * PI / 180.0).sqrt()",
        "theta / 180.0_f64",
        "x * 180.0 / PI",
        "x*180/PI",
        "deg * 57.29577951308232",
    ] {
        assert!(
            FACTOR_PREFIXES.iter().any(|p| starts_a_literal(code, p))
                || writes_boltzmann_in_kcal(code)
                || scales_by_180(code),
            "{code}"
        );
    }
    for code in [
        "let alpha = 1.987;",
        "phase in 0..=180",
        "\"180°\"",
        "n180 / 2.0",
        "x * 1180.0",
        "assert_eq!(n, 180);",
        "\"k*(theta-theta0*(pi/180))^2; pi=3.141592653589793\"",
    ] {
        assert!(
            !(FACTOR_PREFIXES.iter().any(|p| starts_a_literal(code, p))
                || writes_boltzmann_in_kcal(code)
                || scales_by_180(code)),
            "{code}"
        );
    }
}
