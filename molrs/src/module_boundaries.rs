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
//! * `ff`'s submodules are layers, each naming only the layers beneath it:
//!   `ir` (vocabulary) < `potential` (kernels) < `style_registry` (which
//!   kernel prices which style) < `forcefield` (the data model) < `compile`
//!   (a force field bound to its kernels) < `form_conversion` (a force field
//!   rewritten between the styles of a family). Beside them, `params` (the
//!   shipped tables) names no other `ff` module, and `typifier` never names
//!   `charge` (a charge model types its atoms with a typifier, not the
//!   reverse).
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
// ff: its submodules are layers.
//
// Unlike the checks above, these resolve every path a file names —
// `crate::…`, `molrs::…`, `super::…`, `self::…`, through `use` brace groups
// across lines — to the module it reaches, so a relative import counts too.
// ---------------------------------------------------------------------------

/// `ff`'s layers, lowest first: each names only the layers before it.
const FF_LAYERS: [&str; 6] = [
    "ir",
    "potential",
    "style_registry",
    "forcefield",
    "compile",
    "form_conversion",
];

/// Every `ff` submodule.
const FF_MODULES: [&str; 10] = [
    "ir",
    "potential",
    "style_registry",
    "forcefield",
    "compile",
    "form_conversion",
    "typifier",
    "charge",
    "params",
    "clpol_scaling",
];

/// The module path (`["ff", "ir", "form"]`) a source file under `src` is.
fn module_path(rel: &Path) -> Vec<String> {
    let mut segs: Vec<String> = rel
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if segs.last().is_some_and(|s| s == "mod") {
        segs.pop();
    }
    segs
}

/// The non-test code of `text`, comment lines blanked, cut at the trailing
/// `#[cfg(test)]` block — line numbers kept.
fn code_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for line in text.lines() {
        let code = line.trim();
        if is_test_cfg(code) {
            break;
        }
        out.push(if code.starts_with("//") { "" } else { line });
    }
    out
}

/// Every module path `lines` names, resolved from inside `here`, with the
/// line (1-based) it is named on: `crate::a::b` and `molrs::a::b` as
/// `[a, b]`, `super::x` and `self::x` against `here`, and each member of a
/// `use` brace group (nested, across lines) under its prefix.
fn named_paths(here: &[String], lines: &[&str]) -> Vec<(usize, Vec<String>)> {
    #[derive(PartialEq)]
    enum Tok {
        Ident(String),
        Sep,
        Open,
        Close,
        Comma,
        Other,
    }
    let mut toks: Vec<(usize, Tok)> = Vec::new();
    for (n, line) in lines.iter().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '"' {
                // Skip a string literal (one line; enough for paths).
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    i += if chars[i] == '\\' { 2 } else { 1 };
                }
                i += 1;
                toks.push((n, Tok::Other));
            } else if word(c) {
                let start = i;
                while i < chars.len() && word(chars[i]) {
                    i += 1;
                }
                toks.push((n, Tok::Ident(chars[start..i].iter().collect())));
            } else if c == ':' && chars.get(i + 1) == Some(&':') {
                toks.push((n, Tok::Sep));
                i += 2;
            } else {
                i += 1;
                let tok = match c {
                    '{' => Tok::Open,
                    '}' => Tok::Close,
                    ',' => Tok::Comma,
                    _ if c.is_whitespace() => continue,
                    _ => Tok::Other,
                };
                toks.push((n, tok));
            }
        }
    }

    let resolve = |path: &[String]| -> Option<Vec<String>> {
        let first = path.first()?.as_str();
        match first {
            "crate" | "molrs" => Some(path[1..].to_vec()),
            "self" => Some(
                here.iter()
                    .cloned()
                    .chain(path[1..].iter().cloned())
                    .collect(),
            ),
            "super" => {
                let ups = path.iter().take_while(|s| *s == "super").count();
                let base = here.len().checked_sub(ups)?;
                Some(
                    here[..base]
                        .iter()
                        .cloned()
                        .chain(path[ups..].iter().cloned())
                        .collect(),
                )
            }
            _ => None,
        }
    };

    let mut out = Vec::new();
    // The prefix of each open brace: `Some` for a `path::{` group.
    let mut groups: Vec<Option<Vec<String>>> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let Tok::Ident(_) = &toks[i].1 else {
            match toks[i].1 {
                Tok::Open => groups.push(None),
                Tok::Close => {
                    groups.pop();
                }
                _ => {}
            }
            i += 1;
            continue;
        };
        // A path: ident (:: ident)*, maybe ending in `::{`.
        let line = toks[i].0;
        let in_group = i > 0
            && matches!(toks[i - 1].1, Tok::Open | Tok::Comma)
            && groups.last().is_some_and(|g| g.is_some());
        let mut path = Vec::new();
        let mut opens_group = false;
        while let Some((_, Tok::Ident(s))) = toks.get(i) {
            path.push(s.clone());
            i += 1;
            if toks.get(i).map(|t| &t.1) != Some(&Tok::Sep) {
                break;
            }
            i += 1;
            if toks.get(i).map(|t| &t.1) == Some(&Tok::Open) {
                opens_group = true;
                i += 1;
                break;
            }
        }
        let full = if in_group {
            let prefix = groups.last().cloned().flatten().unwrap_or_default();
            let mut p = prefix;
            if path.first().is_some_and(|s| s == "self") {
                path.remove(0);
            }
            p.extend(path);
            Some(p)
        } else {
            resolve(&path)
        };
        if opens_group {
            groups.push(full.clone());
        }
        if let Some(p) = full
            && !opens_group
        {
            out.push((line + 1, p));
        }
    }
    out
}

/// The non-test code lines under `src/ff/<module>` that reach
/// `ff::<forbidden>`, as `file:line: path`.
fn ff_crossings(module: &str, forbidden: &str) -> Vec<String> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let dir = src.join("ff").join(module);
    let mut files = Vec::new();
    if dir.is_dir() {
        sources(&dir, &mut files);
    } else {
        files.push(dir.with_extension("rs"));
    }
    let test_only = test_only_modules(&files);
    let mut out = Vec::new();
    for file in &files {
        if test_only.iter().any(|t| file.starts_with(t)) {
            continue;
        }
        let rel = file.strip_prefix(&src).unwrap();
        let here = module_path(rel);
        let text = std::fs::read_to_string(file).unwrap();
        for (line, path) in named_paths(&here, &code_lines(&text)) {
            if path.len() >= 2 && path[0] == "ff" && path[1] == forbidden {
                out.push(format!("{}:{line}: {}", rel.display(), path.join("::")));
            }
        }
    }
    out
}

fn assert_no_ff_crossing(module: &str, forbidden: &str) {
    let found = ff_crossings(module, forbidden);
    assert!(
        found.is_empty(),
        "ff::{module} names ff::{forbidden} outside test code:\n{}",
        found.join("\n")
    );
}

#[test]
fn every_ff_layer_names_only_the_layers_beneath_it() {
    for (i, lower) in FF_LAYERS.iter().enumerate() {
        for upper in &FF_LAYERS[i + 1..] {
            assert_no_ff_crossing(lower, upper);
        }
    }
}

#[test]
fn ff_params_names_no_other_ff_module() {
    for other in FF_MODULES.iter().filter(|m| **m != "params") {
        assert_no_ff_crossing("params", other);
    }
}

#[test]
fn ff_typifier_names_no_charge_model() {
    assert_no_ff_crossing("typifier", "charge");
}

#[test]
fn the_ff_layer_check_sees_the_edges_that_exist() {
    // The compiler does read the force field and the kernels: a scan that
    // found nothing would pass every layer check vacuously.
    assert!(!ff_crossings("compile", "forcefield").is_empty());
    assert!(!ff_crossings("compile", "potential").is_empty());
    assert!(!ff_crossings("clpol_scaling", "forcefield").is_empty());
}

#[test]
fn the_ff_path_resolver_sees_relative_and_grouped_imports() {
    let here: Vec<String> = ["ff", "ir", "form"].map(String::from).to_vec();
    let lines = [
        "use super::super::potential::Potential;",
        "use crate::ff::{",
        "    compile::{self, PotentialCompiler},",
        "    forcefield::ForceField,",
        "};",
        "let s = \"crate::ff::charge\"; self::torsion::FAMILY",
    ];
    let paths: Vec<String> = named_paths(&here, &lines)
        .into_iter()
        .map(|(_, p)| p.join("::"))
        .collect();
    assert_eq!(
        paths,
        [
            "ff::potential::Potential",
            "ff::compile",
            "ff::compile::PotentialCompiler",
            "ff::forcefield::ForceField",
            "ff::ir::form::torsion::FAMILY",
        ]
    );
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
