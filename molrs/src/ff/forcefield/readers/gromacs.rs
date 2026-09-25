//! GROMACS `.top` / `.itp` force-field reader.
//!
//! Parses section tables (`[ atomtypes ]`, `[ atoms ]`, `[ bonds ]`, …) — the
//! same layout the historical molpy `GromacsTopReader` consumed — into a molrs
//! [`ForceField`] in molrs units (Å, kcal/mol, radians, e).
//!
//! # Notes
//!
//! - `[ atomtypes ]` rows are the only source of atom-type parameters. Each is
//!   defined on atom style `full` with numeric `mass` (amu), `charge` (e),
//!   `sigma` (Å = nm × 10), `epsilon` (kcal/mol = kJ/mol ÷ 4.184) and
//!   `atomic_number` when present, and string `ptype` and `bond_type` when
//!   present. Columns resolve from the right: the last five are
//!   `mass charge ptype V W` (`ptype` one of `A S V D B`); the one to three
//!   leading tokens are `name`, an optional `bond_type` and an optional
//!   integer `at.num`. V/W are σ/ε only under comb-rule 2, so `[ atomtypes ]`
//!   requires `[ defaults ]`.
//! - `[ atoms ]` is molecule topology (read by `io::data::top::read_top`), not
//!   force-field data: its rows only give the per-atom type list that resolves
//!   bonded rows' **1-based atom indices** to type names. Nothing from it is
//!   stored in the force field. A type it names with no `[ atomtypes ]` row is
//!   defined once, with empty params, after every `[ atomtypes ]` row.
//! - Bonded rows define their type tuple through the conflict rule: instances
//!   of one tuple with equal parameters are one type, with different
//!   parameters a `TypeConflict`. Numeric parameters (when present) are
//!   converted from GROMACS units.
//! - `#include` is optional (`include: true`); unresolved includes fail fast.
//!   Rows repeated across includes follow the same conflict rule.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::ForceFieldReader;
use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use molrs::store::type_labels::TypeName;

const KJ_PER_KCAL: f64 = 4.184;
const NM_TO_ANGSTROM: f64 = 10.0;

/// Reader for GROMACS topology force-field / molecule parameter tables.
#[derive(Debug, Clone, Default)]
pub struct GromacsTopFfReader {
    /// Follow `#include` directives (default false — matches historical molpy).
    pub include: bool,
    /// Optional extra include search roots (force-field directories).
    pub include_dirs: Vec<PathBuf>,
}

impl GromacsTopFfReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_include(mut self, include: bool) -> Self {
        self.include = include;
        self
    }
}

impl ForceFieldReader for GromacsTopFfReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        // In-memory path has no cwd for includes.
        let sections = parse_sections_text(text, self.include, None, &self.include_dirs)?;
        build_forcefield(&sections)
    }

    fn read(&self, path: &str) -> Result<ForceField, String> {
        let p = Path::new(path);
        if !p.is_file() {
            return Err(format!("file not found: {path}"));
        }
        let text = std::fs::read_to_string(p).map_err(|e| format!("read {path}: {e}"))?;
        let sections = parse_sections_text(
            &text,
            self.include,
            Some(p.parent().unwrap_or(Path::new("."))),
            &self.include_dirs,
        )?;
        build_forcefield(&sections)
    }
}

/// Convenience free function.
pub fn read_gromacs_top_ff(path: impl AsRef<Path>) -> Result<ForceField, String> {
    GromacsTopFfReader::new().read(
        path.as_ref()
            .to_str()
            .ok_or_else(|| "path is not valid UTF-8".to_string())?,
    )
}

// ---------------------------------------------------------------------------
// Section parse + #include
// ---------------------------------------------------------------------------

fn parse_sections_text(
    text: &str,
    include: bool,
    cwd: Option<&Path>,
    include_dirs: &[PathBuf],
) -> Result<HashMap<String, Vec<String>>, String> {
    let mut store: HashMap<String, Vec<String>> = HashMap::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    parse_into(
        text,
        cwd,
        include,
        include_dirs,
        &mut store,
        &mut visited,
        cwd.map(|p| p.to_path_buf()),
    )?;
    Ok(store)
}

fn parse_into(
    text: &str,
    file_cwd: Option<&Path>,
    include: bool,
    include_dirs: &[PathBuf],
    store: &mut HashMap<String, Vec<String>>,
    visited: &mut HashSet<PathBuf>,
    visit_key: Option<PathBuf>,
) -> Result<(), String> {
    if let Some(ref key) = visit_key
        && !visited.insert(key.clone())
    {
        return Ok(());
    }

    let mut current: Option<String> = None;
    for raw in text.lines() {
        let mut line = raw;
        // strip ; comments
        if let Some(pos) = line.find(';') {
            line = &line[..pos];
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            if include && let Some(inc) = parse_include(line) {
                let resolved = resolve_include(&inc, file_cwd, include_dirs)?;
                let body = std::fs::read_to_string(&resolved)
                    .map_err(|e| format!("include {}: {e}", resolved.display()))?;
                let parent = resolved.parent().map(|p| p.to_path_buf());
                parse_into(
                    &body,
                    parent.as_deref(),
                    include,
                    include_dirs,
                    store,
                    visited,
                    Some(resolved),
                )?;
            }
            continue;
        }
        if let Some(sec) = parse_section_header(line) {
            current = Some(sec);
            store.entry(current.clone().unwrap()).or_default();
            continue;
        }
        let key = current.get_or_insert_with(|| "__preamble__".to_string());
        store.entry(key.clone()).or_default().push(line.to_string());
    }
    Ok(())
}

fn parse_section_header(line: &str) -> Option<String> {
    let t = line.trim();
    if t.starts_with('[') && t.ends_with(']') {
        Some(t[1..t.len() - 1].trim().to_ascii_lowercase())
    } else {
        None
    }
}

fn parse_include(line: &str) -> Option<String> {
    // #include "foo" or #include <foo>
    let rest = line.trim().strip_prefix('#')?.trim();
    let rest = rest.strip_prefix("include")?.trim();
    let rest = rest.trim_matches(|c| c == '"' || c == '<' || c == '>');
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

fn resolve_include(
    inc: &str,
    cwd: Option<&Path>,
    include_dirs: &[PathBuf],
) -> Result<PathBuf, String> {
    let p = Path::new(inc);
    if p.is_absolute() && p.is_file() {
        return Ok(p.to_path_buf());
    }
    if let Some(cwd) = cwd {
        let c = cwd.join(inc);
        if c.is_file() {
            return Ok(c);
        }
    }
    for d in include_dirs {
        let c = d.join(inc);
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(format!("Could not resolve include '{inc}'"))
}

// ---------------------------------------------------------------------------
// Unit conversion
// ---------------------------------------------------------------------------

fn bond_params_to_internal(style: &str, values: &[f64]) -> Result<Vec<(String, f64)>, String> {
    let names: &[&str] = match style {
        "harmonic" | "G96" => &["r0", "k"],
        "morse" => &["r0", "De", "alpha"],
        "cubic" => &["r0", "k2", "k3", "k4"],
        _ => return Err(format!("Unknown bond style {style}")),
    };
    let mut out = Vec::new();
    for (n, &v) in names.iter().zip(values.iter()) {
        let conv = match *n {
            "r0" => v * NM_TO_ANGSTROM,
            "k" | "k2" => v / (KJ_PER_KCAL * NM_TO_ANGSTROM * NM_TO_ANGSTROM),
            "k3" => v / (KJ_PER_KCAL * NM_TO_ANGSTROM.powi(3)),
            "k4" => v / (KJ_PER_KCAL * NM_TO_ANGSTROM.powi(4)),
            "De" => v / KJ_PER_KCAL,
            "alpha" => v / NM_TO_ANGSTROM,
            _ => v,
        };
        out.push(((*n).to_string(), conv));
    }
    Ok(out)
}

fn angle_params_to_internal(style: &str, values: &[f64]) -> Result<Vec<(String, f64)>, String> {
    let names: &[&str] = match style {
        "harmonic" | "G96" => &["theta0", "k"],
        "quartic" => &["c0", "c1", "c2", "c3"],
        "ub" => &["theta0", "k", "r0", "k_ub"],
        _ => return Err(format!("Unknown angle style {style}")),
    };
    let mut out = Vec::new();
    for (n, &v) in names.iter().zip(values.iter()) {
        let conv = match *n {
            "theta0" => v.to_radians(),
            "k" => v / KJ_PER_KCAL,
            "r0" => v * NM_TO_ANGSTROM,
            "k_ub" => v / (KJ_PER_KCAL * NM_TO_ANGSTROM * NM_TO_ANGSTROM),
            _ => v,
        };
        out.push(((*n).to_string(), conv));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

fn parse_defaults_section(lines: &[String]) -> Result<SpecialBonds, String> {
    for line in lines {
        let t = line.trim();
        if t.is_empty() || t.starts_with(';') {
            continue;
        }
        let cols: Vec<&str> = t.split_whitespace().collect();
        if cols.len() < 5 {
            return Err("[ defaults ] is missing fudgeLJ or fudgeQQ".into());
        }
        let nbfunc: i32 = cols[0]
            .parse()
            .map_err(|_| format!("[ defaults ] nbfunc is not an integer: {}", cols[0]))?;
        if nbfunc != 1 {
            return Err(format!("[ defaults ] nbfunc {nbfunc} is not supported"));
        }
        if cols[1] != "2" {
            return Err(format!(
                "[ defaults ] comb-rule {} is not supported",
                cols[1]
            ));
        }
        if cols[2] != "yes" {
            return Err("[ defaults ] gen-pairs must be yes".into());
        }
        let fudge_lj: f64 = cols[3]
            .parse()
            .map_err(|_| format!("[ defaults ] fudgeLJ is not a number: {}", cols[3]))?;
        let fudge_qq: f64 = cols[4]
            .parse()
            .map_err(|_| format!("[ defaults ] fudgeQQ is not a number: {}", cols[4]))?;
        return Ok(SpecialBonds {
            lj: [0.0, 0.0, fudge_lj],
            coul: [0.0, 0.0, fudge_qq],
        });
    }
    Err("[ defaults ] section is empty".into())
}

fn build_forcefield(sections: &HashMap<String, Vec<String>>) -> Result<ForceField, String> {
    let mut ff = ForceField::new("GROMACS");
    match sections.get("defaults") {
        Some(lines) => ff.set_special_bonds(parse_defaults_section(lines)?),
        None if sections.contains_key("pairs") => {
            return Err("[ defaults ] is required when [ pairs ] is present".into());
        }
        None => {}
    }

    // `[ atomtypes ]` rows are the atom types' parameters; V/W mean σ/ε only
    // under comb-rule 2, which `parse_defaults_section` enforces.
    if sections.contains_key("atomtypes") && !sections.contains_key("defaults") {
        return Err("[ defaults ] is required when [ atomtypes ] is present".into());
    }
    let style = ff
        .def_style("atom", "full", Params::new())
        .map_err(|e| e.to_string())?;
    for line in sections.get("atomtypes").into_iter().flatten() {
        let (name, params) = parse_atomtypes_row(line)?;
        style.def_type(name, params).map_err(|e| e.to_string())?;
    }

    // `[ atoms ]` is topology: only its type column is read, to resolve bonded
    // rows' atom indices. A type it names without an `[ atomtypes ]` row is
    // referenced, not parameterised — defined once, with empty params.
    let mut atom_type_names: Vec<String> = Vec::new();
    for line in sections.get("atoms").into_iter().flatten() {
        let tname = line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| format!("[ atoms ] row has no type column: {line}"))?;
        if style.get_atomtype(tname).is_none() {
            style
                .def_type(tname, Params::new())
                .map_err(|e| e.to_string())?;
        }
        atom_type_names.push(tname.to_owned());
    }

    // Bonds — empty style name matches historical BondStyle("harmonic") bug
    // where the name landed on the `ff` slot. Keep empty for style_names parity.
    parse_bond_section(
        sections.get("bonds").map(|v| v.as_slice()).unwrap_or(&[]),
        &atom_type_names,
        &mut ff,
    )?;
    parse_angle_section(
        sections.get("angles").map(|v| v.as_slice()).unwrap_or(&[]),
        &atom_type_names,
        &mut ff,
    )?;
    parse_dihedral_section(
        sections
            .get("dihedrals")
            .map(|v| v.as_slice())
            .unwrap_or(&[]),
        &atom_type_names,
        &mut ff,
    )?;
    parse_pair_section(
        sections.get("pairs").map(|v| v.as_slice()).unwrap_or(&[]),
        &atom_type_names,
        &mut ff,
    )?;

    Ok(ff)
}

/// One `[ atomtypes ]` row → `(name, params)` in molrs units.
///
/// Columns resolve from the right: the last five are `mass charge ptype V W`;
/// the one to three leading tokens are `name`, then an optional `bond_type`
/// and an optional integer `at.num`. Anything else is `Err` naming the row.
fn parse_atomtypes_row(line: &str) -> Result<(&str, Params), String> {
    let cols: Vec<&str> = line.split_whitespace().collect();
    let bad = |why: &str| format!("[ atomtypes ] row '{line}': {why}");
    if cols.len() < 6 {
        return Err(bad(
            "expected `name [bond_type] [at.num] mass charge ptype V W`",
        ));
    }
    let (lead, tail) = cols.split_at(cols.len() - 5);
    let is_int = |tok: &str| tok.parse::<i64>().is_ok();
    let (name, bond_type, at_num) = match *lead {
        [name] => (name, None, None),
        [name, second] if is_int(second) => (name, None, Some(second)),
        [name, second] => (name, Some(second), None),
        [name, bond_type, at_num] if is_int(at_num) => (name, Some(bond_type), Some(at_num)),
        [_, _, _] => return Err(bad("at.num is not an integer")),
        _ => return Err(bad("more than three tokens before `mass charge ptype V W`")),
    };
    let number = |tok: &str, what: &str| {
        tok.parse::<f64>()
            .map_err(|_| bad(&format!("{what} is not a number: {tok}")))
    };
    let mass = number(tail[0], "mass")?;
    let charge = number(tail[1], "charge")?;
    let ptype = tail[2];
    if !matches!(ptype, "A" | "S" | "V" | "D" | "B") {
        return Err(bad(&format!("ptype '{ptype}' is not one of A S V D B")));
    }
    let sigma_nm = number(tail[3], "V (sigma)")?;
    let epsilon_kj = number(tail[4], "W (epsilon)")?;

    let mut params = Params::from_pairs(&[
        ("mass", mass),
        ("charge", charge),
        ("sigma", sigma_nm * NM_TO_ANGSTROM),
        ("epsilon", epsilon_kj / KJ_PER_KCAL),
    ]);
    if let Some(tok) = at_num {
        params.set("atomic_number", number(tok, "at.num")?);
    }
    params.set_str("ptype", ptype);
    if let Some(bond_type) = bond_type {
        params.set_str("bond_type", bond_type);
    }
    Ok((name, params))
}

fn atom_name_at(names: &[String], idx_1based: usize) -> Result<String, String> {
    names
        .get(idx_1based.wrapping_sub(1))
        .cloned()
        .ok_or_else(|| format!("atom index {idx_1based} out of range"))
}

fn parse_bond_section(
    lines: &[String],
    atom_names: &[String],
    ff: &mut ForceField,
) -> Result<(), String> {
    let func_types: HashMap<&str, &str> = [
        ("1", "harmonic"),
        ("2", "G96"),
        ("3", "morse"),
        ("4", "cubic"),
    ]
    .into_iter()
    .collect();

    // One empty-named bond style (historical molpy surface).
    ff.def_style("bond", "", Params::new())
        .map_err(|e| e.to_string())?;
    for raw in lines {
        let cols: Vec<&str> = raw.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        let i: usize = cols[0]
            .parse()
            .map_err(|_| format!("bad bond i: {}", cols[0]))?;
        let j: usize = cols[1]
            .parse()
            .map_err(|_| format!("bad bond j: {}", cols[1]))?;
        let funct = cols[2];
        let style_name = *func_types
            .get(funct)
            .ok_or_else(|| format!("Unknown bond funct '{funct}' in line: {raw}"))?;
        // Historical reader always used empty style name via BondStyle(style_name) bug.
        // Types still go on the empty-named style.
        let _ = style_name;
        let params: Vec<f64> = cols[3..]
            .iter()
            .map(|t| t.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("bond params: {e}"))?;
        let converted = if params.is_empty() {
            Vec::new()
        } else {
            bond_params_to_internal(style_name, &params)?
        };
        let iname = atom_name_at(atom_names, i)?;
        let jname = atom_name_at(atom_names, j)?;
        let owned: Vec<(&str, f64)> = converted.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        ff.get_style_mut("bond", "")
            .ok_or("bond style missing")?
            .def_type_at(
                TypeName::join(&[&iname, &jname])?.as_str(),
                &[&iname, &jname],
                Params::from_pairs(&owned),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn parse_angle_section(
    lines: &[String],
    atom_names: &[String],
    ff: &mut ForceField,
) -> Result<(), String> {
    let func_types: HashMap<&str, &str> = [
        ("1", "harmonic"),
        ("2", "G96"),
        ("3", "quartic"),
        ("4", "ub"),
    ]
    .into_iter()
    .collect();

    ff.def_style("angle", "", Params::new())
        .map_err(|e| e.to_string())?;
    for raw in lines {
        let cols: Vec<&str> = raw.split_whitespace().collect();
        if cols.len() < 4 {
            continue;
        }
        let i: usize = cols[0].parse().map_err(|_| "bad angle i".to_string())?;
        let j: usize = cols[1].parse().map_err(|_| "bad angle j".to_string())?;
        let k: usize = cols[2].parse().map_err(|_| "bad angle k".to_string())?;
        let funct = cols[3];
        let style_name = *func_types
            .get(funct)
            .ok_or_else(|| format!("Unknown angle funct '{funct}' in line: {raw}"))?;
        let params: Vec<f64> = cols[4..]
            .iter()
            .map(|t| t.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("angle params: {e}"))?;
        let converted = if params.is_empty() {
            Vec::new()
        } else {
            angle_params_to_internal(style_name, &params)?
        };
        let owned: Vec<(&str, f64)> = converted.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let iname = atom_name_at(atom_names, i)?;
        let jname = atom_name_at(atom_names, j)?;
        let kname = atom_name_at(atom_names, k)?;
        ff.get_style_mut("angle", "")
            .ok_or("angle style missing")?
            .def_type_at(
                TypeName::join(&[&iname, &jname, &kname])?.as_str(),
                &[&iname, &jname, &kname],
                Params::from_pairs(&owned),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn parse_dihedral_section(
    lines: &[String],
    atom_names: &[String],
    ff: &mut ForceField,
) -> Result<(), String> {
    let func_types: HashMap<&str, &str> = [("1", "periodic"), ("2", "rb"), ("3", "harmonic")]
        .into_iter()
        .collect();
    let param_names: HashMap<&str, &[&str]> = [
        ("periodic", &["phase", "k", "periodicity"][..]),
        ("rb", &["c0", "c1", "c2", "c3", "c4", "c5"][..]),
        ("harmonic", &["psi0", "k"][..]),
    ]
    .into_iter()
    .collect();

    ff.def_style("dihedral", "", Params::new())
        .map_err(|e| e.to_string())?;
    for raw in lines {
        let cols: Vec<&str> = raw.split_whitespace().collect();
        if cols.len() < 5 {
            continue;
        }
        let i: usize = cols[0].parse().map_err(|_| "bad dihedral i")?;
        let j: usize = cols[1].parse().map_err(|_| "bad dihedral j")?;
        let k: usize = cols[2].parse().map_err(|_| "bad dihedral k")?;
        let l: usize = cols[3].parse().map_err(|_| "bad dihedral l")?;
        let funct = cols[4];
        let style_name = *func_types
            .get(funct)
            .ok_or_else(|| format!("Unknown dihedral funct '{funct}' in line: {raw}"))?;
        let params: Vec<f64> = cols[5..]
            .iter()
            .map(|t| t.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("dihedral params: {e}"))?;
        let names = param_names[style_name];
        // Normalize to the canonical vocabulary at the reader boundary (spec
        // ff-params-01): angles in radians, energies in kcal/mol. The GROMACS
        // file spells the phase in degrees and every barrier in kJ/mol.
        let converted: Vec<(String, f64)> = names
            .iter()
            .zip(params.iter())
            .map(|(n, v)| {
                let conv = match *n {
                    "phase" | "psi0" => v.to_radians(),
                    "k" | "c0" | "c1" | "c2" | "c3" | "c4" | "c5" => v / KJ_PER_KCAL,
                    _ => *v,
                };
                ((*n).to_string(), conv)
            })
            .collect();
        let owned: Vec<(&str, f64)> = converted.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let iname = atom_name_at(atom_names, i)?;
        let jname = atom_name_at(atom_names, j)?;
        let kname = atom_name_at(atom_names, k)?;
        let lname = atom_name_at(atom_names, l)?;
        ff.get_style_mut("dihedral", "")
            .ok_or("dihedral style missing")?
            .def_type_at(
                TypeName::join(&[&iname, &jname, &kname, &lname])?.as_str(),
                &[&iname, &jname, &kname, &lname],
                Params::from_pairs(&owned),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn parse_pair_section(
    lines: &[String],
    atom_names: &[String],
    ff: &mut ForceField,
) -> Result<(), String> {
    let func_types: HashMap<&str, &str> =
        [("1", "lj12-6"), ("2", "buckingham")].into_iter().collect();
    // GROMACS buckingham is `a  b  c6` with `b = 1/rho` (1/nm) — a different
    // quantity from the canonical `rho`, so it is inverted here rather than
    // stored under a third spelling (spec ff-params-01).
    let param_names: HashMap<&str, &[&str]> = [
        ("lj12-6", &["c6", "c12"][..]),
        ("buckingham", &["a", "rho", "c"][..]),
    ]
    .into_iter()
    .collect();

    ff.def_style("pair", "", Params::new())
        .map_err(|e| e.to_string())?;
    for raw in lines {
        let cols: Vec<&str> = raw.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        let i: usize = cols[0].parse().map_err(|_| "bad pair i")?;
        let j: usize = cols[1].parse().map_err(|_| "bad pair j")?;
        let funct = cols[2];
        let style_name = *func_types
            .get(funct)
            .ok_or_else(|| format!("Unknown pair funct '{funct}' in line: {raw}"))?;
        let params: Vec<f64> = cols[3..]
            .iter()
            .map(|t| t.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("pair params: {e}"))?;
        let names = param_names[style_name];
        let converted: Vec<(String, f64)> = names
            .iter()
            .zip(params.iter())
            .map(|(n, v)| {
                let conv = match *n {
                    // kJ/mol → kcal/mol
                    "a" => v / KJ_PER_KCAL,
                    // b (1/nm) → rho (Å)
                    "rho" => {
                        if *v == 0.0 {
                            0.0
                        } else {
                            NM_TO_ANGSTROM / v
                        }
                    }
                    // kJ/mol·nm⁶ → kcal/mol·Å⁶
                    "c" | "c6" => v / KJ_PER_KCAL * NM_TO_ANGSTROM.powi(6),
                    // kJ/mol·nm¹² → kcal/mol·Å¹²
                    "c12" => v / KJ_PER_KCAL * NM_TO_ANGSTROM.powi(12),
                    _ => *v,
                };
                ((*n).to_string(), conv)
            })
            .collect();
        let owned: Vec<(&str, f64)> = converted.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let iname = atom_name_at(atom_names, i)?;
        let jname = atom_name_at(atom_names, j)?;
        ff.get_style_mut("pair", "")
            .ok_or("pair style missing")?
            .def_type_at(
                TypeName::pair(&iname, &jname)?.as_str(),
                &[&iname, &jname],
                Params::from_pairs(&owned),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atoms_only_minimal() {
        let text = r#"
[ atoms ]
1  opls_135  1  LIG  C  1  -0.18  12.011
2  opls_140  1  LIG  H  2   0.06   1.008
[ bonds ]
1  2  1
"#;
        let ff = GromacsTopFfReader::new().read_str(text).expect("parse");
        assert_eq!(ff.get_atomtypes().len(), 2);
        assert_eq!(ff.get_bondtypes().len(), 1);
    }

    // -- [ atomtypes ] ---------------------------------------------------------

    /// `nbfunc 1`, comb-rule 2 (V/W are σ/ε), `gen-pairs yes`.
    const DEFAULTS: &str = "[ defaults ]\n1  2  yes  0.5  0.8333\n";

    fn read(text: &str) -> ForceField {
        GromacsTopFfReader::new()
            .read_str(text)
            .unwrap_or_else(|e| panic!("read_str: {e}"))
    }

    fn read_err(text: &str) -> String {
        GromacsTopFfReader::new()
            .read_str(text)
            .expect_err("expected Err from read_str")
    }

    fn with_atomtypes(row: &str) -> String {
        format!("{DEFAULTS}[ atomtypes ]\n{row}\n")
    }

    fn atom_type<'a>(ff: &'a ForceField, name: &str) -> &'a crate::ff::forcefield::AtomType {
        ff.get_style("atom", "full")
            .and_then(|s| s.get_atomtype(name))
            .unwrap_or_else(|| panic!("no atom type {name}"))
    }

    /// 6-column form `name mass charge ptype V W`. Hand conversion:
    /// σ = 0.35 nm × 10 = 3.5 Å; ε = 0.276144 kJ/mol ÷ 4.184 = 0.066 kcal/mol.
    #[test]
    fn atomtypes_row_is_defined_in_molrs_units() {
        let ff = read(&with_atomtypes(
            "opls_135  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert!((p.get("sigma").unwrap() - 3.5).abs() < 1e-12);
        assert!((p.get("epsilon").unwrap() - 0.066).abs() < 1e-10);
        assert_eq!(p.get("mass"), Some(12.011));
        assert_eq!(p.get("charge"), Some(-0.18));
        assert_eq!(p.get_str("ptype"), Some("A"));
    }

    #[test]
    fn atomtypes_six_column_row_has_no_bond_type_or_atomic_number() {
        let ff = read(&with_atomtypes(
            "opls_135  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get_str("bond_type"), None);
        assert_eq!(p.get("atomic_number"), None);
    }

    /// Seven columns with an integer second token: `name at.num mass …`.
    #[test]
    fn atomtypes_seven_column_row_with_an_integer_reads_the_atomic_number() {
        let ff = read(&with_atomtypes(
            "opls_135  6  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get("atomic_number"), Some(6.0));
        assert_eq!(p.get_str("bond_type"), None);
        assert_eq!(p.get("mass"), Some(12.011));
    }

    /// Seven columns with a non-integer second token: `name bond_type mass …`.
    #[test]
    fn atomtypes_seven_column_row_with_a_label_reads_the_bond_type() {
        let ff = read(&with_atomtypes(
            "opls_135  CT  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get_str("bond_type"), Some("CT"));
        assert_eq!(p.get("atomic_number"), None);
        assert_eq!(p.get("mass"), Some(12.011));
    }

    /// Eight columns: `name bond_type at.num mass charge ptype V W`.
    #[test]
    fn atomtypes_eight_column_row_reads_bond_type_and_atomic_number() {
        let ff = read(&with_atomtypes(
            "opls_135  CT  6  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get_str("bond_type"), Some("CT"));
        assert_eq!(p.get("atomic_number"), Some(6.0));
        assert_eq!(p.get("mass"), Some(12.011));
        assert_eq!(p.get_str("ptype"), Some("A"));
    }

    /// `at.num` must be an integer token.
    #[test]
    fn atomtypes_eight_column_row_with_a_non_integer_atomic_number_is_an_error() {
        let err = read_err(&with_atomtypes(
            "opls_135  CT  C  12.011  -0.18  A  0.35  0.276144",
        ));
        assert!(err.contains("opls_135"), "error should name the row: {err}");
    }

    /// More than three leading tokens before `mass charge ptype V W`.
    #[test]
    fn atomtypes_row_with_four_leading_tokens_is_an_error() {
        let err = read_err(&with_atomtypes(
            "opls_135  CT  6  extra  12.011  -0.18  A  0.35  0.276144",
        ));
        assert!(err.contains("opls_135"), "error should name the row: {err}");
    }

    /// `ptype` is one of `A S V D B`.
    #[test]
    fn atomtypes_row_with_an_unknown_ptype_is_an_error() {
        let err = read_err(&with_atomtypes(
            "opls_135  12.011  -0.18  X  0.35  0.276144",
        ));
        assert!(err.contains("opls_135"), "error should name the row: {err}");
    }

    /// V/W mean σ/ε only under comb-rule 2, which only `[ defaults ]` declares.
    #[test]
    fn atomtypes_without_defaults_is_an_error() {
        let text = "[ atomtypes ]\nopls_135  12.011  -0.18  A  0.35  0.276144\n";
        let err = read_err(text);
        assert!(
            err.contains("defaults"),
            "error should name [ defaults ]: {err}"
        );
    }

    // -- [ atoms ] is topology, not force-field parameters ---------------------

    /// A full 11-column `[ atoms ]` row (with the B-state columns) leaves no
    /// per-atom string on any atom type.
    #[test]
    fn atoms_rows_put_no_per_atom_field_on_an_atom_type() {
        let text = format!(
            "{DEFAULTS}[ atomtypes ]\n\
             opls_135  12.011  -0.18  A  0.35  0.276144\n\
             [ atoms ]\n\
             1  opls_135  1  LIG  C1  1  -0.18  12.011  opls_135  -0.18  12.011\n\
             2  opls_140  1  LIG  H1  2   0.06   1.008  opls_140   0.06   1.008\n"
        );
        let ff = read(&text);
        let per_atom = [
            "nr", "resnr", "residu", "atom", "cgnr", "charge", "mass", "typeB", "chargeB", "massB",
        ];
        let types = ff.get_atomtypes();
        assert_eq!(types.len(), 2);
        for t in types {
            for key in per_atom {
                assert_eq!(
                    t.params.get_str(key),
                    None,
                    "{} carries per-atom string `{key}`",
                    t.name
                );
            }
        }
    }

    /// Two `[ atoms ]` rows of a type with no `[ atomtypes ]` row: the type is
    /// referenced, not parameterised — one definition, empty params.
    #[test]
    fn a_type_named_only_in_atoms_is_defined_once_with_empty_params() {
        let text = "[ atoms ]\n\
                    1  opls_140  1  LIG  H1  1  0.06  1.008\n\
                    2  opls_140  1  LIG  H2  1  0.06  1.008\n";
        let ff = read(text);
        let types = ff.get_atomtypes();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "opls_140");
        assert_eq!(types[0].params.iter().count(), 0);
        assert_eq!(types[0].params.iter_strings().count(), 0);
    }

    /// A type with an `[ atomtypes ]` row keeps those params when `[ atoms ]`
    /// names it: the reference is not a second, empty definition.
    #[test]
    fn a_type_named_in_atoms_keeps_its_atomtypes_params() {
        let text = format!(
            "{DEFAULTS}[ atomtypes ]\n\
             opls_135  12.011  -0.18  A  0.35  0.276144\n\
             [ atoms ]\n\
             1  opls_135  1  LIG  C1  1  -0.18  12.011\n"
        );
        let ff = read(&text);
        assert_eq!(ff.get_atomtypes().len(), 1);
        assert!((atom_type(&ff, "opls_135").params.get("sigma").unwrap() - 3.5).abs() < 1e-12);
    }

    // -- bonded rows follow the conflict rule ----------------------------------

    const THREE_ATOMS: &str = "[ atoms ]\n\
                               1  opls_135  1  LIG  C1  1  -0.18  12.011\n\
                               2  opls_140  1  LIG  H1  1   0.06   1.008\n\
                               3  opls_140  1  LIG  H2  1   0.06   1.008\n";

    /// Two instances of one type pair with equal per-instance params: one type.
    #[test]
    fn two_bonds_rows_of_one_type_pair_with_equal_params_define_one_type() {
        let text = format!(
            "{THREE_ATOMS}[ bonds ]\n\
             1  2  1  0.1090  284512.0\n\
             1  3  1  0.1090  284512.0\n"
        );
        let ff = read(&text);
        assert_eq!(ff.get_bondtypes().len(), 1);
    }

    /// Two instances of one type pair with different `k`: a conflict, not
    /// last-writer-wins.
    #[test]
    fn two_bonds_rows_of_one_type_pair_with_different_k_are_an_error() {
        let text = format!(
            "{THREE_ATOMS}[ bonds ]\n\
             1  2  1  0.1090  284512.0\n\
             1  3  1  0.1090  300000.0\n"
        );
        let err = read_err(&text);
        assert!(
            err.contains("opls_135-opls_140"),
            "error should name the type: {err}"
        );
    }
}
