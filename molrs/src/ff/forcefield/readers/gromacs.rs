//! GROMACS `.top` / `.itp` force-field reader.
//!
//! Parses section tables (`[ atoms ]`, `[ bonds ]`, …) — the same layout the
//! historical molpy `GromacsTopReader` consumed — into a molrs [`ForceField`]
//! in molrs units (Å, kcal/mol, radians, e).
//!
//! # Notes
//!
//! - Molecule topology files list **per-atom** rows under `[ atoms ]`; each row
//!   becomes an atom-type entry (duplicate type names are kept so bond indices
//!   resolve to the same per-atom handles the Python reader produced).
//! - Bonded rows reference **1-based atom indices** into that list when they
//!   carry no type labels; optional numeric parameters (when present) are
//!   converted from GROMACS units.
//! - `#include` is optional (`include: true`); unresolved includes fail fast.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::ForceFieldReader;
use crate::ff::forcefield::ForceField;

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

fn build_forcefield(sections: &HashMap<String, Vec<String>>) -> Result<ForceField, String> {
    let mut ff = ForceField::new("GROMACS");

    // Atom rows → atom types (one per row, strings for non-float metadata).
    let atom_lines = sections.get("atoms").cloned().unwrap_or_default();
    let atom_header = [
        "nr", "name", "resnr", "residu", "atom", "cgnr", "charge", "mass", "typeB", "chargeB",
        "massB",
    ];
    let mut atom_type_names: Vec<String> = Vec::new();
    let mut pending_strs: Vec<(String, Vec<(String, String)>)> = Vec::new();
    {
        let style = ff.def_atomstyle("full");
        for line in &atom_lines {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 2 {
                continue;
            }
            let mut map: HashMap<&str, &str> = HashMap::new();
            for (h, c) in atom_header.iter().zip(cols.iter()) {
                map.insert(*h, *c);
            }
            // "name" column in the historical header is the type token (2nd field).
            let tname = map.get("name").copied().unwrap_or("").to_string();
            if tname.is_empty() {
                continue;
            }
            // Numeric bag empty — charge/mass stay string params (molpy parity).
            style.def_atomtype(&tname, &[]);
            let strs: Vec<(String, String)> = map
                .iter()
                .filter(|(k, _)| **k != "name")
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            pending_strs.push((tname.clone(), strs));
            atom_type_names.push(tname);
        }
    }
    if let Some(st) = ff.get_style_mut("atom", "full") {
        for (tname, strs) in pending_strs {
            for (k, v) in strs {
                st.set_type_str_param(&tname, &k, &v);
            }
        }
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
    let _ = ff.def_bondstyle("");
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
            .def_bondtype(&iname, &jname, &owned);
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

    let _ = ff.def_anglestyle("");
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
            .def_angletype(&iname, &jname, &kname, &owned);
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
        ("periodic", &["phi0", "k", "n"][..]),
        ("rb", &["c0", "c1", "c2", "c3", "c4", "c5"][..]),
        ("harmonic", &["psi0", "k"][..]),
    ]
    .into_iter()
    .collect();

    let _ = ff.def_dihedralstyle("");
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
        // Dihedral: no unit conversion in historical Python (raw numbers).
        let converted: Vec<(String, f64)> = names
            .iter()
            .zip(params.iter())
            .map(|(n, v)| ((*n).to_string(), *v))
            .collect();
        let owned: Vec<(&str, f64)> = converted.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let iname = atom_name_at(atom_names, i)?;
        let jname = atom_name_at(atom_names, j)?;
        let kname = atom_name_at(atom_names, k)?;
        let lname = atom_name_at(atom_names, l)?;
        ff.get_style_mut("dihedral", "")
            .ok_or("dihedral style missing")?
            .def_dihedraltype(&iname, &jname, &kname, &lname, &owned);
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
    let param_names: HashMap<&str, &[&str]> = [
        ("lj12-6", &["c6", "c12"][..]),
        ("buckingham", &["A", "B", "C"][..]),
    ]
    .into_iter()
    .collect();

    let _ = ff.def_pairstyle("", &[]);
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
            .map(|(n, v)| ((*n).to_string(), *v))
            .collect();
        let owned: Vec<(&str, f64)> = converted.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let iname = atom_name_at(atom_names, i)?;
        let jname = atom_name_at(atom_names, j)?;
        ff.get_style_mut("pair", "")
            .ok_or("pair style missing")?
            .def_pairtype(&iname, Some(&jname), &owned);
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
}
