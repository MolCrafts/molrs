//! AMBER prmtop force-field reader.
//!
//! Builds a molrs [`ForceField`] from the parameter tables of an AMBER
//! topology (prmtop / parm7). Structure/connectivity lives in
//! [`crate::io::data::prmtop`]; this module owns styles + type params only.
//!
//! # Amber FileFormats (I/O) vs LAMMPS/molrs potentials
//!
//! **Parsing** follows <https://ambermd.org/FileFormats.php> (and the
//! expanded Swails prmtop appendix for section layout). **Stored potentials**
//! use the same form map as the LAMMPS force-field boundary:
//!
//! | Term | Amber prmtop storage | LAMMPS file form | molrs store |
//! |------|----------------------|------------------|-------------|
//! | Bond | `RK` in `E = RK·(r−r₀)²` (no ½) | `bond_style harmonic` same | `k = 2·RK`, `E = ½k(r−r₀)²` |
//! | Angle | `TK` in `E = TK·(θ−θ₀)²` (no ½), `θ₀` rad | `angle_style harmonic` same | `k = 2·TK`, `θ₀` rad |
//! | Dihedral | `PK·[1 + cos(nφ − δ)]`, `δ` rad | `dihedral_style fourier` | `k/n/d` as-is (`d` rad) |
//! | Improper | same form; 4th pointer negative | `improper_style periodic` | `k/n/d` as-is |
//! | LJ | `A/r¹² − B/r⁶` via ICO | `lj/cut` σ/ε | `σ = 2^{−1/6} r_min`, `ε = B²/(4A)` |
//! | 1-4 scales | `SCEE`/`SCNB` divisors (default 1.2 / 2.0) | `special_bonds amber` | `coul_14 = 1/SCEE`, `lj_14 = 1/SCNB` |
//!
//! Notes:
//! - OpenMM multiplies Amber bond/angle `RK`/`TK` by 2 when loading into a ½k
//!   kernel — same as our bond/angle map. Dihedral `PK` is **not** doubled
//!   (matches LAMMPS fourier / molrs periodic).
//! - Swails’ appendix writes bond/angle as ½k and torsion as `k cos(…)`; those
//!   equations disagree with Amber parameter files, OpenMM’s converter, and
//!   the FileFormats `parm.dat` section. We follow FileFormats + OpenMM.
//! - `%COMMENT` lines are skipped. Section order is free (flag map).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, Error, ErrorKind};
use std::path::Path;

use super::ForceFieldReader;
use crate::ff::forcefield::{ForceField, SpecialBonds};

/// Default AMBER 1-4 Coulomb divisor (`SCEE`) when the section is absent.
const DEFAULT_SCEE: f64 = 1.2;
/// Default AMBER 1-4 VDW divisor (`SCNB`) when the section is absent.
const DEFAULT_SCNB: f64 = 2.0;

/// Reader for AMBER prmtop force-field parameter tables.
#[derive(Debug, Default, Clone, Copy)]
pub struct AmberPrmtopFfReader;

impl AmberPrmtopFfReader {
    pub fn new() -> Self {
        Self
    }
}

impl ForceFieldReader for AmberPrmtopFfReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        let sections = parse_flag_sections(text.as_bytes()).map_err(|e| e.to_string())?;
        build_forcefield(&sections)
    }

    fn read(&self, path: &str) -> Result<ForceField, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        self.read_str(&text)
    }
}

/// Convenience: read prmtop path → ForceField.
pub fn read_amber_prmtop_ff(path: impl AsRef<Path>) -> Result<ForceField, String> {
    let text = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("read {}: {e}", path.as_ref().display()))?;
    AmberPrmtopFfReader::new().read_str(&text)
}

// ---------------------------------------------------------------------------
// Section parse
// ---------------------------------------------------------------------------

fn parse_flag_sections<R: BufRead>(mut reader: R) -> std::io::Result<HashMap<String, Vec<String>>> {
    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut flag: Option<String> = None;
    let mut data: Vec<String> = Vec::new();
    let mut buf = String::new();

    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("%FLAG") {
            if let Some(f) = flag.take() {
                sections
                    .entry(f)
                    .or_default()
                    .extend(std::mem::take(&mut data));
            }
            let name = line
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| Error::new(ErrorKind::InvalidData, "malformed %FLAG"))?
                .to_string();
            flag = Some(name);
            data = Vec::new();
        } else if line.starts_with("%FORMAT")
            || line.starts_with("%VERSION")
            || line.starts_with("%COMMENT")
        {
            // ignore
        } else {
            data.push(line.to_string());
        }
    }
    if let Some(f) = flag {
        sections.entry(f).or_default().extend(data);
    }
    Ok(sections)
}

fn parse_tokens<T: std::str::FromStr>(lines: &[String]) -> Result<Vec<T>, String>
where
    T::Err: std::fmt::Display,
{
    let mut out = Vec::new();
    for line in lines {
        for tok in line.split_whitespace() {
            out.push(
                tok.parse::<T>()
                    .map_err(|e| format!("token {tok:?}: {e}"))?,
            );
        }
    }
    Ok(out)
}

fn read_a4_names(lines: &[String]) -> Vec<String> {
    let mut names = Vec::new();
    for line in lines {
        let mut i = 0;
        while i < line.len() {
            let end = (i + 4).min(line.len());
            names.push(line[i..end].trim().to_string());
            i += 4;
        }
    }
    names
}

fn section_f64(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Vec<f64>, String> {
    match sections.get(key) {
        Some(lines) => parse_tokens(lines),
        None => Ok(Vec::new()),
    }
}

fn section_i64(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Vec<i64>, String> {
    match sections.get(key) {
        Some(lines) => parse_tokens(lines),
        None => Ok(Vec::new()),
    }
}

/// Pick a representative non-zero SCEE/SCNB divisor (Amber default if absent).
fn scale_divisor(values: &[f64], default: f64) -> f64 {
    values.iter().copied().find(|&v| v > 0.0).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Build ForceField
// ---------------------------------------------------------------------------

fn build_forcefield(sections: &HashMap<String, Vec<String>>) -> Result<ForceField, String> {
    let pointers = sections
        .get("POINTERS")
        .ok_or_else(|| "POINTERS section missing".to_string())?;
    let ptr_vals: Vec<i64> = parse_tokens(pointers)?;
    // NATOM, NTYPES, … — first two always present when POINTERS is valid.
    let n_atom = *ptr_vals.first().unwrap_or(&0) as usize;
    let n_types = *ptr_vals.get(1).unwrap_or(&0) as usize;

    let mut atom_types = sections
        .get("AMBER_ATOM_TYPE")
        .map(|l| read_a4_names(l))
        .unwrap_or_default();
    if atom_types.len() > n_atom {
        atom_types.truncate(n_atom);
    }
    while atom_types.len() < n_atom {
        atom_types.push(String::new());
    }

    let masses: Vec<f64> = sections
        .get("MASS")
        .map(|l| parse_tokens(l))
        .transpose()?
        .unwrap_or_default();
    let type_index: Vec<i64> = section_i64(sections, "ATOM_TYPE_INDEX")?;

    // 1-4 scales: FileFormats stores SCEE/SCNB as *divisors* (default 1.2 / 2.0).
    // molrs SpecialBonds wants multiplicative 1-4 weights.
    let scee = section_f64(sections, "SCEE_SCALE_FACTOR")?;
    let scnb = section_f64(sections, "SCNB_SCALE_FACTOR")?;
    let scee_div = scale_divisor(&scee, DEFAULT_SCEE);
    let scnb_div = scale_divisor(&scnb, DEFAULT_SCNB);

    let mut ff = ForceField::new("AMBER");
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 1.0 / scnb_div],
        coul: [0.0, 0.0, 1.0 / scee_div],
    });

    // Atom types (unique by name; id from ATOM_TYPE_INDEX).
    {
        let style = ff.def_atomstyle("full");
        let mut seen: HashSet<String> = HashSet::new();
        for (i, name) in atom_types.iter().enumerate() {
            if name.is_empty() || !seen.insert(name.clone()) {
                continue;
            }
            let id = type_index.get(i).copied().unwrap_or((i + 1) as i64) as f64;
            let mass = masses.get(i).copied().unwrap_or(0.0);
            style.def_atomtype(name, &[("id", id), ("mass", mass)]);
        }
    }

    // Bonds: unique by sorted endpoint type names; k = 2·RK (LAMMPS→molrs map).
    let bond_k = section_f64(sections, "BOND_FORCE_CONSTANT")?;
    let bond_r0 = section_f64(sections, "BOND_EQUIL_VALUE")?;
    let mut bond_ptrs = section_i64(sections, "BONDS_INC_HYDROGEN")?;
    bond_ptrs.extend(section_i64(sections, "BONDS_WITHOUT_HYDROGEN")?);
    {
        let style = ff.def_bondstyle("harmonic");
        let mut seen: HashSet<String> = HashSet::new();
        for chunk in bond_ptrs.as_chunks::<3>().0 {
            let a = chunk[0];
            let b = chunk[1];
            if a < 0 || b < 0 {
                return Err(format!("negative bonded atom pointers ({a}, {b})"));
            }
            let i = (a / 3) as usize;
            let j = (b / 3) as usize;
            let tid = (chunk[2] - 1) as usize;
            let mut ends = [
                atom_types.get(i).cloned().unwrap_or_default(),
                atom_types.get(j).cloned().unwrap_or_default(),
            ];
            ends.sort();
            let name = format!("{}-{}", ends[0], ends[1]);
            if !seen.insert(name.clone()) {
                continue;
            }
            let k = 2.0 * bond_k.get(tid).copied().unwrap_or(0.0);
            let r0 = bond_r0.get(tid).copied().unwrap_or(0.0);
            style.def_bondtype(
                &ends[0],
                &ends[1],
                &[("k", k), ("r0", r0), ("id", (tid + 1) as f64)],
            );
        }
    }

    // Angles: k = 2·TK, theta0 already radians in prmtop.
    let angle_k = section_f64(sections, "ANGLE_FORCE_CONSTANT")?;
    let angle_eq = section_f64(sections, "ANGLE_EQUIL_VALUE")?;
    let mut angle_ptrs = section_i64(sections, "ANGLES_INC_HYDROGEN")?;
    angle_ptrs.extend(section_i64(sections, "ANGLES_WITHOUT_HYDROGEN")?);
    {
        let style = ff.def_anglestyle("harmonic");
        let mut seen: HashSet<String> = HashSet::new();
        for chunk in angle_ptrs.as_chunks::<4>().0 {
            let a = chunk[0];
            let b = chunk[1];
            let c = chunk[2];
            if a < 0 || b < 0 || c < 0 {
                return Err(format!("negative angle atom pointers ({a}, {b}, {c})"));
            }
            let i = (a / 3) as usize;
            let j = (b / 3) as usize;
            let k_idx = (c / 3) as usize;
            let tid = (chunk[3] - 1) as usize;
            let mut ends_ik = [
                atom_types.get(i).cloned().unwrap_or_default(),
                atom_types.get(k_idx).cloned().unwrap_or_default(),
            ];
            ends_ik.sort();
            let jname = atom_types.get(j).cloned().unwrap_or_default();
            let name = format!("{}-{}-{}", ends_ik[0], jname, ends_ik[1]);
            if !seen.insert(name) {
                continue;
            }
            let k = 2.0 * angle_k.get(tid).copied().unwrap_or(0.0);
            let theta0 = angle_eq.get(tid).copied().unwrap_or(0.0);
            style.def_angletype(
                &ends_ik[0],
                &jname,
                &ends_ik[1],
                &[("k", k), ("theta0", theta0), ("id", (tid + 1) as f64)],
            );
        }
    }

    // Dihedrals / impropers.
    // FileFormats: 3rd pointer negative → ignore 1-4; 4th negative → improper.
    // Potential: PK * [1 + cos(n·φ − phase)], phase in radians — matches LAMMPS
    // fourier / improper_style periodic (no form factor on K).
    let dih_k = section_f64(sections, "DIHEDRAL_FORCE_CONSTANT")?;
    let dih_phase = section_f64(sections, "DIHEDRAL_PHASE")?;
    let dih_per = section_f64(sections, "DIHEDRAL_PERIODICITY")?;
    let mut dih_ptrs = section_i64(sections, "DIHEDRALS_INC_HYDROGEN")?;
    dih_ptrs.extend(section_i64(sections, "DIHEDRALS_WITHOUT_HYDROGEN")?);

    // name → (handles, tid → (k, n, d))
    type DihedralTerm = (f64, f64, f64);
    type DihedralEntry = ([String; 4], BTreeMap<i64, DihedralTerm>);
    let mut proper: BTreeMap<String, DihedralEntry> = BTreeMap::new();
    let mut improper: BTreeMap<String, DihedralEntry> = BTreeMap::new();

    for chunk in dih_ptrs.as_chunks::<5>().0 {
        let a = chunk[0];
        let b = chunk[1];
        if a < 0 || b < 0 {
            return Err(format!("negative dihedral atom pointers ({a}, {b})"));
        }
        let i = (a / 3) as usize;
        let j = (b / 3) as usize;
        let k_raw = chunk[2];
        let l_raw = chunk[3];
        let is_improper = l_raw < 0;
        let k_idx = (k_raw.unsigned_abs() as usize) / 3;
        let l = (l_raw.unsigned_abs() as usize) / 3;
        let tid = chunk[4];

        let mut i_name = atom_types.get(i).cloned().unwrap_or_default();
        let mut j_name = atom_types.get(j).cloned().unwrap_or_default();
        let mut k_name = atom_types.get(k_idx).cloned().unwrap_or_default();
        let mut l_name = atom_types.get(l).cloned().unwrap_or_default();
        if j_name > k_name {
            std::mem::swap(&mut j_name, &mut k_name);
            std::mem::swap(&mut i_name, &mut l_name);
        }
        let name = format!("{i_name}-{j_name}-{k_name}-{l_name}");
        let table = if is_improper {
            &mut improper
        } else {
            &mut proper
        };
        let entry = table
            .entry(name)
            .or_insert_with(|| ([i_name, j_name, k_name, l_name], BTreeMap::new()));

        // Multiterm: FileFormats — negative PN means the *next* PK/PN/PHASE
        // entries continue this torsion until a positive PN is seen.
        for term_tid in expand_multiterm_tids(tid, &dih_per) {
            if entry.1.contains_key(&term_tid) {
                continue;
            }
            let idx = (term_tid - 1) as usize;
            let k = dih_k.get(idx).copied().unwrap_or(0.0);
            let n = dih_per.get(idx).copied().unwrap_or(0.0).abs();
            let d = dih_phase.get(idx).copied().unwrap_or(0.0);
            entry.1.insert(term_tid, (k, n, d));
        }
    }

    {
        let style = ff.def_dihedralstyle("fourier");
        for (handles, terms) in proper.values() {
            let owned = terms_to_fourier_params(terms);
            let refs: Vec<(&str, f64)> = owned.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            style.def_dihedraltype(&handles[0], &handles[1], &handles[2], &handles[3], &refs);
        }
    }

    if !improper.is_empty() {
        let style = ff.def_improperstyle("periodic");
        for (handles, terms) in improper.values() {
            // Improper periodic kernel is single-term; take first Fourier term.
            let (k, n, d) = terms.values().next().copied().unwrap_or((0.0, 0.0, 0.0));
            style.def_impropertype(
                &handles[0],
                &handles[1],
                &handles[2],
                &handles[3],
                &[("k", k), ("n", n), ("d", d)],
            );
        }
    }

    // Pair LJ from A/B diagonal of NONBONDED matrix.
    // FileFormats: index = ICO[NTYPES*(IAC(i)-1) + IAC(j)] (1-based Fortran).
    // Diagonal i=j: 0-based ICO index = (NTYPES+1)*(IAC-1).
    let acoef = section_f64(sections, "LENNARD_JONES_ACOEF")?;
    let bcoef = section_f64(sections, "LENNARD_JONES_BCOEF")?;
    let nb_index = section_i64(sections, "NONBONDED_PARM_INDEX")?;
    let hbond_a = sections.get("HBOND_ACOEF").cloned().unwrap_or_default();
    let hbond_b = sections.get("HBOND_BCOEF").cloned().unwrap_or_default();
    for line in hbond_a.iter().chain(hbond_b.iter()) {
        for tok in line.split_whitespace() {
            if let Ok(v) = tok.parse::<f64>()
                && v != 0.0
            {
                return Err("10-12 interactions are not supported".into());
            }
        }
    }

    {
        let style = ff.def_pairstyle(
            "lj/cut/coul/long",
            &[("cutoff_lj", 9.0), ("cutoff_coul", 10.0)],
        );
        let mut seen: HashSet<String> = HashSet::new();
        for i_atom in 0..n_atom {
            let itype = type_index.get(i_atom).copied().unwrap_or(1);
            // 0-based into ICO for self-type: NTYPES*(IAC-1)+(IAC-1)
            let index = n_types
                .saturating_mul(itype as usize - 1)
                .saturating_add(itype as usize - 1);
            let nb = *nb_index.get(index).unwrap_or(&0);
            if nb < 0 {
                return Err("10-12 interactions are not supported".into());
            }
            if nb == 0 {
                continue;
            }
            let nb_idx = (nb - 1) as usize;
            let a = acoef.get(nb_idx).copied().unwrap_or(0.0);
            let b = bcoef.get(nb_idx).copied().unwrap_or(0.0);
            let (sigma, epsilon) = if a == 0.0 || b == 0.0 {
                (1.0, 0.0)
            } else {
                // E = A/r^12 − B/r^6 → r_min = (2A/B)^{1/6}, ε = B²/(4A)
                let r_min = (2.0 * a / b).powf(1.0 / 6.0);
                let eps = 0.25 * b * b / a;
                let sigma = 2f64.powf(-1.0 / 6.0) * r_min;
                (sigma, eps)
            };
            let tname = atom_types.get(i_atom).cloned().unwrap_or_default();
            if tname.is_empty() || !seen.insert(tname.clone()) {
                continue;
            }
            style.def_pairtype(&tname, None, &[("epsilon", epsilon), ("sigma", sigma)]);
        }
    }

    Ok(ff)
}

/// Expand a 1-based dihedral type id through negative-periodicity multiterms.
fn expand_multiterm_tids(tid: i64, periods: &[f64]) -> Vec<i64> {
    let mut out = Vec::new();
    let mut t = tid;
    loop {
        out.push(t);
        let idx = (t - 1) as usize;
        let pn = periods.get(idx).copied().unwrap_or(0.0);
        if pn >= 0.0 {
            break;
        }
        t += 1;
        if t as usize > periods.len() {
            break;
        }
    }
    out
}

fn terms_to_fourier_params(terms: &BTreeMap<i64, (f64, f64, f64)>) -> Vec<(String, f64)> {
    let mut params = Vec::new();
    for (m, (_tid, (k, n, d))) in terms.iter().enumerate() {
        let i = m + 1;
        params.push((format!("k{i}"), *k));
        params.push((format!("n{i}"), *n));
        params.push((format!("d{i}"), *d));
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_prmtop_errors() {
        let r = AmberPrmtopFfReader::new().read_str("%VERSION 1\n");
        assert!(r.is_err());
    }

    #[test]
    fn multiterm_expansion() {
        // type 1 has PN=-2 → continues to type 2 (PN=+3)
        let periods = vec![-2.0, 3.0];
        assert_eq!(expand_multiterm_tids(1, &periods), vec![1, 2]);
        assert_eq!(expand_multiterm_tids(2, &periods), vec![2]);
    }

    #[test]
    fn comment_lines_ignored() {
        // Minimal broken POINTERS after a comment — parser must not treat
        // %COMMENT as a numeric token.
        let text = "\
%VERSION VERSION_STAMP = V0001.000
%FLAG POINTERS
%COMMENT ignore me
%FORMAT(10I8)
       0       0
";
        // Missing sections → may fail later, but not on the comment token.
        let sections = parse_flag_sections(text.as_bytes()).unwrap();
        let vals: Vec<i64> = parse_tokens(sections.get("POINTERS").unwrap()).unwrap();
        assert_eq!(vals, vec![0, 0]);
    }
}
