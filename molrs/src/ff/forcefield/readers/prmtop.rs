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

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use super::ForceFieldReader;
use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use crate::ff::params::amber::{AMBER_COULOMB, AMBER_SCEE, AMBER_SCNB};
use crate::io::data::prmtop::parse_flag_sections;
use crate::io::data::prmtop_tables::{parse_a4_names, parse_tokens};
use crate::math::pair_form::lj_ab_to_sigma_epsilon;
use molrs::store::type_labels::TypeName;

/// `(type_name, sigma_Å, epsilon_kcal_per_mol)` for one self LJ type.
type LjSelfRow = (String, f64, f64);

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

/// Uniform SCEE/SCNB divisor over non-suppressed proper torsion type ids.
///
/// Empty `values` (section absent) → `default`. A used id out of range or a
/// non-positive value is an `Err` naming `flag` and the id. Two used values
/// differing by more than 1e-6 relative is an `Err` naming `flag` and both
/// values.
fn uniform_divisor(
    values: &[f64],
    used_tids: &[i64],
    flag: &str,
    default: f64,
) -> Result<f64, String> {
    if values.is_empty() {
        return Ok(default);
    }
    let mut chosen: Option<f64> = None;
    for &tid in used_tids {
        let idx = (tid - 1) as usize;
        let Some(&v) = values.get(idx) else {
            return Err(format!("{flag} type {tid} is out of range"));
        };
        if v <= 0.0 {
            return Err(format!("{flag} type {tid} has non-positive divisor {v}"));
        }
        match chosen {
            None => chosen = Some(v),
            Some(prev) => {
                let denom = prev.abs().max(v.abs());
                if denom > 0.0 && (prev - v).abs() / denom > 1e-6 {
                    return Err(format!("{flag} is not uniform: {prev:?} vs {v:?}"));
                }
            }
        }
    }
    Ok(chosen.unwrap_or(default))
}

fn ico_entry(n_types: usize, iac_i: usize, iac_j: usize, nb_index: &[i64]) -> Result<i64, String> {
    let index = n_types
        .saturating_mul(iac_i.saturating_sub(1))
        .saturating_add(iac_j.saturating_sub(1));
    let nb = *nb_index.get(index).unwrap_or(&0);
    if nb < 0 {
        return Err("10-12 interactions are not supported".into());
    }
    Ok(nb)
}

fn first_name_for_type(atom_types: &[String], type_index: &[i64], itype: i64) -> String {
    for (i, name) in atom_types.iter().enumerate() {
        if type_index.get(i).copied().unwrap_or(0) == itype && !name.is_empty() {
            return name.clone();
        }
    }
    format!("type{itype}")
}

/// Full-ICO LJ decode: one self term per atom, in atom order; off-diagonal
/// entries must match Lorentz–Berthelot or the topology is refused.
///
/// Atoms sharing a type name share its LJ class (the atom-type definition
/// makes a second class under one name a `TypeConflict`), so their rows are
/// equal and the definition rule collapses them to one pair type.
fn decode_lj_types(
    n_types: usize,
    atom_types: &[String],
    type_index: &[i64],
    nb_index: &[i64],
    acoef: &[f64],
    bcoef: &[f64],
) -> Result<Vec<LjSelfRow>, String> {
    if n_types == 0 {
        return Ok(Vec::new());
    }
    let mut self_params = vec![(1.0, 0.0); n_types];
    for t in 1..=n_types {
        let nb = ico_entry(n_types, t, t, nb_index)?;
        if nb == 0 {
            continue;
        }
        let idx = (nb - 1) as usize;
        self_params[t - 1] = lj_ab_to_sigma_epsilon(
            acoef.get(idx).copied().unwrap_or(0.0),
            bcoef.get(idx).copied().unwrap_or(0.0),
        );
    }
    for i in 1..=n_types {
        for j in (i + 1)..=n_types {
            let nb = ico_entry(n_types, i, j, nb_index)?;
            if nb == 0 {
                continue;
            }
            let idx = (nb - 1) as usize;
            let (sigma, eps) = lj_ab_to_sigma_epsilon(
                acoef.get(idx).copied().unwrap_or(0.0),
                bcoef.get(idx).copied().unwrap_or(0.0),
            );
            let (si, ei) = self_params[i - 1];
            let (sj, ej) = self_params[j - 1];
            let lb_s = 0.5 * (si + sj);
            let lb_e = (ei * ej).sqrt();
            let rel = |got: f64, exp: f64| {
                let denom = got.abs().max(exp.abs());
                if denom == 0.0 {
                    0.0
                } else {
                    (got - exp).abs() / denom
                }
            };
            if rel(sigma, lb_s) > 1e-6 || rel(eps, lb_e) > 1e-6 {
                let ni = first_name_for_type(atom_types, type_index, i as i64);
                let nj = first_name_for_type(atom_types, type_index, j as i64);
                return Err(format!(
                    "off-diagonal (NBFIX) Lennard-Jones pairs are not supported: {ni}-{nj} deviates from Lorentz-Berthelot"
                ));
            }
        }
    }
    let mut out = Vec::new();
    for (i, name) in atom_types.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        let itype = type_index.get(i).copied().unwrap_or(1) as usize;
        let (sigma, epsilon) = if itype == 0 || itype > n_types {
            (1.0, 0.0)
        } else {
            self_params[itype - 1]
        };
        out.push((name.clone(), sigma, epsilon));
    }
    Ok(out)
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
        .map(|l| parse_a4_names(l))
        .unwrap_or_default();
    if atom_types.len() > n_atom {
        atom_types.truncate(n_atom);
    }
    while atom_types.len() < n_atom {
        atom_types.push(String::new());
    }
    let type_name = |atom: usize| atom_types.get(atom).map(String::as_str).unwrap_or_default();

    let masses: Vec<f64> = sections
        .get("MASS")
        .map(|l| parse_tokens(l))
        .transpose()?
        .unwrap_or_default();
    let type_index: Vec<i64> = parse_tokens(
        sections
            .get("ATOM_TYPE_INDEX")
            .ok_or_else(|| "%FLAG ATOM_TYPE_INDEX section missing".to_string())?,
    )?;

    if sections.contains_key("LENNARD_JONES_CCOEF") {
        return Err("12-6-4 Lennard-Jones prmtop files are not supported".into());
    }

    // 1-4 scales: FileFormats stores SCEE/SCNB as *divisors*. Applied after
    // the dihedral walk so only non-suppressed proper type ids constrain
    // uniformity.
    let scee = section_f64(sections, "SCEE_SCALE_FACTOR")?;
    let scnb = section_f64(sections, "SCNB_SCALE_FACTOR")?;

    let mut ff = ForceField::new("AMBER");
    // prmtop stores Å, kcal/mol and e: LAMMPS `real`.
    ff.set_units("real");

    // Atom types: one per name; id from ATOM_TYPE_INDEX (the LJ class). Every
    // atom defines its type, so two atoms of one name with a different class
    // or mass are a TypeConflict.
    {
        let style = ff
            .def_style("atom", "full", Params::new())
            .map_err(|e| e.to_string())?;
        for (i, name) in atom_types.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let id = type_index
                .get(i)
                .copied()
                .ok_or_else(|| format!("ATOM_TYPE_INDEX has no entry for atom {}", i + 1))?
                as f64;
            let mass = masses.get(i).copied().unwrap_or(0.0);
            style
                .def_type(name, &[], Params::from_pairs(&[("id", id), ("mass", mass)]))
                .map_err(|e| e.to_string())?;
        }
    }

    // Bonds: one type per endpoint type pair in `TypeName::orient`'s spelling;
    // k = 2·RK (LAMMPS→molrs map). Every bond row defines its type, so two
    // rows under one name with different k/r0 are a TypeConflict.
    let bond_k = section_f64(sections, "BOND_FORCE_CONSTANT")?;
    let bond_r0 = section_f64(sections, "BOND_EQUIL_VALUE")?;
    let mut bond_ptrs = section_i64(sections, "BONDS_INC_HYDROGEN")?;
    bond_ptrs.extend(section_i64(sections, "BONDS_WITHOUT_HYDROGEN")?);
    {
        let style = ff
            .def_style("bond", "harmonic", Params::new())
            .map_err(|e| e.to_string())?;
        for chunk in bond_ptrs.as_chunks::<3>().0 {
            let a = chunk[0];
            let b = chunk[1];
            if a < 0 || b < 0 {
                return Err(format!("negative bonded atom pointers ({a}, {b})"));
            }
            let i = (a / 3) as usize;
            let j = (b / 3) as usize;
            let tid = (chunk[2] - 1) as usize;
            let ends = TypeName::orient(&[type_name(i), type_name(j)]);
            let k = 2.0 * bond_k.get(tid).copied().unwrap_or(0.0);
            let r0 = bond_r0.get(tid).copied().unwrap_or(0.0);
            style
                .def_type(
                    TypeName::join(&ends)?.as_str(),
                    &ends,
                    Params::from_pairs(&[("k", k), ("r0", r0)]),
                )
                .map_err(|e| e.to_string())?;
        }
    }

    // Angles: k = 2·TK, theta0 already radians in prmtop. As for bonds, every
    // angle row defines its type.
    let angle_k = section_f64(sections, "ANGLE_FORCE_CONSTANT")?;
    let angle_eq = section_f64(sections, "ANGLE_EQUIL_VALUE")?;
    let mut angle_ptrs = section_i64(sections, "ANGLES_INC_HYDROGEN")?;
    angle_ptrs.extend(section_i64(sections, "ANGLES_WITHOUT_HYDROGEN")?);
    {
        let style = ff
            .def_style("angle", "harmonic", Params::new())
            .map_err(|e| e.to_string())?;
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
            let ends = TypeName::orient(&[type_name(i), type_name(j), type_name(k_idx)]);
            let k = 2.0 * angle_k.get(tid).copied().unwrap_or(0.0);
            let theta0 = angle_eq.get(tid).copied().unwrap_or(0.0);
            style
                .def_type(
                    TypeName::join(&ends)?.as_str(),
                    &ends,
                    Params::from_pairs(&[("k", k), ("theta0", theta0)]),
                )
                .map_err(|e| e.to_string())?;
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
    let mut used_tids: Vec<i64> = Vec::new();

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
        if is_improper && expand_multiterm_tids(tid, &dih_per).len() > 1 {
            return Err(format!("multi-term improper type {tid} is not supported"));
        }
        if !is_improper && k_raw >= 0 {
            used_tids.extend(expand_multiterm_tids(tid, &dih_per));
        }

        let mut quartet = [type_name(i), type_name(j), type_name(k_idx), type_name(l)];
        // A proper torsion reads the same backwards, so it is stored in one
        // orientation (`TypeName::orient`'s, the frame reader's too). An
        // improper does not: AMBER puts its central atom third, and reversing
        // it would move the centre to second place.
        if !is_improper && TypeName::reads_reversed(&quartet) {
            quartet.reverse();
        }
        let name = TypeName::join(&quartet)?.to_string();
        let table = if is_improper {
            &mut improper
        } else {
            &mut proper
        };
        let entry = table
            .entry(name)
            .or_insert_with(|| (quartet.map(str::to_owned), BTreeMap::new()));

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

    let scee_div = uniform_divisor(&scee, &used_tids, "SCEE_SCALE_FACTOR", AMBER_SCEE)?;
    let scnb_div = uniform_divisor(&scnb, &used_tids, "SCNB_SCALE_FACTOR", AMBER_SCNB)?;
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 1.0 / scnb_div],
        coul: [0.0, 0.0, 1.0 / scee_div],
    });

    {
        let style = ff
            .def_style("dihedral", "fourier", Params::new())
            .map_err(|e| e.to_string())?;
        for (handles, terms) in proper.values() {
            let owned = terms_to_fourier_params(terms);
            let refs: Vec<(&str, f64)> = owned.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            let ends = [&*handles[0], &*handles[1], &*handles[2], &*handles[3]];
            style
                .def_type(
                    TypeName::join(&ends)?.as_str(),
                    &ends,
                    Params::from_pairs(&refs),
                )
                .map_err(|e| e.to_string())?;
        }
    }

    if !improper.is_empty() {
        let style = ff
            .def_style("improper", "periodic", Params::new())
            .map_err(|e| e.to_string())?;
        for (handles, terms) in improper.values() {
            let (k, n, d) = terms.values().next().copied().unwrap_or((0.0, 0.0, 0.0));
            let ends = [&*handles[0], &*handles[1], &*handles[2], &*handles[3]];
            style
                .def_type(
                    TypeName::join(&ends)?.as_str(),
                    &ends,
                    Params::from_pairs(&[("k", k), ("periodicity", n), ("phase", d)]),
                )
                .map_err(|e| e.to_string())?;
        }
    }

    // Pair LJ from the full ICO matrix. FileFormats:
    // index = ICO[NTYPES*(IAC(i)-1) + IAC(j)] (1-based Fortran).
    // Neither pair style gets a `cutoff`: a prmtop carries none (it lives in
    // the mdin), so the caller declares it.
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

    let rows = decode_lj_types(n_types, &atom_types, &type_index, &nb_index, &acoef, &bcoef)?;
    {
        let style = ff
            .def_style("pair", "lj/cut", Params::new())
            .map_err(|e| e.to_string())?;
        for (tname, sigma, epsilon) in &rows {
            style
                .def_type(
                    tname,
                    &[tname],
                    Params::from_pairs(&[("epsilon", *epsilon), ("sigma", *sigma)]),
                )
                .map_err(|e| e.to_string())?;
        }
    }
    ff.def_style(
        "pair",
        "coul/cut",
        Params::from_pairs(&[
            ("coulomb", AMBER_COULOMB),
            ("dielectric", VACUUM_DIELECTRIC),
        ]),
    )
    .map_err(|e| e.to_string())?;

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
        params.push((format!("periodicity{i}"), *n));
        params.push((format!("phase{i}"), *d));
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

    /// 4-atom c3–c3–c3–hc chain, NTYPES=2.
    ///
    /// c3 self LJ from published GAFF R*=1.9080 Å, ε=0.1094 kcal/mol
    /// (`A=1043080.23`, `B=675.612248`). hc from GAFF R*=1.4870 Å, ε=0.0157.
    /// Off-diagonal ICO is Lorentz–Berthelot of the two self terms.
    /// Goldens are hand-derived from the A/B closed form; no AmberTools.
    const GAFF_MINI: &str = "\
%VERSION VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
GAFF_MINI
%FLAG POINTERS
%FORMAT(10I8)
       4       2       1       2       1       1       0       1       0       0
       0       1       2       1       1       2       2       2       2       0
       0       0       0       0       0       0       0       0       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
C1  C2  C3  H1
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.20100000E+01  1.20100000E+01  1.20100000E+01  1.00800000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
c3  c3  c3  hc
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1       1       2
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       2       2       3
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  3.00000000E+02  3.40000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  1.53500000E+00  1.09000000E+00
%FLAG ANGLE_FORCE_CONSTANT
%FORMAT(5E16.8)
  5.00000000E+01  4.00000000E+01
%FLAG ANGLE_EQUIL_VALUE
%FORMAT(5E16.8)
  1.91113553E+00  1.91113553E+00
%FLAG DIHEDRAL_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.50000000E-01  2.00000000E-01
%FLAG DIHEDRAL_PERIODICITY
%FORMAT(5E16.8)
  3.00000000E+00  2.00000000E+00
%FLAG DIHEDRAL_PHASE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG SCEE_SCALE_FACTOR
%FORMAT(5E16.8)
  1.20000000E+00  1.20000000E+00
%FLAG SCNB_SCALE_FACTOR
%FORMAT(5E16.8)
  2.00000000E+00  2.00000000E+00
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  1.04308023E+06  9.717081166135172E+04  7.516077034091E+03
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  6.75612248E+02  1.2691914994192737E+02  2.17257827878E+01
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       6       9       2
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       3       6       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
       3       6       9       2
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       9       1
";

    fn read_ff(text: &str) -> ForceField {
        AmberPrmtopFfReader::new()
            .read_str(text)
            .unwrap_or_else(|e| panic!("read_str: {e}"))
    }

    fn read_err(text: &str) -> String {
        AmberPrmtopFfReader::new()
            .read_str(text)
            .expect_err("expected Err from read_str")
    }

    fn without_flag(text: &str, flag: &str) -> String {
        let mut out = String::new();
        let mut skipping = false;
        for line in text.lines() {
            if line.starts_with("%FLAG") {
                skipping = line.split_whitespace().nth(1) == Some(flag);
            }
            if !skipping {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    fn rel_close(got: f64, expected: f64, tol: f64) -> bool {
        (got - expected).abs() <= tol * expected.abs()
    }

    #[test]
    fn pair_styles_are_registered_lj_cut_and_coul_cut() {
        let ff = read_ff(GAFF_MINI);
        assert!(
            ff.get_style("pair", "lj/cut").is_some(),
            "lj/cut pair style"
        );
        let coul = ff
            .get_style("pair", "coul/cut")
            .expect("coul/cut pair style");
        assert!(
            ff.get_style("pair", "lj/cut/coul/long").is_none(),
            "combined lj/cut/coul/long must not be registered"
        );

        let coulomb = coul.params.get("coulomb").expect("coulomb");
        let dielectric = coul.params.get("dielectric").expect("dielectric");
        assert!(
            (coulomb - 332.052_217_29).abs() < 1e-10,
            "coulomb={coulomb}"
        );
        assert!((dielectric - 1.0).abs() < 1e-12, "dielectric={dielectric}");

        for style in ff.get_styles("pair") {
            // A prmtop carries no cutoff (it lives in the mdin); the caller
            // declares one.
            assert!(
                style.params.get("cutoff").is_none(),
                "{} carries an invented cutoff",
                style.name
            );
            assert!(
                style.params.get("cutoff_lj").is_none(),
                "{} still has cutoff_lj",
                style.name
            );
            assert!(
                style.params.get("cutoff_coul").is_none(),
                "{} still has cutoff_coul",
                style.name
            );
        }
    }

    #[test]
    fn special_bonds_are_reciprocal_divisors() {
        let ff = read_ff(GAFF_MINI);
        let sb = ff.special_bonds();
        assert!((sb.lj[2] - 0.5).abs() < 1e-12, "lj_14={}", sb.lj[2]);
        assert!(
            (sb.coul[2] - 1.0 / 1.2).abs() < 1e-12,
            "coul_14={}",
            sb.coul[2]
        );
    }

    #[test]
    fn uniform_divisor_defaults_when_absent() {
        let text = without_flag(
            &without_flag(GAFF_MINI, "SCEE_SCALE_FACTOR"),
            "SCNB_SCALE_FACTOR",
        );
        let ff = read_ff(&text);
        let sb = ff.special_bonds();
        assert!((sb.lj[2] - 0.5).abs() < 1e-12, "lj_14={}", sb.lj[2]);
        assert!(
            (sb.coul[2] - 1.0 / 1.2).abs() < 1e-12,
            "coul_14={}",
            sb.coul[2]
        );
    }

    #[test]
    fn uniform_divisor_rejects_mixed_scee() {
        let text = GAFF_MINI
            .replace(
                "  1.20000000E+00  1.20000000E+00",
                "  1.20000000E+00  1.00000000E+00",
            )
            .replace(
                "       0       3       6       9       1",
                "       0       3       6       9       1       0       3       6       9       2",
            );
        let err = read_err(&text);
        assert!(
            err.contains("SCEE_SCALE_FACTOR"),
            "error should name the flag: {err}"
        );
        assert!(err.contains("1.2"), "error should name 1.2: {err}");
        assert!(err.contains("1.0"), "error should name 1.0: {err}");
    }

    #[test]
    fn uniform_divisor_ignores_suppressed_torsions() {
        let text = GAFF_MINI
            .replace(
                "  1.20000000E+00  1.20000000E+00",
                "  1.20000000E+00  1.00000000E+00",
            )
            .replace(
                "       0       3       6       9       1",
                "       0       3       6       9       1       0       3      -6       9       2",
            );
        let ff = read_ff(&text);
        let sb = ff.special_bonds();
        assert!((sb.lj[2] - 0.5).abs() < 1e-12, "lj_14={}", sb.lj[2]);
        assert!(
            (sb.coul[2] - 1.0 / 1.2).abs() < 1e-12,
            "coul_14={}",
            sb.coul[2]
        );
    }

    #[test]
    fn uniform_divisor_rejects_nonpositive() {
        let text = GAFF_MINI
            .replace(
                "  1.20000000E+00  1.20000000E+00",
                "  1.20000000E+00  0.00000000E+00",
            )
            .replace(
                "       0       3       6       9       1",
                "       0       3       6       9       1       0       3       6       9       2",
            );
        let err = read_err(&text);
        assert!(
            err.contains("SCEE_SCALE_FACTOR"),
            "error should name the flag: {err}"
        );
        assert!(err.contains("2"), "error should name used type id 2: {err}");
    }

    #[test]
    fn rejects_12_6_4() {
        let text = format!(
            "{GAFF_MINI}%FLAG LENNARD_JONES_CCOEF\n%FORMAT(5E16.8)\n  \
             1.00000000E+00  0.00000000E+00  0.00000000E+00\n"
        );
        let err = read_err(&text);
        assert!(err.contains("12-6-4"), "error should name 12-6-4: {err}");
    }

    #[test]
    fn an_improper_keeps_its_central_atom_third() {
        // i=C1 (c3), j=H1 (hc), k=C2 (c3, the centre), l=C3 (c3): j > k by
        // name, which reverses a proper but must not reverse an improper.
        let text = GAFF_MINI.replace(
            "       0       3       6       9       1",
            "       0       9       3      -6       1",
        );
        let ff = read_ff(&text);
        let style = ff
            .get_style("improper", "periodic")
            .expect("improper style");
        let rows = style.type_rows();
        let names: Vec<&str> = rows.iter().map(|(n, _, _)| n.as_ref()).collect();
        assert_eq!(names, vec!["c3-hc-c3-c3"]);
    }

    #[test]
    fn rejects_multiterm_improper() {
        let text = GAFF_MINI
            .replace(
                "       0       3       6       9       1",
                "       0       3       6      -9       1",
            )
            .replace(
                "  3.00000000E+00  2.00000000E+00",
                " -2.00000000E+00  3.00000000E+00",
            );
        let err = read_err(&text);
        assert!(
            err.contains("1"),
            "error should name improper type id 1: {err}"
        );
    }

    #[test]
    fn rejects_negative_ico() {
        let text = GAFF_MINI.replace(
            "       1       2       2       3",
            "       1      -1       2       3",
        );
        let err = read_err(&text);
        assert!(
            err.contains("10-12 interactions are not supported"),
            "{err}"
        );
    }

    #[test]
    fn rejects_nonzero_hbond() {
        let text = format!("{GAFF_MINI}%FLAG HBOND_ACOEF\n%FORMAT(5E16.8)\n  1.00000000E+00\n");
        let err = read_err(&text);
        assert!(
            err.contains("10-12 interactions are not supported"),
            "{err}"
        );
    }

    #[test]
    fn decode_lj_types_self_terms_match_closed_form() {
        let ff = read_ff(GAFF_MINI);
        let lj = ff.get_style("pair", "lj/cut").expect("lj/cut pair style");
        let pt = lj.get_pairtype("c3", None).expect("c3 self pair type");
        let eps = pt.params.get("epsilon").expect("epsilon");
        let sigma = pt.params.get("sigma").expect("sigma");
        assert!(rel_close(eps, 0.109400, 1e-6), "epsilon={eps} vs 0.109400");
        assert!(
            rel_close(sigma, 3.3996695, 1e-6),
            "sigma={sigma} vs 3.3996695"
        );
    }

    #[test]
    fn decode_lj_types_refuses_nbfix_cross_terms() {
        let nbfix = GAFF_MINI
            .replace(
                "  1.04308023E+06  9.717081166135172E+04  7.516077034091E+03",
                "  1.04308023E+06  9.814251977796524E+04  7.516077034091E+03",
            )
            .replace(
                "  6.75612248E+02  1.2691914994192737E+02  2.17257827878E+01",
                "  6.75612248E+02  1.281883414413926E+02  2.17257827878E+01",
            );
        let err = read_err(&nbfix);
        assert!(err.contains("c3"), "error should name c3: {err}");
        assert!(err.contains("hc"), "error should name hc: {err}");
        assert!(
            err.contains("not supported"),
            "error should say not supported: {err}"
        );

        let ff = read_ff(GAFF_MINI);
        let lj = ff.get_style("pair", "lj/cut").expect("lj/cut pair style");
        assert!(
            lj.get_pairtype("c3", Some("hc")).is_none(),
            "LB-consistent cross must not emit a two-endpoint pair type"
        );
        for pt in ff.get_pairtypes() {
            assert_eq!(
                pt.itom, pt.jtom,
                "unexpected cross pair type {}-{}",
                pt.itom, pt.jtom
            );
        }
    }

    /// `text` with the single occurrence of `from` replaced by `to`.
    fn replaced_once(text: &str, from: &str, to: &str) -> String {
        assert_eq!(
            text.matches(from).count(),
            1,
            "fixture edit must be unique: {from:?}"
        );
        text.replacen(from, to, 1)
    }

    /// [`GAFF_MINI`] with a third bond-parameter row equal to row 1
    /// (`RK = 300`, `r₀ = 1.535`), and the second c3–c3 bond pointing at it.
    /// Two table rows, one parameter set, one type name.
    fn gaff_mini_with_equal_bond_rows_under_one_name() -> String {
        let text = replaced_once(
            GAFF_MINI,
            "  3.00000000E+02  3.40000000E+02\n",
            "  3.00000000E+02  3.40000000E+02  3.00000000E+02\n",
        );
        let text = replaced_once(
            &text,
            "  1.53500000E+00  1.09000000E+00\n",
            "  1.53500000E+00  1.09000000E+00  1.53500000E+00\n",
        );
        replaced_once(
            &text,
            "       0       3       1       3       6       1\n",
            "       0       3       1       3       6       3\n",
        )
    }

    #[test]
    fn two_bond_rows_with_equal_k_and_r0_under_one_name_define_one_type() {
        let ff = read_ff(&gaff_mini_with_equal_bond_rows_under_one_name());
        let c3c3: Vec<_> = ff
            .get_bondtypes()
            .into_iter()
            .filter(|t| t.name == "c3-c3")
            .collect();
        assert_eq!(c3c3.len(), 1);
        // k = 2·RK
        assert_eq!(c3c3[0].params.get("k"), Some(600.0));
        assert_eq!(c3c3[0].params.get("r0"), Some(1.535));
    }

    /// The second c3–c3 bond points at row 2 (`RK = 340`, `r₀ = 1.09`): two
    /// parameter sets under one type name is a conflict, not a silent
    /// first-wins drop.
    #[test]
    fn two_bond_rows_with_unequal_params_under_one_name_are_an_error() {
        let text = replaced_once(
            GAFF_MINI,
            "       0       3       1       3       6       1\n",
            "       0       3       1       3       6       2\n",
        );
        let err = read_err(&text);
        assert!(err.contains("c3-c3"), "error should name the type: {err}");
    }

    /// No `%FLAG ATOM_TYPE_INDEX` at all is a malformed file, reported as a
    /// missing section, not as a per-atom gap.
    #[test]
    fn absent_atom_type_index_section_is_reported_missing() {
        let err = read_err(&without_flag(GAFF_MINI, "ATOM_TYPE_INDEX"));
        assert!(
            err.contains("%FLAG ATOM_TYPE_INDEX"),
            "error should name the section: {err}"
        );
        assert!(err.contains("missing"), "error should say missing: {err}");
    }

    /// A present `ATOM_TYPE_INDEX` with fewer entries than atoms names the
    /// first atom lacking an entry, and is a different error from the absent
    /// section.
    #[test]
    fn short_atom_type_index_names_the_atom_without_entry() {
        let text = replaced_once(
            GAFF_MINI,
            "%FLAG ATOM_TYPE_INDEX\n%FORMAT(10I8)\n       1       1       1       2\n",
            "%FLAG ATOM_TYPE_INDEX\n%FORMAT(10I8)\n       1       1       1\n",
        );
        let err = read_err(&text);
        assert!(err.contains("atom 4"), "error should name atom 4: {err}");

        let missing = read_err(&without_flag(GAFF_MINI, "ATOM_TYPE_INDEX"));
        assert_ne!(
            err, missing,
            "short section and absent section must be distinct errors"
        );
    }

    /// The bond table row number is file layout, not a parameter.
    #[test]
    fn bond_types_carry_no_row_id() {
        let ff = read_ff(GAFF_MINI);
        let bonds = ff.get_bondtypes();
        assert!(!bonds.is_empty());
        for bt in bonds {
            assert_eq!(bt.params.get("id"), None, "{} carries an id", bt.name);
        }
    }

    /// The angle table row number is file layout, not a parameter.
    #[test]
    fn angle_types_carry_no_row_id() {
        let ff = read_ff(GAFF_MINI);
        let angles = ff.get_angletypes();
        assert!(!angles.is_empty());
        for at in angles {
            assert_eq!(at.params.get("id"), None, "{} carries an id", at.name);
        }
    }

    #[test]
    fn amber_coulomb_is_18_2223_squared() {
        use crate::ff::params::amber::{AMBER_COULOMB, AMBER_SCEE, AMBER_SCNB};
        use molrs::units::constants::COULOMB_REAL;

        assert!(
            (AMBER_COULOMB - 18.2223_f64.powi(2)).abs() < 1e-9,
            "AMBER_COULOMB={AMBER_COULOMB}"
        );
        let rel = (COULOMB_REAL - AMBER_COULOMB) / COULOMB_REAL;
        assert!(
            (rel - 3.4610e-5).abs() < 1e-8,
            "CODATA relative offset={rel}"
        );
        assert!((1.0 / AMBER_SCEE - 1.0 / 1.2).abs() < 1e-12);
        assert!((1.0 / AMBER_SCNB - 0.5).abs() < 1e-12);
    }
}
