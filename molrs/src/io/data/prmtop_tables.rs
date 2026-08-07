//! Amber prmtop **parameter-table** decode helpers.
//!
//! These mirror the historical molpy `AmberPrmtopReader` helper surface used by
//! tests and AmberTools workflows: bond/angle/dihedral rows and per-atom LJ
//! σ/ε, plus POINTERS / 20a4 name parsing. They operate on already-tokenized
//! arrays (not the full file) so call sites can inject tables in unit tests.
//!
//! Atom indexes in returned bond/angle/dihedral rows are **1-based** (Amber /
//! historical molpy helper contract). Structure [`Frame`](molrs::store::frame::Frame)
//! connectivity remains 0-based.

use std::collections::HashMap;

/// Parse POINTERS lines into the historical molpy meta map.
///
/// Includes both raw Amber fields (`NATOM`, …) and derived counts
/// (`n_atoms`, `n_bonds`, …). Accepts 30- or 31-value POINTERS (NCOPY optional).
pub fn parse_pointers(lines: &[String]) -> Result<HashMap<String, i64>, String> {
    let values: Vec<i64> = parse_tokens(lines)?;
    const FIELDS: &[&str] = &[
        "NATOM", "NTYPES", "NBONH", "MBONA", "NTHETH", "MTHETA", "NPHIH", "MPHIA", "NHPARM",
        "NPARM", "NNB", "NRES", "NBONA", "NTHETA", "NPHIA", "NUMBND", "NUMANG", "NPTRA", "NATYP",
        "NPHB", "IFPERT", "NBPER", "NGPER", "NDPER", "MBPER", "MGPER", "MDPER", "IFBOX", "NMXRS",
        "IFCAP", "NUMEXTRA", "NCOPY",
    ];
    let mut meta_data: HashMap<String, i64> = HashMap::new();
    for (name, val) in FIELDS.iter().zip(values.iter()) {
        meta_data.insert((*name).to_string(), *val);
    }
    // Graceful short POINTERS: missing keys stay absent; derived use unwrap_or(0).
    let natom = *meta_data.get("NATOM").unwrap_or(&0);
    let nbonh = *meta_data.get("NBONH").unwrap_or(&0);
    let mbona = *meta_data.get("MBONA").unwrap_or(&0);
    let ntheth = *meta_data.get("NTHETH").unwrap_or(&0);
    let mtheta = *meta_data.get("MTHETA").unwrap_or(&0);
    let nphih = *meta_data.get("NPHIH").unwrap_or(&0);
    let mphia = *meta_data.get("MPHIA").unwrap_or(&0);
    let natyp = *meta_data.get("NATYP").unwrap_or(&0);
    let numbnd = *meta_data.get("NUMBND").unwrap_or(&0);
    let numang = *meta_data.get("NUMANG").unwrap_or(&0);
    let nptra = *meta_data.get("NPTRA").unwrap_or(&0);

    let mut meta = meta_data;
    meta.insert("n_atoms".into(), natom);
    meta.insert("n_bonds".into(), nbonh + mbona);
    meta.insert("n_angles".into(), ntheth + mtheta);
    meta.insert("n_dihedrals".into(), nphih + mphia);
    meta.insert("n_atomtypes".into(), natyp);
    meta.insert("n_bondtypes".into(), numbnd);
    meta.insert("n_angletypes".into(), numang);
    meta.insert("n_dihedraltypes".into(), nptra);
    Ok(meta)
}

/// Fortran `20a4` name fields (strip each 4-char window).
pub fn parse_a4_names(lines: &[String]) -> Vec<String> {
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

/// Bond row: `(type_id, atom_i, atom_j, K, r0)` — atoms **1-based**.
pub type BondParamRow = (i64, i64, i64, f64, f64);

/// Decode bond pointer triples + force/equil tables.
pub fn decode_bond_params(
    pointers: &[i64],
    force_k: &[f64],
    equil: &[f64],
) -> Result<Vec<BondParamRow>, String> {
    if !pointers.len().is_multiple_of(3) {
        return Err(format!(
            "bond pointer length {} not multiple of 3",
            pointers.len()
        ));
    }
    let mut out = Vec::with_capacity(pointers.len() / 3);
    for chunk in pointers.chunks_exact(3) {
        let a = chunk[0];
        let b = chunk[1];
        if a < 0 || b < 0 {
            return Err(format!("Found negative bonded atom pointers ({a}, {b})"));
        }
        let type_id = chunk[2];
        let mut i = a / 3 + 1;
        let mut j = b / 3 + 1;
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        let tid = (type_id - 1) as usize;
        let k = *force_k
            .get(tid)
            .ok_or_else(|| format!("bond type {type_id} out of range"))?;
        let r0 = *equil
            .get(tid)
            .ok_or_else(|| format!("bond type {type_id} out of range"))?;
        out.push((type_id, i, j, k, r0));
    }
    Ok(out)
}

/// Angle row: `(type_id, i, j, k, K, theta0_deg)` — atoms **1-based**, θ in **degrees**.
pub type AngleParamRow = (i64, i64, i64, i64, f64, f64);

/// Decode angle pointer quads. Equilibrium angles in the table are radians;
/// returned `theta0` is converted to **degrees** (historical molpy helper).
pub fn decode_angle_params(
    pointers: &[i64],
    force_k: &[f64],
    equil_rad: &[f64],
) -> Result<Vec<AngleParamRow>, String> {
    if !pointers.len().is_multiple_of(4) {
        return Err(format!(
            "angle pointer length {} not multiple of 4",
            pointers.len()
        ));
    }
    let mut out = Vec::with_capacity(pointers.len() / 4);
    for chunk in pointers.chunks_exact(4) {
        let a = chunk[0];
        let b = chunk[1];
        let c = chunk[2];
        if a < 0 || b < 0 || c < 0 {
            return Err(format!(
                "Found negative angle atom pointers ({a}, {b}, {c})"
            ));
        }
        let type_id = chunk[3];
        let mut i = a / 3 + 1;
        let j = b / 3 + 1;
        let mut k = c / 3 + 1;
        if i > k {
            std::mem::swap(&mut i, &mut k);
        }
        let tid = (type_id - 1) as usize;
        let fk = *force_k
            .get(tid)
            .ok_or_else(|| format!("angle type {type_id} out of range"))?;
        let teq = *equil_rad
            .get(tid)
            .ok_or_else(|| format!("angle type {type_id} out of range"))?;
        out.push((type_id, i, j, k, fk, teq.to_degrees()));
    }
    Ok(out)
}

/// Dihedral row: `(type_id, i, j, k, l, K, phase_rad, n)` — atoms **1-based**.
pub type DihedralParamRow = (i64, i64, i64, i64, i64, f64, f64, i64);

/// Decode dihedral pointer quints. Phase stays radians; periodicity is rounded
/// to the nearest integer (historical molpy).
pub fn decode_dihedral_params(
    pointers: &[i64],
    force_k: &[f64],
    phase: &[f64],
    periodicity: &[f64],
) -> Result<Vec<DihedralParamRow>, String> {
    if !pointers.len().is_multiple_of(5) {
        return Err(format!(
            "dihedral pointer length {} not multiple of 5",
            pointers.len()
        ));
    }
    let mut out = Vec::with_capacity(pointers.len() / 5);
    for chunk in pointers.chunks_exact(5) {
        let a = chunk[0];
        let b = chunk[1];
        if a < 0 || b < 0 {
            return Err(format!(
                "Found negative dihedral atom pointers ({a}, {b}, {}, {})",
                chunk[2], chunk[3]
            ));
        }
        let type_id = chunk[4];
        let mut i = a / 3 + 1;
        let mut j = b / 3 + 1;
        let mut k = chunk[2].unsigned_abs() as i64 / 3 + 1;
        let mut l = chunk[3].unsigned_abs() as i64 / 3 + 1;
        if j > k {
            std::mem::swap(&mut i, &mut l);
            std::mem::swap(&mut j, &mut k);
        }
        let tid = (type_id - 1) as usize;
        let fk = *force_k
            .get(tid)
            .ok_or_else(|| format!("dihedral type {type_id} out of range"))?;
        let ph = *phase
            .get(tid)
            .ok_or_else(|| format!("dihedral type {type_id} out of range"))?;
        let pn = *periodicity
            .get(tid)
            .ok_or_else(|| format!("dihedral type {type_id} out of range"))?;
        // Match historical molpy: int(0.5 + pn) — truncate toward zero.
        let n = (0.5 + pn) as i64;
        out.push((type_id, i, j, k, l, fk, ph, n));
    }
    Ok(out)
}

/// Nonbond row: `(atom_1based, sigma, epsilon)`.
pub type NonbondParamRow = (i64, f64, f64);

/// Per-atom LJ σ/ε from diagonal ICO + A/B coefficients.
///
/// `hbond_a` / `hbond_b` must be flattened coefficient lists (any non-zero → error).
#[allow(clippy::too_many_arguments)]
pub fn decode_nonbond_params(
    n_atom: usize,
    n_types: usize,
    atom_type_index: &[i64],
    nonbonded_parm_index: &[i64],
    acoef: &[f64],
    bcoef: &[f64],
    hbond_a: &[f64],
    hbond_b: &[f64],
) -> Result<Vec<NonbondParamRow>, String> {
    for (&x, &y) in hbond_a.iter().zip(hbond_b.iter()) {
        if x != 0.0 || y != 0.0 {
            return Err("10-12 interactions are not supported".into());
        }
    }
    // Also reject any leftover non-zero if lengths differ
    for &x in hbond_a.iter().chain(hbond_b.iter()) {
        if x != 0.0 {
            return Err("10-12 interactions are not supported".into());
        }
    }

    let mut out = Vec::with_capacity(n_atom);
    for i_atom in 0..n_atom {
        let itype = *atom_type_index
            .get(i_atom)
            .ok_or_else(|| format!("ATOM_TYPE_INDEX missing atom {i_atom}"))?;
        // Historical diagonal: 0-based ICO index = (NTYPES+1)*(IAC-1)
        let index = (n_types + 1) * (itype as usize - 1);
        let nb = *nonbonded_parm_index.get(index).unwrap_or(&0);
        if nb < 0 {
            return Err("10-12 interactions are not supported".into());
        }
        let nb_idx = (nb - 1) as usize;
        let a = *acoef.get(nb_idx).unwrap_or(&0.0);
        let b = *bcoef.get(nb_idx).unwrap_or(&0.0);
        let (sigma, epsilon) = if a == 0.0 || b == 0.0 {
            (1.0, 0.0)
        } else {
            let r_min = (2.0 * a / b).powf(1.0 / 6.0);
            let eps = 0.25 * b * b / a;
            let sigma = 2f64.powf(-1.0 / 6.0) * r_min;
            (sigma, eps)
        };
        out.push(((i_atom as i64) + 1, sigma, epsilon));
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_1based_and_sorted() {
        // pointers 33,36,1 → atoms 12,13 type 1
        let rows = decode_bond_params(&[33, 36, 1], &[100.0], &[1.5]).unwrap();
        assert_eq!(rows, vec![(1, 12, 13, 100.0, 1.5)]);
        let rows = decode_bond_params(&[36, 33, 1], &[100.0], &[1.5]).unwrap();
        assert_eq!(rows[0].1, 12);
        assert_eq!(rows[0].2, 13);
    }

    #[test]
    fn angle_theta_degrees() {
        let teq = std::f64::consts::FRAC_PI_2; // 90°
        let rows = decode_angle_params(&[0, 3, 6, 1], &[50.0], &[teq]).unwrap();
        assert!((rows[0].5 - 90.0).abs() < 1e-9);
    }

    #[test]
    fn dihedral_abs_and_swap() {
        // l=-18 → abs//3+1 = 7
        let rows = decode_dihedral_params(&[0, 6, 12, -18, 1], &[0.5], &[0.0], &[2.0]).unwrap();
        assert_eq!(rows[0].4, 7);
    }
}
