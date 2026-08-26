//! AMBER prmtop **structure** reader.
//!
//! Parses topology/connectivity into a [`Frame`]. Force-field parameter tables
//! (harmonic constants, LJ coefficients, Fourier terms) are intentionally
//! **not** assembled here — those remain a separate product (FF reader) for
//! now. Structure fields mirror the historical molpy `AmberPrmtopReader` Frame
//! contract so molpy can thin to a molrs call.
//!
//! ## Output Frame
//!
//! - `"atoms"`: `id` (uint, 1-based), `name` (str), `type` (str, AMBER atom
//!   type), `charge` (float, electron units — prmtop value / 18.2223),
//!   `mass` (float), optional `atomic_number` (uint) + `element` (str),
//!   `res_id` (uint, 0-based from `RESIDUE_POINTER`)
//! - `"bonds"` / `"angles"` / `"dihedrals"`: connectivity (`atomi`/… 0-based
//!   uint), `type` (str label from atom types), `type_id` (uint, prmtop index),
//!   `id` (uint, 1-based row id). Empty systems still get schema-typed empty
//!   blocks.
//! - `frame.meta`: POINTERS raw fields (`NATOM`, …) plus derived
//!   `n_atoms` / `n_bonds` / `n_angles` / `n_dihedrals` / `n_atomtypes` /
//!   `n_bondtypes` / `n_angletypes` / `n_dihedraltypes`, and optional `title`.
//!
//! ## Encoding notes (Amber [FileFormats](https://ambermd.org/FileFormats.php))
//!
//! - Bonded atom pointers are coordinate-array indexes: true 1-based atom number
//!   is `|N|/3 + 1` (0-based index `|N|/3`).
//! - Dihedral: 3rd pointer negative → ignore end-group (1-4) interactions;
//!   4th pointer negative → improper torsion. Atom index uses absolute value.
//! - `ATOM_NAME` / `AMBER_ATOM_TYPE` / residue labels are Fortran `20a4`
//!   (exactly 4-char fields; may not be whitespace-delimited).
//! - `%COMMENT` lines are optional and skipped; section order is not required
//!   to be fixed (we index by `%FLAG` name).
//! - Charges are Amber internal units (`E = q1*q2/r` with kcal/mol, Å);
//!   we divide by 18.2223 to electron charge for the Frame.

use std::collections::HashMap;
use std::io::{BufRead, Error, ErrorKind, Result};
use std::path::Path;

use ndarray::{Array1, IxDyn};

use molrs::Element;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, Idx};

/// AMBER stores charges × √(332.0636 kcal·Å/mol/e²) ≈ 18.2223.
pub const CHARGE_CONVERSION_FACTOR: F = 18.2223;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn invalid_data<E: std::fmt::Display>(e: E) -> Error {
    Error::new(ErrorKind::InvalidData, e.to_string())
}

fn insert_float_col(block: &mut Block, key: &str, vals: Vec<F>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    block.insert(key, arr).map_err(invalid_data)
}

fn insert_uint_col(block: &mut Block, key: &str, vals: Vec<Idx>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    block.insert(key, arr).map_err(invalid_data)
}

fn insert_str_col(block: &mut Block, key: &str, vals: Vec<String>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    block.insert(key, arr).map_err(invalid_data)
}

/// Split whitespace-joined section lines into tokens and parse as `T`.
fn parse_tokens<T: std::str::FromStr>(lines: &[String]) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    let mut out = Vec::new();
    for line in lines {
        for tok in line.split_whitespace() {
            let v: T = tok
                .parse()
                .map_err(|e| invalid_data(format!("bad token {tok:?}: {e}")))?;
            out.push(v);
        }
    }
    Ok(out)
}

/// Fortran `20a4` — extract 4-character name fields (byte slices, ASCII).
///
/// Matches molpy: each 4-char window is stripped and always pushed (including
/// empty strings for blank pad windows).
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

// ---------------------------------------------------------------------------
// Section map
// ---------------------------------------------------------------------------

/// Parse a prmtop into flag → data lines (sanitized, non-empty).
///
/// Public for the FF reader and for Python helpers that still need raw
/// parameter tables (`BOND_FORCE_CONSTANT`, …) without re-implementing the
/// `%FLAG` / `%COMMENT` scan in molpy.
pub fn parse_flag_sections<R: BufRead>(mut reader: R) -> Result<HashMap<String, Vec<String>>> {
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
                .ok_or_else(|| invalid_data("malformed %FLAG line"))?
                .to_string();
            flag = Some(name);
            data = Vec::new();
        } else if line.starts_with("%FORMAT")
            || line.starts_with("%VERSION")
            || line.starts_with("%COMMENT")
        {
            // Amber FileFormats: any number of %COMMENT lines may appear
            // between %FLAG and data; ignore them.
        } else {
            data.push(line.to_string());
        }
    }
    if let Some(f) = flag {
        sections.entry(f).or_default().extend(data);
    }
    Ok(sections)
}

// ---------------------------------------------------------------------------
// POINTERS
// ---------------------------------------------------------------------------

const POINTER_FIELDS: &[&str] = &[
    "NATOM", "NTYPES", "NBONH", "MBONA", "NTHETH", "MTHETA", "NPHIH", "MPHIA", "NHPARM", "NPARM",
    "NNB", "NRES", "NBONA", "NTHETA", "NPHIA", "NUMBND", "NUMANG", "NPTRA", "NATYP", "NPHB",
    "IFPERT", "NBPER", "NGPER", "NDPER", "MBPER", "MGPER", "MDPER", "IFBOX", "NMXRS", "IFCAP",
    "NUMEXTRA", "NCOPY",
];

fn read_pointers(lines: &[String]) -> Result<HashMap<String, i64>> {
    let values: Vec<i64> = parse_tokens(lines)?;
    let mut meta = HashMap::new();
    for (name, val) in POINTER_FIELDS.iter().zip(values.iter()) {
        meta.insert((*name).to_string(), *val);
    }
    let natom = *meta
        .get("NATOM")
        .ok_or_else(|| invalid_data("POINTERS missing NATOM"))?;
    let nbonh = meta.get("NBONH").copied().unwrap_or(0);
    let mbona = meta.get("MBONA").copied().unwrap_or(0);
    let ntheth = meta.get("NTHETH").copied().unwrap_or(0);
    let mtheta = meta.get("MTHETA").copied().unwrap_or(0);
    let nphih = meta.get("NPHIH").copied().unwrap_or(0);
    let mphia = meta.get("MPHIA").copied().unwrap_or(0);
    let natyp = meta.get("NATYP").copied().unwrap_or(0);
    let numbnd = meta.get("NUMBND").copied().unwrap_or(0);
    let numang = meta.get("NUMANG").copied().unwrap_or(0);
    let nptra = meta.get("NPTRA").copied().unwrap_or(0);

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

// ---------------------------------------------------------------------------
// Connectivity decode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BondRow {
    type_id: Idx,
    atomi: Idx,
    atomj: Idx,
    type_name: String,
}

#[derive(Debug, Clone)]
struct AngleRow {
    type_id: Idx,
    atomi: Idx,
    atomj: Idx,
    atomk: Idx,
    type_name: String,
}

#[derive(Debug, Clone)]
struct DihedralRow {
    type_id: Idx,
    atomi: Idx,
    atomj: Idx,
    atomk: Idx,
    atoml: Idx,
    type_name: String,
    /// True when the raw 4th pointer was negative (Amber improper flag).
    is_improper: bool,
}

fn decode_bonds(pointers: &[i64], atom_types: &[String]) -> Result<Vec<BondRow>> {
    if !pointers.len().is_multiple_of(3) {
        return Err(invalid_data(format!(
            "bond pointer length {} not multiple of 3",
            pointers.len()
        )));
    }
    let mut out = Vec::with_capacity(pointers.len() / 3);
    for chunk in pointers.as_chunks::<3>().0 {
        let a = chunk[0];
        let b = chunk[1];
        if a < 0 || b < 0 {
            return Err(invalid_data(format!(
                "Found negative bonded atom pointers ({a}, {b})"
            )));
        }
        let type_id = chunk[2] as Idx;
        let mut i = (a / 3) as Idx; // 0-based
        let mut j = (b / 3) as Idx;
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        let ti = atom_types
            .get(i as usize)
            .ok_or_else(|| invalid_data(format!("bond atom index {i} out of range")))?;
        let tj = atom_types
            .get(j as usize)
            .ok_or_else(|| invalid_data(format!("bond atom index {j} out of range")))?;
        let mut pair = [ti.as_str(), tj.as_str()];
        pair.sort();
        let type_name = format!("{}-{}", pair[0], pair[1]);
        out.push(BondRow {
            type_id,
            atomi: i,
            atomj: j,
            type_name,
        });
    }
    Ok(out)
}

fn decode_angles(pointers: &[i64], atom_types: &[String]) -> Result<Vec<AngleRow>> {
    if !pointers.len().is_multiple_of(4) {
        return Err(invalid_data(format!(
            "angle pointer length {} not multiple of 4",
            pointers.len()
        )));
    }
    let mut out = Vec::with_capacity(pointers.len() / 4);
    for chunk in pointers.as_chunks::<4>().0 {
        let a = chunk[0];
        let b = chunk[1];
        let c = chunk[2];
        if a < 0 || b < 0 || c < 0 {
            return Err(invalid_data(format!(
                "Found negative angle atom pointers ({a}, {b}, {c})"
            )));
        }
        let type_id = chunk[3] as Idx;
        let mut i = (a / 3) as Idx;
        let j = (b / 3) as Idx;
        let mut k = (c / 3) as Idx;
        if i > k {
            std::mem::swap(&mut i, &mut k);
        }
        let ti = atom_types
            .get(i as usize)
            .ok_or_else(|| invalid_data(format!("angle atom index {i} out of range")))?;
        let tj = atom_types
            .get(j as usize)
            .ok_or_else(|| invalid_data(format!("angle atom index {j} out of range")))?;
        let tk = atom_types
            .get(k as usize)
            .ok_or_else(|| invalid_data(format!("angle atom index {k} out of range")))?;
        let type_name = format!("{ti}-{tj}-{tk}");
        out.push(AngleRow {
            type_id,
            atomi: i,
            atomj: j,
            atomk: k,
            type_name,
        });
    }
    Ok(out)
}

fn decode_dihedrals(pointers: &[i64], atom_types: &[String]) -> Result<Vec<DihedralRow>> {
    if !pointers.len().is_multiple_of(5) {
        return Err(invalid_data(format!(
            "dihedral pointer length {} not multiple of 5",
            pointers.len()
        )));
    }
    let mut out = Vec::with_capacity(pointers.len() / 5);
    for chunk in pointers.as_chunks::<5>().0 {
        let a = chunk[0];
        let b = chunk[1];
        if a < 0 || b < 0 {
            return Err(invalid_data(format!(
                "Found negative dihedral atom pointers ({a}, {b}, {}, {})",
                chunk[2], chunk[3]
            )));
        }
        let type_id = chunk[4] as Idx;
        let is_improper = chunk[3] < 0;
        let mut i = (a / 3) as Idx;
        let mut j = (b / 3) as Idx;
        let mut k = (chunk[2].unsigned_abs() / 3) as Idx;
        let mut l = (chunk[3].unsigned_abs() / 3) as Idx;
        // Canonicalise so type name is direction-independent (j ≤ k).
        if j > k {
            std::mem::swap(&mut i, &mut l);
            std::mem::swap(&mut j, &mut k);
        }
        let ti = atom_types
            .get(i as usize)
            .ok_or_else(|| invalid_data(format!("dihedral atom index {i} out of range")))?;
        let tj = atom_types
            .get(j as usize)
            .ok_or_else(|| invalid_data(format!("dihedral atom index {j} out of range")))?;
        let tk = atom_types
            .get(k as usize)
            .ok_or_else(|| invalid_data(format!("dihedral atom index {k} out of range")))?;
        let tl = atom_types
            .get(l as usize)
            .ok_or_else(|| invalid_data(format!("dihedral atom index {l} out of range")))?;
        let type_name = format!("{ti}-{tj}-{tk}-{tl}");
        out.push(DihedralRow {
            type_id,
            atomi: i,
            atomj: j,
            atomk: k,
            atoml: l,
            type_name,
            is_improper,
        });
    }
    Ok(out)
}

fn residue_ids(pointer_lines: Option<&Vec<String>>, n_atoms: usize) -> Result<Vec<Idx>> {
    let Some(lines) = pointer_lines else {
        return Ok(vec![0; n_atoms]);
    };
    let mut ptrs: Vec<i64> = parse_tokens(lines)?;
    if ptrs.is_empty() {
        return Ok(vec![0; n_atoms]);
    }
    // RESIDUE_POINTER is 1-based start atom; append sentinel n_atoms+1.
    ptrs.push((n_atoms as i64) + 1);
    let mut res_id = Vec::with_capacity(n_atoms);
    for (r, window) in ptrs.windows(2).enumerate() {
        let start = (window[0] - 1) as usize;
        let end = (window[1] - 1) as usize;
        if end < start || end > n_atoms {
            return Err(invalid_data(format!(
                "bad RESIDUE_POINTER slice {start}..{end} for n_atoms={n_atoms}"
            )));
        }
        for _ in start..end {
            res_id.push(r as Idx);
        }
    }
    if res_id.len() != n_atoms {
        return Err(invalid_data(format!(
            "RESIDUE_POINTER assigned {} atoms, expected {n_atoms}",
            res_id.len()
        )));
    }
    Ok(res_id)
}

// ---------------------------------------------------------------------------
// Build Frame
// ---------------------------------------------------------------------------

fn build_bond_block(rows: &[BondRow]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut type_id = Vec::with_capacity(n);
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut type_name = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    for (idx, r) in rows.iter().enumerate() {
        type_id.push(r.type_id);
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        type_name.push(r.type_name.clone());
        id.push((idx as Idx) + 1);
    }
    insert_uint_col(&mut block, "type_id", type_id)?;
    insert_uint_col(&mut block, "atomi", atomi)?;
    insert_uint_col(&mut block, "atomj", atomj)?;
    insert_str_col(&mut block, "type", type_name)?;
    insert_uint_col(&mut block, "id", id)?;
    Ok(block)
}

fn build_angle_block(rows: &[AngleRow]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut type_id = Vec::with_capacity(n);
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut atomk = Vec::with_capacity(n);
    let mut type_name = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    for (idx, r) in rows.iter().enumerate() {
        type_id.push(r.type_id);
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        atomk.push(r.atomk);
        type_name.push(r.type_name.clone());
        id.push((idx as Idx) + 1);
    }
    insert_uint_col(&mut block, "type_id", type_id)?;
    insert_uint_col(&mut block, "atomi", atomi)?;
    insert_uint_col(&mut block, "atomj", atomj)?;
    insert_uint_col(&mut block, "atomk", atomk)?;
    insert_str_col(&mut block, "type", type_name)?;
    insert_uint_col(&mut block, "id", id)?;
    Ok(block)
}

fn build_dihedral_block(rows: &[DihedralRow]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut type_id = Vec::with_capacity(n);
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut atomk = Vec::with_capacity(n);
    let mut atoml = Vec::with_capacity(n);
    let mut type_name = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    for (idx, r) in rows.iter().enumerate() {
        type_id.push(r.type_id);
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        atomk.push(r.atomk);
        atoml.push(r.atoml);
        type_name.push(r.type_name.clone());
        id.push((idx as Idx) + 1);
    }
    insert_uint_col(&mut block, "type_id", type_id)?;
    insert_uint_col(&mut block, "atomi", atomi)?;
    insert_uint_col(&mut block, "atomj", atomj)?;
    insert_uint_col(&mut block, "atomk", atomk)?;
    insert_uint_col(&mut block, "atoml", atoml)?;
    insert_str_col(&mut block, "type", type_name)?;
    insert_uint_col(&mut block, "id", id)?;
    Ok(block)
}

fn section_ints(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Vec<i64>> {
    match sections.get(key) {
        Some(lines) => parse_tokens(lines),
        None => Ok(Vec::new()),
    }
}

fn build_frame(sections: HashMap<String, Vec<String>>) -> Result<Frame> {
    let pointers_lines = sections.get("POINTERS").ok_or_else(|| {
        invalid_data(
            "Invalid or empty prmtop file: POINTERS section missing. \
             This typically means the external tool (tleap) failed to create the file.",
        )
    })?;
    let meta_map = read_pointers(pointers_lines)?;
    let n_atoms = *meta_map
        .get("n_atoms")
        .ok_or_else(|| invalid_data("n_atoms missing after POINTERS"))? as usize;

    let names = sections
        .get("ATOM_NAME")
        .map(|l| read_a4_names(l))
        .unwrap_or_default();
    if !names.is_empty() && names.len() != n_atoms {
        // Trailing empty pad fields from a short last line can inflate count;
        // trim to n_atoms when we have at least n_atoms non-pad entries.
        // Prefer exact n_atoms prefix if longer (pad empties at end).
        if names.len() < n_atoms {
            return Err(invalid_data(format!(
                "ATOM_NAME has {} entries, expected {n_atoms}",
                names.len()
            )));
        }
    }
    let mut names = names;
    if names.len() > n_atoms {
        names.truncate(n_atoms);
    }
    if names.len() < n_atoms {
        names.resize(n_atoms, String::new());
    }

    let raw_charges: Vec<F> = sections
        .get("CHARGE")
        .map(|l| parse_tokens(l))
        .transpose()?
        .unwrap_or_default();
    if raw_charges.len() != n_atoms {
        return Err(invalid_data(format!(
            "CHARGE has {} entries, expected {n_atoms}",
            raw_charges.len()
        )));
    }
    let charges: Vec<F> = raw_charges
        .into_iter()
        .map(|q| q / CHARGE_CONVERSION_FACTOR)
        .collect();

    let masses: Vec<F> = sections
        .get("MASS")
        .map(|l| parse_tokens(l))
        .transpose()?
        .unwrap_or_default();
    if masses.len() != n_atoms {
        return Err(invalid_data(format!(
            "MASS has {} entries, expected {n_atoms}",
            masses.len()
        )));
    }

    let mut atom_types = sections
        .get("AMBER_ATOM_TYPE")
        .map(|l| read_a4_names(l))
        .unwrap_or_default();
    if atom_types.len() > n_atoms {
        atom_types.truncate(n_atoms);
    }
    if atom_types.len() < n_atoms {
        atom_types.resize(n_atoms, String::new());
    }

    let atomic_numbers_raw: Option<Vec<i64>> = sections
        .get("ATOMIC_NUMBER")
        .map(|l| parse_tokens(l))
        .transpose()?;
    // AMBER writes -1 for unknown element; omit the column rather than fake 0.
    let atomic_numbers: Option<Vec<Idx>> = match atomic_numbers_raw {
        Some(nums) if !nums.is_empty() && nums.iter().all(|&z| z >= 0) => {
            if nums.len() != n_atoms {
                return Err(invalid_data(format!(
                    "ATOMIC_NUMBER has {} entries, expected {n_atoms}",
                    nums.len()
                )));
            }
            Some(nums.into_iter().map(|z| z as Idx).collect())
        }
        _ => None,
    };

    let res_ids = residue_ids(sections.get("RESIDUE_POINTER"), n_atoms)?;

    // Connectivity (inc + without H).
    let mut bond_ptrs = section_ints(&sections, "BONDS_INC_HYDROGEN")?;
    bond_ptrs.extend(section_ints(&sections, "BONDS_WITHOUT_HYDROGEN")?);
    let bonds = decode_bonds(&bond_ptrs, &atom_types)?;

    let mut angle_ptrs = section_ints(&sections, "ANGLES_INC_HYDROGEN")?;
    angle_ptrs.extend(section_ints(&sections, "ANGLES_WITHOUT_HYDROGEN")?);
    let angles = decode_angles(&angle_ptrs, &atom_types)?;

    let mut dihe_ptrs = section_ints(&sections, "DIHEDRALS_INC_HYDROGEN")?;
    dihe_ptrs.extend(section_ints(&sections, "DIHEDRALS_WITHOUT_HYDROGEN")?);
    let dihedrals = decode_dihedrals(&dihe_ptrs, &atom_types)?;

    // ---- atoms block ----
    let mut atoms = Block::new();
    let ids: Vec<Idx> = (1..=n_atoms as Idx).collect();
    insert_uint_col(&mut atoms, "id", ids)?;
    insert_str_col(&mut atoms, "name", names)?;
    insert_str_col(&mut atoms, "type", atom_types)?;
    insert_float_col(&mut atoms, "charge", charges)?;
    insert_float_col(&mut atoms, "mass", masses)?;
    insert_uint_col(&mut atoms, "res_id", res_ids)?;
    if let Some(zs) = atomic_numbers {
        let elements: Vec<String> = zs
            .iter()
            .map(|&z| {
                Element::by_number(z as u8)
                    .map(|e| e.symbol().to_string())
                    .unwrap_or_else(|| {
                        // z==0 or out of range: leave empty (should not occur
                        // when all z >= 0 and valid; fall back to "?").
                        if z == 0 {
                            String::new()
                        } else {
                            format!("Z{z}")
                        }
                    })
            })
            .collect();
        insert_uint_col(&mut atoms, "atomic_number", zs)?;
        insert_str_col(&mut atoms, "element", elements)?;
    }

    let mut frame = Frame::new();
    // meta
    if let Some(title_lines) = sections.get("TITLE")
        && let Some(t) = title_lines.first()
    {
        frame.meta.insert("title", t.clone());
    }
    for (k, v) in meta_map {
        frame.meta.insert(k, v);
    }

    frame.insert("atoms", atoms);
    frame.insert("bonds", build_bond_block(&bonds)?);
    frame.insert("angles", build_angle_block(&angles)?);
    // Keep full list in "dihedrals" so meta n_dihedrals (NPHIH+MPHIA) still
    // matches historical Frame contracts. Amber impropers (4th pointer
    // negative) are *also* mirrored into "impropers" for LAMMPS-style consumers.
    frame.insert("dihedrals", build_dihedral_block(&dihedrals)?);
    let impropers: Vec<DihedralRow> = dihedrals
        .iter()
        .filter(|r| r.is_improper)
        .cloned()
        .collect();
    if !impropers.is_empty() {
        frame.insert("impropers", build_dihedral_block(&impropers)?);
    }

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read an AMBER prmtop structure file at `path` into a [`Frame`].
pub fn read_amber_prmtop<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    read_amber_prmtop_from_reader(std::io::BufReader::new(file))
}

/// Alias for [`read_amber_prmtop`].
pub fn read_prmtop<P: AsRef<Path>>(path: P) -> Result<Frame> {
    read_amber_prmtop(path)
}

/// Parse AMBER prmtop structure text from any [`BufRead`].
pub fn read_amber_prmtop_from_reader<R: BufRead>(reader: R) -> Result<Frame> {
    let sections = parse_flag_sections(reader)?;
    build_frame(sections)
}

/// Read raw `%FLAG` sections from a prmtop path (flag name → data lines).
///
/// Skips `%VERSION` / `%FORMAT` / `%COMMENT`. Used by force-field helpers that
/// still inspect parameter tables without a second Python text scan.
pub fn read_amber_prmtop_sections<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Vec<String>>> {
    let file = std::fs::File::open(path.as_ref())?;
    parse_flag_sections(std::io::BufReader::new(file))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const LITFSI_HEAD: &str = "\
%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
TFSI
%FLAG POINTERS
%FORMAT(10I8)
      16       6       0      14       0      25       0      27       0       0
      65       2      14      25      27       7      12       4       7       0
       0       0       0       0       0       0       0       0      15       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
F   C   F1  F2  S   O   O3  N   S1  O1  O2  C1  F4  F5  F3  LI
%FLAG CHARGE
%FORMAT(5E16.8)
 -4.94977802E+00  1.01079098E+01 -4.94977802E+00 -4.94977802E+00  2.70364265E+01
 -1.11739144E+01 -1.19684066E+01 -1.92992379E+01  3.06571975E+01 -1.19684066E+01
 -1.19684066E+01  1.01079098E+01 -4.94977802E+00 -4.94977802E+00 -4.94977802E+00
  1.82223000E+01
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       9       6       9       9      16       8       8       7      16       8
       8       6       9       9       9       3
%FLAG MASS
%FORMAT(5E16.8)
  1.90000000E+01  1.20100000E+01  1.90000000E+01  1.90000000E+01  3.20600000E+01
  1.60000000E+01  1.60000000E+01  1.40100000E+01  3.20600000E+01  1.60000000E+01
  1.60000000E+01  1.20100000E+01  1.90000000E+01  1.90000000E+01  1.90000000E+01
  6.94000000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
f   c3  f   f   s6  o   o   ne  sy  o   o   c3  f   f   f   Li+
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1      16
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
      33      36       1      33      39       1      33      42       1      24
      27       2      24      30       2      24      33       3      21      24
       4      12      15       5      12      18       5      12      21       6
       3       6       1       3       9       1       3      12       7       0
       3       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
      39      33      42       1      36      33      39       1      36      33
      42       1      30      24      33       2      27      24      30       3
      27      24      33       2      24      33      36       4      24      33
      39       4      24      33      42       4      21      24      27       5
      21      24      30       5      21      24      33       6      18      12
      21       7      15      12      18       8      15      12      21       7
      12      21      24       9       9       3      12      10       6       3
       9       1       6       3      12      10       3      12      15      11
       3      12      18      11       3      12      21      12       0       3
       6       1       0       3       9       1       0       3      12      10
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
      30      24      33      36       1      30      24      33      39       1
      30      24      33      42       1      27      24      33      36       1
      27      24      33      39       1      27      24      33      42       1
      21      24      33      36       1      21      24      33      39       1
      21      24      33      42       1      18      12      21      24       2
      15      12      21      24       2      12      21      24      27       3
      12      21     -24      27       4      12      21      24      30       3
      12      21     -24      30       4      12      21      24      33       3
      12      21     -24      33       4       9       3      12      15       1
       9       3      12      18       1       9       3      12      21       1
       6       3      12      15       1       6       3      12      18       1
       6       3      12      21       1       3      12      21      24       2
       0       3      12      15       1       0       3      12      18       1
       0       3      12      21       1
";

    fn frame_from(s: &str) -> Frame {
        read_amber_prmtop_from_reader(Cursor::new(s.as_bytes())).expect("parse")
    }

    #[test]
    fn a4_names_chunking() {
        let names = read_a4_names(&["F   C   F1  F2  S   O   O3  N   ".to_string()]);
        assert_eq!(names, vec!["F", "C", "F1", "F2", "S", "O", "O3", "N"]);
    }

    #[test]
    fn litfsi_counts() {
        let frame = frame_from(LITFSI_HEAD);
        assert_eq!(frame.meta.get("n_atoms").and_then(|v| v.as_i64()), Some(16));
        assert_eq!(frame.meta.get("n_bonds").and_then(|v| v.as_i64()), Some(14));
        assert_eq!(
            frame.meta.get("n_angles").and_then(|v| v.as_i64()),
            Some(25)
        );
        assert_eq!(
            frame.meta.get("n_dihedrals").and_then(|v| v.as_i64()),
            Some(27)
        );
        assert_eq!(
            frame.meta.get("title").and_then(|v| v.as_str()),
            Some("TFSI")
        );
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(16));
        let bonds = frame.get("bonds").unwrap();
        assert_eq!(bonds.nrows(), Some(14));
        let angles = frame.get("angles").unwrap();
        assert_eq!(angles.nrows(), Some(25));
        let dihedrals = frame.get("dihedrals").unwrap();
        assert_eq!(dihedrals.nrows(), Some(27));
    }

    #[test]
    fn litfsi_charge_and_li() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let charge = atoms.get_float("charge").unwrap();
        assert!((charge[[15]] - 1.0).abs() < 1e-5);
        let total: F = (0..16).map(|i| charge[[i]]).sum();
        assert!(total.abs() < 0.01);
        let z = atoms.get_uint("atomic_number").unwrap();
        assert_eq!(z[[15]], 3);
        assert_eq!(z[[0]], 9);
        let names = atoms.get_string("name").unwrap();
        assert_eq!(names[[0]], "F");
        assert_eq!(names[[15]], "LI");
        let types = atoms.get_string("type").unwrap();
        assert_eq!(types[[0]], "f");
        assert_eq!(types[[1]], "c3");
        assert_eq!(types[[15]], "Li+");
        let res = atoms.get_uint("res_id").unwrap();
        assert_eq!(res[[0]], 0);
        assert_eq!(res[[14]], 0);
        assert_eq!(res[[15]], 1);
    }

    #[test]
    fn bond_first_pair_zero_based() {
        let frame = frame_from(LITFSI_HEAD);
        let bonds = frame.get("bonds").unwrap();
        let ai = bonds.get_uint("atomi").unwrap();
        let aj = bonds.get_uint("atomj").unwrap();
        let mut found = false;
        for i in 0..ai.len() {
            let a = ai[[i]];
            let b = aj[[i]];
            if (a == 11 && b == 12) || (a == 12 && b == 11) {
                found = true;
                break;
            }
        }
        assert!(found, "expected bond (11,12)");
    }

    #[test]
    fn missing_pointers() {
        let text = "%FLAG TITLE\n%FORMAT(20a4)\ntest\n";
        let err = read_amber_prmtop_from_reader(Cursor::new(text.as_bytes())).unwrap_err();
        assert!(err.to_string().contains("POINTERS section missing"));
    }

    #[test]
    fn bond_index_encoding_unit() {
        // raw 33,36 → 11,12 0-based; type 1
        let types: Vec<String> = (0..16).map(|i| format!("t{i}")).collect();
        let rows = decode_bonds(&[33, 36, 1], &types).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].atomi, 11);
        assert_eq!(rows[0].atomj, 12);
        assert_eq!(rows[0].type_id, 1);
    }

    #[test]
    fn dihedral_negative_k() {
        let types: Vec<String> = (0..16).map(|i| format!("t{i}")).collect();
        // (12, 21, -24, 27, 1): k=abs(-24)/3=8, l=9; j=7
        let rows = decode_dihedrals(&[12, 21, -24, 27, 1], &types).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].atomk, 8);
        assert_eq!(rows[0].atoml, 9);
    }
}
