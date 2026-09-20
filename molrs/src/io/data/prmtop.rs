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
//!   `res_id` (uint, 0-based from `RESIDUE_POINTER`), optional `res_name`
//!   (from `RESIDUE_LABEL`), optional `mol_id` (1-based, from
//!   `ATOMS_PER_MOLECULE`; never inferred from bonds). Format-local
//!   unregistered columns — a debt, not a licence, with no cross-format
//!   consumer: `tree` (`TREE_CHAIN_CLASSIFICATION`), `gb_radius` (Å, `RADII`),
//!   `gb_screen` (`SCREEN`). Each is absent when its section is absent.
//! - `"bonds"` / `"angles"` / `"dihedrals"`: connectivity (`atomi`/… 0-based
//!   uint), `type` (str label from atom types), `type_id` (uint, prmtop index),
//!   `id` (uint, 1-based row id). `"dihedrals"` also carries `exclude_14`
//!   (bool: negative 3rd pointer). Empty systems still get schema-typed empty
//!   blocks. Amber impropers (negative 4th pointer) are mirrored into
//!   `"impropers"` from the same builder, so `exclude_14` cannot diverge.
//! - `"exclusions"`: `atomi`/`atomj` (uint, 0-based, `atomi < atomj`), the
//!   Ewald real-space correction set from `NUMBER_EXCLUDED_ATOMS` /
//!   `EXCLUDED_ATOMS_LIST`. `0` placeholders are dropped. An all-placeholder
//!   list yields a schema-typed empty block.
//! - `frame.simbox`: from `BOX_DIMENSIONS` when `IFBOX = 1` (ortho) or
//!   `IFBOX = 2` (truncated octahedron at `arccos(-1/3)`). An inpcrd box
//!   takes precedence over a prmtop box: the prmtop cell is the topology-time
//!   default and the inpcrd cell is the current state. The two readers stay
//!   independent — callers who read both files apply that rule.
//! - `frame.meta`: POINTERS raw fields (`NATOM`, …) plus derived
//!   `n_atoms` / `n_bonds` / `n_angles` / `n_dihedrals` / `n_atomtypes` /
//!   `n_bondtypes` / `n_angletypes` / `n_dihedraltypes`, optional `title`,
//!   and optional `radius_set` / `oldbeta` / `solvent_iptres` /
//!   `solvent_nspm` / `solvent_nspsol`.
//!
//! ## Encoding notes (Amber [FileFormats](https://ambermd.org/FileFormats.php))
//!
//! - Bonded atom pointers are coordinate-array indexes: true 1-based atom number
//!   is `|N|/3 + 1` (0-based index `|N|/3`).
//! - Dihedral: 3rd pointer negative → ignore end-group (1-4) interactions;
//!   4th pointer negative → improper torsion. Atom index uses absolute value.
//!   Pointer `0` denotes atom 1 — the sign, not the value, carries the flags.
//! - `ATOM_NAME` / `AMBER_ATOM_TYPE` / residue labels are Fortran `20a4`
//!   (exactly 4-char fields; may not be whitespace-delimited).
//! - `%COMMENT` lines are optional and skipped; section order is not required
//!   to be fixed (we index by `%FLAG` name).
//! - Charges are Amber internal units (`E = q1*q2/r` with kcal/mol, Å);
//!   we divide by the literal 18.2223 to electron charge for the Frame.

use std::collections::HashMap;
use std::io::{BufRead, Error, ErrorKind, Result};
use std::path::Path;

use ndarray::{Array1, IxDyn, array};

use molrs::Element;
use molrs::spatial::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::store::keys;
use molrs::types::{F, Idx};

use super::prmtop_tables;

/// AMBER `CHARGE` stores `q · 18.2223`. The literal is Amber's own factor
/// (`18.2223² = 332.05221729` kcal·Å·mol⁻¹·e⁻²), 3.46e-5 below molrs CODATA
/// `COULOMB_REAL`. Do not re-derive it from `√332.06371`.
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

fn insert_bool_col(block: &mut Block, key: &str, vals: Vec<bool>) -> Result<()> {
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

fn unsupported(what: &str) -> Error {
    invalid_data(format!("{what} are not supported"))
}

fn length_mismatch(flag: &str, got: usize, expected: impl std::fmt::Display) -> Error {
    invalid_data(format!("{flag} has {got} entries, expected {expected}"))
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
    /// True when the raw 3rd pointer was negative (Amber 1-4 suppression).
    exclude_14: bool,
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
        let exclude_14 = chunk[2] < 0;
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
            exclude_14,
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
    let mut exclude_14 = Vec::with_capacity(n);
    for (idx, r) in rows.iter().enumerate() {
        type_id.push(r.type_id);
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        atomk.push(r.atomk);
        atoml.push(r.atoml);
        type_name.push(r.type_name.clone());
        id.push((idx as Idx) + 1);
        exclude_14.push(r.exclude_14);
    }
    insert_uint_col(&mut block, "type_id", type_id)?;
    insert_uint_col(&mut block, "atomi", atomi)?;
    insert_uint_col(&mut block, "atomj", atomj)?;
    insert_uint_col(&mut block, "atomk", atomk)?;
    insert_uint_col(&mut block, "atoml", atoml)?;
    insert_str_col(&mut block, "type", type_name)?;
    insert_uint_col(&mut block, "id", id)?;
    insert_bool_col(&mut block, keys::EXCLUDE_14, exclude_14)?;
    Ok(block)
}

fn section_ints(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Vec<i64>> {
    match sections.get(key) {
        Some(lines) => parse_tokens(lines),
        None => Ok(Vec::new()),
    }
}

fn first_i64(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Option<i64>> {
    let Some(lines) = sections.get(key) else {
        return Ok(None);
    };
    let vals: Vec<i64> = parse_tokens(lines)?;
    Ok(vals.first().copied())
}

fn trim_trailing_empty(mut names: Vec<String>) -> Vec<String> {
    while names.last().is_some_and(|s| s.is_empty()) {
        names.pop();
    }
    names
}

fn a4_exact(
    sections: &HashMap<String, Vec<String>>,
    flag: &str,
    expected: usize,
) -> Result<Option<Vec<String>>> {
    let Some(lines) = sections.get(flag) else {
        return Ok(None);
    };
    let names = trim_trailing_empty(prmtop_tables::parse_a4_names(lines));
    if names.len() != expected {
        return Err(length_mismatch(flag, names.len(), expected));
    }
    Ok(Some(names))
}

fn floats_n(
    sections: &HashMap<String, Vec<String>>,
    flag: &str,
    expected: usize,
) -> Result<Option<Vec<F>>> {
    let Some(lines) = sections.get(flag) else {
        return Ok(None);
    };
    let vals: Vec<F> = parse_tokens(lines)?;
    if vals.len() != expected {
        return Err(length_mismatch(flag, vals.len(), expected));
    }
    Ok(Some(vals))
}

fn check_dead_int_section(
    sections: &HashMap<String, Vec<String>>,
    flag: &str,
    n_atoms: usize,
) -> Result<()> {
    let Some(lines) = sections.get(flag) else {
        return Ok(());
    };
    let vals: Vec<i64> = parse_tokens(lines)?;
    if vals.len() != n_atoms {
        return Err(length_mismatch(flag, vals.len(), n_atoms));
    }
    Ok(())
}

fn refuse_unsupported(
    sections: &HashMap<String, Vec<String>>,
    meta_map: &HashMap<String, i64>,
) -> Result<()> {
    if meta_map.get("IFPERT").copied().unwrap_or(0) > 0 {
        return Err(unsupported("perturbed (IFPERT>0) prmtop files"));
    }
    if meta_map.get("IFCAP").copied().unwrap_or(0) > 0 {
        return Err(unsupported("solvent-cap (IFCAP>0) prmtop files"));
    }
    if meta_map.get("IFBOX").copied().unwrap_or(0) == 3 {
        return Err(unsupported("IFBOX=3 prmtop files"));
    }
    if sections.contains_key("CTITLE") || sections.contains_key("FORCE_FIELD_TYPE") {
        return Err(unsupported("CHAMBER prmtop files"));
    }
    if sections.contains_key("LENNARD_JONES_CCOEF") {
        return Err(unsupported("12-6-4 Lennard-Jones prmtop files"));
    }
    if first_i64(sections, "CMAP_COUNT")?.unwrap_or(0) > 0 {
        return Err(unsupported("CMAP terms"));
    }
    if first_i64(sections, "IPOL")?.unwrap_or(0) == 1 {
        return Err(unsupported("polarizable (IPOL=1) prmtop files"));
    }
    if let Some(lines) = sections.get("NONBONDED_PARM_INDEX") {
        let vals: Vec<i64> = parse_tokens(lines)?;
        if vals.iter().any(|&v| v < 0) {
            return Err(unsupported("10-12 hydrogen-bond prmtop files"));
        }
    }
    Ok(())
}

fn residue_names(
    sections: &HashMap<String, Vec<String>>,
    res_ids: &[Idx],
    n_res: usize,
) -> Result<Option<Vec<String>>> {
    let Some(labels) = a4_exact(sections, "RESIDUE_LABEL", n_res)? else {
        return Ok(None);
    };
    let mut names = Vec::with_capacity(res_ids.len());
    for &rid in res_ids {
        let name = labels
            .get(rid as usize)
            .ok_or_else(|| invalid_data(format!("res_id {rid} out of range for RESIDUE_LABEL")))?;
        names.push(name.clone());
    }
    Ok(Some(names))
}

fn molecule_ids(
    sections: &HashMap<String, Vec<String>>,
    n_atoms: usize,
) -> Result<Option<Vec<Idx>>> {
    let Some(lines) = sections.get("ATOMS_PER_MOLECULE") else {
        return Ok(None);
    };
    let counts: Vec<i64> = parse_tokens(lines)?;
    let mut mol_id = Vec::with_capacity(n_atoms);
    for (i, &n) in counts.iter().enumerate() {
        if n < 0 {
            return Err(invalid_data(format!(
                "ATOMS_PER_MOLECULE entry {n} is negative"
            )));
        }
        let id = (i as Idx) + 1;
        for _ in 0..n {
            mol_id.push(id);
        }
    }
    if mol_id.len() != n_atoms {
        return Err(length_mismatch("ATOMS_PER_MOLECULE", mol_id.len(), n_atoms));
    }
    Ok(Some(mol_id))
}

fn build_exclusions_block(
    sections: &HashMap<String, Vec<String>>,
    nnb: i64,
    n_atoms: usize,
) -> Result<Option<Block>> {
    let has_numex = sections.contains_key("NUMBER_EXCLUDED_ATOMS");
    let has_list = sections.contains_key("EXCLUDED_ATOMS_LIST");
    if !has_numex && !has_list {
        return Ok(None);
    }
    if has_numex != has_list {
        return Err(invalid_data(
            "NUMBER_EXCLUDED_ATOMS and EXCLUDED_ATOMS_LIST must appear together",
        ));
    }
    let numex: Vec<i64> = parse_tokens(&sections["NUMBER_EXCLUDED_ATOMS"])?;
    let list: Vec<i64> = parse_tokens(&sections["EXCLUDED_ATOMS_LIST"])?;
    if list.len() != nnb as usize {
        return Err(length_mismatch(
            "EXCLUDED_ATOMS_LIST",
            list.len(),
            format!("NNB={nnb}"),
        ));
    }
    if numex.len() != n_atoms {
        return Err(length_mismatch(
            "NUMBER_EXCLUDED_ATOMS",
            numex.len(),
            n_atoms,
        ));
    }
    let mut atomi = Vec::new();
    let mut atomj = Vec::new();
    let mut cursor = 0usize;
    for (i, &count) in numex.iter().enumerate() {
        if count < 0 {
            return Err(invalid_data("NUMBER_EXCLUDED_ATOMS entry is negative"));
        }
        let n = count as usize;
        if cursor + n > list.len() {
            return Err(length_mismatch(
                "EXCLUDED_ATOMS_LIST",
                list.len(),
                format!("NNB={nnb}"),
            ));
        }
        for &partner in &list[cursor..cursor + n] {
            if partner == 0 {
                continue;
            }
            if partner < 1 {
                return Err(invalid_data(format!(
                    "EXCLUDED_ATOMS_LIST partner {partner} is not a 1-based atom number"
                )));
            }
            let mut a = i as Idx;
            let mut b = (partner as Idx) - 1;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            if a != b {
                atomi.push(a);
                atomj.push(b);
            }
        }
        cursor += n;
    }
    if cursor != list.len() {
        return Err(invalid_data(format!(
            "NUMBER_EXCLUDED_ATOMS sums to {cursor}, expected NNB={nnb}"
        )));
    }
    let mut block = Block::new();
    insert_uint_col(&mut block, keys::ATOMI, atomi)?;
    insert_uint_col(&mut block, keys::ATOMJ, atomj)?;
    Ok(Some(block))
}

fn apply_box(frame: &mut Frame, sections: &HashMap<String, Vec<String>>, ifbox: i64) -> Result<()> {
    let Some(lines) = sections.get("BOX_DIMENSIONS") else {
        return Ok(());
    };
    let vals: Vec<F> = parse_tokens(lines)?;
    if vals.len() < 4 {
        return Err(length_mismatch("BOX_DIMENSIONS", vals.len(), 4usize));
    }
    frame.meta.insert("oldbeta", vals[0]);
    let lengths = array![vals[1], vals[2], vals[3]];
    let origin = array![0.0 as F, 0.0, 0.0];
    let simbox = match ifbox {
        1 => {
            SimBox::ortho(lengths, origin, [true; 3]).map_err(|e| invalid_data(format!("{e:?}")))?
        }
        2 => {
            let angle = (-1.0 as F / 3.0).acos().to_degrees();
            let h = SimBox::matrix_from_lengths_angles([vals[1], vals[2], vals[3]], [angle; 3])
                .map_err(|e| invalid_data(format!("{e:?}")))?;
            SimBox::new(h, origin, [true; 3]).map_err(|e| invalid_data(format!("{e:?}")))?
        }
        _ => return Ok(()),
    };
    frame.simbox = Some(simbox);
    Ok(())
}

fn apply_meta_scalars(frame: &mut Frame, sections: &HashMap<String, Vec<String>>) -> Result<()> {
    if let Some(lines) = sections.get("RADIUS_SET")
        && let Some(line) = lines.first()
    {
        frame.meta.insert("radius_set", line.clone());
    }
    if let Some(lines) = sections.get("SOLVENT_POINTERS") {
        let vals: Vec<i64> = parse_tokens(lines)?;
        if vals.len() != 3 {
            return Err(length_mismatch("SOLVENT_POINTERS", vals.len(), 3usize));
        }
        frame.meta.insert("solvent_iptres", vals[0]);
        frame.meta.insert("solvent_nspm", vals[1]);
        frame.meta.insert("solvent_nspsol", vals[2]);
    }
    Ok(())
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
    let n_res = *meta_map.get("NRES").unwrap_or(&0) as usize;
    let nnb = *meta_map.get("NNB").unwrap_or(&0);
    let ifbox = meta_map.get("IFBOX").copied().unwrap_or(0);
    refuse_unsupported(&sections, &meta_map)?;
    check_dead_int_section(&sections, "JOIN_ARRAY", n_atoms)?;
    check_dead_int_section(&sections, "IROTAT", n_atoms)?;

    let names = sections
        .get("ATOM_NAME")
        .map(|l| prmtop_tables::parse_a4_names(l))
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
        .map(|l| prmtop_tables::parse_a4_names(l))
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
    let res_name = residue_names(&sections, &res_ids, n_res)?;
    let mol_id = molecule_ids(&sections, n_atoms)?;
    let tree = a4_exact(&sections, "TREE_CHAIN_CLASSIFICATION", n_atoms)?;
    let gb_radius = floats_n(&sections, "RADII", n_atoms)?;
    let gb_screen = floats_n(&sections, "SCREEN", n_atoms)?;
    let exclusions = build_exclusions_block(&sections, nnb, n_atoms)?;

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
    if let Some(names) = res_name {
        insert_str_col(&mut atoms, keys::RES_NAME, names)?;
    }
    if let Some(ids) = mol_id {
        insert_uint_col(&mut atoms, keys::MOL_ID, ids)?;
    }
    if let Some(vals) = tree {
        insert_str_col(&mut atoms, "tree", vals)?;
    }
    if let Some(vals) = gb_radius {
        insert_float_col(&mut atoms, "gb_radius", vals)?;
    }
    if let Some(vals) = gb_screen {
        insert_float_col(&mut atoms, "gb_screen", vals)?;
    }
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
    if let Some(excl) = exclusions {
        frame.insert("exclusions", excl);
    }
    apply_meta_scalars(&mut frame, &sections)?;
    apply_box(&mut frame, &sections, ifbox)?;

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

    /// A 16-atom LiTFSI prmtop head (TFSI anion + Li+), hand-written against
    /// the Amber PARM spec <https://ambermd.org/FileFormats.php> — no
    /// AmberTools is involved at test time.
    ///
    /// Self-consistency the goldens lean on, all derived from the text itself:
    /// `NATOM = 16`, `NRES = 2`, `NNB = 65`, `IFBOX = 0`. The bond graph is a
    /// chain `0-1(-2,-3)-4(-5,-6)-7-8(-9,-10)-11(-12,-13,-14)` with atom 15
    /// (Li+) isolated, so the 1-2/1-3/1-4 exclusion counts are
    /// `7 7 5 4 7 3 2 7 6 5 4 3 2 1 1 1`, which sum to exactly 65 with the two
    /// trailing `0` placeholders (atoms 14 and 15 exclude nothing) — hence 63
    /// exclusion rows. `DIHEDRALS_INC_HYDROGEN` is empty, so the 27 torsions
    /// keep `WITHOUT_HYDROGEN` order and the three negative 3rd pointers
    /// (`12 21 -24 …`) land at rows 12, 14 and 16.
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
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       7       7       5       4       7       3       2       7       6       5
       4       3       2       1       1       1
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       2       3       4       5       6       7       8       3       4       5
       6       7       8       9       4       5       6       7       8       5
       6       7       8       6       7       8       9      10      11      12
       7       8       9       8       9       9      10      11      12      13
      14      15      10      11      12      13      14      15      11      12
      13      14      15      12      13      14      15      13      14      15
      14      15      15       0       0
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
TF  LI
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
%FLAG TREE_CHAIN_CLASSIFICATION
%FORMAT(20a4)
E   M   E   E   M   E   E   M   M   E   E   M   E   E   E   BLA
%FLAG JOIN_ARRAY
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0
%FLAG IROTAT
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0
%FLAG SOLVENT_POINTERS
%FORMAT(3I8)
       2       2       3
%FLAG ATOMS_PER_MOLECULE
%FORMAT(10I8)
      15       1
%FLAG RADIUS_SET
%FORMAT(1a80)
modified Bondi radii (mbondi2)
%FLAG RADII
%FORMAT(5E16.8)
  1.50000000E+00  1.70000000E+00  1.50000000E+00  1.50000000E+00  1.80000000E+00
  1.50000000E+00  1.50000000E+00  1.55000000E+00  1.80000000E+00  1.50000000E+00
  1.50000000E+00  1.70000000E+00  1.50000000E+00  1.50000000E+00  1.50000000E+00
  1.50000000E+00
%FLAG SCREEN
%FORMAT(5E16.8)
  8.80000000E-01  7.20000000E-01  8.80000000E-01  8.80000000E-01  9.60000000E-01
  8.50000000E-01  8.50000000E-01  7.90000000E-01  9.60000000E-01  8.50000000E-01
  8.50000000E-01  7.20000000E-01  8.80000000E-01  8.80000000E-01  8.80000000E-01
  8.00000000E-01
";

    fn frame_from(s: &str) -> Frame {
        read_amber_prmtop_from_reader(Cursor::new(s.as_bytes())).expect("parse")
    }

    #[test]
    fn a4_names_chunking() {
        let names =
            prmtop_tables::parse_a4_names(&["F   C   F1  F2  S   O   O3  N   ".to_string()]);
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
    // -----------------------------------------------------------------
    // Fixtures for amber-prmtop-complete-01-structure
    //
    // Goldens are hand-derived from the Amber PARM/prmtop specification
    // (<https://ambermd.org/FileFormats.php>, accessed 2026-09-04) and from
    // the fixture text itself. No AmberTools is involved at test time.
    // -----------------------------------------------------------------

    /// POINTERS line 3 of [`LITFSI_HEAD`] — fields 21..30
    /// (IFPERT, NBPER, NGPER, NDPER, MBPER, MGPER, MDPER, IFBOX, NMXRS, IFCAP).
    const LITFSI_POINTERS_LINE3: &str =
        "       0       0       0       0       0       0       0       0      15       0";

    /// `LITFSI_HEAD` with IFPERT / IFBOX / IFCAP rewritten in POINTERS.
    fn litfsi_with_pointers3(ifpert: i64, ifbox: i64, ifcap: i64) -> String {
        assert_eq!(
            LITFSI_HEAD.matches(LITFSI_POINTERS_LINE3).count(),
            1,
            "POINTERS line 3 must occur exactly once in LITFSI_HEAD"
        );
        let line = format!(
            "{ifpert:>8}{z:>8}{z:>8}{z:>8}{z:>8}{z:>8}{z:>8}{ifbox:>8}{nmxrs:>8}{ifcap:>8}",
            z = 0,
            nmxrs = 15
        );
        LITFSI_HEAD.replace(LITFSI_POINTERS_LINE3, &line)
    }

    /// Drop one `%FLAG <name>` section (flag line, `%FORMAT` line and data).
    ///
    /// The "absent when the section is absent" half of every column assertion.
    fn without_section(text: &str, flag: &str) -> String {
        assert!(
            text.contains(&format!("%FLAG {flag}\n")),
            "fixture does not carry %FLAG {flag}"
        );
        let mut out = String::with_capacity(text.len());
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

    /// `(OLDBETA, bx, by, bz)` for a 30 Å cube — IFBOX=1.
    const BOX_ORTHO: &str = "\
%FLAG BOX_DIMENSIONS
%FORMAT(5E16.8)
  9.00000000E+01  3.00000000E+01  3.00000000E+01  3.00000000E+01
";

    /// `(OLDBETA, bx, by, bz)` for a 30 Å truncated octahedron — IFBOX=2.
    ///
    /// Amber writes OLDBETA truncated to `1.09471219E+02`; the cell is built
    /// from `arccos(-1/3) = 109.4712206344907`, never from this number.
    const BOX_OCTAHEDRON: &str = "\
%FLAG BOX_DIMENSIONS
%FORMAT(5E16.8)
  1.09471219E+02  3.00000000E+01  3.00000000E+01  3.00000000E+01
";

    /// Minimal 4-atom prmtop: a star (atom 1 bonded to 0, 2 and 3) whose
    /// torsion table carries one proper row and two Amber impropers, one of
    /// which also has a negative **3rd** pointer (`exclude_14`).
    ///
    /// `nnb` / `numex` / `excluded` are parameters so the exclusion-block edge
    /// cases reuse one legal topology.
    fn star4_prmtop(nnb: i64, numex: &str, excluded: &str) -> String {
        format!(
            "\
%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
STAR
%FLAG POINTERS
%FORMAT(10I8)
       4       2       0       3       0       3       0       3       0       0
{nnb:>8}       1       3       3       3       1       1       2       2       0
       0       0       0       0       0       0       0       0       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
C1  C2  C3  C4
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       6       6       6       6
%FLAG MASS
%FORMAT(5E16.8)
  1.20100000E+01  1.20100000E+01  1.20100000E+01  1.20100000E+01
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
{numex}
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
{excluded}
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
MOL
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
c3  c3  c3  c3
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       3       6       1       3       9       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1       0       3       9       1       6       3
       9       1
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       9       1       0       3      -6      -9       2
       3       0       6      -9       2
"
        )
    }

    /// Exclusion counts of the 4-atom star: 3, 2, 1 and one `0` placeholder.
    const STAR4_NUMEX: &str = "       3       2       1       1";
    /// The matching `EXCLUDED_ATOMS_LIST` — NNB = 7, one `0` placeholder.
    const STAR4_EXCLUDED: &str = "       2       3       4       3       4       4       0";

    /// The 4-atom star with its true exclusion list.
    fn star4() -> String {
        star4_prmtop(7, STAR4_NUMEX, STAR4_EXCLUDED)
    }

    fn refusal_from(text: &str) -> Error {
        read_amber_prmtop_from_reader(Cursor::new(text.as_bytes()))
            .expect_err("expected the reader to refuse this file")
    }

    /// Every unsupported-feature refusal: `InvalidData`, names the subject and
    /// says `"are not supported"` (a refusal a user cannot act on is not one).
    fn assert_refusal(text: &str, subject: &str) {
        let err = refusal_from(text);
        assert_eq!(
            err.kind(),
            ErrorKind::InvalidData,
            "refusal for {subject:?} must be InvalidData"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(subject),
            "refusal message {msg:?} does not name {subject:?}"
        );
        assert!(
            msg.contains("are not supported"),
            "refusal message {msg:?} does not say \"are not supported\""
        );
    }

    // -----------------------------------------------------------------
    // Per-atom columns (ac-002)
    // -----------------------------------------------------------------

    #[test]
    fn res_name_follows_residue_label() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let res_name = atoms
            .get_string("res_name")
            .expect("atoms block must carry res_name from RESIDUE_LABEL");
        assert_eq!(res_name.len(), 16);
        for i in 0..15 {
            assert_eq!(res_name[[i]], "TF", "atom {i} is in residue TF");
        }
        assert_eq!(res_name[[15]], "LI");
    }

    #[test]
    fn res_name_absent_without_residue_label() {
        let text = without_section(LITFSI_HEAD, "RESIDUE_LABEL");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(
            atoms.get_string("res_name").is_none(),
            "res_name must be absent, never fabricated, when RESIDUE_LABEL is missing"
        );
    }

    #[test]
    fn mol_id_from_atoms_per_molecule() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let mol_id = atoms
            .get_uint("mol_id")
            .expect("atoms block must carry mol_id from ATOMS_PER_MOLECULE");
        assert_eq!(mol_id.len(), 16);
        for i in 0..15 {
            assert_eq!(mol_id[[i]], 1, "atom {i} is in the TFSI molecule");
        }
        assert_eq!(mol_id[[15]], 2);
    }

    #[test]
    fn mol_id_absent_without_atoms_per_molecule() {
        // Amber writes ATOMS_PER_MOLECULE and SOLVENT_POINTERS only for
        // IFBOX > 0, so "no molecule partition" means both are gone. The
        // partition is never re-derived from bond connectivity.
        let text = without_section(LITFSI_HEAD, "ATOMS_PER_MOLECULE");
        let text = without_section(&text, "SOLVENT_POINTERS");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(
            atoms.get_uint("mol_id").is_none(),
            "mol_id must be absent, never inferred from bonds"
        );
    }

    #[test]
    fn tree_from_tree_chain_classification() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let tree = atoms
            .get_string("tree")
            .expect("atoms block must carry tree from TREE_CHAIN_CLASSIFICATION");
        assert_eq!(tree.len(), 16);
        assert_eq!(tree[[0]], "E");
        assert_eq!(tree[[1]], "M");
        assert_eq!(tree[[7]], "M");
        assert_eq!(tree[[15]], "BLA");
    }

    #[test]
    fn tree_absent_without_tree_chain_classification() {
        let text = without_section(LITFSI_HEAD, "TREE_CHAIN_CLASSIFICATION");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(atoms.get_string("tree").is_none());
    }

    #[test]
    fn gb_radius_from_radii() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let r = atoms
            .get_float("gb_radius")
            .expect("atoms block must carry gb_radius from RADII");
        assert_eq!(r.len(), 16);
        assert!((r[[0]] - 1.50).abs() < 1e-12, "F radius");
        assert!((r[[1]] - 1.70).abs() < 1e-12, "C radius");
        assert!((r[[7]] - 1.55).abs() < 1e-12, "N radius");
        assert!((r[[15]] - 1.50).abs() < 1e-12, "Li radius");
    }

    #[test]
    fn gb_radius_absent_without_radii() {
        let text = without_section(LITFSI_HEAD, "RADII");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(atoms.get_float("gb_radius").is_none());
    }

    #[test]
    fn gb_screen_from_screen() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let s = atoms
            .get_float("gb_screen")
            .expect("atoms block must carry gb_screen from SCREEN");
        assert_eq!(s.len(), 16);
        assert!((s[[0]] - 0.88).abs() < 1e-12, "F screen");
        assert!((s[[1]] - 0.72).abs() < 1e-12, "C screen");
        assert!((s[[4]] - 0.96).abs() < 1e-12, "S screen");
        assert!((s[[15]] - 0.80).abs() < 1e-12, "Li screen");
    }

    #[test]
    fn gb_screen_absent_without_screen() {
        let text = without_section(LITFSI_HEAD, "SCREEN");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(atoms.get_float("gb_screen").is_none());
    }

    // -----------------------------------------------------------------
    // exclusions block (ac-003)
    // -----------------------------------------------------------------

    #[test]
    fn exclusions_rows_are_zero_based_ordered_pairs() {
        let frame = frame_from(LITFSI_HEAD);
        let excl = frame
            .get("exclusions")
            .expect("prmtop must produce the exclusions block PME already reads");
        // NNB = 65 entries, two of which are the `0` placeholders of the two
        // atoms that exclude nothing (F3, index 14; Li, index 15).
        assert_eq!(excl.nrows(), Some(63));
        let atomi = excl.get_uint("atomi").expect("exclusions.atomi");
        let atomj = excl.get_uint("atomj").expect("exclusions.atomj");
        for row in 0..63 {
            assert!(
                atomi[[row]] < atomj[[row]],
                "row {row}: {} !< {}",
                atomi[[row]],
                atomj[[row]]
            );
            assert!(
                atomj[[row]] <= 14,
                "row {row} references a non-existent partner"
            );
        }
        // File numbers are 1-based: atom 1 excludes 2..8 -> (0,1)..(0,7).
        assert_eq!((atomi[[0]], atomj[[0]]), (0, 1));
        assert_eq!((atomi[[6]], atomj[[6]]), (0, 7));
        assert_eq!((atomi[[7]], atomj[[7]]), (1, 2));
        assert_eq!((atomi[[62]], atomj[[62]]), (13, 14));
    }

    #[test]
    fn exclusions_all_zero_list_is_an_empty_block() {
        let text = star4_prmtop(
            4,
            "       1       1       1       1",
            "       0       0       0       0",
        );
        let frame = frame_from(&text);
        let excl = frame
            .get("exclusions")
            .expect("an all-placeholder list still yields a schema-typed empty block");
        assert_eq!(excl.nrows(), Some(0));
        assert!(excl.get_uint("atomi").is_some(), "atomi column must exist");
        assert!(excl.get_uint("atomj").is_some(), "atomj column must exist");
    }

    #[test]
    fn exclusions_length_mismatch_names_nnb() {
        // NNB = 7 in POINTERS, but the flattened list carries only 6 entries.
        let text = star4_prmtop(
            7,
            STAR4_NUMEX,
            "       2       3       4       3       4       4",
        );
        let err = refusal_from(&text);
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(msg.contains("NNB"), "message {msg:?} must name NNB");
        assert!(
            msg.contains('7') && msg.contains('6'),
            "message {msg:?} must name both counts"
        );
    }

    // -----------------------------------------------------------------
    // exclude_14 (ac-004)
    // -----------------------------------------------------------------

    #[test]
    fn dihedral_exclude_14_marks_the_negative_third_pointer() {
        let frame = frame_from(LITFSI_HEAD);
        let dihedrals = frame.get("dihedrals").unwrap();
        assert_eq!(dihedrals.nrows(), Some(27));
        let flag = dihedrals
            .get_bool("exclude_14")
            .expect("dihedrals must carry exclude_14");
        let flagged: Vec<usize> = (0..27).filter(|&i| flag[[i]]).collect();
        // DIHEDRALS_INC_HYDROGEN is empty, so WITHOUT_HYDROGEN order stands:
        // rows 12, 14 and 16 are `12 21 -24 {27,30,33} 4`.
        assert_eq!(flagged, vec![12, 14, 16]);
    }

    #[test]
    fn improper_exclude_14_matches_its_dihedral_row() {
        let frame = frame_from(&star4());
        let dihedrals = frame.get("dihedrals").unwrap();
        let impropers = frame
            .get("impropers")
            .expect("a negative 4th pointer mirrors the row into impropers");
        assert_eq!(dihedrals.nrows(), Some(3));
        assert_eq!(impropers.nrows(), Some(2));

        let key = |b: &Block, row: usize| {
            (
                b.get_uint("atomi").unwrap()[[row]],
                b.get_uint("atomj").unwrap()[[row]],
                b.get_uint("atomk").unwrap()[[row]],
                b.get_uint("atoml").unwrap()[[row]],
                b.get_uint("type_id").unwrap()[[row]],
            )
        };
        let d_flag = dihedrals
            .get_bool("exclude_14")
            .expect("dihedrals must carry exclude_14");
        let i_flag = impropers
            .get_bool("exclude_14")
            .expect("impropers must carry exclude_14 from the same builder");
        for irow in 0..impropers.nrows().unwrap() {
            let k = key(impropers, irow);
            let drow = (0..dihedrals.nrows().unwrap())
                .find(|&d| key(dihedrals, d) == k)
                .unwrap_or_else(|| panic!("improper row {irow} has no matching dihedral row"));
            assert_eq!(
                i_flag[[irow]],
                d_flag[[drow]],
                "exclude_14 diverged between the two homes of improper row {irow}"
            );
        }
        // The fixture pins the values, so the parity check cannot pass vacuously.
        assert!(i_flag[[0]], "row `0 3 -6 -9 2` suppresses its 1-4 term");
        assert!(!i_flag[[1]], "row `3 0 6 -9 2` keeps its 1-4 term");
    }

    #[test]
    fn zero_third_pointer_is_atom_zero_and_not_excluded() {
        // Pointer 0 denotes atom 1 (index 0): the *sign*, not the value,
        // carries the 1-4 suppression flag.
        let types: Vec<String> = (0..4).map(|i| format!("t{i}")).collect();
        let rows = decode_dihedrals(&[3, 0, 0, 9, 1], &types).unwrap();
        let block = build_dihedral_block(&rows).unwrap();
        assert_eq!(block.get_uint("atomk").unwrap()[[0]], 0);
        let flag = block
            .get_bool("exclude_14")
            .expect("build_dihedral_block must emit exclude_14");
        assert!(!flag[[0]], "a literal 0 pointer is not a negative pointer");
    }

    // -----------------------------------------------------------------
    // Box (ac-005)
    // -----------------------------------------------------------------

    #[test]
    fn ifbox_1_builds_an_orthogonal_cell() {
        let text = litfsi_with_pointers3(0, 1, 0) + BOX_ORTHO;
        let frame = frame_from(&text);
        let simbox = frame
            .simbox
            .as_ref()
            .expect("IFBOX=1 with BOX_DIMENSIONS must build a SimBox");
        let lengths = simbox.lengths();
        for axis in 0..3 {
            assert!(
                (lengths[axis] - 30.0).abs() < 1e-9,
                "axis {axis} length {} != 30.0",
                lengths[axis]
            );
        }
        assert!(
            (simbox.volume() - 27000.0).abs() < 1e-9,
            "volume {} != 27000.0",
            simbox.volume()
        );
    }

    #[test]
    fn ifbox_2_builds_the_truncated_octahedron_cell() {
        let text = litfsi_with_pointers3(0, 2, 0) + BOX_OCTAHEDRON;
        let frame = frame_from(&text);
        let simbox = frame
            .simbox
            .as_ref()
            .expect("IFBOX=2 with BOX_DIMENSIONS must build a SimBox");
        let h = simbox.h_view();
        // arccos(-1/3) = 109.4712206344907 deg, so 30*cos(alpha) = -10 exactly.
        assert!((h[[0, 0]] - 30.0).abs() < 1e-9, "h[0][0] = {}", h[[0, 0]]);
        assert!((h[[0, 1]] + 10.0).abs() < 1e-9, "h[0][1] = {}", h[[0, 1]]);
        assert!((h[[0, 2]] + 10.0).abs() < 1e-9, "h[0][2] = {}", h[[0, 2]]);
        // 30^3 * sqrt(16/27) = 20784.6096908265...
        assert!(
            (simbox.volume() - 20784.6096908265).abs() < 1e-6,
            "volume {} != 20784.6096908265",
            simbox.volume()
        );
    }

    #[test]
    fn ifbox_3_is_refused() {
        let text = litfsi_with_pointers3(0, 3, 0) + BOX_ORTHO;
        assert_refusal(&text, "IFBOX=3");
    }

    #[test]
    fn ifbox_0_leaves_simbox_none() {
        let frame = frame_from(LITFSI_HEAD);
        assert!(
            frame.simbox.is_none(),
            "IFBOX=0 and no BOX_DIMENSIONS means no cell — nothing is fabricated"
        );
    }

    #[test]
    fn charge_conversion_keeps_the_amber_literal() {
        // 1.82223000E+01 / 18.2223 == 1.0 exactly; re-deriving the factor from
        // molrs' own C = 332.0637133 would shift this by ~1.7e-5.
        let frame = frame_from(LITFSI_HEAD);
        let charge = frame.get("atoms").unwrap().get_float("charge").unwrap();
        assert!(
            (charge[[15]] - 1.0).abs() < 1e-12,
            "Li+ charge {}",
            charge[[15]]
        );
    }

    // -----------------------------------------------------------------
    // meta scalars (ac-006)
    // -----------------------------------------------------------------

    #[test]
    fn meta_radius_set_is_the_verbatim_line() {
        let frame = frame_from(LITFSI_HEAD);
        assert_eq!(
            frame.meta.get("radius_set").and_then(|v| v.as_str()),
            Some("modified Bondi radii (mbondi2)")
        );
    }

    #[test]
    fn meta_solvent_pointers_are_three_integers() {
        let frame = frame_from(LITFSI_HEAD);
        assert_eq!(
            frame.meta.get("solvent_iptres").and_then(|v| v.as_i64()),
            Some(2)
        );
        assert_eq!(
            frame.meta.get("solvent_nspm").and_then(|v| v.as_i64()),
            Some(2)
        );
        assert_eq!(
            frame.meta.get("solvent_nspsol").and_then(|v| v.as_i64()),
            Some(3)
        );
    }

    #[test]
    fn meta_oldbeta_is_the_file_value_not_the_cell_angle() {
        let ortho = frame_from(&(litfsi_with_pointers3(0, 1, 0) + BOX_ORTHO));
        assert_eq!(
            ortho.meta.get("oldbeta").and_then(|v| v.as_f64()),
            Some(90.0)
        );

        let octa = frame_from(&(litfsi_with_pointers3(0, 2, 0) + BOX_OCTAHEDRON));
        let oldbeta = octa
            .meta
            .get("oldbeta")
            .and_then(|v| v.as_f64())
            .expect("oldbeta must be recorded for provenance");
        assert!(
            (oldbeta - 109.4712190).abs() < 1e-9,
            "oldbeta {oldbeta} is not the truncated file value"
        );
        assert!(
            (oldbeta - 109.4712206).abs() > 1e-7,
            "oldbeta must be the file's number, not the arccos(-1/3) used for the cell"
        );
    }

    #[test]
    fn meta_scalars_absent_when_their_sections_are() {
        let text = without_section(LITFSI_HEAD, "RADIUS_SET");
        let text = without_section(&text, "SOLVENT_POINTERS");
        let frame = frame_from(&text);
        assert!(frame.meta.get("radius_set").is_none());
        assert!(frame.meta.get("solvent_iptres").is_none());
        assert!(frame.meta.get("solvent_nspm").is_none());
        assert!(frame.meta.get("solvent_nspsol").is_none());
        assert!(frame.meta.get("oldbeta").is_none());
    }

    // -----------------------------------------------------------------
    // Length-checked and discarded (ac-007)
    // -----------------------------------------------------------------

    #[test]
    fn join_array_and_irotat_produce_no_column() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        for dead in ["join_array", "join", "irotat", "rotat"] {
            assert!(
                atoms.get_uint(dead).is_none() && atoms.get_int(dead).is_none(),
                "{dead} is a dead Amber field: length-checked, then discarded"
            );
        }
    }

    #[test]
    fn join_array_wrong_length_is_refused() {
        let short = "\
%FLAG JOIN_ARRAY
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0
";
        let text = without_section(LITFSI_HEAD, "JOIN_ARRAY") + short;
        let err = refusal_from(&text);
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("JOIN_ARRAY"),
            "message {msg:?} must name the section"
        );
        assert!(
            msg.contains("15") && msg.contains("16"),
            "message {msg:?} must name both counts"
        );
    }

    #[test]
    fn irotat_wrong_length_is_refused() {
        let short = "\
%FLAG IROTAT
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0
";
        let text = without_section(LITFSI_HEAD, "IROTAT") + short;
        let err = refusal_from(&text);
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("IROTAT"),
            "message {msg:?} must name the section"
        );
        assert!(
            msg.contains("15") && msg.contains("16"),
            "message {msg:?} must name both counts"
        );
    }

    #[test]
    fn section_length_mismatch_with_pointers_is_refused() {
        // NRES = 2, but RESIDUE_LABEL carries three 20a4 fields.
        let three = "\
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
TF  LI  XX
";
        let text = without_section(LITFSI_HEAD, "RESIDUE_LABEL") + three;
        let err = refusal_from(&text);
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("RESIDUE_LABEL"),
            "message {msg:?} must name the section"
        );
        assert!(
            msg.contains('3') && msg.contains('2'),
            "message {msg:?} must name both counts"
        );
    }

    // -----------------------------------------------------------------
    // Loud refusals (ac-007)
    // -----------------------------------------------------------------

    #[test]
    fn cmap_terms_are_refused() {
        let cmap = "\
%FLAG CMAP_COUNT
%FORMAT(2I8)
       2       1
";
        assert_refusal(&(LITFSI_HEAD.to_string() + cmap), "CMAP");
    }

    #[test]
    fn perturbed_prmtop_is_refused() {
        assert_refusal(&litfsi_with_pointers3(1, 0, 0), "IFPERT");
    }

    #[test]
    fn solvent_cap_prmtop_is_refused() {
        let cap = "\
%FLAG CAP_INFO
%FORMAT(10I8)
      12
%FLAG CAP_INFO2
%FORMAT(5E16.8)
  8.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
";
        assert_refusal(&(litfsi_with_pointers3(0, 0, 1) + cap), "IFCAP");
    }

    #[test]
    fn polarizable_prmtop_is_refused() {
        let ipol = "\
%FLAG IPOL
%FORMAT(1I8)
       1
";
        assert_refusal(&(LITFSI_HEAD.to_string() + ipol), "IPOL");
    }

    #[test]
    fn chamber_ctitle_prmtop_is_refused() {
        let ctitle = "\
%FLAG CTITLE
%FORMAT(a80)
CHARMM topology written by chamber
";
        assert_refusal(&(LITFSI_HEAD.to_string() + ctitle), "CHAMBER");
    }

    #[test]
    fn chamber_force_field_type_prmtop_is_refused() {
        let fftype = "\
%FLAG FORCE_FIELD_TYPE
%FORMAT(i2,a78)
 1 CHARMM  31
";
        assert_refusal(&(LITFSI_HEAD.to_string() + fftype), "CHAMBER");
    }

    #[test]
    fn negative_nonbonded_parm_index_is_refused() {
        // NTYPES = 6 -> 36 entries; a negative ICO indexes HBOND_ACOEF (10-12).
        let ico = "\
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       2       4       7      11      -1       2       3       5       8
      12      17       4       5       6       9      13      18       7       8
       9      10      14      19      11      12      13      14      15      20
      -1      17      18      19      20      21
";
        assert_refusal(&(LITFSI_HEAD.to_string() + ico), "10-12");
    }

    #[test]
    fn lennard_jones_ccoef_prmtop_is_refused() {
        // NTYPES*(NTYPES+1)/2 = 21 C4 terms of the 12-6-4 model.
        let ccoef = "\
%FLAG LENNARD_JONES_CCOEF
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00
";
        assert_refusal(&(LITFSI_HEAD.to_string() + ccoef), "12-6-4");
    }
}
