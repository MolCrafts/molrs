//! GROMACS topology (`.top` / `.itp`) **structure** reader and writer.
//!
//! Structure only — not force-field parameter tables. Sections honoured:
//!
//! - `[ atoms ]` → `"atoms"` block
//! - `[ bonds ]` / `[ pairs ]` / `[ angles ]` / `[ dihedrals ]` → same-named blocks
//!
//! `#include` directives are **not** expanded (skipped like other `#` lines).
//! Atom indices in connectivity blocks are kept **1-based as written in the
//! file** (GROMACS convention; matches the historical molpy contract).
//!
//! ## Output Frame
//!
//! - `"atoms"`: `id` (uint, 1-based), `type` (str), `resnr` (int), `residu` (str),
//!   `name` (str), `cgnr` (int), `charge` (float), `mass` (float),
//!   `atomic_number` (uint, guessed from `name` / `type`).
//! - connectivity blocks: `atomi`/`atomj`[/`atomk`/`atoml`] (uint, 1-based),
//!   `type` (str — the GROMACS `funct` field).

use std::io::{BufRead, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use ndarray::{Array1, IxDyn};

use molrs::Element;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, I, U};

use crate::io::reader::{FrameReader, Reader};
use crate::io::writer::{FrameWriter, Writer};

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

fn insert_int_col(block: &mut Block, key: &str, vals: Vec<I>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    block.insert(key, arr).map_err(invalid_data)
}

fn insert_uint_col(block: &mut Block, key: &str, vals: Vec<U>) -> Result<()> {
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

/// Strip a GROMACS inline `;` comment and trim.
fn strip_comment(line: &str) -> &str {
    let cut = line.find(';').unwrap_or(line.len());
    line[..cut].trim()
}

/// Normalize a section header to lowercase inner text, e.g. `"[ atoms ]"`.
fn parse_section_header(line: &str) -> Option<String> {
    let t = line.trim();
    if t.starts_with('[') && t.ends_with(']') {
        let inner = t[1..t.len() - 1].trim().to_ascii_lowercase();
        Some(format!("[ {inner} ]"))
    } else {
        None
    }
}

/// Guess atomic number from atom name, falling back to atom type letters.
///
/// Unknown symbols yield `0` (matches the historical molpy TopReader contract).
fn guess_atomic_number(name: &str, atom_type: &str) -> U {
    let letters: String = name.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if !letters.is_empty() {
        if let Some(e) = Element::by_symbol(&letters) {
            return e.z() as U;
        }
        if letters.len() > 1 {
            let mut chars = letters.chars();
            let title: String = chars
                .next()
                .map(|c| c.to_ascii_uppercase())
                .into_iter()
                .chain(chars.flat_map(|c| c.to_lowercase()))
                .collect();
            if let Some(e) = Element::by_symbol(&title) {
                return e.z() as U;
            }
        }
    }
    let type_letters: String = atom_type
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();
    if !type_letters.is_empty()
        && let Some(e) = Element::by_symbol(&type_letters)
    {
        return e.z() as U;
    }
    0
}

// ---------------------------------------------------------------------------
// Parsed records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TopAtom {
    id: U,
    atype: String,
    resnr: I,
    residu: String,
    name: String,
    cgnr: I,
    charge: F,
    mass: F,
}

#[derive(Debug, Clone)]
struct Conn2 {
    atomi: U,
    atomj: U,
    funct: String,
}

#[derive(Debug, Clone)]
struct Conn3 {
    atomi: U,
    atomj: U,
    atomk: U,
    funct: String,
}

#[derive(Debug, Clone)]
struct Conn4 {
    atomi: U,
    atomj: U,
    atomk: U,
    atoml: U,
    funct: String,
}

fn parse_atom_line(line: &str) -> Option<TopAtom> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 8 {
        return None;
    }
    Some(TopAtom {
        id: parts[0].parse().ok()?,
        atype: parts[1].to_string(),
        resnr: parts[2].parse().ok()?,
        residu: parts[3].to_string(),
        name: parts[4].to_string(),
        cgnr: parts[5].parse().ok()?,
        charge: parts[6].parse().ok()?,
        mass: parts[7].parse().ok()?,
    })
}

fn parse_conn2(line: &str) -> Option<Conn2> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    Some(Conn2 {
        atomi: parts[0].parse().ok()?,
        atomj: parts[1].parse().ok()?,
        funct: parts[2].to_string(),
    })
}

fn parse_conn3(line: &str) -> Option<Conn3> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 {
        return None;
    }
    Some(Conn3 {
        atomi: parts[0].parse().ok()?,
        atomj: parts[1].parse().ok()?,
        atomk: parts[2].parse().ok()?,
        funct: parts[3].to_string(),
    })
}

fn parse_conn4(line: &str) -> Option<Conn4> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    Some(Conn4 {
        atomi: parts[0].parse().ok()?,
        atomj: parts[1].parse().ok()?,
        atomk: parts[2].parse().ok()?,
        atoml: parts[3].parse().ok()?,
        funct: parts[4].to_string(),
    })
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read a GROMACS topology structure file into a [`Frame`].
///
/// `#include` is not expanded. Connectivity atom indices stay 1-based.
pub fn read_top<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut reader = std::io::BufReader::new(file);
    read_top_frame(&mut reader)
}

/// Parse topology text from any [`BufRead`] into a [`Frame`].
pub fn read_top_frame<R: BufRead>(reader: &mut R) -> Result<Frame> {
    let mut atoms: Vec<TopAtom> = Vec::new();
    let mut bonds: Vec<Conn2> = Vec::new();
    let mut pairs: Vec<Conn2> = Vec::new();
    let mut angles: Vec<Conn3> = Vec::new();
    let mut dihedrals: Vec<Conn4> = Vec::new();

    let mut section: Option<String> = None;
    let mut buf = String::new();

    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let line = strip_comment(&buf);
        if line.is_empty() {
            continue;
        }
        if let Some(sec) = parse_section_header(line) {
            section = Some(sec);
            continue;
        }
        // Directives (#include, #ifdef, …) and bare comments already stripped.
        if line.starts_with('#') {
            continue;
        }
        match section.as_deref() {
            Some("[ atoms ]") => {
                if let Some(a) = parse_atom_line(line) {
                    atoms.push(a);
                }
            }
            Some("[ bonds ]") => {
                if let Some(b) = parse_conn2(line) {
                    bonds.push(b);
                }
            }
            Some("[ pairs ]") => {
                if let Some(p) = parse_conn2(line) {
                    pairs.push(p);
                }
            }
            Some("[ angles ]") => {
                if let Some(a) = parse_conn3(line) {
                    angles.push(a);
                }
            }
            Some("[ dihedrals ]") => {
                if let Some(d) = parse_conn4(line) {
                    dihedrals.push(d);
                }
            }
            _ => {}
        }
    }

    build_frame(atoms, bonds, pairs, angles, dihedrals)
}

fn build_frame(
    atoms: Vec<TopAtom>,
    bonds: Vec<Conn2>,
    pairs: Vec<Conn2>,
    angles: Vec<Conn3>,
    dihedrals: Vec<Conn4>,
) -> Result<Frame> {
    let mut frame = Frame::new();

    if !atoms.is_empty() {
        let n = atoms.len();
        let mut block = Block::new();
        let mut id = Vec::with_capacity(n);
        let mut atype = Vec::with_capacity(n);
        let mut resnr = Vec::with_capacity(n);
        let mut residu = Vec::with_capacity(n);
        let mut name = Vec::with_capacity(n);
        let mut cgnr = Vec::with_capacity(n);
        let mut charge = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut atomic_number = Vec::with_capacity(n);
        for a in &atoms {
            id.push(a.id);
            atype.push(a.atype.clone());
            resnr.push(a.resnr);
            residu.push(a.residu.clone());
            name.push(a.name.clone());
            cgnr.push(a.cgnr);
            charge.push(a.charge);
            mass.push(a.mass);
            atomic_number.push(guess_atomic_number(&a.name, &a.atype));
        }
        insert_uint_col(&mut block, "id", id)?;
        insert_str_col(&mut block, "type", atype)?;
        insert_int_col(&mut block, "resnr", resnr)?;
        insert_str_col(&mut block, "residu", residu)?;
        insert_str_col(&mut block, "name", name)?;
        insert_int_col(&mut block, "cgnr", cgnr)?;
        insert_float_col(&mut block, "charge", charge)?;
        insert_float_col(&mut block, "mass", mass)?;
        insert_uint_col(&mut block, "atomic_number", atomic_number)?;
        frame.insert("atoms", block);
    }

    if !bonds.is_empty() {
        frame.insert("bonds", conn2_block(&bonds)?);
    }
    if !pairs.is_empty() {
        frame.insert("pairs", conn2_block(&pairs)?);
    }
    if !angles.is_empty() {
        frame.insert("angles", conn3_block(&angles)?);
    }
    if !dihedrals.is_empty() {
        frame.insert("dihedrals", conn4_block(&dihedrals)?);
    }

    Ok(frame)
}

fn conn2_block(rows: &[Conn2]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut funct = Vec::with_capacity(n);
    for r in rows {
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        funct.push(r.funct.clone());
    }
    insert_uint_col(&mut block, "atomi", atomi)?;
    insert_uint_col(&mut block, "atomj", atomj)?;
    // Schema key `type` is string (force-field / funct label).
    insert_str_col(&mut block, "type", funct)?;
    Ok(block)
}

fn conn3_block(rows: &[Conn3]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut atomk = Vec::with_capacity(n);
    let mut funct = Vec::with_capacity(n);
    for r in rows {
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        atomk.push(r.atomk);
        funct.push(r.funct.clone());
    }
    insert_uint_col(&mut block, "atomi", atomi)?;
    insert_uint_col(&mut block, "atomj", atomj)?;
    insert_uint_col(&mut block, "atomk", atomk)?;
    insert_str_col(&mut block, "type", funct)?;
    Ok(block)
}

fn conn4_block(rows: &[Conn4]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut atomk = Vec::with_capacity(n);
    let mut atoml = Vec::with_capacity(n);
    let mut funct = Vec::with_capacity(n);
    for r in rows {
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        atomk.push(r.atomk);
        atoml.push(r.atoml);
        funct.push(r.funct.clone());
    }
    insert_uint_col(&mut block, "atomi", atomi)?;
    insert_uint_col(&mut block, "atomj", atomj)?;
    insert_uint_col(&mut block, "atomk", atomk)?;
    insert_uint_col(&mut block, "atoml", atoml)?;
    insert_str_col(&mut block, "type", funct)?;
    Ok(block)
}

/// `FrameReader` wrapper (single-frame topology file).
pub struct TopReader<R: BufRead> {
    reader: R,
    done: bool,
}

impl<R: BufRead> Reader for TopReader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self {
            reader,
            done: false,
        }
    }
}

impl<R: BufRead> FrameReader for TopReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let frame = read_top_frame(&mut self.reader)?;
        crate::io::reader::validated(Some(frame))
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Frame as a minimal GROMACS topology structure file.
pub fn write_top<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    write_top_frame(&mut w, frame)?;
    w.flush()
}

/// Emit one Frame in GROMACS topology structure form.
pub fn write_top_frame<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    let mol_name = frame
        .meta
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| frame.meta.get("title").and_then(|v| v.as_str()))
        .unwrap_or("MOL");

    writeln!(writer, "; Generated by molrs TopWriter")?;
    writeln!(writer)?;
    writeln!(writer, "[ moleculetype ]")?;
    writeln!(writer, "; name  nrexcl")?;
    writeln!(writer, "{}  3", mol_name)?;
    writeln!(writer)?;

    if let Some(atoms) = frame.get("atoms") {
        let n = atoms
            .nrows()
            .ok_or_else(|| invalid_data("atoms block has no rows"))?;
        writeln!(writer, "[ atoms ]")?;
        writeln!(
            writer,
            ";  nr  type  resnr  residu  atom  cgnr  charge  mass"
        )?;

        let id_col = atoms.get_uint("id");
        let type_col = atoms.get_string("type");
        // Free-form `resnr` (int) preferred; canonical `res_id` (uint) as fallback.
        let resnr_i = atoms.get_int("resnr");
        let resnr_u = atoms.get_uint("res_id");
        let residu_col = atoms
            .get_string("residu")
            .or_else(|| atoms.get_string("res_name"));
        let name_col = atoms.get_string("name");
        let cgnr_col = atoms.get_int("cgnr");
        let charge_col = atoms.get_float("charge");
        let mass_col = atoms.get_float("mass");

        for i in 0..n {
            let aid = id_col.map(|c| c[[i]]).unwrap_or((i as U) + 1);
            let atype = type_col.map(|c| c[[i]].as_str()).unwrap_or("X");
            let resnr = resnr_i
                .map(|c| c[[i]])
                .or_else(|| resnr_u.map(|c| c[[i]] as I))
                .unwrap_or(1);
            let residu = residu_col.map(|c| c[[i]].as_str()).unwrap_or(mol_name);
            let name = name_col.map(|c| c[[i]].as_str()).unwrap_or(atype);
            let cgnr = cgnr_col.map(|c| c[[i]]).unwrap_or((i as I) + 1);
            let charge = charge_col.map(|c| c[[i]]).unwrap_or(0.0);
            let mass = mass_col.map(|c| c[[i]]).unwrap_or(0.0);
            writeln!(
                writer,
                "  {}  {}  {}  {}  {}  {}  {:.4}  {:.3}",
                aid, atype, resnr, residu, name, cgnr, charge, mass
            )?;
        }
        writeln!(writer)?;
    }

    write_index_section(writer, frame, "bonds", "bonds", &["atomi", "atomj"])?;
    write_index_section(writer, frame, "pairs", "pairs", &["atomi", "atomj"])?;
    write_index_section(
        writer,
        frame,
        "angles",
        "angles",
        &["atomi", "atomj", "atomk"],
    )?;
    write_index_section(
        writer,
        frame,
        "dihedrals",
        "dihedrals",
        &["atomi", "atomj", "atomk", "atoml"],
    )?;

    writeln!(writer, "[ system ]")?;
    writeln!(writer, "{}", mol_name)?;
    writeln!(writer)?;
    writeln!(writer, "[ molecules ]")?;
    writeln!(writer, "{}  1", mol_name)?;
    writeln!(writer)?;
    Ok(())
}

fn write_index_section<W: Write>(
    writer: &mut W,
    frame: &Frame,
    section_name: &str,
    block_key: &str,
    columns: &[&str],
) -> Result<()> {
    let Some(block) = frame.get(block_key) else {
        return Ok(());
    };
    let n = match block.nrows() {
        Some(n) if n > 0 => n,
        _ => return Ok(()),
    };

    writeln!(writer, "[ {} ]", section_name)?;
    let header = columns
        .iter()
        .copied()
        .chain(std::iter::once("funct"))
        .collect::<Vec<_>>()
        .join("  ");
    writeln!(writer, "; {}", header)?;

    let col_views: Vec<_> = columns
        .iter()
        .map(|c| {
            block
                .get_uint(c)
                .ok_or_else(|| invalid_data(format!("{block_key}.{c} missing or not uint")))
        })
        .collect::<Result<Vec<_>>>()?;

    let type_str = block.get_string("type");
    let type_id = block.get_uint("type_id");

    for i in 0..n {
        let vals: Vec<String> = col_views.iter().map(|c| c[[i]].to_string()).collect();
        let funct = type_str
            .and_then(|c| c[[i]].parse::<i64>().ok())
            .or_else(|| type_id.map(|c| c[[i]] as i64))
            .unwrap_or(1);
        writeln!(writer, "  {}  {}", vals.join("  "), funct)?;
    }
    writeln!(writer)?;
    Ok(())
}

/// `FrameWriter` wrapper.
pub struct TopFrameWriter<W: Write> {
    writer: W,
}

impl<W: Write> Writer for TopFrameWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for TopFrameWriter<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        crate::io::writer::check_before_write(frame)?;
        write_top_frame(&mut self.writer, frame)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const BENZENE_ATOMS: &str = r#"; comment
#include "forcefield.itp"
[ moleculetype ]
benzene  3

[ atoms ]
; nr type resnr residu atom cgnr charge mass
1  opls_145  1  LIG  C  1  -0.115  12.011
2  opls_145  1  LIG  C  2  -0.115  12.011

[ bonds ]
1  2  1
"#;

    #[test]
    fn reads_atoms_and_bonds_one_based() {
        let frame = read_top_frame(&mut Cursor::new(BENZENE_ATOMS.as_bytes())).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(2));
        assert_eq!(atoms.get_string("type").unwrap()[[0]], "opls_145");
        assert!((atoms.get_float("charge").unwrap()[[0]] + 0.115).abs() < 1e-9);
        let bonds = frame.get("bonds").unwrap();
        assert_eq!(bonds.nrows(), Some(1));
        assert_eq!(bonds.get_uint("atomi").unwrap()[[0]], 1);
        assert_eq!(bonds.get_uint("atomj").unwrap()[[0]], 2);
        assert_eq!(bonds.get_string("type").unwrap()[[0]], "1");
        assert_eq!(atoms.get_uint("atomic_number").unwrap()[[0]], 6);
    }

    #[test]
    fn section_header_without_spaces() {
        let text = "[atoms]\n1  CT  1  MOL  C  1  -0.1  12.011\n[bonds]\n";
        let frame = read_top_frame(&mut Cursor::new(text.as_bytes())).unwrap();
        assert_eq!(frame.get("atoms").unwrap().nrows(), Some(1));
    }

    #[test]
    fn empty_file_yields_empty_frame() {
        let frame = read_top_frame(&mut Cursor::new(b"; just a comment\n")).unwrap();
        assert!(frame.get("atoms").is_none());
    }

    #[test]
    fn round_trip_minimal() {
        let text = r#"
[ moleculetype ]
MOL  3
[ atoms ]
1  CT  1  MOL  C  1  -0.1000  12.011
2  HC  1  MOL  H  2  0.1000  1.008
[ bonds ]
1  2  1
"#;
        let frame = read_top_frame(&mut Cursor::new(text.as_bytes())).unwrap();
        let mut buf = Vec::new();
        write_top_frame(&mut buf, &frame).unwrap();
        let frame2 = read_top_frame(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(frame2.get("atoms").unwrap().nrows(), Some(2));
        assert_eq!(frame2.get("bonds").unwrap().nrows(), Some(1));
        assert_eq!(
            frame2.get("bonds").unwrap().get_uint("atomi").unwrap()[[0]],
            1
        );
    }
}
