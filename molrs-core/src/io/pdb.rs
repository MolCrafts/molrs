//! PDB file format reader and writer.
//!
//! Implements PDB 3.3 specification for coordinate section records:
//! https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html

use crate::block::Block;
use crate::frame::Frame;
use crate::io::reader::{FrameReader, Reader};
use crate::io::writer::FrameWriter;
use crate::region::simbox::SimBox;
use crate::types::{F, U};
use ndarray::{Array1, IxDyn, array};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

// ============================================================================
// Record Structs (per PDB 3.3 spec)
// ============================================================================

/// ATOM record - standard amino acid/nucleotide atoms
/// Columns per spec: 1-6 record, 7-11 serial, 13-16 name, 17 altLoc,
/// 18-20 resName, 22 chainID, 23-26 resSeq, 27 iCode, 31-38 x, 39-46 y,
/// 47-54 z, 55-60 occupancy, 61-66 tempFactor, 77-78 element, 79-80 charge
#[derive(Debug, Clone, PartialEq)]
pub struct AtomRecord {
    pub serial: i32,
    pub name: String,
    pub alt_loc: char,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub i_code: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub occupancy: f32,
    pub temp_factor: f32,
    pub element: String,
    pub charge: String,
}

/// HETATM record - non-polymer chemical atoms (same fields as ATOM)
pub type HetAtmRecord = AtomRecord;

/// CONECT record - bond connectivity
#[derive(Debug, Clone, PartialEq)]
pub struct ConectRecord {
    pub serial: i32,
    pub bonded: Vec<i32>,
}

/// CRYST1 record - unit cell parameters
#[derive(Debug, Clone, PartialEq)]
pub struct Cryst1Record {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub space_group: String,
    pub z: i32,
}

/// MODEL record
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRecord {
    pub serial: i32,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get char at position, defaulting to space
fn char_at(s: &str, idx: usize) -> char {
    s.chars().nth(idx).unwrap_or(' ')
}

/// Get substring safely, returning empty string if out of bounds
fn substr(s: &str, start: usize, end: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    let end = end.min(len);
    &s[start..end]
}

fn infer_element_from_atom_name(name_raw: &str) -> Option<String> {
    let mut chars = name_raw.chars();
    let first = chars.next()?;
    let second = chars.next();

    if first.is_whitespace() {
        if let Some(c) = second
            && c.is_ascii_alphabetic()
        {
            return Some(c.to_ascii_uppercase().to_string());
        }
        return None;
    }

    if first.is_ascii_alphabetic() {
        let mut symbol = String::new();
        symbol.push(first.to_ascii_uppercase());
        if let Some(c) = second
            && c.is_ascii_alphabetic()
        {
            symbol.push(c.to_ascii_lowercase());
        }
        return Some(symbol);
    }

    None
}

// ============================================================================
// Record Parsers
// ============================================================================

/// Internal implementation for parsing ATOM or HETATM records
fn parse_atom_or_hetatm_impl(
    line: &str,
    expected_record: &str,
) -> std::io::Result<Option<AtomRecord>> {
    // Minimum length check
    if line.len() < 54 {
        return Ok(None);
    }

    // Check record type (columns 1-6, 0-indexed: 0-6)
    let record = substr(line, 0, 6).trim();
    if record != expected_record {
        return Ok(None);
    }

    // Parse fixed-width fields (PDB uses 1-indexed columns, we use 0-indexed)
    let serial = substr(line, 6, 11)
        .trim()
        .parse::<i32>()
        .map_err(err_mapper)?;
    let name_raw = substr(line, 12, 16);
    let name = name_raw.trim().to_string();
    let alt_loc = char_at(line, 16);
    let res_name = substr(line, 17, 20).trim().to_string();
    let chain_id = char_at(line, 21);
    let res_seq_str = substr(line, 22, 26).trim();
    if res_seq_str.is_empty() {
        return Err(err_mapper("missing res_seq"));
    }
    let res_seq = res_seq_str.parse::<i32>().map_err(err_mapper)?;
    let i_code = char_at(line, 26);
    let x = substr(line, 30, 38)
        .trim()
        .parse::<f32>()
        .map_err(err_mapper)?;
    let y = substr(line, 38, 46)
        .trim()
        .parse::<f32>()
        .map_err(err_mapper)?;
    let z = substr(line, 46, 54)
        .trim()
        .parse::<f32>()
        .map_err(err_mapper)?;

    // Optional fields
    let occupancy_str = substr(line, 54, 60).trim();
    let occupancy = if occupancy_str.is_empty() {
        1.0
    } else {
        occupancy_str.parse::<f32>().map_err(err_mapper)?
    };
    // make sure occupancy is between 0 and 1
    if !(0.0..=1.0).contains(&occupancy) {
        return Err(err_mapper(
            "occupancy out of range (0.0 - 1.0) in ".to_string() + line,
        ));
    }
    let temp_factor_str = substr(line, 60, 66).trim();
    let temp_factor = if temp_factor_str.is_empty() {
        0.0
    } else {
        temp_factor_str.parse::<f32>().map_err(err_mapper)?
    };
    // make sure temp_factor is non-negative
    if temp_factor < 0.0 {
        return Err(err_mapper("temp_factor negative in ".to_string() + line));
    }

    // Element (columns 77-78, 0-indexed: 76-78)
    let mut element = if line.len() >= 78 {
        substr(line, 76, 78).trim().to_string()
    } else {
        String::new()
    };
    if element.is_empty() {
        if let Some(inferred) = infer_element_from_atom_name(name_raw) {
            element = inferred;
        } else {
            element = "X".to_string();
        }
    }

    // Charge (columns 79-80)
    let charge = if line.len() >= 80 {
        substr(line, 78, 80).trim().to_string()
    } else {
        String::new()
    };

    Ok(Some(AtomRecord {
        serial,
        name,
        alt_loc,
        res_name,
        chain_id,
        res_seq,
        i_code,
        x,
        y,
        z,
        occupancy,
        temp_factor,
        element,
        charge,
    }))
}

/// Parse ATOM record from line
pub fn parse_atom_record(line: &str) -> std::io::Result<Option<AtomRecord>> {
    parse_atom_or_hetatm_impl(line, "ATOM")
}

/// Parse HETATM record from line (same format as ATOM)
pub fn parse_hetatm_record(line: &str) -> std::io::Result<Option<HetAtmRecord>> {
    parse_atom_or_hetatm_impl(line, "HETATM")
}

/// Parse CONECT record
pub fn parse_conect_record(line: &str) -> Option<ConectRecord> {
    if !line.starts_with("CONECT") {
        return None;
    }

    // Parse serial numbers from columns 7-11, 12-16, 17-21, 22-26, 27-31
    let mut bonded = Vec::new();
    let serial = substr(line, 6, 11).trim().parse::<i32>().ok()?;

    // Parse bonded atoms (up to 4 per line in standard CONECT)
    for i in 0..4 {
        let start = 11 + i * 5;
        let end = start + 5;
        if line.len() >= end
            && let Ok(bonded_serial) = substr(line, start, end).trim().parse::<i32>()
            && bonded_serial > 0
        {
            bonded.push(bonded_serial);
        }
    }

    Some(ConectRecord { serial, bonded })
}

/// Parse CRYST1 record
pub fn parse_cryst1_record(line: &str) -> Option<Cryst1Record> {
    if !line.starts_with("CRYST1") {
        return None;
    }

    if line.len() < 54 {
        return None;
    }

    let a = substr(line, 6, 15).trim().parse::<f32>().ok()?;
    let b = substr(line, 15, 24).trim().parse::<f32>().ok()?;
    let c = substr(line, 24, 33).trim().parse::<f32>().ok()?;
    let alpha = substr(line, 33, 40).trim().parse::<f32>().unwrap_or(90.0);
    let beta = substr(line, 40, 47).trim().parse::<f32>().unwrap_or(90.0);
    let gamma = substr(line, 47, 54).trim().parse::<f32>().unwrap_or(90.0);

    let space_group = if line.len() >= 66 {
        substr(line, 55, 66).trim().to_string()
    } else {
        String::new()
    };

    let z = if line.len() >= 70 {
        substr(line, 66, 70).trim().parse::<i32>().unwrap_or(1)
    } else {
        1
    };

    Some(Cryst1Record {
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        space_group,
        z,
    })
}

/// Parse MODEL record
pub fn parse_model_record(line: &str) -> Option<ModelRecord> {
    if !line.starts_with("MODEL") {
        return None;
    }

    let serial = substr(line, 10, 14).trim().parse::<i32>().unwrap_or(1);
    Some(ModelRecord { serial })
}

/// Check if line is ENDMDL
pub fn is_endmdl(line: &str) -> bool {
    line.trim().starts_with("ENDMDL")
}

/// Check if line is END
pub fn is_end(line: &str) -> bool {
    line.trim() == "END"
}

/// Check if line is TER
pub fn is_ter(line: &str) -> bool {
    line.trim().starts_with("TER")
}

// ============================================================================
// Frame Building
// ============================================================================

/// Build a Frame from parsed records
fn to_array_float(vec: Vec<F>, len: usize) -> std::io::Result<ndarray::ArrayD<F>> {
    Ok(Array1::from_vec(vec)
        .into_shape_with_order(IxDyn(&[len]))
        .map_err(err_mapper)?
        .into_dyn())
}

fn to_array_uint(vec: Vec<U>, len: usize) -> std::io::Result<ndarray::ArrayD<U>> {
    Ok(Array1::<U>::from_vec(vec)
        .into_shape_with_order(IxDyn(&[len]))
        .map_err(err_mapper)?
        .into_dyn())
}

fn build_atoms_block(atoms: &[AtomRecord]) -> std::io::Result<(Block, String, HashMap<i32, U>)> {
    let n = atoms.len();
    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut z_vec = Vec::with_capacity(n);
    let mut ids_vec: Vec<U> = Vec::with_capacity(n);
    let mut elements = Vec::with_capacity(n);
    let mut serial_map: HashMap<i32, U> = HashMap::with_capacity(n);

    for (i, atom) in atoms.iter().enumerate() {
        x_vec.push(atom.x as F);
        y_vec.push(atom.y as F);
        z_vec.push(atom.z as F);
        ids_vec.push(atom.serial as U);
        elements.push(if atom.element.trim().is_empty() {
            "X".to_string()
        } else {
            atom.element.clone()
        });
        serial_map.insert(atom.serial, i as U);
    }

    let unique_elements = collect_unique_elements(&elements);

    let mut block = Block::new();
    block
        .insert("x", to_array_float(x_vec, n)?)
        .map_err(err_mapper)?;
    block
        .insert("y", to_array_float(y_vec, n)?)
        .map_err(err_mapper)?;
    block
        .insert("z", to_array_float(z_vec, n)?)
        .map_err(err_mapper)?;
    block
        .insert("id", to_array_uint(ids_vec, n)?)
        .map_err(err_mapper)?;

    let elements_arr = Array1::from_vec(elements)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(err_mapper)?
        .into_dyn();
    block.insert("element", elements_arr).map_err(err_mapper)?;

    Ok((block, unique_elements, serial_map))
}

fn collect_unique_elements(elements: &[String]) -> String {
    let mut unique = Vec::new();
    for elem in elements {
        if !unique.contains(elem) {
            unique.push(elem.clone());
        }
    }
    unique.join("|")
}

fn build_bonds_block(
    conects: &[ConectRecord],
    serial_map: &HashMap<i32, U>,
) -> std::io::Result<Option<Block>> {
    if conects.is_empty() {
        return Ok(None);
    }

    let mut i_indices: Vec<U> = Vec::new();
    let mut j_indices: Vec<U> = Vec::new();

    for conect in conects {
        if let Some(&idx1) = serial_map.get(&conect.serial) {
            for &bonded_serial in &conect.bonded {
                if let Some(&idx2) = serial_map.get(&bonded_serial) {
                    i_indices.push(idx1);
                    j_indices.push(idx2);
                }
            }
        }
    }

    if i_indices.is_empty() {
        return Ok(None);
    }

    let bn = i_indices.len();
    let mut block = Block::new();
    block
        .insert("atomi", to_array_uint(i_indices, bn)?)
        .map_err(err_mapper)?;
    block
        .insert("atomj", to_array_uint(j_indices, bn)?)
        .map_err(err_mapper)?;

    Ok(Some(block))
}

fn add_simbox_from_cryst1(frame: &mut Frame, cryst1: Option<&Cryst1Record>) {
    if let Some(cryst) = cryst1
        && cryst.a > 0.0
        && cryst.b > 0.0
        && cryst.c > 0.0
    {
        let lengths = array![cryst.a as F, cryst.b as F, cryst.c as F];
        let origin = array![0.0 as F, 0.0, 0.0];
        let pbc = [true, true, true];
        if let Ok(simbox) = SimBox::ortho(lengths, origin, pbc) {
            frame.simbox = Some(simbox);
        }
    }
}

pub fn build_frame(
    atoms: &[AtomRecord],
    cryst1: Option<&Cryst1Record>,
    conects: &[ConectRecord],
) -> std::io::Result<Frame> {
    if atoms.is_empty() {
        return Ok(Frame::new());
    }

    let mut frame = Frame::new();

    let (atoms_block, elements_metadata, serial_map) = build_atoms_block(atoms)?;
    frame.insert("atoms", atoms_block);
    frame.meta.insert("elements".to_string(), elements_metadata);

    if let Some(bonds_block) = build_bonds_block(conects, &serial_map)? {
        frame.insert("bonds", bonds_block);
    }

    add_simbox_from_cryst1(&mut frame, cryst1);

    Ok(frame)
}

fn err_mapper<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

// ============================================================================
// Reader (single-frame)
// ============================================================================

/// PDB reader (single-frame).
pub struct PDBReader<R: BufRead> {
    reader: R,
}

impl<R: BufRead> PDBReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    fn read_single_frame(&mut self) -> std::io::Result<Option<Frame>> {
        let mut atoms = Vec::new();
        let mut conects = Vec::new();
        let mut cryst1 = None;
        let mut line = String::new();

        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();

            if is_endmdl(trimmed) || is_end(trimmed) {
                break;
            } else if let Some(cryst) = parse_cryst1_record(&line) {
                cryst1 = Some(cryst);
            } else if let Some(atom) = parse_atom_record(&line)? {
                atoms.push(atom);
            } else if let Some(hetatm) = parse_hetatm_record(&line)? {
                atoms.push(hetatm);
            } else if let Some(conect) = parse_conect_record(&line) {
                conects.push(conect);
            }
        }

        if atoms.is_empty() {
            return Ok(None);
        }

        Ok(Some(build_frame(&atoms, cryst1.as_ref(), &conects)?))
    }
}

impl<R: BufRead> Reader for PDBReader<R> {
    type R = R;
    type Frame = Frame;

    fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R: BufRead> FrameReader for PDBReader<R> {
    fn read_frame(&mut self) -> std::io::Result<Option<Self::Frame>> {
        self.read_single_frame()
    }
}

// ============================================================================
// Writer
// ============================================================================

pub struct PDBWriter<W: Write> {
    writer: W,
}

impl<W: Write> crate::io::writer::Writer for PDBWriter<W> {
    type W = W;
    type FrameLike = Frame;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for PDBWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> std::io::Result<()> {
        write_pdb_frame(&mut self.writer, frame)
    }
}

impl<W: Write> PDBWriter<W> {
    /// Create a new PDB writer from an output sink.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

pub fn write_pdb_frame<W: Write>(writer: &mut W, _frame: &Frame) -> std::io::Result<()> {
    let atoms = _frame
        .get("atoms")
        .ok_or_else(|| err_mapper("Missing 'atoms' block"))?;
    let n = atoms
        .nrows()
        .ok_or_else(|| err_mapper("Empty 'atoms' block"))?;

    let x = atoms
        .get_float("x")
        .ok_or_else(|| err_mapper("Missing 'x' column"))?;
    let y = atoms
        .get_float("y")
        .ok_or_else(|| err_mapper("Missing 'y' column"))?;
    let z = atoms
        .get_float("z")
        .ok_or_else(|| err_mapper("Missing 'z' column"))?;

    let x_slice = x
        .as_slice_memory_order()
        .ok_or_else(|| err_mapper("Non-contiguous 'x' column"))?;
    let y_slice = y
        .as_slice_memory_order()
        .ok_or_else(|| err_mapper("Non-contiguous 'y' column"))?;
    let z_slice = z
        .as_slice_memory_order()
        .ok_or_else(|| err_mapper("Non-contiguous 'z' column"))?;

    let elements = atoms.get_string("element");
    let elements_slice = elements.and_then(|arr| arr.as_slice_memory_order());

    let ids_u32 = atoms.get_uint("id");
    let ids_u32_slice = ids_u32.and_then(|arr| arr.as_slice_memory_order());

    // Write CRYST1 from SimBox if present
    if let Some(simbox) = &_frame.simbox {
        let lengths = simbox.lengths();
        writeln!(
            writer,
            "CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} {:<11}{:>4}",
            lengths[0], lengths[1], lengths[2], 90.00, 90.00, 90.00, "P 1", 1
        )?;
    }

    let mut serials = Vec::with_capacity(n);
    for i in 0..n {
        let serial = ids_u32_slice.map(|ids| ids[i] as usize).unwrap_or(i + 1);
        serials.push(serial);
        let elem_raw = elements_slice
            .and_then(|elts| elts.get(i))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("X");
        let name = format!("{:<4}", elem_raw);
        let elem_field = format!("{:>2}", elem_raw);

        writeln!(
            writer,
            "ATOM  {:>5} {} MOL A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00           {}",
            serial, name, 1, x_slice[i], y_slice[i], z_slice[i], elem_field
        )?;
    }

    if let Some(bonds) = _frame.get("bonds") {
        let bn = bonds.nrows().unwrap_or(0);
        if bn > 0 {
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

            let (u32_i, u32_j) =
                if let (Some(i), Some(j)) = (bonds.get_uint("atomi"), bonds.get_uint("atomj")) {
                    (Some(i), Some(j))
                } else {
                    (None, None)
                };

            if let (Some(i_arr), Some(j_arr)) = (u32_i, u32_j) {
                let i_slice = i_arr
                    .as_slice_memory_order()
                    .ok_or_else(|| err_mapper("Non-contiguous bonds 'atom_i'/'i' column"))?;
                let j_slice = j_arr
                    .as_slice_memory_order()
                    .ok_or_else(|| err_mapper("Non-contiguous bonds 'atom_j'/'j' column"))?;

                for b in 0..bn {
                    let idx_i = i_slice[b] as usize;
                    let idx_j = j_slice[b] as usize;
                    if idx_i >= n || idx_j >= n {
                        return Err(err_mapper("Bond index out of range for atoms"));
                    }
                    adj[idx_i].push(serials[idx_j]);
                    adj[idx_j].push(serials[idx_i]);
                }
            } else {
                return Err(err_mapper(
                    "Bonds block missing 'atom_i'/'atom_j' or 'i'/'j' columns",
                ));
            }

            for (atom_idx, neighbors) in adj.iter_mut().enumerate() {
                if neighbors.is_empty() {
                    continue;
                }
                neighbors.sort_unstable();
                neighbors.dedup();

                let serial = serials[atom_idx];
                for chunk in neighbors.chunks(4) {
                    write!(writer, "CONECT{:>5}", serial)?;
                    for &bond_serial in chunk {
                        write!(writer, "{:>5}", bond_serial)?;
                    }
                    writeln!(writer)?;
                }
            }
        }
    }

    writeln!(writer, "END")?;
    Ok(())
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Read a single frame from a PDB file
///
/// # Examples
///
/// ```no_run
/// use molrs::io::pdb::read_pdb_frame;
///
/// # fn main() -> std::io::Result<()> {
/// let frame = read_pdb_frame("protein.pdb")?;
/// # Ok(())
/// # }
/// ```
pub fn read_pdb_frame<P: AsRef<Path>>(path: P) -> std::io::Result<Frame> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut pdb_reader = PDBReader::new(reader);
    pdb_reader.read_frame()?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "No frame found in PDB file",
        )
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ATOM_LINE: &str =
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N  ";
    const SAMPLE_HETATM_LINE: &str =
        "HETATM  100  O   HOH A 501      10.000  20.000  30.000  1.00  0.00           O  ";
    const SAMPLE_HETATM_LINE_NO_OCC: &str =
        "HETATM    1  O   HOH     1      10.203   7.604  12.673";
    const SAMPLE_CRYST1_LINE: &str =
        "CRYST1   50.000   60.000   70.000  90.00  90.00  90.00 P 1           1";
    const SAMPLE_CONECT_LINE: &str = "CONECT    1    2    3    4";

    #[test]
    fn test_parse_atom_record() {
        let atom = parse_atom_record(SAMPLE_ATOM_LINE)
            .expect("Failed to parse ATOM")
            .expect("Missing ATOM record");
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "N");
        assert_eq!(atom.res_name, "ALA");
        assert_eq!(atom.chain_id, 'A');
        assert_eq!(atom.res_seq, 1);
        assert!((atom.x - 1.0).abs() < 0.001);
        assert!((atom.y - 2.0).abs() < 0.001);
        assert!((atom.z - 3.0).abs() < 0.001);
        assert!((atom.occupancy - 1.0).abs() < 0.001);
        assert!((atom.temp_factor - 20.0).abs() < 0.001);
        assert_eq!(atom.element, "N");
    }

    #[test]
    fn test_parse_hetatm_record() {
        let atom = parse_hetatm_record(SAMPLE_HETATM_LINE)
            .expect("Failed to parse HETATM")
            .expect("Missing HETATM record");
        assert_eq!(atom.serial, 100);
        assert_eq!(atom.name, "O");
        assert_eq!(atom.res_name, "HOH");
        assert!((atom.x - 10.0).abs() < 0.001);
        assert_eq!(atom.element, "O");
    }

    #[test]
    fn test_parse_hetatm_record_missing_occupancy_temp() {
        let atom = parse_hetatm_record(SAMPLE_HETATM_LINE_NO_OCC)
            .expect("Failed to parse HETATM")
            .expect("Missing HETATM record");
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "O");
        assert!((atom.occupancy - 1.0).abs() < 0.001);
        assert!((atom.temp_factor - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_cryst1_record() {
        let cryst = parse_cryst1_record(SAMPLE_CRYST1_LINE).expect("Failed to parse CRYST1");
        assert!((cryst.a - 50.0).abs() < 0.001);
        assert!((cryst.b - 60.0).abs() < 0.001);
        assert!((cryst.c - 70.0).abs() < 0.001);
        assert!((cryst.alpha - 90.0).abs() < 0.001);
        assert_eq!(cryst.space_group, "P 1");
        assert_eq!(cryst.z, 1);
    }

    #[test]
    fn test_parse_conect_record() {
        let conect = parse_conect_record(SAMPLE_CONECT_LINE).expect("Failed to parse CONECT");
        assert_eq!(conect.serial, 1);
        assert_eq!(conect.bonded, vec![2, 3, 4]);
    }

    #[test]
    fn test_write_pdb_frame_conect_from_bonds() {
        use crate::block::Block;
        use crate::frame::Frame;
        use ndarray::{Array1, IxDyn};

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        let n = 3;

        let x = Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let y = Array1::from_vec(vec![0.0 as F, 0.0 as F, 0.0 as F])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let z = Array1::from_vec(vec![0.0 as F, 0.0 as F, 0.0 as F])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let elements = Array1::from_vec(vec!["C".to_string(), "O".to_string(), "N".to_string()])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let ids = Array1::from_vec(vec![10 as U, 20 as U, 30 as U])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();

        atoms.insert("x", x).unwrap();
        atoms.insert("y", y).unwrap();
        atoms.insert("z", z).unwrap();
        atoms.insert("element", elements).unwrap();
        atoms.insert("id", ids).unwrap();
        frame.insert("atoms", atoms);

        let mut bonds = Block::new();
        let atom_i = Array1::from_vec(vec![0 as U, 1 as U])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        let atom_j = Array1::from_vec(vec![2 as U, 2 as U])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        bonds.insert("atomi", atom_i).unwrap();
        bonds.insert("atomj", atom_j).unwrap();
        frame.insert("bonds", bonds);

        let mut out = Vec::new();
        write_pdb_frame(&mut out, &frame).expect("write pdb");
        let output = String::from_utf8(out).expect("utf8");
        assert!(output.contains("CONECT   10   30"));
        assert!(output.contains("CONECT   20   30"));
        assert!(output.contains("CONECT   30   10   20"));
    }

    #[test]
    fn test_parse_pdb_missing_element_infers_from_name() {
        let pdb_content = r#"ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00 20.00
ATOM      2 FE   HEM A   1       2.000   3.000   4.000  1.00 20.00
END
"#;

        let mut reader = PDBReader::new(std::io::Cursor::new(pdb_content));
        let frame = reader.read_frame().expect("IO error").expect("No frame");

        let atom_block = frame.get("atoms").expect("No atoms block");
        let elements = atom_block.get_string("element").expect("No element column");
        assert_eq!(elements[[0]], "C");
        assert_eq!(elements[[1]], "Fe");
    }

    #[test]
    fn test_atom_line_short() {
        // Line too short should return None
        let short_line = "ATOM      1  N";
        assert!(
            parse_atom_record(short_line)
                .expect("Parse error")
                .is_none()
        );
    }

    #[test]
    fn test_read_real_pdb_water() {
        let test_data_dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tests-data");
        if !test_data_dir.exists() {
            panic!("Test data not found. Run: scripts/fetch-test-data.sh");
        }
        let test_data_dir = test_data_dir.to_str().unwrap();
        let path = format!("{}/pdb/water.pdb", test_data_dir);
        let frame = read_pdb_frame(&path).expect("Failed to open water.pdb");

        let atoms = frame.get("atoms").expect("No atoms block");
        assert!(atoms.nrows().unwrap() > 0, "Should have atoms");

        // Water should have O and H elements
        let elements = atoms.get_string("element").expect("No element column");
        let has_oxygen = elements.iter().any(|e| e == "O");
        let has_hydrogen = elements.iter().any(|e| e == "H");
        assert!(has_oxygen, "Water should have oxygen");
        assert!(has_hydrogen, "Water should have hydrogen");
    }

    #[test]
    fn test_read_real_pdb_with_bonds() {
        let test_data_dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tests-data");
        if !test_data_dir.exists() {
            panic!("Test data not found. Run: scripts/fetch-test-data.sh");
        }
        let test_data_dir = test_data_dir.to_str().unwrap();
        let path = format!("{}/pdb/1avg.pdb", test_data_dir);
        let frame = read_pdb_frame(&path).expect("Failed to open 1avg.pdb");

        let atoms = frame.get("atoms").expect("No atoms block");
        assert!(atoms.nrows().unwrap() > 0, "Should have atoms");

        // Check if bonds exist (CONECT records)
        if let Some(bonds) = frame.get("bonds") {
            assert!(bonds.nrows().unwrap() > 0, "Should have bonds");

            // Bonds should have i and j columns
            let i_atoms = bonds.get_uint("atomi").expect("No atomi column");
            let j_atoms = bonds.get_uint("atomj").expect("No atomj column");
            assert_eq!(i_atoms.len(), j_atoms.len(), "Bond arrays should match");
        }
    }
}
