//! GROMACS GRO structure / trajectory reader and writer.
//!
//! GRO is a fixed-column text format used by GROMACS for input structures and
//! single-precision trajectories. One frame layout:
//!
//! ```text
//! line 1   : title comment                        → frame.meta["title"]
//! line 2   : atom count `n` (decimal integer)
//! line 3..3+n : atom records (fixed columns; see below)
//! line 3+n : box vectors (3 floats orthorhombic; 9 floats triclinic; nm)
//! ```
//!
//! Multi-frame `.gro` files concatenate this layout. [`GroReader::read_frame`]
//! returns one frame per call.
//!
//! ## Atom record columns (1-indexed)
//!
//! | Cols | Meaning                | Example       |
//! |------|------------------------|---------------|
//! | 1-5  | Residue number (i32)   | `    1`       |
//! | 6-10 | Residue name (str)     | `LIG  `       |
//! | 11-15| Atom name (str)        | `   CA`       |
//! | 16-20| Atom number (i32)      | `    1`       |
//! | 21-28| x (nm, %8.3f)          | `   0.310`    |
//! | 29-36| y                      | `   0.862`    |
//! | 37-44| z                      | `   1.316`    |
//! | 45-52| vx (optional, %8.4f)   |               |
//! | 53-60| vy                     |               |
//! | 61-68| vz                     |               |
//!
//! ## GRO triclinic box convention (line 3+n)
//!
//! Tokens, in file order: `v1x v2y v3z v1y v1z v2x v2z v3x v3y`. When only 3
//! tokens are present, the box is orthorhombic: off-diagonals = 0.
//!
//! ## Output Frame
//!
//! - `"atoms"` block: `res_id` (uint), `resname` (str), `atom_name` (str),
//!   `element` (str, inferred — see [`element_from_atom_name`]), `id` (uint),
//!   `x`/`y`/`z` (F, **Å**), and optional `vx`/`vy`/`vz` (F, **Å/ps**).
//! - `frame.simbox`: triclinic [`SimBox`] from the box-vector line, in Å.
//! - `frame.meta["title"]`.
//!
//! ## Units
//!
//! GRO is nm; molrs is Å. The reader multiplies by [`NM_TO_ANGSTROM`] and the
//! writer divides by it, so a frame in memory is never in nm and no consumer
//! has to ask where it came from. There is no `gro_units` tag: a unit that is
//! normalised at the boundary is not a property of the frame.

use std::io::{BufRead, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use ndarray::{Array1, Array2, IxDyn, array};

use molrs::spatial::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, I, Idx};

use crate::io::reader::{FrameReader, Reader};
use crate::io::writer::{FrameWriter, Writer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Nanometre → ångström. GRO stores lengths in nm; every other molrs format
/// and every consumer above the reader works in Å, so the conversion happens
/// here, at the boundary, exactly as the Cube reader converts Bohr → Å and
/// every force-field reader converts degrees → radians.
const NM_TO_ANGSTROM: F = 10.0;

/// The element a GROMACS atom name denotes, or `None` when nothing plausible
/// resolves.
///
/// A GRO file carries no element column, so this reads a *naming convention*
/// rather than parsing a field. GROMACS names are ambiguous in isolation —
/// `CA` is a protein C-alpha in one residue and calcium in another — and the
/// disambiguator is in the file:
///
/// * `monatomic_residue`: the atom is alone in its residue and its name is a
///   real element symbol, so it is that ion — `NA` → Na, `CL` → Cl, `CA` → Ca.
/// * otherwise the atom belongs to a polyatomic residue and follows the
///   biomolecular convention, where the leading letter is the element and the
///   rest says where it sits: `OW` → O, `HW1` → H, `CA` / `CB` → C.
///
/// The PDB reader's inference is deliberately **not** reused: it leans on the
/// column-13 rule (a leading blank means a one-letter element), and a GRO name
/// is already trimmed, so that signal does not survive the parse.
///
/// Known limit: a two-letter element written inside a polyatomic residue —
/// heme's `FE`, say — reads as its leading letter (F). Only a residue-name
/// table could fix that, and molrs has no business shipping one.
fn element_from_atom_name(name: &str, monatomic_residue: bool) -> Option<String> {
    let letters: String = name
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if letters.is_empty() {
        return None;
    }
    let title = |s: &str| -> String {
        let mut c = s.chars();
        let first = c.next().unwrap_or_default().to_ascii_uppercase();
        let rest: String = c.flat_map(|ch| ch.to_lowercase()).collect();
        format!("{first}{rest}")
    };

    if monatomic_residue {
        let whole = title(&letters);
        if molrs::Element::by_symbol(&whole).is_some() {
            return Some(whole);
        }
    }
    let one = title(&letters[..1]);
    if molrs::Element::by_symbol(&one).is_some() {
        return Some(one);
    }
    if letters.len() >= 2 {
        let two = title(&letters[..2]);
        if molrs::Element::by_symbol(&two).is_some() {
            return Some(two);
        }
    }
    None
}

fn invalid_data<E: std::fmt::Display>(e: E) -> Error {
    Error::new(ErrorKind::InvalidData, e.to_string())
}

fn substr(s: &str, start: usize, end: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    &s[start..end.min(len)]
}

fn insert_float_col(block: &mut Block, key: &str, vals: Vec<F>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    block.insert(key, arr).map_err(invalid_data)
}

/// Insert an unsigned column, rejecting negatives with a message that names
/// the key — used for the canonical identifier columns.
fn insert_uint_col(block: &mut Block, key: &str, vals: Vec<I>) -> Result<()> {
    let unsigned: Vec<molrs::types::Idx> = vals
        .iter()
        .map(|&v| {
            Idx::try_from(v).map_err(|_| {
                invalid_data(format!(
                    "GRO column '{key}' is unsigned in the Frame schema, got {v}"
                ))
            })
        })
        .collect::<Result<_>>()?;
    let n = unsigned.len();
    block
        .insert(
            key,
            Array1::from_vec(unsigned)
                .into_shape_with_order(IxDyn(&[n]))
                .map_err(invalid_data)?,
        )
        .map_err(invalid_data)
}

fn insert_str_col(block: &mut Block, key: &str, vals: Vec<String>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    block.insert(key, arr).map_err(invalid_data)
}

// ---------------------------------------------------------------------------
// Parsed atom record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GroAtom {
    resid: I,
    resname: String,
    atom_name: String,
    atom_id: I,
    x: F,
    y: F,
    z: F,
    velocity: Option<[F; 3]>,
}

fn parse_atom_line(line: &str, line_no: usize) -> Result<GroAtom> {
    // Strip trailing newline only — leading/trailing spaces inside fields are significant.
    let line = line.trim_end_matches(['\r', '\n']);
    if line.len() < 44 {
        return Err(invalid_data(format!(
            "line {}: GRO atom record too short ({} chars; need ≥44)",
            line_no,
            line.len()
        )));
    }
    let resid = substr(line, 0, 5)
        .trim()
        .parse::<I>()
        .map_err(|_| invalid_data(format!("line {}: bad residue number", line_no)))?;
    let resname = substr(line, 5, 10).trim().to_string();
    let atom_name = substr(line, 10, 15).trim().to_string();
    let atom_id = substr(line, 15, 20)
        .trim()
        .parse::<I>()
        .map_err(|_| invalid_data(format!("line {}: bad atom number", line_no)))?;
    let x = substr(line, 20, 28)
        .trim()
        .parse::<F>()
        .map_err(|_| invalid_data(format!("line {}: bad x", line_no)))?;
    let y = substr(line, 28, 36)
        .trim()
        .parse::<F>()
        .map_err(|_| invalid_data(format!("line {}: bad y", line_no)))?;
    let z = substr(line, 36, 44)
        .trim()
        .parse::<F>()
        .map_err(|_| invalid_data(format!("line {}: bad z", line_no)))?;

    let velocity = if line.len() >= 68 {
        let vx = substr(line, 44, 52)
            .trim()
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad vx", line_no)))?;
        let vy = substr(line, 52, 60)
            .trim()
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad vy", line_no)))?;
        let vz = substr(line, 60, 68)
            .trim()
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad vz", line_no)))?;
        Some([vx, vy, vz])
    } else {
        None
    };

    Ok(GroAtom {
        resid,
        resname,
        atom_name,
        atom_id,
        x,
        y,
        z,
        velocity,
    })
}

/// Parse the GROMACS box-vector line (3 floats orthorhombic, 9 triclinic).
/// Returns the 3x3 H matrix with H columns = lattice vectors.
fn parse_box_line(line: &str, line_no: usize) -> Result<[[F; 3]; 3]> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() != 3 && tokens.len() != 9 {
        return Err(invalid_data(format!(
            "line {}: box vector line must have 3 or 9 floats, got {}",
            line_no,
            tokens.len()
        )));
    }
    let parse = |idx: usize| -> Result<F> {
        tokens[idx]
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad box float '{}'", line_no, tokens[idx])))
    };
    if tokens.len() == 3 {
        let v1x = parse(0)?;
        let v2y = parse(1)?;
        let v3z = parse(2)?;
        return Ok([[v1x, 0.0, 0.0], [0.0, v2y, 0.0], [0.0, 0.0, v3z]]);
    }
    // GROMACS triclinic box order:
    // v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
    let v1x = parse(0)?;
    let v2y = parse(1)?;
    let v3z = parse(2)?;
    let v1y = parse(3)?;
    let v1z = parse(4)?;
    let v2x = parse(5)?;
    let v2z = parse(6)?;
    let v3x = parse(7)?;
    let v3y = parse(8)?;
    Ok([[v1x, v1y, v1z], [v2x, v2y, v2z], [v3x, v3y, v3z]])
}

/// Format a 3x3 H matrix back into a GRO box line. Emits 3 numbers if the box
/// is orthorhombic (off-diagonals all zero), else the 9-number triclinic form.
fn format_box_line(h: &Array2<F>) -> String {
    let v1x = h[(0, 0)];
    let v2y = h[(1, 1)];
    let v3z = h[(2, 2)];
    let v1y = h[(1, 0)];
    let v1z = h[(2, 0)];
    let v2x = h[(0, 1)];
    let v2z = h[(2, 1)];
    let v3x = h[(0, 2)];
    let v3y = h[(1, 2)];
    let off_diag = [v1y, v1z, v2x, v2z, v3x, v3y];
    let is_ortho = off_diag.iter().all(|v| v.abs() < 1e-10);
    if is_ortho {
        format!("{:10.5}{:10.5}{:10.5}", v1x, v2y, v3z)
    } else {
        format!(
            "{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}",
            v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y
        )
    }
}

// ---------------------------------------------------------------------------
// Public reader API
// ---------------------------------------------------------------------------

/// Read all frames from a `.gro` file at `path`.
pub fn read_gro<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = std::io::BufReader::new(file);
    let mut gr = GroReader::new(reader);
    crate::io::reader::collect_frames(&mut gr)
}

/// Read a single GRO frame from any [`BufRead`]. Returns `Ok(None)` at EOF.
pub fn read_gro_frame<R: BufRead>(reader: &mut R) -> Result<Option<Frame>> {
    let mut buf = String::new();

    // Title
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Ok(None);
    }
    let title = buf.trim().to_string();

    // Atom count
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Err(invalid_data("missing atom-count line"));
    }
    let n_atoms: usize = buf
        .trim()
        .parse()
        .map_err(|_| invalid_data(format!("bad atom-count line: {:?}", buf.trim())))?;

    // Atoms
    let mut atoms = Vec::with_capacity(n_atoms);
    let mut have_velocities: Option<bool> = None;
    for i in 0..n_atoms {
        buf.clear();
        if reader.read_line(&mut buf)? == 0 {
            return Err(invalid_data(format!(
                "unexpected EOF after {} of {} atom records",
                i, n_atoms
            )));
        }
        let atom = parse_atom_line(&buf, i + 3)?;
        // Velocity presence must be consistent across all rows.
        let has_v = atom.velocity.is_some();
        match have_velocities {
            None => have_velocities = Some(has_v),
            Some(expected) if expected != has_v => {
                return Err(invalid_data(format!(
                    "atom {} has {} velocities; expected {} based on first row",
                    i + 1,
                    if has_v { "" } else { "no" },
                    if expected { "yes" } else { "no" }
                )));
            }
            _ => {}
        }
        atoms.push(atom);
    }

    // Box vectors
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Err(invalid_data("missing box-vector line"));
    }
    let cell_rows = parse_box_line(&buf, 3 + n_atoms)?;

    // -----------------------------------------------------------------------
    // Build the Frame
    // -----------------------------------------------------------------------
    // How many atoms share each residue number — the signal that tells a
    // monatomic ion from a C-alpha (see `element_from_atom_name`).
    let mut residue_size: std::collections::HashMap<I, usize> = std::collections::HashMap::new();
    for a in &atoms {
        *residue_size.entry(a.resid).or_insert(0) += 1;
    }

    let mut block = Block::new();
    let mut resid = Vec::with_capacity(n_atoms);
    let mut resname = Vec::with_capacity(n_atoms);
    let mut atom_name = Vec::with_capacity(n_atoms);
    let mut element = Vec::with_capacity(n_atoms);
    let mut atom_id = Vec::with_capacity(n_atoms);
    let mut x = Vec::with_capacity(n_atoms);
    let mut y = Vec::with_capacity(n_atoms);
    let mut z = Vec::with_capacity(n_atoms);
    let mut vx = Vec::with_capacity(n_atoms);
    let mut vy = Vec::with_capacity(n_atoms);
    let mut vz = Vec::with_capacity(n_atoms);
    for a in &atoms {
        resid.push(a.resid);
        resname.push(a.resname.clone());
        atom_name.push(a.atom_name.clone());
        let alone = residue_size.get(&a.resid).copied().unwrap_or(0) == 1;
        element
            .push(element_from_atom_name(&a.atom_name, alone).unwrap_or_else(|| "X".to_string()));
        atom_id.push(a.atom_id);
        x.push(a.x * NM_TO_ANGSTROM);
        y.push(a.y * NM_TO_ANGSTROM);
        z.push(a.z * NM_TO_ANGSTROM);
        if let Some(v) = a.velocity {
            // nm/ps → Å/ps: the time unit is untouched, so the length scale is
            // the whole conversion.
            vx.push(v[0] * NM_TO_ANGSTROM);
            vy.push(v[1] * NM_TO_ANGSTROM);
            vz.push(v[2] * NM_TO_ANGSTROM);
        }
    }
    insert_uint_col(&mut block, "res_id", resid)?;
    insert_str_col(&mut block, "resname", resname)?;
    insert_str_col(&mut block, "atom_name", atom_name)?;
    insert_str_col(&mut block, "element", element)?;
    insert_uint_col(&mut block, "id", atom_id)?;
    insert_float_col(&mut block, "x", x)?;
    insert_float_col(&mut block, "y", y)?;
    insert_float_col(&mut block, "z", z)?;
    if have_velocities == Some(true) {
        insert_float_col(&mut block, "vx", vx)?;
        insert_float_col(&mut block, "vy", vy)?;
        insert_float_col(&mut block, "vz", vz)?;
    }

    let mut frame = Frame::new();
    if !title.is_empty() {
        frame.meta.insert("title", title);
    }
    frame.insert("atoms", block);

    // SimBox: H columns = lattice vectors. cell_rows[i] = lattice vector i.
    let h = Array2::from_shape_fn((3, 3), |(i, j)| cell_rows[j][i] * NM_TO_ANGSTROM);
    let origin = array![0.0 as F, 0.0, 0.0];
    let simbox = SimBox::new(h, origin, [true; 3]).map_err(|e| invalid_data(format!("{:?}", e)))?;
    frame.simbox = Some(simbox);

    Ok(Some(frame))
}

/// `FrameReader`-trait wrapper. Multi-frame `.gro` files are supported by
/// repeated calls to `read_frame`; each call advances to the next frame.
pub struct GroReader<R: BufRead> {
    reader: R,
}

impl<R: BufRead> Reader for GroReader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R: BufRead> FrameReader for GroReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        crate::io::reader::validated(read_gro_frame(&mut self.reader)?)
    }
}

// ---------------------------------------------------------------------------
// Public writer API
// ---------------------------------------------------------------------------

/// Write a frame as `.gro` at `path`.
pub fn write_gro<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    write_gro_frame(&mut w, frame)?;
    w.flush()
}

/// Write a single frame in GRO format.
pub fn write_gro_frame<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("GRO write: frame has no atoms block"))?;
    let n = atoms
        .nrows()
        .ok_or_else(|| invalid_data("GRO write: atoms block has no rows"))?;

    let title = frame
        .meta
        .get("title")
        .and_then(|value| value.as_str())
        .unwrap_or("molrs GRO");
    writeln!(writer, "{}", title)?;
    writeln!(writer, "{:>5}", n)?;

    let xs = atoms
        .get_float("x")
        .ok_or_else(|| invalid_data("atoms.x missing"))?;
    let ys = atoms
        .get_float("y")
        .ok_or_else(|| invalid_data("atoms.y missing"))?;
    let zs = atoms
        .get_float("z")
        .ok_or_else(|| invalid_data("atoms.z missing"))?;
    let vx = atoms.get_float("vx");
    let vy = atoms.get_float("vy");
    let vz = atoms.get_float("vz");
    let resid = atoms.get_uint("res_id");
    let resname = atoms.get_string("resname");
    let atom_name = atoms.get_string("atom_name");
    let element = atoms.get_string("element");
    let atom_id = atoms.get_uint("id");

    for i in 0..n {
        let r = resid.map(|c| c[[i]]).unwrap_or(1);
        let rn = resname.map(|c| c[[i]].as_str()).unwrap_or("UNK");
        // A frame from a format with no atom names (XYZ, say) still knows its
        // elements; writing "X" there would throw away the identity the reader
        // is expected to infer back out.
        let an = atom_name
            .map(|c| c[[i]].as_str())
            .or_else(|| element.map(|c| c[[i]].as_str()))
            .unwrap_or("X");
        let aid = atom_id
            .map(|c| c[[i]])
            .unwrap_or((i as molrs::types::Idx) + 1);
        // GROMACS truncates the residue number and atom number at 5 digits via modulo.
        let r_mod = r.rem_euclid(100_000);
        let aid_mod = aid.rem_euclid(100_000);
        // Å → nm on the way out, mirroring the reader's normalisation.
        write!(
            writer,
            "{:>5}{:<5}{:>5}{:>5}{:>8.3}{:>8.3}{:>8.3}",
            r_mod,
            truncate_to_5(rn),
            truncate_to_5(an),
            aid_mod,
            xs[[i]] / NM_TO_ANGSTROM,
            ys[[i]] / NM_TO_ANGSTROM,
            zs[[i]] / NM_TO_ANGSTROM
        )?;
        if let (Some(vxc), Some(vyc), Some(vzc)) = (vx, vy, vz) {
            write!(
                writer,
                "{:>8.4}{:>8.4}{:>8.4}",
                vxc[[i]] / NM_TO_ANGSTROM,
                vyc[[i]] / NM_TO_ANGSTROM,
                vzc[[i]] / NM_TO_ANGSTROM
            )?;
        }
        writeln!(writer)?;
    }

    let h = frame
        .simbox
        .as_ref()
        .map(|sb| sb.h_view().to_owned() / NM_TO_ANGSTROM)
        .unwrap_or_else(|| Array2::<F>::zeros((3, 3)));
    writeln!(writer, "{}", format_box_line(&h))?;

    Ok(())
}

fn truncate_to_5(s: &str) -> &str {
    let mut end = 0;
    for (i, _) in s.char_indices().take(5) {
        end = i + s[i..].chars().next().map(char::len_utf8).unwrap_or(0);
    }
    &s[..end.min(s.len())]
}

/// `FrameWriter`-trait wrapper.
pub struct GroFrameWriter<W: Write> {
    writer: W,
}

impl<W: Write> Writer for GroFrameWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for GroFrameWriter<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_before_write(frame)?;
        write_gro_frame(&mut self.writer, frame)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn water_gro() -> String {
        // Each atom line must be at least 44 chars, fixed-column.
        // resid(5) + resname(5) + atom_name(5) + atom_id(5) + x(8) + y(8) + z(8) = 44
        let lines = [
            "Water box",
            "    3",
            "    1WAT     OW    1   0.000   0.000   0.000",
            "    1WAT    HW1    2   0.100   0.000   0.000",
            "    1WAT    HW2    3   0.000   0.100   0.000",
            "   2.00000   2.00000   2.00000",
        ];
        let mut out = String::new();
        for l in lines {
            out.push_str(l);
            out.push('\n');
        }
        out
    }

    #[test]
    fn reads_basic_gro() {
        let frame = read_gro_frame(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(3));
        let xs = atoms.get_float("x").unwrap();
        assert!((xs[[1]] - 1.0).abs() < 1e-9); // 0.100 nm → 1.0 Å
        let names = atoms.get_string("atom_name").unwrap();
        assert_eq!(names[[0]], "OW");
        assert_eq!(names[[1]], "HW1");
        assert!(frame.simbox.is_some());
    }

    #[test]
    fn reads_lengths_as_angstrom() {
        // GRO is nm; every other molrs format is Å. The reader normalises at
        // its boundary, so nothing downstream has to know where a frame came
        // from — 0.100 nm is 1.0 Å, and the 2 nm box is 20 Å.
        let frame = read_gro_frame(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let xs = frame.get("atoms").unwrap().get_float("x").unwrap();
        assert!((xs[[1]] - 1.0).abs() < 1e-9, "x[1] = {}", xs[[1]]);
        let h = frame.simbox.as_ref().unwrap().h_view().to_owned();
        assert!((h[[0, 0]] - 20.0).abs() < 1e-9, "box = {}", h[[0, 0]]);
        assert!(
            !frame.meta.contains_key("gro_units"),
            "the frame is in Å; a nm tag would be a lie"
        );
    }

    #[test]
    fn reads_velocities_as_angstrom_per_ps() {
        // A velocity is a length per time: leaving it in nm/ps beside an Å
        // position is what makes `x += v * dt` silently wrong.
        let with_v = "Water box\n    1\n    1WAT     OW    1   0.000   0.000   0.000  0.1000  0.2000  0.3000\n   2.00000   2.00000   2.00000\n";
        let frame = read_gro_frame(&mut Cursor::new(with_v.as_bytes().to_vec()))
            .unwrap()
            .unwrap();
        let vx = frame.get("atoms").unwrap().get_float("vx").unwrap();
        assert!((vx[[0]] - 1.0).abs() < 1e-9, "vx = {}", vx[[0]]);
    }

    #[test]
    fn infers_element_from_the_atom_name() {
        // A GRO file has no element column. The names are there, though, and a
        // reader that drops them leaves every downstream consumer with "X".
        let frame = read_gro_frame(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let elements = frame.get("atoms").unwrap().get_string("element").unwrap();
        assert_eq!(elements[[0]], "O", "OW is water oxygen");
        assert_eq!(elements[[1]], "H", "HW1 is a water hydrogen");
        assert_eq!(elements[[2]], "H");
    }

    #[test]
    fn a_monatomic_residue_reads_its_name_as_the_element() {
        // `CA` is a protein C-alpha in one residue and calcium in another. The
        // residue decides, and it is in the file: a one-atom residue whose name
        // is an element symbol is that ion.
        let ions = concat!(
            "ions\n",
            "    6\n",
            "    1ALA      N    1   0.000   0.000   0.000\n",
            "    1ALA     CA    2   0.100   0.000   0.000\n",
            "    1ALA     CB    3   0.200   0.000   0.000\n",
            "    2CA      CA    4   1.000   0.000   0.000\n",
            "    3CL      CL    5   0.000   1.000   0.000\n",
            "    4SOL     OW    6   0.000   0.000   1.000\n",
            "   2.00000   2.00000   2.00000\n",
        );
        let frame = read_gro_frame(&mut Cursor::new(ions.as_bytes().to_vec()))
            .unwrap()
            .unwrap();
        let elements = frame.get("atoms").unwrap().get_string("element").unwrap();
        assert_eq!(elements[[0]], "N");
        assert_eq!(elements[[1]], "C", "CA inside a residue is the C-alpha");
        assert_eq!(elements[[2]], "C");
        assert_eq!(elements[[3]], "Ca", "CA alone in its residue is calcium");
        assert_eq!(elements[[4]], "Cl");
        assert_eq!(elements[[5]], "O");
    }

    #[test]
    fn writes_the_element_as_the_atom_name_when_there_is_none() {
        // The mirror of the read-side inference: a frame that came from a
        // format without atom names still knows its elements, and "X" would
        // throw away the identity the next reader is expected to recover.
        use molrs::store::block::Block;
        use ndarray::Array1;

        let mut atoms = Block::new();
        for (key, v) in [("x", 1.0_f64), ("y", 0.0), ("z", 0.0)] {
            atoms
                .insert(key, Array1::from_vec(vec![v]).into_dyn())
                .unwrap();
        }
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["Na".to_string()]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let mut buf = Vec::new();
        write_gro_frame(&mut buf, &frame).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let atom_line = text.lines().nth(2).expect("atom line");
        assert_eq!(
            substr(atom_line, 10, 15).trim(),
            "Na",
            "line: {atom_line:?}"
        );
    }

    #[test]
    fn round_trip_basic_gro() {
        let frame = read_gro_frame(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let mut buf = Vec::new();
        write_gro_frame(&mut buf, &frame).unwrap();
        let frame2 = read_gro_frame(&mut Cursor::new(&buf)).unwrap().unwrap();
        let xs1 = frame.get("atoms").unwrap().get_float("x").unwrap();
        let xs2 = frame2.get("atoms").unwrap().get_float("x").unwrap();
        assert_eq!(xs1.len(), xs2.len());
        for i in 0..xs1.len() {
            assert!((xs1[[i]] - xs2[[i]]).abs() < 1e-3);
        }
    }
}
