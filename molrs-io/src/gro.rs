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
//! | 21-28| x (Å nm, %8.3f)        | `   0.310`    |
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
//! - `"atoms"` block: `resid` (i32), `resname` (str), `atom_name` (str),
//!   `atom_id` (i32), `x`/`y`/`z` (F, nm), and optional `vx`/`vy`/`vz` (F).
//! - `frame.simbox`: triclinic [`SimBox`] from the box-vector line.
//! - `frame.meta["title"]` and `frame.meta["gro_units"] = "nm"`.

use std::io::{BufRead, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use ndarray::{Array1, Array2, IxDyn, array};

use molrs::block::Block;
use molrs::frame::Frame;
use molrs::region::simbox::SimBox;
use molrs::types::{F, I};

use crate::reader::{FrameReader, Reader};
use crate::writer::{FrameWriter, Writer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn insert_int_col(block: &mut Block, key: &str, vals: Vec<I>) -> Result<()> {
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
    gr.read_all()
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
    let mut block = Block::new();
    let mut resid = Vec::with_capacity(n_atoms);
    let mut resname = Vec::with_capacity(n_atoms);
    let mut atom_name = Vec::with_capacity(n_atoms);
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
        atom_id.push(a.atom_id);
        x.push(a.x);
        y.push(a.y);
        z.push(a.z);
        if let Some(v) = a.velocity {
            vx.push(v[0]);
            vy.push(v[1]);
            vz.push(v[2]);
        }
    }
    insert_int_col(&mut block, "resid", resid)?;
    insert_str_col(&mut block, "resname", resname)?;
    insert_str_col(&mut block, "atom_name", atom_name)?;
    insert_int_col(&mut block, "atom_id", atom_id)?;
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
        frame.meta.insert("title".into(), title);
    }
    frame.meta.insert("gro_units".into(), "nm".into());
    frame.insert("atoms", block);

    // SimBox: H columns = lattice vectors. cell_rows[i] = lattice vector i.
    let h = Array2::from_shape_fn((3, 3), |(i, j)| cell_rows[j][i]);
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
    type Frame = Frame;
    fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R: BufRead> FrameReader for GroReader<R> {
    fn read_frame(&mut self) -> Result<Option<Frame>> {
        read_gro_frame(&mut self.reader)
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
        .cloned()
        .unwrap_or_else(|| "molrs GRO".to_string());
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
    let resid = atoms.get_int("resid");
    let resname = atoms.get_string("resname");
    let atom_name = atoms.get_string("atom_name");
    let atom_id = atoms.get_int("atom_id");

    for i in 0..n {
        let r = resid.map(|c| c[[i]]).unwrap_or(1);
        let rn = resname.map(|c| c[[i]].as_str()).unwrap_or("UNK");
        let an = atom_name.map(|c| c[[i]].as_str()).unwrap_or("X");
        let aid = atom_id.map(|c| c[[i]]).unwrap_or((i as I) + 1);
        // GROMACS truncates the residue number and atom number at 5 digits via modulo.
        let r_mod = r.rem_euclid(100_000);
        let aid_mod = aid.rem_euclid(100_000);
        write!(
            writer,
            "{:>5}{:<5}{:>5}{:>5}{:>8.3}{:>8.3}{:>8.3}",
            r_mod,
            truncate_to_5(rn),
            truncate_to_5(an),
            aid_mod,
            xs[[i]],
            ys[[i]],
            zs[[i]]
        )?;
        if let (Some(vxc), Some(vyc), Some(vzc)) = (vx, vy, vz) {
            write!(
                writer,
                "{:>8.4}{:>8.4}{:>8.4}",
                vxc[[i]],
                vyc[[i]],
                vzc[[i]]
            )?;
        }
        writeln!(writer)?;
    }

    let h = frame
        .simbox
        .as_ref()
        .map(|sb| sb.h_view().to_owned())
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
    type FrameLike = Frame;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for GroFrameWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> Result<()> {
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
        assert!((xs[[1]] - 0.1).abs() < 1e-9);
        let names = atoms.get_string("atom_name").unwrap();
        assert_eq!(names[[0]], "OW");
        assert_eq!(names[[1]], "HW1");
        assert!(frame.simbox.is_some());
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
