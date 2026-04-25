//! VASP POSCAR / CONTCAR structure file reader and writer.
//!
//! POSCAR is the VASP input format describing a crystalline cell and the atoms
//! within it. Format outline:
//!
//! ```text
//! line 1   : comment / system name              → frame.meta["title"]
//! line 2   : global scale factor (Å)
//! line 3-5 : lattice vectors (row-per-line, Å after scaling)
//! line 6   : element symbols  (VASP5 only — omitted in VASP4)
//! line 7   : atom counts per element
//! line 8?  : "Selective dynamics" (optional)
//! line N   : "Direct" or "Cartesian"
//! line N+1+: atom coordinates (3 floats; + T/F flags if selective dynamics)
//! [opt]    : blank line then "Cartesian"/"Direct" + N velocity rows
//! ```
//!
//! The returned [`Frame`] contains:
//!
//! - `"atoms"` block:
//!   - `x`, `y`, `z` — Cartesian Å (`Direct` files are converted on read).
//!   - `symbol` — element symbol (omitted when the file did not declare them).
//!   - `sd_x`, `sd_y`, `sd_z` — selective-dynamics flags, if present.
//!   - `vx`, `vy`, `vz` — atomic velocities, if present.
//! - `frame.simbox` — periodic [`SimBox`] from the lattice vectors.
//! - `frame.meta` — `title`, plus `poscar_mode = "direct" | "cartesian"`.

use std::io::{BufRead, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use ndarray::{Array1, Array2, IxDyn, array};

use molrs::block::Block;
use molrs::frame::Frame;
use molrs::region::simbox::SimBox;
use molrs::types::F;

use crate::reader::{FrameReader, Reader};
use crate::vasp_common::{
    AtomRow, CoordMode, expand_symbols, fractional_to_cartesian, parse_atom_row, read_coords,
    read_header,
};
use crate::writer::{FrameWriter, Writer};

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn invalid_data<E: std::fmt::Display>(e: E) -> Error {
    Error::new(ErrorKind::InvalidData, e.to_string())
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read one POSCAR file from `path`.
pub fn read_poscar<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    read_poscar_from_reader(std::io::BufReader::new(file))
}

/// Read one POSCAR-format frame from `reader`.
pub fn read_poscar_from_reader<R: BufRead>(mut reader: R) -> Result<Frame> {
    let mut line_no = 0usize;
    let header = read_header(&mut reader, &mut line_no)?;
    let n = header.total_atoms();
    if n == 0 {
        return Err(invalid_data("POSCAR declares zero atoms"));
    }

    let (raw_x, raw_y, raw_z, sd_flags) =
        read_coords(&mut reader, n, &mut line_no, header.selective_dynamics)?;

    let (cart_x, cart_y, cart_z) = match header.mode {
        CoordMode::Direct => fractional_to_cartesian(&raw_x, &raw_y, &raw_z, &header.cell),
        CoordMode::Cartesian => {
            // Already Cartesian; lattice vectors were pre-scaled by `scale`,
            // but Cartesian atom coords are also scaled by VASP convention.
            let s = header.scale;
            (
                raw_x.iter().map(|v| v * s).collect(),
                raw_y.iter().map(|v| v * s).collect(),
                raw_z.iter().map(|v| v * s).collect(),
            )
        }
    };

    // Optional velocity block after a blank line. Best-effort: silently stop
    // if the file does not contain one or stops early.
    let velocities = try_read_velocities(&mut reader, n, &mut line_no);

    // ---------------------------------------------------------------------
    // Assemble Frame
    // ---------------------------------------------------------------------
    let mut atoms = Block::new();
    insert_float_col(&mut atoms, "x", cart_x)?;
    insert_float_col(&mut atoms, "y", cart_y)?;
    insert_float_col(&mut atoms, "z", cart_z)?;

    let symbols = expand_symbols(&header.symbols, &header.counts);
    if !symbols.is_empty() {
        let arr = Array1::from_vec(symbols)
            .into_shape_with_order(IxDyn(&[n]))
            .map_err(invalid_data)?
            .into_dyn();
        atoms.insert("symbol", arr).map_err(invalid_data)?;
    }

    if let Some(flags) = sd_flags {
        let (sx, sy, sz): (Vec<bool>, Vec<bool>, Vec<bool>) = flags.iter().fold(
            (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            ),
            |(mut x, mut y, mut z), f| {
                x.push(f[0]);
                y.push(f[1]);
                z.push(f[2]);
                (x, y, z)
            },
        );
        insert_bool_col(&mut atoms, "sd_x", sx)?;
        insert_bool_col(&mut atoms, "sd_y", sy)?;
        insert_bool_col(&mut atoms, "sd_z", sz)?;
    }

    if let Some((vx, vy, vz)) = velocities {
        insert_float_col(&mut atoms, "vx", vx)?;
        insert_float_col(&mut atoms, "vy", vy)?;
        insert_float_col(&mut atoms, "vz", vz)?;
    }

    let mut frame = Frame::new();
    if !header.title.is_empty() {
        frame.meta.insert("title".into(), header.title);
    }
    frame.meta.insert(
        "poscar_mode".into(),
        match header.mode {
            CoordMode::Direct => "direct".into(),
            CoordMode::Cartesian => "cartesian".into(),
        },
    );

    // SimBox: H columns = lattice vectors. cell rows = lattice vectors, so
    // h[col, row] = cell[row][col].
    let h = Array2::from_shape_fn((3, 3), |(i, j)| header.cell[j][i]);
    let origin = array![0.0 as F, 0.0, 0.0];
    let simbox = SimBox::new(h, origin, [true; 3]).map_err(|e| invalid_data(format!("{:?}", e)))?;
    frame.simbox = Some(simbox);

    frame.insert("atoms", atoms);
    Ok(frame)
}

fn insert_float_col(block: &mut Block, key: &str, vals: Vec<F>) -> Result<()> {
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

/// Best-effort velocity read. Returns `None` if no velocity block present.
fn try_read_velocities<R: BufRead>(
    reader: &mut R,
    n: usize,
    line_no: &mut usize,
) -> Option<(Vec<F>, Vec<F>, Vec<F>)> {
    // Consume any blank line(s).
    let mut buf = String::new();
    loop {
        buf.clear();
        let bytes = reader.read_line(&mut buf).ok()?;
        if bytes == 0 {
            return None;
        }
        *line_no += 1;
        if !buf.trim().is_empty() {
            break;
        }
    }
    // `buf` now holds the first non-blank line. It is either a velocity row
    // (3 floats) or a coord-mode keyword for the velocity block.
    let trimmed = buf.trim();
    let first_char = trimmed.chars().next()?;
    let first_row_is_keyword = matches!(first_char, 'D' | 'd' | 'C' | 'c' | 'K' | 'k');

    let mut vx = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    let mut vz = Vec::with_capacity(n);

    if !first_row_is_keyword {
        let row = parse_atom_row(&buf, *line_no, false).ok()?;
        let AtomRow { x, y, z, .. } = row;
        vx.push(x);
        vy.push(y);
        vz.push(z);
    }

    while vx.len() < n {
        buf.clear();
        let bytes = reader.read_line(&mut buf).ok()?;
        if bytes == 0 {
            return None;
        }
        *line_no += 1;
        if buf.trim().is_empty() {
            continue;
        }
        let row = parse_atom_row(&buf, *line_no, false).ok()?;
        vx.push(row.x);
        vy.push(row.y);
        vz.push(row.z);
    }
    Some((vx, vy, vz))
}

/// `FrameReader`-trait wrapper for `read_poscar_from_reader`.
///
/// POSCAR holds at most one frame. The first `read_frame` call returns it;
/// subsequent calls return `Ok(None)`.
pub struct PoscarReader<R: BufRead> {
    reader: R,
    consumed: bool,
}

impl<R: BufRead> Reader for PoscarReader<R> {
    type R = R;
    type Frame = Frame;
    fn new(reader: R) -> Self {
        Self {
            reader,
            consumed: false,
        }
    }
}

impl<R: BufRead> FrameReader for PoscarReader<R> {
    fn read_frame(&mut self) -> Result<Option<Frame>> {
        if self.consumed {
            return Ok(None);
        }
        self.consumed = true;
        Ok(Some(read_poscar_from_reader(&mut self.reader)?))
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Frame as a POSCAR file at `path`.
pub fn write_poscar<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    write_poscar_to_writer(&mut w, frame)?;
    w.flush()
}

/// Write a Frame in POSCAR format to any writer.
///
/// The output mode (Direct / Cartesian) follows `frame.meta["poscar_mode"]`,
/// defaulting to Cartesian when absent.
pub fn write_poscar_to_writer<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("POSCAR write: frame has no atoms block"))?;
    let n = atoms.nrows().unwrap_or(0);
    if n == 0 {
        return Err(invalid_data("POSCAR write: atoms block is empty"));
    }
    let simbox = frame
        .simbox
        .as_ref()
        .ok_or_else(|| invalid_data("POSCAR write: frame has no SimBox"))?;

    let title = frame
        .meta
        .get("title")
        .cloned()
        .unwrap_or_else(|| "molrs POSCAR".to_string());
    writeln!(writer, "{}", title)?;
    writeln!(writer, "1.0")?;

    // Lattice rows: h columns are lattice vectors → row i = (h[0,i], h[1,i], h[2,i]).
    let h = simbox_h_matrix(simbox);
    for i in 0..3 {
        writeln!(
            writer,
            "  {:.10}  {:.10}  {:.10}",
            h[(0, i)],
            h[(1, i)],
            h[(2, i)]
        )?;
    }

    // Group atoms by symbol so we can emit POSCAR's per-element runs.
    let symbol_col = atoms.get_string("symbol");
    let (symbols_line, counts_line, order) = group_by_symbol(n, symbol_col);

    if let Some(syms) = &symbols_line {
        writeln!(writer, "{}", syms.join(" "))?;
    }
    writeln!(
        writer,
        "{}",
        counts_line
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    )?;

    // Mode: Cartesian (default) or Direct based on frame.meta.
    let direct = frame.meta.get("poscar_mode").map(|s| s.as_str()) == Some("direct");
    writeln!(writer, "{}", if direct { "Direct" } else { "Cartesian" })?;

    let xs = atoms
        .get_float("x")
        .ok_or_else(|| invalid_data("atoms.x missing"))?;
    let ys = atoms
        .get_float("y")
        .ok_or_else(|| invalid_data("atoms.y missing"))?;
    let zs = atoms
        .get_float("z")
        .ok_or_else(|| invalid_data("atoms.z missing"))?;

    if direct {
        // Convert Cartesian → fractional via H^{-1}.
        let inv = invert_h(&h)?;
        for &i in &order {
            let x = xs[[i]];
            let y = ys[[i]];
            let z = zs[[i]];
            let fx = inv[(0, 0)] * x + inv[(0, 1)] * y + inv[(0, 2)] * z;
            let fy = inv[(1, 0)] * x + inv[(1, 1)] * y + inv[(1, 2)] * z;
            let fz = inv[(2, 0)] * x + inv[(2, 1)] * y + inv[(2, 2)] * z;
            writeln!(writer, "  {:.10}  {:.10}  {:.10}", fx, fy, fz)?;
        }
    } else {
        for &i in &order {
            writeln!(
                writer,
                "  {:.10}  {:.10}  {:.10}",
                xs[[i]],
                ys[[i]],
                zs[[i]]
            )?;
        }
    }
    Ok(())
}

/// Pull the H matrix (3x3, columns = lattice vectors) out of a SimBox.
fn simbox_h_matrix(simbox: &SimBox) -> ndarray::Array2<F> {
    simbox.h_view().to_owned()
}

/// Group atoms by their `symbol` column (if present).
///
/// Returns `(Some(symbols), counts, order)` when symbols are present, where
/// `order` is the permutation of original indices that puts each element in a
/// run. Returns `(None, [n], 0..n)` when no symbol column exists.
fn group_by_symbol(
    n: usize,
    symbol_col: Option<&ndarray::ArrayD<String>>,
) -> (Option<Vec<String>>, Vec<usize>, Vec<usize>) {
    if let Some(col) = symbol_col {
        let mut runs: Vec<(String, Vec<usize>)> = Vec::new();
        for i in 0..n {
            let s = col[[i]].clone();
            if let Some((sym, idxs)) = runs.last_mut() {
                if *sym == s {
                    idxs.push(i);
                    continue;
                }
            }
            runs.push((s, vec![i]));
        }
        let symbols = runs.iter().map(|(s, _)| s.clone()).collect();
        let counts = runs.iter().map(|(_, v)| v.len()).collect();
        let order: Vec<usize> = runs.into_iter().flat_map(|(_, v)| v).collect();
        (Some(symbols), counts, order)
    } else {
        (None, vec![n], (0..n).collect())
    }
}

/// Compute the inverse of a 3x3 matrix via cofactor expansion.
fn invert_h(h: &ndarray::Array2<F>) -> Result<ndarray::Array2<F>> {
    let m = |i: usize, j: usize| h[(i, j)];
    let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
        - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    if det.abs() < 1e-30 {
        return Err(invalid_data("singular cell matrix"));
    }
    let inv_det = 1.0 / det;
    let mut inv = ndarray::Array2::<F>::zeros((3, 3));
    inv[(0, 0)] = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) * inv_det;
    inv[(0, 1)] = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * inv_det;
    inv[(0, 2)] = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * inv_det;
    inv[(1, 0)] = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * inv_det;
    inv[(1, 1)] = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * inv_det;
    inv[(1, 2)] = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * inv_det;
    inv[(2, 0)] = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) * inv_det;
    inv[(2, 1)] = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * inv_det;
    inv[(2, 2)] = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * inv_det;
    Ok(inv)
}

/// Convenience writer: implements [`FrameWriter`].
pub struct PoscarFrameWriter<W: Write> {
    writer: W,
}

impl<W: Write> Writer for PoscarFrameWriter<W> {
    type W = W;
    type FrameLike = Frame;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for PoscarFrameWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        write_poscar_to_writer(&mut self.writer, frame)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const POSCAR_BN: &str = "BN bulk\n\
1.0\n\
2.5  0.0  0.0\n\
0.0  2.5  0.0\n\
0.0  0.0  2.5\n\
B N\n\
1 1\n\
Direct\n\
0.0 0.0 0.0\n\
0.5 0.5 0.5\n";

    #[test]
    fn reads_basic_poscar() {
        let frame = read_poscar_from_reader(Cursor::new(POSCAR_BN.as_bytes())).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(2));
        assert!(frame.simbox.is_some());

        let xs = atoms.get_float("x").unwrap();
        // (0.5, 0.5, 0.5) fractional with 2.5Å cube → (1.25, 1.25, 1.25)
        assert!((xs[[1]] - 1.25).abs() < 1e-10);
    }

    #[test]
    fn round_trip_basic_poscar() {
        let frame = read_poscar_from_reader(Cursor::new(POSCAR_BN.as_bytes())).unwrap();
        let mut buf = Vec::new();
        write_poscar_to_writer(&mut buf, &frame).unwrap();
        let frame2 = read_poscar_from_reader(Cursor::new(&buf)).unwrap();

        let xs1 = frame.get("atoms").unwrap().get_float("x").unwrap();
        let xs2 = frame2.get("atoms").unwrap().get_float("x").unwrap();
        for i in 0..xs1.len() {
            assert!(
                (xs1[[i]] - xs2[[i]]).abs() < 1e-6,
                "x[{}]: {} vs {}",
                i,
                xs1[[i]],
                xs2[[i]]
            );
        }
    }
}
