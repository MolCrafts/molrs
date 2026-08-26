//! GROMACS TRR binary trajectory reader and writer.
//!
//! TRR is the full-precision GROMACS trajectory format: an XDR (big-endian)
//! stream of self-describing per-frame records carrying any subset of box,
//! coordinates, velocities, and forces, in single **or** double precision.
//!
//! # Per-frame layout (XDR, big-endian)
//!
//! ```text
//! magic        i32 = 1993
//! ver_len      i32              (strlen+1 allocation hint; ignored)
//! version      XDR string       ("GMX_trn_file")
//! ir_size      i32              (legacy; must be 0)
//! e_size       i32              (legacy; must be 0)
//! box_size     i32 (bytes)      (>0 ⇒ a 3×3 box follows)
//! vir_size     i32 (bytes)      (virial; skipped)
//! pres_size    i32 (bytes)      (pressure; skipped)
//! top_size     i32              (legacy; must be 0)
//! sym_size     i32              (legacy; must be 0)
//! x_size       i32 (bytes)      (>0 ⇒ coordinates follow)
//! v_size       i32 (bytes)      (>0 ⇒ velocities follow)
//! f_size       i32 (bytes)      (>0 ⇒ forces follow)
//! natoms       i32
//! step         i32
//! nre          i32
//! t            real             (time, ps)
//! lambda       real
//! [box]   3×3 reals             if box_size  != 0
//! [vir]   3×3 reals             if vir_size  != 0   (skipped)
//! [pres]  3×3 reals             if pres_size != 0   (skipped)
//! [x]     natoms×3 reals (nm)   if x_size    != 0
//! [v]     natoms×3 reals        if v_size    != 0
//! [f]     natoms×3 reals        if f_size    != 0
//! ```
//!
//! `real` is `f32` or `f64` per frame; the width is inferred from a known
//! block's byte size (`box_size / 9`, falling back to `x_size / (natoms*3)`).
//!
//! # Output Frame
//!
//! - `atoms` block: `id` (1-based), `x`/`y`/`z` (nm), plus `vx`/`vy`/`vz` and
//!   `fx`/`fy`/`fz` when the frame carries velocities / forces.
//! - `frame.simbox`: from the box (GROMACS row-stored vectors → column-stored
//!   `SimBox` H matrix). Absent when `box_size == 0`.
//! - `frame.meta`: `step`, `time`, `lambda`.
//!
//! Coordinates stay in **nm** (GROMACS-native), matching the GRO reader.
//!
//! # Examples
//!
//! ```no_run
//! use molrs::io::trajectory::trr::{read_trr, open_trr, write_trr};
//! use molrs::io::reader::TrajectoryReader;
//!
//! # fn main() -> std::io::Result<()> {
//! let frames = read_trr("traj.trr")?;          // sequential, all frames
//! let mut r = open_trr("traj.trr")?;            // random access
//! let frame_5 = r.read_step(5)?;
//! write_trr("out.trr", &frames)?;
//! # Ok(())
//! # }
//! ```

use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::streaming::{BinaryFrameScanner, FrameIndexBuilder, FrameIndexEntry};
use crate::io::trajectory::xdr;
use crate::io::writer::{FrameWriter, Writer};
use molrs::spatial::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::store::frame_access::FrameAccess;
use molrs::types::{F, Idx};
use ndarray::{Array1, Array2, IxDyn, array};
use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::{BufRead, BufWriter, Cursor, Read, Result, Seek, SeekFrom, Write};
use std::path::Path;

const TRR_MAGIC: i32 = 1993;
const TRR_VERSION: &str = "GMX_trn_file";
const DIM: usize = 3;

fn invalid<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

fn unsupported<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, e.to_string())
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Parsed TRR per-frame header (sizes resolved to bytes / counts).
#[derive(Debug, Clone)]
pub struct TrrHeader {
    /// Whether reals in this frame are double precision.
    pub is_double: bool,
    /// Box block size in bytes (0 ⇒ no box).
    pub box_size: usize,
    /// Virial block size in bytes (skipped).
    pub vir_size: usize,
    /// Pressure block size in bytes (skipped).
    pub pres_size: usize,
    /// Coordinate block size in bytes (0 ⇒ no coordinates).
    pub x_size: usize,
    /// Velocity block size in bytes (0 ⇒ no velocities).
    pub v_size: usize,
    /// Force block size in bytes (0 ⇒ no forces).
    pub f_size: usize,
    /// Number of atoms.
    pub natoms: usize,
    /// Integration step.
    pub step: i32,
    /// Time (ps).
    pub t: f64,
    /// Free-energy lambda.
    pub lambda: f64,
}

impl TrrHeader {
    /// Total byte size of the data blocks that follow the header.
    fn data_len(&self) -> u64 {
        (self.box_size + self.vir_size + self.pres_size + self.x_size + self.v_size + self.f_size)
            as u64
    }
}

/// Infer real precision from a known block's byte size. Returns `true` for
/// double precision (`f64`), `false` for single (`f32`).
fn detect_double(
    box_size: i32,
    x_size: i32,
    v_size: i32,
    f_size: i32,
    natoms: usize,
) -> Result<bool> {
    let float_size = if box_size != 0 {
        box_size as usize / (DIM * DIM)
    } else if x_size != 0 {
        x_size as usize / (natoms * DIM)
    } else if v_size != 0 {
        v_size as usize / (natoms * DIM)
    } else if f_size != 0 {
        f_size as usize / (natoms * DIM)
    } else {
        return Err(invalid(
            "TRR frame has no box/x/v/f block to infer precision",
        ));
    };
    match float_size {
        4 => Ok(false),
        8 => Ok(true),
        other => Err(invalid(format!(
            "cannot infer TRR precision: implied real size {other} bytes (expected 4 or 8)"
        ))),
    }
}

/// Read one TRR frame header. Leaves the reader positioned at the first data
/// block. Propagates `UnexpectedEof` so the index scanner can stop cleanly.
fn read_header<R: Read>(r: &mut R) -> Result<TrrHeader> {
    let magic = xdr::read_i32(r)?;
    if magic != TRR_MAGIC {
        return Err(invalid(format!(
            "bad TRR magic {magic} (expected {TRR_MAGIC})"
        )));
    }
    // Version: a strlen+1 allocation hint, then the XDR string itself.
    let _ver_len = xdr::read_i32(r)?;
    let _version = xdr::read_string(r)?;

    let ir_size = xdr::read_i32(r)?;
    let e_size = xdr::read_i32(r)?;
    let box_size = xdr::read_i32(r)?;
    let vir_size = xdr::read_i32(r)?;
    let pres_size = xdr::read_i32(r)?;
    let top_size = xdr::read_i32(r)?;
    let sym_size = xdr::read_i32(r)?;
    let x_size = xdr::read_i32(r)?;
    let v_size = xdr::read_i32(r)?;
    let f_size = xdr::read_i32(r)?;
    let natoms = xdr::read_i32(r)?;
    let step = xdr::read_i32(r)?;
    let _nre = xdr::read_i32(r)?;

    if ir_size != 0 || e_size != 0 || top_size != 0 || sym_size != 0 {
        return Err(unsupported(
            "TRR with legacy ir/e/top/sym blocks is not supported",
        ));
    }
    if natoms <= 0 {
        return Err(invalid(format!("invalid TRR natoms {natoms}")));
    }
    let natoms = natoms as usize;
    for (name, size) in [
        ("box", box_size),
        ("vir", vir_size),
        ("pres", pres_size),
        ("x", x_size),
        ("v", v_size),
        ("f", f_size),
    ] {
        if size < 0 {
            return Err(invalid(format!("negative TRR {name}_size {size}")));
        }
    }

    let is_double = detect_double(box_size, x_size, v_size, f_size, natoms)?;
    let t = xdr::read_real(r, is_double)?;
    let lambda = xdr::read_real(r, is_double)?;

    Ok(TrrHeader {
        is_double,
        box_size: box_size as usize,
        vir_size: vir_size as usize,
        pres_size: pres_size as usize,
        x_size: x_size as usize,
        v_size: v_size as usize,
        f_size: f_size as usize,
        natoms,
        step,
        t,
        lambda,
    })
}

// ---------------------------------------------------------------------------
// Frame parsing
// ---------------------------------------------------------------------------

/// Read `count` reals at the header precision into an `f64` vector.
fn read_reals<R: Read>(r: &mut R, count: usize, is_double: bool) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(xdr::read_real(r, is_double)?);
    }
    Ok(out)
}

/// Build a `SimBox` from 9 row-stored box reals (`box[i][j]` = component `j`
/// of lattice vector `i`). `SimBox` stores lattice vectors as H columns, so
/// `H[r][c] = vals[c*3 + r]`.
/// Nanometre → ångström. GROMACS binary trajectories are nm; molrs is Å, so
/// the reader normalises here and the writer inverts it — the same boundary
/// rule the GRO reader follows.
pub(crate) const NM_TO_ANGSTROM: F = 10.0;

fn build_simbox(vals: &[f64]) -> Result<SimBox> {
    let h = Array2::from_shape_fn((DIM, DIM), |(r, c)| vals[c * DIM + r] as F);
    let origin = array![0.0 as F, 0.0, 0.0];
    SimBox::new(h, origin, [true; 3]).map_err(|e| invalid(format!("TRR box: {e:?}")))
}

fn insert_float_col(block: &mut Block, key: &str, vals: Vec<F>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid)?;
    block.insert(key, arr).map_err(invalid)
}

/// De-interleave an `rvec` block (`[x0,y0,z0, x1,y1,z1, …]`) into three axis
/// columns and insert them under `kx`/`ky`/`kz`.
fn insert_rvec_cols(
    block: &mut Block,
    rvec: &[f64],
    natoms: usize,
    kx: &str,
    ky: &str,
    kz: &str,
    scale: F,
) -> Result<()> {
    let mut x = Vec::with_capacity(natoms);
    let mut y = Vec::with_capacity(natoms);
    let mut z = Vec::with_capacity(natoms);
    for i in 0..natoms {
        x.push(rvec[i * DIM] as F * scale);
        y.push(rvec[i * DIM + 1] as F * scale);
        z.push(rvec[i * DIM + 2] as F * scale);
    }
    insert_float_col(block, kx, x)?;
    insert_float_col(block, ky, y)?;
    insert_float_col(block, kz, z)
}

/// Read the frame whose header starts at byte `offset`.
fn parse_frame_at<R: BufRead + Seek>(r: &mut R, offset: u64) -> Result<Frame> {
    r.seek(SeekFrom::Start(offset))?;
    let hdr = read_header(r)?;
    let natoms = hdr.natoms;

    let simbox = if hdr.box_size != 0 {
        let vals = read_reals(r, DIM * DIM, hdr.is_double)?;
        Some(scale_simbox(build_simbox(&vals)?, NM_TO_ANGSTROM)?)
    } else {
        None
    };
    if hdr.vir_size != 0 {
        let _ = read_reals(r, DIM * DIM, hdr.is_double)?;
    }
    if hdr.pres_size != 0 {
        let _ = read_reals(r, DIM * DIM, hdr.is_double)?;
    }
    let x = if hdr.x_size != 0 {
        Some(read_reals(r, natoms * DIM, hdr.is_double)?)
    } else {
        None
    };
    let v = if hdr.v_size != 0 {
        Some(read_reals(r, natoms * DIM, hdr.is_double)?)
    } else {
        None
    };
    let f = if hdr.f_size != 0 {
        Some(read_reals(r, natoms * DIM, hdr.is_double)?)
    } else {
        None
    };

    let mut atoms = Block::new();
    let id_arr = Array1::from_iter(1..=natoms as Idx)
        .into_shape_with_order(IxDyn(&[natoms]))
        .map_err(invalid)?;
    atoms.insert("id", id_arr).map_err(invalid)?;
    if let Some(x) = &x {
        insert_rvec_cols(&mut atoms, x, natoms, "x", "y", "z", NM_TO_ANGSTROM)?;
    }
    if let Some(v) = &v {
        // nm/ps → Å/ps.
        insert_rvec_cols(&mut atoms, v, natoms, "vx", "vy", "vz", NM_TO_ANGSTROM)?;
    }
    if let Some(f) = &f {
        // kJ/mol/nm → kJ/mol/Å: same energy, per Å instead of per nm.
        insert_rvec_cols(
            &mut atoms,
            f,
            natoms,
            "fx",
            "fy",
            "fz",
            1.0 / NM_TO_ANGSTROM,
        )?;
    }

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = simbox;
    frame.meta.insert("step", hdr.step);
    frame.meta.insert("time", hdr.t);
    frame.meta.insert("lambda", hdr.lambda);
    Ok(frame)
}

/// Scan the whole file, recording the byte offset of each frame header.
fn scan_offsets<R: BufRead + Seek>(r: &mut R) -> Result<Vec<u64>> {
    let end = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    let mut offsets = Vec::new();
    loop {
        let pos = r.stream_position()?;
        if pos >= end {
            break;
        }
        let hdr = match read_header(r) {
            Ok(h) => h,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };
        r.seek(SeekFrom::Current(hdr.data_len() as i64))?;
        offsets.push(pos);
    }
    Ok(offsets)
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// TRR trajectory reader with O(1) random access (offsets indexed lazily on
/// first use).
pub struct TrrReader<R: BufRead + Seek> {
    reader: R,
    offsets: OnceCell<Vec<u64>>,
    cursor: usize,
}

impl<R: BufRead + Seek> TrrReader<R> {
    /// Wrap `reader`; index building is deferred.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            offsets: OnceCell::new(),
            cursor: 0,
        }
    }

    fn ensure_index(&mut self) -> Result<()> {
        if self.offsets.get().is_some() {
            return Ok(());
        }
        let offs = scan_offsets(&mut self.reader)?;
        self.offsets
            .set(offs)
            .map_err(|_| std::io::Error::other("failed to set TRR index"))?;
        Ok(())
    }
}

impl<R: BufRead + Seek> Reader for TrrReader<R> {
    type R = R;
    fn new(reader: Self::R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for TrrReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        self.ensure_index()?;
        let cursor = self.cursor;
        let off = match self.offsets.get().and_then(|o| o.get(cursor).copied()) {
            Some(o) => o,
            None => return Ok(None),
        };
        let frame = parse_frame_at(&mut self.reader, off)?;
        self.cursor += 1;
        crate::io::reader::validated(Some(frame))
    }
}

impl<R: BufRead + Seek> TrajectoryReader for TrrReader<R> {
    fn build_index(&mut self) -> Result<()> {
        self.ensure_index()
    }

    fn read_step(&mut self, step: usize) -> Result<Option<Frame>> {
        self.ensure_index()?;
        let off = match self.offsets.get().and_then(|o| o.get(step).copied()) {
            Some(o) => o,
            None => return Ok(None),
        };
        parse_frame_at(&mut self.reader, off).map(Some)
    }

    fn len(&mut self) -> Result<usize> {
        self.ensure_index()?;
        Ok(self.offsets.get().expect("index set").len())
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

fn axis<FA: FrameAccess>(frame: &FA, key: &str) -> Option<Vec<f64>> {
    frame
        .get_float("atoms", key)
        .map(|view| view.iter().copied().collect::<Vec<f64>>())
}

fn write_rvecs<W: Write>(w: &mut W, x: &[f64], y: &[f64], z: &[f64], scale: F) -> Result<()> {
    for i in 0..x.len() {
        xdr::write_f32(w, (x[i] * scale) as f32)?;
        xdr::write_f32(w, (y[i] * scale) as f32)?;
        xdr::write_f32(w, (z[i] * scale) as f32)?;
    }
    Ok(())
}

/// A copy of `sb` with every length multiplied by `scale`.
fn scale_simbox(sb: SimBox, scale: F) -> Result<SimBox> {
    let h = sb.h_view().to_owned() * scale;
    let origin = sb.origin_view().to_owned() * scale;
    SimBox::new(h, origin, sb.pbc()).map_err(|e| invalid(format!("TRR box scale: {e:?}")))
}

/// Write one frame in single-precision TRR format.
fn write_trr_frame<W: Write, FA: FrameAccess>(w: &mut W, frame: &FA) -> Result<()> {
    let natoms = frame
        .visit_block("atoms", |a| a.nrows().unwrap_or(0))
        .ok_or_else(|| invalid("TRR write: frame has no atoms block"))?;
    if natoms == 0 {
        return Err(invalid("TRR write: atoms block is empty"));
    }

    let xs = axis(frame, "x").ok_or_else(|| invalid("TRR write: atoms.x missing"))?;
    let ys = axis(frame, "y").ok_or_else(|| invalid("TRR write: atoms.y missing"))?;
    let zs = axis(frame, "z").ok_or_else(|| invalid("TRR write: atoms.z missing"))?;
    if xs.len() != natoms || ys.len() != natoms || zs.len() != natoms {
        return Err(invalid("TRR write: coordinate columns disagree on length"));
    }
    let vel = match (axis(frame, "vx"), axis(frame, "vy"), axis(frame, "vz")) {
        (Some(a), Some(b), Some(c)) => Some((a, b, c)),
        _ => None,
    };
    let force = match (axis(frame, "fx"), axis(frame, "fy"), axis(frame, "fz")) {
        (Some(a), Some(b), Some(c)) => Some((a, b, c)),
        _ => None,
    };
    let has_box = frame.simbox_ref().is_some();

    let meta = frame.meta_ref();
    let step = meta
        .get("step")
        .and_then(|value| value.as_i32())
        .unwrap_or(0);
    let time = meta
        .get("time")
        .and_then(|value| value.as_f32())
        .unwrap_or(0.0);
    let lambda: f32 = meta
        .get("lambda")
        .and_then(|value| value.as_f32())
        .unwrap_or(0.0);

    const RSIZE: usize = 4; // single precision
    let rvec_bytes = (natoms * DIM * RSIZE) as i32;

    xdr::write_i32(w, TRR_MAGIC)?;
    xdr::write_i32(w, (TRR_VERSION.len() + 1) as i32)?;
    xdr::write_string(w, TRR_VERSION)?;
    xdr::write_i32(w, 0)?; // ir_size
    xdr::write_i32(w, 0)?; // e_size
    xdr::write_i32(
        w,
        if has_box {
            (DIM * DIM * RSIZE) as i32
        } else {
            0
        },
    )?;
    xdr::write_i32(w, 0)?; // vir_size
    xdr::write_i32(w, 0)?; // pres_size
    xdr::write_i32(w, 0)?; // top_size
    xdr::write_i32(w, 0)?; // sym_size
    xdr::write_i32(w, rvec_bytes)?; // x_size
    xdr::write_i32(w, if vel.is_some() { rvec_bytes } else { 0 })?;
    xdr::write_i32(w, if force.is_some() { rvec_bytes } else { 0 })?;
    xdr::write_i32(w, natoms as i32)?;
    xdr::write_i32(w, step)?;
    xdr::write_i32(w, 0)?; // nre
    xdr::write_f32(w, time)?;
    xdr::write_f32(w, lambda)?;

    if has_box {
        let sb = frame.simbox_ref().expect("box present");
        let h = sb.h_view().to_owned() / NM_TO_ANGSTROM;
        // GROMACS row-stored: box[i][j] = component j of lattice vector i = H[j][i].
        for i in 0..DIM {
            for j in 0..DIM {
                xdr::write_f32(w, h[(j, i)] as f32)?;
            }
        }
    }
    // Å → nm on the way out, mirroring the reader.
    write_rvecs(w, &xs, &ys, &zs, 1.0 / NM_TO_ANGSTROM)?;
    if let Some((vx, vy, vz)) = &vel {
        write_rvecs(w, vx, vy, vz, 1.0 / NM_TO_ANGSTROM)?;
    }
    if let Some((fx, fy, fz)) = &force {
        write_rvecs(w, fx, fy, fz, NM_TO_ANGSTROM)?;
    }
    Ok(())
}

/// TRR trajectory writer (single precision).
pub struct TrrWriter<W: Write> {
    writer: W,
}

impl<W: Write> TrrWriter<W> {
    /// Create a new TRR writer.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> Writer for TrrWriter<W> {
    type W = W;
    fn new(writer: Self::W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write> FrameWriter for TrrWriter<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_before_write(frame)?;
        write_trr_frame(&mut self.writer, frame)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Read every frame of a TRR file into memory.
pub fn read_trr<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>> {
    let reader = crate::io::reader::open_seekable(path)?;
    crate::io::reader::collect_frames(&mut TrrReader::new(reader))
}

/// Open a TRR file for trajectory-style random access.
pub fn open_trr<P: AsRef<Path>>(path: P) -> Result<TrrReader<Box<dyn ReadSeek>>> {
    Ok(TrrReader::new(crate::io::reader::open_seekable(path)?))
}

/// Write a slice of frames to a TRR file (single precision).
pub fn write_trr<P: AsRef<Path>, FA: FrameAccess>(path: P, frames: &[FA]) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    for f in frames {
        write_trr_frame(&mut w, f)?;
    }
    w.flush()
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Byte length of one complete TRR frame starting at `bytes[0]`.
///
/// `Ok(None)` means the prefix is valid so far but truncated.
fn try_trr_frame_len(bytes: &[u8]) -> Result<Option<u32>> {
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes.len() < 4 {
        return Ok(None);
    }
    let magic = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
    if magic != TRR_MAGIC {
        return Err(invalid(format!(
            "bad TRR magic {magic} (expected {TRR_MAGIC})"
        )));
    }
    let mut cursor = Cursor::new(bytes);
    let header = match read_header(&mut cursor) {
        Ok(h) => h,
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    };
    let total = cursor.position().saturating_add(header.data_len());
    if (bytes.len() as u64) < total {
        return Ok(None);
    }
    u32::try_from(total)
        .map(Some)
        .map_err(|_| invalid("TRR frame larger than 4 GiB"))
}

/// Parse exactly one TRR frame from a tightly-bounded byte slice.
pub fn parse_frame_bytes(bytes: &[u8]) -> Result<Frame> {
    let mut cursor = Cursor::new(bytes);
    parse_frame_at(&mut cursor, 0)
}

/// Streaming frame indexer for TRR files. Each frame header carries the
/// sizes of the following data blocks.
pub struct TrrIndexBuilder {
    scan: BinaryFrameScanner,
    error: Option<std::io::Error>,
}

impl Default for TrrIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TrrIndexBuilder {
    /// Create an empty indexer.
    pub fn new() -> Self {
        Self {
            scan: BinaryFrameScanner::new(),
            error: None,
        }
    }
}

impl FrameIndexBuilder for TrrIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        if self.error.is_some() {
            return;
        }
        if let Err(err) = self.scan.feed(chunk, global_offset, try_trr_frame_len) {
            self.error = Some(err);
        }
    }

    fn drain(&mut self) -> Vec<FrameIndexEntry> {
        self.scan.drain()
    }

    fn finish(self: Box<Self>) -> Result<Vec<FrameIndexEntry>> {
        if let Some(err) = self.error {
            return Err(err);
        }
        self.scan.finish(try_trr_frame_len)
    }

    fn bytes_seen(&self) -> u64 {
        self.scan.bytes_seen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_double_from_box_size() {
        assert!(
            !detect_double(36, 0, 0, 0, 10).unwrap(),
            "36/9 = 4 → single"
        );
        assert!(detect_double(72, 0, 0, 0, 10).unwrap(), "72/9 = 8 → double");
    }

    #[test]
    fn detect_double_falls_back_to_coord_blocks() {
        // No box: infer from x_size = natoms*3*size (120/30 = 4, 240/30 = 8).
        assert!(!detect_double(0, 120, 0, 0, 10).unwrap());
        assert!(detect_double(0, 240, 0, 0, 10).unwrap());
        // Then velocities, then forces.
        assert!(!detect_double(0, 0, 120, 0, 10).unwrap());
        assert!(detect_double(0, 0, 0, 240, 10).unwrap());
    }

    #[test]
    fn detect_double_errors_without_any_block() {
        assert!(detect_double(0, 0, 0, 0, 10).is_err());
        assert!(detect_double(45, 0, 0, 0, 10).is_err()); // 45/9 = 5, nonsense
    }

    #[test]
    fn build_simbox_transposes_row_vectors_to_columns() {
        // Row-stored: vector i = (vals[3i], vals[3i+1], vals[3i+2]).
        let vals = vec![1.0, 0.0, 0.0, 0.5, 2.0, 0.0, 0.1, 0.2, 3.0];
        let sb = build_simbox(&vals).unwrap();
        let h = sb.h_view().to_owned();
        for c in 0..3 {
            for r in 0..3 {
                assert!(
                    (h[(r, c)] - vals[c * 3 + r]).abs() < 1e-12,
                    "H[{r}][{c}] = {} expected {}",
                    h[(r, c)],
                    vals[c * 3 + r]
                );
            }
        }
    }

    fn trr_frame(x0: f64) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert("id", Array1::from(vec![1u64, 2]).into_dyn())
            .unwrap();
        atoms
            .insert("x", Array1::from(vec![x0, x0 + 1.0]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from(vec![0.0, 0.0]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from(vec![0.0, 0.0]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let h = array![[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]];
        frame.simbox = Some(SimBox::new(h, array![0.0, 0.0, 0.0], [true; 3]).unwrap());
        frame
    }

    fn write_trr_mem(frames: &[Frame]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut writer = TrrWriter::new(&mut buf);
            for frame in frames {
                FrameWriter::write(&mut writer, frame).expect("write trr");
            }
        }
        buf
    }

    fn trr_index_chunked(bytes: &[u8], chunk: usize) -> Vec<FrameIndexEntry> {
        let mut builder = Box::new(TrrIndexBuilder::new());
        let mut offset = 0u64;
        let mut out = Vec::new();
        for piece in bytes.chunks(chunk.max(1)) {
            builder.feed(piece, offset);
            offset += piece.len() as u64;
            out.extend(builder.drain());
        }
        out.extend(builder.finish().expect("finish trr index"));
        out
    }

    #[test]
    fn trr_streaming_single_shot_matches_chunks() {
        let frames = [trr_frame(0.25), trr_frame(1.25)];
        let bytes = write_trr_mem(&frames);
        let one_shot = trr_index_chunked(&bytes, bytes.len());
        assert_eq!(one_shot.len(), 2);
        for cs in [1usize, 8, 21, 64] {
            assert_eq!(
                trr_index_chunked(&bytes, cs),
                one_shot,
                "chunk size {cs} produced a different index"
            );
        }
        for (i, entry) in one_shot.iter().enumerate() {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            let parsed = parse_frame_bytes(&bytes[lo..hi]).expect("parse trr");
            let x = parsed.get("atoms").unwrap().get_float("x").unwrap();
            assert!(
                (x[0] - frames[i].get("atoms").unwrap().get_float("x").unwrap()[0]).abs() < 1e-5
            );
        }
    }
}
