//! DCD binary trajectory file reader and writer.
//!
//! DCD is the de-facto binary trajectory format used by CHARMM, NAMD,
//! OpenMM, and (optionally) LAMMPS. The on-disk layout is a sequence of
//! Fortran "unformatted" records: each record is `[len][payload][len]`
//! with two equal length markers framing the payload. Marker width is 4
//! bytes by default and 8 bytes when CHARMM was compiled with
//! `-frecord-marker=8`. Both little- and big-endian byte orders are
//! supported.
//!
//! # Supported features
//!
//! - 32- and 64-bit Fortran record markers
//! - Little- and big-endian byte order
//! - Optional periodic-box record (NAMD/CHARMM "extra block")
//! - Triclinic boxes encoded as either cosines or degrees
//! - Optional 4D dynamics (W coordinate)
//! - Fixed-atom subsets (frame 0 carries full coords; later frames only
//!   the free atoms)
//! - O(1) random access via `TrajReader::read_step` (frame size is
//!   constant, no scan needed)
//! - Writer (NAMD-style: little-endian, 4-byte markers, cosine angles,
//!   no fixed atoms, no 4D)
//!
//! # Examples
//!
//! ```no_run
//! use molrs_io::dcd::{read_dcd, open_dcd, write_dcd};
//! use molrs_io::reader::TrajReader;
//!
//! # fn main() -> std::io::Result<()> {
//! // Read all frames
//! let frames = read_dcd("trajectory.dcd")?;
//!
//! // Random access via TrajReader
//! let mut reader = open_dcd("trajectory.dcd")?;
//! let frame_5 = reader.read_step(5)?;
//!
//! // Write frames
//! write_dcd("output.dcd", &frames)?;
//! # Ok(())
//! # }
//! ```

use crate::reader::{FrameReader, ReadSeek, Reader, TrajReader};
use crate::writer::{FrameWriter, Writer};
use molrs::block::Block;
use molrs::frame::Frame;
use molrs::frame_access::FrameAccess;
use molrs::region::simbox::SimBox;
use molrs::types::{F, I, Pbc3};
use ndarray::{Array1, Array2, IxDyn, array};
use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::{BufRead, Read, Seek, SeekFrom, Write};
use std::path::Path;

// ============================================================================
// Helpers
// ============================================================================

fn err_mapper<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

fn unsupported<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, e.to_string())
}

/// Byte order of a DCD file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little-endian (x86, ARM, NAMD on most platforms).
    Le,
    /// Big-endian (s390x, older PowerPC).
    Be,
}

/// Width of Fortran record-length markers in a DCD file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerSize {
    /// 4-byte markers (default).
    Four,
    /// 8-byte markers (`-frecord-marker=8` builds of CHARMM).
    Eight,
}

impl MarkerSize {
    #[inline]
    fn bytes(self) -> u64 {
        match self {
            MarkerSize::Four => 4,
            MarkerSize::Eight => 8,
        }
    }
}

#[inline]
fn read_u32(buf: &[u8; 4], byte_order: ByteOrder) -> u32 {
    match byte_order {
        ByteOrder::Le => u32::from_le_bytes(*buf),
        ByteOrder::Be => u32::from_be_bytes(*buf),
    }
}

#[inline]
fn read_i32(buf: &[u8; 4], byte_order: ByteOrder) -> i32 {
    match byte_order {
        ByteOrder::Le => i32::from_le_bytes(*buf),
        ByteOrder::Be => i32::from_be_bytes(*buf),
    }
}

#[inline]
fn read_f32(buf: &[u8; 4], byte_order: ByteOrder) -> f32 {
    match byte_order {
        ByteOrder::Le => f32::from_le_bytes(*buf),
        ByteOrder::Be => f32::from_be_bytes(*buf),
    }
}

#[inline]
fn read_f64(buf: &[u8; 8], byte_order: ByteOrder) -> f64 {
    match byte_order {
        ByteOrder::Le => f64::from_le_bytes(*buf),
        ByteOrder::Be => f64::from_be_bytes(*buf),
    }
}

/// Read a Fortran record-length marker at the current reader position.
fn read_marker<R: Read>(
    reader: &mut R,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
) -> std::io::Result<u64> {
    match marker_size {
        MarkerSize::Four => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(read_u32(&buf, byte_order) as u64)
        }
        MarkerSize::Eight => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            let v = match byte_order {
                ByteOrder::Le => u64::from_le_bytes(buf),
                ByteOrder::Be => u64::from_be_bytes(buf),
            };
            Ok(v)
        }
    }
}

/// Write a Fortran record-length marker.
fn write_marker<W: Write>(
    writer: &mut W,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
    value: u64,
) -> std::io::Result<()> {
    match marker_size {
        MarkerSize::Four => {
            let v = u32::try_from(value).map_err(err_mapper)?;
            let bytes = match byte_order {
                ByteOrder::Le => v.to_le_bytes(),
                ByteOrder::Be => v.to_be_bytes(),
            };
            writer.write_all(&bytes)
        }
        MarkerSize::Eight => {
            let bytes = match byte_order {
                ByteOrder::Le => value.to_le_bytes(),
                ByteOrder::Be => value.to_be_bytes(),
            };
            writer.write_all(&bytes)
        }
    }
}

/// Read one Fortran record's payload, asserting the trailing marker matches.
fn read_record<R: Read>(
    reader: &mut R,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
) -> std::io::Result<Vec<u8>> {
    let leading = read_marker(reader, byte_order, marker_size)?;
    let mut payload = vec![0u8; leading as usize];
    reader.read_exact(&mut payload)?;
    let trailing = read_marker(reader, byte_order, marker_size)?;
    if leading != trailing {
        return Err(err_mapper(format!(
            "Fortran record marker mismatch: leading={}, trailing={}",
            leading, trailing
        )));
    }
    Ok(payload)
}

/// Read one Fortran record into a caller-supplied buffer (resizes as needed).
fn read_record_into<R: Read>(
    reader: &mut R,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
    buf: &mut Vec<u8>,
) -> std::io::Result<()> {
    let leading = read_marker(reader, byte_order, marker_size)?;
    buf.resize(leading as usize, 0);
    reader.read_exact(buf)?;
    let trailing = read_marker(reader, byte_order, marker_size)?;
    if leading != trailing {
        return Err(err_mapper(format!(
            "Fortran record marker mismatch: leading={}, trailing={}",
            leading, trailing
        )));
    }
    Ok(())
}

// ============================================================================
// Header
// ============================================================================

/// Parsed DCD header.
#[derive(Debug, Clone)]
pub struct DcdHeader {
    /// Byte order of the file.
    pub byte_order: ByteOrder,
    /// Width of Fortran record-length markers.
    pub marker_size: MarkerSize,
    /// CHARMM version field; 0 indicates an X-PLOR-style header.
    pub charmm_ver: i32,
    /// Number of frames as recorded in the header (NSET).
    pub nset: u32,
    /// Starting timestep (ISTART).
    pub istart: i32,
    /// Number of integration steps between saved frames (NSAVC).
    pub nsavc: i32,
    /// Total number of atoms (NATOMS).
    pub natoms: u32,
    /// Number of fixed atoms (NAMNF).
    pub namnf: u32,
    /// Whether per-frame box records are present.
    pub has_box: bool,
    /// Whether per-frame W coordinates are present (4D dynamics).
    pub has_4d: bool,
    /// Integration timestep in AKMA units (DELTA).
    pub delta: f64,
    /// Concatenated title strings, trimmed.
    pub title: String,
    /// 1-indexed free-atom indices (only present when NAMNF > 0).
    pub free_atoms: Option<Vec<i32>>,
    /// Frame-0 X/Y/Z coords, cached when NAMNF > 0 so later frames can be
    /// reconstructed by overlaying the free-atom subset on top of these.
    fixed_seed: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Byte offset of the first per-frame record.
    pub data_offset: u64,
    /// Byte size of frame 0 (full NATOMS).
    pub frame_size_first: u64,
    /// Byte size of frames 1..NSET (NATOMS - NAMNF when NAMNF > 0; otherwise
    /// equal to `frame_size_first`).
    pub frame_size_rest: u64,
}

impl DcdHeader {
    /// Byte offset of frame `n` in the file.
    fn frame_offset(&self, n: usize) -> u64 {
        if n == 0 {
            self.data_offset
        } else {
            self.data_offset + self.frame_size_first + (n as u64 - 1) * self.frame_size_rest
        }
    }

    /// Number of effective coordinate values in frame `n`.
    fn natoms_eff(&self, n: usize) -> u32 {
        if self.namnf == 0 || n == 0 {
            self.natoms
        } else {
            self.natoms - self.namnf
        }
    }
}

/// Detect byte order and Fortran marker width.
///
/// The first record always wraps the 84-byte CORD header. The marker layout
/// is ambiguous when read in isolation (`84u64.to_le_bytes()` and
/// `84u32.to_le_bytes() ++ [0u8; 4]` are byte-for-byte identical), so we
/// disambiguate by checking which marker width places the ASCII `"CORD"`
/// magic immediately after the leading marker.
fn detect_endianness_and_marker<R: Read>(
    reader: &mut R,
) -> std::io::Result<(ByteOrder, MarkerSize)> {
    let mut buf = [0u8; 12];
    reader.read_exact(&mut buf)?;

    // 4-byte marker: leading marker at [0..4], CORD at [4..8].
    if &buf[4..8] == b"CORD" {
        let leading: [u8; 4] = buf[..4].try_into().unwrap();
        if u32::from_le_bytes(leading) == 84 {
            return Ok((ByteOrder::Le, MarkerSize::Four));
        }
        if u32::from_be_bytes(leading) == 84 {
            return Ok((ByteOrder::Be, MarkerSize::Four));
        }
    }

    // 8-byte marker: leading marker at [0..8], CORD at [8..12].
    if &buf[8..12] == b"CORD" {
        let leading: [u8; 8] = buf[..8].try_into().unwrap();
        if u64::from_le_bytes(leading) == 84 {
            return Ok((ByteOrder::Le, MarkerSize::Eight));
        }
        if u64::from_be_bytes(leading) == 84 {
            return Ok((ByteOrder::Be, MarkerSize::Eight));
        }
    }

    Err(err_mapper(format!(
        "not a DCD file: first 12 bytes {:02x?} do not match any known marker layout",
        buf
    )))
}

/// Parse the full DCD header from the current reader position. Advances the
/// reader past the header (and, when NAMNF > 0, past frame 0 — frame 0's
/// coordinates are cached in the returned header).
fn parse_header<R: BufRead + Seek>(reader: &mut R) -> std::io::Result<DcdHeader> {
    reader.seek(SeekFrom::Start(0))?;
    let (byte_order, marker_size) = detect_endianness_and_marker(reader)?;

    // Re-read from the start now that we know the layout.
    reader.seek(SeekFrom::Start(0))?;

    // -- Header record 1 (84 bytes) --
    let h1 = read_record(reader, byte_order, marker_size)?;
    if h1.len() != 84 {
        return Err(err_mapper(format!(
            "header record 1 has {} bytes, expected 84",
            h1.len()
        )));
    }
    if &h1[0..4] != b"CORD" {
        return Err(err_mapper(format!(
            "header magic is {:?}, expected 'CORD'",
            std::str::from_utf8(&h1[0..4]).unwrap_or("?")
        )));
    }

    let read_i32_at = |off: usize| -> i32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&h1[off..off + 4]);
        read_i32(&buf, byte_order)
    };

    let nset = read_i32_at(4) as u32;
    let istart = read_i32_at(8);
    let nsavc = read_i32_at(12);
    let namnf = read_i32_at(36) as u32;
    let charmm_ver = read_i32_at(80);

    // DELTA position depends on header flavor: CHARMM stores f32 at icntrl[9]
    // (payload offset 40); X-PLOR stores f64 spanning icntrl[9..10] (payload
    // offsets 40..48). icntrl[19] (payload offset 80) chooses the flavor.
    let delta = if charmm_ver != 0 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&h1[40..44]);
        read_f32(&buf, byte_order) as f64
    } else {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&h1[40..48]);
        read_f64(&buf, byte_order)
    };

    // -- Header record 2 (title) --
    let h2 = read_record(reader, byte_order, marker_size)?;
    if h2.len() < 4 {
        return Err(err_mapper("title record too short"));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&h2[0..4]);
    let ntitle = read_i32(&buf, byte_order);
    if ntitle < 0 {
        return Err(err_mapper(format!("invalid NTITLE={}", ntitle)));
    }
    let expected_title_len = 4 + (ntitle as usize) * 80;
    if h2.len() < expected_title_len {
        return Err(err_mapper(format!(
            "title record has {} bytes, expected at least {}",
            h2.len(),
            expected_title_len
        )));
    }
    let mut title = String::with_capacity((ntitle as usize) * 80);
    for i in 0..ntitle as usize {
        let off = 4 + i * 80;
        let s = String::from_utf8_lossy(&h2[off..off + 80]);
        title.push_str(s.trim_end_matches([' ', '\0']));
        title.push('\n');
    }
    let title = title.trim_end_matches('\n').to_owned();

    // -- Header record 3 (NATOMS) --
    let h3 = read_record(reader, byte_order, marker_size)?;
    if h3.len() != 4 {
        return Err(err_mapper(format!(
            "NATOMS record has {} bytes, expected 4",
            h3.len()
        )));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&h3[0..4]);
    let natoms = read_i32(&buf, byte_order);
    if natoms <= 0 {
        return Err(err_mapper(format!("invalid NATOMS={}", natoms)));
    }
    let natoms = natoms as u32;

    if namnf > natoms {
        return Err(err_mapper(format!(
            "NAMNF={} exceeds NATOMS={}",
            namnf, natoms
        )));
    }

    // -- Optional fixed-atom record --
    let free_atoms = if namnf > 0 {
        let payload = read_record(reader, byte_order, marker_size)?;
        let nfree = (natoms - namnf) as usize;
        if payload.len() != nfree * 4 {
            return Err(err_mapper(format!(
                "fixed-atom record has {} bytes, expected {}",
                payload.len(),
                nfree * 4
            )));
        }
        let mut indices = Vec::with_capacity(nfree);
        for i in 0..nfree {
            let mut b = [0u8; 4];
            b.copy_from_slice(&payload[i * 4..(i + 1) * 4]);
            indices.push(read_i32(&b, byte_order));
        }
        Some(indices)
    } else {
        None
    };

    // -- Now positioned at the start of per-frame data. --
    let data_offset = reader.stream_position()?;
    let m = marker_size.bytes();

    // The header flag positions (CHARMM icntrl[10..11] vs X-PLOR
    // icntrl[11..12]) disagree across writers. Trying to honor the bits in
    // the file leads to wrong answers on real-world fixtures. Instead, we
    // probe the first per-frame record by peeking at its leading marker:
    // a box record is exactly 48 bytes; a coord record is `natoms * 4`.
    // From there file size disambiguates 3 vs 4 coord records (4D dynamics).
    let end = reader.seek(SeekFrom::End(0))?;
    let trailing = end.saturating_sub(data_offset);

    if trailing == 0 {
        return Ok(DcdHeader {
            byte_order,
            marker_size,
            charmm_ver,
            nset: 0,
            istart,
            nsavc,
            natoms,
            namnf,
            has_box: false,
            has_4d: false,
            delta,
            title,
            free_atoms,
            fixed_seed: None,
            data_offset,
            frame_size_first: 0,
            frame_size_rest: 0,
        });
    }

    reader.seek(SeekFrom::Start(data_offset))?;
    let first_marker = read_marker(reader, byte_order, marker_size)?;
    reader.seek(SeekFrom::Start(data_offset))?;

    let coord_marker_full = (natoms as u64) * 4;
    let coord_rec_full = 2 * m + coord_marker_full;
    let coord_rec_eff = 2 * m + ((natoms - namnf) as u64) * 4;

    // The first marker constrains has_box, but only loosely: a 48-byte
    // record can be either the box block or an X-coord record when natoms is
    // exactly 12. We enumerate all four (has_box, has_4d) combos that match
    // the first marker, and pick the unique one whose frame layout divides
    // the file's data section evenly.
    let mut candidates: Vec<(bool, bool, u64, u64)> = Vec::new();
    for &has_box_candidate in &[true, false] {
        let box_part = if has_box_candidate { 2 * m + 48 } else { 0 };

        // First marker must match what this candidate predicts.
        let predicted_first = if has_box_candidate {
            48
        } else {
            coord_marker_full
        };
        if first_marker != predicted_first {
            continue;
        }

        for &has_4d_candidate in &[false, true] {
            let n_recs = if has_4d_candidate { 4 } else { 3 };
            let f_first = box_part + coord_rec_full * n_recs;
            let f_rest = box_part + coord_rec_eff * n_recs;
            if f_first == 0 || f_rest == 0 || trailing < f_first {
                continue;
            }
            let after_first = trailing - f_first;
            if after_first % f_rest != 0 {
                continue;
            }
            candidates.push((has_box_candidate, has_4d_candidate, f_first, f_rest));
        }
    }
    if candidates.is_empty() {
        return Err(err_mapper(format!(
            "no per-frame layout (has_box, has_4d) is consistent with file size {} after header (natoms={}, namnf={}, first_marker={})",
            trailing, natoms, namnf, first_marker
        )));
    }
    // When multiple candidates fit, prefer the simpler one: no 4D, then no
    // box. In practice ambiguity is rare (it requires natoms == 12).
    candidates.sort_by_key(|&(hb, h4, _, _)| (h4 as u8, hb as u8));
    let (has_box, has_4d, frame_size_first, frame_size_rest) = candidates[0];

    let actual_nset = 1 + (trailing - frame_size_first) / frame_size_rest;

    let mut header = DcdHeader {
        byte_order,
        marker_size,
        charmm_ver,
        nset: u32::try_from(actual_nset).unwrap_or(u32::MAX),
        istart,
        nsavc,
        natoms,
        namnf,
        has_box,
        has_4d,
        delta,
        title,
        free_atoms,
        fixed_seed: None,
        data_offset,
        frame_size_first,
        frame_size_rest,
    };
    let _ = nset; // header NSET hint is no longer authoritative; trust file size

    reader.seek(SeekFrom::Start(data_offset))?;

    // Cache frame-0 coordinates if we need them to reconstruct fixed atoms.
    if header.namnf > 0 && header.nset > 0 {
        let (xs, ys, zs, _w) = read_coord_payload(reader, &header, 0)?;
        header.fixed_seed = Some((xs, ys, zs));
        // Reset; read_step will re-read frame 0 if requested.
        reader.seek(SeekFrom::Start(data_offset))?;
    }

    Ok(header)
}

// ============================================================================
// Frame parsing
// ============================================================================

/// Read the coordinate (and box) payload for frame `n`. Reader must be
/// positioned at the start of frame `n`. Returns `(x, y, z, w)` where `w` is
/// `Some` iff `header.has_4d`.
///
/// For fixed-atom files with `n >= 1`, the returned arrays still have length
/// `natoms`: fixed atoms are filled from `header.fixed_seed`.
type Coords = (Vec<f32>, Vec<f32>, Vec<f32>, Option<Vec<f32>>);

fn read_coord_payload<R: BufRead + Seek>(
    reader: &mut R,
    header: &DcdHeader,
    n: usize,
) -> std::io::Result<Coords> {
    let natoms = header.natoms as usize;
    let natoms_eff = header.natoms_eff(n) as usize;

    // Skip past the box record — caller handles it separately when needed.
    if header.has_box {
        let _ = read_record(reader, header.byte_order, header.marker_size)?;
    }

    let mut buf = Vec::with_capacity(natoms_eff * 4);

    let read_axis = |reader: &mut R, buf: &mut Vec<u8>| -> std::io::Result<Vec<f32>> {
        read_record_into(reader, header.byte_order, header.marker_size, buf)?;
        if buf.len() != natoms_eff * 4 {
            return Err(err_mapper(format!(
                "coord record has {} bytes, expected {}",
                buf.len(),
                natoms_eff * 4
            )));
        }
        let mut out = vec![0.0f32; natoms_eff];
        for i in 0..natoms_eff {
            let mut b = [0u8; 4];
            b.copy_from_slice(&buf[i * 4..(i + 1) * 4]);
            out[i] = read_f32(&b, header.byte_order);
        }
        Ok(out)
    };

    let xs_eff = read_axis(reader, &mut buf)?;
    let ys_eff = read_axis(reader, &mut buf)?;
    let zs_eff = read_axis(reader, &mut buf)?;
    let w_eff = if header.has_4d {
        Some(read_axis(reader, &mut buf)?)
    } else {
        None
    };

    if header.namnf == 0 || n == 0 {
        return Ok((xs_eff, ys_eff, zs_eff, w_eff));
    }

    // Reconstruct full-length arrays by overlaying free atoms onto the cached
    // frame-0 fixed coords.
    let (sx, sy, sz) = header
        .fixed_seed
        .as_ref()
        .ok_or_else(|| err_mapper("fixed_seed missing for fixed-atom DCD"))?;
    let free = header
        .free_atoms
        .as_ref()
        .ok_or_else(|| err_mapper("free_atoms missing for fixed-atom DCD"))?;
    if free.len() != natoms_eff {
        return Err(err_mapper(format!(
            "free_atoms has {} entries, expected {}",
            free.len(),
            natoms_eff
        )));
    }

    let mut xs = sx.clone();
    let mut ys = sy.clone();
    let mut zs = sz.clone();
    for (slot, &one_indexed) in free.iter().enumerate() {
        if one_indexed < 1 || (one_indexed as usize) > natoms {
            return Err(err_mapper(format!(
                "free atom index {} out of range 1..={}",
                one_indexed, natoms
            )));
        }
        let i = (one_indexed - 1) as usize;
        xs[i] = xs_eff[slot];
        ys[i] = ys_eff[slot];
        zs[i] = zs_eff[slot];
    }
    // 4D w is not propagated through the seed (no test corpus exercises it).
    Ok((xs, ys, zs, w_eff))
}

/// Parse the box record at the current reader position into a `SimBox`.
fn parse_box_payload(payload: &[u8], byte_order: ByteOrder) -> std::io::Result<SimBox> {
    if payload.len() != 48 {
        return Err(err_mapper(format!(
            "box record has {} bytes, expected 48",
            payload.len()
        )));
    }
    let read_d = |off: usize| -> f64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&payload[off..off + 8]);
        read_f64(&b, byte_order)
    };

    // CHARMM order: A, gamma, B, beta, alpha, C
    let a = read_d(0);
    let raw_gamma = read_d(8);
    let b = read_d(16);
    let raw_beta = read_d(24);
    let raw_alpha = read_d(32);
    let c = read_d(40);

    // Cosine-vs-degree heuristic: NAMD/CHARMM ≥ 25 stores cosines, older
    // CHARMM stores degrees. If all three angle slots fit inside [-1, 1] we
    // treat them as cosines.
    let in_unit = |v: f64| (-1.0..=1.0).contains(&v);
    let (alpha, beta, gamma) = if in_unit(raw_alpha) && in_unit(raw_beta) && in_unit(raw_gamma) {
        (
            raw_alpha.acos().to_degrees(),
            raw_beta.acos().to_degrees(),
            raw_gamma.acos().to_degrees(),
        )
    } else {
        (raw_alpha, raw_beta, raw_gamma)
    };

    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return Err(err_mapper(format!(
            "non-positive box length: A={}, B={}, C={}",
            a, b, c
        )));
    }

    let pbc: Pbc3 = [true; 3];
    let origin = array![0.0 as F, 0.0, 0.0];

    if (alpha - 90.0).abs() < 1e-6 && (beta - 90.0).abs() < 1e-6 && (gamma - 90.0).abs() < 1e-6 {
        return SimBox::ortho(array![a as F, b as F, c as F], origin, pbc)
            .map_err(|e| err_mapper(format!("ortho box: {:?}", e)));
    }

    let h = abc_to_h(a, b, c, alpha, beta, gamma);
    SimBox::new(h, origin, pbc).map_err(|e| err_mapper(format!("triclinic box: {:?}", e)))
}

/// Build a 3×3 H matrix whose **columns** are the lattice vectors, using the
/// "a along x, b in xy plane" convention. Matches the column-major convention
/// used by `SimBox` and `lammps_data` / `poscar` readers.
fn abc_to_h(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Array2<F> {
    let to_rad = std::f64::consts::PI / 180.0;
    let ca = (alpha * to_rad).cos();
    let cb = (beta * to_rad).cos();
    let cg = (gamma * to_rad).cos();
    let sg = (gamma * to_rad).sin();

    let v1 = [a, 0.0, 0.0];
    let v2 = [b * cg, b * sg, 0.0];
    let v3x = c * cb;
    let v3y = if sg.abs() > 0.0 {
        c * (ca - cb * cg) / sg
    } else {
        0.0
    };
    let v3z2 = c * c - v3x * v3x - v3y * v3y;
    let v3z = if v3z2 > 0.0 { v3z2.sqrt() } else { 0.0 };
    let v3 = [v3x, v3y, v3z];

    // SimBox treats columns as lattice vectors: H[:, i] = v_{i+1}.
    array![
        [v1[0], v2[0], v3[0]],
        [v1[1], v2[1], v3[1]],
        [v1[2], v2[2], v3[2]],
    ]
}

/// Inverse of `abc_to_h`: extract `(a, b, c, alpha, beta, gamma)` from an
/// H matrix whose columns are lattice vectors.
fn h_to_abc(h: &Array2<F>) -> (f64, f64, f64, f64, f64, f64) {
    let col = |j: usize| [h[[0, j]], h[[1, j]], h[[2, j]]];
    let v1 = col(0);
    let v2 = col(1);
    let v3 = col(2);
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let a = norm(v1);
    let b = norm(v2);
    let c = norm(v3);
    let alpha = (dot(v2, v3) / (b * c)).clamp(-1.0, 1.0).acos().to_degrees();
    let beta = (dot(v1, v3) / (a * c)).clamp(-1.0, 1.0).acos().to_degrees();
    let gamma = (dot(v1, v2) / (a * b)).clamp(-1.0, 1.0).acos().to_degrees();
    (a, b, c, alpha, beta, gamma)
}

/// Read frame `n` at its computed offset. Reader must be Seek.
fn parse_frame_at<R: BufRead + Seek>(
    reader: &mut R,
    header: &DcdHeader,
    n: usize,
) -> std::io::Result<Option<Frame>> {
    if n >= header.nset as usize {
        return Ok(None);
    }
    reader.seek(SeekFrom::Start(header.frame_offset(n)))?;

    // Optional box record at the head of the frame.
    let simbox = if header.has_box {
        let payload = read_record(reader, header.byte_order, header.marker_size)?;
        Some(parse_box_payload(&payload, header.byte_order)?)
    } else {
        None
    };

    // Coordinate records — reuse `read_coord_payload` minus its leading box
    // skip by re-seeking past the box we just consumed.
    let coords_pos = reader.stream_position()?;
    // read_coord_payload always tries to read the box record first; rewind
    // and ask it to skip nothing by toggling the flag locally.
    let (xs, ys, zs, w) = {
        // Build a temporary header view with has_box=false so the helper
        // doesn't try to re-read a box record we already consumed.
        let mut hdr = header.clone();
        hdr.has_box = false;
        reader.seek(SeekFrom::Start(coords_pos))?;
        read_coord_payload(reader, &hdr, n)?
    };

    let natoms = header.natoms as usize;

    let mut atoms = Block::new();

    let id_arr = Array1::from_iter(1..=natoms as I)
        .into_shape_with_order(IxDyn(&[natoms]))
        .map_err(err_mapper)?;
    atoms.insert("id", id_arr).map_err(err_mapper)?;

    let to_f64 = |v: &[f32]| -> Array1<F> { v.iter().map(|&x| x as F).collect() };

    atoms
        .insert(
            "x",
            to_f64(&xs)
                .into_shape_with_order(IxDyn(&[natoms]))
                .map_err(err_mapper)?,
        )
        .map_err(err_mapper)?;
    atoms
        .insert(
            "y",
            to_f64(&ys)
                .into_shape_with_order(IxDyn(&[natoms]))
                .map_err(err_mapper)?,
        )
        .map_err(err_mapper)?;
    atoms
        .insert(
            "z",
            to_f64(&zs)
                .into_shape_with_order(IxDyn(&[natoms]))
                .map_err(err_mapper)?,
        )
        .map_err(err_mapper)?;
    if let Some(w) = w {
        atoms
            .insert(
                "w",
                to_f64(&w)
                    .into_shape_with_order(IxDyn(&[natoms]))
                    .map_err(err_mapper)?,
            )
            .map_err(err_mapper)?;
    }

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = simbox;

    let timestep = (header.istart as i64) + (n as i64) * (header.nsavc as i64);
    frame
        .meta
        .insert("timestep".to_string(), timestep.to_string());
    frame
        .meta
        .insert("delta".to_string(), header.delta.to_string());
    if !header.title.is_empty() {
        frame.meta.insert("title".to_string(), header.title.clone());
    }

    Ok(Some(frame))
}

// ============================================================================
// Reader
// ============================================================================

/// DCD trajectory reader implementing `TrajReader` for random access.
///
/// Header parsing is lazy — performed on the first call to a method that
/// needs it. This keeps `Reader::new` infallible.
pub struct DcdReader<R: BufRead + Seek> {
    reader: R,
    header: OnceCell<DcdHeader>,
    cursor: usize,
}

impl<R: BufRead + Seek> DcdReader<R> {
    /// Wrap `reader` in a `DcdReader`. Header parsing is deferred.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            header: OnceCell::new(),
            cursor: 0,
        }
    }

    /// Force header parsing and return a reference to the parsed header.
    pub fn header(&mut self) -> std::io::Result<&DcdHeader> {
        self.ensure_header()?;
        Ok(self.header.get().expect("header set"))
    }

    fn ensure_header(&mut self) -> std::io::Result<()> {
        if self.header.get().is_some() {
            return Ok(());
        }
        let header = parse_header(&mut self.reader)?;
        self.header
            .set(header)
            .map_err(|_| std::io::Error::other("failed to set header"))?;
        Ok(())
    }
}

impl<R: BufRead + Seek> Reader for DcdReader<R> {
    type R = R;
    type Frame = Frame;

    fn new(reader: Self::R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for DcdReader<R> {
    fn read_frame(&mut self) -> std::io::Result<Option<Self::Frame>> {
        self.ensure_header()?;
        let cursor = self.cursor;
        let header = self.header.get().expect("header set");
        if cursor >= header.nset as usize {
            return Ok(None);
        }
        let header = header.clone();
        let frame = parse_frame_at(&mut self.reader, &header, cursor)?;
        if frame.is_some() {
            self.cursor += 1;
        }
        Ok(frame)
    }
}

impl<R: BufRead + Seek> TrajReader for DcdReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        self.ensure_header()
    }

    fn read_step(&mut self, step: usize) -> std::io::Result<Option<Self::Frame>> {
        self.ensure_header()?;
        let header = self.header.get().expect("header set").clone();
        parse_frame_at(&mut self.reader, &header, step)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        self.ensure_header()?;
        Ok(self.header.get().expect("header set").nset as usize)
    }
}

// ============================================================================
// Writer
// ============================================================================

const WRITER_TITLE_DEFAULT: &str = "Created by molcrafts-molrs-io";

struct WriterMeta {
    natoms: u32,
    has_box: bool,
    nset_offset: u64,
    byte_order: ByteOrder,
    marker_size: MarkerSize,
}

/// DCD trajectory writer.
///
/// Writes NAMD-compatible files: little-endian, 4-byte Fortran markers,
/// CHARMM v24 header, cosine-encoded box angles. Does not support writing
/// 4D dynamics or fixed atoms — both produce `ErrorKind::Unsupported`.
///
/// Requires `Seek` so the NSET field can be patched after each frame.
pub struct DcdWriter<W: Write + Seek> {
    writer: W,
    meta: Option<WriterMeta>,
    nset: u32,
}

impl<W: Write + Seek> DcdWriter<W> {
    /// Create a new DCD writer. The header is emitted on the first
    /// `write_frame` call so it can match the frame's atom count and box
    /// presence.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            meta: None,
            nset: 0,
        }
    }
}

impl<W: Write + Seek> Writer for DcdWriter<W> {
    type W = W;
    type FrameLike = Frame;

    fn new(writer: Self::W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write + Seek> FrameWriter for DcdWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> std::io::Result<()> {
        write_dcd_frame(self, frame)
    }
}

fn write_dcd_frame<W: Write + Seek>(
    state: &mut DcdWriter<W>,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    // Reject 4D / fixed-atom frames eagerly.
    let has_w = frame
        .visit_block("atoms", |a| a.get_float_view("w").is_some())
        .unwrap_or(false);
    if has_w {
        return Err(unsupported(
            "DcdWriter does not support 4D (w-column) frames",
        ));
    }

    let natoms_in_frame = frame
        .visit_block("atoms", |a| a.nrows().unwrap_or(0))
        .ok_or_else(|| err_mapper("frame must contain 'atoms' block"))?
        as u32;
    if natoms_in_frame == 0 {
        return Err(err_mapper("frame has no atoms"));
    }

    let has_box = frame.simbox_ref().is_some();

    if state.meta.is_none() {
        let title = frame
            .meta_ref()
            .get("title")
            .map(String::as_str)
            .unwrap_or(WRITER_TITLE_DEFAULT);
        let istart: i32 = frame
            .meta_ref()
            .get("timestep")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let nset_offset =
            write_header_stub(&mut state.writer, natoms_in_frame, has_box, title, istart)?;
        state.meta = Some(WriterMeta {
            natoms: natoms_in_frame,
            has_box,
            nset_offset,
            byte_order: ByteOrder::Le,
            marker_size: MarkerSize::Four,
        });
    }
    let meta = state.meta.as_ref().expect("meta set");

    if natoms_in_frame != meta.natoms {
        return Err(err_mapper(format!(
            "DCD requires constant atom count: frame has {}, header has {}",
            natoms_in_frame, meta.natoms
        )));
    }
    if has_box != meta.has_box {
        return Err(err_mapper(
            "DCD requires consistent box presence across frames",
        ));
    }

    write_frame_payload(&mut state.writer, meta, frame)?;
    state.nset = state.nset.saturating_add(1);

    // Patch NSET in place.
    let end_pos = state.writer.stream_position()?;
    state.writer.seek(SeekFrom::Start(meta.nset_offset))?;
    let bytes = state.nset.to_le_bytes();
    state.writer.write_all(&bytes)?;
    state.writer.seek(SeekFrom::Start(end_pos))?;
    Ok(())
}

/// Write a stub header. Returns the byte offset of the NSET field so it can
/// be patched as frames are appended.
fn write_header_stub<W: Write + Seek>(
    writer: &mut W,
    natoms: u32,
    has_box: bool,
    title: &str,
    istart: i32,
) -> std::io::Result<u64> {
    let byte_order = ByteOrder::Le;
    let marker_size = MarkerSize::Four;

    // -- Header record 1 (84 bytes) --
    write_marker(writer, byte_order, marker_size, 84)?;
    writer.write_all(b"CORD")?;

    // Remember NSET position for later patching.
    let nset_offset = writer.stream_position()?;
    writer.write_all(&0u32.to_le_bytes())?; // NSET (patched later)
    writer.write_all(&istart.to_le_bytes())?;
    writer.write_all(&1i32.to_le_bytes())?; // NSAVC
    for _ in 0..5 {
        writer.write_all(&0i32.to_le_bytes())?;
    }
    writer.write_all(&0i32.to_le_bytes())?; // NAMNF
    writer.write_all(&(if has_box { 1i32 } else { 0i32 }).to_le_bytes())?;
    writer.write_all(&0i32.to_le_bytes())?; // 4D flag
    for _ in 0..5 {
        writer.write_all(&0i32.to_le_bytes())?;
    }
    writer.write_all(&1.0f32.to_le_bytes())?; // DELTA
    for _ in 0..2 {
        writer.write_all(&0i32.to_le_bytes())?;
    }
    writer.write_all(&24i32.to_le_bytes())?; // CHARMM_VER (NAMD-compatible)
    write_marker(writer, byte_order, marker_size, 84)?;

    // -- Header record 2 (title) --
    let title_lines: Vec<String> = if title.is_empty() {
        vec![WRITER_TITLE_DEFAULT.to_string()]
    } else {
        title.lines().map(|l| l.to_string()).collect()
    };
    let ntitle = title_lines.len() as i32;
    let payload_len = 4 + (ntitle as u64) * 80;
    write_marker(writer, byte_order, marker_size, payload_len)?;
    writer.write_all(&ntitle.to_le_bytes())?;
    for line in &title_lines {
        let mut padded = [b' '; 80];
        let bytes = line.as_bytes();
        let n = bytes.len().min(80);
        padded[..n].copy_from_slice(&bytes[..n]);
        writer.write_all(&padded)?;
    }
    write_marker(writer, byte_order, marker_size, payload_len)?;

    // -- Header record 3 (NATOMS) --
    write_marker(writer, byte_order, marker_size, 4)?;
    writer.write_all(&natoms.to_le_bytes())?;
    write_marker(writer, byte_order, marker_size, 4)?;

    Ok(nset_offset)
}

/// Write the box record + X/Y/Z payload for one frame.
fn write_frame_payload<W: Write>(
    writer: &mut W,
    meta: &WriterMeta,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    if meta.has_box {
        let simbox = frame
            .simbox_ref()
            .ok_or_else(|| err_mapper("frame missing simbox but header advertised one"))?;
        let h = simbox.h_view().to_owned();
        let (a, b, c, alpha, beta, gamma) = h_to_abc(&h);
        let cos_alpha = alpha.to_radians().cos();
        let cos_beta = beta.to_radians().cos();
        let cos_gamma = gamma.to_radians().cos();

        write_marker(writer, meta.byte_order, meta.marker_size, 48)?;
        writer.write_all(&a.to_le_bytes())?;
        writer.write_all(&cos_gamma.to_le_bytes())?;
        writer.write_all(&b.to_le_bytes())?;
        writer.write_all(&cos_beta.to_le_bytes())?;
        writer.write_all(&cos_alpha.to_le_bytes())?;
        writer.write_all(&c.to_le_bytes())?;
        write_marker(writer, meta.byte_order, meta.marker_size, 48)?;
    }

    let natoms = meta.natoms as usize;
    let coord_record_len = (natoms as u64) * 4;

    let write_axis = |writer: &mut W, vals: &[f64]| -> std::io::Result<()> {
        write_marker(writer, meta.byte_order, meta.marker_size, coord_record_len)?;
        for &v in vals.iter().take(natoms) {
            writer.write_all(&(v as f32).to_le_bytes())?;
        }
        write_marker(writer, meta.byte_order, meta.marker_size, coord_record_len)?;
        Ok(())
    };

    let extract_axis = |key: &str| -> std::io::Result<Vec<f64>> {
        frame
            .get_float("atoms", key)
            .map(|view| view.iter().copied().collect::<Vec<f64>>())
            .ok_or_else(|| err_mapper(format!("atoms.{} missing or not float", key)))
    };

    let xs = extract_axis("x")?;
    let ys = extract_axis("y")?;
    let zs = extract_axis("z")?;
    if xs.len() != natoms || ys.len() != natoms || zs.len() != natoms {
        return Err(err_mapper(format!(
            "atom count mismatch: x={}, y={}, z={}, expected {}",
            xs.len(),
            ys.len(),
            zs.len(),
            natoms
        )));
    }
    write_axis(writer, &xs)?;
    write_axis(writer, &ys)?;
    write_axis(writer, &zs)?;
    Ok(())
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Read every frame of a DCD file into memory.
///
/// For large trajectories prefer [`open_dcd`] with [`TrajReader::read_step`]
/// for random access without loading all frames at once.
pub fn read_dcd<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Frame>> {
    let reader = crate::reader::open_seekable(path)?;
    let mut dcd_reader = DcdReader::new(reader);
    dcd_reader.read_all()
}

/// Open a DCD file for trajectory-style random access.
pub fn open_dcd<P: AsRef<Path>>(path: P) -> std::io::Result<DcdReader<Box<dyn ReadSeek>>> {
    let reader = crate::reader::open_seekable(path)?;
    Ok(DcdReader::new(reader))
}

/// Write a slice of frames to a DCD file.
///
/// All frames must have the same atom count and the same box presence as the
/// first frame. 4D-dynamics frames (a `w` column on `atoms`) are rejected
/// with `ErrorKind::Unsupported`.
pub fn write_dcd<P: AsRef<Path>, FA: FrameAccess>(path: P, frames: &[FA]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    // BufWriter is Write + Seek when the inner is Seek — File is Seek.
    let mut dcd = DcdWriter::new(writer);
    for f in frames {
        write_dcd_frame(&mut dcd, f)?;
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_detect_endianness_and_marker_le4() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u32.to_le_bytes());
        buf.extend_from_slice(b"CORD");
        buf.extend_from_slice(&[0u8; 4]);
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Le, MarkerSize::Four)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_be4() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u32.to_be_bytes());
        buf.extend_from_slice(b"CORD");
        buf.extend_from_slice(&[0u8; 4]);
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Be, MarkerSize::Four)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_le8() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u64.to_le_bytes());
        buf.extend_from_slice(b"CORD");
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Le, MarkerSize::Eight)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_be8() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&84u64.to_be_bytes());
        buf.extend_from_slice(b"CORD");
        let mut cur = Cursor::new(buf);
        assert_eq!(
            detect_endianness_and_marker(&mut cur).unwrap(),
            (ByteOrder::Be, MarkerSize::Eight)
        );
    }

    #[test]
    fn test_detect_endianness_and_marker_garbage() {
        let mut cur = Cursor::new(vec![0u8; 12]);
        assert!(detect_endianness_and_marker(&mut cur).is_err());
    }

    #[test]
    fn test_box_cosine_vs_degree_heuristic_agree() {
        // Cubic box: A=B=C=10, alpha=beta=gamma=90°.
        let mut deg_payload = Vec::with_capacity(48);
        for v in [10.0f64, 90.0, 10.0, 90.0, 90.0, 10.0] {
            deg_payload.extend_from_slice(&v.to_le_bytes());
        }
        let mut cos_payload = Vec::with_capacity(48);
        for v in [10.0f64, 0.0, 10.0, 0.0, 0.0, 10.0] {
            cos_payload.extend_from_slice(&v.to_le_bytes());
        }
        let bx_deg = parse_box_payload(&deg_payload, ByteOrder::Le).unwrap();
        let bx_cos = parse_box_payload(&cos_payload, ByteOrder::Le).unwrap();
        let lengths_deg = bx_deg.lengths();
        let lengths_cos = bx_cos.lengths();
        for i in 0..3 {
            assert!((lengths_deg[i] - lengths_cos[i]).abs() < 1e-9);
            assert!((lengths_deg[i] - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_fortran_record_marker_mismatch() {
        let mut buf = vec![];
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&7u32.to_le_bytes()); // mismatch on purpose
        let mut cur = Cursor::new(buf);
        let err = read_record(&mut cur, ByteOrder::Le, MarkerSize::Four).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_abc_to_h_round_trip() {
        let h = abc_to_h(10.0, 12.0, 15.0, 70.0, 80.0, 95.0);
        let (a, b, c, al, be, ga) = h_to_abc(&h);
        assert!((a - 10.0).abs() < 1e-9);
        assert!((b - 12.0).abs() < 1e-9);
        assert!((c - 15.0).abs() < 1e-9);
        assert!((al - 70.0).abs() < 1e-7);
        assert!((be - 80.0).abs() < 1e-7);
        assert!((ga - 95.0).abs() < 1e-7);
    }
}
