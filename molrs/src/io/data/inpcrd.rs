//! AMBER ASCII inpcrd / restrt coordinate reader.
//!
//! Fixed-width Fortran ``6F12.7`` layout (old-style ``*.inpcrd`` / restart):
//!
//! ```text
//! line 1   : title
//! line 2   : natom [time]
//! next     : coordinates — 6 floats per line, 12 chars each
//! optional : velocities (same layout; only when `time` is present on line 2)
//! optional : box line — 3–6 floats (first three → orthorhombic diagonal)
//! ```
//!
//! ## Output Frame
//!
//! - `"atoms"` block: `id` (uint, 1-based), `name` (str, `"ATM{i}"`),
//!   `x`/`y`/`z` (F), optional `vel` (F, shape `[n, 3]`)
//! - `frame.meta["title"]` (string); optional `frame.meta["timestep"]` (i64,
//!   truncates the header time)
//! - `frame.simbox`: orthorhombic cell from the first three box floats when present

use std::io::{BufRead, Error, ErrorKind, Result};
use std::path::Path;

use ndarray::{Array1, Array2, IxDyn, array};

use molrs::spatial::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, Idx};

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

/// Parse a line with fixed-width columns (default width 12).
///
/// AMBER writes Fortran ``6F12.7``; a negative sign may abut the previous field
/// (e.g. ``50.5413286-100.7101036``), so whitespace splitting is wrong.
fn parse_fixed_width(raw: &str, width: usize) -> Result<Vec<F>> {
    let mut out = Vec::new();
    let mut i = 0;
    let bytes = raw.as_bytes();
    while i < bytes.len() {
        let end = (i + width).min(bytes.len());
        let token = std::str::from_utf8(&bytes[i..end])
            .map_err(|_| invalid_data("non-utf8 in fixed-width field"))?
            .trim();
        if !token.is_empty() {
            let v: F = token
                .parse()
                .map_err(|_| invalid_data(format!("bad float in fixed-width field: {token:?}")))?;
            out.push(v);
        }
        i += width;
    }
    Ok(out)
}

/// Parse floats from one coordinate/velocity/box line.
///
/// Prefer fixed-width 12-char fields; fall back to whitespace split if that fails.
fn parse_floats_line(raw: &str) -> Result<Vec<F>> {
    match parse_fixed_width(raw, 12) {
        Ok(v) if !v.is_empty() || raw.trim().is_empty() => Ok(v),
        Ok(_) | Err(_) => {
            let mut out = Vec::new();
            for tok in raw.split_whitespace() {
                let v: F = tok
                    .parse()
                    .map_err(|_| invalid_data(format!("bad float token: {tok:?}")))?;
                out.push(v);
            }
            Ok(out)
        }
    }
}

/// Collect floats from a slice of lines into an `[n_atoms × 3]` flat vector.
fn eat_section(lines: &[String], n_atoms: usize) -> Result<Vec<F>> {
    let need = n_atoms * 3;
    let mut data = Vec::with_capacity(need);
    for line in lines {
        let raw = line.trim_end_matches(['\n', '\r']);
        if raw.trim().is_empty() {
            continue;
        }
        data.extend(parse_floats_line(raw)?);
    }
    if data.len() < need {
        return Err(invalid_data(format!(
            "expected {need} floats for {n_atoms} atoms, got {}",
            data.len()
        )));
    }
    data.truncate(need);
    Ok(data)
}

// ---------------------------------------------------------------------------
// Public reader API
// ---------------------------------------------------------------------------

/// Read an AMBER ASCII inpcrd / restrt file at `path`.
pub fn read_amber_inpcrd<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    read_amber_inpcrd_from_reader(std::io::BufReader::new(file))
}

/// Alias for [`read_amber_inpcrd`].
pub fn read_inpcrd<P: AsRef<Path>>(path: P) -> Result<Frame> {
    read_amber_inpcrd(path)
}

/// Read an AMBER ASCII inpcrd / restrt from any [`BufRead`].
pub fn read_amber_inpcrd_from_reader<R: BufRead>(mut reader: R) -> Result<Frame> {
    let mut raw_lines: Vec<String> = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        // Keep original line content (minus only the trailing newline) so
        // fixed-width columns that include leading spaces stay intact.
        let line = buf.trim_end_matches(['\n', '\r']).to_string();
        raw_lines.push(line);
    }

    if raw_lines.len() < 2 {
        return Err(invalid_data("inpcrd too short"));
    }

    let title = raw_lines[0].trim().to_string();
    let header_tokens: Vec<&str> = raw_lines[1].split_whitespace().collect();
    if header_tokens.is_empty() {
        return Err(invalid_data("inpcrd header missing atom count"));
    }
    let n_atoms: usize = header_tokens[0]
        .parse()
        .map_err(|_| invalid_data(format!("bad atom count: {:?}", header_tokens[0])))?;
    let time: Option<F> = if header_tokens.len() > 1 {
        Some(
            header_tokens[1]
                .parse()
                .map_err(|_| invalid_data(format!("bad time token: {:?}", header_tokens[1])))?,
        )
    } else {
        None
    };

    // AMBER: 6 values per line → ceil(n_atoms * 3 / 6) coordinate lines.
    let n_coord_lines = n_atoms.saturating_mul(3).saturating_add(5) / 6;
    if raw_lines.len() < 2 + n_coord_lines {
        return Err(invalid_data(format!(
            "Not enough lines for {n_atoms} atoms: need {n_coord_lines} coordinate lines, got {}",
            raw_lines.len().saturating_sub(2)
        )));
    }

    let mut cursor = 2 + n_coord_lines;
    let coords_flat = eat_section(&raw_lines[2..cursor], n_atoms)?;

    // Velocities only appear in restart files (timestamp on the header line).
    let mut velocity_flat: Option<Vec<F>> = None;
    if time.is_some() {
        let non_blank_remaining = raw_lines[cursor..]
            .iter()
            .filter(|l| !l.trim().is_empty())
            .count();
        if non_blank_remaining >= n_coord_lines {
            let maybe = eat_section(&raw_lines[cursor..cursor + n_coord_lines], n_atoms)?;
            if maybe.len() == n_atoms * 3 {
                velocity_flat = Some(maybe);
                cursor += n_coord_lines;
            }
        }
    }

    // Optional box line (3–6 floats; first three → diagonal lengths).
    let mut box_lengths: Option<[F; 3]> = None;
    if cursor < raw_lines.len() {
        let raw = raw_lines[cursor].as_str();
        if !raw.trim().is_empty() {
            let box_floats = parse_floats_line(raw)?;
            if box_floats.len() >= 3 {
                box_lengths = Some([box_floats[0], box_floats[1], box_floats[2]]);
            }
        }
    }

    // ---- Build Frame -------------------------------------------------------
    let mut block = Block::new();
    let mut ids = Vec::with_capacity(n_atoms);
    let mut names = Vec::with_capacity(n_atoms);
    let mut xs = Vec::with_capacity(n_atoms);
    let mut ys = Vec::with_capacity(n_atoms);
    let mut zs = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        ids.push((i as Idx) + 1);
        names.push(format!("ATM{}", i + 1));
        xs.push(coords_flat[i * 3]);
        ys.push(coords_flat[i * 3 + 1]);
        zs.push(coords_flat[i * 3 + 2]);
    }
    insert_uint_col(&mut block, "id", ids)?;
    insert_str_col(&mut block, "name", names)?;
    insert_float_col(&mut block, "x", xs)?;
    insert_float_col(&mut block, "y", ys)?;
    insert_float_col(&mut block, "z", zs)?;

    if let Some(vel) = velocity_flat {
        let arr = Array2::from_shape_vec((n_atoms, 3), vel)
            .map_err(invalid_data)?
            .into_dyn();
        block.insert("vel", arr).map_err(invalid_data)?;
    }

    let mut frame = Frame::new();
    frame.meta.insert("title", title);
    if let Some(t) = time {
        frame.meta.insert("timestep", t as i64);
    }
    frame.insert("atoms", block);

    if let Some(lens) = box_lengths {
        let origin = array![0.0 as F, 0.0, 0.0];
        let simbox = SimBox::ortho(array![lens[0], lens[1], lens[2]], origin, [true; 3])
            .map_err(|e| invalid_data(format!("{:?}", e)))?;
        frame.simbox = Some(simbox);
    }

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn frame_from(s: &str) -> Frame {
        read_amber_inpcrd_from_reader(Cursor::new(s.as_bytes())).expect("parse")
    }

    #[test]
    fn basic_coords_only() {
        let text = "\
Simple 3-atom system
  3
  0.0000000   1.0000000   2.0000000   3.0000000   4.0000000   5.0000000
  6.0000000   7.0000000   8.0000000
";
        let frame = frame_from(text);
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(3));
        assert_eq!(
            frame.meta.get("title").and_then(|v| v.as_str()),
            Some("Simple 3-atom system")
        );
        let x = atoms.get_float("x").unwrap();
        let y = atoms.get_float("y").unwrap();
        let z = atoms.get_float("z").unwrap();
        assert!((x[[0]] - 0.0).abs() < 1e-9);
        assert!((y[[1]] - 4.0).abs() < 1e-9);
        assert!((z[[2]] - 8.0).abs() < 1e-9);
        assert!(frame.simbox.is_none());
    }

    #[test]
    fn with_time() {
        let text = "\
Test with time
  2   100.5
  1.0         2.0         3.0         4.0         5.0         6.0
";
        let frame = frame_from(text);
        assert_eq!(
            frame.meta.get("timestep").and_then(|v| v.as_i64()),
            Some(100)
        );
    }

    #[test]
    fn with_velocities() {
        let text = "\
Test with velocities
  2   25.0
  1.0         2.0         3.0         4.0         5.0         6.0
  0.1         0.2         0.3         0.4         0.5         0.6
";
        let frame = frame_from(text);
        let atoms = frame.get("atoms").unwrap();
        let vel = atoms.get_float("vel").expect("vel column");
        assert_eq!(vel.shape(), &[2, 3]);
        assert!((vel[[0, 0]] - 0.1).abs() < 1e-9);
        assert!((vel[[1, 2]] - 0.6).abs() < 1e-9);
    }

    #[test]
    fn with_box() {
        let text = "\
Test with box
  2
  1.0         2.0         3.0         4.0         5.0         6.0
 10.0        20.0        30.0        90.0        90.0        90.0
";
        let frame = frame_from(text);
        let sb = frame.simbox.as_ref().expect("box");
        let lens = sb.lengths();
        assert!((lens[0] - 10.0).abs() < 1e-9);
        assert!((lens[1] - 20.0).abs() < 1e-9);
        assert!((lens[2] - 30.0).abs() < 1e-9);
    }

    #[test]
    fn abutting_negatives() {
        // Fortran 6F12.7 can place a minus flush against the previous field.
        let abutting = "  50.5413286-100.7101036  12.3456789 -44.5678901  88.8888888  -0.1234567";
        let text = format!("Abutting negatives\n  2\n{abutting}\n");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        let x = atoms.get_float("x").unwrap();
        let y = atoms.get_float("y").unwrap();
        let z = atoms.get_float("z").unwrap();
        assert!((x[[0]] - 50.5413286).abs() < 1e-6);
        assert!((y[[0]] + 100.7101036).abs() < 1e-6);
        assert!((z[[0]] - 12.3456789).abs() < 1e-6);
        assert!((x[[1]] + 44.5678901).abs() < 1e-6);
    }

    #[test]
    fn too_short() {
        let err = read_amber_inpcrd_from_reader(Cursor::new(b"Only title\n")).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn insufficient_coords() {
        let text = "\
Insufficient data
  5
  1.0         2.0         3.0         4.0         5.0         6.0
";
        let err = read_amber_inpcrd_from_reader(Cursor::new(text.as_bytes())).unwrap_err();
        assert!(err.to_string().contains("Not enough lines"));
    }
}
