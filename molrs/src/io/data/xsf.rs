//! XSF (XCrySDen Structure File) structure reader and writer.
//!
//! Supports crystal and molecular structures:
//!
//! - `CRYSTAL` / `MOLECULE` keywords
//! - `PRIMVEC` / `CONVVEC` — three lattice-vector lines (Å)
//! - `PRIMCOORD` — atom count / multiplicity, then `Z x y z` rows
//!
//! Returned [`Frame`]:
//!
//! - `"atoms"` block: `atomic_number` (U), `element` (str), `x`/`y`/`z` (F)
//! - `frame.simbox` — periodic cell from PRIMVEC (else CONVVEC) for `CRYSTAL`,
//!   or a free (no-cell) box for `MOLECULE` / unspecified

use std::io::{BufRead, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use ndarray::{Array1, Array2, IxDyn, array};

use molrs::Element;
use molrs::spatial::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, I, U};

// ---------------------------------------------------------------------------
// Error helpers
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

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read one XSF structure file from `path`.
pub fn read_xsf<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    read_xsf_from_reader(std::io::BufReader::new(file))
}

/// Read one XSF structure from a buffered reader.
pub fn read_xsf_from_reader<R: BufRead>(mut reader: R) -> Result<Frame> {
    let mut raw_lines = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let trimmed = buf.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        raw_lines.push(trimmed.to_string());
    }

    if raw_lines.is_empty() {
        return Err(invalid_data("Empty XSF file"));
    }

    let mut structure_type: Option<&str> = None;
    let mut primvec: Option<[[F; 3]; 3]> = None;
    let mut convvec: Option<[[F; 3]; 3]> = None;
    let mut atomic_numbers: Vec<U> = Vec::new();
    let mut xs: Vec<F> = Vec::new();
    let mut ys: Vec<F> = Vec::new();
    let mut zs: Vec<F> = Vec::new();
    let mut elements: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < raw_lines.len() {
        let line = raw_lines[i].as_str();
        let upper = line.to_ascii_uppercase();
        match upper.as_str() {
            "CRYSTAL" => {
                structure_type = Some("CRYSTAL");
                i += 1;
            }
            "MOLECULE" => {
                structure_type = Some("MOLECULE");
                i += 1;
            }
            "PRIMVEC" => {
                primvec = Some(parse_vectors(&raw_lines[i + 1..])?);
                i += 4;
            }
            "CONVVEC" => {
                convvec = Some(parse_vectors(&raw_lines[i + 1..])?);
                i += 4;
            }
            "PRIMCOORD" => {
                if i + 1 >= raw_lines.len() {
                    return Err(invalid_data("PRIMCOORD section incomplete"));
                }
                let header: Vec<&str> = raw_lines[i + 1].split_whitespace().collect();
                if header.len() < 2 {
                    return Err(invalid_data("Invalid PRIMCOORD header"));
                }
                let n_atoms: usize = header[0].parse().map_err(|_| {
                    invalid_data(format!("Invalid PRIMCOORD header: {}", raw_lines[i + 1]))
                })?;
                // Multiplicity (header[1]) is accepted and ignored.
                let start = i + 2;
                let end = start + n_atoms;
                if end > raw_lines.len() {
                    return Err(invalid_data("PRIMCOORD section incomplete"));
                }
                parse_atoms(
                    &raw_lines[start..end],
                    &mut atomic_numbers,
                    &mut xs,
                    &mut ys,
                    &mut zs,
                    &mut elements,
                )?;
                i = end;
            }
            _ => {
                i += 1;
            }
        }
    }

    let mut frame = Frame::new();

    if !atomic_numbers.is_empty() {
        let mut atoms = Block::new();
        insert_uint_col(&mut atoms, "atomic_number", atomic_numbers)?;
        insert_str_col(&mut atoms, "element", elements)?;
        insert_float_col(&mut atoms, "x", xs)?;
        insert_float_col(&mut atoms, "y", ys)?;
        insert_float_col(&mut atoms, "z", zs)?;
        frame.insert("atoms", atoms);
    }

    frame.simbox = Some(match structure_type {
        Some("CRYSTAL") => {
            let mat =
                primvec
                    .or(convvec)
                    .unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            // PRIMVEC lines are lattice vectors; H columns are lattice vectors.
            let h = Array2::from_shape_fn((3, 3), |(r, c)| mat[c][r]);
            let origin = array![0.0 as F, 0.0, 0.0];
            SimBox::new(h, origin, [true, true, true])
                .map_err(|e| invalid_data(format!("{:?}", e)))?
        }
        _ => free_simbox()?,
    });

    Ok(frame)
}

fn free_simbox() -> Result<SimBox> {
    let h = array![[1.0 as F, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let origin = array![0.0 as F, 0.0, 0.0];
    SimBox::new_cell(h, origin, [false, false, false], false)
        .map_err(|e| invalid_data(format!("{:?}", e)))
}

/// Parse three lattice-vector lines (row-per-vector in the file).
fn parse_vectors(lines: &[String]) -> Result<[[F; 3]; 3]> {
    if lines.len() < 3 {
        return Err(invalid_data("Incomplete vector specification"));
    }
    let mut matrix = [[0.0 as F; 3]; 3];
    for (i, line) in lines.iter().take(3).enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(invalid_data(format!("Invalid vector line: {line}")));
        }
        for j in 0..3 {
            matrix[i][j] = parts[j]
                .parse::<F>()
                .map_err(|_| invalid_data(format!("Invalid vector coordinates: {line}")))?;
        }
    }
    Ok(matrix)
}

fn parse_atoms(
    lines: &[String],
    atomic_numbers: &mut Vec<U>,
    xs: &mut Vec<F>,
    ys: &mut Vec<F>,
    zs: &mut Vec<F>,
    elements: &mut Vec<String>,
) -> Result<()> {
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(invalid_data(format!("Invalid atom line: {line}")));
        }
        let z: I = parts[0]
            .parse()
            .map_err(|_| invalid_data(format!("Invalid atom data: {line}")))?;
        if z < 0 {
            return Err(invalid_data(format!("Invalid atom data: {line}")));
        }
        let z_u = z as U;
        let x: F = parts[1]
            .parse()
            .map_err(|_| invalid_data(format!("Invalid atom data: {line}")))?;
        let y: F = parts[2]
            .parse()
            .map_err(|_| invalid_data(format!("Invalid atom data: {line}")))?;
        let zc: F = parts[3]
            .parse()
            .map_err(|_| invalid_data(format!("Invalid atom data: {line}")))?;

        let symbol = if z_u > 0 && z_u <= 118 {
            Element::by_number(z_u as u8)
                .map(|e| e.symbol().to_string())
                .unwrap_or_else(|| "X".to_string())
        } else {
            "X".to_string()
        };

        atomic_numbers.push(z_u);
        xs.push(x);
        ys.push(y);
        zs.push(zc);
        elements.push(symbol);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Frame to an XSF file at `path`.
pub fn write_xsf<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    write_xsf_frame(&mut w, frame)?;
    w.flush()
}

/// Emit one Frame in XSF format.
pub fn write_xsf_frame<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    writeln!(writer, "# XSF file generated by molpy")?;

    let has_crystal = frame
        .simbox
        .as_ref()
        .is_some_and(|sb| sb.is_cell_defined() && !sb.is_free());

    if has_crystal {
        let sb = frame.simbox.as_ref().unwrap();
        let h = sb.h_view();
        writeln!(writer, "CRYSTAL")?;
        writeln!(writer, "PRIMVEC")?;
        for i in 0..3 {
            // Lattice vector i = column i of H → written as one PRIMVEC row.
            writeln!(
                writer,
                "    {:12.8}    {:12.8}    {:12.8}",
                h[[0, i]],
                h[[1, i]],
                h[[2, i]]
            )?;
        }
        writeln!(writer, "CONVVEC")?;
        for i in 0..3 {
            writeln!(
                writer,
                "    {:12.8}    {:12.8}    {:12.8}",
                h[[0, i]],
                h[[1, i]],
                h[[2, i]]
            )?;
        }
    } else {
        writeln!(writer, "MOLECULE")?;
    }

    if let Some(atoms) = frame.get("atoms") {
        let n = atoms.nrows().unwrap_or(0);
        let xs = atoms
            .get_float("x")
            .ok_or_else(|| invalid_data("XSF write: atoms.x missing"))?;
        let ys = atoms
            .get_float("y")
            .ok_or_else(|| invalid_data("XSF write: atoms.y missing"))?;
        let zs = atoms
            .get_float("z")
            .ok_or_else(|| invalid_data("XSF write: atoms.z missing"))?;
        let z_u = atoms.get_uint("atomic_number");
        let z_i = atoms.get_int("atomic_number");

        writeln!(writer, "PRIMCOORD")?;
        writeln!(writer, "       {n} 1")?;
        for idx in 0..n {
            let an = if let Some(col) = z_u {
                col[[idx]] as i32
            } else if let Some(col) = z_i {
                col[[idx]]
            } else {
                return Err(invalid_data("XSF write: atoms.atomic_number missing"));
            };
            writeln!(
                writer,
                "{:2}    {:12.8}    {:12.8}    {:12.8}",
                an,
                xs[[idx]],
                ys[[idx]],
                zs[[idx]]
            )?;
        }
    } else {
        writeln!(writer, "PRIMCOORD")?;
        writeln!(writer, "        0 1")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_crystal_orthogonal() {
        let text = "\
CRYSTAL
PRIMVEC
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
PRIMCOORD
2 1
1  0.0  0.0  0.0
8  1.5  1.5  1.5
";
        let frame = read_xsf_from_reader(Cursor::new(text)).expect("parse");
        let atoms = frame.get("atoms").expect("atoms");
        assert_eq!(atoms.nrows(), Some(2));
        let z = atoms.get_uint("atomic_number").unwrap();
        assert_eq!(z[[0]], 1);
        assert_eq!(z[[1]], 8);
        let sb = frame.simbox.as_ref().unwrap();
        assert!(sb.is_cell_defined());
        assert!(!sb.is_free());
        assert_eq!(sb.style(), "orthogonal");
    }

    #[test]
    fn read_molecule_free() {
        let text = "\
MOLECULE
PRIMCOORD
2 1
1  0.0  0.0  0.0
1  1.0  0.0  0.0
";
        let frame = read_xsf_from_reader(Cursor::new(text)).expect("parse");
        let atoms = frame.get("atoms").expect("atoms");
        assert_eq!(atoms.nrows(), Some(2));
        let sb = frame.simbox.as_ref().unwrap();
        assert!(!sb.is_cell_defined());
        assert!(sb.is_free());
        assert_eq!(sb.style(), "free");
    }

    #[test]
    fn empty_file_errors() {
        let err = read_xsf_from_reader(Cursor::new("")).unwrap_err();
        assert!(err.to_string().contains("Empty XSF file"));
    }

    #[test]
    fn roundtrip_crystal() {
        let text = "\
CRYSTAL
PRIMVEC
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
PRIMCOORD
2 1
6  0.0  0.0  0.0
1  1.0  0.0  0.0
";
        let frame = read_xsf_from_reader(Cursor::new(text)).unwrap();
        let mut out = Vec::new();
        write_xsf_frame(&mut out, &frame).unwrap();
        let back = read_xsf_from_reader(Cursor::new(out)).unwrap();
        let z0 = frame
            .get("atoms")
            .unwrap()
            .get_uint("atomic_number")
            .unwrap();
        let z1 = back
            .get("atoms")
            .unwrap()
            .get_uint("atomic_number")
            .unwrap();
        assert_eq!(z0[[0]], z1[[0]]);
        assert_eq!(z0[[1]], z1[[1]]);
    }

    #[test]
    fn malformed_primcoord_header() {
        let text = "MOLECULE\nPRIMCOORD\ninvalid_number 1\n";
        let err = read_xsf_from_reader(Cursor::new(text)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }
}
