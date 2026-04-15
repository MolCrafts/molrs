//! SDF / MDL molfile reader (V2000 CTAB).
//!
//! Parses the connection table (CTAB) inside MDL `.mol` files and the
//! record blocks of `.sdf` files. V2000 only — V3000 records are rejected.
//! Only the first record of a multi-record SDF is returned.
//!
//! Produces a [`Frame`] with:
//! - `"atoms"` block: `element` (string), `id` (u32, 1-based),
//!   `x`, `y`, `z` (F, angstrom)
//! - `"bonds"` block (if any): `atomi`, `atomj` (u32, 0-based indices
//!   into the atoms block), `order` (u32)

use crate::reader::{FrameReader, Reader};
use molrs::block::Block;
use molrs::frame::Frame;
use molrs::types::{F, U};
use ndarray::{Array1, IxDyn};
use std::io::BufRead;

fn err_mapper<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

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

/// Slice `[start..end)` clamped to `s.len()`.
fn substr(s: &str, start: usize, end: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    &s[start..end.min(len)]
}

/// Parsed V2000 counts line.
#[derive(Debug, Clone, Copy)]
struct Counts {
    atoms: usize,
    bonds: usize,
}

fn parse_counts_line(line: &str) -> std::io::Result<Counts> {
    if line.len() < 6 {
        return Err(err_mapper("counts line too short"));
    }
    let version = substr(line, 33, 39).trim();
    if version.eq_ignore_ascii_case("V3000") {
        return Err(err_mapper("V3000 SDF not supported"));
    }
    let atoms = substr(line, 0, 3)
        .trim()
        .parse::<usize>()
        .map_err(err_mapper)?;
    let bonds = substr(line, 3, 6)
        .trim()
        .parse::<usize>()
        .map_err(err_mapper)?;
    Ok(Counts { atoms, bonds })
}

#[derive(Debug, Clone)]
struct SdfAtom {
    element: String,
    x: F,
    y: F,
    z: F,
}

fn parse_atom_line(line: &str) -> std::io::Result<SdfAtom> {
    if line.len() < 34 {
        return Err(err_mapper("atom line too short"));
    }
    let x = substr(line, 0, 10).trim().parse::<F>().map_err(err_mapper)?;
    let y = substr(line, 10, 20).trim().parse::<F>().map_err(err_mapper)?;
    let z = substr(line, 20, 30).trim().parse::<F>().map_err(err_mapper)?;
    let element = substr(line, 31, 34).trim().to_string();
    Ok(SdfAtom { element, x, y, z })
}

#[derive(Debug, Clone, Copy)]
struct SdfBond {
    i: U, // 1-based in source
    j: U,
    order: U,
}

fn parse_bond_line(line: &str) -> std::io::Result<SdfBond> {
    if line.len() < 9 {
        return Err(err_mapper("bond line too short"));
    }
    let i = substr(line, 0, 3)
        .trim()
        .parse::<U>()
        .map_err(err_mapper)?;
    let j = substr(line, 3, 6)
        .trim()
        .parse::<U>()
        .map_err(err_mapper)?;
    let order = substr(line, 6, 9)
        .trim()
        .parse::<U>()
        .map_err(err_mapper)?;
    Ok(SdfBond { i, j, order })
}

fn build_frame(atoms: &[SdfAtom], bonds: &[SdfBond]) -> std::io::Result<Frame> {
    let n = atoms.len();
    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut z_vec = Vec::with_capacity(n);
    let mut id_vec: Vec<U> = Vec::with_capacity(n);
    let mut elements = Vec::with_capacity(n);

    for (i, a) in atoms.iter().enumerate() {
        x_vec.push(a.x);
        y_vec.push(a.y);
        z_vec.push(a.z);
        id_vec.push((i as U) + 1);
        elements.push(if a.element.is_empty() {
            "X".to_string()
        } else {
            a.element.clone()
        });
    }

    let mut atoms_block = Block::new();
    atoms_block
        .insert("x", to_array_float(x_vec, n)?)
        .map_err(err_mapper)?;
    atoms_block
        .insert("y", to_array_float(y_vec, n)?)
        .map_err(err_mapper)?;
    atoms_block
        .insert("z", to_array_float(z_vec, n)?)
        .map_err(err_mapper)?;
    atoms_block
        .insert("id", to_array_uint(id_vec, n)?)
        .map_err(err_mapper)?;
    let elements_arr = Array1::from_vec(elements)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(err_mapper)?
        .into_dyn();
    atoms_block
        .insert("element", elements_arr)
        .map_err(err_mapper)?;

    let mut frame = Frame::new();
    frame.insert("atoms", atoms_block);

    if !bonds.is_empty() {
        let bn = bonds.len();
        let mut i_vec: Vec<U> = Vec::with_capacity(bn);
        let mut j_vec: Vec<U> = Vec::with_capacity(bn);
        let mut order_vec: Vec<U> = Vec::with_capacity(bn);
        for b in bonds {
            if b.i == 0 || b.j == 0 || (b.i as usize) > n || (b.j as usize) > n {
                return Err(err_mapper(format!(
                    "bond references out-of-range atom: {}-{}",
                    b.i, b.j
                )));
            }
            // Convert 1-based to 0-based indices into the atoms block.
            i_vec.push(b.i - 1);
            j_vec.push(b.j - 1);
            order_vec.push(b.order);
        }
        let mut bonds_block = Block::new();
        bonds_block
            .insert("atomi", to_array_uint(i_vec, bn)?)
            .map_err(err_mapper)?;
        bonds_block
            .insert("atomj", to_array_uint(j_vec, bn)?)
            .map_err(err_mapper)?;
        bonds_block
            .insert("order", to_array_uint(order_vec, bn)?)
            .map_err(err_mapper)?;
        frame.insert("bonds", bonds_block);
    }

    Ok(frame)
}

/// V2000 SDF / MDL molfile reader.
///
/// Multi-record SDF files are supported; `read_frame` returns one record
/// per call and advances past the `$$$$` terminator.
pub struct SDFReader<R: BufRead> {
    reader: R,
}

impl<R: BufRead> SDFReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    fn read_single_record(&mut self) -> std::io::Result<Option<Frame>> {
        // Header (3 lines): title, program info, comment.
        let mut header = [String::new(), String::new(), String::new()];
        for slot in &mut header {
            if self.reader.read_line(slot)? == 0 {
                return Ok(None);
            }
        }

        // Counts line.
        let mut counts_line = String::new();
        if self.reader.read_line(&mut counts_line)? == 0 {
            return Err(err_mapper("missing counts line"));
        }
        let counts = parse_counts_line(&counts_line)?;
        if counts.atoms == 0 {
            return Err(err_mapper("SDF record has zero atoms"));
        }

        // Atom block.
        let mut atoms = Vec::with_capacity(counts.atoms);
        for _ in 0..counts.atoms {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                return Err(err_mapper("unexpected EOF in atom block"));
            }
            atoms.push(parse_atom_line(&line)?);
        }

        // Bond block.
        let mut bonds = Vec::with_capacity(counts.bonds);
        for _ in 0..counts.bonds {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                return Err(err_mapper("unexpected EOF in bond block"));
            }
            bonds.push(parse_bond_line(&line)?);
        }

        // Drain remaining lines of this record (properties, data items)
        // until `M  END` or `$$$$` (record terminator in multi-record SDF).
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim_end();
            if trimmed == "$$$$" {
                break;
            }
        }

        Ok(Some(build_frame(&atoms, &bonds)?))
    }
}

impl<R: BufRead> Reader for SDFReader<R> {
    type R = R;
    type Frame = Frame;

    fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R: BufRead> FrameReader for SDFReader<R> {
    fn read_frame(&mut self) -> std::io::Result<Option<Self::Frame>> {
        self.read_single_record()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::frame_access::FrameAccess;
    use std::io::Cursor;

    const WATER_SDF: &str = "962\n  -OEChem-\n\n  3  2  0     0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7572    0.0000   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.7572    0.0000   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0  0  0  0\n  1  3  1  0  0  0  0\nM  END\n$$$$\n";

    #[test]
    fn reads_water_record() {
        let mut reader = SDFReader::new(Cursor::new(WATER_SDF.as_bytes()));
        let frame = reader
            .read_frame()
            .expect("read ok")
            .expect("frame present");

        let x = frame.get_float("atoms", "x").expect("x column");
        assert_eq!(x.shape(), &[3]);
        assert!((x[[0]] - 0.0).abs() < 1e-6);
        assert!((x[[1]] - 0.7572).abs() < 1e-6);

        let atomi = frame.get_uint("bonds", "atomi").expect("atomi column");
        assert_eq!(atomi.shape(), &[2]);
        assert_eq!(atomi[[0]], 0); // 1-based -> 0-based
        assert_eq!(atomi[[1]], 0);
        let atomj = frame.get_uint("bonds", "atomj").expect("atomj column");
        assert_eq!(atomj[[0]], 1);
        assert_eq!(atomj[[1]], 2);
    }

    #[test]
    fn rejects_v3000() {
        let bad = "name\n\n\n  0  0  0  0  0  0  0  0  0  0999 V3000\n";
        let mut reader = SDFReader::new(Cursor::new(bad.as_bytes()));
        assert!(reader.read_frame().is_err());
    }
}
