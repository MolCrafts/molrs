//! Gaussian Cube file reader and writer.
//!
//! ## File layout
//!
//! ```text
//! <comment line 1>                    ← free text → frame.meta["comment1"]
//! <comment line 2>                    ← free text → frame.meta["comment2"]
//! NATOMS  origin_x  origin_y  origin_z
//!   N1    vx1_x  vx1_y  vx1_z         ← voxel step vector for axis 1
//!   N2    vx2_x  vx2_y  vx2_z         ← voxel step vector for axis 2
//!   N3    vx3_x  vx3_y  vx3_z         ← voxel step vector for axis 3
//!   Z  charge  x  y  z                ← one atom per line
//!   …
//! [MO only: NVALS  idx1  idx2  …]     ← present if NATOMS < 0
//! <val> <val> …                        ← volumetric data, 6 per line
//! ```
//!
//! ## Sign conventions
//!
//! - **NATOMS**: negative → MO mode (extra header line after atoms listing
//!   orbital count and indices).
//! - **N1** (first voxel count): positive → file units are Bohr; negative →
//!   Angstrom. The absolute value is the actual grid count.
//!
//! ## Data layout
//!
//! Volumetric data is stored with x as the outermost loop and z as the
//! innermost (row-major). This matches the internal [`Grid`] convention, so
//! **no reordering** is needed (unlike CHGCAR).
//!
//! ## Unit handling
//!
//! molrs is **unit-agnostic**: values are stored as-is from the file. The
//! detected unit system is recorded in `frame.meta["cube_units"]` as either
//! `"bohr"` or `"angstrom"`.
//!
//! ## Frame contents
//!
//! | Block / Grid | Key(s)            | Content                        |
//! |--------------|-------------------|--------------------------------|
//! | `"atoms"`    | symbol, x, y, z, atomic_number, charge | Atom data |
//! | grid `"cube"`| `"density"` or `"mo_<idx>"` | Volumetric scalar fields |

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use ndarray::{Array1, ArrayD, IxDyn};

use crate::block::Block;
use crate::element::Element;
use crate::error::MolRsError;
use crate::frame::Frame;
use crate::grid::Grid;
use crate::types::{F, I};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read a Gaussian Cube file from disk.
pub fn read_cube<P: AsRef<Path>>(path: P) -> Result<Frame, MolRsError> {
    let file = std::fs::File::open(path.as_ref()).map_err(MolRsError::Io)?;
    read_cube_from_reader(BufReader::new(file))
}

/// Read a Gaussian Cube file from any [`BufRead`] source.
pub fn read_cube_from_reader<R: BufRead>(mut reader: R) -> Result<Frame, MolRsError> {
    let mut line_no = 0usize;

    macro_rules! next_line {
        () => {{
            let mut s = String::new();
            reader.read_line(&mut s).map_err(MolRsError::Io)?;
            line_no += 1;
            s
        }};
    }

    macro_rules! parse_err {
        ($msg:expr) => {
            MolRsError::parse_error(line_no, $msg)
        };
    }

    // -----------------------------------------------------------------------
    // Comment lines
    // -----------------------------------------------------------------------
    let comment1 = next_line!().trim_end().to_string();
    let comment2 = next_line!().trim_end().to_string();

    // -----------------------------------------------------------------------
    // Line 3: NATOMS  origin_x  origin_y  origin_z
    // -----------------------------------------------------------------------
    let line3 = next_line!();
    let toks3: Vec<&str> = line3.split_whitespace().collect();
    if toks3.len() < 4 {
        return Err(parse_err!("expected NATOMS origin_x origin_y origin_z"));
    }
    let natoms_signed: i32 = toks3[0]
        .parse()
        .map_err(|_| parse_err!("bad NATOMS integer"))?;
    let has_mo = natoms_signed < 0;
    let n_atoms = natoms_signed.unsigned_abs() as usize;
    let origin = [
        parse_f64(toks3[1], line_no)?,
        parse_f64(toks3[2], line_no)?,
        parse_f64(toks3[3], line_no)?,
    ];

    // -----------------------------------------------------------------------
    // Lines 4-6: voxel axes  N_i  vx vy vz
    // -----------------------------------------------------------------------
    let mut dims = [0usize; 3];
    let mut voxel_vecs = [[0.0f64; 3]; 3];
    let mut is_angstrom = false;

    for i in 0..3 {
        let line = next_line!();
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(parse_err!("expected N vx vy vz for voxel axis"));
        }
        let n_signed: i32 = toks[0].parse().map_err(|_| parse_err!("bad voxel count"))?;
        if i == 0 && n_signed < 0 {
            is_angstrom = true;
        }
        dims[i] = n_signed.unsigned_abs() as usize;
        voxel_vecs[i][0] = parse_f64(toks[1], line_no)?;
        voxel_vecs[i][1] = parse_f64(toks[2], line_no)?;
        voxel_vecs[i][2] = parse_f64(toks[3], line_no)?;
    }

    // -----------------------------------------------------------------------
    // Atom lines: Z  charge  x  y  z
    // -----------------------------------------------------------------------
    let mut atomic_numbers = Vec::with_capacity(n_atoms);
    let mut charges = Vec::with_capacity(n_atoms);
    let mut xs = Vec::with_capacity(n_atoms);
    let mut ys = Vec::with_capacity(n_atoms);
    let mut zs = Vec::with_capacity(n_atoms);
    let mut symbols = Vec::with_capacity(n_atoms);

    for _ in 0..n_atoms {
        let line = next_line!();
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 5 {
            return Err(parse_err!("expected Z charge x y z for atom"));
        }
        let z: i32 = toks[0]
            .parse()
            .map_err(|_| parse_err!("bad atomic number"))?;
        atomic_numbers.push(z);
        charges.push(parse_f64(toks[1], line_no)?);
        xs.push(parse_f64(toks[2], line_no)?);
        ys.push(parse_f64(toks[3], line_no)?);
        zs.push(parse_f64(toks[4], line_no)?);

        let sym = if z > 0 && z <= 118 {
            Element::by_number(z as u8)
                .map(|e| e.symbol().to_string())
                .unwrap_or_else(|| "X".to_string())
        } else {
            "X".to_string()
        };
        symbols.push(sym);
    }

    // -----------------------------------------------------------------------
    // Optional MO header: NVALS  idx1  idx2  …
    // -----------------------------------------------------------------------
    let mut mo_indices: Vec<usize> = Vec::new();

    let n_vals_per_point: usize = if has_mo {
        let line = next_line!();
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            return Err(parse_err!("expected MO header (NVALS idx1 idx2 …)"));
        }
        let nvals: usize = toks[0]
            .parse()
            .map_err(|_| parse_err!("bad NVALS in MO header"))?;
        for &t in &toks[1..] {
            let idx: usize = t.parse().map_err(|_| parse_err!("bad MO index"))?;
            mo_indices.push(idx);
        }
        if mo_indices.len() != nvals {
            return Err(parse_err!(format!(
                "NVALS={} but {} indices given",
                nvals,
                mo_indices.len()
            )));
        }
        nvals
    } else {
        1
    };

    // -----------------------------------------------------------------------
    // Volumetric data
    // -----------------------------------------------------------------------
    let n_voxels = dims[0] * dims[1] * dims[2];
    let total_values = n_voxels * n_vals_per_point;
    let flat_data = read_volumetric_data(&mut reader, total_values, &mut line_no)?;

    // -----------------------------------------------------------------------
    // Build Grid
    // -----------------------------------------------------------------------
    // cell[i] = voxel_vec[i] * N_i  (total span per axis)
    let cell: [[F; 3]; 3] = [
        [
            voxel_vecs[0][0] * dims[0] as f64,
            voxel_vecs[0][1] * dims[0] as f64,
            voxel_vecs[0][2] * dims[0] as f64,
        ],
        [
            voxel_vecs[1][0] * dims[1] as f64,
            voxel_vecs[1][1] * dims[1] as f64,
            voxel_vecs[1][2] * dims[1] as f64,
        ],
        [
            voxel_vecs[2][0] * dims[2] as f64,
            voxel_vecs[2][1] * dims[2] as f64,
            voxel_vecs[2][2] * dims[2] as f64,
        ],
    ];
    let grid_origin: [F; 3] = [origin[0], origin[1], origin[2]];

    let mut grid = Grid::new(dims, grid_origin, cell, [false; 3]);

    if has_mo {
        // Deinterleave: data is [v0_pt0, v1_pt0, v2_pt0, v0_pt1, v1_pt1, …]
        for (k, &idx) in mo_indices.iter().enumerate() {
            let arr: Vec<F> = (0..n_voxels)
                .map(|i| flat_data[i * n_vals_per_point + k])
                .collect();
            grid.insert(format!("mo_{}", idx), arr)
                .map_err(|e| MolRsError::parse(format!("grid insert error: {}", e)))?;
        }
    } else {
        grid.insert("density", flat_data)
            .map_err(|e| MolRsError::parse(format!("grid insert error: {}", e)))?;
    }

    // -----------------------------------------------------------------------
    // Build atoms Block
    // -----------------------------------------------------------------------
    let mut atoms = Block::new();
    atoms
        .insert(
            "x",
            Array1::from_vec(xs.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "y",
            Array1::from_vec(ys.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "z",
            Array1::from_vec(zs.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "atomic_number",
            Array1::from_vec(atomic_numbers.iter().map(|&v| v as I).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "charge",
            Array1::from_vec(charges.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "symbol",
            ArrayD::from_shape_vec(IxDyn(&[n_atoms]), symbols)
                .expect("shape matches")
                .into_dyn(),
        )
        .map_err(MolRsError::Block)?;

    // -----------------------------------------------------------------------
    // Assemble Frame
    // -----------------------------------------------------------------------
    let mut frame = Frame::new();
    if !comment1.is_empty() {
        frame.meta.insert("comment1".into(), comment1);
    }
    if !comment2.is_empty() {
        frame.meta.insert("comment2".into(), comment2);
    }
    frame.meta.insert(
        "cube_units".into(),
        if is_angstrom {
            "angstrom".into()
        } else {
            "bohr".into()
        },
    );
    if has_mo {
        let indices_str = mo_indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        frame.meta.insert("cube_mo_indices".into(), indices_str);
    }

    frame.insert("atoms", atoms);
    frame.insert_grid("cube", grid);

    Ok(frame)
}

/// Write a Gaussian Cube file to disk.
pub fn write_cube<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<(), MolRsError> {
    let file = std::fs::File::create(path.as_ref()).map_err(MolRsError::Io)?;
    let mut writer = std::io::BufWriter::new(file);
    write_cube_to_writer(&mut writer, frame)
}

/// Write a Gaussian Cube file to any [`Write`] destination.
pub fn write_cube_to_writer<W: Write>(writer: &mut W, frame: &Frame) -> Result<(), MolRsError> {
    let grid = frame
        .get_grid("cube")
        .ok_or_else(|| MolRsError::validation("frame has no 'cube' grid"))?;
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| MolRsError::validation("frame has no 'atoms' block"))?;

    let n_atoms = atoms.nrows().unwrap_or(0);
    let [nx, ny, nz] = grid.dim;

    // Determine MO mode from metadata
    let mo_indices: Option<Vec<usize>> = frame.meta.get("cube_mo_indices").map(|s| {
        s.split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect()
    });
    let has_mo = mo_indices.is_some();

    // Determine unit sign convention
    let is_angstrom = frame
        .meta
        .get("cube_units")
        .is_some_and(|u| u == "angstrom");

    // Comment lines
    let c1 = frame.meta.get("comment1").cloned().unwrap_or_default();
    let c2 = frame.meta.get("comment2").cloned().unwrap_or_default();
    writeln!(writer, "{}", c1).map_err(MolRsError::Io)?;
    writeln!(writer, "{}", c2).map_err(MolRsError::Io)?;

    // NATOMS line
    let natoms_signed: i32 = if has_mo {
        -(n_atoms as i32)
    } else {
        n_atoms as i32
    };
    writeln!(
        writer,
        "{:5}{:12.6}{:12.6}{:12.6}",
        natoms_signed, grid.origin[0], grid.origin[1], grid.origin[2]
    )
    .map_err(MolRsError::Io)?;

    // Voxel axis lines
    for i in 0..3 {
        let n = grid.dim[i];
        let n_signed: i32 = if i == 0 && is_angstrom {
            -(n as i32)
        } else {
            n as i32
        };
        // voxel_vec = cell[i] / N
        let vx = grid.cell[i][0] / n as F;
        let vy = grid.cell[i][1] / n as F;
        let vz = grid.cell[i][2] / n as F;
        writeln!(writer, "{:5}{:12.6}{:12.6}{:12.6}", n_signed, vx, vy, vz)
            .map_err(MolRsError::Io)?;
    }

    // Atom lines
    let atom_x = atoms.get_float("x");
    let atom_y = atoms.get_float("y");
    let atom_z = atoms.get_float("z");
    let atom_z_num = atoms.get_int("atomic_number");
    let atom_charge = atoms.get_float("charge");
    let atom_symbol = atoms.get_string("symbol");

    for i in 0..n_atoms {
        let z_num = atom_z_num
            .map(|a| a[[i]])
            .or_else(|| atom_symbol.and_then(|s| Element::by_symbol(&s[[i]]).map(|e| e.z() as I)))
            .unwrap_or(0);
        let charge = atom_charge.map(|a| a[[i]]).unwrap_or(z_num as F);
        let x = atom_x.map(|a| a[[i]]).unwrap_or(0.0);
        let y = atom_y.map(|a| a[[i]]).unwrap_or(0.0);
        let z = atom_z.map(|a| a[[i]]).unwrap_or(0.0);
        writeln!(
            writer,
            "{:5}{:12.6}{:12.6}{:12.6}{:12.6}",
            z_num, charge, x, y, z
        )
        .map_err(MolRsError::Io)?;
    }

    // MO header line
    if let Some(ref indices) = mo_indices {
        let parts: Vec<String> = std::iter::once(indices.len().to_string())
            .chain(indices.iter().map(|i| format!("{:4}", i)))
            .collect();
        writeln!(writer, "{}", parts.join("  ")).map_err(MolRsError::Io)?;
    }

    // Volumetric data
    let n_voxels = nx * ny * nz;
    if let Some(ref indices) = mo_indices {
        // Interleave MO data
        let n_vals = indices.len();
        let arrays: Vec<Option<&[F]>> = indices
            .iter()
            .map(|idx| grid.get_raw(&format!("mo_{}", idx)))
            .collect();
        let mut col = 0;
        for i in 0..n_voxels {
            for (k, arr_opt) in arrays.iter().enumerate() {
                let val = arr_opt.map(|a| a[i]).unwrap_or(0.0);
                write!(writer, "{:13.5E}", val).map_err(MolRsError::Io)?;
                col += 1;
                if col % 6 == 0 || (i == n_voxels - 1 && k == n_vals - 1) {
                    writeln!(writer).map_err(MolRsError::Io)?;
                    col = 0;
                }
            }
        }
        if col != 0 {
            writeln!(writer).map_err(MolRsError::Io)?;
        }
    } else {
        // Single density field
        let data = grid
            .get_raw("density")
            .ok_or_else(|| MolRsError::validation("cube grid has no 'density' array"))?;
        let mut col = 0;
        for &val in data {
            write!(writer, "{:13.5E}", val).map_err(MolRsError::Io)?;
            col += 1;
            if col % 6 == 0 {
                writeln!(writer).map_err(MolRsError::Io)?;
                col = 0;
            }
        }
        if col != 0 {
            writeln!(writer).map_err(MolRsError::Io)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_f64(s: &str, line_no: usize) -> Result<f64, MolRsError> {
    s.parse::<f64>()
        .map_err(|_| MolRsError::parse_error(line_no, format!("expected float, got '{}'", s)))
}

/// Read `count` floating-point values from `reader`, ignoring line structure.
fn read_volumetric_data<R: BufRead>(
    reader: &mut R,
    count: usize,
    line_no: &mut usize,
) -> Result<Vec<F>, MolRsError> {
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).map_err(MolRsError::Io)?;
        *line_no += 1;
        if bytes == 0 {
            return Err(MolRsError::parse_error(
                *line_no,
                format!(
                    "unexpected EOF while reading volumetric data (got {}/{} values)",
                    values.len(),
                    count
                ),
            ));
        }
        for tok in line.split_whitespace() {
            if values.len() >= count {
                break;
            }
            let v = tok.parse::<f64>().map_err(|_| {
                MolRsError::parse_error(
                    *line_no,
                    format!("expected float in volumetric data, got '{}'", tok),
                )
            })?;
            values.push(v as F);
        }
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn parse_f64_valid() {
        assert!((super::parse_f64("1.23E+02", 1).unwrap() - 123.0).abs() < 1e-10);
        assert!((super::parse_f64("-0.5", 1).unwrap() - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn parse_f64_invalid() {
        assert!(super::parse_f64("abc", 1).is_err());
    }
}
