//! VASP CHGCAR / CHGDIF volumetric data file reader.
//!
//! ## File layout
//!
//! ```text
//! <comment>                        ← system name → frame.meta["title"]
//! <scale>                          ← uniform scaling factor
//! <a1x> <a1y> <a1z>               ← lattice vector a1 (Å after scaling)
//! <a2x> <a2y> <a2z>               ← lattice vector a2
//! <a3x> <a3y> <a3z>               ← lattice vector a3
//! <Elem1> <Elem2> …               ← element symbols
//! <n1> <n2> …                     ← element counts
//! Direct | Cartesian               ← coordinate mode
//! <s1> <s2> <s3>                  ← one atom per line
//! …
//!                                  ← blank line
//! <nx> <ny> <nz>                  ← grid dimensions
//! <val> <val> …                   ← nx*ny*nz values, 5 per line, VASP column-major (x fastest)
//!                                  ← optional: augmentation occupancies (skipped)
//!                                  ← optional: blank line + nx ny nz + spin data
//! ```
//!
//! ## Grid stored in Frame
//!
//! The returned [`Frame`] carries a [`Grid`] under key `"chgcar"` containing:
//!
//! | Array key | Content                                   | Always? |
//! |-----------|-------------------------------------------|---------|
//! | `"total"` | Total charge density (raw: ρ·V_cell, e)  | yes     |
//! | `"diff"`  | Spin density α−β (raw: ρ·V_cell, e)      | ISPIN=2 |
//!
//! The values are stored **as-is** from the file (ρ × V_cell).
//! To convert to charge density in e/Å³: divide by `simbox.volume()`.
//!
//! ## Grid axis convention
//!
//! The VASP/FORTRAN data is x-fastest (column-major).
//! `read_chgcar` converts to C row-major `(ix, iy, iz)` order so that
//! `grid["total"][[ix, iy, iz]]` is `ρ(ix, iy, iz) × V`.
//!
//! ## Atom positions
//!
//! Atom positions are returned in Cartesian Å.  If the file uses `Direct`
//! coordinates they are multiplied by the lattice matrix on read.

use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array1;

use molrs::block::Block;
use molrs::error::MolRsError;
use molrs::frame::Frame;
use molrs::grid::Grid;
use molrs::region::simbox::SimBox;
use molrs::types::F;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read a CHGCAR (or CHGDIF) file from disk.
///
/// Returns a [`Frame`] with:
/// - `"atoms"` block: `symbol` (str), `x`/`y`/`z` (float, Å Cartesian)
/// - `simbox`: triclinic periodic box derived from the POSCAR header
/// - grid `"chgcar"`: [`Grid`] containing `"total"` and optionally `"diff"`
///
/// # Errors
///
/// Returns [`MolRsError`] on any I/O or parse failure.
pub fn read_chgcar<P: AsRef<Path>>(path: P) -> Result<Frame, MolRsError> {
    let file = std::fs::File::open(path.as_ref()).map_err(MolRsError::Io)?;
    read_chgcar_from_reader(BufReader::new(file))
}

/// Read a CHGCAR from any [`BufRead`] source.
pub fn read_chgcar_from_reader<R: BufRead>(mut reader: R) -> Result<Frame, MolRsError> {
    let mut line_no = 0usize;

    macro_rules! next_line {
        () => {{
            let mut s = String::new();
            reader.read_line(&mut s).map_err(|e| MolRsError::Io(e))?;
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
    // Header
    // -----------------------------------------------------------------------

    // Line 1: comment / system name
    let title = next_line!().trim().to_string();

    // Line 2: scaling factor
    let scale: F = next_line!()
        .trim()
        .parse::<f64>()
        .map_err(|_| parse_err!("expected scaling factor"))? as F;

    // Lines 3-5: lattice vectors (rows = a1, a2, a3)
    let mut cell = [[0.0f64; 3]; 3];
    for row in &mut cell {
        let s = next_line!();
        let vals = parse_floats(&s, 3, line_no)?;
        row[0] = vals[0] * scale as f64;
        row[1] = vals[1] * scale as f64;
        row[2] = vals[2] * scale as f64;
    }
    let cell_f: [[F; 3]; 3] = cell.map(|row| row.map(|v| v as F));

    // Line 6: element symbols
    let symbols_line = next_line!();
    let symbols: Vec<&str> = symbols_line.split_whitespace().collect();
    // Guard: must be non-empty and non-numeric (VASP4 format had no symbols)
    let has_symbols = !symbols.is_empty() && symbols[0].parse::<u32>().is_err();

    // Line 7 (or 6 if VASP4): counts
    let counts: Vec<usize> = if has_symbols {
        let s = next_line!();
        parse_usize_vec(&s, line_no)?
    } else {
        // VASP4: counts were on the "symbols" line
        symbols
            .iter()
            .map(|t| t.parse::<usize>().map_err(|_| parse_err!("bad count")))
            .collect::<Result<Vec<_>, _>>()?
    };

    if has_symbols && symbols.len() != counts.len() {
        return Err(MolRsError::parse_error(
            line_no,
            format!(
                "element count mismatch: {} symbols but {} counts",
                symbols.len(),
                counts.len()
            ),
        ));
    }

    let n_atoms: usize = counts.iter().sum();

    // Line 8: Selective dynamics (optional) or coordinate mode
    let mut mode_line = next_line!();
    if mode_line.trim().to_ascii_lowercase().starts_with('s') {
        // selective dynamics line — skip and read the real mode line
        mode_line = next_line!();
    }
    let mode_lower = mode_line.trim().to_ascii_lowercase();
    let is_direct = if mode_lower.starts_with('d') {
        true
    } else if mode_lower.starts_with('c') {
        false
    } else {
        return Err(MolRsError::parse_error(
            line_no,
            format!(
                "unrecognised coordinate mode {:?}; expected 'Direct' or 'Cartesian'",
                mode_line.trim()
            ),
        ));
    };

    // Lines 9..9+n_atoms: atom positions
    let mut frac_x = Vec::with_capacity(n_atoms);
    let mut frac_y = Vec::with_capacity(n_atoms);
    let mut frac_z = Vec::with_capacity(n_atoms);

    for _ in 0..n_atoms {
        let s = next_line!();
        let vals = parse_floats(&s, 3, line_no)?;
        frac_x.push(vals[0]);
        frac_y.push(vals[1]);
        frac_z.push(vals[2]);
    }

    // Convert to Cartesian if Direct
    let (cart_x, cart_y, cart_z) = if is_direct {
        fractional_to_cartesian(&frac_x, &frac_y, &frac_z, &cell)
    } else {
        // Already Cartesian; apply scale
        let s = scale as f64;
        (
            frac_x.iter().map(|v| v * s).collect(),
            frac_y.iter().map(|v| v * s).collect(),
            frac_z.iter().map(|v| v * s).collect(),
        )
    };

    // Build atoms block
    let mut atom_syms: Vec<String> = Vec::with_capacity(n_atoms);
    if has_symbols {
        for (sym, &cnt) in symbols.iter().zip(counts.iter()) {
            for _ in 0..cnt {
                atom_syms.push(sym.to_string());
            }
        }
    }

    let mut atoms = Block::new();
    atoms
        .insert(
            "x",
            Array1::from_vec(cart_x.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "y",
            Array1::from_vec(cart_y.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "z",
            Array1::from_vec(cart_z.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    if !atom_syms.is_empty() {
        use ndarray::ArrayD;
        use ndarray::IxDyn;
        atoms
            .insert(
                "symbol",
                ArrayD::from_shape_vec(IxDyn(&[n_atoms]), atom_syms)
                    .expect("shape matches")
                    .into_dyn(),
            )
            .map_err(MolRsError::Block)?;
    }

    // Build SimBox from cell rows (CHGCAR rows = molrs column convention)
    // molrs SimBox h-matrix: h[:,i] = i-th lattice vector
    // cell[i] = i-th lattice vector row → h column i
    let h = ndarray::Array2::from_shape_fn((3, 3), |(i, j)| cell[j][i] as F);
    let origin = ndarray::array![0.0 as F, 0.0 as F, 0.0 as F];
    let simbox = SimBox::new(h, origin, [true, true, true])
        .map_err(|e| MolRsError::parse(format!("invalid cell: {:?}", e)))?;

    // -----------------------------------------------------------------------
    // Skip blank line(s) before grid header
    // -----------------------------------------------------------------------
    skip_blank_lines(&mut reader, &mut line_no)?;

    // -----------------------------------------------------------------------
    // Grid header: nx ny nz
    // -----------------------------------------------------------------------
    let dim_line = next_line!();
    let dims = parse_usize_vec(&dim_line, line_no)?;
    if dims.len() < 3 {
        return Err(parse_err!("expected 'nx ny nz' grid dimensions"));
    }
    let [nx, ny, nz] = [dims[0], dims[1], dims[2]];
    let n_voxels = nx * ny * nz;

    let mut grid = Grid::new(dim_line_to_dim(nx, ny, nz), [0.0; 3], cell_f, [true; 3]);

    // -----------------------------------------------------------------------
    // Read total charge density
    // -----------------------------------------------------------------------
    let total_vasp = read_volumetric_data(&mut reader, n_voxels, &mut line_no)?;
    let total = vasp_to_row_major(total_vasp, nx, ny, nz);
    grid.insert("total", total)
        .map_err(|e| MolRsError::parse(format!("grid insert error: {}", e)))?;

    // -----------------------------------------------------------------------
    // Optional: augmentation occupancies (skip) + spin density
    // -----------------------------------------------------------------------
    if let Some(diff) = try_read_optional_grid(&mut reader, n_voxels, nx, ny, nz, &mut line_no)? {
        grid.insert("diff", diff)
            .map_err(|e| MolRsError::parse(format!("grid insert error: {}", e)))?;
    }

    // -----------------------------------------------------------------------
    // Assemble Frame
    // -----------------------------------------------------------------------
    let mut frame = Frame::new();
    if !title.is_empty() {
        frame.meta.insert("title".into(), title);
    }
    frame.simbox = Some(simbox);
    frame.insert("atoms", atoms);
    frame.insert_grid("chgcar", grid);

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_floats(line: &str, expected: usize, line_no: usize) -> Result<Vec<f64>, MolRsError> {
    let vals: Vec<f64> = line
        .split_whitespace()
        .take(expected)
        .map(|s| {
            s.parse::<f64>().map_err(|_| {
                MolRsError::parse_error(line_no, format!("expected float, got '{}'", s))
            })
        })
        .collect::<Result<_, _>>()?;
    if vals.len() < expected {
        return Err(MolRsError::parse_error(
            line_no,
            format!("expected {} floats, got {}", expected, vals.len()),
        ));
    }
    Ok(vals)
}

fn parse_usize_vec(line: &str, line_no: usize) -> Result<Vec<usize>, MolRsError> {
    line.split_whitespace()
        .map(|s| {
            s.parse::<usize>().map_err(|_| {
                MolRsError::parse_error(line_no, format!("expected integer, got '{}'", s))
            })
        })
        .collect()
}

fn fractional_to_cartesian(
    sx: &[f64],
    sy: &[f64],
    sz: &[f64],
    cell: &[[f64; 3]; 3],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = sx.len();
    let mut cx = Vec::with_capacity(n);
    let mut cy = Vec::with_capacity(n);
    let mut cz = Vec::with_capacity(n);
    for i in 0..n {
        // r = s1*a1 + s2*a2 + s3*a3, cell[k] = a_{k+1}
        cx.push(sx[i] * cell[0][0] + sy[i] * cell[1][0] + sz[i] * cell[2][0]);
        cy.push(sx[i] * cell[0][1] + sy[i] * cell[1][1] + sz[i] * cell[2][1]);
        cz.push(sx[i] * cell[0][2] + sy[i] * cell[1][2] + sz[i] * cell[2][2]);
    }
    (cx, cy, cz)
}

/// Read `n_voxels` floating-point values from `reader`, ignoring line structure.
fn read_volumetric_data<R: BufRead>(
    reader: &mut R,
    n_voxels: usize,
    line_no: &mut usize,
) -> Result<Vec<F>, MolRsError> {
    let mut values = Vec::with_capacity(n_voxels);
    while values.len() < n_voxels {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).map_err(MolRsError::Io)?;
        *line_no += 1;
        if bytes == 0 {
            return Err(MolRsError::parse_error(
                *line_no,
                format!(
                    "unexpected EOF while reading grid data (got {}/{} values)",
                    values.len(),
                    n_voxels
                ),
            ));
        }
        for tok in line.split_whitespace() {
            if values.len() >= n_voxels {
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

/// Convert VASP column-major (x fastest: flat index = ix + nx*iy + nx*ny*iz)
/// to our row-major (z fastest: flat index = ix*ny*nz + iy*nz + iz).
fn vasp_to_row_major(vasp: Vec<F>, nx: usize, ny: usize, nz: usize) -> Vec<F> {
    let n = nx * ny * nz;
    let mut out = vec![0.0 as F; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let src = ix + nx * iy + nx * ny * iz;
                let dst = ix * ny * nz + iy * nz + iz;
                out[dst] = vasp[src];
            }
        }
    }
    out
}

/// Skip blank lines and lines that consist only of whitespace.
fn skip_blank_lines<R: BufRead>(reader: &mut R, line_no: &mut usize) -> Result<(), MolRsError> {
    loop {
        // Peek by filling the buffer without consuming
        let buf = reader.fill_buf().map_err(MolRsError::Io)?;
        if buf.is_empty() {
            return Ok(()); // EOF
        }
        // Find first newline
        let newline_pos = buf.iter().position(|&b| b == b'\n');
        let line_is_blank = match newline_pos {
            None => buf.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\r'),
            Some(pos) => buf[..pos]
                .iter()
                .all(|&b| b == b' ' || b == b'\t' || b == b'\r'),
        };
        if !line_is_blank {
            return Ok(());
        }
        // Consume the blank line
        let consume = newline_pos.map_or(buf.len(), |p| p + 1);
        reader.consume(consume);
        *line_no += 1;
    }
}

/// After the total charge density block, try to read an optional spin-density block.
///
/// Skips any "augmentation occupancies" lines, then looks for another `nx ny nz`
/// header followed by volumetric data.
///
/// Returns `Ok(Some(data))` if a second grid was found, `Ok(None)` otherwise.
fn try_read_optional_grid<R: BufRead>(
    reader: &mut R,
    n_voxels: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    line_no: &mut usize,
) -> Result<Option<Vec<F>>, MolRsError> {
    // Skip blank lines and augmentation occupancy blocks
    loop {
        skip_blank_lines(reader, line_no)?;

        // Peek at the next non-blank line
        let buf = reader.fill_buf().map_err(MolRsError::Io)?;
        if buf.is_empty() {
            return Ok(None);
        }

        // Read the line to inspect it
        let mut line = String::new();
        reader.read_line(&mut line).map_err(MolRsError::Io)?;
        *line_no += 1;

        let trimmed = line.trim().to_ascii_lowercase();

        if trimmed.starts_with("augmentation") {
            // Skip this line and any following data lines (all-numeric)
            // until we hit another blank or "augmentation" or the dim line
            loop {
                skip_blank_lines(reader, line_no)?;
                let buf = reader.fill_buf().map_err(MolRsError::Io)?;
                if buf.is_empty() {
                    return Ok(None);
                }
                // Peek: is the next line a dim line, aug line, or data?
                let mut peek = String::new();
                reader.read_line(&mut peek).map_err(MolRsError::Io)?;
                *line_no += 1;
                let peek_trim = peek.trim().to_ascii_lowercase();
                if peek_trim.starts_with("augmentation") {
                    continue; // another aug block
                }
                if peek_trim.is_empty() {
                    break; // blank → back to outer loop
                }
                // If this line contains only integers that match nx ny nz, it's the dim header
                if is_dim_line(&peek, nx, ny, nz) {
                    let diff_vasp = read_volumetric_data(reader, n_voxels, line_no)?;
                    return Ok(Some(vasp_to_row_major(diff_vasp, nx, ny, nz)));
                }
                // Otherwise it's data we're skipping
            }
            continue;
        }

        // Check if this is a dim line matching nx ny nz
        if is_dim_line(&line, nx, ny, nz) {
            let diff_vasp = read_volumetric_data(reader, n_voxels, line_no)?;
            return Ok(Some(vasp_to_row_major(diff_vasp, nx, ny, nz)));
        }

        // Something else → no second grid
        return Ok(None);
    }
}

fn is_dim_line(line: &str, nx: usize, ny: usize, nz: usize) -> bool {
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 3 {
        return false;
    }
    toks[0].parse::<usize>().ok() == Some(nx)
        && toks[1].parse::<usize>().ok() == Some(ny)
        && toks[2].parse::<usize>().ok() == Some(nz)
}

fn dim_line_to_dim(nx: usize, ny: usize, nz: usize) -> [usize; 3] {
    [nx, ny, nz]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vasp_to_row_major_reorder() {
        // VASP order (x fastest): flat[k] at k = ix + nx*iy + nx*ny*iz
        // Our order (z fastest):  flat[k] at k = ix*ny*nz + iy*nz + iz
        // With nx=ny=nz=2, values 1..8:
        // VASP flat: k=0 → (0,0,0), k=1 → (1,0,0), k=2 → (0,1,0), k=3 → (1,1,0)
        //            k=4 → (0,0,1), k=5 → (1,0,1), k=6 → (0,1,1), k=7 → (1,1,1)
        let vasp: Vec<F> = (1..=8).map(|v| v as F).collect();
        let row_major = vasp_to_row_major(vasp, 2, 2, 2);
        // (ix=0,iy=0,iz=0) → VASP k=0 → value 1 → row-major dst=0*4+0*2+0=0
        assert_eq!(row_major[0 * 4 + 0 * 2 + 0], 1.0); // (0,0,0)
        assert_eq!(row_major[1 * 4 + 0 * 2 + 0], 2.0); // (1,0,0)
        assert_eq!(row_major[0 * 4 + 1 * 2 + 0], 3.0); // (0,1,0)
        assert_eq!(row_major[0 * 4 + 0 * 2 + 1], 5.0); // (0,0,1)
    }
}
