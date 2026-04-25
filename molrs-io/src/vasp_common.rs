//! Shared helpers for VASP-style file headers (POSCAR, CHGCAR, CONTCAR).
//!
//! VASP uses a common preamble across several file types:
//!
//! ```text
//! line 1   : comment / system name
//! line 2   : global scale factor
//! line 3-5 : lattice vectors (rows; one vector per line)
//! line 6   : element symbols (VASP5+) — optional, may be omitted
//! line 7   : atom counts per element
//! line 8   : "Selective dynamics" (optional, single line)
//! line N   : coordinate mode — first non-blank char ∈ {D,d,K,k,C,c}
//! line N+1+: atom records (3 floats; optionally + T/F flags + symbol)
//! ```
//!
//! Helpers in this module read this preamble and return strongly typed values.
//! Errors use `std::io::Error` with `ErrorKind::InvalidData`, matching the
//! convention of other readers in `molrs-io`.

use std::io::{BufRead, Error, ErrorKind, Result};

use molrs::types::F;

/// VASP coordinate mode for atom positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordMode {
    /// Fractional / direct coordinates (in lattice basis).
    Direct,
    /// Cartesian coordinates (in Å, scaled by the global scale factor).
    Cartesian,
}

/// Header section of a VASP file, before the coordinate mode line.
#[derive(Debug, Clone)]
pub struct VaspHeader {
    /// First-line comment / system name.
    pub title: String,
    /// Global scale factor applied to lattice vectors and Cartesian coords.
    pub scale: F,
    /// 3×3 cell with rows = lattice vectors a1, a2, a3 (already scaled).
    pub cell: [[F; 3]; 3],
    /// Element symbols, in order matching `counts`. Empty for VASP4 files.
    pub symbols: Vec<String>,
    /// Atom counts per element.
    pub counts: Vec<usize>,
    /// True when a `Selective dynamics` line was present before the mode line.
    pub selective_dynamics: bool,
    /// Coordinate mode for the atom block.
    pub mode: CoordMode,
}

impl VaspHeader {
    /// Total atom count = sum of `counts`.
    pub fn total_atoms(&self) -> usize {
        self.counts.iter().sum()
    }
}

/// Wrap a parse failure as `InvalidData` with line number context.
fn parse_err(line_no: usize, msg: impl AsRef<str>) -> Error {
    Error::new(
        ErrorKind::InvalidData,
        format!("line {}: {}", line_no, msg.as_ref()),
    )
}

/// Read one line into `buf`, returning `Ok(false)` on EOF.
fn read_line_into<R: BufRead>(reader: &mut R, buf: &mut String) -> Result<bool> {
    buf.clear();
    let n = reader.read_line(buf)?;
    Ok(n > 0)
}

/// Parse exactly `expected` whitespace-separated floats from `line`.
pub fn parse_floats(line: &str, expected: usize, line_no: usize) -> Result<Vec<F>> {
    let vals: Vec<F> = line
        .split_whitespace()
        .take(expected)
        .map(|s| {
            s.parse::<F>()
                .map_err(|_| parse_err(line_no, format!("expected float, got '{}'", s)))
        })
        .collect::<Result<_>>()?;
    if vals.len() < expected {
        return Err(parse_err(
            line_no,
            format!("expected {} floats, got {}", expected, vals.len()),
        ));
    }
    Ok(vals)
}

/// Parse all whitespace-separated `usize` tokens on `line`.
pub fn parse_usize_vec(line: &str, line_no: usize) -> Result<Vec<usize>> {
    line.split_whitespace()
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| parse_err(line_no, format!("expected integer, got '{}'", s)))
        })
        .collect()
}

/// Convert fractional coordinates to Cartesian using a row-major cell.
///
/// `cell[i] = a_{i+1}` (lattice vector i as a 3-element row).
/// For each atom: `r_cart = s_x * a1 + s_y * a2 + s_z * a3`.
pub fn fractional_to_cartesian(
    sx: &[F],
    sy: &[F],
    sz: &[F],
    cell: &[[F; 3]; 3],
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let n = sx.len();
    let mut cx = Vec::with_capacity(n);
    let mut cy = Vec::with_capacity(n);
    let mut cz = Vec::with_capacity(n);
    for i in 0..n {
        cx.push(sx[i] * cell[0][0] + sy[i] * cell[1][0] + sz[i] * cell[2][0]);
        cy.push(sx[i] * cell[0][1] + sy[i] * cell[1][1] + sz[i] * cell[2][1]);
        cz.push(sx[i] * cell[0][2] + sy[i] * cell[1][2] + sz[i] * cell[2][2]);
    }
    (cx, cy, cz)
}

/// Read the VASP header (title, scale, cell, counts, mode flags).
///
/// On return, the reader is positioned at the first atom-coordinate line.
/// `*line_no` is updated to reflect the number of lines consumed.
pub fn read_header<R: BufRead>(reader: &mut R, line_no: &mut usize) -> Result<VaspHeader> {
    let mut buf = String::new();

    // Line 1: title / comment
    if !read_line_into(reader, &mut buf)? {
        return Err(parse_err(*line_no + 1, "unexpected EOF before title"));
    }
    *line_no += 1;
    let title = buf.trim().to_string();

    // Line 2: global scale (POSCAR allows three scales = per-axis; this reader
    // accepts a single uniform scale, which is by far the common case).
    if !read_line_into(reader, &mut buf)? {
        return Err(parse_err(*line_no + 1, "unexpected EOF before scale"));
    }
    *line_no += 1;
    let scale: F = buf
        .split_whitespace()
        .next()
        .ok_or_else(|| parse_err(*line_no, "missing scale factor"))?
        .parse()
        .map_err(|_| parse_err(*line_no, "expected scale factor"))?;

    // Lines 3-5: lattice vectors
    let mut cell = [[0.0 as F; 3]; 3];
    for row in &mut cell {
        if !read_line_into(reader, &mut buf)? {
            return Err(parse_err(*line_no + 1, "unexpected EOF in lattice vectors"));
        }
        *line_no += 1;
        let vals = parse_floats(&buf, 3, *line_no)?;
        row[0] = vals[0] * scale;
        row[1] = vals[1] * scale;
        row[2] = vals[2] * scale;
    }

    // Line 6: symbols (VASP5+) or counts (VASP4)
    if !read_line_into(reader, &mut buf)? {
        return Err(parse_err(*line_no + 1, "unexpected EOF before symbols"));
    }
    *line_no += 1;
    let tokens: Vec<&str> = buf.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(parse_err(*line_no, "empty symbols/counts line"));
    }

    let has_symbols = tokens[0].parse::<u32>().is_err();
    let (symbols, counts) = if has_symbols {
        let symbols: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
        // Line 7: counts
        if !read_line_into(reader, &mut buf)? {
            return Err(parse_err(*line_no + 1, "unexpected EOF before counts"));
        }
        *line_no += 1;
        let counts = parse_usize_vec(&buf, *line_no)?;
        if counts.len() != symbols.len() {
            return Err(parse_err(
                *line_no,
                format!(
                    "element count mismatch: {} symbols but {} counts",
                    symbols.len(),
                    counts.len()
                ),
            ));
        }
        (symbols, counts)
    } else {
        let counts = tokens
            .iter()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| parse_err(*line_no, format!("expected count, got '{}'", s)))
            })
            .collect::<Result<Vec<_>>>()?;
        (Vec::new(), counts)
    };

    // Line 8 (or N): optional Selective dynamics, then coord mode
    if !read_line_into(reader, &mut buf)? {
        return Err(parse_err(*line_no + 1, "unexpected EOF before mode"));
    }
    *line_no += 1;
    let mut selective_dynamics = false;
    let mut mode_line = buf.trim().to_string();
    if mode_line
        .chars()
        .next()
        .map(|c| c.eq_ignore_ascii_case(&'s'))
        .unwrap_or(false)
    {
        selective_dynamics = true;
        if !read_line_into(reader, &mut buf)? {
            return Err(parse_err(*line_no + 1, "unexpected EOF after selective"));
        }
        *line_no += 1;
        mode_line = buf.trim().to_string();
    }

    let mode = match mode_line.chars().next() {
        Some(c) if c.eq_ignore_ascii_case(&'d') => CoordMode::Direct,
        Some(c) if c.eq_ignore_ascii_case(&'c') || c.eq_ignore_ascii_case(&'k') => {
            CoordMode::Cartesian
        }
        _ => {
            return Err(parse_err(
                *line_no,
                format!(
                    "unrecognised coordinate mode {:?}; expected Direct or Cartesian",
                    mode_line
                ),
            ));
        }
    };

    Ok(VaspHeader {
        title,
        scale,
        cell,
        symbols,
        counts,
        selective_dynamics,
        mode,
    })
}

/// One row of an atom-coordinate line — coordinates plus optional selective-dynamics flags.
#[derive(Debug, Clone)]
pub struct AtomRow {
    pub x: F,
    pub y: F,
    pub z: F,
    /// Per-axis "T"/"F" flags from selective dynamics, if present.
    pub flags: Option<[bool; 3]>,
}

/// Parse a single atom row. Selective dynamics flags are detected by the
/// presence of `T`/`F` tokens after the three coordinates.
pub fn parse_atom_row(line: &str, line_no: usize, expect_flags: bool) -> Result<AtomRow> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 3 {
        return Err(parse_err(
            line_no,
            format!("expected 3 floats, got {} tokens", tokens.len()),
        ));
    }
    let x: F = tokens[0]
        .parse()
        .map_err(|_| parse_err(line_no, format!("bad x: '{}'", tokens[0])))?;
    let y: F = tokens[1]
        .parse()
        .map_err(|_| parse_err(line_no, format!("bad y: '{}'", tokens[1])))?;
    let z: F = tokens[2]
        .parse()
        .map_err(|_| parse_err(line_no, format!("bad z: '{}'", tokens[2])))?;

    let flags = if expect_flags && tokens.len() >= 6 {
        let parse_flag = |t: &str| -> Result<bool> {
            match t {
                "T" | "t" => Ok(true),
                "F" | "f" => Ok(false),
                other => Err(parse_err(
                    line_no,
                    format!("expected selective-dynamics T/F, got '{}'", other),
                )),
            }
        };
        Some([
            parse_flag(tokens[3])?,
            parse_flag(tokens[4])?,
            parse_flag(tokens[5])?,
        ])
    } else {
        None
    };

    Ok(AtomRow { x, y, z, flags })
}

/// Per-atom coordinates plus optional selective-dynamics flags.
pub type CoordBlock = (Vec<F>, Vec<F>, Vec<F>, Option<Vec<[bool; 3]>>);

/// Read `n` atom rows from the reader. Coordinates are returned as parallel
/// `(xs, ys, zs)` vectors plus optional selective-dynamics flags.
///
/// **Note**: the values returned are exactly what the file contains — fractional
/// coordinates are NOT yet converted to Cartesian. The caller decides based on
/// `header.mode` whether to call `fractional_to_cartesian`.
pub fn read_coords<R: BufRead>(
    reader: &mut R,
    n: usize,
    line_no: &mut usize,
    expect_flags: bool,
) -> Result<CoordBlock> {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut flags = if expect_flags {
        Some(Vec::with_capacity(n))
    } else {
        None
    };
    let mut buf = String::new();
    for i in 0..n {
        if !read_line_into(reader, &mut buf)? {
            return Err(parse_err(
                *line_no + 1,
                format!("unexpected EOF at atom {}/{}", i, n),
            ));
        }
        *line_no += 1;
        let row = parse_atom_row(&buf, *line_no, expect_flags)?;
        xs.push(row.x);
        ys.push(row.y);
        zs.push(row.z);
        if let (Some(out), Some(f)) = (flags.as_mut(), row.flags) {
            out.push(f);
        }
    }
    Ok((xs, ys, zs, flags))
}

/// Build the per-atom `symbol` column from `symbols` + `counts`. Each element
/// symbol is repeated by its count, in declaration order. Empty if the file
/// did not declare symbols (VASP4 format).
pub fn expand_symbols(symbols: &[String], counts: &[usize]) -> Vec<String> {
    let total: usize = counts.iter().sum();
    let mut out = Vec::with_capacity(total);
    for (sym, &cnt) in symbols.iter().zip(counts.iter()) {
        for _ in 0..cnt {
            out.push(sym.clone());
        }
    }
    out
}

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
    fn parses_basic_poscar_header() {
        let mut reader = Cursor::new(POSCAR_BN.as_bytes());
        let mut line_no = 0usize;
        let h = read_header(&mut reader, &mut line_no).expect("header parses");
        assert_eq!(h.title, "BN bulk");
        assert_eq!(h.scale, 1.0);
        assert_eq!(h.symbols, vec!["B".to_string(), "N".to_string()]);
        assert_eq!(h.counts, vec![1, 1]);
        assert!(!h.selective_dynamics);
        assert_eq!(h.mode, CoordMode::Direct);
        assert_eq!(h.total_atoms(), 2);
    }

    #[test]
    fn fractional_to_cartesian_roundtrip() {
        let cell = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let (cx, cy, cz) = fractional_to_cartesian(&[0.5], &[0.5], &[0.5], &cell);
        assert_eq!((cx[0], cy[0], cz[0]), (1.0, 1.0, 1.0));
    }

    #[test]
    fn parses_atom_row_with_flags() {
        let row = parse_atom_row("0.0 0.5 0.25 T F T", 1, true).unwrap();
        assert_eq!(row.flags, Some([true, false, true]));
    }
}
