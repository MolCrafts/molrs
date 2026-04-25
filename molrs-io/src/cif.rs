//! Crystallographic Information File (CIF) reader and writer — MVP subset.
//!
//! This implementation covers the CIF subset most molecular work needs:
//!
//! - `data_<id>` blocks. Multi-block files yield one [`Frame`] per
//!   [`CifReader::read_frame`] call.
//! - Key-value pairs `_key  value`, including parenthesized esd (`5.917(3)`).
//! - `loop_` tables. Only `_atom_site_*` (small-molecule CIF) and
//!   `_atom_site.*` (mmCIF) loops are extracted into an atoms block — all
//!   other loops are tolerantly skipped.
//! - Cell parameters: `_cell_length_a/b/c`, `_cell_angle_alpha/beta/gamma`.
//! - Atom-site columns: `label`, `type_symbol`, `fract_x/y/z`, `Cartn_x/y/z`,
//!   `occupancy`, `B_iso_or_equiv`.
//! - Comments (`#`) and blank lines.
//! - Multi-line strings (`;...;`) are recognised and skipped when they appear
//!   as a column value — their content is preserved as a single token.
//! - Quoted strings (`'...'`, `"..."`).
//! - Special values `?` and `.` (treated as empty).
//!
//! Out of scope:
//! - Full CIF dictionary validation
//! - `save_` blocks
//! - `loop_` nesting
//! - Symmetry expansion (the parsed coordinates are exactly what the file holds)

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use ndarray::{Array1, Array2, IxDyn, array};

use molrs::block::Block;
use molrs::frame::Frame;
use molrs::region::simbox::SimBox;
use molrs::types::F;

use crate::reader::{FrameReader, Reader};
use crate::writer::{FrameWriter, Writer};

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

fn insert_str_col(block: &mut Block, key: &str, vals: Vec<String>) -> Result<()> {
    let n = vals.len();
    let arr = Array1::from_vec(vals)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    block.insert(key, arr).map_err(invalid_data)
}

/// Strip a parenthesised standard-deviation suffix from a numeric token.
/// `5.917(3)` → `5.917`. Non-numeric tokens are returned unchanged.
fn strip_esd(token: &str) -> &str {
    if let Some(idx) = token.find('(') {
        &token[..idx]
    } else {
        token
    }
}

fn parse_float(token: &str) -> Option<F> {
    strip_esd(token).parse::<F>().ok()
}

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------
//
// We do not aim to be a strict CIF tokeniser — only good enough to feed the
// loop-row parser. Tokens emitted are owned strings (not borrowed) because
// `;`-multi-line tokens span several physical lines.

#[derive(Debug, Clone)]
struct LineSource<R: BufRead> {
    reader: R,
    /// Pending tokens from a single physical line, FIFO.
    pending: Vec<String>,
    /// One-line lookahead used by the block-restart logic. (Reserved for
    /// future use; the current parser only needs token-level pushback.)
    pushed_line: Option<String>,
    /// True after we've seen EOF on the underlying reader.
    eof: bool,
}

impl<R: BufRead> LineSource<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            pending: Vec::new(),
            pushed_line: None,
            eof: false,
        }
    }

    fn read_raw_line(&mut self) -> Result<Option<String>> {
        if let Some(line) = self.pushed_line.take() {
            return Ok(Some(line));
        }
        if self.eof {
            return Ok(None);
        }
        let mut buf = String::new();
        let bytes = self.reader.read_line(&mut buf)?;
        if bytes == 0 {
            self.eof = true;
            return Ok(None);
        }
        Ok(Some(buf))
    }

    /// Pull the next non-empty, non-comment token.
    fn next_token(&mut self) -> Result<Option<String>> {
        loop {
            if let Some(t) = self.pending.pop() {
                return Ok(Some(t));
            }
            let line = match self.read_raw_line()? {
                Some(l) => l,
                None => return Ok(None),
            };

            // Multi-line `;...;` opens when `;` is the first column.
            if let Some(rest) = line.strip_prefix(';') {
                let mut content = String::new();
                content.push_str(rest.trim_end_matches(['\r', '\n']));
                loop {
                    let next = match self.read_raw_line()? {
                        Some(l) => l,
                        None => return Err(invalid_data("unterminated ;-block in CIF")),
                    };
                    if let Some(after) = next.strip_prefix(';') {
                        // Discard this terminator line, push remaining tokens
                        // from after the `;` (rare but allowed).
                        let rest = after.trim();
                        if !rest.is_empty() {
                            self.tokenise_line(rest);
                        }
                        break;
                    }
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(next.trim_end_matches(['\r', '\n']));
                }
                return Ok(Some(content));
            }

            self.tokenise_line(line.trim_end_matches(['\r', '\n']));
            // Loop and pop next token.
        }
    }

    fn tokenise_line(&mut self, line: &str) {
        // Comments: full-line comments only when `#` is at column 0 OR after
        // whitespace (CIF allows `#` to start a comment after whitespace).
        // For simplicity, drop everything from the first `#` whose preceding
        // char is whitespace or start-of-line.
        let mut s = String::from(line);
        let mut cut: Option<usize> = None;
        let mut prev_is_ws = true;
        for (i, c) in s.char_indices() {
            if c == '\'' || c == '"' {
                // Don't cut inside quotes; bail out of comment search until
                // the close quote. (Approximation: track quote state.)
                // For our subset, treat the whole string as no-comment if a
                // quote appears before any '#'.
                cut = None;
                break;
            }
            if c == '#' && prev_is_ws {
                cut = Some(i);
                break;
            }
            prev_is_ws = c.is_whitespace();
        }
        if let Some(idx) = cut {
            s.truncate(idx);
        }
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return;
        }

        // Whitespace-tokenise honoring single & double quotes.
        let mut buf = String::new();
        let mut in_single = false;
        let mut in_double = false;
        let mut tokens: Vec<String> = Vec::new();
        for c in trimmed.chars() {
            if in_single {
                if c == '\'' {
                    in_single = false;
                    tokens.push(std::mem::take(&mut buf));
                } else {
                    buf.push(c);
                }
            } else if in_double {
                if c == '"' {
                    in_double = false;
                    tokens.push(std::mem::take(&mut buf));
                } else {
                    buf.push(c);
                }
            } else if c.is_whitespace() {
                if !buf.is_empty() {
                    tokens.push(std::mem::take(&mut buf));
                }
            } else if c == '\'' && buf.is_empty() {
                in_single = true;
            } else if c == '"' && buf.is_empty() {
                in_double = true;
            } else {
                buf.push(c);
            }
        }
        if !buf.is_empty() {
            tokens.push(buf);
        }
        // Reverse so we can pop in source order.
        tokens.reverse();
        for t in tokens {
            self.pending.push(t);
        }
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

struct FrameInProgress {
    name: String,
    cell_a: Option<F>,
    cell_b: Option<F>,
    cell_c: Option<F>,
    cell_alpha: Option<F>,
    cell_beta: Option<F>,
    cell_gamma: Option<F>,
    /// Misc string key-value pairs — copied to `frame.meta`.
    meta: HashMap<String, String>,
    /// Per-atom column name → list of token values, in row order.
    atom_cols: HashMap<String, Vec<String>>,
}

impl FrameInProgress {
    fn new(name: String) -> Self {
        Self {
            name,
            cell_a: None,
            cell_b: None,
            cell_c: None,
            cell_alpha: None,
            cell_beta: None,
            cell_gamma: None,
            meta: HashMap::new(),
            atom_cols: HashMap::new(),
        }
    }

    fn set_cell(&mut self, key: &str, value: &str) {
        let v = parse_float(value);
        match key {
            "_cell_length_a" | "_cell.length_a" => self.cell_a = v,
            "_cell_length_b" | "_cell.length_b" => self.cell_b = v,
            "_cell_length_c" | "_cell.length_c" => self.cell_c = v,
            "_cell_angle_alpha" | "_cell.angle_alpha" => self.cell_alpha = v,
            "_cell_angle_beta" | "_cell.angle_beta" => self.cell_beta = v,
            "_cell_angle_gamma" | "_cell.angle_gamma" => self.cell_gamma = v,
            _ => {}
        }
    }

    fn build_frame(self) -> Result<Frame> {
        if self.atom_cols.is_empty() {
            return Err(invalid_data(format!(
                "CIF block '{}' has no _atom_site_* loop",
                self.name
            )));
        }

        let n = self.atom_cols.values().next().map(|v| v.len()).unwrap_or(0);
        if n == 0 {
            return Err(invalid_data(format!(
                "CIF block '{}' atom loop has zero rows",
                self.name
            )));
        }
        for (k, v) in &self.atom_cols {
            if v.len() != n {
                return Err(invalid_data(format!(
                    "CIF block '{}': column '{}' has {} rows, expected {}",
                    self.name,
                    k,
                    v.len(),
                    n
                )));
            }
        }

        let mut atoms = Block::new();
        let mut frame = Frame::new();
        if !self.name.is_empty() {
            frame.meta.insert("title".into(), self.name);
        }
        for (k, v) in self.meta {
            frame.meta.insert(k, v);
        }

        // Cell → SimBox (only if all six are present)
        let cell_present = self.cell_a.is_some()
            && self.cell_b.is_some()
            && self.cell_c.is_some()
            && self.cell_alpha.is_some()
            && self.cell_beta.is_some()
            && self.cell_gamma.is_some();
        let h = if cell_present {
            Some(cell_to_h(
                self.cell_a.unwrap(),
                self.cell_b.unwrap(),
                self.cell_c.unwrap(),
                self.cell_alpha.unwrap(),
                self.cell_beta.unwrap(),
                self.cell_gamma.unwrap(),
            ))
        } else {
            None
        };

        // Coordinates: prefer Cartn over fract.
        let cartn_x = column_floats(
            &self.atom_cols,
            &["_atom_site_Cartn_x", "_atom_site.Cartn_x"],
        );
        let cartn_y = column_floats(
            &self.atom_cols,
            &["_atom_site_Cartn_y", "_atom_site.Cartn_y"],
        );
        let cartn_z = column_floats(
            &self.atom_cols,
            &["_atom_site_Cartn_z", "_atom_site.Cartn_z"],
        );
        let fract_x = column_floats(
            &self.atom_cols,
            &["_atom_site_fract_x", "_atom_site.fract_x"],
        );
        let fract_y = column_floats(
            &self.atom_cols,
            &["_atom_site_fract_y", "_atom_site.fract_y"],
        );
        let fract_z = column_floats(
            &self.atom_cols,
            &["_atom_site_fract_z", "_atom_site.fract_z"],
        );

        let (xs, ys, zs): (Vec<F>, Vec<F>, Vec<F>) = if let (Some(x), Some(y), Some(z)) =
            (&cartn_x, &cartn_y, &cartn_z)
        {
            (x.clone(), y.clone(), z.clone())
        } else if let (Some(fx), Some(fy), Some(fz), Some(h)) = (&fract_x, &fract_y, &fract_z, &h) {
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            let mut zs = Vec::with_capacity(n);
            for i in 0..n {
                let f = [fx[i], fy[i], fz[i]];
                let cart = h_times(h, &f);
                xs.push(cart[0]);
                ys.push(cart[1]);
                zs.push(cart[2]);
            }
            (xs, ys, zs)
        } else {
            return Err(invalid_data(
                "CIF atom loop: missing both Cartesian and fractional coordinates",
            ));
        };

        insert_float_col(&mut atoms, "x", xs)?;
        insert_float_col(&mut atoms, "y", ys)?;
        insert_float_col(&mut atoms, "z", zs)?;

        // Optional columns.
        if let Some(syms) = string_column(
            &self.atom_cols,
            &["_atom_site_type_symbol", "_atom_site.type_symbol"],
        ) {
            insert_str_col(&mut atoms, "symbol", syms)?;
        }
        if let Some(labels) = string_column(
            &self.atom_cols,
            &["_atom_site_label", "_atom_site.label_atom_id"],
        ) {
            insert_str_col(&mut atoms, "label", labels)?;
        }
        if let Some(occ) = column_floats(
            &self.atom_cols,
            &["_atom_site_occupancy", "_atom_site.occupancy"],
        ) {
            insert_float_col(&mut atoms, "occupancy", occ)?;
        }
        if let Some(b) = column_floats(
            &self.atom_cols,
            &["_atom_site_B_iso_or_equiv", "_atom_site.B_iso_or_equiv"],
        ) {
            insert_float_col(&mut atoms, "b_iso", b)?;
        }

        frame.insert("atoms", atoms);

        if let Some(h) = h {
            let h_arr = Array2::from_shape_fn((3, 3), |(i, j)| h[i][j]);
            let origin = array![0.0 as F, 0.0, 0.0];
            let simbox = SimBox::new(h_arr, origin, [true; 3])
                .map_err(|e| invalid_data(format!("CIF cell → SimBox: {:?}", e)))?;
            frame.simbox = Some(simbox);
        }

        Ok(frame)
    }
}

fn column_floats(map: &HashMap<String, Vec<String>>, keys: &[&str]) -> Option<Vec<F>> {
    for k in keys {
        if let Some(col) = map.get(*k) {
            return Some(col.iter().map(|s| parse_float(s).unwrap_or(0.0)).collect());
        }
    }
    None
}

fn string_column(map: &HashMap<String, Vec<String>>, keys: &[&str]) -> Option<Vec<String>> {
    for k in keys {
        if let Some(col) = map.get(*k) {
            return Some(col.clone());
        }
    }
    None
}

/// Build the H matrix (3 lattice vectors as rows) from a/b/c lengths and
/// alpha/beta/gamma angles in degrees. Convention: a along x, b in xy plane.
fn cell_to_h(a: F, b: F, c: F, alpha: F, beta: F, gamma: F) -> [[F; 3]; 3] {
    let to_rad = std::f64::consts::PI / 180.0;
    let ca = (alpha * to_rad).cos();
    let cb = (beta * to_rad).cos();
    let cg = (gamma * to_rad).cos();
    let sg = (gamma * to_rad).sin();

    let v1 = [a, 0.0, 0.0];
    let v2 = [b * cg, b * sg, 0.0];
    let v3x = c * cb;
    let v3y = c * (ca - cb * cg) / sg;
    let v3z2 = c * c - v3x * v3x - v3y * v3y;
    let v3z = if v3z2 > 0.0 { v3z2.sqrt() } else { 0.0 };
    let v3 = [v3x, v3y, v3z];
    [v1, v2, v3]
}

/// Multiply 3-vector by H matrix where H rows = lattice vectors:
/// `cart = fx*a1 + fy*a2 + fz*a3`.
fn h_times(h_rows: &[[F; 3]; 3], frac: &[F; 3]) -> [F; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = frac[0] * h_rows[0][i] + frac[1] * h_rows[1][i] + frac[2] * h_rows[2][i];
    }
    out
}

/// Read one CIF data block from `src`. Returns `Ok(None)` at EOF.
fn read_one_block<R: BufRead>(src: &mut LineSource<R>) -> Result<Option<Frame>> {
    let mut current: Option<FrameInProgress> = None;

    while let Some(tok) = src.next_token()? {
        if let Some(name) = tok.strip_prefix("data_") {
            if let Some(prev) = current.take() {
                // Push back: we rebuild a fake "data_" token via the pending stack.
                // Easiest: wrap in a pseudo-line and put back through pending.
                src.pending.push(format!("data_{}", name));
                return Ok(Some(prev.build_frame()?));
            }
            current = Some(FrameInProgress::new(name.to_string()));
            continue;
        }
        let frame = match current.as_mut() {
            Some(f) => f,
            None => continue, // skip stray tokens before any data_ block
        };

        if tok == "loop_" {
            handle_loop(src, frame)?;
            continue;
        }

        if let Some(stripped) = tok.strip_prefix('_') {
            // Key-value: read the next token as value.
            let key = format!("_{}", stripped);
            let value = src
                .next_token()?
                .ok_or_else(|| invalid_data(format!("missing value for {}", key)))?;
            frame.set_cell(&key, &value);
            // Save scalar string values into meta when small.
            if value.len() < 256 && key.starts_with("_chemical_") {
                frame.meta.insert(key, value);
            }
            continue;
        }

        // Stray tokens outside any structure — ignore.
    }

    if let Some(f) = current {
        return Ok(Some(f.build_frame()?));
    }
    Ok(None)
}

/// Handle a `loop_` table. Reads header `_keys` until a non-`_` token, then
/// consumes `n_keys * n_rows` tokens as data values.
fn handle_loop<R: BufRead>(src: &mut LineSource<R>, frame: &mut FrameInProgress) -> Result<()> {
    let mut keys: Vec<String> = Vec::new();

    while let Some(tok) = src.next_token()? {
        if let Some(rest) = tok.strip_prefix('_') {
            keys.push(format!("_{}", rest));
            continue;
        }
        // First non-`_` token is the first data value — push back via pending.
        src.pending.push(tok);
        break;
    }
    if keys.is_empty() {
        return Ok(()); // empty loop_; rare but allowed
    }

    // Only the *coordinate-bearing* atom_site loop is a true atom loop.
    // Anisotropic-displacement loops (`_atom_site_aniso_*`) and similar use
    // the same prefix but lack coordinates and must be skipped. Likewise,
    // ignore any second atom loop in the same block — only the first wins.
    let has_coord_col = keys.iter().any(|k| {
        k == "_atom_site_fract_x"
            || k == "_atom_site.fract_x"
            || k == "_atom_site_Cartn_x"
            || k == "_atom_site.Cartn_x"
    });
    let already_have_atoms = !frame.atom_cols.is_empty();
    let is_atom_loop = has_coord_col && !already_have_atoms;

    let n_cols = keys.len();
    if is_atom_loop {
        // Pre-allocate columns we care about.
        for k in &keys {
            frame.atom_cols.entry(k.clone()).or_default();
        }
    }

    // Collect data rows: read `n_cols` tokens at a time. End the loop when:
    //  - we run out of tokens (EOF), OR
    //  - the next token starts with `_`, `loop_`, `data_`, `save_`, or `global_`.
    'outer: loop {
        let mut row: Vec<String> = Vec::with_capacity(n_cols);
        for col_idx in 0..n_cols {
            let tok = match src.next_token()? {
                Some(t) => t,
                None => {
                    if col_idx == 0 {
                        break 'outer;
                    } else {
                        return Err(invalid_data(format!(
                            "CIF loop ended mid-row (got {} of {} cols)",
                            col_idx, n_cols
                        )));
                    }
                }
            };
            if col_idx == 0 {
                let lower = tok.as_str();
                if lower.starts_with('_')
                    || lower == "loop_"
                    || lower.starts_with("data_")
                    || lower.starts_with("save_")
                    || lower == "global_"
                {
                    src.pending.push(tok);
                    break 'outer;
                }
            }
            row.push(tok);
        }
        if is_atom_loop {
            for (key, val) in keys.iter().zip(row) {
                frame.atom_cols.get_mut(key).unwrap().push(val);
            }
        }
    }

    Ok(())
}

/// `FrameReader`-trait wrapper.
pub struct CifReader<R: BufRead> {
    src: LineSource<R>,
}

impl<R: BufRead> Reader for CifReader<R> {
    type R = R;
    type Frame = Frame;
    fn new(reader: R) -> Self {
        Self {
            src: LineSource::new(reader),
        }
    }
}

impl<R: BufRead> FrameReader for CifReader<R> {
    fn read_frame(&mut self) -> Result<Option<Frame>> {
        read_one_block(&mut self.src)
    }
}

/// Read every block in `path` as a Vec of frames.
pub fn read_cif_all<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut cr = CifReader::new(BufReader::new(file));
    cr.read_all()
}

/// Read the first block from `path`.
pub fn read_cif<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut cr = CifReader::new(BufReader::new(file));
    cr.read_frame()?
        .ok_or_else(|| invalid_data("CIF file has no data_ block"))
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Frame as a single-block CIF file at `path`.
pub fn write_cif<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    write_cif_frame(&mut w, frame)?;
    w.flush()
}

/// Write a Frame as a single CIF data block. Cell parameters come from
/// `frame.simbox`; atoms from the `atoms` block (Cartesian or fractional).
pub fn write_cif_frame<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("CIF write: frame has no atoms block"))?;
    let n = atoms
        .nrows()
        .ok_or_else(|| invalid_data("CIF write: atoms block has no rows"))?;

    let title = frame
        .meta
        .get("title")
        .cloned()
        .unwrap_or_else(|| "molrs".to_string());
    writeln!(writer, "data_{}", sanitise_data_name(&title))?;

    if let Some(sb) = frame.simbox.as_ref() {
        let lengths = sb.lengths();
        writeln!(writer, "_cell_length_a    {:.6}", lengths[0])?;
        writeln!(writer, "_cell_length_b    {:.6}", lengths[1])?;
        writeln!(writer, "_cell_length_c    {:.6}", lengths[2])?;
        // Compute angles from H columns.
        let h = sb.h_view();
        let v1 = [h[(0, 0)], h[(1, 0)], h[(2, 0)]];
        let v2 = [h[(0, 1)], h[(1, 1)], h[(2, 1)]];
        let v3 = [h[(0, 2)], h[(1, 2)], h[(2, 2)]];
        let alpha = angle_deg(&v2, &v3);
        let beta = angle_deg(&v1, &v3);
        let gamma = angle_deg(&v1, &v2);
        writeln!(writer, "_cell_angle_alpha {:.6}", alpha)?;
        writeln!(writer, "_cell_angle_beta  {:.6}", beta)?;
        writeln!(writer, "_cell_angle_gamma {:.6}", gamma)?;
    }
    writeln!(writer)?;
    writeln!(writer, "loop_")?;
    writeln!(writer, "_atom_site_label")?;
    writeln!(writer, "_atom_site_type_symbol")?;
    writeln!(writer, "_atom_site_Cartn_x")?;
    writeln!(writer, "_atom_site_Cartn_y")?;
    writeln!(writer, "_atom_site_Cartn_z")?;

    let xs = atoms
        .get_float("x")
        .ok_or_else(|| invalid_data("atoms.x missing"))?;
    let ys = atoms
        .get_float("y")
        .ok_or_else(|| invalid_data("atoms.y missing"))?;
    let zs = atoms
        .get_float("z")
        .ok_or_else(|| invalid_data("atoms.z missing"))?;
    let labels = atoms.get_string("label");
    let symbols = atoms.get_string("symbol");

    for i in 0..n {
        let label = labels
            .map(|c| c[[i]].clone())
            .unwrap_or_else(|| format!("X{}", i + 1));
        let sym = symbols.map(|c| c[[i]].clone()).unwrap_or("X".to_string());
        writeln!(
            writer,
            "{} {} {:.6} {:.6} {:.6}",
            label,
            sym,
            xs[[i]],
            ys[[i]],
            zs[[i]]
        )?;
    }
    Ok(())
}

fn sanitise_data_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}

fn angle_deg(u: &[F; 3], v: &[F; 3]) -> F {
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let c = (dot / (nu * nv)).clamp(-1.0, 1.0);
    c.acos() * 180.0 / std::f64::consts::PI
}

/// `FrameWriter`-trait wrapper.
pub struct CifFrameWriter<W: Write> {
    writer: W,
}

impl<W: Write> Writer for CifFrameWriter<W> {
    type W = W;
    type FrameLike = Frame;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for CifFrameWriter<W> {
    fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        write_cif_frame(&mut self.writer, frame)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SMALL_CIF: &str = "\
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
C2 C 0.5 0.5 0.5
";

    #[test]
    fn reads_small_cif() {
        let mut reader = CifReader::new(Cursor::new(SMALL_CIF.as_bytes()));
        let frame = reader.read_frame().unwrap().unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(2));
        let xs = atoms.get_float("x").unwrap();
        assert!((xs[[1]] - 2.5).abs() < 1e-9, "got {}", xs[[1]]);
        assert!(frame.simbox.is_some());
    }

    #[test]
    fn round_trip_small_cif() {
        let frame = {
            let mut reader = CifReader::new(Cursor::new(SMALL_CIF.as_bytes()));
            reader.read_frame().unwrap().unwrap()
        };
        let mut buf = Vec::new();
        write_cif_frame(&mut buf, &frame).unwrap();
        let mut reader2 = CifReader::new(Cursor::new(&buf));
        let frame2 = reader2.read_frame().unwrap().unwrap();
        let xs1 = frame.get("atoms").unwrap().get_float("x").unwrap();
        let xs2 = frame2.get("atoms").unwrap().get_float("x").unwrap();
        for i in 0..xs1.len() {
            assert!((xs1[[i]] - xs2[[i]]).abs() < 1e-4);
        }
    }

    #[test]
    fn esd_strip() {
        assert_eq!(strip_esd("5.917(3)"), "5.917");
        assert_eq!(strip_esd("90.000"), "90.000");
    }
}
