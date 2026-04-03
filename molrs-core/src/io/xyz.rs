use crate::block::Block;
use crate::frame::Frame;
use crate::io::reader::{FrameIndex, FrameReader, Reader, TrajReader};
use crate::io::writer::{FrameWriter, Writer};
use crate::region::simbox::SimBox;
use crate::types::{F, I};
use ndarray::{Array1, Array2, ArrayD};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::io::{BufRead, Seek, SeekFrom, Write};

// XYZ now produces a core::Frame consisting of blocks of NdArray columns

/// Primitive value in extended XYZ comment
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    Str(String),
    Int(i64),
    Real(f64),
    Logical(bool),
}

/// Extended value (nested arrays)
#[derive(Debug, Clone, PartialEq)]
pub enum ExtValue {
    Primitive(Primitive),
    Array1(Vec<Primitive>),
    Array2(Vec<Vec<Primitive>>),
}

/// Property type (S=string, I=int, R=real, L=logical)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropType {
    S,
    I,
    R,
    L,
}

/// Property specification: name, type and multiplicity
#[derive(Debug, Clone, PartialEq)]
pub struct PropertySpec {
    /// Property name
    pub name: String,
    /// Property type
    pub ty: PropType,
    /// Multiplicity (1 for scalar)
    pub m: usize,
}

/// Parsed XYZ comment line with optional extended fields
#[derive(Debug, Clone, PartialEq)]
pub struct XYZComment {
    /// Key-value pairs
    pub kv: HashMap<String, ExtValue>,
    /// Parsed properties (from key "Properties"), if present
    pub properties: Option<Vec<PropertySpec>>, // parsed from key "Properties" if present
    /// Original comment line when treated as extxyz
    pub comment: Option<String>, // original comment line when treated as extxyz
    /// True if treated as plain XYZ
    pub is_plain_xyz: bool,
}

fn parse_logical_token(tok: &str) -> Option<bool> {
    match tok.to_ascii_lowercase().as_str() {
        "t" | "true" => Some(true),
        "f" | "false" => Some(false),
        _ => None,
    }
}

fn parse_primitive_token(tok: &str) -> Primitive {
    if let Some(b) = parse_logical_token(tok) {
        Primitive::Logical(b)
    } else if tok.contains('.') || tok.contains('e') || tok.contains('E') {
        match tok.parse::<f64>() {
            Ok(v) => Primitive::Real(v),
            Err(_) => Primitive::Str(tok.to_string()),
        }
    } else {
        match tok.parse::<i64>() {
            Ok(v) => Primitive::Int(v),
            Err(_) => Primitive::Str(tok.to_string()),
        }
    }
}

fn parse_array_from_quoted(s: &str) -> ExtValue {
    // Try 2D using row separators ';' or '|' or comma between rows
    let has_row_sep = s.contains(';') || s.contains('|') || s.contains('\n');
    if has_row_sep {
        let rows: Vec<Vec<Primitive>> = s
            .split([';', '|', '\n'])
            .filter(|row| !row.trim().is_empty())
            .map(|row| row.split_whitespace().map(parse_primitive_token).collect())
            .collect();
        return ExtValue::Array2(rows);
    }

    // Try comma-separated values or whitespace-separated
    let elements: Vec<&str> = if s.contains(',') {
        s.split(',').collect()
    } else {
        s.split_whitespace().collect()
    };
    if elements.len() > 1 {
        let vals = elements
            .into_iter()
            .map(|t| parse_primitive_token(t.trim()))
            .collect();
        ExtValue::Array1(vals)
    } else {
        ExtValue::Primitive(Primitive::Str(s.to_string()))
    }
}

fn parse_properties(spec: &str) -> Option<Vec<PropertySpec>> {
    // Expect a colon-separated stream of triplets: name:T:m: name2:T:m: ...
    let parts: Vec<&str> = spec.split(':').filter(|s| !s.is_empty()).collect();
    if parts.len() < 3 || parts.len() % 3 != 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i + 2 < parts.len() {
        let name = parts[i].to_string();
        let ty = match parts[i + 1] {
            "S" => PropType::S,
            "I" => PropType::I,
            "R" => PropType::R,
            "L" => PropType::L,
            other => {
                // try tolerate lower-case
                match other.to_ascii_uppercase().as_str() {
                    "S" => PropType::S,
                    "I" => PropType::I,
                    "R" => PropType::R,
                    "L" => PropType::L,
                    _ => return None,
                }
            }
        };
        let m = match parts[i + 2].parse::<usize>() {
            Ok(v) if v > 0 => v,
            _ => return None,
        };
        out.push(PropertySpec { name, ty, m });
        i += 3;
    }
    Some(out)
}

pub fn parse_comment_line(line: &str) -> std::result::Result<XYZComment, String> {
    let original = line.to_string();
    let input = line.trim();

    // Quick check: if no '=' present, treat as plain xyz comment
    if !input.contains('=') {
        let mut kv = HashMap::new();
        kv.insert(
            "comment".to_string(),
            ExtValue::Primitive(Primitive::Str(original.clone())),
        );
        return Ok(XYZComment {
            kv,
            properties: None,
            comment: None,
            is_plain_xyz: true,
        });
    }

    let bytes = input.as_bytes();
    let mut idx = 0usize;
    let len = bytes.len();
    let mut kv: HashMap<String, ExtValue> = HashMap::new();
    let mut properties: Option<Vec<PropertySpec>> = None;

    let skip_ws = |pos: &mut usize| {
        while *pos < len && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
    };

    while idx < len {
        skip_ws(&mut idx);
        if idx >= len {
            break;
        }

        let key = if bytes[idx] == b'"' {
            idx += 1;
            let start = idx;
            while idx < len && bytes[idx] != b'"' {
                idx += 1;
            }
            if idx >= len {
                return Err("unterminated quoted key".to_string());
            }
            let key = input[start..idx].to_string();
            idx += 1;
            key
        } else {
            let start = idx;
            while idx < len && !bytes[idx].is_ascii_whitespace() && bytes[idx] != b'=' {
                idx += 1;
            }
            if start == idx {
                return Err("missing key".to_string());
            }
            input[start..idx].to_string()
        };

        skip_ws(&mut idx);
        if idx >= len || bytes[idx] != b'=' {
            // Bare boolean key (no '=' follows) — valid EXTXYZ, treat as true.
            kv.insert(key, ExtValue::Primitive(Primitive::Logical(true)));
            continue;
        }
        idx += 1;
        skip_ws(&mut idx);
        if idx >= len {
            return Err(format!("missing value for key '{key}'"));
        }

        let value = if bytes[idx] == b'"' {
            idx += 1;
            let start = idx;
            while idx < len && bytes[idx] != b'"' {
                idx += 1;
            }
            if idx >= len {
                return Err(format!("unterminated quoted value for key '{key}'"));
            }
            let value_str = &input[start..idx];
            idx += 1;
            parse_array_from_quoted(value_str)
        } else {
            let start = idx;
            while idx < len && !bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            let token = &input[start..idx];
            ExtValue::Primitive(parse_primitive_token(token))
        };

        if key.eq_ignore_ascii_case("properties") {
            let spec_str = match &value {
                ExtValue::Primitive(Primitive::Str(s)) => s.clone(),
                ExtValue::Array1(vs) => vs
                    .iter()
                    .map(|p| match p {
                        Primitive::Str(s) => s.clone(),
                        _ => "".into(),
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => String::new(),
            };
            properties = parse_properties(&spec_str);
        }
        kv.insert(key, value);
    }

    if properties.is_none() {
        kv.insert(
            "comment".to_string(),
            ExtValue::Primitive(Primitive::Str(original.clone())),
        );
        Ok(XYZComment {
            kv,
            properties: None,
            comment: None,
            is_plain_xyz: true,
        })
    } else {
        Ok(XYZComment {
            kv,
            properties,
            comment: Some(original),
            is_plain_xyz: false,
        })
    }
}

fn expand_property_columns(props: &[PropertySpec]) -> Vec<(String, PropType)> {
    let mut cols = Vec::new();
    for p in props {
        if p.m == 1 {
            cols.push((p.name.clone(), p.ty));
        } else {
            // Special-case: map pos:R:3 -> x,y,z (LAMMPS naming)
            if p.name.eq_ignore_ascii_case("pos") && p.ty == PropType::R && p.m == 3 {
                cols.push(("x".to_string(), PropType::R));
                cols.push(("y".to_string(), PropType::R));
                cols.push(("z".to_string(), PropType::R));
            } else {
                for i in 0..p.m {
                    cols.push((format!("{}_{}", p.name, i + 1), p.ty));
                }
            }
        }
    }
    cols
}

fn parse_bool_token(tok: &str) -> Option<bool> {
    match tok.to_ascii_lowercase().as_str() {
        "t" | "true" => Some(true),
        "f" | "false" => Some(false),
        _ => None,
    }
}

fn line_to_tokens(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

/// Build schema from parsed properties
fn build_complete_schema(ec: &XYZComment) -> Vec<PropertySpec> {
    // If no Properties key, return plain XYZ schema (4 columns: element, x, y, z)
    // Otherwise, return the properties as-is
    ec.properties.clone().unwrap_or_else(|| {
        vec![
            PropertySpec {
                name: "element".into(),
                ty: PropType::S,
                m: 1,
            },
            PropertySpec {
                name: "x".into(),
                ty: PropType::R,
                m: 1,
            },
            PropertySpec {
                name: "y".into(),
                ty: PropType::R,
                m: 1,
            },
            PropertySpec {
                name: "z".into(),
                ty: PropType::R,
                m: 1,
            },
        ]
    })
}

fn build_block_from_props(
    n: usize,
    lines: &[String],
    props: &[PropertySpec],
) -> Result<Block, String> {
    let cols = expand_property_columns(props);
    let m_total = cols.len();
    if lines.len() != n {
        return Err("insufficient atom lines".into());
    }

    // Prepare column buffers by type (use compile-time float for real values)
    enum ColBuf {
        S(Vec<String>),
        I(Vec<I>),
        R(Vec<F>),
        L(Vec<bool>),
    }
    let mut buffers: Vec<ColBuf> = cols
        .iter()
        .map(|(_, t)| match t {
            PropType::S => ColBuf::S(Vec::with_capacity(n)),
            PropType::I => ColBuf::I(Vec::with_capacity(n)),
            PropType::R => ColBuf::R(Vec::with_capacity(n)),
            PropType::L => ColBuf::L(Vec::with_capacity(n)),
        })
        .collect();

    for (row_i, line) in lines.iter().enumerate() {
        let toks = line_to_tokens(line);
        if toks.len() < m_total {
            return Err(format!(
                "line {}: expected at least {} tokens, got {}",
                row_i,
                m_total,
                toks.len()
            ));
        }
        // Iterate over props but push into flattened buffers
        for (buf_idx, (_, ty)) in cols.iter().enumerate() {
            let tok = toks[buf_idx];
            match (&mut buffers[buf_idx], ty) {
                (ColBuf::S(v), PropType::S) => v.push(tok.to_string()),
                (ColBuf::I(v), PropType::I) => v.push(tok.parse::<I>().map_err(|_| {
                    format!("line {} col {}: invalid int '{}", row_i, buf_idx, tok)
                })?),
                (ColBuf::R(v), PropType::R) => v.push(tok.parse::<F>().map_err(|_| {
                    format!("line {} col {}: invalid float '{}", row_i, buf_idx, tok)
                })?),
                (ColBuf::L(v), PropType::L) => v.push(parse_bool_token(tok).ok_or_else(|| {
                    format!("line {} col {}: invalid bool '{}", row_i, buf_idx, tok)
                })?),
                _ => return Err(format!("type mismatch at line {} col {}", row_i, buf_idx)),
            }
        }
    }

    // Assemble core::Block: drop string columns (S) as Block stores numeric/boolean arrays only
    let mut block = Block::new();
    for ((name, ty), buf) in cols.into_iter().zip(buffers.into_iter()) {
        match (ty, buf) {
            (PropType::I, ColBuf::I(v)) => {
                // Store as i64
                let arr = Array1::from_vec(v).into_dyn();
                block.insert(name, arr).map_err(|e| e.to_string())?;
            }
            (PropType::R, ColBuf::R(v)) => {
                // Store as float
                let arr: ArrayD<F> = Array1::from_vec(v).into_dyn();
                block.insert(name, arr).map_err(|e| e.to_string())?;
            }
            (PropType::L, ColBuf::L(v)) => {
                // Store as bool
                let arr = Array1::from_vec(v).into_dyn();
                block.insert(name, arr).map_err(|e| e.to_string())?;
            }
            (PropType::S, ColBuf::S(v)) => {
                // Store as String
                let arr = Array1::from_vec(v).into_dyn();
                block.insert(name, arr).map_err(|e| e.to_string())?;
            }
            _ => { /* type mismatch shouldn't happen due to construction; skip */ }
        }
    }

    Ok(block)
}

/// Parse 9 floats from an ExtValue into a 3×3 H matrix.
fn parse_lattice_values(v: &ExtValue) -> Option<Vec<F>> {
    let values: Vec<F> = match v {
        ExtValue::Array1(vals) => vals
            .iter()
            .filter_map(|p| match p {
                Primitive::Real(r) => Some(*r as F),
                Primitive::Int(i) => Some(*i as F),
                _ => None,
            })
            .collect(),
        ExtValue::Primitive(Primitive::Str(s)) => s
            .split_whitespace()
            .filter_map(|tok| tok.parse::<F>().ok())
            .collect(),
        _ => return None,
    };
    if values.len() == 9 {
        Some(values)
    } else {
        None
    }
}

/// Parse 3 floats from an Origin ExtValue.
fn parse_origin_values(v: &ExtValue) -> Option<[F; 3]> {
    let values: Vec<F> = match v {
        ExtValue::Array1(vals) => vals
            .iter()
            .filter_map(|p| match p {
                Primitive::Real(r) => Some(*r as F),
                Primitive::Int(i) => Some(*i as F),
                _ => None,
            })
            .collect(),
        ExtValue::Primitive(Primitive::Str(s)) => s
            .split_whitespace()
            .filter_map(|tok| tok.parse::<F>().ok())
            .collect(),
        _ => return None,
    };
    if values.len() == 3 {
        Some([values[0], values[1], values[2]])
    } else {
        None
    }
}

/// Build a SimBox from Lattice + optional Origin ExtValues.
///
/// `Origin` defaults to `[0, 0, 0]` when absent, matching the extxyz convention.
fn parse_simbox(lattice: &ExtValue, origin: Option<&ExtValue>) -> Option<SimBox> {
    let h_vals = parse_lattice_values(lattice)?;
    let h = Array2::from_shape_vec((3, 3), h_vals).ok()?;
    let origin_arr = origin
        .and_then(parse_origin_values)
        .map(|o| ndarray::array![o[0], o[1], o[2]])
        .unwrap_or_else(|| ndarray::array![0.0 as F, 0.0, 0.0]);
    SimBox::new(h, origin_arr, [true, true, true]).ok()
}

/// Read one XYZ/EXTXYZ frame from the current position of a buffered reader.
/// Returns Ok(None) on EOF before the first line.
pub fn read_xyz_frame_from_reader<R: BufRead>(reader: &mut R) -> std::io::Result<Option<Frame>> {
    // Read first non-empty line as atom count
    let mut line = String::new();
    let n = loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None); // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed.parse::<usize>() {
            Ok(v) => break v,
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid atom count line: {}", trimmed),
                ));
            }
        }
    };

    // Read comment line (can be empty)
    line.clear();
    let _ = reader.read_line(&mut line)?; // if EOF after count, it's malformed but we allow empty
    let comment = line.trim_end_matches(['\r', '\n']);

    // Parse comment to metadata and properties
    let ec = parse_comment_line(comment)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut kv_meta: HashMap<String, ExtValue> = HashMap::new();
    let mut lattice_value: Option<ExtValue> = None;
    let mut origin_value: Option<ExtValue> = None;
    for (k, v) in ec.kv.iter() {
        if k.eq_ignore_ascii_case("Properties") {
            continue;
        }
        if k.eq_ignore_ascii_case("Lattice") {
            lattice_value = Some(v.clone());
            continue;
        }
        if k.eq_ignore_ascii_case("Origin") {
            origin_value = Some(v.clone());
            continue;
        }
        kv_meta.insert(k.clone(), v.clone());
    }

    // Read N atom lines
    let mut atom_lines: Vec<String> = Vec::with_capacity(n);
    for _ in 0..n {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "unexpected EOF in atom lines",
            ));
        }
        atom_lines.push(line.trim_end_matches(['\r', '\n']).to_string());
    }

    // Build complete schema (base properties + derived columns)
    let schema = build_complete_schema(&ec);

    // Parse columns according to schema -> atoms block
    let atoms_block = build_block_from_props(n, &atom_lines, &schema)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut frame = Frame::new();
    frame.insert("atoms", atoms_block);
    // Stringify metadata into frame.meta
    for (k, v) in kv_meta.into_iter() {
        frame.meta.insert(k, ext_value_to_string(&v));
    }

    // Build SimBox from Lattice + optional Origin
    if let Some(ref lat) = lattice_value
        && let Some(simbox) = parse_simbox(lat, origin_value.as_ref())
    {
        frame.simbox = Some(simbox);
    }

    Ok(Some(frame))
}

// =============== Winnow-based frame parser for a complete frame string ===============

/// Parse a complete XYZ/EXTXYZ frame from a &str using line-by-line parsing.
pub fn parse_xyz_frame_str(s: &str) -> std::result::Result<Frame, String> {
    let mut lines = s.lines();

    let count_line = lines
        .next()
        .ok_or_else(|| "missing atom count line".to_string())?;
    let n = count_line
        .trim()
        .parse::<usize>()
        .map_err(|e| format!("parse atom count: {e}"))?;

    let comment = lines
        .next()
        .ok_or_else(|| "missing comment line".to_string())?
        .to_string();

    let mut atom_lines = Vec::with_capacity(n);
    for _ in 0..n {
        let line = lines
            .next()
            .ok_or_else(|| "insufficient atom lines".to_string())?;
        atom_lines.push(line.to_string());
    }

    // Parse comment to metadata and properties
    let ec = parse_comment_line(&comment)?;
    let mut kv_meta: HashMap<String, ExtValue> = HashMap::new();
    let mut lattice_value: Option<ExtValue> = None;
    let mut origin_value: Option<ExtValue> = None;
    for (k, v) in ec.kv.iter() {
        if k.eq_ignore_ascii_case("Properties") {
            continue;
        }
        if k.eq_ignore_ascii_case("Lattice") {
            lattice_value = Some(v.clone());
            continue;
        }
        if k.eq_ignore_ascii_case("Origin") {
            origin_value = Some(v.clone());
            continue;
        }
        kv_meta.insert(k.clone(), v.clone());
    }

    // Build complete schema (base properties + derived columns)
    let schema = build_complete_schema(&ec);

    // Parse columns according to schema
    let atoms_block = build_block_from_props(n, &atom_lines, &schema)?;
    let mut frame = Frame::new();
    frame.insert("atoms", atoms_block);
    for (k, v) in kv_meta.into_iter() {
        frame.meta.insert(k, ext_value_to_string(&v));
    }

    // Build SimBox from Lattice + optional Origin
    if let Some(ref lat) = lattice_value
        && let Some(simbox) = parse_simbox(lat, origin_value.as_ref())
    {
        frame.simbox = Some(simbox);
    }

    Ok(frame)
}

fn ext_value_to_string(v: &ExtValue) -> String {
    match v {
        ExtValue::Primitive(Primitive::Str(s)) => s.clone(),
        ExtValue::Primitive(Primitive::Int(i)) => i.to_string(),
        ExtValue::Primitive(Primitive::Real(r)) => format!("{}", r),
        ExtValue::Primitive(Primitive::Logical(b)) => b.to_string(),
        ExtValue::Array1(vals) => vals
            .iter()
            .map(|p| match p {
                Primitive::Str(s) => s.clone(),
                Primitive::Int(i) => i.to_string(),
                Primitive::Real(r) => format!("{}", r),
                Primitive::Logical(b) => b.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" "),
        ExtValue::Array2(rows) => rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|p| match p {
                        Primitive::Str(s) => s.clone(),
                        Primitive::Int(i) => i.to_string(),
                        Primitive::Real(r) => format!("{}", r),
                        Primitive::Logical(b) => b.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("; "),
    }
}

// =============== XYZReader ===============

/// Unified XYZ/ExtXYZ reader treating all files as trajectories
///
/// Single-frame files are treated as 1-step trajectories. This reader
/// implements lazy indexing: the first `read_step(0)` call reads immediately
/// without scanning the file, while accessing later frames triggers index
/// building for efficient random access.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::xyz::XYZReader;
/// use molrs::io::reader::TrajReader;
/// use std::io::BufReader;
/// use std::fs::File;
///
/// # fn main() -> std::io::Result<()> {
/// let file = File::open("trajectory.xyz")?;
/// let mut reader = XYZReader::new(BufReader::new(file));
///
/// // Read first frame (no indexing)
/// if let Some(frame) = reader.read_step(0)? {
///     println!("First frame loaded");
/// }
///
/// // Read frame 5 (triggers indexing)
/// if let Some(frame) = reader.read_step(5)? {
///     println!("Frame 5 loaded");
/// }
///
/// // Iterate all frames
/// for result in reader.iter() {
///     let frame = result?;
///     // Process frame
/// }
/// # Ok(())
/// # }
/// ```
pub struct XYZReader<R: BufRead> {
    reader: R,
    index: OnceCell<FrameIndex>,
}

impl<R: BufRead + Seek> XYZReader<R> {
    /// Create a new XYZ reader from a buffered reader
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: OnceCell::new(),
        }
    }

    /// Build frame index by scanning the entire file
    fn build_index(&mut self) -> std::io::Result<()> {
        if self.index.get().is_some() {
            return Ok(()); // Already built
        }

        let mut frame_index = FrameIndex::new();

        // Try to seek to beginning if seekable
        let start_pos = if let Ok(pos) = self.reader.stream_position() {
            // Seekable stream - record starting position
            self.reader.seek(SeekFrom::Start(0))?;
            Some(pos)
        } else {
            // Non-seekable stream (e.g., gzip) - can only scan forward
            None
        };

        let mut current_pos: u64 = 0;
        let mut line = String::new();

        loop {
            // Record frame start position
            frame_index.add_frame(current_pos);

            // Read atom count
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                // Remove the last empty frame we just added
                if !frame_index.is_empty()
                    && frame_index.offsets[frame_index.len() - 1] == current_pos
                {
                    frame_index.offsets.pop();
                }
                break; // EOF
            }
            current_pos += bytes as u64;

            let n = line.trim().parse::<usize>().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid atom count: {}", line.trim()),
                )
            })?;

            // Skip comment line
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            current_pos += bytes as u64;

            // Skip N atom lines
            for _ in 0..n {
                line.clear();
                let bytes = self.reader.read_line(&mut line)?;
                if bytes == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "unexpected EOF in atom lines",
                    ));
                }
                current_pos += bytes as u64;
            }
        }

        // Restore original position if seekable
        if let Some(pos) = start_pos {
            self.reader.seek(SeekFrom::Start(pos))?;
        }

        self.index
            .set(frame_index)
            .map_err(|_| std::io::Error::other("failed to set index"))?;

        Ok(())
    }

    /// Read frame at specific offset
    fn read_at_offset(&mut self, offset: u64) -> std::io::Result<Option<Frame>> {
        self.reader.seek(SeekFrom::Start(offset))?;
        read_xyz_frame_from_reader(&mut self.reader)
    }
}

impl<R: BufRead + Seek> Reader for XYZReader<R> {
    type R = R;
    type Frame = Frame;

    fn new(reader: Self::R) -> Self {
        Self {
            reader,
            index: OnceCell::new(),
        }
    }
}

impl<R: BufRead + Seek> FrameReader for XYZReader<R> {
    fn read_frame(&mut self) -> std::io::Result<Option<Self::Frame>> {
        // Always read the first frame from the current stream position.
        read_xyz_frame_from_reader(&mut self.reader)
    }
}

impl<R: BufRead + Seek> TrajReader for XYZReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        self.build_index()
    }

    fn read_step(&mut self, step: usize) -> std::io::Result<Option<Self::Frame>> {
        // Fast path for first frame: read immediately without indexing
        if step == 0
            && self.index.get().is_none()
            && let Ok(start_pos) = self.reader.stream_position()
        {
            let frame = read_xyz_frame_from_reader(&mut self.reader)?;
            self.reader.seek(SeekFrom::Start(start_pos))?;
            return Ok(frame);
        }

        if self.index.get().is_none() {
            self.build_index()?;
        }

        let index = self.index.get().unwrap();
        if step >= index.len() {
            return Ok(None);
        }

        let offset = index.get(step).unwrap();
        self.read_at_offset(offset)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        if self.index.get().is_none() {
            self.build_index()?;
        }
        Ok(self.index.get().unwrap().len())
    }
}

/// Read a single frame from an XYZ file
///
/// # Examples
///
/// ```no_run
/// use molrs::io::xyz::read_xyz_frame;
///
/// # fn main() -> std::io::Result<()> {
/// let frame = read_xyz_frame("water.xyz")?;
/// println!("Loaded {} atoms", frame.get("atoms").map(|b| b.nrows().unwrap_or(0)).unwrap_or(0));
/// # Ok(())
/// # }
/// ```
pub fn read_xyz_frame<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Frame> {
    use crate::io::reader::open_file;
    let reader = open_file(path)?;
    let mut xyz_reader = XYZReader::new(reader);
    xyz_reader
        .read_step(0)?
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "empty file"))
}

/// Read all frames from an XYZ trajectory file
///
/// # Examples
///
/// ```no_run
/// use molrs::io::xyz::read_xyz_traj;
///
/// # fn main() -> std::io::Result<()> {
/// let frames = read_xyz_traj("md_run.xyz")?;
/// println!("Loaded {} frames", frames.len());
/// # Ok(())
/// # }
/// ```
pub fn read_xyz_traj<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Vec<Frame>> {
    use crate::io::reader::open_file;
    let reader = open_file(path)?;
    let mut xyz_reader = XYZReader::new(reader);
    xyz_reader.iter().collect()
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn parse_properties_triplets() {
        let line = "Properties=species:S:1:pos:R:3:mass:R:1";
        let ec = parse_comment_line(line).expect("parse");
        assert!(ec.properties.is_some());
        let props = ec.properties.unwrap();
        assert_eq!(props.len(), 3);
        assert_eq!(
            props[0],
            PropertySpec {
                name: "species".into(),
                ty: PropType::S,
                m: 1
            }
        );
        assert_eq!(
            props[1],
            PropertySpec {
                name: "pos".into(),
                ty: PropType::R,
                m: 3
            }
        );
        assert_eq!(
            props[2],
            PropertySpec {
                name: "mass".into(),
                ty: PropType::R,
                m: 1
            }
        );
        assert!(!ec.is_plain_xyz);
    }

    #[test]
    fn parse_comment_with_properties() {
        let line = r#"Lattice="8.43116035 0.0 0.0 0.158219155128 14.5042431863 0.0 1.16980663624 4.4685149855 14.9100096405" Properties=species:S:1:pos:R:3:CS:R:2 ENERGY=-2069.84934116 Natoms=192 NAME=COBHUW"#;
        let ec = parse_comment_line(line).expect("parse");
        assert!(ec.properties.is_some());
        assert!(!ec.is_plain_xyz);
        // Lattice
        match ec.kv.get("Lattice").unwrap() {
            ExtValue::Array1(v) => {
                assert_eq!(v.len(), 9);
                assert!(matches!(v[0], Primitive::Real(_)) || matches!(v[0], Primitive::Int(_)));
            }
            other => panic!("unexpected Lattice value: {other:?}"),
        }
        // energy
        match ec.kv.get("ENERGY").unwrap() {
            ExtValue::Primitive(Primitive::Real(x)) => assert!((x - -2069.84934116).abs() < 1e-6),
            other => panic!("unexpected energy value: {other:?}"),
        }
    }

    #[test]
    fn test_xyz_invalid_atom_count() {
        use std::io::Cursor;

        let data = b"abc\nComment\nH 0 0 0\n";
        let mut cursor = Cursor::new(&data[..]);
        let err = read_xyz_frame_from_reader(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

}

// =============== XYZFrameWriter ===============

/// A writer for XYZ/EXTXYZ single-frame files
pub struct XYZFrameWriter<W: Write> {
    writer: W,
}

impl<W: Write> Writer for XYZFrameWriter<W> {
    type W = W;
    type FrameLike = Frame;

    fn new(writer: Self::W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for XYZFrameWriter<W> {
    fn write_frame(&mut self, frame: &Self::FrameLike) -> std::io::Result<()> {
        write_xyz_frame(&mut self.writer, frame)
    }
}

/// Write a single frame to the writer in Extended XYZ format
pub fn write_xyz_frame<W: Write>(writer: &mut W, frame: &Frame) -> std::io::Result<()> {
    // 1. Get atoms block
    let atoms = match frame.get("atoms") {
        Some(b) => b,
        None => {
            // Write 0 atoms
            writeln!(writer, "0")?;
            writeln!(writer)?;
            return Ok(());
        }
    };

    let n = atoms.nrows().unwrap_or(0);
    writeln!(writer, "{}", n)?;

    // 2. Construct Properties string and comment
    // Collect keys and sort them to be deterministic.
    // Prioritize x, y, z.
    let mut keys: Vec<&str> = atoms.keys().collect();
    keys.sort_by(|a, b| {
        // Custom sort: x, y, z first, then alphabetical
        let rank_a = match *a {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => 3,
        };
        let rank_b = match *b {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => 3,
        };
        if rank_a != rank_b {
            rank_a.cmp(&rank_b)
        } else {
            a.cmp(b)
        }
    });

    let mut props_parts = Vec::new();
    // Order: species, pos, id, mol, then others
    props_parts.push("species:S:1".to_string());

    // Check if x, y, z exist and add pos
    let has_xyz = keys.contains(&"x") && keys.contains(&"y") && keys.contains(&"z");
    if has_xyz {
        props_parts.push("pos:R:3".to_string());
    }

    // Add other properties in a specific order: id, mol, then rest
    let priority_keys = vec!["id", "mol"];
    for priority_k in &priority_keys {
        if keys.contains(priority_k)
            && let Some(col) = atoms.get(priority_k)
        {
            let shape = col.shape();
            let m: usize = shape.iter().skip(1).product();
            let m = if m == 0 { 1 } else { m };
            let type_char = match col.dtype() {
                crate::block::DType::Float => "R",
                crate::block::DType::Int | crate::block::DType::UInt | crate::block::DType::U8 => {
                    "I"
                }
                crate::block::DType::Bool => "L",
                crate::block::DType::String => "S",
            };
            props_parts.push(format!("{}:{}:{}", priority_k, type_char, m));
        }
    }

    // Add remaining keys (excluding x, y, z, element, and already added priority keys)
    for k in &keys {
        if *k == "x" || *k == "y" || *k == "z" || *k == "element" || priority_keys.contains(k) {
            continue;
        }
        if let Some(col) = atoms.get(k) {
            let shape = col.shape();
            let m: usize = shape.iter().skip(1).product();
            let m = if m == 0 { 1 } else { m };
            let type_char = match col.dtype() {
                crate::block::DType::Float => "R",
                crate::block::DType::Int | crate::block::DType::UInt | crate::block::DType::U8 => {
                    "I"
                }
                crate::block::DType::Bool => "L",
                crate::block::DType::String => "S",
            };
            props_parts.push(format!("{}:{}:{}", k, type_char, m));
        }
    }
    let properties_str = props_parts.join(":");

    // Construct full comment line
    let mut comment_parts = Vec::new();
    // Add Lattice (and Origin when non-zero) from SimBox if present
    if let Some(simbox) = &frame.simbox {
        let h = simbox.h_view();
        let mut lattice_values = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                lattice_values.push(format!("{}", h[[i, j]]));
            }
        }
        comment_parts.push(format!("Lattice=\"{}\"", lattice_values.join(" ")));

        let o = simbox.origin_view();
        if o[0] != 0.0 || o[1] != 0.0 || o[2] != 0.0 {
            comment_parts.push(format!("Origin=\"{} {} {}\"", o[0], o[1], o[2]));
        }
    }
    // Add other meta
    for (k, v) in &frame.meta {
        if k == "Lattice" || k == "Origin" || k == "Properties" || k == "comment" || k == "elements"
        {
            continue;
        }
        let val_str = if v.contains(' ') {
            format!("\"{}\"", v)
        } else {
            v.clone()
        };
        comment_parts.push(format!("{}={}", k, val_str));
    }

    // Add Properties
    comment_parts.push(format!("Properties={}", properties_str));

    writeln!(writer, "{}", comment_parts.join(" "))?;

    // 3. Write atom lines
    // Helper to format a value from a column at row i
    use crate::block::DType;

    // Read element symbols from atoms["element"] column.
    let elements: Vec<String> = atoms
        .get_string("element")
        .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
        .unwrap_or_else(|| vec!["X".to_string(); n]);

    for i in 0..n {
        let mut line_parts = Vec::new();
        for k in &keys {
            // "element" is already written as the species column (first field)
            if *k == "element" {
                continue;
            }
            if let Some(col) = atoms.get(k) {
                // Extract values based on dtype
                match col.dtype() {
                    DType::Float => {
                        if let Some(arr) = col.as_float() {
                            let row = arr.index_axis(ndarray::Axis(0), i);
                            for val in row.iter() {
                                line_parts.push(format!("{}", val));
                            }
                        }
                    }
                    DType::Int => {
                        if let Some(arr) = col.as_int() {
                            let row = arr.index_axis(ndarray::Axis(0), i);
                            for val in row.iter() {
                                line_parts.push(format!("{}", val));
                            }
                        }
                    }
                    DType::Bool => {
                        if let Some(arr) = col.as_bool() {
                            let row = arr.index_axis(ndarray::Axis(0), i);
                            for val in row.iter() {
                                line_parts.push(if *val { "T" } else { "F" }.to_string());
                            }
                        }
                    }
                    DType::UInt => {
                        if let Some(arr) = col.as_uint() {
                            let row = arr.index_axis(ndarray::Axis(0), i);
                            for val in row.iter() {
                                line_parts.push(format!("{}", val));
                            }
                        }
                    }
                    DType::U8 => {
                        if let Some(arr) = col.as_u8() {
                            let row = arr.index_axis(ndarray::Axis(0), i);
                            for val in row.iter() {
                                line_parts.push(format!("{}", val));
                            }
                        }
                    }
                    DType::String => {
                        if let Some(arr) = col.as_string() {
                            let row = arr.index_axis(ndarray::Axis(0), i);
                            for val in row.iter() {
                                line_parts.push(val.clone());
                            }
                        }
                    }
                }
            }
        }
        // Prepend element symbol
        let species = elements.get(i).cloned().unwrap_or_else(|| "X".to_string());
        line_parts.insert(0, species);
        writeln!(writer, "{}", line_parts.join(" "))?;
    }

    Ok(())
}
