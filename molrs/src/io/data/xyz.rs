use crate::io::reader::{FrameIndex, FrameReader, Reader, TrajReader};
use crate::io::writer::{FrameWriter, Writer};
use molrs::spatial::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::store::frame_access::FrameAccess;
use molrs::store::meta::MetaValue;
use molrs::types::{F, I, U};
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
        let values: Vec<Primitive> = elements
            .into_iter()
            .map(|t| parse_primitive_token(t.trim()))
            .collect();
        if values
            .iter()
            .all(|value| matches!(value, Primitive::Str(_)))
        {
            ExtValue::Primitive(Primitive::Str(s.to_string()))
        } else {
            ExtValue::Array1(values)
        }
    } else {
        ExtValue::Primitive(Primitive::Str(s.to_string()))
    }
}

fn parse_properties(spec: &str) -> Option<Vec<PropertySpec>> {
    // Expect a colon-separated stream of triplets: name:T:m: name2:T:m: ...
    let parts: Vec<&str> = spec.split(':').filter(|s| !s.is_empty()).collect();
    if parts.len() < 3 || !parts.len().is_multiple_of(3) {
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
    for ((name, ty), buf) in cols.into_iter().zip(buffers) {
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

/// Parse the MolCrafts `Connct` XYZ comment extension.
///
/// The value is a flat, zero-based list of atom-index pairs, for example
/// `Connct="[0,1,0,2]"` describes bonds 0-1 and 0-2. Bond order is implicitly
/// one. Brackets are required by the public convention but are accepted
/// leniently here so older hand-written inputs remain readable.
fn parse_connct(value: &ExtValue, n_atoms: usize) -> Result<Vec<(U, U)>, String> {
    fn append_primitive(raw: &mut String, value: &Primitive) -> Result<(), String> {
        if !raw.is_empty() {
            raw.push(',');
        }
        match value {
            Primitive::Int(value) => raw.push_str(&value.to_string()),
            Primitive::Str(value) => raw.push_str(value),
            Primitive::Real(_) | Primitive::Logical(_) => {
                return Err("Connct accepts integer atom indices only".to_string());
            }
        }
        Ok(())
    }

    let mut raw = String::new();
    match value {
        ExtValue::Primitive(value) => append_primitive(&mut raw, value)?,
        ExtValue::Array1(values) => {
            for value in values {
                append_primitive(&mut raw, value)?;
            }
        }
        ExtValue::Array2(_) => return Err("Connct must be a flat list of atom indices".to_string()),
    }

    let indices = raw
        .split(|ch: char| ch == '[' || ch == ']' || ch == ',' || ch.is_whitespace())
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse::<U>()
                .map_err(|_| format!("Connct contains invalid atom index '{token}'"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if !indices.len().is_multiple_of(2) {
        return Err(format!(
            "Connct requires atom-index pairs, but received {} indices",
            indices.len()
        ));
    }

    let mut pairs = Vec::with_capacity(indices.len() / 2);
    for pair in indices.chunks_exact(2) {
        let (atomi, atomj) = (pair[0], pair[1]);
        if atomi as usize >= n_atoms || atomj as usize >= n_atoms {
            return Err(format!(
                "Connct bond {atomi}-{atomj} is out of range for {n_atoms} atoms"
            ));
        }
        pairs.push((atomi, atomj));
    }
    Ok(pairs)
}

fn connct_block(value: &ExtValue, n_atoms: usize) -> Result<Option<Block>, String> {
    let pairs = parse_connct(value, n_atoms)?;
    if pairs.is_empty() {
        return Ok(None);
    }

    let (atomi, atomj): (Vec<U>, Vec<U>) = pairs.into_iter().unzip();
    let mut block = Block::new();
    block
        .insert("atomi", Array1::from_vec(atomi).into_dyn())
        .map_err(|error| error.to_string())?;
    block
        .insert("atomj", Array1::from_vec(atomj).into_dyn())
        .map_err(|error| error.to_string())?;
    Ok(Some(block))
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
    let mut connct_value: Option<ExtValue> = None;
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
        if k.eq_ignore_ascii_case("Connct") {
            connct_value = Some(v.clone());
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
    if let Some(ref connct) = connct_value
        && let Some(bonds) = connct_block(connct, n)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
    {
        frame.insert("bonds", bonds);
    }
    for (k, v) in kv_meta.into_iter() {
        frame.meta.insert(
            k,
            ext_value_to_meta(v)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
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
    let mut connct_value: Option<ExtValue> = None;
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
        if k.eq_ignore_ascii_case("Connct") {
            connct_value = Some(v.clone());
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
    if let Some(ref connct) = connct_value
        && let Some(bonds) = connct_block(connct, n)?
    {
        frame.insert("bonds", bonds);
    }
    for (k, v) in kv_meta.into_iter() {
        frame.meta.insert(k, ext_value_to_meta(v)?);
    }

    // Build SimBox from Lattice + optional Origin
    if let Some(ref lat) = lattice_value
        && let Some(simbox) = parse_simbox(lat, origin_value.as_ref())
    {
        frame.simbox = Some(simbox);
    }

    Ok(frame)
}

fn ext_value_to_meta(value: ExtValue) -> std::result::Result<MetaValue, String> {
    match value {
        ExtValue::Primitive(Primitive::Str(value)) => Ok(MetaValue::String(value)),
        ExtValue::Primitive(Primitive::Int(value)) => Ok(MetaValue::I64(value)),
        ExtValue::Primitive(Primitive::Real(value)) => Ok(MetaValue::F64(value)),
        ExtValue::Primitive(Primitive::Logical(value)) => Ok(MetaValue::Bool(value)),
        ExtValue::Array2(rows) => ext_array_to_meta(rows.into_iter().flatten().collect()),
        ExtValue::Array1(values) => ext_array_to_meta(values),
    }
}

fn ext_array_to_meta(values: Vec<Primitive>) -> std::result::Result<MetaValue, String> {
    if values.len() == 3 && values.iter().all(|v| matches!(v, Primitive::Logical(_))) {
        let values: Vec<bool> = values
            .into_iter()
            .map(|v| match v {
                Primitive::Logical(v) => v,
                _ => unreachable!(),
            })
            .collect();
        return Ok(MetaValue::Bool3(values.try_into().unwrap()));
    }
    if values.len() == 3 && values.iter().all(|v| matches!(v, Primitive::Int(_))) {
        let values: Vec<i64> = values
            .into_iter()
            .map(|v| match v {
                Primitive::Int(v) => v,
                _ => unreachable!(),
            })
            .collect();
        return Ok(MetaValue::I64x3(values.try_into().unwrap()));
    }
    if matches!(values.len(), 3 | 6 | 9) && values.iter().all(|v| matches!(v, Primitive::Real(_))) {
        let values: Vec<f64> = values
            .into_iter()
            .map(|v| match v {
                Primitive::Real(v) => v,
                _ => unreachable!(),
            })
            .collect();
        return Ok(match values.len() {
            3 => MetaValue::F64x3(values.try_into().unwrap()),
            6 => MetaValue::F64x6(values.try_into().unwrap()),
            9 => MetaValue::F64x9(values.try_into().unwrap()),
            _ => unreachable!(),
        });
    }
    Err(format!(
        "extended XYZ metadata array has unsupported type/length {}; expected a homogeneous numeric vector of length 3, 6, or 9, or bool[3]",
        values.len()
    ))
}

fn meta_to_extxyz(value: &MetaValue) -> String {
    macro_rules! joined {
        ($values:expr) => {
            $values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        };
    }
    let raw = match value {
        MetaValue::Bool(v) => v.to_string(),
        MetaValue::I32(v) => v.to_string(),
        MetaValue::I64(v) => v.to_string(),
        MetaValue::U32(v) => v.to_string(),
        MetaValue::U64(v) => v.to_string(),
        MetaValue::F32(v) => v.to_string(),
        MetaValue::F64(v) => v.to_string(),
        MetaValue::String(v) => v.clone(),
        MetaValue::Bool3(v) => joined!(v),
        MetaValue::I32x3(v) => joined!(v),
        MetaValue::I64x3(v) => joined!(v),
        MetaValue::U32x3(v) => joined!(v),
        MetaValue::U64x3(v) => joined!(v),
        MetaValue::F32x3(v) => joined!(v),
        MetaValue::F64x3(v) => joined!(v),
        MetaValue::F32x6(v) => joined!(v),
        MetaValue::F64x6(v) => joined!(v),
        MetaValue::F32x9(v) => joined!(v),
        MetaValue::F64x9(v) => joined!(v),
    };
    if raw.contains(char::is_whitespace) {
        format!("\"{raw}\"")
    } else {
        raw
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
/// use molrs::io::data::xyz::XYZReader;
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

        'frames: loop {
            // Locate the next atom-count line. Blank lines between frames
            // (and trailing blanks at EOF) are skipped — same rule as
            // `read_xyz_frame_from_reader` and `XyzIndexBuilder` in the
            // AwaitingNatoms state. Recording a frame only after a valid
            // count avoids the old "invalid atom count: " crash on `\n`.
            let n = loop {
                let frame_start = current_pos;
                line.clear();
                let bytes = self.reader.read_line(&mut line)?;
                if bytes == 0 {
                    break 'frames; // EOF while searching for a frame
                }
                current_pos += bytes as u64;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match trimmed.parse::<usize>() {
                    Ok(v) => {
                        frame_index.add_frame(frame_start);
                        break v;
                    }
                    Err(_) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("invalid atom count: {}", trimmed),
                        ));
                    }
                }
            };

            // Skip comment line
            line.clear();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unexpected EOF after atom count (missing comment line)",
                ));
            }
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
/// use molrs::io::data::xyz::read_xyz_frame;
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
/// use molrs::io::data::xyz::read_xyz_traj;
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

// ============================================================================
// Streaming
// ============================================================================

use crate::io::streaming::{FrameIndexBuilder, FrameIndexEntry, LineAccumulator};
use std::io::Cursor;

/// Parse exactly one XYZ / Extended-XYZ frame from a tightly-bounded byte
/// slice. The slice must be a `[byte_offset, byte_offset + byte_len)` window
/// produced by [`XyzIndexBuilder`].
pub fn parse_frame_bytes(bytes: &[u8]) -> std::io::Result<Frame> {
    let mut cursor = Cursor::new(bytes);
    read_xyz_frame_from_reader(&mut cursor)?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "XYZ frame slice is empty",
        )
    })
}

/// Phase of the XYZ indexer state machine.
#[derive(Debug, Clone, Copy)]
enum XyzPhase {
    /// Looking for the natoms count line.
    AwaitingNatoms,
    /// natoms parsed; the next line is the comment line.
    AwaitingComment { natoms: usize },
    /// Consuming `remaining` atom lines.
    ConsumingAtoms { remaining: usize },
}

/// Streaming frame indexer for XYZ / Extended-XYZ files.
///
/// State machine: `AwaitingNatoms` → `AwaitingComment` → `ConsumingAtoms` →
/// emit and reset to `AwaitingNatoms`. Blank or non-integer lines in
/// `AwaitingNatoms` are skipped (matching the lenient
/// `read_xyz_frame_from_reader` behavior).
pub struct XyzIndexBuilder {
    lines: LineAccumulator,
    phase: XyzPhase,
    pending_frame_start: Option<u64>,
    pending_entries: Vec<FrameIndexEntry>,
    /// Total bytes scanned so far (line offset + line len of the most
    /// recently completed atom line) — needed to compute byte_len on
    /// frame completion.
    last_line_end: u64,
}

impl Default for XyzIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl XyzIndexBuilder {
    pub fn new() -> Self {
        Self {
            lines: LineAccumulator::new(),
            phase: XyzPhase::AwaitingNatoms,
            pending_frame_start: None,
            pending_entries: Vec::new(),
            last_line_end: 0,
        }
    }

    fn process_line(&mut self, line: &str, line_offset: u64, line_len: u32) -> std::io::Result<()> {
        let line_end = line_offset + line_len as u64;
        self.last_line_end = line_end;
        match self.phase {
            XyzPhase::AwaitingNatoms => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    // Blank line — skip in the AwaitingNatoms state, matching
                    // the legacy `read_xyz_frame_from_reader` semantics.
                    return Ok(());
                }
                let n = match trimmed.parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => {
                        // Garbage line in AwaitingNatoms is also skipped per
                        // spec ("blank/garbage lines in AwaitingNatoms state
                        // are skipped not crashed").
                        return Ok(());
                    }
                };
                self.pending_frame_start = Some(line_offset);
                self.phase = XyzPhase::AwaitingComment { natoms: n };
                Ok(())
            }
            XyzPhase::AwaitingComment { natoms } => {
                if natoms == 0 {
                    // 0-atom frame — emit now, spanning the count + comment.
                    let start = self.pending_frame_start.take().unwrap_or(line_offset);
                    let span = line_end - start;
                    self.push_entry(start, span)?;
                    self.phase = XyzPhase::AwaitingNatoms;
                } else {
                    self.phase = XyzPhase::ConsumingAtoms { remaining: natoms };
                }
                Ok(())
            }
            XyzPhase::ConsumingAtoms { remaining } => {
                let new_remaining = remaining - 1;
                if new_remaining == 0 {
                    let start = self.pending_frame_start.take().unwrap_or(line_offset);
                    let span = line_end - start;
                    self.push_entry(start, span)?;
                    self.phase = XyzPhase::AwaitingNatoms;
                } else {
                    self.phase = XyzPhase::ConsumingAtoms {
                        remaining: new_remaining,
                    };
                }
                Ok(())
            }
        }
    }

    fn push_entry(&mut self, byte_offset: u64, span: u64) -> std::io::Result<()> {
        if span > u32::MAX as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "XYZ frame size exceeds 4 GiB",
            ));
        }
        self.pending_entries.push(FrameIndexEntry {
            byte_offset,
            byte_len: span as u32,
        });
        Ok(())
    }
}

impl FrameIndexBuilder for XyzIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        // Drain into a buffer first, then process — the closure can't borrow
        // self mutably and call self.process_line. Collect pending lines and
        // process post-loop.
        let mut staged: Vec<(String, u64, u32)> = Vec::new();
        self.lines
            .feed(chunk, global_offset, |line, line_offset, line_len| {
                staged.push((line.to_string(), line_offset, line_len));
            });
        for (line, off, len) in staged {
            // Errors during streaming index building are silently ignored —
            // the legacy reader's strict-error behavior surfaces when
            // parse_frame_bytes is later called on the resulting slice.
            // For oversized frames we still surface via finish().
            let _ = self.process_line(&line, off, len);
        }
    }

    fn drain(&mut self) -> Vec<FrameIndexEntry> {
        std::mem::take(&mut self.pending_entries)
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<Vec<FrameIndexEntry>> {
        let mut staged: Vec<(String, u64, u32)> = Vec::new();
        self.lines.finish(|line, line_offset, line_len| {
            staged.push((line.to_string(), line_offset, line_len));
        });
        for (line, off, len) in staged {
            self.process_line(&line, off, len)?;
        }
        // Trailing partial frame: if pending_frame_start is still Some,
        // we have an incomplete frame at EOF. The spec for XYZ says
        // ConsumingAtoms must reach zero before emit; an incomplete final
        // frame is malformed. Be lenient: drop it (legacy reader would
        // error at parse time on this slice).
        Ok(std::mem::take(&mut self.pending_entries))
    }

    fn bytes_seen(&self) -> u64 {
        self.lines.bytes_seen()
    }
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
    fn quoted_metadata_with_spaces_stays_a_string() {
        let data = b"1\ntitle=\"Water box\" Properties=species:S:1:pos:R:3\nH 0 0 0\n";
        let mut cursor = std::io::Cursor::new(&data[..]);
        let frame = read_xyz_frame_from_reader(&mut cursor)
            .expect("read XYZ")
            .expect("one frame");

        assert_eq!(
            frame.meta.get("title").and_then(MetaValue::as_str),
            Some("Water box")
        );
    }

    #[test]
    fn connct_comment_builds_zero_based_bonds() {
        let data = b"3\nname=water Connct=\"[0,1,0,2]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n";
        let mut cursor = std::io::Cursor::new(&data[..]);
        let frame = read_xyz_frame_from_reader(&mut cursor)
            .expect("read XYZ")
            .expect("one frame");
        let bonds = frame.get("bonds").expect("Connct creates bonds block");

        assert_eq!(
            bonds.get_uint("atomi").unwrap().as_slice().unwrap(),
            &[0, 0]
        );
        assert_eq!(
            bonds.get_uint("atomj").unwrap().as_slice().unwrap(),
            &[1, 2]
        );
        assert!(!frame.meta.contains_key("Connct"));
    }

    #[test]
    fn xyz_writer_preserves_connct_bonds() {
        let frame =
            parse_xyz_frame_str("3\nname=water Connct=\"[0,1,0,2]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
                .expect("parse XYZ");
        let mut output = Vec::new();

        write_xyz_frame(&mut output, &frame).expect("write XYZ");
        let output = String::from_utf8(output).expect("UTF-8 XYZ");

        assert!(
            output
                .lines()
                .nth(1)
                .unwrap()
                .contains("Connct=\"[0,1,0,2]\"")
        );
        let round_trip = parse_xyz_frame_str(&output).expect("read written XYZ");
        assert_eq!(
            round_trip
                .get("bonds")
                .unwrap()
                .get_uint("atomj")
                .unwrap()
                .as_slice()
                .unwrap(),
            &[1, 2]
        );
    }

    #[test]
    fn connct_comment_rejects_invalid_pairs() {
        let odd = "3\nConnct=\"[0,1,0]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n";
        assert!(
            parse_xyz_frame_str(odd)
                .unwrap_err()
                .contains("atom-index pairs")
        );

        let out_of_range = "3\nConnct=\"[0,3]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n";
        assert!(
            parse_xyz_frame_str(out_of_range)
                .unwrap_err()
                .contains("out of range")
        );
    }

    #[test]
    fn test_xyz_invalid_atom_count() {
        use std::io::Cursor;

        let data = b"abc\nComment\nH 0 0 0\n";
        let mut cursor = Cursor::new(&data[..]);
        let err = read_xyz_frame_from_reader(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    // -----------------------------------------------------------------
    // Streaming index tests
    // -----------------------------------------------------------------

    fn xyz_build_chunked(bytes: &[u8], chunk_size: usize) -> Vec<FrameIndexEntry> {
        let mut builder = Box::new(XyzIndexBuilder::new());
        let mut offset: u64 = 0;
        let mut out: Vec<FrameIndexEntry> = Vec::new();
        for piece in bytes.chunks(chunk_size.max(1)) {
            builder.feed(piece, offset);
            offset += piece.len() as u64;
            out.extend(builder.drain());
        }
        out.extend(builder.finish().expect("finish"));
        out
    }

    const TWO_FRAME_XYZ: &str =
        "2\nframe 0\nH 0 0 0\nH 1 0 0\n3\nframe 1\nO 0 0 0\nH 1 0 0\nH -1 0 0\n";

    #[test]
    fn xyz_streaming_single_shot_matches_chunks() {
        let bytes = TWO_FRAME_XYZ.as_bytes();
        let one_shot = xyz_build_chunked(bytes, bytes.len());
        assert_eq!(one_shot.len(), 2);
        for cs in [1usize, 3, 7, 13, 31, 64, 1024] {
            let chunked = xyz_build_chunked(bytes, cs);
            assert_eq!(
                one_shot, chunked,
                "chunk size {} produced different index",
                cs
            );
        }

        // Each entry slice must be parseable.
        for entry in &one_shot {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            parse_frame_bytes(&bytes[lo..hi]).expect("parse_frame_bytes");
        }
    }

    /// Edge: chunk boundary inside the natoms count line.
    #[test]
    fn xyz_streaming_boundary_inside_natoms_count() {
        // Multi-digit count.
        let s = "12\nframe\n".to_string()
            + &(0..12).map(|i| format!("H 0 0 {i}\n")).collect::<String>()
            + "8\nframe2\n"
            + &(0..8).map(|i| format!("H 0 0 {i}\n")).collect::<String>();
        let bytes = s.as_bytes();
        // Split at byte 1 — middle of "12".
        let mut builder = Box::new(XyzIndexBuilder::new());
        builder.feed(&bytes[..1], 0);
        builder.feed(&bytes[1..], 1);
        let mut got = builder.drain();
        got.extend(builder.finish().expect("finish"));
        let one_shot = xyz_build_chunked(bytes, bytes.len());
        assert_eq!(one_shot.len(), 2);
        assert_eq!(got, one_shot);
    }

    /// Edge: blank/garbage lines in `AwaitingNatoms` are skipped, not crashed.
    #[test]
    fn xyz_streaming_skips_blanks_in_awaiting_natoms() {
        let s = "\n\n  \n2\nframe 0\nH 0 0 0\nH 1 0 0\n";
        let bytes = s.as_bytes();
        let entries = xyz_build_chunked(bytes, bytes.len());
        assert_eq!(entries.len(), 1);
        // Frame must start at the "2\n" line, not at byte 0.
        let lo = entries[0].byte_offset as usize;
        assert!(bytes[lo..].starts_with(b"2\n"));
        let hi = lo + entries[0].byte_len as usize;
        let frame = parse_frame_bytes(&bytes[lo..hi]).expect("parse");
        assert_eq!(frame.get("atoms").unwrap().nrows().unwrap(), 2);
    }

    /// Legacy `XYZReader::build_index` (used by `len()` / random-access
    /// `read_step`) must tolerate blank lines between frames and trailing
    /// blanks — the same AwaitingNatoms rule as the streaming index.
    /// Regression: blank lines used to surface as
    /// `XYZ len error: invalid atom count:` through the wasm binding.
    #[test]
    fn xyz_legacy_index_skips_inter_frame_and_trailing_blanks() {
        use crate::io::reader::TrajReader;
        use std::io::{BufReader, Cursor};

        let s = "\
2
frame 0
H 0 0 0
H 1 0 0

2
frame 1
H 0 0 1
H 1 0 1

";
        let mut reader = XYZReader::new(BufReader::new(Cursor::new(s.as_bytes())));
        assert_eq!(reader.len().expect("len"), 2);

        let f0 = reader.read_step(0).expect("step0").expect("some");
        let f1 = reader.read_step(1).expect("step1").expect("some");
        assert_eq!(f0.get("atoms").unwrap().nrows().unwrap(), 2);
        assert_eq!(f1.get("atoms").unwrap().nrows().unwrap(), 2);
        let z0 = f0.get("atoms").unwrap().get_float("z").unwrap();
        let z1 = f1.get("atoms").unwrap().get_float("z").unwrap();
        assert!((z0[0] - 0.0).abs() < 1e-12);
        assert!((z1[0] - 1.0).abs() < 1e-12);
    }

    /// Leading blanks before the first frame must not confuse the legacy index.
    #[test]
    fn xyz_legacy_index_skips_leading_blanks() {
        use crate::io::reader::TrajReader;
        use std::io::{BufReader, Cursor};

        let s = "\n\n  \n2\nframe 0\nH 0 0 0\nH 1 0 0\n";
        let mut reader = XYZReader::new(BufReader::new(Cursor::new(s.as_bytes())));
        assert_eq!(reader.len().expect("len"), 1);
        let f0 = reader.read_step(0).expect("step0").expect("some");
        assert_eq!(f0.get("atoms").unwrap().nrows().unwrap(), 2);
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

/// Write a single frame to the writer in Extended XYZ format.
///
/// Accepts any type implementing [`FrameAccess`], including both [`Frame`] and
/// [`FrameView`](molrs::store::frame_view::FrameView). Existing callers passing `&Frame`
/// continue to work without changes.
pub fn write_xyz_frame<W: Write>(writer: &mut W, frame: &impl FrameAccess) -> std::io::Result<()> {
    use molrs::store::block::DType;

    // 1. Build per-atom data from the atoms block via visit_block.
    //    We collect everything we need into owned data structures inside the closure,
    //    then write outside it.
    struct AtomBlockData {
        n: usize,
        properties_str: String,
        elements: Vec<String>,
        /// Per-row, per-column values (excluding element). Outer = rows, inner = values.
        row_values: Vec<Vec<String>>,
    }

    let atom_data: Option<AtomBlockData> = frame.visit_block("atoms", |atoms| {
        let n = atoms.nrows().unwrap_or(0);

        // Collect and sort keys
        let mut keys = atoms.column_keys();
        keys.sort_by(|a, b| {
            let rank = |s: &str| match s {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => 3,
            };
            let ra = rank(a);
            let rb = rank(b);
            if ra != rb { ra.cmp(&rb) } else { a.cmp(b) }
        });

        // Build Properties string
        let mut props_parts = Vec::new();
        props_parts.push("species:S:1".to_string());

        let has_xyz = keys.contains(&"x") && keys.contains(&"y") && keys.contains(&"z");
        if has_xyz {
            props_parts.push("pos:R:3".to_string());
        }

        let dtype_to_char = |dt: DType| -> &'static str {
            match dt {
                DType::Float => "R",
                DType::Int | DType::UInt | DType::U8 => "I",
                DType::Bool => "L",
                DType::String => "S",
            }
        };

        let priority_keys = ["id", "mol"];
        for pk in &priority_keys {
            if keys.contains(pk)
                && let (Some(dt), Some(shape)) = (atoms.column_dtype(pk), atoms.column_shape(pk))
            {
                let m: usize = shape.iter().skip(1).product();
                let m = if m == 0 { 1 } else { m };
                props_parts.push(format!("{}:{}:{}", pk, dtype_to_char(dt), m));
            }
        }

        for k in &keys {
            if *k == "x" || *k == "y" || *k == "z" || *k == "element" || priority_keys.contains(k) {
                continue;
            }
            if let (Some(dt), Some(shape)) = (atoms.column_dtype(k), atoms.column_shape(k)) {
                let m: usize = shape.iter().skip(1).product();
                let m = if m == 0 { 1 } else { m };
                props_parts.push(format!("{}:{}:{}", k, dtype_to_char(dt), m));
            }
        }
        let properties_str = props_parts.join(":");

        // Read element symbols
        let elements: Vec<String> = atoms
            .get_string_view("element")
            .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
            .unwrap_or_else(|| vec!["X".to_string(); n]);

        // Build per-row values
        let mut row_values: Vec<Vec<String>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut line_parts = Vec::new();
            for k in &keys {
                if *k == "element" {
                    continue;
                }
                if let Some(dt) = atoms.column_dtype(k) {
                    match dt {
                        DType::Float => {
                            if let Some(arr) = atoms.get_float_view(k) {
                                let row = arr.index_axis(ndarray::Axis(0), i);
                                for val in row.iter() {
                                    line_parts.push(format!("{}", val));
                                }
                            }
                        }
                        DType::Int => {
                            if let Some(arr) = atoms.get_int_view(k) {
                                let row = arr.index_axis(ndarray::Axis(0), i);
                                for val in row.iter() {
                                    line_parts.push(format!("{}", val));
                                }
                            }
                        }
                        DType::Bool => {
                            if let Some(arr) = atoms.get_bool_view(k) {
                                let row = arr.index_axis(ndarray::Axis(0), i);
                                for val in row.iter() {
                                    line_parts.push(if *val { "T" } else { "F" }.to_string());
                                }
                            }
                        }
                        DType::UInt => {
                            if let Some(arr) = atoms.get_uint_view(k) {
                                let row = arr.index_axis(ndarray::Axis(0), i);
                                for val in row.iter() {
                                    line_parts.push(format!("{}", val));
                                }
                            }
                        }
                        DType::U8 => {
                            if let Some(arr) = atoms.get_u8_view(k) {
                                let row = arr.index_axis(ndarray::Axis(0), i);
                                for val in row.iter() {
                                    line_parts.push(format!("{}", val));
                                }
                            }
                        }
                        DType::String => {
                            if let Some(arr) = atoms.get_string_view(k) {
                                let row = arr.index_axis(ndarray::Axis(0), i);
                                for val in row.iter() {
                                    line_parts.push(val.clone());
                                }
                            }
                        }
                    }
                }
            }
            row_values.push(line_parts);
        }

        AtomBlockData {
            n,
            properties_str,
            elements,
            row_values,
        }
    });

    let atom_data = match atom_data {
        Some(d) => d,
        None => {
            writeln!(writer, "0")?;
            writeln!(writer)?;
            return Ok(());
        }
    };

    writeln!(writer, "{}", atom_data.n)?;

    // 2. Construct comment line using FrameAccess for simbox and meta
    let mut comment_parts = Vec::new();
    if let Some(simbox) = frame.simbox_ref() {
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
    for (k, v) in frame.meta_ref() {
        if k == "Lattice"
            || k == "Origin"
            || k == "Properties"
            || k == "comment"
            || k == "elements"
            || k.eq_ignore_ascii_case("Connct")
        {
            continue;
        }
        let val_str = meta_to_extxyz(v);
        comment_parts.push(format!("{}={}", k, val_str));
    }
    if let (Some(atomi), Some(atomj)) = (
        frame.get_uint("bonds", "atomi"),
        frame.get_uint("bonds", "atomj"),
    ) && atomi.len() == atomj.len()
        && !atomi.is_empty()
    {
        let indices = atomi
            .iter()
            .zip(atomj.iter())
            .flat_map(|(atomi, atomj)| [atomi.to_string(), atomj.to_string()])
            .collect::<Vec<_>>()
            .join(",");
        comment_parts.push(format!("Connct=\"[{indices}]\""));
    }
    comment_parts.push(format!("Properties={}", atom_data.properties_str));
    writeln!(writer, "{}", comment_parts.join(" "))?;

    // 3. Write atom lines
    for i in 0..atom_data.n {
        let species = atom_data
            .elements
            .get(i)
            .cloned()
            .unwrap_or_else(|| "X".to_string());
        let mut line_parts = vec![species];
        line_parts.extend(atom_data.row_values[i].iter().cloned());
        writeln!(writer, "{}", line_parts.join(" "))?;
    }

    Ok(())
}
