//! LAMMPS molecule template files (native text + JSON).
//!
//! Native format: header counts + body sections (`Coords`, `Types`, `Charges`,
//! `Bonds`, …). JSON format follows the LAMMPS molecule JSON schema
//! (`format: "molecule"`).
//!
//! Output Frame (canonical columns):
//! - `"atoms"`: `id`, `type` (string), optional `x`/`y`/`z`, `charge`, `mass`,
//!   `mol_id`, `diameter`
//! - `"bonds"` / `"angles"` / `"dihedrals"` / `"impropers"`: `id`, `type`,
//!   `atomi`… (0-based indices into atoms)
//! - meta: `format=lammps_molecule`, `source_format`, counts, optional
//!   `title` / `units` / `center_of_mass` / `total_mass` / `inertia`

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use ndarray::{Array1, IxDyn};
use serde_json::{Value as JsonValue, json};

use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::store::meta::MetaValue;
use molrs::types::{F, I, Idx};

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

// ── public path API ─────────────────────────────────────────────────────────

/// Read a LAMMPS molecule file (native or `.json`).
pub fn read_lammps_molecule<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let path = path.as_ref();
    let is_json = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"));
    if is_json {
        read_lammps_molecule_json(path)
    } else {
        read_lammps_molecule_native(path)
    }
}

/// Write a Frame as a LAMMPS molecule file.
///
/// `format`: `"native"` or `"json"`.
pub fn write_lammps_molecule<P: AsRef<Path>>(path: P, frame: &Frame, format: &str) -> Result<()> {
    match format.to_ascii_lowercase().as_str() {
        "json" => write_lammps_molecule_json(path, frame),
        "native" => write_lammps_molecule_native(path, frame),
        other => Err(invalid_data(format!(
            "format must be 'native' or 'json', got `{other}`"
        ))),
    }
}

// ── native reader ───────────────────────────────────────────────────────────

fn read_lammps_molecule_native(path: &Path) -> Result<Frame> {
    let f = std::fs::File::open(path)?;
    let mut lines: Vec<String> = BufReader::new(f)
        .lines()
        .collect::<std::result::Result<_, _>>()?;
    if lines.is_empty() {
        return Err(invalid_data("Empty molecule file"));
    }
    // First line is title/comment.
    let title = lines[0].trim().trim_start_matches('#').trim().to_string();
    lines = lines.into_iter().skip(1).collect();

    let (header, sections) = parse_native_sections(&lines)?;
    let mut frame = Frame::new();
    frame.meta.insert("format", "lammps_molecule");
    frame.meta.insert("source_format", "native");
    frame.meta.insert("source_file", path.display().to_string());
    frame.meta.insert("title", title);
    for (k, v) in [
        ("n_atoms", header.n_atoms),
        ("n_bonds", header.n_bonds),
        ("n_angles", header.n_angles),
        ("n_dihedrals", header.n_dihedrals),
        ("n_impropers", header.n_impropers),
    ] {
        if let Some(n) = v {
            frame.meta.insert(k, MetaValue::I64(n as i64));
        }
    }
    if let Some(m) = header.total_mass {
        frame.meta.insert("total_mass", MetaValue::F64(m));
    }
    if let Some(com) = header.com {
        frame.meta.insert("center_of_mass", MetaValue::F64x3(com));
    }
    if let Some(inertia) = header.inertia {
        frame.meta.insert("inertia", MetaValue::F64x6(inertia));
    }

    let (atoms, id_to_idx) = parse_native_atoms(&sections)?;
    if atoms.nrows().unwrap_or(0) > 0 {
        frame.insert("atoms", atoms);
    }
    for (name, arity) in [
        ("Bonds", 2usize),
        ("Angles", 3),
        ("Dihedrals", 4),
        ("Impropers", 4),
    ] {
        if let Some(block) = parse_native_connectivity(&sections, name, arity, &id_to_idx)? {
            frame.insert(name.to_ascii_lowercase(), block);
        }
    }
    Ok(frame)
}

#[derive(Default)]
struct NativeHeader {
    n_atoms: Option<usize>,
    n_bonds: Option<usize>,
    n_angles: Option<usize>,
    n_dihedrals: Option<usize>,
    n_impropers: Option<usize>,
    total_mass: Option<F>,
    com: Option<[F; 3]>,
    inertia: Option<[F; 6]>,
}

fn parse_native_sections(lines: &[String]) -> Result<(NativeHeader, HashMap<String, Vec<String>>)> {
    let mut header = NativeHeader::default();
    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut current: Option<String> = None;

    for raw in lines {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if try_header_line(line, &mut header) {
            continue;
        }
        if let Some(sec) = section_name(line) {
            current = Some(sec);
            sections.entry(current.clone().unwrap()).or_default();
            continue;
        }
        if let Some(ref sec) = current {
            sections.get_mut(sec).unwrap().push(line.to_string());
        }
    }
    Ok((header, sections))
}

fn try_header_line(line: &str, header: &mut NativeHeader) -> bool {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return false;
    }
    let last = parts[parts.len() - 1];
    match last {
        "atoms" => {
            if let Ok(n) = parts[0].parse() {
                header.n_atoms = Some(n);
                return true;
            }
        }
        "bonds" => {
            if let Ok(n) = parts[0].parse() {
                header.n_bonds = Some(n);
                return true;
            }
        }
        "angles" => {
            if let Ok(n) = parts[0].parse() {
                header.n_angles = Some(n);
                return true;
            }
        }
        "dihedrals" => {
            if let Ok(n) = parts[0].parse() {
                header.n_dihedrals = Some(n);
                return true;
            }
        }
        "impropers" => {
            if let Ok(n) = parts[0].parse() {
                header.n_impropers = Some(n);
                return true;
            }
        }
        "mass" => {
            if let Ok(m) = parts[0].parse() {
                header.total_mass = Some(m);
                return true;
            }
        }
        "com" if parts.len() >= 4 => {
            if let (Ok(x), Ok(y), Ok(z)) = (parts[0].parse(), parts[1].parse(), parts[2].parse()) {
                header.com = Some([x, y, z]);
                return true;
            }
        }
        "inertia" if parts.len() >= 7 => {
            let mut arr = [0.0_f64; 6];
            for (i, p) in parts.iter().take(6).enumerate() {
                if let Ok(v) = p.parse() {
                    arr[i] = v;
                } else {
                    return false;
                }
            }
            header.inertia = Some(arr);
            return true;
        }
        _ => {}
    }
    false
}

fn section_name(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    const NAMES: &[&str] = &[
        "coords",
        "types",
        "molecules",
        "charges",
        "diameters",
        "masses",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
    ];
    for n in NAMES {
        if lower == *n {
            // Canonical section key: capitalized as in LAMMPS docs for multiword,
            // single word capitalized first letter.
            return Some(match *n {
                "coords" => "Coords".into(),
                "types" => "Types".into(),
                "molecules" => "Molecules".into(),
                "charges" => "Charges".into(),
                "diameters" => "Diameters".into(),
                "masses" => "Masses".into(),
                "bonds" => "Bonds".into(),
                "angles" => "Angles".into(),
                "dihedrals" => "Dihedrals".into(),
                "impropers" => "Impropers".into(),
                _ => n.to_string(),
            });
        }
    }
    None
}

fn parse_native_atoms(
    sections: &HashMap<String, Vec<String>>,
) -> Result<(Block, HashMap<Idx, Idx>)> {
    let types_lines = sections
        .get("Types")
        .ok_or_else(|| invalid_data("Native molecule file must contain Types section"))?;
    let mut ids: Vec<Idx> = Vec::new();
    let mut types: Vec<String> = Vec::new();
    for line in types_lines {
        let parts: Vec<&str> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if parts.len() >= 2 {
            ids.push(parts[0].parse().map_err(invalid_data)?);
            types.push(parts[1].to_string());
        }
    }
    if ids.is_empty() {
        return Err(invalid_data("Types section is empty"));
    }
    let id_to_idx: HashMap<Idx, Idx> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as Idx))
        .collect();
    let n = ids.len();
    let mut block = Block::new();
    insert_uint_col(&mut block, "id", ids)?;
    insert_str_col(&mut block, "type", types)?;

    if let Some(lines) = sections.get("Coords") {
        let mut x = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];
        for line in lines {
            let parts: Vec<&str> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            if parts.len() >= 4 {
                let id: Idx = parts[0].parse().map_err(invalid_data)?;
                if let Some(&idx) = id_to_idx.get(&id) {
                    let i = idx as usize;
                    x[i] = parts[1].parse().map_err(invalid_data)?;
                    y[i] = parts[2].parse().map_err(invalid_data)?;
                    z[i] = parts[3].parse().map_err(invalid_data)?;
                }
            }
        }
        insert_float_col(&mut block, "x", x)?;
        insert_float_col(&mut block, "y", y)?;
        insert_float_col(&mut block, "z", z)?;
    }

    for (sec, col) in [
        ("Charges", "charge"),
        ("Masses", "mass"),
        ("Diameters", "diameter"),
    ] {
        if let Some(lines) = sections.get(sec) {
            let mut vals = vec![0.0_f64; n];
            for line in lines {
                let parts: Vec<&str> = line
                    .split('#')
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .collect();
                if parts.len() >= 2 {
                    let id: Idx = parts[0].parse().map_err(invalid_data)?;
                    if let Some(&idx) = id_to_idx.get(&id) {
                        vals[idx as usize] = parts[1].parse().map_err(invalid_data)?;
                    }
                }
            }
            insert_float_col(&mut block, col, vals)?;
        }
    }
    if let Some(lines) = sections.get("Molecules") {
        let mut vals = vec![0_u64; n];
        for line in lines {
            let parts: Vec<&str> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            if parts.len() >= 2 {
                let id: Idx = parts[0].parse().map_err(invalid_data)?;
                if let Some(&idx) = id_to_idx.get(&id) {
                    vals[idx as usize] = parts[1].parse().map_err(invalid_data)?;
                }
            }
        }
        insert_uint_col(&mut block, "mol_id", vals)?;
    }
    Ok((block, id_to_idx))
}

fn parse_native_connectivity(
    sections: &HashMap<String, Vec<String>>,
    section: &str,
    arity: usize,
    id_to_idx: &HashMap<Idx, Idx>,
) -> Result<Option<Block>> {
    let Some(lines) = sections.get(section) else {
        return Ok(None);
    };
    if lines.is_empty() {
        return Ok(None);
    }
    let mut ids = Vec::new();
    let mut types = Vec::new();
    let mut members: Vec<Vec<Idx>> = vec![Vec::new(); arity];
    for line in lines {
        let parts: Vec<&str> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        // id type a1 a2 …  → need 2 + arity fields
        if parts.len() < 2 + arity {
            continue;
        }
        ids.push(parts[0].parse::<Idx>().map_err(invalid_data)?);
        types.push(parts[1].to_string());
        for (k, mcol) in members.iter_mut().enumerate() {
            let atom_id: Idx = parts[2 + k].parse().map_err(invalid_data)?;
            let idx = *id_to_idx
                .get(&atom_id)
                .ok_or_else(|| invalid_data(format!("unknown atom id {atom_id} in {section}")))?;
            mcol.push(idx);
        }
    }
    if ids.is_empty() {
        return Ok(None);
    }
    let mut block = Block::new();
    insert_uint_col(&mut block, "id", ids)?;
    insert_str_col(&mut block, "type", types)?;
    let keys = ["atomi", "atomj", "atomk", "atoml"];
    for (k, col) in members.into_iter().enumerate() {
        insert_uint_col(&mut block, keys[k], col)?;
    }
    Ok(Some(block))
}

// ── native writer ───────────────────────────────────────────────────────────

fn write_lammps_molecule_native<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    let n = atoms
        .nrows()
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    if n == 0 {
        return Err(invalid_data("Frame must contain atoms data"));
    }
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    let title = frame
        .meta
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Molecule template written by molrs");
    writeln!(w, "# {title}")?;
    writeln!(w)?;
    writeln!(w, "{n} atoms")?;
    for name in ["bonds", "angles", "dihedrals", "impropers"] {
        if let Some(b) = frame.get(name)
            && let Some(nb) = b.nrows()
            && nb > 0
        {
            writeln!(w, "{nb} {name}")?;
        }
    }
    if let Some(MetaValue::F64(m)) = frame.meta.get("total_mass") {
        writeln!(w, "{m:.6} mass")?;
    }
    if let Some(MetaValue::F64x3(c)) = frame.meta.get("center_of_mass") {
        writeln!(w, "{:.6} {:.6} {:.6} com", c[0], c[1], c[2])?;
    }
    if let Some(MetaValue::F64x6(inn)) = frame.meta.get("inertia") {
        write!(w, "{:.6}", inn[0])?;
        for v in &inn[1..] {
            write!(w, " {v:.6}")?;
        }
        writeln!(w, " inertia")?;
    }
    writeln!(w)?;

    let ids = atom_ids(atoms);
    if atoms.contains_key("x") && atoms.contains_key("y") && atoms.contains_key("z") {
        writeln!(w, "Coords")?;
        writeln!(w)?;
        let x = atoms
            .get_float("x")
            .ok_or_else(|| invalid_data("x missing"))?;
        let y = atoms
            .get_float("y")
            .ok_or_else(|| invalid_data("y missing"))?;
        let z = atoms
            .get_float("z")
            .ok_or_else(|| invalid_data("z missing"))?;
        for i in 0..n {
            writeln!(w, "{} {:.6} {:.6} {:.6}", ids[i], x[[i]], y[[i]], z[[i]])?;
        }
        writeln!(w)?;
    }
    writeln!(w, "Types")?;
    writeln!(w)?;
    let type_ids = type_ids_for_block(atoms)?;
    for i in 0..n {
        writeln!(w, "{} {}", ids[i], type_ids[i])?;
    }
    writeln!(w)?;

    if atoms.contains_key("mol_id") {
        writeln!(w, "Molecules")?;
        writeln!(w)?;
        if let Some(vals) = atoms.get_uint("mol_id") {
            for i in 0..n {
                writeln!(w, "{} {}", ids[i], vals[[i]])?;
            }
        } else if let Some(vals) = atoms.get_int("mol_id") {
            for i in 0..n {
                writeln!(w, "{} {}", ids[i], vals[[i]])?;
            }
        } else {
            return Err(invalid_data("mol_id"));
        }
        writeln!(w)?;
    }
    for (col, heading) in [
        ("charge", "Charges"),
        ("mass", "Masses"),
        ("diameter", "Diameters"),
    ] {
        if !atoms.contains_key(col) {
            continue;
        }
        writeln!(w, "{heading}")?;
        writeln!(w)?;
        let vals = atoms.get_float(col).ok_or_else(|| invalid_data(col))?;
        for i in 0..n {
            writeln!(w, "{} {:.6}", ids[i], vals[[i]])?;
        }
        writeln!(w)?;
    }

    for (name, arity) in [
        ("bonds", 2usize),
        ("angles", 3),
        ("dihedrals", 4),
        ("impropers", 4),
    ] {
        let Some(block) = frame.get(name) else {
            continue;
        };
        let Some(nb) = block.nrows() else {
            continue;
        };
        if nb == 0 {
            continue;
        }
        writeln!(w, "{}", capitalize(name))?;
        writeln!(w)?;
        let item_ids: Vec<Idx> = if block.contains_key("id") {
            if let Some(col) = block.get_uint("id") {
                (0..nb).map(|i| col[[i]]).collect()
            } else if let Some(col) = block.get_int("id") {
                (0..nb).map(|i| col[[i]] as Idx).collect()
            } else {
                return Err(invalid_data("id"));
            }
        } else {
            (1..=nb as Idx).collect()
        };
        let t_ids = type_ids_for_block(block)?;
        let member_keys = ["atomi", "atomj", "atomk", "atoml"];
        let mut member_cols = Vec::new();
        for key in member_keys.iter().take(arity) {
            let col = block.get_uint(key).ok_or_else(|| invalid_data(*key))?;
            member_cols.push(col);
        }
        for i in 0..nb {
            write!(w, "{} {}", item_ids[i], t_ids[i])?;
            for m in &member_cols {
                let atom_idx = m[[i]] as usize;
                let atom_id = ids
                    .get(atom_idx)
                    .copied()
                    .ok_or_else(|| invalid_data("connectivity atom index out of range"))?;
                write!(w, " {atom_id}")?;
            }
            writeln!(w)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn atom_ids(atoms: &Block) -> Vec<Idx> {
    let n = atoms.nrows().unwrap_or(0);
    if let Some(col) = atoms.get_uint("id") {
        return (0..n).map(|i| col[[i]]).collect();
    }
    if let Some(col) = atoms.get_int("id") {
        return (0..n).map(|i| col[[i]] as Idx).collect();
    }
    (1..=n as Idx).collect()
}

fn type_ids_for_block(block: &Block) -> Result<Vec<I>> {
    let n = block.nrows().ok_or_else(|| invalid_data("empty block"))?;
    if let Some(col) = block.get_int("type_id") {
        return Ok((0..n).map(|i| col[[i]]).collect());
    }
    if let Some(col) = block.get_uint("type_id") {
        return Ok((0..n).map(|i| col[[i]] as I).collect());
    }
    if let Some(col) = block.get_string("type") {
        let labels: Vec<String> = (0..n).map(|i| col[[i]].clone()).collect();
        let mut uniq = labels.clone();
        uniq.sort();
        uniq.dedup();
        if uniq.iter().all(|s| s.parse::<I>().is_ok()) {
            return labels
                .iter()
                .map(|s| s.parse::<I>().map_err(invalid_data))
                .collect();
        }
        let map: HashMap<&str, I> = uniq
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), (i + 1) as I))
            .collect();
        return Ok(labels.iter().map(|s| map[s.as_str()]).collect());
    }
    if let Some(col) = block.get_int("type") {
        return Ok((0..n).map(|i| col[[i]]).collect());
    }
    Err(invalid_data("block has no type / type_id"))
}

// ── JSON ────────────────────────────────────────────────────────────────────

fn read_lammps_molecule_json(path: &Path) -> Result<Frame> {
    let text = std::fs::read_to_string(path)?;
    let data: JsonValue = serde_json::from_str(&text).map_err(invalid_data)?;
    if data.get("format").and_then(|v| v.as_str()) != Some("molecule") {
        return Err(invalid_data("JSON file must have format='molecule'"));
    }
    let mut frame = Frame::new();
    frame.meta.insert("format", "lammps_molecule");
    frame.meta.insert("source_format", "json");
    frame.meta.insert("source_file", path.display().to_string());
    frame.meta.insert(
        "title",
        data.get("title").and_then(|v| v.as_str()).unwrap_or(""),
    );
    frame.meta.insert(
        "units",
        data.get("units").and_then(|v| v.as_str()).unwrap_or("lj"),
    );
    frame.meta.insert(
        "revision",
        MetaValue::I64(data.get("revision").and_then(|v| v.as_i64()).unwrap_or(1)),
    );

    let types_data = data
        .pointer("/types/data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| invalid_data("JSON molecule file must contain 'types' section"))?;
    let mut ids = Vec::new();
    let mut types = Vec::new();
    for entry in types_data {
        let a = entry
            .as_array()
            .ok_or_else(|| invalid_data("types data entry must be array"))?;
        ids.push(a[0].as_i64().ok_or_else(|| invalid_data("atom id"))? as Idx);
        let t = if let Some(s) = a[1].as_str() {
            s.to_string()
        } else if let Some(n) = a[1].as_i64() {
            n.to_string()
        } else {
            return Err(invalid_data("type must be str or int"));
        };
        types.push(t);
    }
    let id_to_idx: HashMap<Idx, Idx> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as Idx))
        .collect();
    let n = ids.len();
    let mut atoms = Block::new();
    insert_uint_col(&mut atoms, "id", ids.clone())?;
    insert_str_col(&mut atoms, "type", types)?;

    if let Some(coords) = data.pointer("/coords/data").and_then(|v| v.as_array()) {
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut z = vec![0.0; n];
        for entry in coords {
            let a = entry.as_array().ok_or_else(|| invalid_data("coords row"))?;
            let id = a[0].as_i64().ok_or_else(|| invalid_data("id"))? as Idx;
            let idx = *id_to_idx.get(&id).ok_or_else(|| invalid_data("id"))? as usize;
            x[idx] = a[1].as_f64().ok_or_else(|| invalid_data("x"))?;
            y[idx] = a[2].as_f64().ok_or_else(|| invalid_data("y"))?;
            z[idx] = a[3].as_f64().ok_or_else(|| invalid_data("z"))?;
        }
        insert_float_col(&mut atoms, "x", x)?;
        insert_float_col(&mut atoms, "y", y)?;
        insert_float_col(&mut atoms, "z", z)?;
    }
    for (json_key, col, as_int) in [
        ("charges", "charge", false),
        ("masses", "mass", false),
        ("diameters", "diameter", false),
        ("molecule", "mol_id", true),
    ] {
        if let Some(rows) = data
            .pointer(&format!("/{json_key}/data"))
            .and_then(|v| v.as_array())
        {
            if as_int {
                let mut vals = vec![0_u64; n];
                for entry in rows {
                    let a = entry.as_array().ok_or_else(|| invalid_data(json_key))?;
                    let id = a[0].as_i64().ok_or_else(|| invalid_data("id"))? as Idx;
                    let idx = *id_to_idx.get(&id).ok_or_else(|| invalid_data("id"))? as usize;
                    vals[idx] = a[1].as_i64().ok_or_else(|| invalid_data(col))? as u64;
                }
                insert_uint_col(&mut atoms, col, vals)?;
            } else {
                let mut vals = vec![0.0_f64; n];
                for entry in rows {
                    let a = entry.as_array().ok_or_else(|| invalid_data(json_key))?;
                    let id = a[0].as_i64().ok_or_else(|| invalid_data("id"))? as Idx;
                    let idx = *id_to_idx.get(&id).ok_or_else(|| invalid_data("id"))? as usize;
                    vals[idx] = a[1].as_f64().ok_or_else(|| invalid_data(col))?;
                }
                insert_float_col(&mut atoms, col, vals)?;
            }
        }
    }
    frame.insert("atoms", atoms);

    for (key, arity) in [
        ("bonds", 2usize),
        ("angles", 3),
        ("dihedrals", 4),
        ("impropers", 4),
    ] {
        if let Some(rows) = data
            .pointer(&format!("/{key}/data"))
            .and_then(|v| v.as_array())
        {
            if rows.is_empty() {
                continue;
            }
            let mut cids = Vec::new();
            let mut ctypes = Vec::new();
            let mut members: Vec<Vec<Idx>> = vec![Vec::new(); arity];
            for (i, entry) in rows.iter().enumerate() {
                let a = entry.as_array().ok_or_else(|| invalid_data(key))?;
                cids.push((i + 1) as Idx);
                let t = if let Some(s) = a[0].as_str() {
                    s.to_string()
                } else if let Some(n) = a[0].as_i64() {
                    n.to_string()
                } else {
                    return Err(invalid_data("connectivity type"));
                };
                ctypes.push(t);
                for k in 0..arity {
                    let atom_id = a[1 + k].as_i64().ok_or_else(|| invalid_data("atom id"))? as Idx;
                    let idx = *id_to_idx
                        .get(&atom_id)
                        .ok_or_else(|| invalid_data(format!("unknown atom {atom_id}")))?;
                    members[k].push(idx);
                }
            }
            let mut block = Block::new();
            insert_uint_col(&mut block, "id", cids)?;
            insert_str_col(&mut block, "type", ctypes)?;
            let keys = ["atomi", "atomj", "atomk", "atoml"];
            for (k, col) in members.into_iter().enumerate() {
                insert_uint_col(&mut block, keys[k], col)?;
            }
            frame.insert(key, block);
        }
    }

    if let Some(com) = data.get("com").and_then(|v| v.as_array())
        && com.len() >= 3
    {
        frame.meta.insert(
            "center_of_mass",
            MetaValue::F64x3([
                com[0].as_f64().unwrap_or(0.0),
                com[1].as_f64().unwrap_or(0.0),
                com[2].as_f64().unwrap_or(0.0),
            ]),
        );
    }
    if let Some(m) = data.get("masstotal").and_then(|v| v.as_f64()) {
        frame.meta.insert("total_mass", MetaValue::F64(m));
    }
    if let Some(inn) = data.get("inertia").and_then(|v| v.as_array())
        && inn.len() >= 6
    {
        let mut a = [0.0_f64; 6];
        for (i, v) in inn.iter().take(6).enumerate() {
            a[i] = v.as_f64().unwrap_or(0.0);
        }
        frame.meta.insert("inertia", MetaValue::F64x6(a));
    }
    Ok(frame)
}

fn write_lammps_molecule_json<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    let n = atoms
        .nrows()
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    let ids = atom_ids(atoms);
    let type_labels: Vec<String> = if let Some(col) = atoms.get_string("type") {
        (0..n).map(|i| col[[i]].clone()).collect()
    } else {
        type_ids_for_block(atoms)?
            .into_iter()
            .map(|v| v.to_string())
            .collect()
    };
    let types_data: Vec<JsonValue> = (0..n)
        .map(|i| {
            if let Ok(t) = type_labels[i].parse::<i64>() {
                json!([ids[i], t])
            } else {
                json!([ids[i], type_labels[i]])
            }
        })
        .collect();

    let title = frame
        .meta
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Molecule template written by molrs");
    let units = frame
        .meta
        .get("units")
        .and_then(|v| v.as_str())
        .unwrap_or("lj");
    let mut data = json!({
        "application": "LAMMPS",
        "format": "molecule",
        "revision": 1,
        "title": title,
        "schema": "https://download.lammps.org/json/molecule-schema.json",
        "units": units,
        "types": { "format": ["atom-id", "type"], "data": types_data },
    });

    if atoms.contains_key("x") && atoms.contains_key("y") && atoms.contains_key("z") {
        let x = atoms.get_float("x").ok_or_else(|| invalid_data("x"))?;
        let y = atoms.get_float("y").ok_or_else(|| invalid_data("y"))?;
        let z = atoms.get_float("z").ok_or_else(|| invalid_data("z"))?;
        let coords: Vec<JsonValue> = (0..n)
            .map(|i| json!([ids[i], x[[i]], y[[i]], z[[i]]]))
            .collect();
        data["coords"] = json!({ "format": ["atom-id", "x", "y", "z"], "data": coords });
    }
    if let Some(c) = atoms.get_float("charge") {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], c[[i]]])).collect();
        data["charges"] = json!({ "format": ["atom-id", "charge"], "data": rows });
    }
    if let Some(m) = atoms.get_float("mass") {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], m[[i]]])).collect();
        data["masses"] = json!({ "format": ["atom-id", "mass"], "data": rows });
    }
    if let Some(m) = atoms.get_uint("mol_id") {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], m[[i]]])).collect();
        data["molecule"] = json!({ "format": ["atom-id", "molecule-id"], "data": rows });
    } else if let Some(m) = atoms.get_int("mol_id") {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], m[[i]]])).collect();
        data["molecule"] = json!({ "format": ["atom-id", "molecule-id"], "data": rows });
    }

    for (name, arity, type_label) in [
        ("bonds", 2usize, "bond-type"),
        ("angles", 3, "angle-type"),
        ("dihedrals", 4, "dihedral-type"),
        ("impropers", 4, "improper-type"),
    ] {
        let Some(block) = frame.get(name) else {
            continue;
        };
        let Some(nb) = block.nrows() else {
            continue;
        };
        if nb == 0 {
            continue;
        }
        let t_ids = type_ids_for_block(block)?;
        let keys = ["atomi", "atomj", "atomk", "atoml"];
        let mut members = Vec::new();
        for key in keys.iter().take(arity) {
            members.push(block.get_uint(key).ok_or_else(|| invalid_data(*key))?);
        }
        let mut rows = Vec::new();
        let mut format = vec![type_label.to_string()];
        for k in 0..arity {
            format.push(format!("atom{}", k + 1));
        }
        for i in 0..nb {
            let mut row = vec![json!(t_ids[i])];
            for m in &members {
                row.push(json!(ids[m[[i]] as usize]));
            }
            rows.push(JsonValue::Array(row));
        }
        data[name] = json!({ "format": format, "data": rows });
    }

    if let Some(MetaValue::F64x3(c)) = frame.meta.get("center_of_mass") {
        data["com"] = json!([c[0], c[1], c[2]]);
    }
    if let Some(MetaValue::F64(m)) = frame.meta.get("total_mass") {
        data["masstotal"] = json!(m);
    }
    if let Some(MetaValue::F64x6(inn)) = frame.meta.get("inertia") {
        data["inertia"] = json!(inn.to_vec());
    }

    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &data).map_err(invalid_data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_water_roundtrip_shape() {
        let text = r#"# Water molecule. TIP3P geometry
3 atoms
2 bonds
1 angles

Coords

1    0.00000  -0.06556   0.00000
2    0.75695   0.52032   0.00000
3   -0.75695   0.52032   0.00000

Types

1        1
2        2
3        2

Charges

1       -0.834
2        0.417
3        0.417

Bonds

1   1      1      2
2   1      1      3

Angles

1   1      2      1      3
"#;
        let dir = std::env::temp_dir();
        let path = dir.join("molrs_water_test.mol");
        std::fs::write(&path, text).unwrap();
        let frame = read_lammps_molecule(&path).unwrap();
        assert_eq!(frame.get("atoms").unwrap().nrows(), Some(3));
        assert_eq!(frame.get("bonds").unwrap().nrows(), Some(2));
        assert_eq!(frame.get("angles").unwrap().nrows(), Some(1));
        assert!(frame.get("atoms").unwrap().contains_key("charge"));
        let out = dir.join("molrs_water_out.mol");
        write_lammps_molecule(&out, &frame, "native").unwrap();
        let frame2 = read_lammps_molecule(&out).unwrap();
        assert_eq!(frame2.get("atoms").unwrap().nrows(), Some(3));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&out);
    }
}
