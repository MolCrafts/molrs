//! Fragment data loading helpers for Embed embedding.
//!
//! The loader reads external fragment tables from `data/embed/`
//! (or override paths via env vars).

use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::PathBuf;

use once_cell::sync::Lazy;

use molrs::element::Element;

/// Ring fragment template (atom symbols + coordinates).
#[derive(Debug, Clone)]
pub(crate) struct RingTemplate {
    pub symbols: Vec<String>,
    pub coords: Vec<[f64; 3]>,
}

/// Rigid fragment template parsed from rigid-fragment data.
#[derive(Debug, Clone)]
pub(crate) struct RigidTemplate {
    pub name: String,
    pub symbols: Vec<String>,
    pub coords: Vec<[f64; 3]>,
    /// Undirected bonds as `(i, j, order)`, 0-based atom indices.
    pub bonds: Vec<(usize, usize, u8)>,
}

#[derive(Debug, Default)]
struct RawFragmentRecord {
    smarts: String,
    coords: Vec<[f64; 3]>,
    atomic_numbers: Vec<u8>,
}

static RING_TEMPLATES: Lazy<Vec<RingTemplate>> = Lazy::new(load_ring_templates);
static RIGID_TEMPLATES: Lazy<Vec<RigidTemplate>> = Lazy::new(load_rigid_templates);

/// Access loaded ring templates.
pub(crate) fn ring_templates() -> &'static [RingTemplate] {
    &RING_TEMPLATES
}

/// Access loaded rigid templates.
pub(crate) fn rigid_templates() -> &'static [RigidTemplate] {
    &RIGID_TEMPLATES
}

fn load_ring_templates() -> Vec<RingTemplate> {
    let Some(path) = find_data_file("ring-fragments.txt") else {
        return Vec::new();
    };
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };

    parse_fragment_records(&text)
        .into_iter()
        .filter_map(|rec| {
            if rec.coords.len() < 3 {
                return None;
            }
            let mut symbols = extract_smarts_atom_symbols(&rec.smarts);
            if symbols.len() != rec.coords.len() {
                symbols = vec!["*".to_string(); rec.coords.len()];
            }
            Some(RingTemplate {
                symbols,
                coords: rec.coords,
            })
        })
        .collect()
}

fn load_rigid_templates() -> Vec<RigidTemplate> {
    let Some(path) = find_data_file("rigid-fragments.txt") else {
        return Vec::new();
    };
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };

    parse_fragment_records(&text)
        .into_iter()
        .filter_map(|rec| {
            if rec.coords.len() < 3 || rec.atomic_numbers.len() != rec.coords.len() {
                return None;
            }

            let symbols = rec
                .atomic_numbers
                .iter()
                .map(|&z| {
                    Element::by_number(z)
                        .map(|e| e.symbol().to_string())
                        .unwrap_or_else(|| "*".to_string())
                })
                .collect::<Vec<_>>();

            let bonds = infer_bonds_from_coords(&symbols, &rec.coords);
            if bonds.is_empty() {
                return None;
            }

            Some(RigidTemplate {
                name: shorten_name(&rec.smarts),
                symbols,
                coords: rec.coords,
                bonds,
            })
        })
        .collect()
}

fn shorten_name(smarts: &str) -> String {
    const MAX_CHARS: usize = 80;
    let mut out = String::new();
    for ch in smarts.chars().take(MAX_CHARS) {
        out.push(ch);
    }
    if smarts.chars().count() > MAX_CHARS {
        out.push_str("...");
    }
    if out.is_empty() {
        "rigid_fragment".to_string()
    } else {
        out
    }
}

fn infer_bonds_from_coords(symbols: &[String], coords: &[[f64; 3]]) -> Vec<(usize, usize, u8)> {
    #[derive(Debug)]
    struct Candidate {
        i: usize,
        j: usize,
        d: f64,
    }

    let n = coords.len();
    if n < 2 {
        return Vec::new();
    }

    let max_deg = symbols
        .iter()
        .map(|sym| {
            Element::by_symbol(sym)
                .and_then(|e| e.default_valences().iter().max().copied())
                .map(|v| v as usize)
                .unwrap_or(4)
        })
        .collect::<Vec<_>>();

    let mut candidates = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distance(coords[i], coords[j]);
            if !(0.4..=bond_cutoff(&symbols[i], &symbols[j])).contains(&d) {
                continue;
            }
            candidates.push(Candidate { i, j, d });
        }
    }

    candidates.sort_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal));

    let mut deg = vec![0usize; n];
    let mut bonds = Vec::new();
    for c in candidates {
        if deg[c.i] >= max_deg[c.i] || deg[c.j] >= max_deg[c.j] {
            continue;
        }
        deg[c.i] += 1;
        deg[c.j] += 1;
        bonds.push((c.i, c.j, 1));
    }

    bonds
}

fn bond_cutoff(sym_i: &str, sym_j: &str) -> f64 {
    let ri = Element::by_symbol(sym_i)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77);
    let rj = Element::by_symbol(sym_j)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77);
    1.25 * (ri + rj).max(0.8) + 0.2
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn find_data_file(name: &str) -> Option<PathBuf> {
    for d in candidate_data_dirs() {
        let p = d.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn candidate_data_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(v) = env::var("MOLRS_GEN3D_DATA_DIR")
        && !v.trim().is_empty()
    {
        dirs.push(PathBuf::from(v));
    }
    if let Ok(v) = env::var("GEN3D_DATADIR")
        && !v.trim().is_empty()
    {
        dirs.push(PathBuf::from(v));
    }

    if let Ok(cwd) = env::current_dir() {
        dirs.push(cwd.join("data/embed"));
        dirs.push(cwd.join("third_party/embed/data"));
    }

    // Workspace-relative fallback when running from the crate directory.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dirs.push(crate_dir.join("../data/embed"));
    dirs.push(crate_dir.join("data/embed"));

    dirs
}

fn parse_fragment_records(text: &str) -> Vec<RawFragmentRecord> {
    let mut records = Vec::new();
    let mut cur = RawFragmentRecord::default();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some((z, xyz)) = parse_zxyz_line(trimmed) {
            if cur.smarts.is_empty() {
                continue;
            }
            cur.atomic_numbers.push(z);
            cur.coords.push(xyz);
            continue;
        }

        if let Some(xyz) = parse_xyz_line(trimmed) {
            if cur.smarts.is_empty() {
                continue;
            }
            cur.coords.push(xyz);
            continue;
        }

        if !cur.smarts.is_empty() && !cur.coords.is_empty() {
            records.push(cur);
            cur = RawFragmentRecord::default();
        }
        cur.smarts = trimmed.to_string();
    }

    if !cur.smarts.is_empty() && !cur.coords.is_empty() {
        records.push(cur);
    }

    records
}

fn parse_zxyz_line(line: &str) -> Option<(u8, [f64; 3])> {
    let toks = line.split_whitespace().collect::<Vec<_>>();
    if toks.len() < 4 {
        return None;
    }
    let z = toks[0].parse::<u16>().ok()?;
    let x = toks[1].parse::<f64>().ok()?;
    let y = toks[2].parse::<f64>().ok()?;
    let zc = toks[3].parse::<f64>().ok()?;
    Some((u8::try_from(z).ok()?, [x, y, zc]))
}

fn parse_xyz_line(line: &str) -> Option<[f64; 3]> {
    let toks = line.split_whitespace().collect::<Vec<_>>();
    if toks.len() != 3 {
        return None;
    }
    let x = toks[0].parse::<f64>().ok()?;
    let y = toks[1].parse::<f64>().ok()?;
    let z = toks[2].parse::<f64>().ok()?;
    Some([x, y, z])
}

fn extract_smarts_atom_symbols(smarts: &str) -> Vec<String> {
    let chars = smarts.chars().collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        if c == '[' {
            let start = i + 1;
            i += 1;
            while i < chars.len() && chars[i] != ']' {
                i += 1;
            }
            let inner = chars[start..i].iter().collect::<String>();
            if let Some(sym) = extract_first_symbol_from_atom_token(&inner) {
                out.push(sym);
            }
            if i < chars.len() {
                i += 1;
            }
            continue;
        }

        if c == '*' {
            out.push("*".to_string());
            i += 1;
            continue;
        }

        if c.is_ascii_alphabetic()
            && let Some((sym, step)) = parse_symbol_at(&chars, i)
        {
            out.push(sym);
            i += step;
            continue;
        }

        i += 1;
    }

    out
}

fn extract_first_symbol_from_atom_token(token: &str) -> Option<String> {
    let chars = token.chars().collect::<Vec<_>>();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '*' {
            return Some("*".to_string());
        }
        if chars[i].is_ascii_alphabetic() {
            return parse_symbol_at(&chars, i).map(|(s, _)| s);
        }
        i += 1;
    }
    None
}

fn parse_symbol_at(chars: &[char], i: usize) -> Option<(String, usize)> {
    let c = *chars.get(i)?;

    if c == 'A' || c == 'a' {
        return Some(("*".to_string(), 1));
    }

    if c.is_ascii_uppercase() {
        if let Some(&next) = chars.get(i + 1)
            && next.is_ascii_lowercase()
        {
            let two = format!("{}{}", c, next);
            if Element::by_symbol(&two).is_some() {
                return Some((two, 2));
            }
        }
        let one = c.to_string();
        if Element::by_symbol(&one).is_some() {
            return Some((one, 1));
        }
        return Some(("*".to_string(), 1));
    }

    if c.is_ascii_lowercase() {
        if let Some(&next) = chars.get(i + 1)
            && next.is_ascii_lowercase()
        {
            let two = format!("{}{}", c.to_ascii_uppercase(), next);
            if Element::by_symbol(&two).is_some() {
                return Some((two, 2));
            }
        }
        let one = c.to_ascii_uppercase().to_string();
        if Element::by_symbol(&one).is_some() {
            return Some((one, 1));
        }
        return Some(("*".to_string(), 1));
    }

    None
}
