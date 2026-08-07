//! AMBER FRCMOD section reader / writer.
//!
//! Parses keyword sections (`MASS`, `BOND`, `ANGLE`, `DIHE`, `IMPROPER`,
//! `NONBON`) into a map of section name → raw body text. Round-trips the
//! historical molpy `read_frcmod` / `write_frcmod` surface.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Error, Result};
use std::path::Path;

const SECTIONS: &[&str] = &["MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"];

/// Parsed FRCMOD file: section name (lower-case) → body, plus remark / raw.
#[derive(Debug, Clone, Default)]
pub struct FrcmodFile {
    pub remark: String,
    pub sections: BTreeMap<String, String>,
    pub raw_text: String,
}

/// Read an FRCMOD file from disk.
pub fn read_frcmod(path: impl AsRef<Path>) -> Result<FrcmodFile> {
    let text = fs::read_to_string(path.as_ref())?;
    Ok(parse_frcmod(&text))
}

/// Parse FRCMOD text.
pub fn parse_frcmod(content: &str) -> FrcmodFile {
    let mut result = FrcmodFile {
        raw_text: content.to_string(),
        ..Default::default()
    };
    let mut current: Option<String> = None;
    let mut body: Vec<String> = Vec::new();

    for line in content.lines() {
        let stripped = line.trim();
        let upper = stripped.to_ascii_uppercase();
        if SECTIONS.contains(&upper.as_str()) {
            if let Some(sec) = current.take() {
                result
                    .sections
                    .insert(sec, body.join("\n").trim().to_string());
            }
            current = Some(upper.to_ascii_lowercase());
            body.clear();
        } else if current.is_none() && !stripped.is_empty() {
            result.remark = stripped.to_string();
        } else if !stripped.is_empty() {
            if current.is_some() {
                body.push(line.to_string());
            }
        } else if current.is_some() {
            // keep blank lines inside a section only if we already have content
            if !body.is_empty() {
                body.push(String::new());
            }
        }
    }
    if let Some(sec) = current {
        result
            .sections
            .insert(sec, body.join("\n").trim().to_string());
    }
    result
}

/// Write FRCMOD sections to a string.
pub fn format_frcmod(file: &FrcmodFile) -> String {
    let mut lines: Vec<String> = Vec::new();
    if !file.remark.is_empty() {
        lines.push(file.remark.clone());
        lines.push(String::new());
    }
    for name in SECTIONS {
        let key = name.to_ascii_lowercase();
        if let Some(body) = file.sections.get(&key) {
            if body.is_empty() {
                continue;
            }
            lines.push(name.to_string());
            lines.push(body.clone());
            lines.push(String::new());
        }
    }
    lines.join("\n")
}

/// Write FRCMOD to disk.
pub fn write_frcmod(path: impl AsRef<Path>, file: &FrcmodFile) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format_frcmod(file)).map_err(Error::other)
}
