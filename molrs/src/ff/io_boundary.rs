//! `ff` reads no file: the force-field data model, kernels, typifiers and
//! parameter tables never name [`crate::io`]. Every file reader and writer —
//! force-field files included — is `io`'s ([`crate::io::forcefield`]), and
//! `io` depends on `ff`, never the reverse.
//!
//! Only test code may cross: a `#[cfg(test)]` module file (the engine
//! checks — `equivalence_check`, `engine_codec_check`, cmap `lammps_check`,
//! …) and a file's trailing `#[cfg(test)]` block read engine files to compare
//! against. This walks every source file under `src/ff` and fails on any other
//! line that names `io`.

use std::path::{Path, PathBuf};

/// Every `.rs` file under `dir`.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The files of modules declared `#[cfg(test)]` (`#[cfg(test)] mod x;` in
/// `parent.rs` / `parent/mod.rs` is `parent/x.rs` or `parent/x/…`).
fn test_only_modules(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(file).unwrap();
        let dir = if file.file_name().is_some_and(|n| n == "mod.rs") {
            file.parent().unwrap().to_path_buf()
        } else {
            file.with_extension("")
        };
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        for pair in lines.windows(2) {
            if pair[0] != "#[cfg(test)]" {
                continue;
            }
            let decl = pair[1]
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub(super) ");
            if let Some(name) = decl.strip_prefix("mod ").and_then(|d| d.strip_suffix(';')) {
                out.push(dir.join(format!("{name}.rs")));
                out.push(dir.join(name));
            }
        }
    }
    out
}

#[test]
fn ff_names_io_only_in_test_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ff");
    let mut files = Vec::new();
    sources(&root, &mut files);
    let test_only = test_only_modules(&files);

    let mut crossings = Vec::new();
    for file in &files {
        if test_only.iter().any(|t| file.starts_with(t)) {
            continue;
        }
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.trim();
            if code == "#[cfg(test)]" {
                break; // the trailing test block
            }
            if code.starts_with("//") {
                continue;
            }
            if code.contains("crate::io") || code.contains("molrs::io") {
                crossings.push(format!(
                    "{}:{}: {code}",
                    file.strip_prefix(&root).unwrap().display(),
                    n + 1
                ));
            }
        }
    }
    assert!(
        crossings.is_empty(),
        "ff names io outside test code:\n{}",
        crossings.join("\n")
    );
}
