//! Structural gates for the one-way Element export chain.
//!
//! The only hand-written periodic table lives in the Rust core. The crate root,
//! CXX bridge and Python binding are adapters over that type; none owns another
//! table, alias, or public module path.

use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    crate_root()
        .parent()
        .expect("molrs crate has a parent")
        .to_path_buf()
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("read source directory") {
            let path = entry.expect("source directory entry").path();
            if path.is_dir() {
                collect(&path, out);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    collect(dir, &mut files);
    files.sort();
    files
}

fn count(source: &str, needle: &str) -> usize {
    source.match_indices(needle).count()
}

#[test]
fn rust_has_one_definition_and_one_public_export() {
    let src = crate_root().join("src");
    let declaration = concat!("pub enum Ele", "ment {");
    let definitions: Vec<_> = rust_files(&src)
        .into_iter()
        .filter(|path| {
            fs::read_to_string(path)
                .expect("read Rust source")
                .contains(declaration)
        })
        .collect();
    assert_eq!(
        definitions,
        [src.join("core/system/element.rs")],
        "Element must have exactly one hand-written Rust definition"
    );

    let root = fs::read_to_string(src.join("lib.rs")).expect("read crate root");
    let export = concat!("pub use crate::core::system::element::Ele", "ment;");
    assert_eq!(
        count(&root, export),
        1,
        "crate root must own the sole export"
    );

    let core = fs::read_to_string(src.join("core/mod.rs")).expect("read core facade");
    assert_eq!(
        count(&core, concat!("pub use system::element::Ele", "ment;")),
        0,
        "core::* must not create a second re-export chain"
    );

    let system = fs::read_to_string(src.join("core/system/mod.rs")).expect("read system module");
    assert!(
        system
            .lines()
            .any(|line| line.trim() == "pub(crate) mod element;"),
        "the implementation module must remain crate-private"
    );
    assert!(
        !system.lines().any(|line| line.trim() == "pub mod element;"),
        "molrs::system::element is a forbidden public compatibility path"
    );
}

#[test]
fn internal_code_consumes_only_the_crate_root_type() {
    let src = crate_root().join("src");
    let root_export = src.join("lib.rs");
    let forbidden = ["system", "element", "Element"].join("::");
    let mut hits = Vec::new();
    for path in rust_files(&src) {
        if path == root_export {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read Rust source");
        for (line_no, line) in source.lines().enumerate() {
            if line.contains(&forbidden) {
                hits.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    line_no + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "internal consumers bypass molrs::Element:\n{}",
        hits.join("\n")
    );
}

#[test]
fn binders_delegate_to_the_canonical_type_without_compatibility_surfaces() {
    let repo = repo_root();
    let cxx = fs::read_to_string(repo.join("molrs-cxxapi/src/lib.rs")).expect("read CXX adapter");
    assert!(cxx.contains("use molrs::Element;"));
    assert!(cxx.contains(".and_then(Element::by_number)"));
    assert!(!cxx.contains(&["Core", "Element"].concat()));
    assert!(!cxx.contains(concat!("1 => ", "\"H\"")));
    assert!(cxx.contains(".get_string(\"species\")"));
    assert!(
        !cxx.contains("frame_atomic_numbers"),
        "the deleted type/element/species fallback bridge must not return"
    );

    let build_script =
        fs::read_to_string(repo.join("molrs-cxxapi/build.rs")).expect("read CXX generator");
    assert!(build_script.contains("cxx_element_variants(&element_source)"));
    assert!(build_script.contains("join(\"element.rs\")"));
    assert!(!build_script.contains("frame_atomic_numbers"));

    let py_binding = fs::read_to_string(repo.join("molrs-python/src/core/system/element.rs"))
        .expect("read Python Element adapter");
    assert!(py_binding.contains("use molrs::Element;"));
    assert!(!py_binding.contains("Option<Element>"));
    assert!(!py_binding.contains(concat!("fn init", "ialize")));

    let compatibility_name = ["Element", "Data"].concat();
    for relative in [
        "molrs-python/src/lib.rs",
        "molrs-python/python/molrs/__init__.py",
        "molrs-python/python/molrs/molrs.pyi",
    ] {
        let path = repo.join(relative);
        let source = fs::read_to_string(&path).expect("read Python export source");
        assert!(
            !source.contains(&compatibility_name),
            "legacy Python Element alias remains in {}",
            path.display()
        );
    }
}
