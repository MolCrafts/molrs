//! Locating the shared `tests-data/` directory (cloned by
//! `scripts/fetch-test-data.sh` to the workspace root). Small, local to the io
//! test target — no separate crate.

use std::path::PathBuf;

/// The shared `tests-data/` dir: `$MOLRS_TESTS_DATA` if set, else `../tests-data`
/// relative to this crate (the workspace root).
pub fn tests_data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("MOLRS_TESTS_DATA").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests-data");
    assert!(
        dir.is_dir(),
        "tests-data not found at {} — run `bash scripts/fetch-test-data.sh`",
        dir.display()
    );
    dir
}

/// Path to a file inside `tests-data/`, e.g. `data_path("pdb/water.pdb")`.
pub fn data_path(relative: &str) -> PathBuf {
    tests_data_dir().join(relative)
}

/// Path to a **required** fixture. Panics if the file is absent.
///
/// Prefer this over `if !path.exists() { return; }`: a missing real fixture
/// must fail the suite, not report green with zero assertions. See
/// `.claude/notes/testing.md` (vacuous-green rule).
// Shared test-common module: not every test binary that includes it uses
// every helper, so the per-binary dead-code lint is a false positive here.
#[allow(dead_code)]
pub fn require_fixture(relative: &str) -> PathBuf {
    let path = data_path(relative);
    assert!(
        path.is_file(),
        "required fixture missing: {} — run `bash scripts/fetch-test-data.sh` \
         (or set MOLRS_TESTS_DATA)",
        path.display()
    );
    path
}

/// Every file directly inside `tests-data/<format>/`, sorted. Backs the rule
/// that format readers are tested against *all* real files, not a subset.
// Shared test-common module: not every test binary that includes it uses
// every helper, so the per-binary dead-code lint is a false positive here.
#[allow(dead_code)]
pub fn format_files(format: &str) -> Vec<PathBuf> {
    let dir = tests_data_dir().join(format);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    assert!(!files.is_empty(), "tests-data/{format}/ is empty");
    files
}
