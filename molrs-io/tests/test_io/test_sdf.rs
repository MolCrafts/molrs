//! Integration tests for the SDF / MDL molfile reader.
//!
//! Iterates every file in `tests-data/sdf/` and checks that:
//! - the reader returns at least one record with a non-empty atoms block
//!   carrying x/y/z float columns, plus an element string column,
//! - bonds (when present) reference 0-based atom indices.
//!
//! Files in `tests-data/sdf/bad/` must fail to parse.
//!
//! No writer exists for SDF — only read coverage is asserted.

use molrs::frame::Frame;
use molrs_io::reader::FrameReader;
use molrs_io::sdf::SDFReader;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

fn open_reader(path: &std::path::Path) -> SDFReader<BufReader<File>> {
    let file =
        File::open(path).unwrap_or_else(|e| panic!("Cannot open SDF file {:?}: {}", path, e));
    SDFReader::new(BufReader::new(file))
}

fn all_sdf_good_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("sdf");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read sdf dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if !p.is_file() {
                return None;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "gz" | "bz2" | "xz" | "zst") {
                return None;
            }
            Some(p)
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No SDF files found in tests-data/sdf/");
    paths
}

fn all_sdf_bad_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("sdf/bad");
    if !dir.exists() {
        return Vec::new();
    }
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read sdf/bad dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    paths
}

fn verify_frame(frame: &Frame, path: &std::path::Path) {
    let atoms = frame
        .get("atoms")
        .unwrap_or_else(|| panic!("{:?}: missing atoms block", path));
    let n = atoms
        .nrows()
        .unwrap_or_else(|| panic!("{:?}: nrows() failed", path));
    assert!(n > 0, "{:?}: atoms block is empty", path);
    assert!(
        atoms.get_float("x").is_some()
            && atoms.get_float("y").is_some()
            && atoms.get_float("z").is_some(),
        "{:?}: atoms block must have x/y/z float columns",
        path
    );
    assert!(
        atoms.get_string("element").is_some(),
        "{:?}: atoms block must have element string column",
        path
    );

    if let Some(bonds) = frame.get("bonds") {
        let bn = bonds.nrows().unwrap_or(0);
        if bn > 0 {
            let atomi = bonds
                .get_uint("atomi")
                .unwrap_or_else(|| panic!("{:?}: bonds missing atomi", path));
            let atomj = bonds
                .get_uint("atomj")
                .unwrap_or_else(|| panic!("{:?}: bonds missing atomj", path));
            for i in 0..bn {
                assert!(
                    (atomi[[i]] as usize) < n && (atomj[[i]] as usize) < n,
                    "{:?}: bond[{}] out of range",
                    path,
                    i
                );
            }
        }
    }
}

#[test]
fn read_all_good_sdf_files() {
    for path in all_sdf_good_files() {
        let mut reader = open_reader(&path);
        let mut count = 0usize;
        loop {
            match reader.read_frame() {
                Ok(Some(frame)) => {
                    verify_frame(&frame, &path);
                    count += 1;
                }
                Ok(None) => break,
                Err(e) => panic!("{:?}: read failed at record {}: {}", path, count, e),
            }
        }
        assert!(count > 0, "{:?}: SDF file produced no records", path);
    }
}

#[test]
fn bad_sdf_files_fail() {
    for path in all_sdf_bad_files() {
        let mut reader = open_reader(&path);
        let result = reader.read_frame();
        assert!(
            result.is_err(),
            "{:?}: bad SDF file unexpectedly parsed successfully",
            path
        );
    }
}
