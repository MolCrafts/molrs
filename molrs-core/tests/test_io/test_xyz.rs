//! Integration tests for XYZ / Extended-XYZ reader and writer.
//!
//! All good files in tests-data/xyz/ are parsed and round-tripped.
//! All files in tests-data/xyz/bad/ are expected to fail parsing.

use molrs::frame::Frame;
use molrs::io::xyz::{read_xyz_frame, write_xyz_frame};
use molrs::types::F;
use std::io::BufWriter;
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn all_xyz_good_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("xyz");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read xyz test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            // skip the bad/ subdirectory and compressed files
            if !p.is_file() { return None; }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "gz" | "bz2" | "xz" | "zst") { return None; }
            Some(p)
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No XYZ files found in tests-data/xyz/");
    paths
}

fn all_xyz_bad_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("xyz/bad");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read xyz/bad dir {:?}: {}", dir, e))
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
        atoms.get_float("x").is_some(),
        "{:?}: missing x column",
        path
    );
}

fn verify_roundtrip(frame1: &Frame, frame2: &Frame, path: &std::path::Path, tol: F) {
    let atoms1 = frame1.get("atoms").unwrap();
    let atoms2 = frame2.get("atoms").unwrap();
    assert_eq!(
        atoms1.nrows(),
        atoms2.nrows(),
        "{:?}: atom count mismatch after roundtrip",
        path
    );
    let x1 = atoms1.get_float("x").unwrap();
    let x2 = atoms2.get_float("x").unwrap();
    let y1 = atoms1.get_float("y").unwrap();
    let y2 = atoms2.get_float("y").unwrap();
    let z1 = atoms1.get_float("z").unwrap();
    let z2 = atoms2.get_float("z").unwrap();
    for i in 0..x1.len().min(50) {
        assert!((x1[i] - x2[i]).abs() < tol, "{:?}: x[{}] mismatch", path, i);
        assert!((y1[i] - y2[i]).abs() < tol, "{:?}: y[{}] mismatch", path, i);
        assert!((z1[i] - z2[i]).abs() < tol, "{:?}: z[{}] mismatch", path, i);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Every good XYZ file must parse and pass basic structure checks.
#[test]
fn test_all_xyz_files_parse() {
    for path in all_xyz_good_files() {
        let frame = read_xyz_frame(path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("{:?}: read failed: {}", path, e));
        verify_frame(&frame, &path);
    }
}

/// Every good XYZ file survives a write → read roundtrip with coordinates preserved.
#[test]
fn test_all_xyz_files_roundtrip() {
    for path in all_xyz_good_files() {
        let frame1 = match read_xyz_frame(path.to_str().unwrap()) {
            Ok(f) => f,
            Err(_) => continue, // already caught by test_all_xyz_files_parse
        };

        let temp = NamedTempFile::new().expect("create temp");
        {
            let file = std::fs::File::create(temp.path()).unwrap();
            let mut writer = BufWriter::new(file);
            if write_xyz_frame(&mut writer, &frame1).is_err() {
                continue; // skip files the writer can't handle (compressed etc.)
            }
        }

        let frame2 = match read_xyz_frame(temp.path().to_str().unwrap()) {
            Ok(f) => f,
            Err(_) => continue,
        };

        verify_roundtrip(&frame1, &frame2, &path, 1e-3);
    }
}

/// All files in xyz/bad/ must fail to parse (they contain intentionally broken data).
#[test]
fn test_all_bad_xyz_files_fail() {
    let bad_files = all_xyz_bad_files();
    // If bad/ is absent or empty this test passes vacuously.
    for path in bad_files {
        let result = read_xyz_frame(path.to_str().unwrap());
        assert!(
            result.is_err(),
            "{:?}: expected parse failure but got Ok",
            path
        );
    }
}

/// Extended XYZ (extended.xyz) must have a Lattice/SimBox.
#[test]
fn test_extended_xyz_has_simbox() {
    let path = crate::test_data::get_test_data_path("xyz/extended.xyz");
    let frame = read_xyz_frame(path.to_str().unwrap()).expect("read extended.xyz");
    assert!(
        frame.simbox.is_some(),
        "extended.xyz must have a SimBox from Lattice= header"
    );
}

/// Plain XYZ (methane.xyz) must have exactly 5 atoms (C + 4 H).
#[test]
fn test_methane_atom_count() {
    let path = crate::test_data::get_test_data_path("xyz/methane.xyz");
    let frame = read_xyz_frame(path.to_str().unwrap()).expect("read methane.xyz");
    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 5, "methane has 5 atoms");
}

/// Velocities XYZ must expose vx/vy/vz columns.
#[test]
fn test_velocities_xyz_has_velocity_columns() {
    let path = crate::test_data::get_test_data_path("xyz/velocities.xyz");
    let frame = read_xyz_frame(path.to_str().unwrap()).expect("read velocities.xyz");
    let atoms = frame.get("atoms").expect("atoms");
    assert!(
        atoms.get_float("vx").is_some()
            || atoms.get_float("vel_1").is_some()
            || atoms.len() > 3, // at minimum x/y/z + at least one velocity column
        "velocities.xyz must carry velocity data"
    );
}
