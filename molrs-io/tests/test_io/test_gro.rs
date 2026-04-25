//! Integration tests for GROMACS GRO reader and writer.
//!
//! Iterates every file in `tests-data/gro/`. Files in `tests-data/gro/bad/`
//! and the explicitly-bad `truncated.gro` (declares more atoms than the file
//! contains) are expected to fail.

use molrs::frame::Frame;
use molrs_io::gro::{read_gro_frame, write_gro_frame};
use std::io::BufReader;
use std::path::PathBuf;

fn all_gro_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("gro");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read gro dir {:?}: {}", dir, e))
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
    assert!(!paths.is_empty(), "No GRO files found in tests-data/gro/");
    paths
}

fn is_known_bad(path: &std::path::Path) -> bool {
    matches!(
        path.file_name().and_then(|s| s.to_str()),
        Some("truncated.gro")
    )
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
        "{:?}: atoms block must have x/y/z",
        path
    );
    assert!(
        frame.simbox.is_some(),
        "{:?}: GRO frame must have a SimBox",
        path
    );
}

#[test]
fn read_all_good_gro_files() {
    for path in all_gro_files() {
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = BufReader::new(file);
        let result = read_gro_frame(&mut reader);
        if is_known_bad(&path) {
            assert!(
                result.is_err(),
                "{:?}: expected parse failure (truncated)",
                path
            );
            continue;
        }
        let frame = result
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", path, e))
            .unwrap_or_else(|| panic!("{:?}: unexpected EOF", path));
        verify_frame(&frame, &path);
    }
}

#[test]
fn round_trip_all_good_gro_files() {
    for path in all_gro_files() {
        if is_known_bad(&path) {
            continue;
        }
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = BufReader::new(file);
        let frame = read_gro_frame(&mut reader)
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", path, e))
            .unwrap();

        let mut buf: Vec<u8> = Vec::new();
        write_gro_frame(&mut buf, &frame)
            .unwrap_or_else(|e| panic!("Failed to write {:?}: {}", path, e));
        let mut cursor = std::io::Cursor::new(&buf);
        let frame2 = read_gro_frame(&mut cursor)
            .unwrap_or_else(|e| panic!("Failed to re-read written {:?}: {}", path, e))
            .unwrap();

        let atoms1 = frame.get("atoms").unwrap();
        let atoms2 = frame2.get("atoms").unwrap();
        assert_eq!(atoms1.nrows(), atoms2.nrows(), "{:?}", path);

        // GRO writes %.3f, so coords agree to ~5e-4 nm.
        let xs1 = atoms1.get_float("x").unwrap();
        let xs2 = atoms2.get_float("x").unwrap();
        for i in 0..xs1.len() {
            assert!(
                (xs1[[i]] - xs2[[i]]).abs() < 1e-3,
                "{:?}: x[{}]: {} vs {}",
                path,
                i,
                xs1[[i]],
                xs2[[i]]
            );
        }
    }
}
