//! Integration tests for VASP POSCAR reader and writer.
//!
//! Iterates every file in `tests-data/poscar/` and checks that:
//! - the reader parses successfully into a Frame with a SimBox and a non-empty
//!   atoms block carrying x/y/z columns,
//! - the writer round-trips back to a frame whose atom coordinates agree to
//!   within 1e-4.
//!
//! Files in `tests-data/poscar/bad/` must fail to parse.

use molrs::frame::Frame;
use molrs_io::poscar::{read_poscar, read_poscar_from_reader, write_poscar_to_writer};
use std::path::PathBuf;

fn all_poscar_good_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("poscar");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read poscar dir {:?}: {}", dir, e))
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
    assert!(
        !paths.is_empty(),
        "No POSCAR files found in tests-data/poscar/"
    );
    paths
}

fn all_poscar_bad_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("poscar/bad");
    if !dir.exists() {
        return Vec::new();
    }
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read poscar/bad dir {:?}: {}", dir, e))
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
        frame.simbox.is_some(),
        "{:?}: POSCAR frame must have a SimBox",
        path
    );
}

#[test]
fn read_all_good_poscar_files() {
    for path in all_poscar_good_files() {
        let frame =
            read_poscar(&path).unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", path, e));
        verify_frame(&frame, &path);
    }
}

#[test]
fn round_trip_all_good_poscar_files() {
    for path in all_poscar_good_files() {
        let frame =
            read_poscar(&path).unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", path, e));

        let mut buf: Vec<u8> = Vec::new();
        write_poscar_to_writer(&mut buf, &frame)
            .unwrap_or_else(|e| panic!("Failed to write {:?}: {}", path, e));
        let frame2 = read_poscar_from_reader(std::io::Cursor::new(&buf))
            .unwrap_or_else(|e| panic!("Failed to re-read written {:?}: {}", path, e));

        let atoms1 = frame.get("atoms").unwrap();
        let atoms2 = frame2.get("atoms").unwrap();
        assert_eq!(
            atoms1.nrows(),
            atoms2.nrows(),
            "{:?}: atom count differs after round-trip",
            path
        );

        // Compare per-atom Cartesian coords; the writer may have permuted atom
        // order to group by symbol. Build sets of sorted (x, y, z) tuples.
        let n = atoms1.nrows().unwrap();
        let xs1 = atoms1.get_float("x").unwrap();
        let ys1 = atoms1.get_float("y").unwrap();
        let zs1 = atoms1.get_float("z").unwrap();
        let xs2 = atoms2.get_float("x").unwrap();
        let ys2 = atoms2.get_float("y").unwrap();
        let zs2 = atoms2.get_float("z").unwrap();

        let mut pts1: Vec<[f64; 3]> = (0..n).map(|i| [xs1[[i]], ys1[[i]], zs1[[i]]]).collect();
        let mut pts2: Vec<[f64; 3]> = (0..n).map(|i| [xs2[[i]], ys2[[i]], zs2[[i]]]).collect();
        // Sort by lexicographic tuple to allow permuted order.
        pts1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pts2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (p1, p2) in pts1.iter().zip(pts2.iter()) {
            for axis in 0..3 {
                assert!(
                    (p1[axis] - p2[axis]).abs() < 1e-4,
                    "{:?}: coord mismatch {:?} vs {:?}",
                    path,
                    p1,
                    p2
                );
            }
        }
    }
}

#[test]
fn bad_poscar_files_fail() {
    for path in all_poscar_bad_files() {
        let result = read_poscar(&path);
        assert!(
            result.is_err(),
            "{:?}: bad POSCAR file unexpectedly parsed successfully",
            path
        );
    }
}
