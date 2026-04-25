//! Integration tests for Tripos MOL2 reader and writer.
//!
//! Iterates every file in `tests-data/mol2/` and verifies parse + round-trip.

use molrs::frame::Frame;
use molrs_io::mol2::{Mol2Reader, write_mol2_frame};
use molrs_io::reader::{FrameReader, Reader};
use std::io::BufReader;
use std::path::PathBuf;

fn all_mol2_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("mol2");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read mol2 dir {:?}: {}", dir, e))
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
    assert!(!paths.is_empty(), "No mol2 files found");
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
        "{:?}: missing x/y/z",
        path
    );
}

#[test]
fn read_all_good_mol2_files() {
    for path in all_mol2_files() {
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = Mol2Reader::new(BufReader::new(file));
        let mut count = 0;
        while let Some(frame) = reader
            .read_frame()
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", path, e))
        {
            verify_frame(&frame, &path);
            count += 1;
        }
        assert!(count >= 1, "{:?}: no molecules parsed", path);
    }
}

#[test]
fn round_trip_all_good_mol2_files() {
    for path in all_mol2_files() {
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = Mol2Reader::new(BufReader::new(file));
        // Round-trip just the first molecule per file.
        let frame = match reader
            .read_frame()
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", path, e))
        {
            Some(f) => f,
            None => continue,
        };

        let mut buf: Vec<u8> = Vec::new();
        write_mol2_frame(&mut buf, &frame)
            .unwrap_or_else(|e| panic!("Failed to write {:?}: {}", path, e));
        let mut reader2 = Mol2Reader::new(std::io::Cursor::new(&buf));
        let frame2 = reader2
            .read_frame()
            .unwrap_or_else(|e| panic!("Failed to re-read {:?}: {}", path, e))
            .unwrap();

        let atoms1 = frame.get("atoms").unwrap();
        let atoms2 = frame2.get("atoms").unwrap();
        assert_eq!(atoms1.nrows(), atoms2.nrows(), "{:?}", path);
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

        // Bond count should match (writer emits same number of bonds we read).
        let b1 = frame.get("bonds").map(|b| b.nrows().unwrap_or(0));
        let b2 = frame2.get("bonds").map(|b| b.nrows().unwrap_or(0));
        assert_eq!(b1, b2, "{:?}: bond count mismatch", path);
    }
}
