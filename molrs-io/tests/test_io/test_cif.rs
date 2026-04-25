//! Integration tests for CIF reader and writer.
//!
//! Iterates every file in `tests-data/cif/`. Files are split into:
//!
//! - **small-molecule CIF** (`1544173.cif`, `Zr-UiO-66-pressure.cif`): full
//!   parse + round-trip, with coordinate tolerance 1e-3.
//! - **mmCIF / large CIF** (`4hhb.cif`, `1j8k.cif`): parse-only — these large
//!   files exercise multi-line strings and many extension columns; the MVP
//!   parser must accept them but full round-trip is out of scope.

use molrs::frame::Frame;
use molrs_io::cif::{CifReader, write_cif_frame};
use molrs_io::reader::{FrameReader, Reader};
use std::io::BufReader;
use std::path::PathBuf;

fn all_cif_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("cif");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read cif dir {:?}: {}", dir, e))
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
    assert!(!paths.is_empty(), "No cif files found");
    paths
}

fn is_small_molecule_cif(path: &std::path::Path) -> bool {
    matches!(
        path.file_name().and_then(|s| s.to_str()),
        Some("1544173.cif") | Some("Zr-UiO-66-pressure.cif")
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
        "{:?}: atoms block missing x/y/z",
        path
    );
}

#[test]
fn read_all_good_cif_files() {
    for path in all_cif_files() {
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = CifReader::new(BufReader::new(file));
        let mut count = 0;
        while let Some(frame) = reader
            .read_frame()
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", path, e))
        {
            verify_frame(&frame, &path);
            count += 1;
        }
        assert!(count >= 1, "{:?}: no blocks parsed", path);
    }
}

#[test]
fn round_trip_small_molecule_cif() {
    for path in all_cif_files() {
        if !is_small_molecule_cif(&path) {
            continue;
        }
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = CifReader::new(BufReader::new(file));
        let frame = reader.read_frame().unwrap().unwrap();

        let mut buf: Vec<u8> = Vec::new();
        write_cif_frame(&mut buf, &frame)
            .unwrap_or_else(|e| panic!("Failed to write {:?}: {}", path, e));
        let mut reader2 = CifReader::new(std::io::Cursor::new(&buf));
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
    }
}
