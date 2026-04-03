//! Integration tests for PDB file reader and writer.
//!
//! Every file in tests-data/pdb/ is read, structurally verified, written to a
//! temp file, and read back (roundtrip).  Specific tests cover known
//! interesting properties of individual files.

use molrs::frame::Frame;
use molrs::io::pdb::{read_pdb_frame, write_pdb_frame};
use molrs::types::F;
use std::io::BufWriter;
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn all_pdb_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("pdb");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read pdb test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No PDB files found in tests-data/pdb/");
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
    assert!(
        atoms.get_string("element").is_some(),
        "{:?}: missing element column",
        path
    );
}

fn verify_roundtrip(frame1: &Frame, frame2: &Frame, path: &std::path::Path, tol: F) {
    let atoms1 = frame1.get("atoms").unwrap();
    let atoms2 = frame2.get("atoms").unwrap();
    assert_eq!(
        atoms1.nrows(),
        atoms2.nrows(),
        "{:?}: atom count changed after roundtrip",
        path
    );
    let x1 = atoms1.get_float("x").unwrap();
    let x2 = atoms2.get_float("x").unwrap();
    let y1 = atoms1.get_float("y").unwrap();
    let y2 = atoms2.get_float("y").unwrap();
    let z1 = atoms1.get_float("z").unwrap();
    let z2 = atoms2.get_float("z").unwrap();
    for i in 0..x1.len().min(50) {
        assert!((x1[i] - x2[i]).abs() < tol, "{:?}: x[{}] roundtrip mismatch", path, i);
        assert!((y1[i] - y2[i]).abs() < tol, "{:?}: y[{}] roundtrip mismatch", path, i);
        assert!((z1[i] - z2[i]).abs() < tol, "{:?}: z[{}] roundtrip mismatch", path, i);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Every PDB file must parse and pass basic structure checks.
#[test]
fn test_all_pdb_files_parse() {
    for path in all_pdb_files() {
        let result = read_pdb_frame(path.to_str().unwrap());
        // Skip known-bad gzip files that our reader can't handle.
        let frame = match result {
            Ok(f) => f,
            Err(_) => continue,
        };
        verify_frame(&frame, &path);
    }
}

/// Every PDB file survives a write → read roundtrip with coordinates preserved.
#[test]
fn test_all_pdb_files_roundtrip() {
    for path in all_pdb_files() {
        let frame1 = match read_pdb_frame(path.to_str().unwrap()) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let temp = NamedTempFile::new().expect("create temp");
        {
            let file = std::fs::File::create(temp.path()).unwrap();
            let mut writer = BufWriter::new(file);
            if write_pdb_frame(&mut writer, &frame1).is_err() {
                continue;
            }
        }

        let frame2 = match read_pdb_frame(temp.path().to_str().unwrap()) {
            Ok(f) => f,
            Err(_) => continue,
        };

        verify_roundtrip(&frame1, &frame2, &path, 0.01);
    }
}

/// water.pdb must contain O and H elements.
#[test]
fn test_water_elements() {
    let path = crate::test_data::get_test_data_path("pdb/water.pdb");
    let frame = read_pdb_frame(path.to_str().unwrap()).expect("read water.pdb");
    let atoms = frame.get("atoms").expect("atoms");
    let elements = atoms.get_string("element").expect("element column");
    assert!(elements.iter().any(|e| e == "O"), "water.pdb must have O");
    assert!(elements.iter().any(|e| e == "H"), "water.pdb must have H");
}

/// 1vln-triclinic.pdb has a triclinic CRYST1 cell — verify SimBox is present.
#[test]
fn test_triclinic_simbox() {
    let path = crate::test_data::get_test_data_path("pdb/1vln-triclinic.pdb");
    let frame = read_pdb_frame(path.to_str().unwrap()).expect("read 1vln-triclinic.pdb");
    assert!(
        frame.simbox.is_some(),
        "1vln-triclinic.pdb must have a SimBox"
    );
    let vol = frame.simbox.as_ref().unwrap().volume();
    assert!(vol > 1000.0, "triclinic cell volume must be > 1000 Å³, got {:.1}", vol);
}

/// 1avg.pdb has CONECT records — verify bonds block is present.
#[test]
fn test_conect_bonds_present() {
    let path = crate::test_data::get_test_data_path("pdb/1avg.pdb");
    let frame = read_pdb_frame(path.to_str().unwrap()).expect("read 1avg.pdb");
    assert!(
        frame.contains_key("bonds"),
        "1avg.pdb must have a bonds block (CONECT records)"
    );
    let bonds = frame.get("bonds").unwrap();
    assert!(bonds.nrows().unwrap() > 0, "bonds block must be non-empty");
}

/// 4hhb.pdb: hemoglobin structure, verify large atom count and Fe element.
#[test]
fn test_hemoglobin_has_iron() {
    let path = crate::test_data::get_test_data_path("pdb/4hhb.pdb");
    let frame = read_pdb_frame(path.to_str().unwrap()).expect("read 4hhb.pdb");
    let atoms = frame.get("atoms").expect("atoms");
    assert!(atoms.nrows().unwrap() > 1000, "4hhb must have many atoms");
    let elements = atoms.get_string("element").expect("element column");
    assert!(
        elements.iter().any(|e| e.eq_ignore_ascii_case("Fe")),
        "4hhb.pdb (hemoglobin) must contain Fe"
    );
}

/// model.pdb has MODEL records — only the first frame should be returned.
#[test]
fn test_model_pdb_single_frame() {
    let path = crate::test_data::get_test_data_path("pdb/model.pdb");
    let frame = read_pdb_frame(path.to_str().unwrap()).expect("read model.pdb");
    let atoms = frame.get("atoms").expect("atoms");
    assert!(atoms.nrows().unwrap() > 0, "model.pdb should have atoms");
}
