//! Integration tests for LAMMPS data file reader and writer.
//!
//! Iterates every file in `tests-data/lammps-data/` and checks that:
//! - the reader parses successfully into a Frame with a SimBox,
//! - the writer round-trips frames whose `atoms` block carries x/y/z columns.
//!
//! Specific feature tests cover labelmaps, molecule IDs, triclinic boxes,
//! whitespace tolerance, angles/dihedrals, and solvated systems.

use molrs::block::Block;
use molrs::frame::Frame;
use molrs::types::{F, I, U};
use molrs_io::lammps_data::{LAMMPSDataReader, read_lammps_data, write_lammps_data};
use molrs_io::reader::FrameReader;
use ndarray::Array1;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers — directory globbing
// ---------------------------------------------------------------------------

fn all_lammps_data_files() -> Vec<PathBuf> {
    let dir = crate::test_data::get_test_data_path("lammps-data");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read lammps-data dir {:?}: {}", dir, e))
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
        "No LAMMPS data files found in tests-data/lammps-data/"
    );
    paths
}

fn frame_has_atoms(frame: &Frame) -> bool {
    frame
        .get("atoms")
        .and_then(|b| b.nrows())
        .map(|n| n > 0)
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Glob tests — every file in tests-data/lammps-data/ must be exercised
// ---------------------------------------------------------------------------

/// Every file must parse. Atom blocks may be empty (e.g. triclinic-1.lmp
/// declares 0 atoms) but when present must carry x/y/z float columns.
/// SimBox is optional — some fixtures omit box dimensions.
#[test]
fn test_all_lammps_data_files_parse() {
    for path in all_lammps_data_files() {
        let frame =
            read_lammps_data(&path).unwrap_or_else(|e| panic!("{:?}: read failed: {}", path, e));
        if let Some(atoms) = frame.get("atoms") {
            if atoms.nrows().unwrap_or(0) > 0 {
                assert!(
                    atoms.get_float("x").is_some()
                        && atoms.get_float("y").is_some()
                        && atoms.get_float("z").is_some(),
                    "{:?}: atoms block missing x/y/z float columns",
                    path
                );
            }
        }
    }
}

/// Every file with at least one atom must survive a write → read roundtrip
/// with atom count preserved.
#[test]
fn test_all_lammps_data_files_roundtrip() {
    for path in all_lammps_data_files() {
        let frame1 =
            read_lammps_data(&path).unwrap_or_else(|e| panic!("{:?}: read failed: {}", path, e));
        if !frame_has_atoms(&frame1) {
            continue;
        }
        let temp = NamedTempFile::new().expect("create temp");
        write_lammps_data(temp.path(), &frame1)
            .unwrap_or_else(|e| panic!("{:?}: write failed: {}", path, e));
        let frame2 = read_lammps_data(temp.path())
            .unwrap_or_else(|e| panic!("{:?}: re-read failed: {}", path, e));
        let n1 = frame1.get("atoms").unwrap().nrows().unwrap_or(0);
        let n2 = frame2.get("atoms").unwrap().nrows().unwrap_or(0);
        assert_eq!(n1, n2, "{:?}: atom count changed across roundtrip", path);
    }
}

// ---------------------------------------------------------------------------
// Specific feature tests — preserve the value of named fixtures
// ---------------------------------------------------------------------------

#[test]
fn test_read_labelmap() {
    let path = crate::test_data::get_test_data_path("lammps-data/labelmap.lmp");
    let frame = read_lammps_data(path).expect("Failed to read labelmap.lmp");

    let atoms = frame.get("atoms").expect("atoms block");
    assert_eq!(atoms.nrows(), Some(16));

    let x = atoms.get_float("x").expect("x coordinates");
    let _y = atoms.get_float("y").expect("y coordinates");
    let _z = atoms.get_float("z").expect("z coordinates");
    let types = atoms.get_int("type").expect("atom types");
    let _charges = atoms.get_float("charge").expect("charges");
    let mol_ids = atoms.get_int("molecule_id").expect("molecule IDs");

    assert_eq!(x.len(), 16);
    assert_eq!(types[0], 1);
    assert_eq!(mol_ids[0], 1);

    let bonds = frame.get("bonds").expect("bonds block");
    assert_eq!(bonds.nrows(), Some(14));
    let atom_i = bonds.get_uint("atomi").expect("atomi");
    let _atom_j = bonds.get_uint("atomj").expect("atomj");
    let _bond_types = bonds.get_int("type").expect("bond types");
    assert_eq!(atom_i.len(), 14);

    let atom_labels = frame
        .meta
        .get("atom_type_labels")
        .expect("atom type labels");
    assert!(atom_labels.contains("1:f"));
    assert!(atom_labels.contains("2:c3"));

    let bond_labels = frame
        .meta
        .get("bond_type_labels")
        .expect("bond type labels");
    assert!(bond_labels.contains("1:c3-f"));
}

#[test]
fn test_read_molid() {
    let path = crate::test_data::get_test_data_path("lammps-data/molid.lmp");
    let frame = read_lammps_data(path).expect("Failed to read molid.lmp");

    let atoms = frame.get("atoms").expect("atoms block");
    assert_eq!(atoms.nrows(), Some(12));

    let mol_ids = atoms.get_int("molecule_id").expect("molecule IDs");
    assert_eq!(mol_ids[0], 0);
    assert_eq!(mol_ids[3], 1);

    let simbox = frame.simbox.as_ref().expect("simbox");
    let lengths = simbox.lengths();
    assert_eq!(lengths[0], 20.0);
    assert_eq!(lengths[1], 20.0);
    assert_eq!(lengths[2], 20.0);
}

#[test]
fn test_read_triclinic() {
    let path = crate::test_data::get_test_data_path("lammps-data/triclinic-1.lmp");
    let frame = read_lammps_data(path).expect("Failed to read triclinic-1.lmp");

    let simbox = frame.simbox.as_ref().expect("simbox");
    let tilts = simbox.tilts();
    assert_eq!(tilts[0], 0.0);
    assert_eq!(tilts[1], 0.0);
    assert_eq!(tilts[2], 0.0);
    assert!(!frame.contains_key("atoms") || frame.get("atoms").unwrap().nrows() == Some(0));
}

#[test]
fn test_read_triclinic_2() {
    let path = crate::test_data::get_test_data_path("lammps-data/triclinic-2.lmp");
    let frame = read_lammps_data(path).expect("Failed to read triclinic-2.lmp");
    let simbox = frame.simbox.as_ref().expect("simbox");
    let _tilts = simbox.tilts();
}

#[test]
fn test_read_whitespaces() {
    let path = crate::test_data::get_test_data_path("lammps-data/whitespaces.lmp");
    let frame = read_lammps_data(path).expect("Failed to read whitespaces.lmp");

    let atoms = frame.get("atoms").expect("atoms block");
    assert_eq!(atoms.nrows(), Some(1));

    let x = atoms.get_float("x").expect("x coordinates");
    assert_eq!(x[0], 5.0);

    let simbox = frame.simbox.as_ref().expect("simbox");
    let lengths = simbox.lengths();
    assert_eq!(lengths[0], 10.0);
    assert_eq!(lengths[1], 10.0);
    assert_eq!(lengths[2], 10.0);
}

#[test]
fn test_read_angles_and_dihedrals() {
    let path = crate::test_data::get_test_data_path("lammps-data/labelmap.lmp");
    let frame = read_lammps_data(path).expect("Failed to read labelmap.lmp");

    let angles = frame.get("angles").expect("angles block");
    assert_eq!(angles.nrows(), Some(25));

    let _atom_i = angles.get_uint("atomi").expect("atomi");
    let _atom_j = angles.get_uint("atomj").expect("atomj");
    let _atom_k = angles.get_uint("atomk").expect("atomk");
    let angle_types = angles.get_int("type").expect("angle types");
    assert_eq!(angle_types.len(), 25);

    let dihedrals = frame.get("dihedrals").expect("dihedrals block");
    assert_eq!(dihedrals.nrows(), Some(27));
    let dihedral_types = dihedrals.get_int("type").expect("dihedral types");
    assert_eq!(dihedral_types.len(), 27);
}

#[test]
fn test_read_solvated_system() {
    let path = crate::test_data::get_test_data_path("lammps-data/solvated.lmp");
    let frame = read_lammps_data(path).expect("Failed to read solvated.lmp");

    let atoms = frame.get("atoms").expect("atoms block");
    let nrows = atoms.nrows().expect("nrows");
    assert!(nrows > 0, "Solvated system should have atoms");
    assert_eq!(atoms.get_float("x").unwrap().len(), nrows);
    assert!(frame.simbox.is_some());
}

#[test]
fn test_molecular_reader_interface() {
    let path = crate::test_data::get_test_data_path("lammps-data/molid.lmp");
    let file = File::open(path).expect("Failed to open file");
    let mut reader = LAMMPSDataReader::new(BufReader::new(file));

    let frame = reader
        .read_frame()
        .expect("Failed to read")
        .expect("No frame");
    assert!(frame.get("atoms").is_some());

    let frame1 = reader.read_frame().expect("Failed to read");
    assert!(frame1.is_none());
}

// ---------------------------------------------------------------------------
// Synthetic-frame writer tests — exercise writer paths not in real fixtures
// ---------------------------------------------------------------------------

#[test]
fn test_write_and_read_roundtrip() {
    let mut frame = Frame::new();
    let mut atoms = Block::new();
    atoms
        .insert(
            "x",
            Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "y",
            Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "z",
            Array1::from_vec(vec![0.0 as F, 0.0 as F, 0.0 as F]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "id",
            Array1::from_vec(vec![1 as I, 2 as I, 3 as I]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "type",
            Array1::from_vec(vec![1 as I, 1 as I, 2 as I]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "charge",
            Array1::from_vec(vec![0.0 as F, -1.0 as F, 1.0 as F]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "molecule_id",
            Array1::from_vec(vec![1 as I, 1 as I, 1 as I]).into_dyn(),
        )
        .unwrap();
    frame.insert("atoms", atoms);

    let mut bonds = Block::new();
    bonds
        .insert("atomi", Array1::from_vec(vec![0 as U, 1 as U]).into_dyn())
        .unwrap();
    bonds
        .insert("atomj", Array1::from_vec(vec![1 as U, 2 as U]).into_dyn())
        .unwrap();
    bonds
        .insert("type", Array1::from_vec(vec![1 as I, 1 as I]).into_dyn())
        .unwrap();
    frame.insert("bonds", bonds);

    use molrs::region::simbox::SimBox;
    use ndarray::array;
    frame.simbox = Some(
        SimBox::ortho(
            array![10.0 as F, 10.0 as F, 10.0 as F],
            array![0.0 as F, 0.0 as F, 0.0 as F],
            [true, true, true],
        )
        .expect("valid simbox"),
    );

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write_lammps_data(temp_file.path(), &frame).expect("Failed to write");

    let frame2 = read_lammps_data(temp_file.path()).expect("Failed to read");
    let atoms2 = frame2.get("atoms").expect("atoms block");
    assert_eq!(atoms2.nrows(), Some(3));
    let x2 = atoms2.get_float("x").expect("x coordinates");
    assert_eq!(x2[0], 1.0);
    assert_eq!(x2[1], 2.0);
    assert_eq!(x2[2], 3.0);

    let bonds2 = frame2.get("bonds").expect("bonds block");
    assert_eq!(bonds2.nrows(), Some(2));

    let sb2 = frame2.simbox.as_ref().expect("simbox");
    let l = sb2.lengths();
    assert_eq!(l[0], 10.0);
    assert_eq!(l[1], 10.0);
    assert_eq!(l[2], 10.0);
}

#[test]
fn test_write_with_type_labels() {
    let mut frame = Frame::new();
    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn())
        .unwrap();
    atoms
        .insert("y", Array1::from_vec(vec![0.0 as F, 1.0 as F]).into_dyn())
        .unwrap();
    atoms
        .insert("z", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
        .unwrap();
    atoms
        .insert("id", Array1::from_vec(vec![1 as I, 2 as I]).into_dyn())
        .unwrap();
    atoms
        .insert("type", Array1::from_vec(vec![1 as I, 2 as I]).into_dyn())
        .unwrap();
    frame.insert("atoms", atoms);

    frame
        .meta
        .insert("atom_type_labels".to_string(), "1:C,2:O".to_string());
    frame
        .meta
        .insert("box".to_string(), "10.0 10.0 10.0".to_string());
    frame
        .meta
        .insert("box_origin".to_string(), "0.0 0.0 0.0".to_string());

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write_lammps_data(temp_file.path(), &frame).expect("Failed to write");
    let frame2 = read_lammps_data(temp_file.path()).expect("Failed to read");

    let labels = frame2
        .meta
        .get("atom_type_labels")
        .expect("atom type labels");
    assert!(labels.contains("1:C"));
    assert!(labels.contains("2:O"));
}

#[test]
fn test_roundtrip_complex_lammps() {
    let path = crate::test_data::get_test_data_path("lammps-data/labelmap.lmp");
    let frame1 = read_lammps_data(path).expect("Failed to read labelmap.lmp");

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write_lammps_data(temp_file.path(), &frame1).expect("Failed to write LAMMPS data");
    let frame2 = read_lammps_data(temp_file.path()).expect("Failed to read written LAMMPS data");

    let atoms1 = frame1.get("atoms").expect("original atoms");
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms1.nrows(), atoms2.nrows(), "Atom count should match");

    let bonds1 = frame1.get("bonds").expect("original bonds");
    let bonds2 = frame2.get("bonds").expect("roundtrip bonds");
    assert_eq!(bonds1.nrows(), bonds2.nrows(), "Bond count should match");

    if frame1.meta.contains_key("atom_type_labels") {
        assert!(frame2.meta.contains_key("atom_type_labels"));
    }
    if frame1.meta.contains_key("bond_type_labels") {
        assert!(frame2.meta.contains_key("bond_type_labels"));
    }
}
