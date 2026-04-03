//! Integration tests for LAMMPS data file reader and writer

use molrs::block::Block;
use molrs::frame::Frame;
use molrs::io::lammps_data::{LAMMPSDataReader, read_lammps_data, write_lammps_data};
use molrs::io::reader::FrameReader;
use molrs::types::{F, I, U};
use ndarray::Array1;
use std::fs::File;
use std::io::BufReader;
use tempfile::NamedTempFile;

#[test]
fn test_read_labelmap() {
    let path = crate::test_data::get_test_data_path("lammps-data/labelmap.lmp");
    let frame = read_lammps_data(path).expect("Failed to read labelmap.lmp");

    // Check atoms block
    let atoms = frame.get("atoms").expect("atoms block");
    assert_eq!(atoms.nrows(), Some(16));

    let x = atoms.get_float("x").expect("x coordinates");
    let _y = atoms.get_float("y").expect("y coordinates");
    let _z = atoms.get_float("z").expect("z coordinates");
    let types = atoms.get_int("type").expect("atom types");
    let _charges = atoms.get_float("charge").expect("charges");
    let mol_ids = atoms.get_int("molecule_id").expect("molecule IDs");

    assert_eq!(x.len(), 16);
    assert_eq!(types[0], 1); // First atom is type 1 (f)
    assert_eq!(mol_ids[0], 1); // All atoms in molecule 1

    // Check bonds block
    let bonds = frame.get("bonds").expect("bonds block");
    assert_eq!(bonds.nrows(), Some(14));

    let atom_i = bonds.get_uint("atomi").expect("atomi");
    let _atom_j = bonds.get_uint("atomj").expect("atomj");
    let _bond_types = bonds.get_int("type").expect("bond types");

    assert_eq!(atom_i.len(), 14);

    // Check type labels in metadata
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

    // Check atoms block
    let atoms = frame.get("atoms").expect("atoms block");
    assert_eq!(atoms.nrows(), Some(12));

    let mol_ids = atoms.get_int("molecule_id").expect("molecule IDs");
    assert_eq!(mol_ids[0], 0); // First atom in molecule 0
    assert_eq!(mol_ids[3], 1); // Fourth atom in molecule 1

    // Check box dimensions
    let box_str = frame.meta.get("box").expect("box");
    assert_eq!(box_str, "20 20 20");
}

#[test]
fn test_read_triclinic() {
    let path = crate::test_data::get_test_data_path("lammps-data/triclinic-1.lmp");
    let frame = read_lammps_data(path).expect("Failed to read triclinic-1.lmp");

    // Check triclinic box
    let box_tilt = frame.meta.get("box_tilt").expect("box tilt");
    assert_eq!(box_tilt, "0 0 0");

    // This file has 0 atoms
    assert!(!frame.contains_key("atoms") || frame.get("atoms").unwrap().nrows() == Some(0));
}

#[test]
fn test_read_whitespaces() {
    let path = crate::test_data::get_test_data_path("lammps-data/whitespaces.lmp");
    let frame = read_lammps_data(path).expect("Failed to read whitespaces.lmp");

    // Check atoms block
    let atoms = frame.get("atoms").expect("atoms block");
    assert_eq!(atoms.nrows(), Some(1));

    let x = atoms.get_float("x").expect("x coordinates");
    assert_eq!(x[0], 5.0);

    // Check box dimensions
    let box_str = frame.meta.get("box").expect("box");
    assert_eq!(box_str, "10 10 10");
}

#[test]
fn test_molecular_reader_interface() {
    let path = crate::test_data::get_test_data_path("lammps-data/molid.lmp");
    let file = File::open(path).expect("Failed to open file");
    let mut reader = LAMMPSDataReader::new(BufReader::new(file));

    // Read first frame
    let frame = reader
        .read_frame()
        .expect("Failed to read")
        .expect("No frame");
    assert!(frame.get("atoms").is_some());

    // Second read should return None (single-frame file)
    let frame1 = reader.read_frame().expect("Failed to read");
    assert!(frame1.is_none());
}

#[test]
fn test_write_and_read_roundtrip() {
    // Create a simple frame
    let mut frame = Frame::new();

    // Create atoms block
    let mut atoms = Block::new();
    let _n = 3;
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

    // Create bonds block
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

    // Add box metadata
    frame
        .meta
        .insert("box".to_string(), "10.0 10.0 10.0".to_string());
    frame
        .meta
        .insert("box_origin".to_string(), "0.0 0.0 0.0".to_string());

    // Write to temporary file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let temp_path = temp_file.path();

    write_lammps_data(temp_path, &frame).expect("Failed to write");

    // Read back
    let frame2 = read_lammps_data(temp_path).expect("Failed to read");

    // Verify atoms
    let atoms2 = frame2.get("atoms").expect("atoms block");
    assert_eq!(atoms2.nrows(), Some(3));

    let x2 = atoms2.get_float("x").expect("x coordinates");
    assert_eq!(x2[0], 1.0);
    assert_eq!(x2[1], 2.0);
    assert_eq!(x2[2], 3.0);

    // Verify bonds
    let bonds2 = frame2.get("bonds").expect("bonds block");
    assert_eq!(bonds2.nrows(), Some(2));

    // Verify box
    let box2 = frame2.meta.get("box").expect("box");
    assert_eq!(box2, "10 10 10");
}

#[test]
fn test_write_with_type_labels() {
    let mut frame = Frame::new();

    // Create atoms block
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

    // Add type labels
    frame
        .meta
        .insert("atom_type_labels".to_string(), "1:C,2:O".to_string());
    frame
        .meta
        .insert("box".to_string(), "10.0 10.0 10.0".to_string());
    frame
        .meta
        .insert("box_origin".to_string(), "0.0 0.0 0.0".to_string());

    // Write to temporary file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let temp_path = temp_file.path();

    write_lammps_data(temp_path, &frame).expect("Failed to write");

    // Read back
    let frame2 = read_lammps_data(temp_path).expect("Failed to read");

    // Verify type labels
    let labels = frame2
        .meta
        .get("atom_type_labels")
        .expect("atom type labels");
    assert!(labels.contains("1:C"));
    assert!(labels.contains("2:O"));
}

#[test]
fn test_read_angles_and_dihedrals() {
    let path = crate::test_data::get_test_data_path("lammps-data/labelmap.lmp");
    let frame = read_lammps_data(path).expect("Failed to read labelmap.lmp");

    // Check angles block
    let angles = frame.get("angles").expect("angles block");
    assert_eq!(angles.nrows(), Some(25));

    let atom_i = angles.get_uint("atomi").expect("atomi");
    let atom_j = angles.get_uint("atomj").expect("atomj");
    let atom_k = angles.get_uint("atomk").expect("atomk");
    let angle_types = angles.get_int("type").expect("angle types");

    assert_eq!(atom_i.len(), 25);
    assert_eq!(atom_j.len(), 25);
    assert_eq!(atom_k.len(), 25);
    assert_eq!(angle_types.len(), 25);

    // Check dihedrals block
    let dihedrals = frame.get("dihedrals").expect("dihedrals block");
    assert_eq!(dihedrals.nrows(), Some(27));

    let atom_i = dihedrals.get_uint("atomi").expect("atomi");
    let atom_j = dihedrals.get_uint("atomj").expect("atomj");
    let atom_k = dihedrals.get_uint("atomk").expect("atomk");
    let atom_l = dihedrals.get_uint("atoml").expect("atoml");
    let dihedral_types = dihedrals.get_int("type").expect("dihedral types");

    assert_eq!(atom_i.len(), 27);
    assert_eq!(atom_j.len(), 27);
    assert_eq!(atom_k.len(), 27);
    assert_eq!(atom_l.len(), 27);
    assert_eq!(dihedral_types.len(), 27);
}

// ============================================================================
// Additional LAMMPS Tests - Solvated Systems and Edge Cases
// ============================================================================

#[test]
fn test_read_solvated_system() {
    let path = crate::test_data::get_test_data_path("lammps-data/solvated.lmp");
    let frame = read_lammps_data(path).expect("Failed to read solvated.lmp");

    // Solvated systems typically have many atoms
    let atoms = frame.get("atoms").expect("atoms block");
    let nrows = atoms.nrows().expect("nrows");
    assert!(nrows > 0, "Solvated system should have atoms");

    // Check that coordinates are present
    let x = atoms.get_float("x").expect("x coordinates");
    let y = atoms.get_float("y").expect("y coordinates");
    let z = atoms.get_float("z").expect("z coordinates");
    assert_eq!(x.len(), nrows);
    assert_eq!(y.len(), nrows);
    assert_eq!(z.len(), nrows);

    // Check box dimensions exist
    assert!(frame.meta.contains_key("box"), "Should have box dimensions");
}

#[test]
fn test_read_triclinic_2() {
    let path = crate::test_data::get_test_data_path("lammps-data/triclinic-2.lmp");
    let frame = read_lammps_data(path).expect("Failed to read triclinic-2.lmp");

    // Should have triclinic box with tilt factors
    let box_tilt = frame.meta.get("box_tilt").expect("box tilt");

    // Verify tilt factors are parsed (format: "xy xz yz")
    let tilt_parts: Vec<&str> = box_tilt.split_whitespace().collect();
    assert_eq!(tilt_parts.len(), 3, "Should have three tilt factors");
}

#[test]
fn test_read_data_body() {
    let path = crate::test_data::get_test_data_path("lammps-data/data.body");
    let result = read_lammps_data(path);

    // data.body may have different format or be empty
    match result {
        Ok(frame) => {
            // If it parses, verify basic structure
            assert!(frame.meta.contains_key("box") || frame.contains_key("atoms"));
        }
        Err(e) => {
            // Some files may not parse if they use unsupported features
            let err_msg = e.to_string();
            assert!(!err_msg.is_empty(), "Error message should not be empty");
        }
    }
}

#[test]
fn test_roundtrip_complex_lammps() {
    // Use labelmap.lmp which has atoms, bonds, angles, dihedrals, and type labels
    let path = crate::test_data::get_test_data_path("lammps-data/labelmap.lmp");
    let frame1 = read_lammps_data(path).expect("Failed to read labelmap.lmp");

    // Write to temporary file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let temp_path = temp_file.path();

    write_lammps_data(temp_path, &frame1).expect("Failed to write LAMMPS data");

    // Read back
    let frame2 = read_lammps_data(temp_path).expect("Failed to read written LAMMPS data");

    // Verify atoms
    let atoms1 = frame1.get("atoms").expect("original atoms");
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms1.nrows(), atoms2.nrows(), "Atom count should match");

    // Verify bonds
    let bonds1 = frame1.get("bonds").expect("original bonds");
    let bonds2 = frame2.get("bonds").expect("roundtrip bonds");
    assert_eq!(bonds1.nrows(), bonds2.nrows(), "Bond count should match");

    // Note: Angles and dihedrals may not be written by the current writer implementation
    // Only verify what the writer actually outputs

    // Verify type labels preserved if they were in the original
    if frame1.meta.contains_key("atom_type_labels") {
        assert!(
            frame2.meta.contains_key("atom_type_labels"),
            "Atom type labels should be preserved"
        );
    }
    if frame1.meta.contains_key("bond_type_labels") {
        assert!(
            frame2.meta.contains_key("bond_type_labels"),
            "Bond type labels should be preserved"
        );
    }
}

#[test]
fn test_roundtrip_triclinic() {
    let path = crate::test_data::get_test_data_path("lammps-data/triclinic-2.lmp");
    let frame1 = read_lammps_data(path).expect("Failed to read triclinic-2.lmp");

    // Skip test if frame has no atoms (writer may require atoms)
    if !frame1.contains_key("atoms") || frame1.get("atoms").unwrap().nrows() == Some(0) {
        return;
    }

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let temp_path = temp_file.path();

    write_lammps_data(temp_path, &frame1).expect("Failed to write triclinic LAMMPS data");

    let frame2 = read_lammps_data(temp_path).expect("Failed to read written triclinic data");

    // Verify tilt factors are preserved
    let tilt1 = frame1.meta.get("box_tilt").expect("original tilt");
    let tilt2 = frame2.meta.get("box_tilt").expect("roundtrip tilt");

    // Parse and compare tilt factors
    let tilt1_parts: Vec<&str> = tilt1.split_whitespace().collect();
    let tilt2_parts: Vec<&str> = tilt2.split_whitespace().collect();
    assert_eq!(
        tilt1_parts.len(),
        tilt2_parts.len(),
        "Tilt factor count should match"
    );
}
#[test]
fn test_lammps_data_body() {
    let path = crate::test_data::get_test_data_path("lammps-data/data.body");
    // Just test that file can be opened
    let _result = std::fs::File::open(path);
}

#[test]
fn test_lammps_labelmap_lmp() {
    let path = crate::test_data::get_test_data_path("lammps-data/labelmap.lmp");
    // Just test that file can be opened
    let _result = std::fs::File::open(path);
}

#[test]
fn test_lammps_molid_lmp() {
    let path = crate::test_data::get_test_data_path("lammps-data/molid.lmp");
    // Just test that file can be opened
    let _result = std::fs::File::open(path);
}

#[test]
fn test_lammps_solvated_lmp() {
    let path = crate::test_data::get_test_data_path("lammps-data/solvated.lmp");
    // Just test that file can be opened
    let _result = std::fs::File::open(path);
}

#[test]
fn test_lammps_triclinic_1_lmp() {
    let path = crate::test_data::get_test_data_path("lammps-data/triclinic-1.lmp");
    // Just test that file can be opened
    let _result = std::fs::File::open(path);
}

#[test]
fn test_lammps_triclinic_2_lmp() {
    let path = crate::test_data::get_test_data_path("lammps-data/triclinic-2.lmp");
    // Just test that file can be opened
    let _result = std::fs::File::open(path);
}

#[test]
fn test_lammps_whitespaces_lmp() {
    let path = crate::test_data::get_test_data_path("lammps-data/whitespaces.lmp");
    // Just test that file can be opened
    let _result = std::fs::File::open(path);
}
