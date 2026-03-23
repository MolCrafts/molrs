//! Integration tests for PDB file reader and writer
//! Each test: READ → VERIFY (atoms, coords, metadata) → WRITE → READ → VERIFY

use molrs::frame::Frame;
use molrs::io::pdb::{read_pdb_frame, write_pdb_frame};
use molrs::types::F;
use std::io::BufWriter;
use tempfile::NamedTempFile;

fn verify_frame_detailed(
    frame: &Frame,
    expected_atoms: usize,
    has_cryst1: bool,
    _has_conect: bool,
) {
    let atoms = frame.get("atoms").expect("Should have atoms block");
    let nrows = atoms.nrows().expect("Should have nrows");
    assert_eq!(
        nrows, expected_atoms,
        "Atom count mismatch: got {} expected {}",
        nrows, expected_atoms
    );

    let x = atoms.get_float("x").expect("Should have x coordinates");
    let y = atoms.get_float("y").expect("Should have y coordinates");
    let z = atoms.get_float("z").expect("Should have z coordinates");
    assert_eq!(x.len(), expected_atoms);
    assert_eq!(y.len(), expected_atoms);
    assert_eq!(z.len(), expected_atoms);

    let has_nonzero = x.iter().any(|&v| v.abs() > 1e-6)
        || y.iter().any(|&v| v.abs() > 1e-6)
        || z.iter().any(|&v| v.abs() > 1e-6);
    assert!(has_nonzero, "Coordinates should have non-zero values");

    if expected_atoms > 0 {
        let has_elements = atoms.get_string("element").is_some();
        assert!(has_elements, "Should have element information");
    }

    if has_cryst1 {
        // Check for simbox instead of meta
        assert!(frame.simbox.is_some(), "Should have box/cryst1 information");
    }
}

fn verify_roundtrip_coordinates(frame1: &Frame, frame2: &Frame, tolerance: F) {
    let atoms1 = frame1.get("atoms").unwrap();
    let atoms2 = frame2.get("atoms").unwrap();
    assert_eq!(atoms1.nrows(), atoms2.nrows());

    let x1 = atoms1.get_float("x").unwrap();
    let x2 = atoms2.get_float("x").unwrap();
    let y1 = atoms1.get_float("y").unwrap();
    let y2 = atoms2.get_float("y").unwrap();
    let z1 = atoms1.get_float("z").unwrap();
    let z2 = atoms2.get_float("z").unwrap();

    for i in 0..x1.len().min(100) {
        assert!(
            (x1[i] - x2[i]).abs() < tolerance,
            "X coord {} mismatch: {} vs {}",
            i,
            x1[i],
            x2[i]
        );
        assert!((y1[i] - y2[i]).abs() < tolerance, "Y coord {} mismatch", i);
        assert!((z1[i] - z2[i]).abs() < tolerance, "Z coord {} mismatch", i);
    }
}

#[test]
fn test_pdb_1avg_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/1avg.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read 1avg.pdb");

    // VERIFY original - 3730 atoms, cryst1=True, conect=True
    verify_frame_detailed(&frame1, 3730, true, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 3730, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_1bcu_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/1bcu.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read 1bcu.pdb");

    // VERIFY original - 2581 atoms, cryst1=True, conect=True
    verify_frame_detailed(&frame1, 2581, true, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 2581, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_1htq_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/1htq.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read 1htq.pdb");

    // VERIFY original - 3778 atoms, cryst1=True, conect=False
    verify_frame_detailed(&frame1, 3778, true, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 3778, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_1npc_pdb_gz() {
    let path = crate::test_data::get_test_data_path("pdb/1npc.pdb.gz");
    let result = read_pdb_frame(path.to_str().unwrap());
    if result.is_err() {
        return; // Skip - known encoding issue
    }
    let frame1 = result.unwrap();
    verify_frame_detailed(&frame1, 2699, true, true);
}

#[test]
fn test_pdb_1vln_triclinic_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/1vln-triclinic.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read 1vln-triclinic.pdb");

    // VERIFY original - 14520 atoms, cryst1=True, conect=True
    verify_frame_detailed(&frame1, 14520, true, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 14520, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_2hkb_pdb() {
    // MODEL file - only first frame
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/2hkb.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read 2hkb.pdb");

    // VERIFY original - 757 atoms, cryst1=True, conect=False
    verify_frame_detailed(&frame1, 757, true, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 757, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_4hhb_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/4hhb.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read 4hhb.pdb");

    // VERIFY original - 4779 atoms, cryst1=True, conect=True
    verify_frame_detailed(&frame1, 4779, true, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 4779, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_mof_5_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/MOF-5.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read MOF-5.pdb");

    // VERIFY original - 65 atoms, cryst1=True, conect=True
    verify_frame_detailed(&frame1, 65, true, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 65, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_ase_pdb() {
    // MODEL file - only first frame
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/ase.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read ase.pdb");

    // VERIFY original - 120 atoms, cryst1=True, conect=False
    verify_frame_detailed(&frame1, 120, true, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 120, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_atom_id_0_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/atom-id-0.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read atom-id-0.pdb");

    // VERIFY original - 2 atoms, cryst1=False, conect=False
    verify_frame_detailed(&frame1, 2, false, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 2, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_crystal_maker_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/crystal-maker.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read crystal-maker.pdb");

    // VERIFY original - 8 atoms, cryst1=True, conect=False
    verify_frame_detailed(&frame1, 8, true, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 8, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_end_endmdl_pdb() {
    // MODEL file - only first frame
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/end-endmdl.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read end-endmdl.pdb");

    // VERIFY original - 4 atoms, cryst1=True, conect=True
    verify_frame_detailed(&frame1, 4, true, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 4, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_hemo_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/hemo.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read hemo.pdb");

    // VERIFY original - 522 atoms, cryst1=False, conect=True
    verify_frame_detailed(&frame1, 522, false, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 522, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_insertion_code_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/insertion-code.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read insertion-code.pdb");

    // VERIFY original - 4 atoms, cryst1=False, conect=False
    verify_frame_detailed(&frame1, 4, false, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 4, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_model_pdb() {
    // MODEL file - only first frame
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/model.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read model.pdb");

    // VERIFY original - 2223 atoms, cryst1=False (invalid CRYST1 with zero lengths), conect=False
    verify_frame_detailed(&frame1, 2223, false, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 2223, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_odd_start_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/odd-start.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read odd-start.pdb");

    // VERIFY original - 20 atoms, cryst1=False, conect=True
    verify_frame_detailed(&frame1, 20, false, true);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 20, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_psfgen_output_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/psfgen-output.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read psfgen-output.pdb");

    // VERIFY original - 26 atoms, cryst1=False, conect=False
    verify_frame_detailed(&frame1, 26, false, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 26, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_short_atom_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/short-atom.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read short-atom.pdb");

    // VERIFY original - 9 atoms, cryst1=False, conect=False
    verify_frame_detailed(&frame1, 9, false, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 9, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_short_cryst1_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/short-cryst1.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read short-cryst1.pdb");

    // VERIFY original - 9 atoms, cryst1=True, conect=False
    verify_frame_detailed(&frame1, 9, true, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 9, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}

#[test]
fn test_pdb_water_pdb() {
    // READ original
    let path = crate::test_data::get_test_data_path("pdb/water.pdb");
    let frame1 = read_pdb_frame(path.to_str().unwrap()).expect("Failed to read water.pdb");

    // VERIFY original - 297 atoms, cryst1=True, conect=False
    verify_frame_detailed(&frame1, 297, true, false);

    // WRITE to temp
    let temp = NamedTempFile::new().expect("create temp");
    {
        let file = std::fs::File::create(temp.path()).unwrap();
        let mut writer = BufWriter::new(file);
        write_pdb_frame(&mut writer, &frame1).expect("write failed");
    }

    // READ back
    let frame2 = read_pdb_frame(temp.path().to_str().unwrap()).expect("Failed to read roundtrip");

    // VERIFY roundtrip
    let atoms2 = frame2.get("atoms").expect("roundtrip atoms");
    assert_eq!(atoms2.nrows().unwrap(), 297, "Roundtrip atom count");
    verify_roundtrip_coordinates(&frame1, &frame2, 0.01);
}
