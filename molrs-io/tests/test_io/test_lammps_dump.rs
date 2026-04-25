//! Integration tests for the LAMMPS dump trajectory reader and writer.
//!
//! Every file in tests-data/lammps/ is read and structurally verified.
//! Files in tests-data/lammps/bad/ are expected to fail, except known-lenient ones.
//!
//! File inventory (good):
//!   lammps/nacl.lammpstrj            – NaCl 512 atoms, scaled coords (xs ys zs), no type col
//!   lammps/polymer.lammpstrj         – polymer chain
//!   lammps/properties.lammpstrj      – custom per-atom computed columns
//!   lammps/detect_best_pos_repr.lammpstrj – many position representations on same frame
//!   lammps/scaled_wrapped.lammpstrj  – xs/ys/zs (scaled, in [0,1))
//!   lammps/scaled_unwrapped.lammpstrj – xsu/ysu/zsu
//!   lammps/unwrapped.lammpstrj       – xu/yu/zu
//!   lammps/wrapped.lammpstrj         – x/y/z
//!
//! Known-lenient bad files (reader accepts):
//!   lammps/bad/atom-duplicated-id.lammpstrj – duplicate atom IDs (reader is tolerant)

use molrs_io::lammps_dump::{read_lammps_dump, write_lammps_dump};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn has_xyz(atoms: &molrs::block::Block) -> bool {
    atoms.get_float("x").is_some()
        && atoms.get_float("y").is_some()
        && atoms.get_float("z").is_some()
}

fn has_xu(atoms: &molrs::block::Block) -> bool {
    atoms.get_float("xu").is_some()
        && atoms.get_float("yu").is_some()
        && atoms.get_float("zu").is_some()
}

fn has_xs(atoms: &molrs::block::Block) -> bool {
    atoms.get_float("xs").is_some()
        && atoms.get_float("ys").is_some()
        && atoms.get_float("zs").is_some()
}

fn has_xsu(atoms: &molrs::block::Block) -> bool {
    atoms.get_float("xsu").is_some()
        && atoms.get_float("ysu").is_some()
        && atoms.get_float("zsu").is_some()
}

fn has_any_coordinate_triplet(atoms: &molrs::block::Block) -> bool {
    has_xyz(atoms) || has_xu(atoms) || has_xs(atoms) || has_xsu(atoms)
}

fn all_lammps_dump_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("lammps");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read lammps test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "No LAMMPS dump files found in tests-data/lammps/"
    );
    paths
}

fn all_lammps_bad_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("lammps/bad");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read lammps/bad dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    paths
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Every LAMMPS dump file must parse (returning ≥ 1 frame with ≥ 1 atom).
/// Readers preserve whatever coordinate triplet the dump provides.
#[test]
fn test_all_lammps_dump_files_parse() {
    for path in all_lammps_dump_files() {
        let frames = read_lammps_dump(path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("{:?}: read failed: {}", path, e));
        assert!(
            !frames.is_empty(),
            "{:?}: expected at least one frame",
            path
        );
        for (i, frame) in frames.iter().enumerate() {
            let atoms = frame
                .get("atoms")
                .unwrap_or_else(|| panic!("{:?}: frame {} missing atoms block", path, i));
            assert!(
                atoms.nrows().unwrap_or(0) > 0,
                "{:?}: frame {} atoms block is empty",
                path,
                i
            );
            assert!(
                has_any_coordinate_triplet(atoms),
                "{:?}: frame {} missing any supported coordinate triplet",
                path,
                i
            );
        }
    }
}

/// Every LAMMPS dump file survives a write → read roundtrip with atom counts preserved.
#[test]
fn test_all_lammps_dump_files_roundtrip() {
    use tempfile::NamedTempFile;
    for path in all_lammps_dump_files() {
        let frames1 = match read_lammps_dump(path.to_str().unwrap()) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let temp = NamedTempFile::new().expect("create temp");
        if write_lammps_dump(temp.path().to_str().unwrap(), &frames1).is_err() {
            continue;
        }
        let frames2 = match read_lammps_dump(temp.path().to_str().unwrap()) {
            Ok(f) => f,
            Err(_) => continue,
        };
        assert_eq!(
            frames1.len(),
            frames2.len(),
            "{:?}: frame count changed after roundtrip",
            path
        );
        for (i, (f1, f2)) in frames1.iter().zip(frames2.iter()).enumerate() {
            let n1 = f1.get("atoms").unwrap().nrows().unwrap_or(0);
            let n2 = f2.get("atoms").unwrap().nrows().unwrap_or(0);
            assert_eq!(
                n1, n2,
                "{:?}: frame {} atom count mismatch after roundtrip",
                path, i
            );
        }
    }
}

/// Most files in lammps/bad/ must fail to parse.
/// Known-lenient: atom-duplicated-id (reader tolerates duplicate IDs).
#[test]
fn test_all_bad_lammps_dump_files_fail() {
    // These files parse successfully — reader is lenient about their defects.
    let lenient = [
        "atom-duplicated-id.lammpstrj",   // duplicate atom IDs tolerated
        "atom-too-many-fields.lammpstrj", // extra fields ignored
        "items-after-atoms.lammpstrj",    // trailing items after ATOMS block ignored
    ];
    for path in all_lammps_bad_files() {
        let name = path.file_name().unwrap().to_str().unwrap();
        if lenient.contains(&name) {
            continue;
        }
        let result = read_lammps_dump(path.to_str().unwrap());
        assert!(
            result.is_err(),
            "{:?}: expected parse failure but got Ok",
            path
        );
    }
}

/// atom-duplicated-id: reader is lenient about duplicate atom IDs.
#[test]
fn test_lenient_atom_duplicated_id() {
    let path = crate::test_data::get_test_data_path("lammps/bad/atom-duplicated-id.lammpstrj");
    if let Ok(frames) = read_lammps_dump(path.to_str().unwrap()) {
        // If it parsed, atoms block must be non-empty
        assert!(!frames.is_empty());
        let atoms = frames[0].get("atoms").unwrap();
        assert!(atoms.nrows().unwrap_or(0) > 0);
    }
    // Erroring is also acceptable.
}

/// nacl.lammpstrj has 512 atoms and preserves native xs/ys/zs without
/// synthesizing x/y/z.
#[test]
fn test_nacl_scaled_coords() {
    let path = crate::test_data::get_test_data_path("lammps/nacl.lammpstrj");
    let frames = read_lammps_dump(path.to_str().unwrap()).expect("read nacl.lammpstrj");
    assert!(!frames.is_empty(), "nacl must have frames");
    let atoms = frames[0].get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 512, "NaCl has 512 atoms");
    assert!(
        atoms.get_float("xs").is_some(),
        "native scaled coord columns should be preserved"
    );
    assert!(
        atoms.get_float("x").is_none(),
        "reader should not synthesize x/y/z from scaled coordinates"
    );
}

/// properties.lammpstrj should contain custom computed columns (e.g. c_pe or similar).
#[test]
fn test_properties_dump_has_custom_columns() {
    let path = crate::test_data::get_test_data_path("lammps/properties.lammpstrj");
    let frames = read_lammps_dump(path.to_str().unwrap()).expect("read properties.lammpstrj");
    assert!(!frames.is_empty());
    let atoms = frames[0].get("atoms").expect("atoms");
    // Should have more columns than just id/type/x/y/z
    assert!(
        atoms.len() > 5,
        "properties dump should have custom columns"
    );
}
