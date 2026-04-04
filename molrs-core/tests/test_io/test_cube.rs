//! Integration tests for the Gaussian Cube reader/writer.
//!
//! File inventory (from h5cube project, MIT-licensed):
//!   cube/grid20.cube       – 16 atoms, 20³ grid, Bohr units
//!   cube/grid20ang.cube    – same system, Angstrom units (negative voxel count)
//!   cube/grid20mo6-8.cube  – MO variant, 7 atoms, 3 orbitals (indices 6,7,8)
//!   cube/grid25mo.cube     – MO variant, 7 atoms, 1 orbital (index 5)
//!   cube/valtest.cube      – minimal: 2 atoms (H), 1×1×5 grid

use molrs::io::cube::{read_cube, write_cube};
use tempfile::NamedTempFile;

fn cube_path(name: &str) -> std::path::PathBuf {
    crate::test_data::get_test_data_path(&format!("cube/{}", name))
}

fn all_cube_good_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("cube");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read cube test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("cube") {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No .cube files in tests-data/cube/");
    paths
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn assert_valid_cube(path: &std::path::Path) {
    let frame = read_cube(path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{:?}: read failed: {}", path, e));

    let atoms = frame
        .get("atoms")
        .unwrap_or_else(|| panic!("{:?}: missing atoms block", path));
    assert!(
        atoms.nrows().unwrap_or(0) > 0,
        "{:?}: atoms block is empty",
        path
    );
    assert!(
        atoms.get_float("x").is_some(),
        "{:?}: missing x column",
        path
    );
    assert!(
        atoms.get_float("y").is_some(),
        "{:?}: missing y column",
        path
    );
    assert!(
        atoms.get_float("z").is_some(),
        "{:?}: missing z column",
        path
    );
    assert!(
        atoms.get_int("atomic_number").is_some(),
        "{:?}: missing atomic_number column",
        path
    );

    assert!(
        frame.simbox.is_none(),
        "{:?}: cube files should not set simbox",
        path
    );

    let grid = frame
        .get_grid("cube")
        .unwrap_or_else(|| panic!("{:?}: missing cube grid", path));
    let [nx, ny, nz] = grid.dim;
    assert!(nx > 0 && ny > 0 && nz > 0, "{:?}: grid dim is zero", path);

    // Must have at least one array
    assert!(!grid.is_empty(), "{:?}: grid has no arrays", path);

    // Units metadata must be present
    assert!(
        frame.meta.contains_key("cube_units"),
        "{:?}: missing cube_units metadata",
        path
    );
}

// ---------------------------------------------------------------------------
// Good-file tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_cube_files_parse() {
    for path in all_cube_good_files() {
        assert_valid_cube(&path);
    }
}

#[test]
fn test_valtest_structure() {
    let path = cube_path("valtest.cube");
    let frame = read_cube(path.to_str().unwrap()).expect("read valtest.cube");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 2, "valtest has 2 atoms");

    let symbols = atoms.get_string("symbol").expect("symbol column");
    assert_eq!(symbols[[0]], "H");
    assert_eq!(symbols[[1]], "H");

    let grid = frame.get_grid("cube").expect("cube grid");
    assert_eq!(grid.dim, [1, 1, 5], "valtest grid is 1×1×5");
    assert!(grid.contains("density"), "must have 'density' array");

    let density = grid.get("density").unwrap();
    assert_eq!(density.len(), 5);

    // Check known values from the file
    let raw = grid.get_raw("density").unwrap();
    assert!(
        (raw[0] - (-100.0)).abs() < 1e-10,
        "first value should be -100"
    );
    assert!((raw[4] - 100.0).abs() < 1e-10, "last value should be 100");

    assert_eq!(
        frame.meta.get("cube_units").map(|s| s.as_str()),
        Some("bohr"),
        "valtest uses Bohr"
    );
}

#[test]
fn test_grid20_bohr() {
    let path = cube_path("grid20.cube");
    let frame = read_cube(path.to_str().unwrap()).expect("read grid20.cube");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 16, "grid20 has 16 atoms");

    let grid = frame.get_grid("cube").expect("cube grid");
    assert_eq!(grid.dim, [20, 20, 20], "grid20 is 20³");
    assert!(grid.contains("density"), "must have 'density'");

    assert_eq!(
        frame.meta.get("cube_units").map(|s| s.as_str()),
        Some("bohr"),
        "grid20 uses Bohr"
    );
}

#[test]
fn test_grid20ang_angstrom() {
    let path = cube_path("grid20ang.cube");
    let frame = read_cube(path.to_str().unwrap()).expect("read grid20ang.cube");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 16, "grid20ang has 16 atoms");

    let grid = frame.get_grid("cube").expect("cube grid");
    assert_eq!(grid.dim, [20, 20, 20], "grid20ang is 20³");

    assert_eq!(
        frame.meta.get("cube_units").map(|s| s.as_str()),
        Some("angstrom"),
        "grid20ang uses Angstrom"
    );
}

#[test]
fn test_grid20mo_orbitals() {
    let path = cube_path("grid20mo6-8.cube");
    let frame = read_cube(path.to_str().unwrap()).expect("read grid20mo6-8.cube");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 7, "grid20mo has 7 atoms");

    let grid = frame.get_grid("cube").expect("cube grid");
    assert_eq!(grid.dim, [20, 20, 20], "grid20mo is 20³");

    // Must have 3 MO arrays
    assert!(grid.contains("mo_6"), "must have mo_6");
    assert!(grid.contains("mo_7"), "must have mo_7");
    assert!(grid.contains("mo_8"), "must have mo_8");
    assert!(
        !grid.contains("density"),
        "MO files should not have 'density'"
    );

    // Check MO indices metadata
    assert_eq!(
        frame.meta.get("cube_mo_indices").map(|s| s.as_str()),
        Some("6,7,8")
    );
}

#[test]
fn test_grid25mo_single_orbital() {
    let path = cube_path("grid25mo.cube");
    let frame = read_cube(path.to_str().unwrap()).expect("read grid25mo.cube");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 7, "grid25mo has 7 atoms");

    let grid = frame.get_grid("cube").expect("cube grid");
    assert_eq!(grid.dim, [25, 25, 25], "grid25mo is 25³");
    assert!(grid.contains("mo_5"), "must have mo_5");

    assert_eq!(
        frame.meta.get("cube_mo_indices").map(|s| s.as_str()),
        Some("5")
    );
}

// ---------------------------------------------------------------------------
// Roundtrip test
// ---------------------------------------------------------------------------

#[test]
fn test_roundtrip_density() {
    let path = cube_path("valtest.cube");
    let frame1 = read_cube(path.to_str().unwrap()).expect("read");

    let temp = NamedTempFile::new().expect("create temp");
    write_cube(temp.path(), &frame1).expect("write");

    let frame2 = read_cube(temp.path().to_str().unwrap()).expect("re-read");

    // Compare atoms
    let a1 = frame1.get("atoms").unwrap();
    let a2 = frame2.get("atoms").unwrap();
    assert_eq!(a1.nrows(), a2.nrows(), "atom count mismatch");

    let x1 = a1.get_float("x").unwrap();
    let x2 = a2.get_float("x").unwrap();
    for i in 0..a1.nrows().unwrap() {
        assert!(
            (x1[[i]] - x2[[i]]).abs() < 1e-4,
            "x coordinate mismatch at atom {}",
            i
        );
    }

    // Compare grid data
    let g1 = frame1.get_grid("cube").unwrap();
    let g2 = frame2.get_grid("cube").unwrap();
    assert_eq!(g1.dim, g2.dim, "grid dim mismatch");

    let d1 = g1.get_raw("density").unwrap();
    let d2 = g2.get_raw("density").unwrap();
    for (i, (&v1, &v2)) in d1.iter().zip(d2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-4,
            "density mismatch at voxel {}: {} vs {}",
            i,
            v1,
            v2
        );
    }
}

#[test]
fn test_roundtrip_grid20() {
    let path = cube_path("grid20.cube");
    let frame1 = read_cube(path.to_str().unwrap()).expect("read");

    let temp = NamedTempFile::new().expect("create temp");
    write_cube(temp.path(), &frame1).expect("write");

    let frame2 = read_cube(temp.path().to_str().unwrap()).expect("re-read");

    let a1 = frame1.get("atoms").unwrap();
    let a2 = frame2.get("atoms").unwrap();
    assert_eq!(a1.nrows(), a2.nrows());

    let g1 = frame1.get_grid("cube").unwrap();
    let g2 = frame2.get_grid("cube").unwrap();
    assert_eq!(g1.dim, g2.dim);

    let d1 = g1.get_raw("density").unwrap();
    let d2 = g2.get_raw("density").unwrap();
    assert_eq!(d1.len(), d2.len());

    // Check relative tolerance for non-zero values
    for (i, (&v1, &v2)) in d1.iter().zip(d2.iter()).enumerate() {
        let tol = if v1.abs() > 1e-20 {
            v1.abs() * 1e-4
        } else {
            1e-20
        };
        assert!(
            (v1 - v2).abs() < tol,
            "density mismatch at voxel {}: {:.6E} vs {:.6E}",
            i,
            v1,
            v2
        );
    }
}
