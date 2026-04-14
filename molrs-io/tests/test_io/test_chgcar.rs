//! Integration tests for the CHGCAR reader.
//!
//! Real fixtures are from the `vaspchg_rs` project (MIT-licensed).
//! Bad fixtures are derived programmatically from the real files.
//!
//! File inventory:
//!   chgcar/CHGCAR.Li_nospin   – Li, 1 atom, 32³ grid, no spin
//!   chgcar/CHGCAR.Li_spin     – Li, 1 atom, 48³ grid, ISPIN=2 (total + diff)
//!   chgcar/CHGCAR.NiO_soc     – NiO, 4 atoms, 56³ grid, SOC (4 blocks)
//!   chgcar/CHGCAR.Fe3O4_spin  – Fe3O4, 14 atoms, spin-polarised
//!   chgcar/CHGCAR.Fe3O4_ref   – Fe3O4 reference calculation
//!
//!   chgcar/bad/CHGCAR.grid_truncated      – grid data cut short → must error
//!   chgcar/bad/CHGCAR.natoms_mismatch     – atom count ≠ coord lines → must error
//!   chgcar/bad/CHGCAR.coord_mode_invalid  – coord mode unrecognised → must error
//!   chgcar/bad/CHGCAR.grid_overflow       – extra values after grid (reader is lenient)
//!   chgcar/bad/CHGCAR.aug_truncated       – truncated augmentation block (reader is lenient)
//!   chgcar/bad/CHGCAR.spin_no_mag         – second spin block absent (reads as nospin)

use molrs_io::chgcar::read_chgcar;

fn chgcar_path(name: &str) -> std::path::PathBuf {
    crate::test_data::get_test_data_path(&format!("chgcar/{}", name))
}

fn all_chgcar_good_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("chgcar");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read chgcar test-data dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "No CHGCAR files in tests-data/chgcar/");
    paths
}

#[allow(dead_code)]
fn all_chgcar_bad_files() -> Vec<std::path::PathBuf> {
    let dir = crate::test_data::get_test_data_path("chgcar/bad");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read chgcar/bad dir {:?}: {}", dir, e))
        .filter_map(|entry| {
            let p = entry.ok()?.path();
            if p.is_file() { Some(p) } else { None }
        })
        .collect();
    paths.sort();
    paths
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn assert_valid_chgcar(path: &std::path::Path) {
    let frame = read_chgcar(path.to_str().unwrap())
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
    assert!(frame.simbox.is_some(), "{:?}: missing simbox", path);

    let grid = frame
        .get_grid("chgcar")
        .unwrap_or_else(|| panic!("{:?}: missing chgcar grid", path));
    assert!(
        grid.contains("total"),
        "{:?}: grid has no 'total' array",
        path
    );

    let [nx, ny, nz] = grid.dim;
    assert!(nx > 0 && ny > 0 && nz > 0, "{:?}: grid dim is zero", path);

    let total = grid.get("total").unwrap();
    assert_eq!(
        total.shape(),
        [nx, ny, nz].as_slice(),
        "{:?}: dim/shape mismatch",
        path
    );

    // Integrated charge must be positive (ρ·V_cell, sum/N_grid = electrons).
    let charge: f64 = total.iter().sum::<f64>() / (nx * ny * nz) as f64;
    assert!(
        charge > 0.0,
        "{:?}: integrated charge ≤ 0 ({:.4})",
        path,
        charge
    );
}

// ---------------------------------------------------------------------------
// Good-file tests
// ---------------------------------------------------------------------------

/// Every good CHGCAR file must parse and pass basic structural checks.
#[test]
fn test_all_chgcar_files_parse() {
    for path in all_chgcar_good_files() {
        assert_valid_chgcar(&path);
    }
}

/// CHGCAR.Li_nospin: 1 Li atom, 32³ grid, no spin → 'diff' must be absent.
#[test]
fn test_nospin_structure() {
    let path = chgcar_path("CHGCAR.Li_nospin");
    let frame = read_chgcar(path.to_str().unwrap()).expect("read CHGCAR.nospin");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 1, "nospin has 1 Li atom");
    let symbols = atoms.get_string("symbol").expect("symbol column");
    assert_eq!(symbols[[0]], "Li", "nospin atom is Li");

    let grid = frame.get_grid("chgcar").expect("chgcar grid");
    assert_eq!(grid.dim, [32, 32, 32], "nospin grid is 32³");
    assert!(!grid.contains("diff"), "nospin must have no 'diff'");
}

/// CHGCAR.Li_spin: ISPIN=2 → 'diff' (magnetization density) must be present
/// with the same shape as 'total'.
#[test]
fn test_spin_has_diff_block() {
    let path = chgcar_path("CHGCAR.Li_spin");
    let frame = read_chgcar(path.to_str().unwrap()).expect("read CHGCAR.spin");

    let grid = frame.get_grid("chgcar").expect("chgcar grid");
    assert_eq!(grid.dim, [48, 48, 48], "spin grid is 48³");
    assert!(grid.contains("diff"), "spin CHGCAR must expose 'diff'");
    assert_eq!(
        grid.get("total").unwrap().shape(),
        grid.get("diff").unwrap().shape(),
        "'total' and 'diff' must have the same shape"
    );
}

/// CHGCAR.NiO_soc: 4-atom NiO with spin-orbit coupling.
/// Must parse with Ni and O elements; grid is 56³.
#[test]
fn test_soc_elements_and_grid() {
    let path = chgcar_path("CHGCAR.NiO_soc");
    let frame = read_chgcar(path.to_str().unwrap()).expect("read CHGCAR.NiO_SOC");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(
        atoms.nrows().unwrap(),
        4,
        "NiO_SOC has 4 atoms (2 Ni + 2 O)"
    );
    let symbols = atoms.get_string("symbol").expect("symbol column");
    assert!(symbols.iter().any(|s| s == "Ni"), "NiO_SOC must contain Ni");
    assert!(symbols.iter().any(|s| s == "O"), "NiO_SOC must contain O");

    let grid = frame.get_grid("chgcar").expect("chgcar grid");
    assert_eq!(grid.dim, [56, 56, 56], "NiO_SOC grid is 56³");
}

/// CHGCAR.Fe3O4_spin: 14-atom Fe3O4 (6 Fe + 8 O), non-orthogonal cell.
#[test]
fn test_fe3o4_composition() {
    let path = chgcar_path("CHGCAR.Fe3O4_spin");
    let frame = read_chgcar(path.to_str().unwrap()).expect("read CHGCAR.Fe3O4");

    let atoms = frame.get("atoms").expect("atoms");
    assert_eq!(atoms.nrows().unwrap(), 14, "Fe3O4 has 14 atoms");
    let symbols = atoms.get_string("symbol").expect("symbol column");
    let n_fe = symbols.iter().filter(|s| *s == "Fe").count();
    let n_o = symbols.iter().filter(|s| *s == "O").count();
    assert_eq!(n_fe, 6, "Fe3O4 must have 6 Fe");
    assert_eq!(n_o, 8, "Fe3O4 must have 8 O");
}

/// CHGCAR.Fe3O4_ref and CHGCAR.Fe3O4 must have the same atom count and
/// grid dimensions (they are a calculation/reference pair).
#[test]
fn test_fe3o4_ref_matches_calc() {
    let calc =
        read_chgcar(chgcar_path("CHGCAR.Fe3O4_spin").to_str().unwrap()).expect("read CHGCAR.Fe3O4");
    let refe = read_chgcar(chgcar_path("CHGCAR.Fe3O4_ref").to_str().unwrap())
        .expect("read CHGCAR.Fe3O4_ref");

    assert_eq!(
        calc.get("atoms").unwrap().nrows(),
        refe.get("atoms").unwrap().nrows(),
        "calc and ref must have the same atom count"
    );
    assert_eq!(
        calc.get_grid("chgcar").unwrap().dim,
        refe.get_grid("chgcar").unwrap().dim,
        "calc and ref must have the same grid dimensions"
    );
}

/// Grid dimensions recorded in the file header must match the actual array shape
/// for every good CHGCAR file.
#[test]
fn test_grid_dimensions_match_header() {
    for path in all_chgcar_good_files() {
        let frame =
            read_chgcar(path.to_str().unwrap()).unwrap_or_else(|e| panic!("{:?}: {}", path, e));
        let grid = frame.get_grid("chgcar").unwrap();
        let [nx, ny, nz] = grid.dim;
        assert_eq!(
            grid.get("total").unwrap().shape(),
            [nx, ny, nz].as_slice(),
            "{:?}: dim {:?} ≠ array shape",
            path,
            grid.dim
        );
    }
}

// ---------------------------------------------------------------------------
// Bad-file tests — files that must produce a hard parse error
// ---------------------------------------------------------------------------

/// grid_truncated: grid data ends before the declared nx*ny*nz values.
#[test]
fn test_bad_grid_truncated_fails() {
    let path = chgcar_path("bad/CHGCAR.grid_truncated");
    assert!(
        read_chgcar(path.to_str().unwrap()).is_err(),
        "CHGCAR.grid_truncated must fail to parse"
    );
}

/// natoms_mismatch: header declares 5 atoms but only 1 coordinate line follows.
#[test]
fn test_bad_natoms_mismatch_fails() {
    let path = chgcar_path("bad/CHGCAR.natoms_mismatch");
    assert!(
        read_chgcar(path.to_str().unwrap()).is_err(),
        "CHGCAR.natoms_mismatch must fail to parse"
    );
}

/// coord_mode_invalid: coordinate mode is unrecognised (not Direct/Cartesian).
#[test]
fn test_bad_coord_mode_fails() {
    let path = chgcar_path("bad/CHGCAR.coord_mode_invalid");
    assert!(
        read_chgcar(path.to_str().unwrap()).is_err(),
        "CHGCAR.coord_mode_invalid must fail to parse"
    );
}

// ---------------------------------------------------------------------------
// Bad-file tests — lenient cases (document reader behaviour)
// ---------------------------------------------------------------------------

/// grid_overflow: 5 extra values appended after the declared grid block.
/// Reader reads exactly N = nx*ny*nz values and stops; extra values ignored.
#[test]
fn test_lenient_grid_overflow() {
    let path = chgcar_path("bad/CHGCAR.grid_overflow");
    if let Ok(_frame) = read_chgcar(path.to_str().unwrap()) {
        assert_valid_chgcar(&path);
        let grid = _frame.get_grid("chgcar").unwrap();
        let [nx, ny, nz] = grid.dim;
        assert_eq!(
            grid.get("total").unwrap().len(),
            nx * ny * nz,
            "extra values must not inflate the array"
        );
    }
    // Erroring is also acceptable.
}

/// aug_truncated: augmentation block is cut short.
/// Reader skips augmentation entirely → must parse successfully.
#[test]
fn test_lenient_aug_truncated_parses() {
    let path = chgcar_path("bad/CHGCAR.aug_truncated");
    read_chgcar(path.to_str().unwrap())
        .expect("CHGCAR.aug_truncated should parse (augmentation is skipped)");
    assert_valid_chgcar(&path);
}

/// spin_no_mag: spin CHGCAR with the magnetization block absent.
/// Reader reads the total block and reaches EOF — acceptable as nospin.
#[test]
fn test_lenient_spin_no_mag() {
    let path = chgcar_path("bad/CHGCAR.spin_no_mag");
    if let Ok(frame) = read_chgcar(path.to_str().unwrap()) {
        let grid = frame.get_grid("chgcar").unwrap();
        assert!(grid.contains("total"), "must still have total block");
    }
    // Erroring is also acceptable.
}
