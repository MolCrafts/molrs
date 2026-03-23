//! Batch example conformance tests.
//! Run all 5 Packmol-sized examples and validate results.
//!
//! These tests are expensive and require test data files.
//! Use `cargo test -p molrs-pack -- --ignored` to run them.

use std::fs;
use std::path::PathBuf;

use molrs_pack::{
    ExampleCase, Molpack, XYZHandler, build_targets, example_dir_from_manifest,
    validate_from_targets,
};

fn run_case(case: ExampleCase) -> Result<(), Box<dyn std::error::Error>> {
    let base = example_dir_from_manifest(case);
    let targets = build_targets(case, &base)?;
    let out_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("examples_batch")
        .join(case.name());
    fs::create_dir_all(&out_root)?;
    let out_path = out_root.join(case.output_xyz());

    let mut packer = Molpack::new()
        .tolerance(2.0)
        .precision(1e-2)
        .add_handler(XYZHandler::new(&out_path, 10));
    let pack_result = packer.pack(&targets, case.max_loops(), Some(case.seed()))?;
    let coords = &pack_result.positions();
    let report = validate_from_targets(&targets, coords, 2.0, 1e-2);

    if !report.is_valid() {
        return Err(format!("validation failed for {}: {:?}", case.name(), report).into());
    }

    let text = fs::read_to_string(&out_path)?;
    let first_line = text.lines().next().unwrap_or_default().trim();
    let natoms: usize = first_line.parse()?;
    if natoms != pack_result.natoms() {
        return Err(format!(
            "xyz atom count mismatch for {}: {} vs {}",
            case.name(),
            natoms,
            pack_result.natoms()
        )
        .into());
    }

    Ok(())
}

/// Full 5-example conformance run.
///
/// This test is intentionally expensive (especially `pack_spherical`).
#[test]
#[ignore = "expensive: runs all 5 Packmol-sized examples"]
fn batch_run_examples_and_validate() -> Result<(), Box<dyn std::error::Error>> {
    for case in ExampleCase::ALL {
        run_case(case)?;
    }
    Ok(())
}
