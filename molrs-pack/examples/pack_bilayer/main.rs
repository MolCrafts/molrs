//! Packmol bilayer example: lipid double layer with water above and below.
//!
//! Based on Packmol's `bilayer.inp` from https://m3g.github.io/packmol/examples.shtml.
//! This version maps Packmol atom-level orientation constraints directly:
//! - atoms 31 32 below plane z=2
//! - atoms 1 2 over plane z=12
//! - atoms 1 2 below plane z=16
//! - atoms 31 32 over plane z=26
//!
//! Run with:
//! ```sh
//! cargo run -p molrs-pack --example pack_bilayer --release
//! ```

use std::fs::create_dir_all;
use std::path::PathBuf;

use molrs::io::pdb::read_pdb_frame;
use molrs_pack::{
    AbovePlaneConstraint, BelowPlaneConstraint, InsideBoxConstraint, Molpack, ProgressHandler,
    Target, XYZHandler,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let base = PathBuf::from(file!())
        .parent()
        .expect("file path has no parent")
        .to_path_buf();
    let water = read_pdb_frame(base.join("water.pdb"))?;
    let lipid = read_pdb_frame(base.join("palmitoil.pdb"))?;

    let water_low = Target::new(water.clone(), 50)
        .with_constraint(InsideBoxConstraint::new(
            [0.0, 0.0, -10.0],
            [40.0, 40.0, 0.0],
        ))
        .with_name("water_low");

    let water_high = Target::new(water, 50)
        .with_constraint(InsideBoxConstraint::new(
            [0.0, 0.0, 28.0],
            [40.0, 40.0, 38.0],
        ))
        .with_name("water_high");

    let lipid_low = Target::new(lipid.clone(), 10)
        .with_constraint(InsideBoxConstraint::new(
            [0.0, 0.0, 0.0],
            [40.0, 40.0, 14.0],
        ))
        .with_constraint_for_atoms(&[31, 32], BelowPlaneConstraint::new([0.0, 0.0, 1.0], 2.0))
        .with_constraint_for_atoms(&[1, 2], AbovePlaneConstraint::new([0.0, 0.0, 1.0], 12.0))
        .with_name("lipid_low");

    let lipid_high = Target::new(lipid, 10)
        .with_constraint(InsideBoxConstraint::new(
            [0.0, 0.0, 14.0],
            [40.0, 40.0, 28.0],
        ))
        .with_constraint_for_atoms(&[1, 2], BelowPlaneConstraint::new([0.0, 0.0, 1.0], 16.0))
        .with_constraint_for_atoms(&[31, 32], AbovePlaneConstraint::new([0.0, 0.0, 1.0], 26.0))
        .with_name("lipid_high");

    let targets = vec![water_low, water_high, lipid_low, lipid_high];
    let mut packer = Molpack::new();
    if std::env::var_os("MOLRS_PACK_EXAMPLE_PROGRESS").is_some() {
        packer = packer.add_handler(ProgressHandler::new());
    }
    if std::env::var_os("MOLRS_PACK_EXAMPLE_XYZ").is_some() {
        let out_dir = base.join("out");
        create_dir_all(&out_dir)?;
        packer = packer.add_handler(XYZHandler::new(out_dir.join("bilayer.xyz"), 10));
    }

    packer.pack(&targets, 800, Some(1_234_567))?;

    Ok(())
}
