//! Packmol solvprotein example: fixed protein + water + ions in a sphere.
//!
//! Equivalent to Packmol's `solvprotein.inp`:
//! ```text
//! tolerance 2.0
//! structure protein.pdb
//!   number 1
//!   fixed 0. 0. 0. 0. 0. 0.
//!   center
//! end structure
//! structure water.pdb
//!   number 1000
//!   inside sphere 0. 0. 0. 50.
//! end structure
//! structure CLA.pdb
//!   number 20
//!   inside sphere 0. 0. 0. 50.
//! end structure
//! structure SOD.pdb
//!   number 30
//!   inside sphere 0. 0. 0. 50.
//! end structure
//! ```
//!
//! Run with:
//! ```sh
//! cargo run -p molrs-pack --example pack_solvprotein --release
//! ```

use std::fs::create_dir_all;
use std::path::PathBuf;

use molrs::io::pdb::read_pdb_frame;
use molrs_pack::{InsideSphereConstraint, Molpack, ProgressHandler, Target, XYZHandler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let base = PathBuf::from(file!())
        .parent()
        .expect("file path has no parent")
        .to_path_buf();
    let protein = read_pdb_frame(base.join("protein.pdb"))?;
    let water = read_pdb_frame(base.join("water.pdb"))?;
    let sodium = read_pdb_frame(base.join("sodium.pdb"))?;
    let chloride = read_pdb_frame(base.join("chloride.pdb"))?;

    let sphere = InsideSphereConstraint::new(50.0, [0.0, 0.0, 0.0]);

    let protein_target = Target::new(protein, 1)
        .with_name("protein")
        .with_center()
        .fixed_at([0.0, 0.0, 0.0]);

    let water_target = Target::new(water, 1000)
        .with_constraint(sphere.clone())
        .with_name("water");

    let sodium_target = Target::new(sodium, 30)
        .with_constraint(sphere.clone())
        .with_name("sodium");

    let chloride_target = Target::new(chloride, 20)
        .with_constraint(sphere)
        .with_name("chloride");

    let mut packer = Molpack::new();
    if std::env::var_os("MOLRS_PACK_EXAMPLE_PROGRESS").is_some() {
        packer = packer.add_handler(ProgressHandler::new());
    }
    if std::env::var_os("MOLRS_PACK_EXAMPLE_XYZ").is_some() {
        let out_dir = base.join("out");
        create_dir_all(&out_dir)?;
        packer = packer.add_handler(XYZHandler::new(out_dir.join("solvprotein.xyz"), 10));
    }

    let targets = vec![protein_target, water_target, sodium_target, chloride_target];
    packer.pack(&targets, 800, Some(1_234_567))?;

    Ok(())
}
