//! Packmol interface example: water/chloroform interface with fixed molecule.
//!
//! Equivalent to Packmol's `interface.inp`:
//! ```text
//! tolerance 2.0
//! output interface.xyz
//! structure water.xyz
//!   number 100
//!   inside box -20. 0. 0. 0. 39. 39.
//! end structure
//! structure chlor.xyz
//!   number 30
//!   inside box 0. 0. 0. 21. 39. 39.
//! end structure
//! structure t3.xyz
//!   centerofmass
//!   fixed 0. 20. 20. 1.57 1.57 1.57
//! end structure
//! ```
//!
//! Run with:
//! ```sh
//! cargo run -p molrs-pack --example pack_interface --release
//! ```

use std::fs::create_dir_all;
use std::path::PathBuf;

use molrs::io::pdb::read_pdb_frame;
use molrs_pack::{
    EarlyStopHandler, InsideBoxConstraint, Molpack, ProgressHandler, Target, XYZHandler,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let base = PathBuf::from(file!())
        .parent()
        .expect("file path has no parent")
        .to_path_buf();
    let water = read_pdb_frame(base.join("water.pdb"))?;
    let chloroform = read_pdb_frame(base.join("chloroform.pdb"))?;
    let t3 = read_pdb_frame(base.join("t3.pdb"))?;

    let water_target = Target::new(water, 200)
        .with_constraint(InsideBoxConstraint::new(
            [-20.0, 0.0, 0.0],
            [0.0, 39.0, 39.0],
        ))
        .with_name("water");

    let chloro_target = Target::new(chloroform, 50)
        .with_constraint(InsideBoxConstraint::new(
            [0.0, 0.0, 0.0],
            [21.0, 39.0, 39.0],
        ))
        .with_name("chloroform");

    let t3_target = Target::new(t3, 1)
        .with_name("t3")
        .with_center_of_mass()
        .fixed_at([0.0, 20.0, 20.0]);

    let out_dir = base.join("out");
    create_dir_all(&out_dir)?;

    let mut packer = Molpack::new()
        .add_handler(ProgressHandler::new())
        .add_handler(EarlyStopHandler::default())
        .add_handler(XYZHandler::new(out_dir.join("interface.xyz"), 10));

    let targets = vec![water_target, chloro_target, t3_target];
    packer.pack(&targets, 200, Some(42u64))?;

    Ok(())
}
