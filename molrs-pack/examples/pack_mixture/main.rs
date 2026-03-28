//! Packmol mixture example: water + urea in a box.
//!
//! Equivalent to Packmol's `mixture.inp`:
//! ```text
//! tolerance 2.0
//! output mixture.pdb
//! structure water.pdb
//!   number 200
//!   inside box 0. 0. 0. 40. 40. 40.
//! end structure
//! structure urea.pdb
//!   number 80
//!   inside box 0. 0. 0. 40. 40. 40.
//! end structure
//! ```
//!
//! Run with:
//! ```sh
//! cargo run -p molrs-pack --example pack_mixture --release
//! ```

use std::fs::create_dir_all;
use std::path::PathBuf;

use molrs::io::pdb::read_pdb_frame;
use molrs_pack::{InsideBoxConstraint, Molpack, ProgressHandler, Target, XYZHandler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let base = PathBuf::from(file!())
        .parent()
        .expect("file path has no parent")
        .to_path_buf();
    let water = read_pdb_frame(base.join("water.pdb"))?;
    let urea = read_pdb_frame(base.join("urea.pdb"))?;

    let box_constraint = InsideBoxConstraint::cube_from_origin(40.0, [0.0, 0.0, 0.0]);

    let water_target = Target::new(water, 1000)
        .with_constraint(box_constraint.clone())
        .with_name("water");

    let urea_target = Target::new(urea, 400)
        .with_constraint(box_constraint)
        .with_name("urea");

    let mut packer = Molpack::new();
    if std::env::var_os("MOLRS_PACK_EXAMPLE_PROGRESS").is_some() {
        packer = packer.add_handler(ProgressHandler::new());
    }
    if std::env::var_os("MOLRS_PACK_EXAMPLE_XYZ").is_some() {
        let out_dir = base.join("out");
        create_dir_all(&out_dir)?;
        packer = packer.add_handler(XYZHandler::new(out_dir.join("mixture.xyz"), 10));
    }

    let targets = vec![water_target, urea_target];
    packer.pack(&targets, 400, Some(1_234_567u64))?;

    Ok(())
}
