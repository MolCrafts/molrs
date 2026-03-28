//! Packmol spherical example: double-layered vesicle with water inside and outside.
//!
//! Based on Packmol's `spherical.inp` from https://m3g.github.io/packmol/examples.shtml.
//!
//! Packmol input structure (full counts: 308 / 90 / 300 / 17536):
//! ```text
//! structure water.pdb              # inner water
//!   number 308
//!   inside sphere 0. 0. 0. 13.
//! end structure
//!
//! structure palmitoil.pdb           # inner lipid layer
//!   number 90
//!   atoms 37
//!     inside sphere 0. 0. 0. 14.
//!   end atoms
//!   atoms 5
//!     outside sphere 0. 0. 0. 26.
//!   end atoms
//! end structure
//!
//! structure palmitoil.pdb           # outer lipid layer
//!   number 300
//!   atoms 5
//!     inside sphere 0. 0. 0. 29.
//!   end atoms
//!   atoms 37
//!     outside sphere 0. 0. 0. 41.
//!   end atoms
//! end structure
//!
//! structure water.pdb              # outer water
//!   number 17536
//!   inside box -47.5 -47.5 -47.5 47.5 47.5 47.5
//!   outside sphere 0. 0. 0. 43.
//! end structure
//! ```
//!
//! Lipids use `with_constraint`, same semantics as all other constraints.
//!
//! Run with:
//! ```sh
//! cargo run -p molrs-pack --example pack_spherical --release
//! ```

use std::path::PathBuf;

use molrs::io::pdb::read_pdb_frame;
use molrs_pack::{
    InsideBoxConstraint, InsideSphereConstraint, Molpack, OutsideSphereConstraint, ProgressHandler,
    RegionConstraint, Target,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let base = PathBuf::from(file!())
        .parent()
        .expect("file path has no parent")
        .to_path_buf();
    let water = read_pdb_frame(base.join("water.pdb"))?;
    let lipid = read_pdb_frame(base.join("palmitoil.pdb"))?;

    let origin = [0.0, 0.0, 0.0];

    // 1. Inner water sphere: 308 molecules inside sphere r=13
    let water_inner = Target::new(water.clone(), 308)
        .with_constraint(InsideSphereConstraint::new(13.0, origin))
        .with_name("water_inner");

    // 2. Inner lipid layer: 90 molecules.
    //    Packmol input constrains only specific atoms:
    //    atom 37 inside sphere r=14, atom 5 outside sphere r=26.
    let lipid_inner = Target::new(lipid.clone(), 90)
        .with_constraint_for_atoms(&[37], InsideSphereConstraint::new(14.0, origin))
        .with_constraint_for_atoms(&[5], OutsideSphereConstraint::new(26.0, origin))
        .with_name("lipid_inner");

    // 3. Outer lipid layer: 300 molecules.
    //    Packmol input constrains only specific atoms:
    //    atom 5 inside sphere r=29, atom 37 outside sphere r=41.
    let lipid_outer = Target::new(lipid, 300)
        .with_constraint_for_atoms(&[5], InsideSphereConstraint::new(29.0, origin))
        .with_constraint_for_atoms(&[37], OutsideSphereConstraint::new(41.0, origin))
        .with_name("lipid_outer");

    // 4. Outer water shell: 17536 molecules, box ±47.5, outside sphere r=43
    let water_outer = Target::new(water, 17536)
        .with_constraint(
            InsideBoxConstraint::new([-47.5, -47.5, -47.5], [47.5, 47.5, 47.5])
                .and(OutsideSphereConstraint::new(43.0, origin)),
        )
        .with_name("water_outer");

    // Target order matches Packmol: water_inner → lipid_inner → lipid_outer → water_outer
    let targets = vec![water_inner, lipid_inner, lipid_outer, water_outer];
    let mut packer = Molpack::new();
    if std::env::var_os("MOLRS_PACK_EXAMPLE_PROGRESS").is_some() {
        packer = packer.add_handler(ProgressHandler::new());
    }

    // Match spherical-comment.inp defaults:
    // - nloop defaults to 200 * ntype (ntype = 4 => 800)
    // - seed defaults to 1234567
    packer.pack(&targets, 800, Some(1_234_567))?;

    Ok(())
}
