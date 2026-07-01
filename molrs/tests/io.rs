//! Integration tests for `molrs-io`.
//!
//! The module tree below mirrors the crate's `src/` layout: one test module per
//! source format module. Every format reader/writer is exercised against **all**
//! real files in `tests-data/<format>/` (located by the local `common` module),
//! never a synthetic happy-path string — see the IO Testing Rules in CLAUDE.md.

#[path = "io/common.rs"]
mod common;

#[path = "io/data/chgcar.rs"]
mod chgcar;
#[path = "io/data/cif.rs"]
mod cif;
#[path = "io/data/cube.rs"]
mod cube;
#[path = "io/data/cube_traj.rs"]
mod cube_traj;
#[path = "io/trajectory/dcd.rs"]
mod dcd;
#[path = "io/data/gro.rs"]
mod gro;
#[path = "io/data/lammps_data.rs"]
mod lammps_data;
#[path = "io/trajectory/lammps_dump.rs"]
mod lammps_dump;
#[path = "io/data/mol2.rs"]
mod mol2;
#[path = "io/data/pdb.rs"]
mod pdb;
#[path = "io/data/poscar.rs"]
mod poscar;
#[path = "io/data/sdf.rs"]
mod sdf;
#[path = "io/streaming.rs"]
mod streaming;
#[path = "io/trajectory/trr.rs"]
mod trr;
#[path = "io/trajectory/xtc.rs"]
mod xtc;
#[path = "io/data/xyz.rs"]
mod xyz;

// SMILES/SMARTS are string parsers (no `tests-data/` files); they mirror
// `src/smiles/` and are gated by the `smiles` feature.
#[path = "io/smiles/mod.rs"]
mod smiles;
