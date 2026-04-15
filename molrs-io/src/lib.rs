//! IO modules: readers and format parsers.
pub mod chgcar;
pub mod cube;
pub mod lammps_data;
pub mod lammps_dump;
pub mod pdb;
pub mod reader;
pub mod sdf;
pub mod writer;
pub mod xyz;

#[cfg(feature = "zarr")]
pub mod zarr;
