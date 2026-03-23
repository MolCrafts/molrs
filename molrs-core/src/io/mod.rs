//! IO modules: readers and format parsers.
pub mod lammps_data;
pub mod pdb;
pub mod reader;
pub mod writer;
pub mod xyz;

#[cfg(feature = "zarr")]
pub mod zarr;
