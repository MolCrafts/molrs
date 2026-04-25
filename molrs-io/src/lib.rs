//! IO modules: readers and format parsers.
pub mod chgcar;
pub mod cif;
pub mod cube;
pub mod dcd;
pub mod gro;
pub mod lammps_data;
pub mod lammps_dump;
pub mod mol2;
pub mod pdb;
pub mod poscar;
pub mod reader;
pub mod sdf;
pub mod vasp_common;
pub mod writer;
pub mod xyz;

#[cfg(feature = "smiles")]
pub mod smiles;

#[cfg(feature = "zarr")]
pub mod zarr;
