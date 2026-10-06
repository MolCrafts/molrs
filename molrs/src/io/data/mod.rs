//! Single-structure data file formats: PDB, XYZ, GRO, mol2, SDF, CIF,
//! LAMMPS data (and the `fix bond/react` file set around it), XSF, AMBER
//! inpcrd / prmtop (structure half), and
//! VASP/Gaussian grid formats (CHGCAR, POSCAR, Cube).
//!
//! Force-field files (GROMACS `.top`/`.itp`, AMBER frcmod and the prmtop
//! parameter half, LAMMPS force-field files, OpenMM XML) map a file to the
//! force-field IR and are read and written by `ff::forcefield::{readers,
//! writers}`; `io` owns structure and trajectory formats. The low-level
//! prmtop `%FLAG` parser ([`prmtop::parse_flag_sections`]) is shared with the
//! force-field reader.

pub mod ac;
pub mod chgcar;
pub mod cif;
pub mod cube;
pub mod gro;
pub mod inpcrd;
pub mod lammps_bond_react;
pub mod lammps_data;
pub mod lammps_molecule;
pub mod mol2;
pub mod pdb;
pub mod poscar;
pub mod prep;
pub mod prmtop;
pub mod prmtop_tables;
pub mod sdf;
pub mod vasp_common;
pub mod xsf;
pub mod xyz;
