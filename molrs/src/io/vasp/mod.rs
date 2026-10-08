//! VASP's structure and volumetric files: POSCAR / CONTCAR and CHGCAR /
//! CHGDIF, which share one header.
//!
//! The doors are functions of [`crate::io`]:
//! [`read_vasp_poscar`](crate::io::read_vasp_poscar),
//! [`read_vasp_poscar_str`](crate::io::read_vasp_poscar_str),
//! [`write_vasp_poscar`](crate::io::write_vasp_poscar),
//! [`write_vasp_poscar_str`](crate::io::write_vasp_poscar_str),
//! [`read_vasp_chgcar`](crate::io::read_vasp_chgcar) and
//! [`read_vasp_chgcar_str`](crate::io::read_vasp_chgcar_str). This module holds
//! the POSCAR classes, [`VaspPoscarReader`] and [`VaspPoscarWriter`], over any
//! byte stream.

pub(super) mod chgcar;
mod header;
pub(super) mod poscar;

pub use poscar::{VaspPoscarReader, VaspPoscarWriter};
