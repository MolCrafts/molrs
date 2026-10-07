//! OpenMM's force-field XML (`<ForceField>` with `<AtomTypes>`,
//! `<HarmonicBondForce>`, …), including the OPLS-AA / CL&P / Foyer packs in
//! the same schema.
//!
//! The doors are functions of [`crate::io`]:
//! [`read_openmm_xml_forcefield`](crate::io::read_openmm_xml_forcefield) /
//! [`read_openmm_xml_forcefield_str`](crate::io::read_openmm_xml_forcefield_str),
//! [`write_openmm_xml_forcefield`](crate::io::write_openmm_xml_forcefield) /
//! [`write_openmm_xml_forcefield_str`](crate::io::write_openmm_xml_forcefield_str)
//! — honest inverses, OpenMM's units (nm, kJ/mol, radians) converted at this
//! boundary — and
//! [`read_openmm_xml_opls_typing_str`](crate::io::read_openmm_xml_opls_typing_str),
//! the OPLS-AA typing annotations of the same file. This module holds the
//! format's classes, [`OpenmmXmlReader`] and [`OpenmmXmlWriter`].

pub(crate) mod opls_typing;
pub(crate) mod reader;
pub(crate) mod writer;

pub use reader::OpenmmXmlReader;
pub use writer::OpenmmXmlWriter;
