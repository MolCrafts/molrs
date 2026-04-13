//! MMFF94 atom/bond/angle/torsion/improper typifier.
//!
//! Bridges [`MolGraph`](crate::molgraph::MolGraph) →
//! [`Frame`](crate::frame::Frame) → [`Potentials`](crate::potential::Potentials):
//! given a molecular graph with element symbols and bond connectivity, assigns
//! MMFF94 integer type IDs and compiles potential energy kernels.
//!
//! # Example
//!
//! ```ignore
//! use molrs::typifier::mmff::MMFFTypifier;
//!
//! let typifier = MMFFTypifier::from_xml_str(MMFF94_XML)?;
//! let potentials = typifier.build(&mol)?;
//! let (energy, forces) = potentials.eval(&coords);
//! ```

#![allow(clippy::type_complexity)]

use std::collections::HashMap;

use crate::forcefield::ForceField;
use molrs::frame::Frame;
use molrs::molgraph::{AtomId, MolGraph};
use crate::potential::Potentials;
use molrs::rings::RingInfo;

use super::Typifier;

pub(crate) mod atom_typing;
pub(crate) mod classify;
pub(crate) mod frame_builder;
pub mod params;

#[cfg(test)]
mod tests;

// Re-exports
pub use params::{MMFFAtomProp, MMFFEquiv, MMFFParams};

/// MMFF94 typifier — owns typing metadata and force-field parameters.
///
/// Primary constructor: [`from_xml_str`](Self::from_xml_str) parses both
/// typing metadata (`MMFFParams`) and potential parameters (`ForceField`)
/// from a single XML string. Users never need to manage these separately.
pub struct MMFFTypifier {
    params: MMFFParams,
    ff: ForceField,
}

impl MMFFTypifier {
    /// Create a typifier from an MMFF94 XML string.
    ///
    /// Parses both typing metadata and force-field parameters in one call.
    pub fn from_xml_str(xml: &str) -> Result<Self, String> {
        let params = crate::forcefield::xml::read_mmff_params_xml_str(xml)?;
        let ff = crate::forcefield::xml::read_forcefield_xml_str(xml)?;
        Ok(Self { params, ff })
    }

    /// Create a typifier from an MMFF94 XML file on disk.
    pub fn from_xml(path: &str) -> Result<Self, String> {
        let xml = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        Self::from_xml_str(&xml)
    }

    /// Create a typifier from the embedded MMFF94 parameter set.
    pub fn mmff94() -> Result<Self, String> {
        Self::from_xml_str(molrs::data::MMFF94_XML)
    }

    /// Access the underlying MMFF typing parameters.
    pub fn params(&self) -> &MMFFParams {
        &self.params
    }

    /// Access the underlying force field.
    pub fn ff(&self) -> &ForceField {
        &self.ff
    }

    /// Typify a molecule and compile potentials in one step.
    ///
    /// `mol → Frame → Potentials`. The intermediate `Frame` is not retained.
    ///
    /// Requires [`Atomistic`](molrs::atomistic::Atomistic) because MMFF94
    /// typing depends on element symbols, bond orders, and ring membership.
    pub fn build(&self, mol: &molrs::atomistic::Atomistic) -> Result<Potentials, String> {
        // Deref: &Atomistic → &MolGraph for the Typifier trait call.
        let frame = self.typify(mol)?;
        self.ff.compile(&frame)
    }

    /// Assign MMFF94 atom types to all atoms.
    pub fn assign_atom_types(&self, mol: &MolGraph, ring_info: &RingInfo) -> HashMap<AtomId, u32> {
        atom_typing::assign_atom_types(mol, ring_info, &self.params)
    }

    /// Classify MMFF bond type: 0=normal, 1=delocalized/aromatic.
    pub fn classify_bond_type(&self, t1: u32, t2: u32, bond_order: f64) -> u32 {
        classify::classify_bond_type(t1, t2, bond_order, &self.params)
    }

    /// Classify MMFF angle type from bond types of the two bonds forming the angle.
    pub fn classify_angle_type(&self, bt_ij: u32, bt_jk: u32) -> u32 {
        classify::classify_angle_type(bt_ij, bt_jk)
    }

    /// Classify MMFF torsion type from the three bond types in the dihedral.
    pub fn classify_torsion_type(&self, bt_ij: u32, bt_jk: u32, bt_kl: u32) -> u32 {
        classify::classify_torsion_type(bt_ij, bt_jk, bt_kl)
    }
}

impl Typifier for MMFFTypifier {
    fn typify(&self, mol: &MolGraph) -> Result<Frame, String> {
        frame_builder::build_mmff_frame(mol, &self.params)
    }
}
