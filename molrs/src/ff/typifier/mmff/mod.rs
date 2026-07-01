//! MMFF94 atom/bond/angle/torsion/improper typifier.
//!
//! Annotates an [`Atomistic`](molrs::Atomistic) with MMFF94 type labels and
//! partial charges (the typifier's job). MMFF carries no bespoke energy path —
//! it is a parameter set plus a topology labeler — so the
//! [`build`](MMFFTypifier::build) convenience just materializes the labeled graph
//! to a [`Frame`](molrs::store::frame::Frame) and routes it through the generic
//! [`ForceField::to_potentials`](crate::ff::potential) compile path.
//!
//! # Example
//!
//! ```no_run
//! use molrs::ff::typifier::mmff::MMFFTypifier;
//! use molrs::Atomistic;
//! # fn main() -> Result<(), String> {
//! let mol = Atomistic::new();                            // build or load your molecule
//! let potentials = MMFFTypifier::mmff94()?.build(&mol)?; // typify → to_frame → to_potentials
//! let coords: Vec<f64> = Vec::new();                     // flat [x,y,z, ...]
//! let (energy, _forces) = potentials.calc_energy_forces(&coords);
//! println!("MMFF94 energy = {energy} kcal/mol");
//! # Ok(())
//! # }
//! ```

#![allow(clippy::type_complexity)]

use crate::Frame;
use crate::ff::forcefield::ForceField;
use crate::ff::potential::{Potentials, intramolecular_pairs};
use molrs::Atomistic;

use super::Typifier;

pub(crate) mod classify;
pub(crate) mod frame_builder;
pub mod params;

#[cfg(test)]
mod tests;

// Re-exports
pub use params::{MMFFAtomProp, MMFFParams};

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
        let params = crate::ff::forcefield::xml::read_mmff_params_xml_str(xml)?;
        let ff = crate::ff::forcefield::xml::read_forcefield_xml_str(xml)?;
        Ok(Self { params, ff })
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
    /// Requires [`Atomistic`](molrs::system::atomistic::Atomistic) because MMFF94
    /// typing depends on element symbols, bond orders, and ring membership.
    pub fn build(&self, mol: &Atomistic) -> Result<Potentials, String> {
        let frame = self.typify_frame(mol)?;
        self.ff.to_potentials(&frame)
    }

    /// Typify a molecule into a `to_potentials`-ready [`Frame`].
    ///
    /// Assigns MMFF94 types, then performs the same assembly that
    /// [`build`](Self::build) does before compiling: merges each angle's two
    /// reference bond lengths (`r0_ij`, `r0_kj`) and reference angle (`theta0`)
    /// onto the `angles` block — the per-bond params the stretch-bend kernel
    /// reads — and inserts the intramolecular `pairs` neighbour list. The
    /// returned `Frame` can be passed straight to
    /// [`ForceField::to_potentials`](crate::ff::potential), unlike the bare
    /// labeled frame from [`typify`](Typifier::typify), which still lacks those
    /// columns and would make the `mmff_stbn` kernel error.
    pub fn typify_frame(&self, mol: &Atomistic) -> Result<Frame, String> {
        // typify → labeled Atomistic → Frame (the generic `to_potentials` input).
        // The typifier bakes every per-instance numeric parameter (bond kb/r0,
        // angle ka/theta0, stretch-bend kba + reference r0/theta0, torsion
        // v1/v2/v3, oop koop) onto the frame via the RDKit-validated energy
        // resolvers, so no post-hoc parameter merge is needed here.
        let mut frame = self.typify(mol)?.to_frame();
        // The neighbour list is the consumer's concern: build it here.
        let pairs = intramolecular_pairs(&frame);
        frame.insert("pairs", pairs);
        Ok(frame)
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
    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        frame_builder::annotate_mmff(mol, &self.params)
    }
}
