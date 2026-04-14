//! Molecular typifiers.
//!
//! Bridges [`MolGraph`](crate::molgraph::MolGraph) to typed
//! [`Frame`](crate::frame::Frame) representations by assigning
//! integer type IDs to atoms, bonds, angles, dihedrals, and impropers.

use molrs::frame::Frame;
use molrs::molgraph::MolGraph;

pub mod mmff;

/// A typifier assigns force-field type IDs to a molecular graph and produces
/// a fully typed [`Frame`].
pub trait Typifier {
    /// Typify a molecular graph, returning a [`Frame`] with topology blocks
    /// and type labels ready for force-field compilation.
    fn typify(&self, mol: &MolGraph) -> Result<Frame, String>;
}
