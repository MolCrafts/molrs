//! Molecular typifiers.
//!
//! A typifier assigns force-field type IDs to a molecular graph, returning the
//! same concrete graph type with labels attached. Materializing a typed graph
//! into a [`Frame`](molrs::store::frame::Frame) for `ForceField::to_potentials`
//! is the graph's `to_frame` job; typifiers stay on the graph boundary.

pub mod am1bcc;
pub mod estimate;
pub mod mmff;
pub mod opls;
pub(crate) mod topology;

pub use am1bcc::{
    AM1BCCTypifier, AM1ChargeTypifier, BCCAtomTypifier, BCCCorrectionTable, BCCCorrector,
    BccParameterSet,
};
pub use estimate::{ParameterEstimator, ParameterInterpolator, TypifierParameterContext};
pub use opls::OPLSAATypifier;

/// A graph typifier.
///
/// `Mol` is the concrete graph type the typifier accepts and returns. Force
/// fields that only make sense for all-atom systems set `Mol = Atomistic`;
/// future united-atom or coarse-grained typifiers can set a different graph
/// type while preserving the `typify(T) -> T` contract.
pub trait Typifier {
    type Mol;

    /// Typify `mol`, returning a labeled graph of the same concrete type.
    fn typify(&self, mol: &Self::Mol) -> Result<Self::Mol, String>;
}
