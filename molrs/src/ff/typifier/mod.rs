//! Molecular typifiers.
//!
//! A typifier assigns force-field type IDs to a molecular graph, returning a
//! **labeled [`Atomistic`]** (atoms typed + charged; bonds/angles/dihedrals/
//! impropers labeled). Materializing it into a [`Frame`](molrs::store::frame::Frame)
//! for `ForceField::to_potentials` is [`Atomistic::to_frame`]'s job, and building
//! the neighbour list is the consumer's — the typifier itself stays on the graph.

use molrs::Atomistic;

pub mod mmff;
pub mod opls;

pub use opls::OplsTypifier;

/// A typifier assigns force-field type IDs to a molecular graph and returns the
/// **labeled graph** (not a [`Frame`]).
pub trait Typifier {
    /// Typify an all-atom molecule, returning a labeled [`Atomistic`]: atoms carry
    /// `type` / `charge`; bonds / angles / dihedrals / impropers carry type
    /// labels. Convert it with [`Atomistic::to_frame`] for force-field compilation.
    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String>;
}
