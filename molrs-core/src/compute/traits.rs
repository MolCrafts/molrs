use crate::Frame;
use crate::neighbors::NeighborList;

use super::error::ComputeError;

/// Analysis that only needs a Frame (positions, simbox, topology already inside).
///
/// `&self` is an immutable parameter container (bins, cutoffs, etc.).
/// Returns an owned result struct — no hidden mutable state.
pub trait Compute {
    /// The per-frame result type.
    type Output;

    /// Run the analysis on a single frame.
    fn compute(&self, frame: &Frame) -> Result<Self::Output, ComputeError>;
}

/// Analysis that additionally needs pre-built neighbor pairs.
///
/// The caller builds a [`NeighborList`] (via [`NeighborQuery`] or [`NbListAlgo`])
/// and passes `&NeighborList`. Multiple `PairCompute` instances with the same
/// cutoff can share a single neighbor build.
pub trait PairCompute {
    /// The per-frame result type.
    type Output;

    /// Run the analysis on a single frame with neighbor pairs.
    fn compute(
        &self,
        frame: &Frame,
        neighbors: &NeighborList,
    ) -> Result<Self::Output, ComputeError>;
}
