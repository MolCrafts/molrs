use molrs::frame_access::FrameAccess;

use super::error::ComputeError;

/// Unified analysis trait for all compute operations.
///
/// `Args` is a GAT that specifies what additional data the compute needs
/// beyond the `Frame`:
///
/// - `()` for frame-only analyses (e.g. MSD)
/// - `&'a NeighborList` for pair-based analyses (e.g. RDF, Cluster)
/// - `&'a ClusterResult` for cluster property analyses
///
/// `&self` is an immutable parameter container (bins, cutoffs, masses, etc.).
/// Returns an owned result struct — no hidden mutable state.
///
/// The `frame` parameter accepts any type implementing [`FrameAccess`], which
/// includes both [`Frame`] and [`FrameView`](crate::frame_view::FrameView).
/// Existing callers passing `&Frame` continue to work without changes.
pub trait Compute {
    /// Additional arguments beyond the Frame (e.g. `&NeighborList`, `&ClusterResult`).
    type Args<'a>;
    /// The per-frame result type.
    type Output;

    /// Run the analysis on a single frame with the given arguments.
    fn compute<FA: FrameAccess>(
        &self,
        frame: &FA,
        args: Self::Args<'_>,
    ) -> Result<Self::Output, ComputeError>;
}
