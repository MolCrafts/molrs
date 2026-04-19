//! Unified [`Compute`] trait — the single public entry point for any analysis.

use molrs::frame_access::FrameAccess;

use crate::error::ComputeError;
use crate::result::ComputeResult;

/// Run an analysis over a sequence of frames and produce a finalized result.
///
/// A single frame is just a length-1 slice — single-frame, trajectory, and
/// dataset analyses are structurally identical. The `Args` GAT carries any
/// non-frame input the analysis needs (neighbor lists, upstream Compute
/// outputs, masses). Concrete `Compute` impls decide whether to iterate the
/// slice, take the first frame as a reference, or treat the whole sequence as
/// a matrix.
///
/// # Contract
///
/// - `&self` is an **immutable parameter bag** (bin count, cutoff, seed, …).
///   No hidden mutable state — two `compute` calls with identical `frames` +
///   `args` must produce identical [`Output`](Self::Output) values.
/// - `Output` is always [`'static`](::std::marker::Send) +
///   [`Send`] + [`Sync`] + [`Clone`]: it must be movable into a
///   [`Store`](crate::Store) and shareable across downstream Compute nodes.
/// - `Output: ComputeResult` — [`Graph::run`](crate::Graph) calls
///   [`finalize`](ComputeResult::finalize) once per node before insertion, so
///   the returned value is always in its user-facing final form.
///
/// # Example (conceptual)
///
/// ```ignore
/// struct COM;
/// struct COMResult { /* ... */ }
/// impl ComputeResult for COMResult {}
///
/// impl Compute for COM {
///     type Args<'a> = &'a Vec<ClusterResult>;
///     type Output = Vec<COMResult>;
///     fn compute<'a, FA: FrameAccess + 'a>(
///         &self,
///         frames: &[&'a FA],
///         clusters: Self::Args<'a>,
///     ) -> Result<Self::Output, ComputeError> {
///         // one COMResult per frame, aligned by index
///         # unimplemented!()
///     }
/// }
/// ```
pub trait Compute {
    /// Non-frame inputs, with a borrow tied to the frame-slice lifetime.
    type Args<'a>;

    /// Finalized output type.
    type Output: ComputeResult + Clone + Send + Sync + 'static;

    /// Run the analysis. `frames` may be empty; implementations that need at
    /// least one frame return [`ComputeError::EmptyInput`].
    ///
    /// `FA: Sync` lets impls that parallelize across frames (e.g. MSD, COM,
    /// inertia) share `&FA` across rayon threads without an explicit clone.
    /// Both `Frame` and `FrameView` satisfy it; user-defined FrameAccess types
    /// need only `Sync`, not `Send`.
    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError>;
}
