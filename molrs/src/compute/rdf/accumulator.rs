//! Streaming (frame-by-frame) RDF accumulation.
//!
//! [`RDFAccumulator`] is the bounded-memory streaming counterpart of the batch
//! [`RDF`](super::RDF) compute: feed one frame + neighbor list at a time via
//! [`accumulate`](RDFAccumulator::accumulate), then read the normalized g(r)
//! from [`finalize`](RDFAccumulator::finalize). State is O(`n_bins`) — never
//! O(trajectory) — so an arbitrarily long MD run can stream through it.
//!
//! It folds exactly the per-frame sums the batch path folds (`n_r`, point
//! counts, volume), in the same order, so `finalize()` reproduces
//! `RDF::compute` over the same frames bit-for-bit. The batch compute is
//! itself implemented on top of this accumulator — one source of truth for
//! the accumulation math.

use molrs::spatial::neighbors::{Neighbors, QueryMode};
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;
use ndarray::Array1;

use super::{RDF, RDFResult};
use crate::compute::error::ComputeError;

/// Streaming g(r) accumulator (bounded memory).
///
/// Construct from a configured [`RDF`], feed frames one at a time, finalize
/// once. See the module docs for the equivalence guarantee with the batch
/// [`RDF::compute`](crate::compute::traits::Compute::compute).
#[derive(Debug, Clone)]
pub struct RDFAccumulator {
    rdf: RDF,
    n_r: Array1<F>,
    n_points: usize,
    n_query_points: usize,
    volume: F,
    n_frames: usize,
    mode: Option<QueryMode>,
}

impl RDFAccumulator {
    /// New accumulator over the given RDF configuration.
    pub fn new(rdf: RDF) -> Self {
        let n_bins = rdf.n_bins();
        Self {
            rdf,
            n_r: Array1::zeros(n_bins),
            n_points: 0,
            n_query_points: 0,
            volume: 0.0,
            n_frames: 0,
            mode: None,
        }
    }

    /// Number of frames accumulated so far.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// The RDF configuration this accumulator bins with.
    pub fn rdf(&self) -> &RDF {
        &self.rdf
    }

    /// Fold one frame's pair distances into the running histogram.
    ///
    /// Errors mirror the batch path: a neighbor-list mode that differs from
    /// frame 0, a missing `SimBox`, or a non-finite/non-positive volume all
    /// reject the frame (the accumulator state is left unchanged).
    pub fn accumulate<FA: FrameAccess>(
        &mut self,
        frame: &FA,
        nlist: &Neighbors,
    ) -> Result<(), ComputeError> {
        // Only the query *kind* has to match frame 0; the point counts the mode
        // carries legitimately vary from frame to frame.
        match self.mode {
            Some(mode)
                if std::mem::discriminant(&nlist.mode()) != std::mem::discriminant(&mode) =>
            {
                return Err(ComputeError::BadShape {
                    expected: format!("{mode:?} (frame 0)"),
                    got: format!("{:?} (frame {})", nlist.mode(), self.n_frames),
                });
            }
            _ => {}
        }
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let vol = simbox.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "RDF::volume",
                value: vol.to_string(),
            });
        }
        if self.mode.is_none() {
            self.mode = Some(nlist.mode());
        }
        self.rdf.accumulate_into(nlist, &mut self.n_r);
        self.n_points += nlist.num_points();
        self.n_query_points += nlist.num_query_points();
        self.volume += vol;
        self.n_frames += 1;
        Ok(())
    }

    /// Normalize the accumulated histogram into a finalized [`RDFResult`].
    ///
    /// Errors with [`ComputeError::EmptyInput`] when no frame has been
    /// accumulated. The accumulator itself is unchanged and may keep
    /// accumulating afterwards.
    pub fn finalize(&self) -> Result<RDFResult, ComputeError> {
        if self.n_frames == 0 {
            return Err(ComputeError::EmptyInput);
        }
        let mut result = RDFResult {
            bin_edges: self.rdf.bin_edges.clone(),
            bin_centers: self.rdf.bin_centers.clone(),
            rdf: Array1::zeros(self.rdf.n_bins()),
            n_r: self.n_r.clone(),
            n_points: self.n_points,
            n_query_points: self.n_query_points,
            mode: self.mode.expect("mode latched with the first frame"),
            volume: self.volume,
            r_min: self.rdf.r_min(),
            n_frames: self.n_frames,
            dimensionality: self.rdf.dimensionality(),
            finalized: false,
        };
        use crate::compute::result::ComputeResult;
        result.finalize();
        Ok(result)
    }
}
