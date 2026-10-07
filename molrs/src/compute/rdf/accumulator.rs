//! Streaming (frame-by-frame) RDF accumulation.

use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array1;

use super::{Rdf, RdfMode, RdfResult};
use crate::compute::ComputeError;

/// Streaming g(r) accumulator (bounded memory).
///
/// Construct from a configured [`Rdf`], feed frames one at a time, finalize
/// once.
///
/// [`RdfAccumulator`] is the bounded-memory streaming counterpart of the batch
/// [`Rdf`](super::Rdf) compute: feed one frame + neighbor list at a time via
/// [`accumulate`](RdfAccumulator::accumulate), then read the normalized g(r)
/// from [`finalize`](RdfAccumulator::finalize). State is O(`n_bins`) — never
/// O(trajectory) — so an arbitrarily long MD run can stream through it.
///
/// It folds exactly the per-frame sums the batch path folds (`n_r`, point
/// counts, volume), in the same order, so `finalize()` reproduces
/// `Rdf::compute` over the same frames bit-for-bit. The batch compute is
/// itself implemented on top of this accumulator — one source of truth for
/// the accumulation math.
#[derive(Debug, Clone)]
pub struct RdfAccumulator {
    rdf: Rdf,
    n_r: Array1<F>,
    n_points: usize,
    n_query_points: usize,
    volume: F,
    n_frames: usize,
    /// Pairing latched from frame 0, count-free: the per-frame point counts go
    /// into `n_points` / `n_query_points`, and only the pairing has to stay
    /// constant across the stream.
    mode: Option<RdfMode>,
}

impl RdfAccumulator {
    /// New accumulator over the given RDF configuration.
    pub fn new(rdf: Rdf) -> Self {
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
    pub fn rdf(&self) -> &Rdf {
        &self.rdf
    }

    /// Fold one frame's pair distances into the running histogram.
    ///
    /// Errors mirror the batch path: a neighbor-list pairing that differs from
    /// frame 0, a table with no `dist_sq` column, a missing `SimBox`, or a
    /// non-finite/non-positive volume all reject the frame (the accumulator
    /// state is left unchanged).
    pub fn accumulate<FA: FrameAccess>(
        &mut self,
        frame: &FA,
        nlist: &Neighbors,
    ) -> Result<(), ComputeError> {
        // Only the pairing has to match frame 0; the point counts the neighbor
        // list carries legitimately vary from frame to frame, which is why the
        // comparison is on the count-free `RdfMode` and not on `QueryMode`.
        let mode = RdfMode::from(nlist.mode());
        match self.mode {
            Some(latched) if latched != mode => {
                return Err(ComputeError::BadShape {
                    expected: format!("{latched:?} (frame 0)"),
                    got: format!("{mode:?} (frame {})", self.n_frames),
                });
            }
            _ => {}
        }
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let vol = simbox.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::volume",
                value: vol.to_string(),
            });
        }
        // Binning first: it is the last thing that can reject the frame, and
        // nothing above it has touched the accumulator yet, so a rejected frame
        // really does leave the state alone.
        self.rdf.accumulate_into(nlist, &mut self.n_r)?;
        if self.mode.is_none() {
            self.mode = Some(mode);
        }
        self.n_points += nlist.n_points();
        self.n_query_points += nlist.n_query_points();
        self.volume += vol;
        self.n_frames += 1;
        Ok(())
    }

    /// Normalize the accumulated histogram into a finalized [`RdfResult`].
    ///
    /// Errors with [`ComputeError::EmptyInput`] when no frame has been
    /// accumulated. The accumulator itself is unchanged and may keep
    /// accumulating afterwards.
    pub fn finalize(&self) -> Result<RdfResult, ComputeError> {
        if self.n_frames == 0 {
            return Err(ComputeError::EmptyInput);
        }
        let mut result = RdfResult {
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
        use crate::compute::ComputeResult;
        result.finalize();
        Ok(result)
    }
}
