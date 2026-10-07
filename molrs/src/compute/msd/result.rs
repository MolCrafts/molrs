//! Result types for [`Msd`](super::Msd): one [`MsdResult`] per lag time,
//! collected in an [`MsdTimeSeries`]. Distances squared, (Å²).

use molrs::op::F;
use ndarray::Array1;

use crate::compute::{ComputeResult, DescriptorRow};

/// Per-particle and mean squared displacement at a single time.
#[derive(Debug, Clone)]
pub struct MsdResult {
    /// Per-particle squared displacement from the reference frame.
    pub per_particle: Array1<F>,
    /// System-average mean squared displacement.
    pub mean: F,
}

impl DescriptorRow for MsdResult {
    fn as_row(&self) -> &[F] {
        self.per_particle
            .as_slice()
            .expect("MsdResult::per_particle must be contiguous")
    }
}

/// Time series of per-frame MSD results, aligned with the original frame slice.
///
/// `per_frame[0]` is the reference frame (its MSD is zero); `per_frame[i]` is
/// the MSD at frame `i` relative to frame `0`.
#[derive(Debug, Clone, Default)]
pub struct MsdTimeSeries {
    /// One result per frame, in frame order.
    pub per_frame: Vec<MsdResult>,
}

impl MsdTimeSeries {
    /// Wrap per-lag results (index = lag frame).
    pub fn new(per_frame: Vec<MsdResult>) -> Self {
        Self { per_frame }
    }
    pub fn len(&self) -> usize {
        self.per_frame.len()
    }
    pub fn is_empty(&self) -> bool {
        self.per_frame.is_empty()
    }
}

impl ComputeResult for MsdTimeSeries {}

impl AsRef<[MsdResult]> for MsdTimeSeries {
    fn as_ref(&self) -> &[MsdResult] {
        &self.per_frame
    }
}
