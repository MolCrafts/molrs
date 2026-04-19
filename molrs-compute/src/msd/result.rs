use molrs::types::F;
use ndarray::Array1;

use crate::result::{ComputeResult, DescriptorRow};

/// Per-particle and mean squared displacement at a single time.
#[derive(Debug, Clone)]
pub struct MSDResult {
    /// Per-particle squared displacement from the reference frame.
    pub per_particle: Array1<F>,
    /// System-average mean squared displacement.
    pub mean: F,
}

impl DescriptorRow for MSDResult {
    fn as_row(&self) -> &[F] {
        self.per_particle
            .as_slice()
            .expect("MSDResult::per_particle must be contiguous")
    }
}

/// Time series of per-frame MSD results, aligned with the original frame slice.
///
/// `data[0]` is the reference frame (its MSD is zero); `data[i]` is the MSD at
/// frame `i` relative to frame `0`.
#[derive(Debug, Clone, Default)]
pub struct MSDTimeSeries {
    pub data: Vec<MSDResult>,
}

impl MSDTimeSeries {
    pub fn new(data: Vec<MSDResult>) -> Self {
        Self { data }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl ComputeResult for MSDTimeSeries {}

impl AsRef<[MSDResult]> for MSDTimeSeries {
    fn as_ref(&self) -> &[MSDResult] {
        &self.data
    }
}
