use crate::types::F;
use ndarray::Array1;

/// Result of an MSD computation on a single frame.
#[derive(Debug, Clone)]
pub struct MSDResult {
    /// Per-particle squared displacement from reference.
    pub per_particle: Array1<F>,
    /// System-average mean squared displacement.
    pub mean: F,
}
