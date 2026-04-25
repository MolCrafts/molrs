use ndarray::Array1;

use crate::result::ComputeResult;

/// Result of a cluster analysis on one frame.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Particle -> cluster ID (0-indexed). `-1` for unassigned (filtered).
    pub cluster_idx: Array1<i64>,
    /// Number of clusters found.
    pub num_clusters: usize,
    /// Size (particle count) of each cluster, indexed by cluster ID.
    pub cluster_sizes: Vec<usize>,
}

impl ComputeResult for ClusterResult {}
