use ndarray::Array1;

use crate::compute::result::ComputeResult;
use molrs::types::U;

/// Result of a cluster analysis on one frame.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Particle -> cluster ID (0-indexed). `-1` for unassigned (filtered).
    pub cluster_idx: Array1<i64>,
    /// Number of clusters found.
    pub num_clusters: usize,
    /// Size (particle count) of each cluster, indexed by cluster ID.
    pub cluster_sizes: Vec<usize>,
    /// The membership keys present in each cluster, indexed by cluster ID
    /// (freud's `cluster_keys`). Empty when clustering without keys; for
    /// key-based grouping each entry holds the single key defining that
    /// cluster.
    pub cluster_keys: Vec<Vec<U>>,
}

impl ComputeResult for ClusterResult {}
