use crate::types::F;
use ndarray::Array1;

/// Result of a cluster analysis.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Particle → cluster ID (0-indexed). All particles are assigned.
    pub cluster_idx: Array1<i64>,
    /// Number of clusters found.
    pub num_clusters: usize,
    /// Size (particle count) of each cluster, indexed by cluster ID.
    pub cluster_sizes: Vec<usize>,
}

/// Result of per-cluster property analysis.
#[derive(Debug, Clone)]
pub struct ClusterPropsResult {
    /// Center-of-mass per cluster (num_clusters × 3).
    pub centers: Vec<[F; 3]>,
    /// Number of particles per cluster.
    pub sizes: Vec<usize>,
}
