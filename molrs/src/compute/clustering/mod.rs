//! Unsupervised clustering of descriptor rows: [`KMeans`] over the
//! projections a [`decomposition`](crate::compute::Pca) produces.
//!
//! | Method | Args | Output |
//! |--------|------|--------|
//! | [`KMeans`] | `&PcaResult` | [`KMeansResult`] — cluster labels + centroids |
//!
//! ```ignore
//! let proj = Pca::new().compute(&[] as &[&Frame], &rows)?;
//! let labels = KMeans::new(k, max_iter, seed)?.compute(&[] as &[&Frame], &proj)?;
//! ```

mod kmeans;

pub use kmeans::{KMeans, KMeansResult};
