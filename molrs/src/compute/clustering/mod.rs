//! Unsupervised clustering of descriptor rows: [`Kmeans`] over the
//! projections a [`decomposition`](crate::compute::Pca) produces.
//!
//! | Method | Args | Output |
//! |--------|------|--------|
//! | [`Kmeans`] | `&PcaResult` | [`KmeansResult`] — cluster labels + centroids |
//!
//! ```ignore
//! let proj = Pca::new().compute(&[] as &[&Frame], &rows)?;
//! let labels = Kmeans::new(k, max_iter, seed)?.compute(&[] as &[&Frame], &proj)?;
//! ```

mod kmeans;

pub use kmeans::{Kmeans, KmeansResult};
