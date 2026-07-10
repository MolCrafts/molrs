//! Unsupervised learning over per-cluster descriptor rows: PCA projection and
//! k-means clustering (mirrors `molrs-python`'s `compute.ml`).
//!
//! | Method | Args | Output |
//! |--------|------|--------|
//! | [`Pca2`] | `&Vec<T: DescriptorRow>` | [`PcaResult`] — 2-component projection |
//! | [`KMeans`] | `&PcaResult` | [`KMeansResult`] — cluster labels + centroids |
//!
//! Both consume the descriptor rows produced by the
//! [`shape`](crate::compute::shape) analyses (gyration/inertia invariants, Rg,
//! …) via [`DescriptorRow`](crate::compute::DescriptorRow).
//!
//! ```ignore
//! let proj = Pca2::new().compute(&[] as &[&Frame], &rows)?;
//! let labels = KMeans::new(k, max_iter, seed)?.compute(&[] as &[&Frame], &proj)?;
//! ```

pub mod kmeans;
pub mod pca;

pub use kmeans::{KMeans, KMeansResult};
pub use pca::{Pca2, PcaResult};
