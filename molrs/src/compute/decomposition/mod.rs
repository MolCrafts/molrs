//! Matrix decomposition of descriptor rows: principal component analysis.
//!
//! | Method | Args | Output |
//! |--------|------|--------|
//! | [`Pca`] | `&Vec<T: DescriptorRow>` | [`PcaResult`] — 2-component projection |
//!
//! It consumes the descriptor rows produced by the
//! [`shape`](crate::compute::GyrationTensor) analyses (gyration/inertia
//! invariants, Rg, …) via [`DescriptorRow`](crate::compute::DescriptorRow).

mod pca;

pub use pca::{Pca, PcaResult};
