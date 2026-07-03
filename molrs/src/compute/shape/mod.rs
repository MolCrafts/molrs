//! Shape and mass-distribution descriptors of clusters / molecules: center of
//! mass, cluster centers, gyration and inertia tensors, radius of gyration.
//!
//! Per-cluster analyses that consume the labels of an upstream
//! [`Cluster`](crate::compute::Cluster) pass (via
//! [`ClusterResult`](crate::compute::ClusterResult)) and return one descriptor
//! row per cluster per frame; the rows compose into PCA / k-means via
//! [`DescriptorRow`](crate::compute::DescriptorRow). Positions are
//! minimum-image-unwrapped per cluster before any moment is accumulated.
//!
//! | Method | Args | Output (per frame) |
//! |--------|------|--------------------|
//! | [`CenterOfMass`] | `&Vec<ClusterResult>` | `Vec<`[`COMResult`]`>` |
//! | [`ClusterCenters`] | `&Vec<ClusterResult>` | `Vec<`[`ClusterCentersResult`]`>` |
//! | [`GyrationTensor`] | `(&Vec<ClusterResult>, &Vec<ClusterCentersResult>)` | `Vec<`[`GyrationTensorResult`]`>` |
//! | [`InertiaTensor`] | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<`[`InertiaTensorResult`]`>` |
//! | [`RadiusOfGyration`] | `(&Vec<ClusterResult>, &Vec<COMResult>)` | `Vec<`[`RgResult`]`>` |
//!
//! ```ignore
//! let clusters = Cluster::new(1).compute(&frames, &nlists)?;
//! let coms = CenterOfMass::new().compute(&frames, &clusters)?;
//! let rg = RadiusOfGyration::new().compute(&frames, (&clusters, &coms))?;
//! ```

pub mod center_of_mass;
pub mod cluster_centers;
pub mod gyration_tensor;
pub mod inertia_tensor;
pub mod radius_of_gyration;

pub use center_of_mass::{COMResult, CenterOfMass};
pub use cluster_centers::{ClusterCenters, ClusterCentersResult};
pub use gyration_tensor::{GyrationTensor, GyrationTensorResult};
pub use inertia_tensor::{InertiaTensor, InertiaTensorResult};
pub use radius_of_gyration::{RadiusOfGyration, RgResult};
