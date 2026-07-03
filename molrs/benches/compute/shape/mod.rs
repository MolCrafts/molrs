//! `shape` category — geometric descriptors over per-frame clusters
//! (center of mass, cluster centers, gyration/inertia tensors, Rg).
//!
//! Mirrors `src/compute/shape/`.

pub mod center_of_mass;
pub mod cluster_centers;
pub mod gyration_tensor;
pub mod inertia_tensor;
pub mod radius_of_gyration;
