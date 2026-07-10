//! Density-related analyzers ported from `freud.density`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`LocalDensity`] | per-particle number density in a sphere of radius `r_max` (Å) |
//! | [`GaussianDensity`] | grid-smeared density (Gaussian kernel, 2-D / 3-D grids) |
//! | [`CorrelationFunction`] | distance-binned correlation `⟨A_i · B_j⟩(r)` of per-particle fields |
//! | [`SpatialDistribution`] | 3-D number density of a target species in a reference molecule's body-fixed frame ([`GridSpec`]) |
//! | [`SphereVoxelization`] | boolean voxel rasterisation of particles as hard spheres |
//!
//! [`kabsch`] hosts the shared BLAS-free Kabsch superposition helper.
//! [`crate::compute::RDF`] (the radial distribution function) lives in `crate::compute::rdf`
//! and is the prototypical member of this family; the others reuse the same
//! histogram and SimBox conventions.

pub mod correlation_function;
pub mod gaussian_density;
pub mod kabsch;
pub mod local_density;
pub mod spatial;
pub mod sphere_voxelization;

pub use correlation_function::{CorrelationFunction, CorrelationFunctionResult};
pub use gaussian_density::{GaussianDensity, GaussianDensityResult};
pub use local_density::{LocalDensity, LocalDensityResult};
pub use spatial::{GridSpec, SpatialDistribution, SpatialDistributionResult};
pub use sphere_voxelization::{SphereVoxelization, SphereVoxelizationResult};

/// Map a (possibly out-of-range) grid index onto a valid voxel.
///
/// Under periodic boundaries the index wraps via `rem_euclid`; otherwise an
/// index outside `[0, n)` is rejected as `(false, 0)`. Shared by the
/// grid-smearing analyzers ([`GaussianDensity`], [`SphereVoxelization`]).
#[inline]
pub(crate) fn wrap_index(i: isize, n: isize, pbc: bool) -> (bool, usize) {
    if pbc {
        (true, i.rem_euclid(n) as usize)
    } else if i < 0 || i >= n {
        (false, 0)
    } else {
        (true, i as usize)
    }
}
