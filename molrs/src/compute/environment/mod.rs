//! Local-environment analyzers ported from `freud.environment`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`AngularSeparationGlobal`] / [`AngularSeparationNeighbor`] | pairwise angular separation between unit quaternions (all-vs-global-refs / per-neighbor) |
//! | [`BondOrientationalOrder`] | 2-D `(θ, φ)` histogram of neighbor bond directions on the unit sphere |
//! | [`LocalBondProjection`] | projection of neighbor bond vectors onto reference directions |
//! | [`LocalDescriptors`] | per-particle spherical-harmonic descriptors of the local neighborhood |
//! | [`EnvironmentMatch`] | environment matching / clustering by neighbor-vector geometry |

mod angular_separation;
mod bond_orientational_order;
mod environment_match;
mod local_bond_projection;
mod local_descriptors;

pub use angular_separation::{
    AngularSeparationGlobal, AngularSeparationGlobalArgs, AngularSeparationGlobalResult,
    AngularSeparationNeighbor, AngularSeparationNeighborArgs, AngularSeparationNeighborResult,
};
pub use bond_orientational_order::{BondOrientationalOrder, BondOrientationalOrderResult};
pub use environment_match::{EnvironmentMatch, EnvironmentMatchResult};
pub use local_bond_projection::{
    LocalBondProjection, LocalBondProjectionArgs, LocalBondProjectionResult,
};
pub use local_descriptors::{LocalDescriptors, LocalDescriptorsResult};
