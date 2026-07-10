//! Local-environment analyzers ported from `freud.environment`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`AngularSeparationGlobal`] / [`AngularSeparationNeighbor`] | pairwise angular separation between unit quaternions (all-vs-global-refs / per-neighbor) |
//! | [`BondOrder`] | 2-D `(θ, φ)` histogram of neighbor bond directions on the unit sphere |
//! | [`LocalBondProjection`] | projection of neighbor bond vectors onto reference directions |
//! | [`LocalDescriptors`] | per-particle spherical-harmonic descriptors of the local neighborhood |
//! | [`MatchEnv`] | environment matching / clustering by neighbor-vector geometry |

pub mod angular_separation;
pub mod bond_order;
pub mod local_bond_projection;
pub mod local_descriptors;
pub mod match_env;

pub use angular_separation::{
    AngularSeparationGlobal, AngularSeparationGlobalResult, AngularSeparationNeighbor,
    AngularSeparationNeighborResult,
};
pub use bond_order::{BondOrder, BondOrderResult};
pub use local_bond_projection::{LocalBondProjection, LocalBondProjectionResult};
pub use local_descriptors::{LocalDescriptors, LocalDescriptorsResult};
pub use match_env::{MatchEnv, MatchEnvResult};
