//! Bond-orientational order parameters ported from `freud.order`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`Steinhardt`] | per-particle `q_ℓ` / `w_ℓ` (+ averaged variants and the raw `q_ℓm` array) |
//! | [`Hexatic`] | 2-D k-atic bond order `ψ_k` |
//! | [`Nematic`] | nematic director + scalar order parameter from particle orientations |
//! | [`Cubatic`] | cubatic order parameter via simulated annealing |
//! | [`SolidLiquid`] | Frenkel–ten Wolde solid/liquid bond classification from `q_ℓm` |
//! | [`ContinuousCoordination`] | continuous (Voronoi-weighted) coordination numbers |
//! | [`LegendreReorientation`] | Legendre reorientational TCFs `C_ℓ(t)` over lag time |
//! | [`RotationalAutocorrelation`] | rotational autocorrelation of an orientation time series |
//!
//! Every method is a [`Compute`](crate::compute::Compute): construct with its
//! parameters, then `compute(&frames, args)` — `args` carries per-frame
//! neighbor lists / orientations where the method needs them.

mod continuous_coordination;
mod cubatic;
mod hexatic;
mod nematic;
mod reorientation_legendre;
mod rotational_autocorrelation;
mod solid_liquid;
mod steinhardt;

pub use continuous_coordination::{ContinuousCoordination, ContinuousCoordinationResult};
pub use cubatic::{Cubatic, CubaticResult};
pub use hexatic::{Hexatic, HexaticResult};
pub use nematic::{Nematic, NematicResult};
pub use reorientation_legendre::{LegendreReorientation, LegendreReorientationResult};
pub use rotational_autocorrelation::{
    RotationalAutocorrelation, RotationalAutocorrelationArgs, RotationalAutocorrelationResult,
};
pub use solid_liquid::{SolidLiquid, SolidLiquidResult};
pub use steinhardt::{Steinhardt, SteinhardtResult, steinhardt_qlm};
