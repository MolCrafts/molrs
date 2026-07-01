//! Bond-orientational order parameters ported from `freud.order`.
//!
//! Currently implemented:
//! - [`Steinhardt`](steinhardt::Steinhardt) — per-particle `q_ℓ`, `w_ℓ`,
//!   averaged variants, and the underlying `q_ℓm` array. Drives the
//!   downstream `SolidLiquid` and `ContinuousCoordination` analyzers.
//!
//! Future phases will add `Nematic`, `Hexatic`, `Cubatic`, `SolidLiquid`,
//! `ContinuousCoordination`, `RotationalAutocorrelation`.

pub mod continuous_coordination;
pub mod cubatic;
pub mod hexatic;
pub mod nematic;
pub mod reorientation_legendre;
pub mod rotational_autocorrelation;
pub mod solid_liquid;
pub mod steinhardt;

pub use continuous_coordination::{ContinuousCoordination, ContinuousCoordinationResult};
pub use cubatic::{Cubatic, CubaticResult};
pub use hexatic::{Hexatic, HexaticResult};
pub use nematic::{Nematic, NematicResult};
pub use reorientation_legendre::{LegendreReorientation, LegendreReorientationResult};
pub use rotational_autocorrelation::{RotationalAutocorrelation, RotationalAutocorrelationResult};
pub use solid_liquid::{SolidLiquid, SolidLiquidResult};
pub use steinhardt::{Steinhardt, SteinhardtResult, compute_qlm};
