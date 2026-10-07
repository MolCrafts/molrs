//! Diffraction / structure-factor analyzers ported from
//! `freud.diffraction`.
//!
//! | Method | Measures |
//! |--------|----------|
//! | [`StaticStructureFactorDebye`] | closed-form Debye `S(k) = N⁻¹ Σ_{i,j} sin(k·r_ij)/(k·r_ij)` on a user k grid (Å⁻¹) |
//! | [`StaticStructureFactorDirect`] | direct k-grid evaluation of `S(k)` from `⟨\|ρ(k)\|²⟩` |
//! | [`DiffractionPattern`] | 2-D FFT diffraction image of a projected frame |

mod debye;
mod diffraction_pattern;
mod direct;

pub use debye::{StaticStructureFactorDebye, StaticStructureFactorDebyeResult};
pub use diffraction_pattern::{DiffractionPattern, DiffractionPatternResult};
pub use direct::{StaticStructureFactorDirect, StaticStructureFactorDirectResult};
