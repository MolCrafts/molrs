//! Transport properties: diffusion, ionic conductivity, and dipolar
//! relaxation — the raw [`Compute`](crate::compute::Compute) observables the
//! [`fit`](crate::compute::fit) layer turns into D, σ, and τ_D.
//!
//! Every method here returns **only a raw curve + scalar metadata**; the fit
//! step (slope, integral, Debye τ) is the analyst's explicit, parameterized
//! choice:
//!
//! | Method | Raw output | Downstream fit |
//! |--------|-----------|----------------|
//! | [`VACF`] / [`GreenKuboDiffusion`] | velocity ACF | [`PowerSpectrum`](crate::compute::spectroscopy::PowerSpectrum) (VDOS) / [`RunningIntegral`](crate::compute::fit::RunningIntegral) (D) |
//! | [`EinsteinDiffusion`] | self-MSD curve | [`LinearFit`](crate::compute::fit::LinearFit) (D = slope/2d) |
//! | [`EinsteinConductivity`] | collective charge-dipole MSD | [`LinearFit`](crate::compute::fit::LinearFit) (σ) |
//! | [`GreenKuboConductivity`] | current ACF | [`RunningIntegral`](crate::compute::fit::RunningIntegral) (σ) |
//! | [`DebyeRelaxation`] | dipole ACF + ⟨M²⟩ + V/T/BC | [`DebyeFit`](crate::compute::fit::DebyeFit) (τ_D, amplitude) |
//! | [`OnsagerCorrelation`] | Onsager L_ij displacement correlations | [`LinearFit`](crate::compute::fit::LinearFit) per pair |
//!
//! [`VACFAccumulator`] is the streaming (frame-by-frame, bounded-memory)
//! counterpart of [`VACF`] for on-the-fly MD analysis. Units follow the MD
//! convention of the caller (time in the `dt` unit, velocities/dipoles as
//! supplied); the fits document the MD→SI prefactors.
//!
//! ```ignore
//! let raw = VACF.compute(&[] as &[&Frame], (&velocities, dt, resolution))?;
//! let d = RunningIntegral.fit((&raw.acf, dt, None))?; // D = integral/3 in MD units
//! ```

pub mod debye_relaxation;
pub mod einstein_conductivity;
pub mod einstein_diffusion;
pub mod green_kubo_conductivity;
pub mod green_kubo_diffusion;
pub mod jacf;
pub mod onsager;
pub mod vacf;
pub mod vacf_accumulator;

pub use debye_relaxation::{
    DebyeRelaxation, DebyeRelaxationArgs, DebyeRelaxationResult, EwaldBoundary,
};
pub use einstein_conductivity::{
    EinsteinConductivity, EinsteinConductivityArgs, EinsteinConductivityResult,
};
pub use einstein_diffusion::{EinsteinDiffusion, EinsteinDiffusionArgs, EinsteinDiffusionResult};
pub use green_kubo_conductivity::{
    GreenKuboConductivity, GreenKuboConductivityArgs, GreenKuboConductivityResult,
};
pub use green_kubo_diffusion::GreenKuboDiffusion;
pub use onsager::{OnsagerCorrelation, OnsagerResult};
pub use vacf::{VACF, VacfArgs, VacfResult};
pub use vacf_accumulator::VACFAccumulator;
