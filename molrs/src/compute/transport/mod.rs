//! Transport properties: diffusion, ionic conductivity, and dipolar
//! relaxation — the raw [`Compute`](crate::compute::Compute) observables the
//! [`fitting`](crate::compute::fitting) layer turns into D, σ, and τ_D.
//!
//! Every method here returns **only a raw curve + scalar metadata**; the fit
//! step (slope, integral, Debye τ) is the analyst's explicit, parameterized
//! choice:
//!
//! | Method | Raw output | Downstream fit |
//! |--------|-----------|----------------|
//! | [`Vacf`] / [`GreenKuboDiffusion`] | velocity ACF | [`PowerSpectrum`](crate::compute::PowerSpectrum) (VDOS) / [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid) (D) |
//! | [`EinsteinDiffusion`] | self-MSD curve | [`LinearFit`](crate::compute::LinearFit) (D = slope/2d) |
//! | [`EinsteinConductivity`] | collective charge-dipole MSD | [`LinearFit`](crate::compute::LinearFit) (σ) |
//! | [`GreenKuboConductivity`] | current ACF | [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid) (σ) |
//! | [`DebyeRelaxation`] | dipole ACF + ⟨M²⟩ + V/T/BC | [`DebyeFit`] (τ_D, amplitude) / [`DipoleAutocorrelationSpectrum`](crate::compute::DipoleAutocorrelationSpectrum) |
//! | [`DipoleRateCross`] | `C_{ṀM}` (FD Ṁ × M) | [`DipoleRateCrossSpectrum`](crate::compute::DipoleRateCrossSpectrum) |
//! | [`OnsagerCorrelation`] | Onsager L_ij displacement correlations | [`LinearFit`](crate::compute::LinearFit) per pair |
//!
//! [`VacfAccumulator`] is the streaming (frame-by-frame, bounded-memory)
//! counterpart of [`Vacf`] for on-the-fly MD analysis. Units follow the MD
//! convention of the caller (time in the `dt` unit, velocities/dipoles as
//! supplied); the fits document the MD→SI prefactors.
//!
//! ```ignore
//! let raw = VACF.compute(&[] as &[&Frame], (&velocities, dt, resolution))?;
//! let d = CumulativeTrapezoid.fit((&raw.acf, dt, None))?; // D = integral/3 in MD units
//! ```

mod correlation;
mod debye_relaxation;
mod einstein_conductivity;
mod einstein_diffusion;
mod green_kubo_conductivity;
mod green_kubo_diffusion;
mod jacf;
mod onsager;
mod vacf;
mod vacf_accumulator;

pub use correlation::{
    DipoleRateCross, DipoleRateCrossArgs, DipoleRateCrossResult, lag_times,
    unbiased_cartesian_xcorr,
};
pub use debye_relaxation::{
    DebyeFit, DebyeFitResult, DebyeRelaxation, DebyeRelaxationArgs, DebyeRelaxationResult,
    EwaldBoundary,
};
pub use einstein_conductivity::{
    EinsteinConductivity, EinsteinConductivityArgs, EinsteinConductivityResult,
};
pub use einstein_diffusion::{EinsteinDiffusion, EinsteinDiffusionArgs, EinsteinDiffusionResult};
pub use green_kubo_conductivity::{
    GreenKuboConductivity, GreenKuboConductivityArgs, GreenKuboConductivityResult,
};
pub use green_kubo_diffusion::GreenKuboDiffusion;
pub use onsager::{OnsagerCorrelation, OnsagerCorrelationArgs, OnsagerCorrelationResult};
pub use vacf::{Vacf, VacfArgs, VacfResult};
pub use vacf_accumulator::VacfAccumulator;
