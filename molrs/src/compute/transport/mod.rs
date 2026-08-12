//! Transport properties: diffusion, ionic conductivity, and dipolar
//! relaxation — the raw [`Compute`](crate::compute::Compute) observables the
//! [`fitting`](crate::compute::fittingting) layer turns into D, σ, and τ_D.
//!
//! Every method here returns **only a raw curve + scalar metadata**; the fit
//! step (slope, integral, Debye τ) is the analyst's explicit, parameterized
//! choice:
//!
//! | Method | Raw output | Downstream fit |
//! |--------|-----------|----------------|
//! | [`VACF`] / [`GreenKuboDiffusion`] | velocity ACF | [`PowerSpectrum`](crate::compute::spectroscopy::PowerSpectrum) (VDOS) / [`CumulativeTrapezoid`](crate::compute::fitting::CumulativeTrapezoid) (D) |
//! | [`EinsteinDiffusion`] | self-MSD curve | [`LinearFit`](crate::compute::fitting::LinearFit) (D = slope/2d) |
//! | [`EinsteinConductivity`] | collective charge-dipole MSD | [`LinearFit`](crate::compute::fitting::LinearFit) (σ) |
//! | [`GreenKuboConductivity`] | current ACF | [`CumulativeTrapezoid`](crate::compute::fitting::CumulativeTrapezoid) (σ) |
//! | [`DebyeRelaxation`] | dipole ACF + ⟨M²⟩ + V/T/BC | [`DebyeFit`] (τ_D, amplitude) / [`DipoleAutocorrelationSpectrum`](crate::compute::spectroscopy::DipoleAutocorrelationSpectrum) |
//! | [`DipoleRateCross`] | `C_{ṀM}` (FD Ṁ × M) | [`DipoleRateCrossSpectrum`](crate::compute::spectroscopy::DipoleRateCrossSpectrum) |
//! | [`OnsagerCorrelation`] | Onsager L_ij displacement correlations | [`LinearFit`](crate::compute::fitting::LinearFit) per pair |
//!
//! [`VACFAccumulator`] is the streaming (frame-by-frame, bounded-memory)
//! counterpart of [`VACF`] for on-the-fly MD analysis. Units follow the MD
//! convention of the caller (time in the `dt` unit, velocities/dipoles as
//! supplied); the fits document the MD→SI prefactors.
//!
//! ```ignore
//! let raw = VACF.compute(&[] as &[&Frame], (&velocities, dt, resolution))?;
//! let d = CumulativeTrapezoid.fit((&raw.acf, dt, None))?; // D = integral/3 in MD units
//! ```

pub mod correlation;
pub mod debye_relaxation;
pub mod einstein_conductivity;
pub mod einstein_diffusion;
pub mod green_kubo_conductivity;
pub mod green_kubo_diffusion;
pub mod jacf;
pub mod onsager;
pub mod vacf;
pub mod vacf_accumulator;

pub use correlation::{
    DipoleRateCross, DipoleRateCrossArgs, DipoleRateCrossResult, apply_unbiased_norm,
    component_means, gradient_axis0_order2, lag_times, unbiased_cartesian_acf,
    unbiased_cartesian_acf_scaled, unbiased_cartesian_xcorr,
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
pub use onsager::{OnsagerCorrelation, OnsagerResult};
pub use vacf::{VACF, VacfArgs, VacfResult};
pub use vacf_accumulator::VACFAccumulator;
