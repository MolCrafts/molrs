//! Transport properties: diffusion, ionic conductivity, and dipolar
//! relaxation — the raw [`Compute`](crate::compute::Compute) observables the
//! [`fitting`](crate::compute::fitting) layer turns into D, σ, and τ_D.

mod correlation;
mod debye_relaxation;
mod einstein_conductivity;
mod einstein_diffusion;
mod green_kubo_conductivity;
mod green_kubo_diffusion;
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
