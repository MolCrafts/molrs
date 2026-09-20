//! Reciprocal-space potential kernels (PME).
//!
//! Kept as a compilation-unit boundary so the FFT dependency can later be
//! gated out of the `ff` feature (0.15). This is not a ForceField category:
//! PME is registered as the pair style `coul/long/pme`.

pub mod pme;

pub use pme::{PmePotential, pme_ctor};
