//! Reciprocal-space potential kernels (PME).
//!
//! Kept as a compilation-unit boundary so the FFT dependency can later be
//! gated out of the `ff` feature (0.15). This is not a ForceField category:
//! PME is registered as the pair style `coul/long/pme`.

pub mod pme;

pub use pme::{PmePotential, pme_ctor};

#[cfg(test)]
mod tests {
    #[test]
    fn fft_gating_reason_is_documented() {
        let src = include_str!("mod.rs");
        assert!(
            src.contains("FFT dependency"),
            "kspace module must document why it remains as an FFT gating boundary"
        );
    }
}
