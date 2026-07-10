//! ROA spectrum transform.

use ndarray::Array1;

use super::raman_spectrum::RamanSpectrum;
use super::spectra::RamanSpectrumResult;
use crate::compute::error::ComputeError;
use crate::compute::traits::Fit;

/// ROA (Raman optical activity) spectrum transform of **raw ROA iso/aniso
/// cross-correlations** from [`RoaCrossTensor`](super::RoaCrossTensor).
///
/// Reuses the exact [`RamanSpectrum`] window + FFT + cross-section/Bose pipeline
/// (so the cm⁻¹ grid and normal-mode peaks coincide with Raman); the ROA
/// difference spectrum is signed and flips between enantiomers because the input
/// `α̇ × Ġ′` cross-correlation flips sign.
#[derive(Debug, Clone, Copy)]
pub struct RoaSpectrum {
    /// Laser frequency, cm⁻¹ (cross-section `(ν₀ − ν)⁴ / ν`). `0.0` to skip.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular components.
    pub averaged: bool,
}

impl Fit for RoaSpectrum {
    /// `(acf_iso, acf_aniso, dt_fs)` — the raw ROA iso/aniso cross-correlations
    /// and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, f64);
    type Output = RamanSpectrumResult;

    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        // ROA shares the Raman spectral pipeline (window + FFT + prefactors);
        // only the upstream cross-correlation differs. Reuse, do not duplicate.
        RamanSpectrum {
            incident_frequency_cm1: self.incident_frequency_cm1,
            temperature_k: self.temperature_k,
            averaged: self.averaged,
        }
        .fit(input)
    }
}
