//! VCD spectrum transform.

use ndarray::Array1;
use rustfft::FftPlanner;

use super::spectra::SpectrumResult;
use super::window_and_fft;
use crate::compute::error::ComputeError;
use crate::compute::traits::Fit;

/// VCD (vibrational circular dichroism) spectrum transform of a **raw VCD
/// cross-correlation** `⟨μ̇(0)·ṁ(τ)⟩` from [`VcdCrossFlux`](super::VcdCrossFlux).
///
/// Identical window + one-sided FFT pipeline as [`IRSpectrum`](super::IRSpectrum)
/// (calls `window_and_fft`), so the cm⁻¹ grid matches
/// IR/Raman exactly — only the supplied cross-correlation differs. The
/// resulting intensities are **signed**: enantiomers produce sign-flipped
/// spectra and an achiral system gives ≈ 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct VcdSpectrum;

impl Fit for VcdSpectrum {
    /// `(acf, dt_fs)` — the raw VCD cross-correlation (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw VCD cross-correlation into a (signed) VCD spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf, dt_fs) = input;
        let n = acf.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }
        let mut planner = FftPlanner::new();
        let (frequencies_cm1, intensities) = window_and_fft(&mut planner, acf, dt_fs)?;
        Ok(SpectrumResult {
            frequencies_cm1,
            intensities,
            resolution: n - 1,
            n_frames: n,
        })
    }
}
