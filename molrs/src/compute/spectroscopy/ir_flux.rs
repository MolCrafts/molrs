//! IR dipole-flux ACF raw compute — the IR-spectrum raw input.

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use molrs::signal as sig;

/// Raw dipole-flux autocorrelation function — the IR-spectrum raw input.
#[derive(Debug, Clone)]
pub struct IRFluxResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Unnormalized dipole-flux ACF `C(τ) = Σ_d ⟨Ṁ_d(0)·Ṁ_d(τ)⟩`, summed over
    /// the 3 Cartesian components — the ACF the
    /// [`IRSpectrum`](super::IRSpectrum) transform consumes. Units: `[Ṁ]²`.
    pub acf: Array1<f64>,
}

impl ComputeResult for IRFluxResult {}

/// Raw dipole-flux-ACF compute (the IR-spectrum input).
///
/// Lifts the central-difference dipole flux + FFT-ACF + component-sum block (the
/// part *before* windowing), returning only the raw ACF. The window + FFT step
/// is then the [`IRSpectrum`](super::IRSpectrum)
/// [`Fit`](crate::compute::traits::Fit).
#[derive(Debug, Clone, Copy, Default)]
pub struct IRFlux;

/// `(dipole_moments, dt, resolution)` argument bundle for [`IRFlux`].
///
/// `dipole_moments` is `(n_frames, 3)`; the central-difference flux loses the
/// first and last frame, so the effective flux length is `n_frames − 2`.
pub type IRFluxArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for IRFlux {
    type Args<'a> = IRFluxArgs<'a>;
    type Output = IRFluxResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole_moments, dt, resolution) = args;
        let shape = dipole_moments.shape();
        let n_frames = shape[0];
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "dipole_moments (expected (n_frames, 3))",
            });
        }
        if n_frames < 3 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }

        let flux_len = n_frames - 2;
        let max_lag = resolution.min(flux_len.saturating_sub(1));
        let inv_2dt = 0.5 / dt;

        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let flux: Array1<f64> = (1..n_frames - 1)
                .map(|t| (dipole_moments[[t + 1, d]] - dipole_moments[[t - 1, d]]) * inv_2dt)
                .collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &flux, max_lag).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                }
            })?;
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(IRFluxResult {
            lag_times,
            acf: acf_sum,
        })
    }
}
