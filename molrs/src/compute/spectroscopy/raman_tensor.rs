//! Raman polarizability iso/aniso ACF raw compute — the Raman-spectrum raw
//! input.

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use molrs::signal as sig;

/// Weight for diagonal anisotropy components in the Raman ACF.
pub(super) const DIAG_ANISO_WEIGHT: f64 = 0.5;
/// Weight for off-diagonal anisotropy components in the Raman ACF.
pub(super) const OFFDIAG_ANISO_WEIGHT: f64 = 3.0;
/// Number of anisotropy components (3 diagonal diffs + 3 off-diagonals).
const N_ANISO_COMPS: usize = 6;

/// Raw isotropic + (weighted) anisotropic polarizability-derivative ACFs — the
/// Raman-spectrum raw input.
#[derive(Debug, Clone)]
pub struct RamanTensorResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Isotropic (trace) polarizability-derivative ACF — the `acf_iso` the
    /// [`RamanSpectrum`](super::RamanSpectrum) transform consumes.
    pub acf_iso: Array1<f64>,
    /// Weighted anisotropic (deviatoric) ACF — the `acf_aniso` the
    /// [`RamanSpectrum`](super::RamanSpectrum) transform consumes.
    pub acf_aniso: Array1<f64>,
}

impl ComputeResult for RamanTensorResult {}

/// Raw Raman-tensor-ACF compute (the Raman-spectrum input).
///
/// Lifts the central-difference polarizability derivative + iso/aniso
/// decomposition + FFT-ACF block (the part *before* windowing +
/// cross-section/Bose prefactors), returning only the raw iso/aniso ACFs. The
/// window + FFT + prefactor step is then the
/// [`RamanSpectrum`](super::RamanSpectrum) [`Fit`](crate::compute::traits::Fit).
#[derive(Debug, Clone, Copy, Default)]
pub struct RamanTensor;

/// `(polarizabilities, dt, resolution)` argument bundle for [`RamanTensor`].
///
/// `polarizabilities` is `(n_frames, 6)` in Voigt notation
/// `[α_xx, α_yy, α_zz, α_xy, α_xz, α_yz]`; the central-difference derivative
/// loses the first and last frame.
pub type RamanTensorArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for RamanTensor {
    type Args<'a> = RamanTensorArgs<'a>;
    type Output = RamanTensorResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (polarizabilities, dt, resolution) = args;
        let shape = polarizabilities.shape();
        let n_frames = shape[0];
        if shape[1] != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: shape[1],
                what: "polarizabilities (expected (n_frames, 6) Voigt)",
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

        let mut iso = Vec::with_capacity(flux_len);
        let mut aniso_comps: [Vec<f64>; N_ANISO_COMPS] = [
            Vec::with_capacity(flux_len), // α_xx − α_yy
            Vec::with_capacity(flux_len), // α_yy − α_zz
            Vec::with_capacity(flux_len), // α_zz − α_xx
            Vec::with_capacity(flux_len), // α_xy
            Vec::with_capacity(flux_len), // α_xz
            Vec::with_capacity(flux_len), // α_yz
        ];

        for t in 1..n_frames - 1 {
            let a_prev = polarizabilities.row(t - 1);
            let a_next = polarizabilities.row(t + 1);

            let xx_dot = (a_next[0] - a_prev[0]) * inv_2dt;
            let yy_dot = (a_next[1] - a_prev[1]) * inv_2dt;
            let zz_dot = (a_next[2] - a_prev[2]) * inv_2dt;

            iso.push((xx_dot + yy_dot + zz_dot) / 3.0);
            aniso_comps[0].push(xx_dot - yy_dot);
            aniso_comps[1].push(yy_dot - zz_dot);
            aniso_comps[2].push(zz_dot - xx_dot);
            aniso_comps[3].push((a_next[3] - a_prev[3]) * inv_2dt); // xy
            aniso_comps[4].push((a_next[4] - a_prev[4]) * inv_2dt); // xz
            aniso_comps[5].push((a_next[5] - a_prev[5]) * inv_2dt); // yz
        }

        let mut planner = FftPlanner::new();

        let iso_series = Array1::from_vec(iso);
        let acf_iso =
            sig::acf_fft_with_planner(&mut planner, &iso_series, max_lag).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                }
            })?;

        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        for (c, comp) in aniso_comps.iter_mut().enumerate() {
            let weight = if c < 3 {
                DIAG_ANISO_WEIGHT
            } else {
                OFFDIAG_ANISO_WEIGHT
            };
            let col = Array1::from_vec(std::mem::take(comp));
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                }
            })?;
            for k in 0..=max_lag {
                acf_aniso[k] += weight * acf[k];
            }
        }

        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(RamanTensorResult {
            lag_times,
            acf_iso,
            acf_aniso,
        })
    }
}
