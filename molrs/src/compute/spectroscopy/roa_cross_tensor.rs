//! ROA polarizability×optical-activity cross-correlation raw compute — the
//! ROA-spectrum raw input.

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use super::raman_tensor::{DIAG_ANISO_WEIGHT, OFFDIAG_ANISO_WEIGHT};
use super::{central_diff_col, cross_correlate};
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;

/// Raw ROA cross-correlation iso/aniso curves — the ROA-spectrum raw input.
#[derive(Debug, Clone)]
pub struct RoaCrossResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: Array1<f64>,
    /// Isotropic ROA cross-correlation of `α̇` (electric polarizability
    /// derivative) with `Ġ′` (magnetic-dipole polarizability / optical-activity
    /// tensor derivative).
    pub acf_iso: Array1<f64>,
    /// Weighted anisotropic ROA cross-correlation (same diagonal/off-diagonal
    /// weighting as the Raman anisotropy).
    pub acf_aniso: Array1<f64>,
}

impl ComputeResult for RoaCrossResult {}

/// Raw ROA cross-tensor compute: cross-correlation of the electric
/// polarizability derivative `α̇` with the optical-activity tensor derivative
/// `Ġ′` (and, in the full theory, the electric-quadrupole tensor `A`).
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_ROA` branch
/// (`src/roa.cpp`): the iso part cross-correlates the polarizability trace with
/// the (negated) `G′` trace, and the anisotropic parts mirror the Raman
/// deviatoric decomposition. Both tensors are passed in Voigt form
/// `[xx, yy, zz, xy, xz, yz]`; the same diagonal (½) / off-diagonal (3) weights
/// as [`RamanTensor`](super::RamanTensor) are used so ROA shares the Raman
/// normal-mode frequencies.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoaCrossTensor;

/// `(electric_pol (n,6), g_tensor (n,6), dt, resolution)` for [`RoaCrossTensor`],
/// both in Voigt notation `[xx, yy, zz, xy, xz, yz]`.
pub type RoaCrossArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl Compute for RoaCrossTensor {
    type Args<'a> = RoaCrossArgs<'a>;
    type Output = RoaCrossResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (el_pol, g_tensor, dt, resolution) = args;
        let n_frames = el_pol.shape()[0];
        if el_pol.shape()[1] != 6 || g_tensor.shape()[1] != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: el_pol.shape()[1].max(g_tensor.shape()[1]),
                what: "ROA (electric_pol, g_tensor) (expected (n_frames, 6) Voigt)",
            });
        }
        if g_tensor.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: g_tensor.shape()[0],
                what: "ROA electric_pol/g_tensor frame counts",
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
        let mut planner = FftPlanner::new();

        // Central-difference derivatives of all six Voigt components.
        let a: Vec<Array1<f64>> = (0..6).map(|c| central_diff_col(el_pol, c, dt)).collect();
        let g: Vec<Array1<f64>> = (0..6).map(|c| central_diff_col(g_tensor, c, dt)).collect();

        // Isotropic: trace × trace.
        let a_iso: Array1<f64> = (0..flux_len)
            .map(|t| (a[0][t] + a[1][t] + a[2][t]) / 3.0)
            .collect();
        let g_iso: Array1<f64> = (0..flux_len)
            .map(|t| (g[0][t] + g[1][t] + g[2][t]) / 3.0)
            .collect();
        let acf_iso = cross_correlate(&mut planner, &a_iso, &g_iso, max_lag);

        // Anisotropic: 3 diagonal differences (weight ½) + 3 off-diagonals
        // (weight 3) — the RamanTensor deviatoric decomposition, cross-correlated.
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        let diag_pairs = [(0usize, 1usize), (1, 2), (2, 0)];
        for (i, j) in diag_pairs {
            let av: Array1<f64> = (0..flux_len).map(|t| a[i][t] - a[j][t]).collect();
            let gv: Array1<f64> = (0..flux_len).map(|t| g[i][t] - g[j][t]).collect();
            let c = cross_correlate(&mut planner, &av, &gv, max_lag);
            for k in 0..=max_lag {
                acf_aniso[k] += DIAG_ANISO_WEIGHT * c[k];
            }
        }
        for off in 3..6 {
            let c = cross_correlate(&mut planner, &a[off], &g[off], max_lag);
            for k in 0..=max_lag {
                acf_aniso[k] += OFFDIAG_ANISO_WEIGHT * c[k];
            }
        }

        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(RoaCrossResult {
            lag_times,
            acf_iso,
            acf_aniso,
        })
    }
}
