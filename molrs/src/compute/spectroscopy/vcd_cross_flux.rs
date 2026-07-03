//! VCD electric×magnetic dipole cross-correlation raw compute — the
//! VCD-spectrum raw input.

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use super::{central_diff_col, cross_correlate};
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;

/// Raw VCD cross-correlation — the VCD-spectrum raw input.
#[derive(Debug, Clone)]
pub struct VcdCrossResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: Array1<f64>,
    /// VCD cross-correlation `C(τ) = Σ_d ⟨μ̇_d(0)·ṁ_d(τ)⟩` summed over the 3
    /// Cartesian components — the (signed) ACF the
    /// [`VcdSpectrum`](super::VcdSpectrum) transform consumes.
    pub acf: Array1<f64>,
}

impl ComputeResult for VcdCrossResult {}

/// Raw VCD cross-flux compute: cross-correlation of the electric-dipole
/// derivative `μ̇` with the magnetic-dipole derivative `ṁ`.
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_VCD` branch
/// (`src/roa.cpp`): for each Cartesian component it cross-correlates
/// `m_vaDElDip` (μ̇) with `m_vaDMagDip` (ṁ) and sums the three components.
///
/// **Deviation from the spec's literal `⟨μ̇·m⟩`:** reference implementation correlates the two
/// *derivatives* (`μ̇` × `ṁ`); the two conventions differ only by a frequency-
/// domain factor and share the same peak positions and enantiomer sign law.
#[derive(Debug, Clone, Copy, Default)]
pub struct VcdCrossFlux;

/// `(electric_dipole (n,3), magnetic_dipole (n,3), dt, resolution)` for
/// [`VcdCrossFlux`]. Both series are `(n_frames, 3)`; the central-difference
/// derivatives drop the first and last frame.
pub type VcdCrossArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl Compute for VcdCrossFlux {
    type Args<'a> = VcdCrossArgs<'a>;
    type Output = VcdCrossResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (electric, magnetic, dt, resolution) = args;
        let n_frames = electric.shape()[0];
        if electric.shape()[1] != 3 || magnetic.shape()[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: electric.shape()[1].max(magnetic.shape()[1]),
                what: "VCD (electric, magnetic) dipoles (expected (n_frames, 3))",
            });
        }
        if magnetic.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: magnetic.shape()[0],
                what: "VCD electric/magnetic frame counts",
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
        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let mu_dot = central_diff_col(electric, d, dt);
            let m_dot = central_diff_col(magnetic, d, dt);
            let c = cross_correlate(&mut planner, &mu_dot, &m_dot, max_lag);
            for k in 0..=max_lag {
                acf[k] += c[k];
            }
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(VcdCrossResult { lag_times, acf })
    }
}
