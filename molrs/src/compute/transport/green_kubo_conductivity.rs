//! Green–Kubo conductivity raw compute — the current-ACF route to σ.

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use molrs::signal as sig;

/// Raw current autocorrelation function — the raw portion of the legacy
/// `JacfResult`, with **no** fitted sigma.
#[derive(Debug, Clone)]
pub struct GreenKuboConductivityResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Current ACF `C(τ) = ⟨J(0)·J(τ)⟩` over time origins, identical to
    /// `JacfResult.jacf`. Units: `(e·Å·ps⁻¹)²`.
    pub jacf: Array1<f64>,
}

impl ComputeResult for GreenKuboConductivityResult {}

/// Raw current-ACF compute. Lifts the unbiased windowed-ACF loop from
/// the Green–Kubo conductivity and stops there (no trapezoid, no σ). The
/// σ = (1/(3·V·k_B·T))·∫⟨JJ⟩ step is a downstream
/// [`CumulativeTrapezoid`](crate::compute::fitting::CumulativeTrapezoid) + scale.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboConductivity;

/// `(current, dt, max_correlation_time)` bundle.
pub type GreenKuboConductivityArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for GreenKuboConductivity {
    type Args<'a> = GreenKuboConductivityArgs<'a>;
    type Output = GreenKuboConductivityResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (current, dt, max_correlation_time) = args;
        let shape = current.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "current (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        // ⟨J(0)·J(τ)⟩ over time origins is the sum of the three Cartesian
        // current autocorrelations — evaluate each with the FFT (Wiener–Khinchin)
        // like VACF / Debye, then apply the unbiased 1/(n − τ) normalization.
        let mut planner = FftPlanner::new();
        let mut jacf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (0..n_frames).map(|t| current[[t, d]]).collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                }
            })?;
            for k in 0..=max_lag {
                jacf[k] += acf[k];
            }
        }
        for tau in 0..=max_lag {
            jacf[tau] /= (n_frames - tau) as f64;
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(GreenKuboConductivityResult { lag_times, jacf })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use ndarray::{Array1 as A1, Array2};
    use rand::{RngExt, SeedableRng};

    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn green_kubo_raw_jacf_matches_direct_acf() {
        // ac-010: GreenKuboConductivity.jacf == the direct unbiased current ACF
        // (the raw observable the removed bundled result also carried).
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let current = rng_series(n, 3, 5);
        let max_lag = mct.min(n - 1);
        let mut expected = A1::<f64>::zeros(max_lag + 1);
        for tau in 0..=max_lag {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    s += current[[t, d]] * current[[t + tau, d]];
                }
                acc += s;
            }
            expected[tau] = acc / count as f64;
        }
        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        assert_eq!(raw.jacf.len(), expected.len());
        for k in 0..raw.jacf.len() {
            assert!((raw.jacf[k] - expected[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - k as f64 * dt).abs() < 1e-12);
        }
    }

    #[test]
    fn green_kubo_raw_plus_cumulative_trapezoid_matches_manual_trapezoid() {
        // ac-015: CumulativeTrapezoid on GreenKuboConductivity.jacf reproduces a manual
        // trapezoidal integral, and σ = prefactor·∫/(V·k_B·T) is well-defined
        // (replaces the removed bundled Green–Kubo conductivity).
        use crate::compute::fitting::CumulativeTrapezoid;
        use crate::compute::traits::Fit;
        use molrs::units::constants::{
            ANGSTROM_M, BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C, PICOSECOND_S,
        };

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let current = rng_series(n, 3, 19);

        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        let integ = CumulativeTrapezoid.fit((&raw.jacf, dt, None)).unwrap();

        // Manual cumulative trapezoid of the JACF.
        let mut manual = 0.0;
        for tau in 1..raw.jacf.len() {
            manual += 0.5 * (raw.jacf[tau - 1] + raw.jacf[tau]) * dt;
        }
        let last = integ.integral.len() - 1;
        assert!((integ.integral[last] - manual).abs() < 1e-12);

        // Green–Kubo 1/3 prefactor.
        let prefactor = (E_C * E_C * ANGSTROM_M * ANGSTROM_M / PICOSECOND_S)
            / (3.0 * ANGSTROM_M.powi(3) * K_B_SI);
        let sigma = prefactor * integ.integral[last] / (volume * temperature);
        assert!(sigma.is_finite());
    }
}
