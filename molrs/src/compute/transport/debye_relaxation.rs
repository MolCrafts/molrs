//! Debye dipole-relaxation raw compute — the dipole-ACF route to ε(ω) and τ_D.

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use molrs::signal as sig;

/// Ewald boundary condition under which the dipole fluctuations were sampled.
///
/// The Debye / Neumann–Kirkwood amplitude `ε₀ − ε∞` depends on this boundary
/// condition, so it travels with the raw result (spec invariant (c)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EwaldBoundary {
    /// Conducting / "tin-foil" boundary (ε' = ∞) — the common MD default.
    TinFoil,
    /// Vacuum boundary (ε' = 1).
    Vacuum,
}

impl EwaldBoundary {
    /// Parse a boundary-condition name (case-insensitive): `"tinfoil"` /
    /// `"tin-foil"` / `"tin_foil"` / `"conducting"` → [`TinFoil`](Self::TinFoil),
    /// `"vacuum"` → [`Vacuum`](Self::Vacuum).
    pub fn from_name(name: &str) -> Result<Self, ComputeError> {
        match name.to_ascii_lowercase().as_str() {
            "tinfoil" | "tin-foil" | "tin_foil" | "conducting" => Ok(Self::TinFoil),
            "vacuum" => Ok(Self::Vacuum),
            other => Err(ComputeError::OutOfRange {
                field: "EwaldBoundary",
                value: other.to_string(),
            }),
        }
    }
}

/// Raw dipole ACF plus the scalar metadata the Debye amplitude needs.
///
/// Unlike the spectral/diffusion raw results, this carries an **unnormalized**
/// ACF and the zero-lag variance ⟨M(0)²⟩ explicitly (invariant (b)): the
/// normalized Φ(t) gives only the relaxation *shape*/τ via
/// [`DebyeFit`](crate::compute::fit::DebyeFit); the amplitude `ε₀ − ε∞` comes
/// from ⟨M²⟩ together with V, T, and the Ewald boundary condition
/// (invariant (c)).
///
/// All four metadata fields are non-optional, so a `DebyeRelaxationResult`
/// **cannot be constructed without them**.
#[derive(Debug, Clone)]
pub struct DebyeRelaxationResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// **Unnormalized** dipole ACF `C(τ) = ⟨δM(0)·δM(τ)⟩` summed over the 3
    /// Cartesian components. Units: `(e·Å)²`.
    pub acf: Array1<f64>,
    /// Zero-lag variance ⟨M(0)²⟩ = `acf[0]`, the Debye amplitude scale
    /// (invariant b). Units: `(e·Å)²`.
    pub zero_lag_variance: f64,
    /// System volume V (invariant c). Units: `Å³`.
    pub volume: f64,
    /// Temperature T (invariant c). Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition the fluctuations were sampled under
    /// (invariant c).
    pub boundary: EwaldBoundary,
}

impl ComputeResult for DebyeRelaxationResult {}

/// Raw dipole-ACF compute for the Debye route. Computes the mean-subtracted,
/// per-component-summed dipole ACF (unbiased estimator) and carries the
/// zero-lag variance + V/T/Ewald-BC metadata the amplitude needs.
#[derive(Debug, Clone, Copy)]
pub struct DebyeRelaxation {
    /// System volume V. Units: `Å³`.
    pub volume: f64,
    /// Temperature T. Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition.
    pub boundary: EwaldBoundary,
}

/// `(dipole_moments, dt, max_correlation_time)` bundle for [`DebyeRelaxation`].
pub type DebyeRelaxationArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for DebyeRelaxation {
    type Args<'a> = DebyeRelaxationArgs<'a>;
    type Output = DebyeRelaxationResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole, dt, max_correlation_time) = args;
        let shape = dipole.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "dipole_moments (expected (n_frames, 3))",
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
        if self.volume <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "volume",
                value: self.volume.to_string(),
            });
        }
        if self.temperature <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "temperature",
                value: self.temperature.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        // Per-component means → fluctuation ACF so acf[0] = ⟨|δM|²⟩.
        let mut mean = [0.0_f64; 3];
        for t in 0..n_frames {
            for d in 0..3 {
                mean[d] += dipole[[t, d]];
            }
        }
        for m in mean.iter_mut() {
            *m /= n_frames as f64;
        }

        let mut planner = FftPlanner::new();
        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (0..n_frames).map(|t| dipole[[t, d]] - mean[d]).collect();
            let component =
                sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
                    ComputeError::OutOfRange {
                        field: "acf_fft",
                        value: e.to_string(),
                    }
                })?;
            for k in 0..=max_lag {
                acf[k] += component[k];
            }
        }
        // Unbiased linear-ACF estimator C(k) = ⟨δM(0)·δM(k·dt)⟩.
        for k in 0..=max_lag {
            acf[k] /= (n_frames - k) as f64;
        }

        let zero_lag_variance = acf[0];
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(DebyeRelaxationResult {
            lag_times,
            acf,
            zero_lag_variance,
            volume: self.volume,
            temperature: self.temperature,
            boundary: self.boundary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use ndarray::Array2;
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
    fn debye_relaxation_carries_unnormalized_acf_and_metadata() {
        // ac-013.
        let n = 128;
        let dt = 0.5;
        let dipole = rng_series(n, 3, 9);
        let res = DebyeRelaxation {
            volume: 1234.5,
            temperature: 298.0,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dipole, dt, 40))
        .unwrap();
        // zero-lag variance == acf[0] and is non-zero (unnormalized).
        assert!((res.zero_lag_variance - res.acf[0]).abs() < 1e-15);
        assert!(res.zero_lag_variance > 0.0);
        assert_eq!(res.volume, 1234.5);
        assert_eq!(res.temperature, 298.0);
        assert_eq!(res.boundary, EwaldBoundary::TinFoil);
    }
}
