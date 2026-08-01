//! [`CumulativeTrapezoid`] — cumulative trapezoidal integral of a curve.
//!
//! Consumes a curve `y` sampled on uniform step `dt` and returns the running
//! integral `∫₀^{k·dt} y(t) dt` at every point. The trapezoid recurrence is the
//! same one lifted into `running_trapezoid` from
//! the Green–Kubo ionic conductivity, so a `CumulativeTrapezoid` over the same JACF
//! and `dt` reproduces that function's running integral bit-for-bit (before the
//! Green–Kubo prefactor).

use ndarray::Array1;

use super::running_trapezoid;
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Fit;

/// Result of a running trapezoidal integration.
#[derive(Debug, Clone)]
pub struct CumulativeTrapezoidResult {
    /// Cumulative integral: `integral[k] = ∫₀^{k·dt} y(t) dt`, length
    /// `y.len()`. `integral[0] = 0`. Units: `[y]·[dt]`.
    pub integral: Array1<f64>,
}

impl ComputeResult for CumulativeTrapezoidResult {}

/// Cumulative trapezoidal integral of a uniformly-sampled curve.
///
/// Stateless: the step `dt` and the optional lag count travel with the input,
/// not the struct, since they are properties of the upstream curve / request.
#[derive(Debug, Clone, Copy, Default)]
pub struct CumulativeTrapezoid;

impl Fit for CumulativeTrapezoid {
    /// `(y, dt, n_lags)` — the curve, its uniform sample step (> 0), and an
    /// optional number of leading samples to integrate. `None` integrates the
    /// whole curve; `Some(m)` integrates the first `m` samples and **errors if
    /// `m` exceeds the curve length** (invariant (a): never silently truncate
    /// a too-short raw curve to satisfy a longer request).
    type Input<'a> = (&'a Array1<f64>, f64, Option<usize>);
    type Output = CumulativeTrapezoidResult;

    /// Integrate `y` cumulatively with the trapezoid rule.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if `y` is empty or `n_lags == Some(0)`.
    /// * [`ComputeError::OutOfRange`] if `dt <= 0`, or if `n_lags == Some(m)`
    ///   with `m` greater than the curve length (invariant a).
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (y, dt, n_lags) = input;
        if y.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let take = match n_lags {
            None => y.len(),
            Some(0) => return Err(ComputeError::EmptyInput),
            Some(m) if m > y.len() => {
                return Err(ComputeError::OutOfRange {
                    field: "n_lags (exceeds curve length)",
                    value: format!("{m} > {}", y.len()),
                });
            }
            Some(m) => m,
        };
        let ys = y.as_slice().ok_or(ComputeError::BadShape {
            expected: "contiguous y".into(),
            got: "non-contiguous".into(),
        })?;
        let integral = Array1::from_vec(running_trapezoid(&ys[..take], dt));
        Ok(CumulativeTrapezoidResult { integral })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_curve_linear_integral() {
        // ac-004: constant c on step dt -> integral[k] = c·k·dt.
        let c = 3.0;
        let dt = 0.25;
        let y = Array1::from_vec(vec![c; 8]);
        let res = CumulativeTrapezoid.fit((&y, dt, None)).unwrap();
        for k in 0..8 {
            let expected = c * k as f64 * dt;
            assert!(
                (res.integral[k] - expected).abs() < 1e-12,
                "k={k}: {} != {expected}",
                res.integral[k]
            );
        }
    }

    #[test]
    fn triangle_signal_analytic_integral() {
        // y = t on [0, 4] step 1 -> ∫₀^T t dt = T²/2 by trapezoid (exact for
        // linear integrand).
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let res = CumulativeTrapezoid.fit((&y, 1.0, None)).unwrap();
        for k in 0..5 {
            let expected = (k as f64) * (k as f64) / 2.0;
            assert!((res.integral[k] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn matches_jacf_running_trapezoid() {
        // ac-004: equals the running trapezoid integral computed inline in
        // the Green–Kubo conductivity running integral on the same JACF + dt.
        let jacf = Array1::from_vec(vec![1.0, 0.8, 0.5, 0.2, 0.05]);
        let dt = 0.5;
        let res = CumulativeTrapezoid.fit((&jacf, dt, None)).unwrap();
        // Inline jacf.rs recurrence.
        let mut expected = vec![0.0; jacf.len()];
        let mut integral = 0.0;
        for tau in 1..jacf.len() {
            integral += 0.5 * (jacf[tau - 1] + jacf[tau]) * dt;
            expected[tau] = integral;
        }
        for (k, (&got, &want)) in res.integral.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "k={k}");
        }
    }

    #[test]
    fn n_lags_subset_integrates_only_prefix() {
        let y = Array1::from_vec(vec![1.0; 10]);
        let res = CumulativeTrapezoid.fit((&y, 1.0, Some(4))).unwrap();
        assert_eq!(res.integral.len(), 4);
    }

    #[test]
    fn n_lags_exceeding_length_errors() {
        // ac-014: requested lag range exceeds the raw curve -> OutOfRange.
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            CumulativeTrapezoid.fit((&y, 1.0, Some(10))),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn empty_curve_errors() {
        let y: Array1<f64> = Array1::from_vec(vec![]);
        assert!(matches!(
            CumulativeTrapezoid.fit((&y, 1.0, None)),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn zero_lags_errors() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            CumulativeTrapezoid.fit((&y, 1.0, Some(0))),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn nonpositive_dt_errors() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            CumulativeTrapezoid.fit((&y, 0.0, None)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
