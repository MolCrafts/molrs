//! [`LinearFit`] — ordinary-least-squares line fit over a fractional window.
//!
//! Consumes an `(x, y)` curve (e.g. `(lag_times, msd)`) and returns the OLS
//! slope, intercept, coefficient of determination, and the inclusive index
//! bounds of the window actually fitted. The slope arithmetic is the same OLS
//! lifted into `ols_slope_intercept_r2` from
//! the Einstein–Helfand ionic-conductivity OLS, so a `LinearFit` over the same
//! curve and fractions reproduces that function's slope bit-for-bit.

use ndarray::Array1;

use super::ols_slope_intercept_r2;
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Fit;

/// Result of a linear (OLS) fit of `y = slope·x + intercept`.
#[derive(Debug, Clone)]
pub struct LinearFitResult {
    /// Fitted slope d`y`/d`x`. Units: `[y]/[x]`.
    pub slope: f64,
    /// Fitted intercept (value of `y` at `x = 0`). Units: `[y]`.
    pub intercept: f64,
    /// Coefficient of determination r², dimensionless in `[0, 1]`. `1.0` for a
    /// perfect fit.
    pub r2: f64,
    /// Inclusive start index of the fitted window into the input curve.
    pub fit_start: usize,
    /// Inclusive end index of the fitted window into the input curve.
    pub fit_end: usize,
}

impl ComputeResult for LinearFitResult {}

/// Ordinary-least-squares line fit over a fractional window of an `(x, y)`
/// curve.
///
/// The window is given as `(start_frac, end_frac)` fractions of the curve's
/// last index `n − 1`, matching the `fit_start_frac` / `fit_end_frac` semantics
/// of the Einstein–Helfand ionic conductivity:
///
/// ```text
///     fit_start = round((n − 1)·start_frac)
///     fit_end   = round((n − 1)·end_frac)
/// ```
///
/// with the same end clamp (`fit_end ≤ n − 1`) and a guard guaranteeing at
/// least two fit points.
#[derive(Debug, Clone, Copy)]
pub struct LinearFit {
    /// `(start_frac, end_frac)` window as fractions of the last index, with
    /// `0 ≤ start_frac < end_frac ≤ 1`.
    pub window: (f64, f64),
}

impl Fit for LinearFit {
    /// `(x, y)` — the abscissa and ordinate curves, equal length.
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>);
    type Output = LinearFitResult;

    /// Fit `y = slope·x + intercept` over `[fit_start, fit_end]`.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if either curve has fewer than two points.
    /// * [`ComputeError::DimensionMismatch`] if `x.len() != y.len()`.
    /// * [`ComputeError::OutOfRange`] if `start_frac >= end_frac`, the fractions
    ///   are out of `[0, 1]`, or the window is degenerate (all `x` equal).
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (x, y) = input;
        let n = x.len();
        if n != y.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: y.len(),
                what: "LinearFit (x, y) lengths",
            });
        }
        if n < 2 {
            return Err(ComputeError::EmptyInput);
        }

        let (start_frac, end_frac) = self.window;
        if !(0.0..=1.0).contains(&start_frac)
            || !(0.0..=1.0).contains(&end_frac)
            || start_frac >= end_frac
        {
            return Err(ComputeError::OutOfRange {
                field: "window (require 0 <= start < end <= 1)",
                value: format!("{start_frac}/{end_frac}"),
            });
        }

        let last = n - 1;
        // Same index derivation as the Einstein–Helfand conductivity (max_lag = last).
        let mut fit_start = (last as f64 * start_frac).round() as usize;
        let mut fit_end = (last as f64 * end_frac).round() as usize;
        if fit_end >= last {
            fit_end = last;
        }
        if fit_end < fit_start + 1 {
            fit_start = fit_start.min(last.saturating_sub(1));
            fit_end = (fit_start + 1).min(last);
        }

        let xs = x.as_slice().ok_or(ComputeError::BadShape {
            expected: "contiguous x".into(),
            got: "non-contiguous".into(),
        })?;
        let ys = y.as_slice().ok_or(ComputeError::BadShape {
            expected: "contiguous y".into(),
            got: "non-contiguous".into(),
        })?;

        let (slope, intercept, r2) =
            ols_slope_intercept_r2(xs, ys, fit_start, fit_end).ok_or(ComputeError::OutOfRange {
                field: "fit window (degenerate: all x equal)",
                value: format!("[{fit_start}, {fit_end}]"),
            })?;

        Ok(LinearFitResult {
            slope,
            intercept,
            r2,
            fit_start,
            fit_end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| i as f64))
    }

    #[test]
    fn recovers_known_slope_intercept_r2() {
        // ac-002: y = 3x + 2 on a uniform grid, window (0,1).
        let x = linspace(10);
        let y: Array1<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();
        let res = LinearFit { window: (0.0, 1.0) }.fit((&x, &y)).unwrap();
        assert!((res.slope - 3.0).abs() < 1e-12, "slope {}", res.slope);
        assert!((res.intercept - 2.0).abs() < 1e-12);
        assert!((res.r2 - 1.0).abs() < 1e-12);
        assert_eq!(res.fit_start, 0);
        assert_eq!(res.fit_end, 9);
    }

    #[test]
    fn full_window_uses_whole_curve() {
        let x = linspace(21);
        let y: Array1<f64> = x.iter().map(|&xi| -0.5 * xi + 7.0).collect();
        let res = LinearFit { window: (0.0, 1.0) }.fit((&x, &y)).unwrap();
        assert_eq!(res.fit_start, 0);
        assert_eq!(res.fit_end, 20);
        assert!((res.slope + 0.5).abs() < 1e-12);
    }

    #[test]
    fn degenerate_window_start_ge_end_errors() {
        // ac-016: start >= end.
        let x = linspace(10);
        let y = x.clone();
        let err = LinearFit { window: (0.7, 0.3) }.fit((&x, &y)).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn degenerate_all_x_equal_errors() {
        // ac-016: all x in the window equal -> denom ~ 0.
        let x = Array1::from_vec(vec![5.0; 6]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let err = LinearFit { window: (0.0, 1.0) }.fit((&x, &y)).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn mismatched_lengths_error() {
        let x = linspace(5);
        let y = linspace(4);
        let err = LinearFit { window: (0.0, 1.0) }.fit((&x, &y)).unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn too_short_errors() {
        let x = linspace(1);
        let y = linspace(1);
        let err = LinearFit { window: (0.0, 1.0) }.fit((&x, &y)).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
