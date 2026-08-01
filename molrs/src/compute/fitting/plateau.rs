//! [`Plateau`] — windowed mean (and spread) of a curve.
//!
//! Reads a plateau value off a curve by averaging it over a fractional window.
//! Typical use: reading the Green–Kubo running-integral plateau, where the
//! transport coefficient is the converged tail value of the running integral.

use ndarray::Array1;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Fit;

/// Result of a plateau (windowed-mean) read.
#[derive(Debug, Clone)]
pub struct PlateauResult {
    /// Mean of the curve over the window. Units: `[y]`.
    pub value: f64,
    /// Number of samples inside the window (`fit_end − fit_start + 1`).
    pub n_samples: usize,
    /// Population standard deviation of the curve over the window. Units: `[y]`.
    pub std: f64,
}

impl ComputeResult for PlateauResult {}

/// Windowed-mean plateau reader over a fractional window of a curve.
///
/// The window `(a, b)` maps to the inclusive index range
/// `[round(a·(n−1)), round(b·(n−1))]`, where `n` is the curve length.
#[derive(Debug, Clone, Copy)]
pub struct Plateau {
    /// `(a, b)` window as fractions of the last index, `0 ≤ a < b ≤ 1`.
    pub window: (f64, f64),
}

impl Fit for Plateau {
    /// The curve to average.
    type Input<'a> = &'a Array1<f64>;
    type Output = PlateauResult;

    /// Average the curve over the fractional window.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the curve is empty.
    /// * [`ComputeError::OutOfRange`] if `a >= b` or the fractions are out of
    ///   `[0, 1]`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let y = input;
        let n = y.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        let (a, b) = self.window;
        if !(0.0..=1.0).contains(&a) || !(0.0..=1.0).contains(&b) || a >= b {
            return Err(ComputeError::OutOfRange {
                field: "window (require 0 <= a < b <= 1)",
                value: format!("{a}/{b}"),
            });
        }

        let last = n - 1;
        let fit_start = (last as f64 * a).round() as usize;
        let mut fit_end = (last as f64 * b).round() as usize;
        if fit_end > last {
            fit_end = last;
        }
        // Single-element curve degenerates to the lone point.
        let fit_end = fit_end.max(fit_start);

        let n_samples = fit_end - fit_start + 1;
        let mut sum = 0.0;
        for i in fit_start..=fit_end {
            sum += y[i];
        }
        let value = sum / n_samples as f64;
        let mut var = 0.0;
        for i in fit_start..=fit_end {
            let d = y[i] - value;
            var += d * d;
        }
        var /= n_samples as f64;
        Ok(PlateauResult {
            value,
            n_samples,
            std: var.sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windowed_mean_of_plateau() {
        // ac-005: curve == P over (a, b). n=11, a=0.5, b=1.0 ->
        // [round(5), round(10)] = [5, 10], 6 samples.
        let p = 4.2;
        let mut y = Array1::zeros(11);
        for i in 5..=10 {
            y[i] = p;
        }
        let res = Plateau { window: (0.5, 1.0) }.fit(&y).unwrap();
        assert!((res.value - p).abs() < 1e-12);
        assert_eq!(res.n_samples, 6);
        assert!(res.std < 1e-12);
    }

    #[test]
    fn sample_count_matches_rounded_bounds() {
        // n=21, a=0.25, b=0.75 -> [round(5), round(15)] = [5, 15] => 11 samples.
        let y = Array1::from_iter((0..21).map(|i| i as f64));
        let res = Plateau {
            window: (0.25, 0.75),
        }
        .fit(&y)
        .unwrap();
        assert_eq!(res.n_samples, 11);
        // Mean of 5..=15 = 10.
        assert!((res.value - 10.0).abs() < 1e-12);
    }

    #[test]
    fn nonzero_std_for_varying_window() {
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 3.0, 5.0]);
        let res = Plateau { window: (0.5, 1.0) }.fit(&y).unwrap();
        // window [round(2), round(4)] = [2, 4] => {1,3,5}, mean 3, std sqrt(8/3).
        assert!((res.value - 3.0).abs() < 1e-12);
        assert!((res.std - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn degenerate_window_errors() {
        let y = Array1::from_iter((0..10).map(|i| i as f64));
        assert!(matches!(
            Plateau { window: (0.8, 0.2) }.fit(&y),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn empty_curve_errors() {
        let y: Array1<f64> = Array1::from_vec(vec![]);
        assert!(matches!(
            Plateau { window: (0.0, 1.0) }.fit(&y),
            Err(ComputeError::EmptyInput)
        ));
    }
}
