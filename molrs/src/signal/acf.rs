use ndarray::Array1;
use rustfft::num_traits::Zero;
use rustfft::{FftPlanner, num_complex::Complex64};

/// Linear autocorrelation via the Wiener-Khinchin theorem (FFT-based).
///
/// Returns the un-normalized linear autocorrelation `r[t] = Σ_{τ=0}^{n-1-t} x[τ]·x[τ+t]`
/// for `t = 0, 1, …, max_lag`. The caller is responsible for any further
/// normalization (e.g. division by `(n - t)` for the unbiased estimator,
/// or by `r[0]` for the normalized ACF).
///
/// Reference: Wiener (1930) "Generalized harmonic analysis";
/// Khinchin (1934). Implementation follows the standard linear-ACF recipe
/// (Press et al., *Numerical Recipes* §13.2): zero-pad to ≥ 2·n,
/// forward FFT, take the squared magnitude, inverse FFT.
///
/// Padding to `(2·n).next_power_of_two()` is required: with `n_pad = n`
/// the inverse FFT yields the *circular* autocorrelation, whose tail
/// wraps around and contaminates the lag-t entries with `r_linear[n − t]`.
pub fn acf_fft(data: &Array1<f64>, max_lag: usize) -> Result<Array1<f64>, SignalError> {
    let mut planner = FftPlanner::new();
    acf_fft_with_planner(&mut planner, data, max_lag)
}

/// Same as [`acf_fft`] but reuses the caller-provided `FftPlanner`.
///
/// `rustfft` caches plans inside the planner, so passing one planner across
/// many calls (e.g. component-wise ACFs in dielectric spectra) avoids paying
/// the plan-construction cost on every invocation.
pub fn acf_fft_with_planner(
    planner: &mut FftPlanner<f64>,
    data: &Array1<f64>,
    max_lag: usize,
) -> Result<Array1<f64>, SignalError> {
    let n = data.len();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }
    if max_lag >= n {
        return Err(SignalError::MaxLagTooLarge { max_lag, len: n });
    }

    let n_pad = (2 * n).next_power_of_two();

    let fwd = planner.plan_fft_forward(n_pad);
    let inv = planner.plan_fft_inverse(n_pad);

    let mut complex_data: Vec<Complex64> = data.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    complex_data.resize(n_pad, Complex64::zero());
    fwd.process(&mut complex_data);

    let power: Vec<Complex64> = complex_data
        .iter()
        .map(|c| Complex64::new(c.norm_sqr(), 0.0))
        .collect();
    let mut acf_raw = power;
    inv.process(&mut acf_raw);

    let scale = 1.0 / n_pad as f64;
    let result: Array1<f64> = acf_raw[..=max_lag]
        .iter()
        .map(|c| c.re * scale)
        .collect::<Vec<_>>()
        .into();

    Ok(result)
}

/// Linear cross-correlation via the Wiener–Khinchin cross spectrum (FFT-based).
///
/// Returns the un-normalized linear cross-correlation
/// `r_ab[t] = Σ_{τ=0}^{n-1-t} a[τ]·b[τ+t]` for `t = 0, 1, …, max_lag`, using the
/// same zero-pad to `(2·n).next_power_of_two()`, `IFFT(conj(A)·B)` recipe, and
/// `1/n_pad` scaling as [`acf_fft_with_planner`]. Consequently
/// `xcorr_fft_with_planner(p, a, a, k)` reproduces `acf_fft_with_planner(p, a, k)`
/// **bit-for-bit** (`conj(A)·A` and `|A|²` agree to the last ULP).
///
/// Both inputs must share the length `n` of `a` (the caller precondition — like
/// [`acf_fft`] this does not itself validate `b.len()`). Errors when `a` is empty
/// or `max_lag ≥ n`.
/// Convenience wrapper around [`xcorr_fft_with_planner`] that owns its planner.
pub fn xcorr_fft(
    a: &Array1<f64>,
    b: &Array1<f64>,
    max_lag: usize,
) -> Result<Array1<f64>, SignalError> {
    let mut planner = FftPlanner::new();
    xcorr_fft_with_planner(&mut planner, a, b, max_lag)
}

pub fn xcorr_fft_with_planner(
    planner: &mut FftPlanner<f64>,
    a: &Array1<f64>,
    b: &Array1<f64>,
    max_lag: usize,
) -> Result<Array1<f64>, SignalError> {
    let n = a.len();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }
    if max_lag >= n {
        return Err(SignalError::MaxLagTooLarge { max_lag, len: n });
    }

    let n_pad = (2 * n).next_power_of_two();

    let fwd = planner.plan_fft_forward(n_pad);
    let inv = planner.plan_fft_inverse(n_pad);

    let mut ca: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut cb: Vec<Complex64> = b.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    ca.resize(n_pad, Complex64::zero());
    cb.resize(n_pad, Complex64::zero());
    fwd.process(&mut ca);
    fwd.process(&mut cb);

    let mut prod: Vec<Complex64> = ca
        .iter()
        .zip(cb.iter())
        .map(|(x, y)| x.conj() * y)
        .collect();
    inv.process(&mut prod);

    let scale = 1.0 / n_pad as f64;
    let result: Array1<f64> = prod[..=max_lag]
        .iter()
        .map(|c| c.re * scale)
        .collect::<Vec<_>>()
        .into();

    Ok(result)
}

/// Failure modes for `molrs-signal` primitives.
#[derive(Debug, PartialEq)]
pub enum SignalError {
    /// Input array has length 0.
    EmptyInput,
    /// `max_lag` does not satisfy `max_lag < data.len()`.
    MaxLagTooLarge { max_lag: usize, len: usize },
    /// Window/axis-aware operation was asked to operate along an axis
    /// that does not exist on the input.
    AxisOutOfBounds { axis: usize, ndim: usize },
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalError::EmptyInput => write!(f, "input array is empty"),
            SignalError::MaxLagTooLarge { max_lag, len } => {
                write!(
                    f,
                    "max_lag ({max_lag}) must be less than data length ({len})"
                )
            }
            SignalError::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis ({axis}) out of bounds for ndim ({ndim})")
            }
        }
    }
}

impl std::error::Error for SignalError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_acf_constant_signal_un_normalized() {
        let data = arr1(&[2.0, 2.0, 2.0, 2.0]);
        let result = acf_fft(&data, 3).unwrap();
        assert_eq!(result.len(), 4);
        let expected_lag0 = 4.0 * 2.0_f64.powi(2); // N * c^2 = 4 * 4 = 16
        assert!((result[0] - expected_lag0).abs() < 1e-10);
    }

    #[test]
    fn test_acf_constant_signal_linear_decay() {
        // Linear ACF of constant c over n samples: r[t] = (n - t) · c²
        let data = arr1(&[3.0, 3.0, 3.0]);
        let result = acf_fft(&data, 2).unwrap();
        assert_eq!(result.len(), 3);
        let c_sq = 3.0_f64.powi(2);
        let n = 3.0;
        for (t, v) in result.iter().enumerate() {
            let expected = (n - t as f64) * c_sq;
            assert!(
                (v - expected).abs() < 1e-10,
                "lag {t}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn test_acf_single_element() {
        let data = arr1(&[5.0]);
        let result = acf_fft(&data, 0).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 25.0).abs() < 1e-10); // 1 * 5^2
    }

    #[test]
    fn test_acf_max_lag_zero() {
        let data = arr1(&[1.0, 2.0, 3.0]);
        let result = acf_fft(&data, 0).unwrap();
        assert_eq!(result.len(), 1);
        let expected = 1.0_f64.powi(2) + 2.0_f64.powi(2) + 3.0_f64.powi(2); // sum of squares
        assert!((result[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_acf_max_lag_too_large() {
        let data = arr1(&[1.0, 2.0]);
        let err = acf_fft(&data, 2).unwrap_err();
        assert_eq!(err, SignalError::MaxLagTooLarge { max_lag: 2, len: 2 });
    }

    #[test]
    fn test_acf_empty_input() {
        let data = Array1::<f64>::zeros(0);
        let err = acf_fft(&data, 0).unwrap_err();
        assert_eq!(err, SignalError::EmptyInput);
    }

    #[test]
    fn test_acf_white_noise_peak_at_zero() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..1000).map(|_| rng.random()).collect();
        let arr = Array1::from_vec(data);
        let result = acf_fft(&arr, 10).unwrap();
        assert_eq!(result.len(), 11);
        for k in 1..result.len() {
            assert!(result[k].abs() < result[0]);
        }
    }

    #[test]
    fn test_acf_sine_wave_oscillatory() {
        let n = 128;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin())
            .collect();
        let arr = Array1::from_vec(data);
        let result = acf_fft(&arr, 32).unwrap();
        assert_eq!(result.len(), 33);
        assert!(result[0] > 0.0);
        // ACF at half-period (lag 8) should be negative (anti-phase)
        assert!(result[8] < 0.0);
    }

    #[test]
    fn xcorr_of_self_equals_acf_bit_for_bit() {
        // The contract: xcorr(a, a) reproduces acf_fft(a) exactly (conj(A)·A == |A|²).
        let data = arr1(&[1.0, -2.0, 3.0, 0.5, -1.5, 4.0, 2.0, -3.0]);
        let mut planner = FftPlanner::new();
        let acf = acf_fft_with_planner(&mut planner, &data, 5).unwrap();
        let xc = xcorr_fft_with_planner(&mut planner, &data, &data, 5).unwrap();
        assert_eq!(acf, xc, "xcorr(a, a) must equal acf_fft(a) exactly");
    }

    #[test]
    fn xcorr_matches_direct_cross_sum() {
        // r_ab[t] = Σ_{τ} a[τ]·b[τ+t] — check against the direct definition.
        let a = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[2.0, 0.0, -1.0, 3.0, 1.0]);
        let n = a.len();
        let max_lag = 3;
        let mut planner = FftPlanner::new();
        let xc = xcorr_fft_with_planner(&mut planner, &a, &b, max_lag).unwrap();
        for t in 0..=max_lag {
            let expected: f64 = (0..n - t).map(|tau| a[tau] * b[tau + t]).sum();
            assert!(
                (xc[t] - expected).abs() < 1e-12,
                "lag {t}: {} vs {expected}",
                xc[t]
            );
        }
    }

    #[test]
    fn xcorr_rejects_bad_max_lag_and_empty() {
        let mut planner = FftPlanner::new();
        let a = arr1(&[1.0, 2.0]);
        assert_eq!(
            xcorr_fft_with_planner(&mut planner, &a, &a, 2).unwrap_err(),
            SignalError::MaxLagTooLarge { max_lag: 2, len: 2 }
        );
        let empty = Array1::<f64>::zeros(0);
        assert_eq!(
            xcorr_fft_with_planner(&mut planner, &empty, &empty, 0).unwrap_err(),
            SignalError::EmptyInput
        );
    }
}
