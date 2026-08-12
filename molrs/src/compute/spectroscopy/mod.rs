//! Vibrational and chiral spectroscopy: raw flux/tensor correlation
//! [`Compute`](crate::compute::Compute)s and the spectral
//! [`Fit`](crate::compute::traits::Fit) transforms that turn them into
//! frequency-domain spectra.
//!
//! Each spectrum is an explicit two-step composition — a raw compute produces
//! an (unwindowed) correlation function, a fit applies window + FFT (+
//! physical prefactors):
//!
//! | Spectrum | Raw compute | Fit transform |
//! |----------|-------------|---------------|
//! | VDOS | [`VACF`](crate::compute::transport::VACF) (velocity ACF) | [`PowerSpectrum`] |
//! | IR | [`IRFlux`] (dipole-flux ACF) | [`IRSpectrum`] |
//! | Raman | [`RamanTensor`] (polarizability iso/aniso ACFs) | [`RamanSpectrum`] |
//! | VCD | [`VcdCrossFlux`] (μ̇ × ṁ cross-correlation) | [`VcdSpectrum`] |
//! | ROA | [`RoaCrossTensor`] (α̇ × Ġ′ cross-correlations) | [`RoaSpectrum`] |
//! | Resonance Raman | [`ResonanceRamanTensor`] (resonant iso/aniso ACFs) | [`ResonanceRamanSpectrum`] |
//! | Dielectric ε(ω) | [`DebyeRelaxation`](crate::compute::transport::DebyeRelaxation) / [`GreenKuboConductivity`](crate::compute::transport::GreenKuboConductivity) / [`DipoleRateCross`](crate::compute::transport::DipoleRateCross) | [`EinsteinHelfandSpectrum`] / [`GreenKuboSpectrum`] / [`DipoleAutocorrelationSpectrum`] / [`DipoleRateCrossSpectrum`] |
//!
//! # Units
//!
//! | quantity   | unit |
//! |------------|------|
//! | time / dt  | fs   |
//! | frequency  | cm⁻¹ |
//! | intensity  | arb. |
//!
//! # Shared spectral primitives
//!
//! The window + one-sided-FFT machinery (`window_and_fft`,
//! `acf_to_spectrum`, `acf_to_intensities`), the physical helpers
//! (`cosine_sq_window`, `bose_factor`), and the flux/correlator
//! primitives (`central_diff_series`, `sum_column_acf`, `sum_column_xcorr`,
//! …) live here so every spectral method shares one implementation. Window
//! coefficients always route through [`molrs::signal`] (never reimplemented);
//! the pad + forward-FFT core is the crate-shared `forward_fft_onesided`.

pub mod dielectric_spectrum;
pub mod ir_flux;
pub mod ir_spectrum;
pub mod power_spectrum;
pub mod raman_spectrum;
pub mod raman_tensor;
pub mod resonance_raman_spectrum;
pub mod resonance_raman_tensor;
pub mod roa_cross_tensor;
pub mod roa_spectrum;
pub mod spectra;
pub mod vcd_cross_flux;
pub mod vcd_spectrum;

pub use dielectric_spectrum::{
    ConductivitySumRule, DielectricSpectrumResult, DipoleAutocorrelationSpectrum,
    DipoleRateCrossSpectrum, EinsteinHelfandSpectrum, GreenKuboSpectrum, KramersKronig,
    KramersKronigCheck, RouteAgreement, RouteAgreementCheck, SumRuleCheck,
};
pub use ir_flux::{IRFlux, IRFluxArgs, IRFluxResult};
pub use ir_spectrum::IRSpectrum;
pub use power_spectrum::PowerSpectrum;
pub use raman_spectrum::RamanSpectrum;
pub use raman_tensor::{RamanTensor, RamanTensorArgs, RamanTensorResult};
pub use resonance_raman_spectrum::ResonanceRamanSpectrum;
pub use resonance_raman_tensor::{ResonanceRamanArgs, ResonanceRamanTensor};
pub use roa_cross_tensor::{RoaCrossArgs, RoaCrossResult, RoaCrossTensor};
pub use roa_spectrum::RoaSpectrum;
pub use spectra::{RamanSpectrumResult, SpectrumResult};
pub use vcd_cross_flux::{VcdCrossArgs, VcdCrossFlux, VcdCrossResult};
pub use vcd_spectrum::VcdSpectrum;

use ndarray::{Array1, Array2, ArrayD};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use crate::compute::error::ComputeError;
use crate::compute::fitting::forward_fft_onesided;
use crate::compute::transport::lag_times as transport_lag_times;
use molrs::signal as sig;

// ── Spectral constants ────────────────────────────────────────────────────────

/// Speed of light in m/s (exact).
const C_MS: f64 = 299_792_458.0;
/// Femtoseconds to seconds.
const FS_TO_S: f64 = 1e-15;
/// Metres to centimetres.
const M_TO_CM: f64 = 100.0;

/// Conversion from angular frequency (rad / fs) to wavenumber (cm⁻¹).
///
/// ν̃ = ω · (2π · c · 10⁻¹⁵ · 100)⁻¹ = ω / (2π · c · FS_TO_S · M_TO_CM)
pub(crate) const ANGULAR_FREQ_TO_CM1: f64 =
    1.0 / (2.0 * std::f64::consts::PI * C_MS * FS_TO_S * M_TO_CM);

/// Largest exponent such that `exp(x)` does not overflow f64.
const MAX_EXP_ARG: f64 = 700.0;

// ── Spectral window + FFT helpers ─────────────────────────────────────────────

/// Apply the CosineSq window to a raw ACF, zero-pad, and forward-FFT into a
/// `(frequencies_cm1, intensities)` spectrum.
///
/// The window routes through [`sig::apply_window`] and the FFT through the
/// shared [`forward_fft_onesided`](crate::compute::fitting::forward_fft_onesided)
/// core.
pub(crate) fn window_and_fft(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt_fs: f64,
) -> Result<(Array1<f64>, Array1<f64>), ComputeError> {
    let n = acf.len();
    let acf_dyn = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), acf.to_vec()).map_err(|e| {
        ComputeError::BadShape {
            expected: "1d".into(),
            got: e.to_string(),
        }
    })?;
    let windowed = sig::apply_window(&acf_dyn, sig::WindowType::CosineSq, 0).map_err(|e| {
        ComputeError::OutOfRange {
            field: "apply_window",
            value: e.to_string(),
        }
    })?;
    let windowed_1d: Array1<f64> = windowed.iter().copied().collect();
    let n_pad = (4 * n).next_power_of_two();
    Ok(acf_to_spectrum(planner, &windowed_1d, dt_fs, n_pad))
}

/// Convert a windowed one-sided ACF to a `(frequencies_cm1, intensities_raw)`
/// spectrum. The caller applies any physical prefactors (cross-section + Bose
/// for Raman). Intensities use the spectra-flavoured `1/n_pad` scaling.
pub(crate) fn acf_to_spectrum(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt_fs: f64,
    n_pad: usize,
) -> (Array1<f64>, Array1<f64>) {
    let freqs_rad = sig::frequency_grid(n_pad, dt_fs);
    let intensities = acf_to_intensities(planner, acf, n_pad);
    let n_freq = intensities.len();
    let mut frequencies_cm1 = Array1::zeros(n_freq);
    for j in 0..n_freq {
        frequencies_cm1[j] = freqs_rad[j] * ANGULAR_FREQ_TO_CM1;
    }
    (frequencies_cm1, intensities)
}

/// FFT a windowed ACF and return only the intensity spectrum (no frequency
/// grid). Used for the second Raman component so we don't allocate a second
/// identical frequency array. Delegates the pad+forward-FFT step to the shared
/// [`forward_fft_onesided`](crate::compute::fitting::forward_fft_onesided) core,
/// then applies the spectra-flavoured `1/n_pad` real-part scaling.
pub(crate) fn acf_to_intensities(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    n_pad: usize,
) -> Array1<f64> {
    let acf_vec;
    let acf_slice = match acf.as_slice() {
        Some(s) => s,
        None => {
            acf_vec = acf.to_vec();
            &acf_vec
        }
    };
    let bins = forward_fft_onesided(planner, acf_slice, n_pad);
    let mut intensities = Array1::zeros(bins.len());
    for (j, b) in bins.iter().enumerate() {
        intensities[j] = b.re / n_pad as f64;
    }
    intensities
}

/// Pre-compute a CosineSq window of length `n`.
pub(crate) fn cosine_sq_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let angle = std::f64::consts::PI * i as f64 / (2.0 * (n - 1) as f64);
            angle.cos().powi(2)
        })
        .collect()
}

/// Evaluate the Bose-Einstein factor at frequency `nu` (cm⁻¹) and temperature
/// `T` (K). Returns 1.0 when `nu <= 0` or the exponent would underflow.
pub(crate) fn bose_factor(nu: f64, temperature_k: f64) -> f64 {
    if nu <= 0.0 || temperature_k <= 0.0 {
        return 1.0;
    }
    // HC_KB = h·c / k_B ≈ 1.438777 cm·K
    let exponent = -1.438777 * nu / temperature_k;
    if exponent > -MAX_EXP_ARG {
        1.0 / (1.0 - exponent.exp())
    } else {
        1.0
    }
}

// ── Flux + correlator primitives (IR / Raman / VCD / ROA) ────────────────────

/// Lag grid shared with transport computes (`τ = i·dt`).
#[inline]
pub(crate) fn lag_times(max_lag: usize, dt: f64) -> Array1<f64> {
    transport_lag_times(max_lag, dt)
}

/// Central-difference time derivative of every column of an `(n_frames, n_cols)`
/// series, dropping first and last frame → shape `(n_frames − 2, n_cols)`.
///
/// `ẋ[t] = (x[t+1] − x[t−1]) / (2·dt)` — the IR / Raman / VCD / ROA flux
/// convention. Row-major walk (time outer) for cache locality.
pub(crate) fn central_diff_series(series: &Array2<f64>, dt: f64) -> Array2<f64> {
    let n = series.shape()[0];
    let n_cols = series.shape()[1];
    debug_assert!(n >= 3 && dt > 0.0);
    let inv_2dt = 0.5 / dt;
    let mut out = Array2::<f64>::zeros((n - 2, n_cols));
    for t in 1..n - 1 {
        let o = t - 1;
        for c in 0..n_cols {
            out[[o, c]] = (series[[t + 1, c]] - series[[t - 1, c]]) * inv_2dt;
        }
    }
    out
}

/// Central-difference of one column into a pre-sized buffer of length `n−2`.
#[allow(dead_code)] // crate API for ad-hoc single-column flux work
pub(crate) fn fill_central_diff_col(
    series: &Array2<f64>,
    col: usize,
    dt: f64,
    out: &mut [f64],
) {
    let n = series.shape()[0];
    debug_assert_eq!(out.len(), n.saturating_sub(2));
    let inv_2dt = 0.5 / dt;
    for (o, t) in (1..n - 1).enumerate() {
        out[o] = (series[[t + 1, col]] - series[[t - 1, col]]) * inv_2dt;
    }
}

/// One-column convenience wrapper (allocates). Prefer
/// [`fill_central_diff_col`] / [`central_diff_series`] in hot paths.
#[allow(dead_code)] // crate API; multi-column paths use `central_diff_series`
pub(crate) fn central_diff_col(series: &Array2<f64>, col: usize, dt: f64) -> Array1<f64> {
    let n = series.shape()[0];
    let mut out = vec![0.0; n.saturating_sub(2)];
    fill_central_diff_col(series, col, dt, &mut out);
    Array1::from_vec(out)
}

/// Unnormalized sum of per-column linear ACFs:
/// `C[k] = Σ_d Σ_τ series[τ,d]·series[τ+k,d]` (no `1/(n−k)`).
///
/// Reuses one column buffer + one complex FFT scratch across components.
/// Used by IR (dipole-flux) and Raman (iso / weighted aniso).
pub(crate) fn sum_column_acf(series: &Array2<f64>, max_lag: usize) -> Array1<f64> {
    let n = series.shape()[0];
    let n_cols = series.shape()[1];
    debug_assert!(n >= 2 && max_lag < n);
    let mut planner = FftPlanner::new();
    let mut out = Array1::<f64>::zeros(max_lag + 1);
    let mut col = vec![0.0_f64; n];
    let mut scratch: Vec<Complex64> = Vec::new();
    let out_s = out.as_slice_mut().expect("zeros contiguous");
    for d in 0..n_cols {
        for (t, slot) in col.iter_mut().enumerate() {
            *slot = series[[t, d]];
        }
        sig::acf_fft_accumulate(&mut planner, &col, max_lag, out_s, &mut scratch)
            .expect("sum_column_acf: max_lag < n by construction");
    }
    out
}

/// Unnormalized ACF of a 1-D series (optionally weighted into `out`).
pub(crate) fn acf_accumulate_into(
    planner: &mut FftPlanner<f64>,
    series: &[f64],
    max_lag: usize,
    out: &mut [f64],
    weight: f64,
    scratch: &mut Vec<Complex64>,
) {
    if weight == 1.0 {
        sig::acf_fft_accumulate(planner, series, max_lag, out, scratch)
            .expect("acf_accumulate_into: max_lag < n by construction");
        return;
    }
    // Weighted: accumulate into a temp slice then scale-add (avoids modifying
    // the unweighted kernel). For weight≠1 use a small stack for short lags.
    let mut tmp = vec![0.0_f64; max_lag + 1];
    sig::acf_fft_accumulate(planner, series, max_lag, &mut tmp, scratch)
        .expect("acf_accumulate_into: max_lag < n by construction");
    for k in 0..=max_lag {
        out[k] += weight * tmp[k];
    }
}

/// Unnormalized sum of per-column linear xcorrs:
/// `C[k] = Σ_d Σ_τ a[τ,d]·b[τ+k,d]`.
///
/// Both series must share shape. Used by VCD (`μ̇ × ṁ`).
pub(crate) fn sum_column_xcorr(a: &Array2<f64>, b: &Array2<f64>, max_lag: usize) -> Array1<f64> {
    debug_assert_eq!(a.shape(), b.shape());
    let n = a.shape()[0];
    let n_cols = a.shape()[1];
    debug_assert!(n >= 2 && max_lag < n);
    let mut planner = FftPlanner::new();
    let mut out = Array1::<f64>::zeros(max_lag + 1);
    let mut ca = vec![0.0_f64; n];
    let mut cb = vec![0.0_f64; n];
    let mut sa: Vec<Complex64> = Vec::new();
    let mut sb: Vec<Complex64> = Vec::new();
    let out_s = out.as_slice_mut().expect("zeros contiguous");
    for d in 0..n_cols {
        for t in 0..n {
            ca[t] = a[[t, d]];
            cb[t] = b[[t, d]];
        }
        sig::xcorr_fft_accumulate(&mut planner, &ca, &cb, max_lag, out_s, &mut sa, &mut sb)
            .expect("sum_column_xcorr: max_lag < n by construction");
    }
    out
}

/// Linear cross-correlation `C_ab[t] = Σ_τ a[τ]·b[τ+t]` for `t = 0..=max_lag`.
///
/// Thin wrapper over [`sig::xcorr_fft_with_planner`] for call sites that need a
/// fresh `Array1`. Multi-component hot paths prefer [`sum_column_xcorr`] or
/// [`xcorr_accumulate_into`].
#[allow(dead_code)] // crate API; ROA/VCD use accumulate paths now
pub(crate) fn cross_correlate(
    planner: &mut FftPlanner<f64>,
    a: &Array1<f64>,
    b: &Array1<f64>,
    max_lag: usize,
) -> Array1<f64> {
    sig::xcorr_fft_with_planner(planner, a, b, max_lag)
        .expect("cross_correlate: equal-length inputs with max_lag < n by construction")
}

/// Cross-correlate two equal-length slices and add `weight * C` into `out`.
pub(crate) fn xcorr_accumulate_into(
    planner: &mut FftPlanner<f64>,
    a: &[f64],
    b: &[f64],
    max_lag: usize,
    out: &mut [f64],
    weight: f64,
    sa: &mut Vec<Complex64>,
    sb: &mut Vec<Complex64>,
) {
    if weight == 1.0 {
        sig::xcorr_fft_accumulate(planner, a, b, max_lag, out, sa, sb)
            .expect("xcorr_accumulate_into: max_lag < n by construction");
        return;
    }
    let mut tmp = vec![0.0_f64; max_lag + 1];
    sig::xcorr_fft_accumulate(planner, a, b, max_lag, &mut tmp, sa, sb)
        .expect("xcorr_accumulate_into: max_lag < n by construction");
    for k in 0..=max_lag {
        out[k] += weight * tmp[k];
    }
}
