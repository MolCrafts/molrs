//! Spectral-transform [`Fit`] impls: [`PowerSpectrum`], [`IRSpectrum`],
//! [`RamanSpectrum`], plus the spectral window + FFT primitives they own.
//!
//! Each [`Fit`] consumes a **raw autocorrelation function** (and `dt`, plus any
//! physical params) and applies window + FFT (+ prefactors) to produce a
//! frequency-domain spectrum. The window + one-sided-FFT machinery
//! ([`window_and_fft`], [`acf_to_spectrum`], [`acf_to_intensities`]) and the
//! physical helpers ([`cosine_sq_window`], [`bose_factor`]) were relocated here
//! from `compute::spectra` in compute-fit-03-cleanup: windowing + transforming a
//! raw ACF is a *fit*, so it belongs in the [`Fit`] layer. Window coefficients
//! always route through [`molrs::signal`] (never reimplemented).
//!
//! The raw, unwindowed ACFs these fits consume come from the raw computes
//! [`VACF`](super::VACF) (velocity), [`IRFlux`](super::IRFlux) (dipole flux),
//! and [`RamanTensor`](super::RamanTensor) (polarizability iso/aniso).
//!
//! # Units
//!
//! | quantity   | unit |
//! |------------|------|
//! | time / dt  | fs   |
//! | frequency  | cm⁻¹ |
//! | intensity  | arb. |

use ndarray::{Array1, ArrayD};
use rustfft::FftPlanner;

use super::forward_fft_onesided;
use crate::compute::error::ComputeError;
use crate::compute::spectra::{RamanSpectrumResult, SpectrumResult};
use crate::compute::traits::Fit;
use molrs::signal as sig;

/// Parallel polarization: `I_∥ = I_iso + (4/45)·I_aniso`.
const PARALLEL_ANISO_COEFF: f64 = 4.0 / 45.0;
/// Perpendicular polarization denominator: `I_⊥ = I_aniso / 15`.
const PERPENDICULAR_DENOM: f64 = 15.0;

// ── Spectral constants (relocated from compute::spectra) ─────────────────────

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

// ── Spectral window + FFT helpers (relocated from compute::spectra) ──────────

/// Apply the CosineSq window to a raw ACF, zero-pad, and forward-FFT into a
/// `(frequencies_cm1, intensities)` spectrum.
///
/// The window routes through [`sig::apply_window`] and the FFT through the
/// shared [`forward_fft_onesided`](super::forward_fft_onesided) core.
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
/// [`forward_fft_onesided`](super::forward_fft_onesided) core, then applies the
/// spectra-flavoured `1/n_pad` real-part scaling.
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

/// Velocity power spectrum (VDOS) transform of a **raw velocity ACF**.
///
/// Applies the CosineSq window + zero-padded forward FFT (the [`window_and_fft`]
/// pipeline) to a raw, unnormalized velocity ACF — the
/// [`VacfResult`](super::VacfResult) of the [`VACF`](super::VACF) compute.
#[derive(Debug, Clone, Copy, Default)]
pub struct PowerSpectrum;

impl Fit for PowerSpectrum {
    /// `(acf, dt_fs)` — the raw velocity ACF (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw ACF into a VDOS spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf, dt_fs) = input;
        let n = acf.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }
        let mut planner = FftPlanner::new();
        let (frequencies_cm1, intensities) = window_and_fft(&mut planner, acf, dt_fs)?;
        Ok(SpectrumResult {
            frequencies_cm1,
            intensities,
            resolution: n - 1,
            n_frames: n,
        })
    }
}

/// Infrared absorption spectrum transform of a **raw dipole-flux ACF**.
///
/// Identical window+FFT pipeline as [`PowerSpectrum`]; the difference between
/// IR and the power spectrum is entirely in *which* ACF is supplied (dipole
/// flux vs velocity), computed upstream by the [`IRFlux`](super::IRFlux) raw
/// compute.
#[derive(Debug, Clone, Copy, Default)]
pub struct IRSpectrum;

impl Fit for IRSpectrum {
    /// `(acf, dt_fs)` — the raw dipole-flux ACF (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw flux ACF into an IR spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf, dt_fs) = input;
        let n = acf.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }
        let mut planner = FftPlanner::new();
        let (frequencies_cm1, intensities) = window_and_fft(&mut planner, acf, dt_fs)?;
        Ok(SpectrumResult {
            frequencies_cm1,
            intensities,
            resolution: n - 1,
            n_frames: n,
        })
    }
}

/// Raman spectrum transform of **raw isotropic + anisotropic ACFs**.
///
/// Applies the one CosineSq window ([`cosine_sq_window`]) to each ACF, FFTs both
/// (reusing the isotropic frequency grid), and applies the cross-section
/// (`(ν₀ − ν)⁴ / ν`) and Bose (`1/(1 − exp(−hcν/kT))`) prefactors. The raw
/// iso/aniso ACFs come from the [`RamanTensor`](super::RamanTensor) raw compute.
#[derive(Debug, Clone, Copy)]
pub struct RamanSpectrum {
    /// Laser frequency, cm⁻¹. The cross-section correction scales as
    /// `(ν₀ − ν)⁴ / ν`. Set to `0.0` to skip.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. Set to `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular polarization spectra.
    pub averaged: bool,
}

impl Fit for RamanSpectrum {
    /// `(acf_iso, acf_aniso, dt_fs)` — the raw isotropic and (weighted)
    /// anisotropic ACFs (equal-length 1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, f64);
    type Output = RamanSpectrumResult;

    /// Window + FFT + physical prefactors on the iso/aniso ACFs.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if either ACF is empty.
    /// * [`ComputeError::DimensionMismatch`] if the two ACFs differ in length.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf_iso, acf_aniso, dt_fs) = input;
        let n = acf_iso.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if acf_aniso.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: acf_aniso.len(),
                what: "Raman (acf_iso, acf_aniso) lengths",
            });
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }

        let max_lag = n - 1;
        // Pre-compute CosineSq window once and apply to both ACFs — identical to
        // the historical Raman transform tail.
        let window = cosine_sq_window(max_lag + 1);
        let win_iso: Array1<f64> = acf_iso.iter().zip(&window).map(|(a, w)| a * w).collect();
        let win_aniso: Array1<f64> = acf_aniso.iter().zip(&window).map(|(a, w)| a * w).collect();

        let mut planner = FftPlanner::new();
        let n_pad = (4 * (max_lag + 1)).next_power_of_two();
        let (frequencies_cm1, raw_iso) = acf_to_spectrum(&mut planner, &win_iso, dt_fs, n_pad);
        let raw_aniso = acf_to_intensities(&mut planner, &win_aniso, n_pad);

        let mut iso_int = raw_iso;
        let mut aniso_int = raw_aniso;

        let apply_cross_section = self.incident_frequency_cm1 > 0.0;
        let apply_bose = self.temperature_k > 0.0;

        for j in 0..frequencies_cm1.len() {
            let nu = frequencies_cm1[j];
            let mut correction = 1.0;
            if apply_cross_section && nu > 0.0 {
                let dnu = self.incident_frequency_cm1 - nu;
                correction *= dnu.powi(4) / nu;
            }
            if apply_bose {
                correction *= bose_factor(nu, self.temperature_k);
            }
            iso_int[j] *= correction;
            aniso_int[j] *= correction;
        }

        let (parallel, perpendicular) = if self.averaged {
            let m = frequencies_cm1.len();
            let mut par = Array1::zeros(m);
            let mut perp = Array1::zeros(m);
            for j in 0..m {
                par[j] = iso_int[j] + PARALLEL_ANISO_COEFF * aniso_int[j];
                perp[j] = aniso_int[j] / PERPENDICULAR_DENOM;
            }
            (Some(par), Some(perp))
        } else {
            (None, None)
        };

        Ok(RamanSpectrumResult {
            frequencies_cm1,
            isotropic: iso_int,
            anisotropic: aniso_int,
            parallel,
            perpendicular,
            resolution: max_lag,
            n_frames: n,
        })
    }
}

/// VCD (vibrational circular dichroism) spectrum transform of a **raw VCD
/// cross-correlation** `⟨μ̇(0)·ṁ(τ)⟩` from [`VcdCrossFlux`](super::VcdCrossFlux).
///
/// Identical window + one-sided FFT pipeline as [`IRSpectrum`] (calls
/// [`window_and_fft`]), so the cm⁻¹ grid matches IR/Raman exactly — only the
/// supplied cross-correlation differs. The resulting intensities are **signed**:
/// enantiomers produce sign-flipped spectra and an achiral system gives ≈ 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct VcdSpectrum;

impl Fit for VcdSpectrum {
    /// `(acf, dt_fs)` — the raw VCD cross-correlation (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw VCD cross-correlation into a (signed) VCD spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf, dt_fs) = input;
        let n = acf.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }
        let mut planner = FftPlanner::new();
        let (frequencies_cm1, intensities) = window_and_fft(&mut planner, acf, dt_fs)?;
        Ok(SpectrumResult {
            frequencies_cm1,
            intensities,
            resolution: n - 1,
            n_frames: n,
        })
    }
}

/// ROA (Raman optical activity) spectrum transform of **raw ROA iso/aniso
/// cross-correlations** from [`RoaCrossTensor`](super::RoaCrossTensor).
///
/// Reuses the exact [`RamanSpectrum`] window + FFT + cross-section/Bose pipeline
/// (so the cm⁻¹ grid and normal-mode peaks coincide with Raman); the ROA
/// difference spectrum is signed and flips between enantiomers because the input
/// `α̇ × Ġ′` cross-correlation flips sign.
#[derive(Debug, Clone, Copy)]
pub struct RoaSpectrum {
    /// Laser frequency, cm⁻¹ (cross-section `(ν₀ − ν)⁴ / ν`). `0.0` to skip.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular components.
    pub averaged: bool,
}

impl Fit for RoaSpectrum {
    /// `(acf_iso, acf_aniso, dt_fs)` — the raw ROA iso/aniso cross-correlations
    /// and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, f64);
    type Output = RamanSpectrumResult;

    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        // ROA shares the Raman spectral pipeline (window + FFT + prefactors);
        // only the upstream cross-correlation differs. Reuse, do not duplicate.
        RamanSpectrum {
            incident_frequency_cm1: self.incident_frequency_cm1,
            temperature_k: self.temperature_k,
            averaged: self.averaged,
        }
        .fit(input)
    }
}

/// Resonance-Raman spectrum transform of **raw resonant iso/aniso ACFs** from
/// [`ResonanceRamanTensor`](super::ResonanceRamanTensor).
///
/// Identical to [`RamanSpectrum`] (same window + FFT + cross-section/Bose
/// prefactors and cm⁻¹ grid); the only difference is upstream — the ACFs come
/// from a resonant (frequency-dependent) polarizability series.
#[derive(Debug, Clone, Copy)]
pub struct ResonanceRamanSpectrum {
    /// Laser / excitation frequency, cm⁻¹. `0.0` to skip the cross-section.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular components.
    pub averaged: bool,
}

impl Fit for ResonanceRamanSpectrum {
    /// `(acf_iso, acf_aniso, dt_fs)` — the raw resonant iso/aniso ACFs and dt.
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, f64);
    type Output = RamanSpectrumResult;

    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        RamanSpectrum {
            incident_frequency_cm1: self.incident_frequency_cm1,
            temperature_k: self.temperature_k,
            averaged: self.averaged,
        }
        .fit(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fit::{IRFlux, RamanTensor, VACF};
    use crate::compute::traits::Compute;
    use molrs::Frame;
    use molrs::signal as sig;
    use ndarray::{Array1, Array2};

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    /// Rebuild the raw velocity ACF the VACF / PowerSpectrum path consumes
    /// (per-dof mean-subtract, FFT-ACF, sum, *= 1/n_dof).
    fn power_acf(velocities: &Array2<f64>, max_lag: usize) -> Array1<f64> {
        let n_frames = velocities.shape()[0];
        let n_dof = velocities.shape()[1];
        let inv_n_frames = 1.0 / n_frames as f64;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..n_dof {
            let mut col: Array1<f64> = (0..n_frames).map(|t| velocities[[t, d]]).collect();
            let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
            for v in col.iter_mut() {
                *v -= mean;
            }
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        let inv_n_dof = 1.0 / n_dof as f64;
        for k in 0..=max_lag {
            acf_sum[k] *= inv_n_dof;
        }
        acf_sum
    }

    /// Rebuild the raw dipole-flux ACF the IRFlux compute / IRSpectrum consume.
    fn ir_acf(dm: &Array2<f64>, dt_fs: f64, max_lag: usize) -> Array1<f64> {
        let n_frames = dm.shape()[0];
        let inv_2dt = 0.5 / dt_fs;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let flux: Array1<f64> = (1..n_frames - 1)
                .map(|t| (dm[[t + 1, d]] - dm[[t - 1, d]]) * inv_2dt)
                .collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &flux, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        acf_sum
    }

    /// Rebuild the raw iso/aniso ACFs the RamanTensor compute / RamanSpectrum consume.
    fn raman_acfs(pol: &Array2<f64>, dt_fs: f64, max_lag: usize) -> (Array1<f64>, Array1<f64>) {
        const DIAG_W: f64 = 0.5;
        const OFFDIAG_W: f64 = 3.0;
        let n_frames = pol.shape()[0];
        let inv_2dt = 0.5 / dt_fs;
        let flux_len = n_frames - 2;
        let mut iso = Vec::with_capacity(flux_len);
        let mut comps: [Vec<f64>; 6] = Default::default();
        for t in 1..n_frames - 1 {
            let p = pol.row(t - 1);
            let q = pol.row(t + 1);
            let xx = (q[0] - p[0]) * inv_2dt;
            let yy = (q[1] - p[1]) * inv_2dt;
            let zz = (q[2] - p[2]) * inv_2dt;
            iso.push((xx + yy + zz) / 3.0);
            comps[0].push(xx - yy);
            comps[1].push(yy - zz);
            comps[2].push(zz - xx);
            comps[3].push((q[3] - p[3]) * inv_2dt);
            comps[4].push((q[4] - p[4]) * inv_2dt);
            comps[5].push((q[5] - p[5]) * inv_2dt);
        }
        let mut planner = FftPlanner::new();
        let iso_series = Array1::from_vec(iso);
        let acf_iso = sig::acf_fft_with_planner(&mut planner, &iso_series, max_lag).unwrap();
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        for (c, comp) in comps.iter_mut().enumerate() {
            let w = if c < 3 { DIAG_W } else { OFFDIAG_W };
            let col = Array1::from_vec(std::mem::take(comp));
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_aniso[k] += w * acf[k];
            }
        }
        (acf_iso, acf_aniso)
    }

    fn sine_velocities(n: usize, dt_fs: f64, freq_thz: f64) -> Array2<f64> {
        let mut v = Array2::zeros((n, 3));
        for t in 0..n {
            let tf = t as f64 * dt_fs;
            v[[t, 0]] = (2.0 * std::f64::consts::PI * freq_thz * 1e-3 * tf).sin();
        }
        v
    }

    #[test]
    fn vacf_plus_vdos_matches_manual_acf_path() {
        // ac-003/ac-006: the VACF raw compute returns exactly the unwindowed ACF
        // the PowerSpectrum transform consumes, and VACF + PowerSpectrum equals
        // the manual-ACF + PowerSpectrum path.
        let n = 1024;
        let dt = 0.5;
        let res = 200;
        let v = sine_velocities(n, dt, 10.0);
        let max_lag = res.min(n - 1);
        let acf = power_acf(&v, max_lag);

        let raw = VACF.compute(&no_frames(), (&v, dt, res)).unwrap();
        assert_eq!(raw.acf, acf); // VACF returns the raw unwindowed ACF.

        let from_raw = PowerSpectrum.fit((&raw.acf, dt)).unwrap();
        let from_manual = PowerSpectrum.fit((&acf, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_eq!(from_raw.intensities, from_manual.intensities);
    }

    #[test]
    fn vdos_sine_peak_at_333_cm1() {
        // ac-008 (scientific): v_x(t) = sin(2π·10 THz·t), dt = 0.5 fs.
        // 10 THz = 0.01 fs⁻¹ → 333.56 cm⁻¹.
        let n = 4096;
        let dt = 0.5;
        let v = sine_velocities(n, dt, 10.0);
        let raw = VACF.compute(&no_frames(), (&v, dt, 200)).unwrap();
        let spec = PowerSpectrum.fit((&raw.acf, dt)).unwrap();
        let n_bins = spec.intensities.len();
        let search_end = n_bins.saturating_sub(3);
        let max_idx = spec
            .intensities
            .iter()
            .enumerate()
            .skip(1)
            .take(search_end.saturating_sub(1))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let peak_cm1 = spec.frequencies_cm1[max_idx];
        assert!(
            (peak_cm1 - 333.56).abs() < 20.0,
            "peak at {peak_cm1} cm⁻¹, expected ~333.56 cm⁻¹"
        );
    }

    #[test]
    fn irflux_plus_ir_transform_matches_manual_acf_path() {
        // ac-003/ac-007 (IR): IRFlux returns the unwindowed dipole-flux ACF the
        // IRSpectrum transform consumes; IRFlux + IRSpectrum == manual path.
        let n = 1024;
        let dt = 0.5;
        let res = 200;
        let mut dm = Array2::zeros((n, 3));
        for t in 0..n {
            let tf = t as f64 * dt;
            dm[[t, 2]] = (2.0 * std::f64::consts::PI * 10.0 * 1e-3 * tf).sin();
        }
        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let acf = ir_acf(&dm, dt, max_lag);

        let raw = IRFlux.compute(&no_frames(), (&dm, dt, res)).unwrap();
        assert_eq!(raw.acf, acf); // IRFlux returns the raw unwindowed ACF.

        let from_raw = IRSpectrum.fit((&raw.acf, dt)).unwrap();
        let from_manual = IRSpectrum.fit((&acf, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_eq!(from_raw.intensities, from_manual.intensities);
    }

    #[test]
    fn ramantensor_plus_raman_transform_matches_manual_acf_path() {
        // ac-003/ac-007 (Raman): RamanTensor returns the unwindowed iso/aniso
        // ACFs; RamanTensor + RamanSpectrum == manual path (iso/aniso/par/perp).
        let n = 256;
        let dt = 0.5;
        let res = 60;
        let mut pol = Array2::zeros((n, 6));
        for t in 0..n {
            let tf = t as f64 * dt;
            let val = (2.0 * std::f64::consts::PI * 30.0 * 1e-3 * tf).sin();
            for c in 0..6 {
                pol[[t, c]] = val * (1.0 + 0.1 * c as f64);
            }
        }
        let incident = 10000.0;
        let temp = 300.0;
        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let (acf_iso, acf_aniso) = raman_acfs(&pol, dt, max_lag);

        let raw = RamanTensor.compute(&no_frames(), (&pol, dt, res)).unwrap();
        assert_eq!(raw.acf_iso, acf_iso); // raw, unwindowed.
        assert_eq!(raw.acf_aniso, acf_aniso);

        let fit = RamanSpectrum {
            incident_frequency_cm1: incident,
            temperature_k: temp,
            averaged: true,
        };
        let from_raw = fit.fit((&raw.acf_iso, &raw.acf_aniso, dt)).unwrap();
        let from_manual = fit.fit((&acf_iso, &acf_aniso, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_eq!(from_raw.isotropic, from_manual.isotropic);
        assert_eq!(from_raw.anisotropic, from_manual.anisotropic);
        assert_eq!(from_raw.parallel, from_manual.parallel);
        assert_eq!(from_raw.perpendicular, from_manual.perpendicular);
    }

    #[test]
    fn raman_rejects_mismatched_acf_lengths() {
        // ac-016: shape mismatch -> DimensionMismatch.
        let iso = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let aniso = Array1::from_vec(vec![1.0, 2.0]);
        let err = RamanSpectrum {
            incident_frequency_cm1: 0.0,
            temperature_k: 0.0,
            averaged: false,
        }
        .fit((&iso, &aniso, 0.5))
        .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn power_rejects_empty_and_bad_dt() {
        let empty: Array1<f64> = Array1::from_vec(vec![]);
        assert!(matches!(
            PowerSpectrum.fit((&empty, 0.5)),
            Err(ComputeError::EmptyInput)
        ));
        let acf = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        assert!(matches!(
            PowerSpectrum.fit((&acf, 0.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
