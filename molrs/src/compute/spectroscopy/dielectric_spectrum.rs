//! Dielectric ε(ω) transform [`Fit`] impls: [`EinsteinHelfandSpectrum`] and
//! [`GreenKuboSpectrum`].
//!
//! Each [`Fit`] consumes a **raw autocorrelation function** (the fluctuation
//! dipole ACF for the Einstein–Helfand route; the current ACF for the
//! Green–Kubo route) plus the physical metadata (`dt`, `V`, `T`, `ε_∞`, and —
//! for EH — the zero-lag variance ⟨|δM|²⟩) and applies window + FFT +
//! prefactors to produce the frequency-dependent permittivity
//! [`DielectricSpectrumResult`].
//!
//! The window + one-sided-FFT machinery (`acf_to_spectrum`,
//! `taper_derivative_spectrum`, `windowed_acf_spectrum`) was relocated here
//! from `compute::dielectric` in compute-fit-04-dielectric: windowing +
//! transforming a raw ACF into ε(ω) is a *fit*, so it belongs in the [`Fit`]
//! layer. Window coefficients always route through [`molrs::signal`] (never
//! reimplemented).
//!
//! The raw, unwindowed ACFs these fits consume come from the raw computes
//! [`DebyeRelaxation`](crate::compute::transport::DebyeRelaxation) (fluctuation dipole ACF +
//! ⟨M(0)²⟩ + V/T/Ewald-BC) and
//! [`GreenKuboConductivity`](crate::compute::transport::GreenKuboConductivity) (current ACF).
//!
//! # Units
//!
//! All inputs and outputs use LAMMPS *real* units throughout:
//!
//! | quantity        | unit                |
//! |-----------------|---------------------|
//! | length          | Å                   |
//! | charge          | e                   |
//! | time / dt       | ps                  |
//! | temperature     | K                   |
//! | volume          | Å³                  |
//! | dipole moment   | e · Å               |
//! | current density | e · Å⁻² · ps⁻¹      |
//! | angular ω       | rad · ps⁻¹          |
//! | ε permittivity  | dimensionless       |

use ndarray::Array1;
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::fit::forward_fft_onesided;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Fit;
use molrs::signal as sig;

// ── Physical constants (MD real units: kcal, mol, Angstrom, e, K) ─────────────

use molrs::units::constants::BOLTZMANN_REAL as K_B;
use molrs::units::constants::COULOMB_REAL as KAPPA;

/// 4π/3 — the isotropic dielectric fluctuation prefactor numerator.
const FOUR_PI_OVER_3: f64 = 4.1887902047863905;

/// Result of a dielectric ε(ω) spectrum transform.
///
/// The complex permittivity is `ε*(ω) = ε′(ω) − i·ε″(ω)`. `eps_imag`
/// stores `ε″(ω)` with the **positive-loss** convention (≥ 0 for stable
/// causal systems). FT convention throughout: `X(ω) = ∫₀^∞ f(t) e^{−iωt} dt`.
#[derive(Debug, Clone)]
pub struct DielectricSpectrumResult {
    /// Angular frequency grid, rad·ps⁻¹, length `n_pad/2 + 1` with
    /// `n_pad = (2·n_correlation_steps).next_power_of_two()`.
    /// Bin 0 is the DC bin; bin 1 is Δω = 2π/(n_pad·dt); the last bin
    /// is the Nyquist frequency π/dt.
    pub frequencies: Array1<f64>,
    /// Real part ε′(ω), dimensionless.
    pub eps_real: Array1<f64>,
    /// Loss spectrum ε″(ω), dimensionless (positive sign convention).
    pub eps_imag: Array1<f64>,
}

impl ComputeResult for DielectricSpectrumResult {}

// ── Shared transform helpers (relocated from compute::dielectric) ────────────

/// `(frequencies, spec_re, spec_im)` triple — the real and imaginary parts of
/// the continuous one-sided Fourier transform `X(ω) = ∫₀^∞ C(t) e^{−iωt} dt`
/// of a (possibly windowed / differentiated) ACF on the rfft frequency grid.
/// No physical prefactor (β, V, k_B T, ε₀, …) is applied.
type RawSpectrum = (Array1<f64>, Array1<f64>, Array1<f64>);

/// Convert a (windowed / differentiated) ACF to the continuous one-sided FT
/// via DFT.
///
/// Returns `X.re` and `X.im` where `X(ω_k) = ∫₀^T C(t) e^{−iωt} dt` under the
/// `e^{−iωt}` Fourier convention. Scaling uses the rectangle-rule `dt` factor,
/// not the FFT's internal `1/n_pad`. The pad+forward-FFT core is the shared
/// [`forward_fft_onesided`](crate::compute::fit::forward_fft_onesided); the dielectric path
/// keeps its own rad·(time)⁻¹ frequency grid and `·dt` 3-tuple scaling.
fn acf_to_spectrum(
    planner: &mut FftPlanner<f64>,
    acf: &Array1<f64>,
    dt: f64,
    n_pad: usize,
) -> RawSpectrum {
    let acf_vec;
    let acf_slice = match acf.as_slice() {
        Some(s) => s,
        None => {
            acf_vec = acf.to_vec();
            &acf_vec
        }
    };
    let bins = forward_fft_onesided(planner, acf_slice, n_pad);

    let frequencies = sig::frequency_grid(n_pad, dt);
    let n_freq = frequencies.len();
    let mut spec_re = Array1::zeros(n_freq);
    let mut spec_im = Array1::zeros(n_freq);

    for j in 0..n_freq {
        let z = bins[j];
        spec_re[j] = z.re * dt;
        spec_im[j] = z.im * dt;
    }

    (frequencies, spec_re, spec_im)
}

/// One-sided cosine² taper → central-difference derivative → one-sided FT of a
/// raw fluctuation dipole ACF.
///
/// Returns `(frequencies, dre, dim)` where `dre + i·dim ≈ Ĉ′(ω) =
/// ∫₀^∞ C′(t) e^{−iωt} dt`. Transforming the ACF *derivative* — rather than
/// forming `ω·X(ω)` from the transform `X(ω)` of the ACF itself — is what keeps
/// the loss spectrum finite. `C′(0) = 0` (the ACF is even) and the cos² taper
/// (1 at `C(0)`, 0 at `C(L)`) drives `C(L) → 0`, so `Ĉ′(ω)` decays toward the
/// Nyquist frequency instead of diverging.
///
/// The input `acf` must be the **fluctuation** (mean-subtracted) dipole ACF
/// `C(k) = ⟨δM(0)·δM(k·dt)⟩` summed over the 3 Cartesian components — exactly
/// the [`DebyeRelaxationResult.acf`](crate::compute::transport::DebyeRelaxationResult::acf).
fn taper_derivative_spectrum(acf: &Array1<f64>, dt: f64) -> RawSpectrum {
    let max_lag = acf.len() - 1;
    let mut tapered = acf.clone();

    // One-sided cosine² taper: 1 at C(0), 0 at C(max_lag).
    let denom = 2.0 * max_lag.max(1) as f64;
    for k in 0..=max_lag {
        let angle = std::f64::consts::PI * k as f64 / denom;
        tapered[k] *= angle.cos().powi(2);
    }

    // Central finite-difference derivative of the tapered ACF. C′(0) = 0
    // exactly because the autocorrelation is even; the tail uses a one-sided
    // stencil (the taper has already driven C(max_lag) to zero).
    let mut deriv = Array1::<f64>::zeros(max_lag + 1);
    for k in 1..max_lag {
        deriv[k] = (tapered[k + 1] - tapered[k - 1]) / (2.0 * dt);
    }
    if max_lag >= 1 {
        deriv[max_lag] = (tapered[max_lag] - tapered[max_lag - 1]) / dt;
    }

    let n_pad = (2 * (max_lag + 1)).next_power_of_two();
    let mut planner = FftPlanner::new();
    acf_to_spectrum(&mut planner, &deriv, dt, n_pad)
}

fn parse_window_type(s: &str) -> Result<sig::WindowType, ComputeError> {
    match s {
        "hann" => Ok(sig::WindowType::Hann),
        "blackman" => Ok(sig::WindowType::Blackman),
        // Preferred for one-sided ACFs (1 at t=0, 0 at t=max_lag).
        // Hann/Blackman zero out C(0), the static-ε signal.
        "cosine_sq" => Ok(sig::WindowType::CosineSq),
        other => Err(ComputeError::OutOfRange {
            field: "window_type",
            value: other.into(),
        }),
    }
}

/// Window a raw current ACF, zero-pad, and forward-FFT into a one-sided FT.
///
/// Returns `(frequencies, spec_re, spec_im)` where `spec_re + i·spec_im` is the
/// continuous FT `X(ω)` of the windowed ACF on the rfft frequency grid. The
/// input `acf` must be the unbiased current ACF `C(k) = ⟨J(0)·J(k·dt)⟩` summed
/// over the 3 Cartesian components — exactly the
/// [`GreenKuboConductivityResult.jacf`](crate::compute::transport::GreenKuboConductivityResult::jacf).
fn windowed_acf_spectrum(
    acf: &Array1<f64>,
    dt: f64,
    window_type: &str,
) -> Result<RawSpectrum, ComputeError> {
    let max_lag = acf.len() - 1;
    let wt = parse_window_type(window_type)?;

    let acf_dyn = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[max_lag + 1]), acf.to_vec())
        .map_err(|e| ComputeError::BadShape {
            expected: "1d".into(),
            got: e.to_string(),
        })?;
    let windowed = sig::apply_window(&acf_dyn, wt, 0).map_err(|e| ComputeError::OutOfRange {
        field: "apply_window",
        value: e.to_string(),
    })?;
    let windowed_1d: Array1<f64> = windowed.iter().copied().collect();

    let n_pad = (2 * (max_lag + 1)).next_power_of_two();
    let mut planner = FftPlanner::new();
    Ok(acf_to_spectrum(&mut planner, &windowed_1d, dt, n_pad))
}

// ── Einstein–Helfand ε(ω) transform ──────────────────────────────────────────

/// Einstein–Helfand (dipole-ACF) route to the frequency-dependent permittivity.
///
/// Consumes the **raw fluctuation dipole ACF** `C(k) = ⟨δM(0)·δM(k·dt)⟩`
/// (summed over the 3 Cartesian components) — the
/// [`DebyeRelaxationResult.acf`](crate::compute::transport::DebyeRelaxationResult::acf) — together
/// with `dt`, `V`, `T`, `ε_∞`, and the zero-lag variance ⟨|δM|²⟩ (the
/// [`zero_lag_variance`](crate::compute::transport::DebyeRelaxationResult::zero_lag_variance), which
/// pins the exact DC bin).
///
/// Implements the integration-by-parts form of Caillol-Levesque-Weis Eq. (30):
///
/// ```text
///     ε*(ω) − ε_∞ = −A · Ĉ′(ω),     A = 4π·KAPPA / (3·V·k_B·T)
///     Ĉ′(ω) = ∫₀^∞ C′(t) e^{−iωt} dt
/// ```
///
/// Under the positive-loss convention `ε* = ε′ − i·ε″`, with
/// `Ĉ′ = Ĉ′_re + i·Ĉ′_im`:
///
/// ```text
///     ε′(ω) − ε_∞ = −A · Ĉ′_re(ω)
///     ε″(ω)       =  A · Ĉ′_im(ω)
/// ```
///
/// At `ω = 0` the result contracts to the Neumann static formula
/// `ε(0) − ε_∞ = A·⟨|δM|²⟩`; the DC bin is set to that exact value, matching
/// [`static_dielectric_constant`](crate::compute::static_dielectric_constant).
///
/// # References
/// - Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986), Eq. (30).
/// - Neumann, *Mol. Phys.* **50**, 841 (1983), static-limit identity.
#[derive(Debug, Clone, Copy)]
pub struct EinsteinHelfandSpectrum {
    /// Frame spacing, **ps**, > 0. Sets the frequency resolution
    /// `Δω = 2π / (n_pad · dt)`.
    pub dt: f64,
    /// System volume V, **Å³**, > 0.
    pub volume: f64,
    /// Temperature T, **K**, > 0.
    pub temperature: f64,
    /// High-frequency / electronic permittivity ε_∞, dimensionless.
    pub epsilon_inf: f64,
    /// Zero-lag variance ⟨|δM|²⟩ = `acf[0]`, **(e·Å)²** — the exact DC term
    /// (the [`DebyeRelaxationResult.zero_lag_variance`](crate::compute::transport::DebyeRelaxationResult::zero_lag_variance)).
    pub zero_lag_variance: f64,
}

impl Fit for EinsteinHelfandSpectrum {
    /// The raw fluctuation dipole ACF (1D, length `max_lag + 1`).
    type Input<'a> = &'a Array1<f64>;
    type Output = DielectricSpectrumResult;

    /// Taper + derivative + FFT the raw dipole ACF into ε(ω).
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF has fewer than 2 lags.
    /// * [`ComputeError::OutOfRange`] if `dt`/`volume`/`temperature` ≤ 0.
    fn fit<'a>(&self, acf: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if acf.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        validate_thermo(self.dt, self.volume, self.temperature)?;

        let (frequencies, dre, dim) = taper_derivative_spectrum(acf, self.dt);

        // ε*(ω) − ε_∞ = −A·Ĉ′(ω), with ε* = ε′ − i·ε″ (positive-loss convention).
        let prefactor = FOUR_PI_OVER_3 * KAPPA / (self.volume * K_B * self.temperature);
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            eps_real[j] = self.epsilon_inf - prefactor * dre[j];
            eps_imag[j] = prefactor * dim[j];
        }
        // DC bin: contract to the exact Neumann static limit. The discrete
        // derivative transform reproduces it only up to O(dt) truncation error.
        eps_real[0] = self.epsilon_inf + prefactor * self.zero_lag_variance;
        eps_imag[0] = 0.0;

        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
        })
    }
}

// ── Green–Kubo ε(ω) transform ─────────────────────────────────────────────────

/// Green–Kubo (current-ACF) route to the frequency-dependent permittivity.
///
/// Consumes the **raw current ACF** `C(k) = ⟨J(0)·J(k·dt)⟩` (summed over the 3
/// Cartesian components) — the
/// [`GreenKuboConductivityResult.jacf`](crate::compute::transport::GreenKuboConductivityResult::jacf)
/// — together with `dt`, `V`, `T`, `ε_∞`, and the window choice.
///
/// Implements
///
/// ```text
///     σ(ω) = (V / (3·k_B·T)) · ∫₀^∞ ⟨J(0)·J(t)⟩ e^{−iωt} dt = σ′ + i·σ″
///     ε*(ω) − ε_∞ = −i · σ(ω) / (ω · ε₀)
/// ```
///
/// (Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20; equivalent to
/// Eq. (39) of Caillol-Levesque-Weis after substituting `J = Ṁ/V`).
/// Separating real/imaginary parts under `ε* = ε′ − i·ε″`:
///
/// ```text
///     ε′(ω) − ε_∞ = σ″(ω) / (ω · ε₀)
///     ε″(ω)       = σ′(ω) / (ω · ε₀)
/// ```
///
/// with `1/ε₀ = 4π·KAPPA` in LAMMPS real units. The DC bin (`ω = 0`) is
/// regularized to `(ε_∞, 0)`; the physical static limit must be taken from
/// [`static_dielectric_constant`](crate::compute::static_dielectric_constant)
/// for insulators or extrapolated from low-ω bins for conductors.
///
/// # References
/// - Caillol, Levesque & Weis, *J. Chem. Phys.* **85**, 6645 (1986), Eqs.
///   (36)–(39).
/// - Hansen & McDonald, *Theory of Simple Liquids*, Eq. 7.7.20.
#[derive(Debug, Clone)]
pub struct GreenKuboSpectrum {
    /// Frame spacing, **ps**, > 0.
    pub dt: f64,
    /// System volume V, **Å³**, > 0.
    pub volume: f64,
    /// Temperature T, **K**, > 0.
    pub temperature: f64,
    /// High-frequency / electronic permittivity ε_∞, dimensionless.
    pub epsilon_inf: f64,
    /// Window applied to the one-sided current ACF: `"cosine_sq"`, `"hann"`, or
    /// `"blackman"`.
    pub window_type: String,
}

impl Fit for GreenKuboSpectrum {
    /// The raw current ACF (1D, length `max_lag + 1`).
    type Input<'a> = &'a Array1<f64>;
    type Output = DielectricSpectrumResult;

    /// Window + FFT the raw current ACF and convert σ(ω) → ε(ω).
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF has fewer than 2 lags.
    /// * [`ComputeError::OutOfRange`] if `dt`/`volume`/`temperature` ≤ 0 or the
    ///   window type is unknown.
    fn fit<'a>(&self, acf: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if acf.len() < 2 {
            return Err(ComputeError::EmptyInput);
        }
        validate_thermo(self.dt, self.volume, self.temperature)?;

        let (frequencies, spec_re, spec_im) =
            windowed_acf_spectrum(acf, self.dt, &self.window_type)?;

        // σ(ω) = sigma_prefactor · (spec_re + i·spec_im) under e^{−iωt}.
        // ε*(ω) − ε_∞ = −i·σ(ω)/(ω·ε₀):
        //   ε′(ω) − ε_∞ = σ″/(ω·ε₀) = (1/ε₀)·sigma_prefactor·spec_im/ω
        //   ε″(ω)       = σ′/(ω·ε₀) = (1/ε₀)·sigma_prefactor·spec_re/ω
        // With J(t) = Ṁ(t)/V: ⟨Ṁ·Ṁ⟩ = V²·⟨J·J⟩, so the textbook
        // 1/(3·V·k_B·T) prefactor for Ṁ becomes V/(3·k_B·T) for J.
        // 1/ε₀ = 4π·KAPPA in MD real units.
        let sigma_prefactor = self.volume / (3.0 * K_B * self.temperature);
        let eps0_factor = 4.0 * std::f64::consts::PI * KAPPA;
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            let omega = frequencies[j];
            if omega == 0.0 {
                // DC bin: σ/ω is 0/0. The true static limit lives in
                // `static_dielectric_constant`; this routine regularizes the
                // bin to (ε_∞, 0) and the caller must consult the static helper.
                eps_real[j] = self.epsilon_inf;
                eps_imag[j] = 0.0;
            } else {
                let sigma_re = sigma_prefactor * spec_re[j];
                let sigma_im = sigma_prefactor * spec_im[j];
                eps_real[j] = self.epsilon_inf + eps0_factor * sigma_im / omega;
                eps_imag[j] = eps0_factor * sigma_re / omega;
            }
        }

        Ok(DielectricSpectrumResult {
            frequencies,
            eps_real,
            eps_imag,
        })
    }
}

/// Validate the shared `dt`/`volume`/`temperature` > 0 contract.
fn validate_thermo(dt: f64, volume: f64, temperature: f64) -> Result<(), ComputeError> {
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    if volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "volume",
            value: volume.to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "temperature",
            value: temperature.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::traits::Compute;
    use crate::compute::transport::{DebyeRelaxation, EwaldBoundary, GreenKuboConductivity};
    use molrs::Frame;
    use ndarray::Array2;
    use rustfft::FftPlanner;

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_dipole(n: usize, seed: u64) -> Array2<f64> {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, 3));
        for t in 0..n {
            for d in 0..3 {
                s[[t, d]] = rng.random::<f64>() - 0.5;
            }
        }
        s
    }

    // ── Legacy-equivalent reference implementations (the pre-migration bodies)
    // These rebuild the exact spectra the removed `einstein_helfand_spectrum` /
    // `green_kubo_spectrum` free fns produced, so the Fit can be locked to them
    // bit-for-bit (ac-001). ──────────────────────────────────────────────────

    /// Pre-migration EH spectrum (the removed `einstein_helfand_spectrum`).
    #[allow(clippy::too_many_arguments)]
    fn legacy_einstein_helfand(
        dipole_moments: &Array2<f64>,
        dt: f64,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
        max_correlation_time: usize,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n_frames = dipole_moments.shape()[0];
        let max_lag = max_correlation_time.min(n_frames - 1);

        // ⟨|δM|²⟩ — the exact static (ω = 0) term.
        let n = n_frames as f64;
        let mut mean_m = [0.0_f64; 3];
        for t in 0..n_frames {
            for d in 0..3 {
                mean_m[d] += dipole_moments[[t, d]];
            }
        }
        for m in mean_m.iter_mut() {
            *m /= n;
        }
        let mut m_sq = 0.0;
        for t in 0..n_frames {
            for d in 0..3 {
                let dev = dipole_moments[[t, d]] - mean_m[d];
                m_sq += dev * dev;
            }
        }
        m_sq /= n;

        // Fluctuation ACF, summed over components, unbiased estimator.
        let mut planner = FftPlanner::new();
        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (0..n_frames)
                .map(|t| dipole_moments[[t, d]] - mean_m[d])
                .collect();
            let component = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf[k] += component[k];
            }
        }
        for k in 0..=max_lag {
            acf[k] /= (n_frames - k) as f64;
        }

        // One-sided cos² taper.
        let denom = 2.0 * max_lag.max(1) as f64;
        for k in 0..=max_lag {
            let angle = std::f64::consts::PI * k as f64 / denom;
            acf[k] *= angle.cos().powi(2);
        }
        // Central finite-difference derivative.
        let mut deriv = Array1::<f64>::zeros(max_lag + 1);
        for k in 1..max_lag {
            deriv[k] = (acf[k + 1] - acf[k - 1]) / (2.0 * dt);
        }
        if max_lag >= 1 {
            deriv[max_lag] = (acf[max_lag] - acf[max_lag - 1]) / dt;
        }
        let n_pad = (2 * (max_lag + 1)).next_power_of_two();
        let (frequencies, dre, dim) = acf_to_spectrum(&mut planner, &deriv, dt, n_pad);

        let prefactor = FOUR_PI_OVER_3 * KAPPA / (volume * K_B * temperature);
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            eps_real[j] = epsilon_inf - prefactor * dre[j];
            eps_imag[j] = prefactor * dim[j];
        }
        eps_real[0] = epsilon_inf + prefactor * m_sq;
        eps_imag[0] = 0.0;
        (frequencies, eps_real, eps_imag)
    }

    /// The unbiased current ACF the legacy `green_kubo_spectrum` built
    /// internally (FFT-based, over the post-NaN `start=1` series). Returned so a
    /// test can lock the Fit's transform tail to the legacy tail on the *same*
    /// ACF (bit-for-bit), independent of the raw-compute estimator.
    fn legacy_gk_acf(current: &Array2<f64>, max_lag: usize) -> Array1<f64> {
        let n_frames = current.shape()[0];
        let start = 1;
        let n_eff = n_frames - start;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (start..n_frames).map(|t| current[[t, d]]).collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        for k in 0..=max_lag {
            acf_sum[k] /= (n_eff - k) as f64;
        }
        acf_sum
    }

    /// Pre-migration GK spectrum tail (the removed `green_kubo_spectrum`),
    /// operating on a pre-built ACF (the `windowed_acf_spectrum` + σ→ε steps).
    fn legacy_green_kubo_from_acf(
        acf: &Array1<f64>,
        dt: f64,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
        window_type: &str,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let max_lag = acf.len() - 1;
        let wt = parse_window_type(window_type).unwrap();
        let acf_dyn =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[max_lag + 1]), acf.to_vec()).unwrap();
        let windowed = sig::apply_window(&acf_dyn, wt, 0).unwrap();
        let windowed_1d: Array1<f64> = windowed.iter().copied().collect();
        let n_pad = (2 * (max_lag + 1)).next_power_of_two();
        let mut planner = FftPlanner::new();
        let (frequencies, spec_re, spec_im) =
            acf_to_spectrum(&mut planner, &windowed_1d, dt, n_pad);

        let sigma_prefactor = volume / (3.0 * K_B * temperature);
        let eps0_factor = 4.0 * std::f64::consts::PI * KAPPA;
        let n_freq = frequencies.len();
        let mut eps_real = Array1::zeros(n_freq);
        let mut eps_imag = Array1::zeros(n_freq);
        for j in 0..n_freq {
            let omega = frequencies[j];
            if omega == 0.0 {
                eps_real[j] = epsilon_inf;
                eps_imag[j] = 0.0;
            } else {
                let sigma_re = sigma_prefactor * spec_re[j];
                let sigma_im = sigma_prefactor * spec_im[j];
                eps_real[j] = epsilon_inf + eps0_factor * sigma_im / omega;
                eps_imag[j] = eps0_factor * sigma_re / omega;
            }
        }
        (frequencies, eps_real, eps_imag)
    }

    #[test]
    fn eh_fit_reproduces_legacy_bit_for_bit() {
        // ac-001: DebyeRelaxation raw ACF + EinsteinHelfandSpectrum reproduces
        // the legacy einstein_helfand_spectrum output bit-for-bit.
        let n = 256;
        let dt = 0.001;
        let (vol, temp, eps_inf) = (1000.0, 300.0, 1.5);
        let mct = 50;
        let dm = rng_dipole(n, 42);

        let (freq_l, re_l, im_l) = legacy_einstein_helfand(&dm, dt, vol, temp, eps_inf, mct);

        let raw = DebyeRelaxation {
            volume: vol,
            temperature: temp,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dm, dt, mct))
        .unwrap();
        let fit = EinsteinHelfandSpectrum {
            dt,
            volume: vol,
            temperature: temp,
            epsilon_inf: eps_inf,
            zero_lag_variance: raw.zero_lag_variance,
        }
        .fit(&raw.acf)
        .unwrap();

        assert_eq!(fit.frequencies, freq_l);
        assert_eq!(fit.eps_real, re_l);
        assert_eq!(fit.eps_imag, im_l);
    }

    #[test]
    fn gk_fit_reproduces_legacy_bit_for_bit() {
        // ac-001: on the SAME raw current ACF, GreenKuboSpectrum reproduces the
        // legacy green_kubo_spectrum transform tail bit-for-bit. The raw ACF is
        // the unbiased current ACF over the start=1 (post-NaN) series — exactly
        // what the legacy fn built internally.
        let n = 256;
        let dt = 0.001;
        let (vol, temp, eps_inf) = (1000.0, 300.0, 1.0);
        let mct = 50;
        // Current density with NaN row 0 (as compute_current_density emits).
        let mut current = rng_dipole(n, 7);
        for d in 0..3 {
            current[[0, d]] = f64::NAN;
        }
        let start = 1;
        let effective_len = n - start;
        let max_lag = mct.min(effective_len.saturating_sub(1));
        let raw_acf = legacy_gk_acf(&current, max_lag);

        for window in ["hann", "blackman", "cosine_sq"] {
            let (freq_l, re_l, im_l) =
                legacy_green_kubo_from_acf(&raw_acf, dt, vol, temp, eps_inf, window);

            let fit = GreenKuboSpectrum {
                dt,
                volume: vol,
                temperature: temp,
                epsilon_inf: eps_inf,
                window_type: window.to_string(),
            }
            .fit(&raw_acf)
            .unwrap();

            assert_eq!(fit.frequencies, freq_l, "window={window}");
            assert_eq!(fit.eps_real, re_l, "window={window}");
            assert_eq!(fit.eps_imag, im_l, "window={window}");
        }
    }

    #[test]
    fn gk_raw_compute_acf_matches_legacy_fft_acf() {
        // The GreenKuboConductivity raw compute (direct-summation estimator) and
        // the legacy FFT-based ACF agree to FP tolerance on the same series, so
        // composing the raw compute with GreenKuboSpectrum reproduces the legacy
        // ε(ω) within that tolerance.
        let n = 256;
        let dt = 0.001;
        let mct = 50;
        let mut current = rng_dipole(n, 7);
        for d in 0..3 {
            current[[0, d]] = f64::NAN;
        }
        let start = 1;
        let effective_len = n - start;
        let max_lag = mct.min(effective_len.saturating_sub(1));
        let legacy = legacy_gk_acf(&current, max_lag);

        let post: Array2<f64> = current.slice(ndarray::s![start.., ..]).to_owned();
        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&post, dt, max_lag))
            .unwrap();
        assert_eq!(raw.jacf.len(), legacy.len());
        for k in 0..legacy.len() {
            assert!((raw.jacf[k] - legacy[k]).abs() < 1e-9, "k={k}");
        }
    }

    #[test]
    fn eh_dc_bin_recovers_neumann_static() {
        // ac-006: ε(ω=0) recovers the Neumann static dielectric constant.
        use crate::compute::static_dielectric_constant;
        let n = 256;
        let dt = 0.001;
        let (vol, temp, eps_inf) = (1000.0, 300.0, 1.5);
        let dm = rng_dipole(n, 99);
        let static_eps = static_dielectric_constant(&dm, vol, temp, eps_inf).unwrap();

        let raw = DebyeRelaxation {
            volume: vol,
            temperature: temp,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dm, dt, 50))
        .unwrap();
        let fit = EinsteinHelfandSpectrum {
            dt,
            volume: vol,
            temperature: temp,
            epsilon_inf: eps_inf,
            zero_lag_variance: raw.zero_lag_variance,
        }
        .fit(&raw.acf)
        .unwrap();
        assert!((fit.eps_real[0] - static_eps).abs() < 1e-10);
        assert_eq!(fit.eps_imag[0], 0.0);
    }

    #[test]
    fn eh_loss_finite_to_nyquist() {
        // ac-006: the loss spectrum ε″ stays finite to Nyquist (derivative-FT).
        let n = 2048;
        let dt = 1.0;
        let dm = rng_dipole(n, 11);
        let raw = DebyeRelaxation {
            volume: 1000.0,
            temperature: 300.0,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dm, dt, 100))
        .unwrap();
        let fit = EinsteinHelfandSpectrum {
            dt,
            volume: 1000.0,
            temperature: 300.0,
            epsilon_inf: 1.0,
            zero_lag_variance: raw.zero_lag_variance,
        }
        .fit(&raw.acf)
        .unwrap();
        assert!(fit.eps_imag.iter().all(|x| x.is_finite()));
        assert!(fit.eps_real.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn fits_reject_short_acf() {
        let tiny = Array1::from_vec(vec![1.0]);
        assert!(matches!(
            EinsteinHelfandSpectrum {
                dt: 1.0,
                volume: 1.0,
                temperature: 1.0,
                epsilon_inf: 1.0,
                zero_lag_variance: 1.0,
            }
            .fit(&tiny),
            Err(ComputeError::EmptyInput)
        ));
        assert!(matches!(
            GreenKuboSpectrum {
                dt: 1.0,
                volume: 1.0,
                temperature: 1.0,
                epsilon_inf: 1.0,
                window_type: "hann".into(),
            }
            .fit(&tiny),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn gk_rejects_bad_window() {
        let acf = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        assert!(matches!(
            GreenKuboSpectrum {
                dt: 1.0,
                volume: 1.0,
                temperature: 1.0,
                epsilon_inf: 1.0,
                window_type: "nope".into(),
            }
            .fit(&acf),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
