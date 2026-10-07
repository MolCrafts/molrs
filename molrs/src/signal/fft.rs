//! One-sided forward FFT of a zero-padded real signal.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::num_traits::Zero;

/// Shared one-sided forward-FFT core: zero-pad a real signal to `n_pad`,
/// forward-FFT, and return the first `n_pad/2 + 1` complex bins **unscaled**.
///
/// The one forward-FFT step shared by the spectra and the dielectric spectra
/// of `molrs::compute`. The callers diverge purely in scaling/units:
///
/// - spectra: `intensity[j] = bin[j].re / n_pad`, frequency grid in cm⁻¹.
/// - dielectric: `re[j] = bin[j].re·dt`, `im[j] = bin[j].im·dt`, frequency grid
///   in rad·(time)⁻¹.
///
/// Each caller keeps its own scaling wrapper; this helper does no scaling.
pub fn forward_fft_onesided(
    planner: &mut FftPlanner<f64>,
    signal: &[f64],
    n_pad: usize,
) -> Vec<Complex64> {
    let fwd = planner.plan_fft_forward(n_pad);
    let mut complex_data: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    complex_data.resize(n_pad, Complex64::zero());
    fwd.process(&mut complex_data);
    let n_freq = n_pad / 2 + 1;
    complex_data.truncate(n_freq);
    complex_data
}
