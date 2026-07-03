//! Vibrational-spectrum result types and shared input validation.
//!
//! The three vibrational spectra (VDOS / IR / Raman) are now the explicit
//! composition of a **raw-ACF compute** with a **spectral
//! [`Fit`](crate::compute::Fit) transform**,
//! keeping "what was measured" separate from "how the analyst transforms it":
//!
//! | spectrum | raw compute (raw ACF) | transform |
//! |----------|-----------------------|-----------|
//! | VDOS  | [`VACF`](crate::compute::transport::VACF) (velocity ACF) | [`PowerSpectrum`](super::PowerSpectrum) |
//! | IR    | [`IRFlux`](super::IRFlux) (dipole-flux ACF) | [`IRSpectrum`](super::IRSpectrum) |
//! | Raman | [`RamanTensor`](super::RamanTensor) (polarizability iso/aniso ACFs) | [`RamanSpectrum`](super::RamanSpectrum) |
//!
//! The legacy `power_spectrum` / `ir_spectrum` / `raman_spectrum` free functions
//! (which baked window + FFT into the raw ACF) and their inline window/FFT
//! helpers were removed in compute-fit-03-cleanup; the window + one-sided-FFT
//! machinery now lives in [`compute::spectroscopy`](crate::compute::spectroscopy) (a
//! windowed transform is a fit), routing every window through
//! [`molrs::signal`]. Only the two result types and the shared input validator
//! remain here.
//!
//! Reference: Dickey & Paskin, *Phys. Rev.* **188**, 1407 (1969).
//!
//! # Units
//!
//! | quantity      | unit   |
//! |---------------|--------|
//! | time / dt     | fs     |
//! | frequency     | cm⁻¹   |
//! | intensity     | arb.   |
//! | temperature   | K      |

use ndarray::Array1;

use crate::compute::result::ComputeResult;

// ── Result types ─────────────────────────────────────────────────────────────

/// Single-spectrum result (VDOS, IR).
#[derive(Debug, Clone)]
pub struct SpectrumResult {
    /// Frequency grid in cm⁻¹, length `n_pad / 2 + 1`.
    pub frequencies_cm1: Array1<f64>,
    /// Spectral intensities (arbitrary units).
    pub intensities: Array1<f64>,
    /// Number of ACF lags retained before windowing.
    pub resolution: usize,
    /// Number of input frames.
    pub n_frames: usize,
}

/// Raman spectrum result with isotropic / anisotropic decomposition.
#[derive(Debug, Clone)]
pub struct RamanSpectrumResult {
    /// Frequency grid in cm⁻¹.
    pub frequencies_cm1: Array1<f64>,
    /// Isotropic (trace) contribution.
    pub isotropic: Array1<f64>,
    /// Anisotropic (deviatoric) contribution.
    pub anisotropic: Array1<f64>,
    /// Parallel polarization `I_∥ = I_iso + (4/45)·I_aniso`. Set when
    /// `averaged = true`.
    pub parallel: Option<Array1<f64>>,
    /// Perpendicular polarization `I_⊥ = I_aniso / 15`. Set when
    /// `averaged = true`.
    pub perpendicular: Option<Array1<f64>>,
    /// Number of ACF lags retained.
    pub resolution: usize,
    /// Number of input frames.
    pub n_frames: usize,
}

impl ComputeResult for SpectrumResult {}
impl ComputeResult for RamanSpectrumResult {}
