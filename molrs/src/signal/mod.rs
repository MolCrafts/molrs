//! Signal-processing primitives for molrs analysis crates.
//!
//! Three families:
//!
//! * [`acf_fft`] / [`xcorr_fft`] (and their accumulating and planner-taking
//!   forms) — linear auto- and cross-correlation via Wiener–Khinchin
//!   (FFT-based).
//! * [`apply_window`] / [`WindowType`] — Hann and Blackman window functions
//!   for spectral estimation.
//! * [`frequency_grid`] — equally-spaced angular-frequency grids matched to
//!   the conventions of [`acf_fft`] + zero-padded FFTs.
//!
//! No unit assumptions; the caller chooses dt and interprets the
//! returned ω accordingly (`rad / [time]` where `[time]` matches dt).
//!
//! Higher-level dielectric / MSD / VACF analyses live in
//! `molrs-compute` and compose these primitives.

mod acf;
mod grid;
mod window;

pub use acf::{
    SignalError, acf_fft, acf_fft_accumulate, acf_fft_with_planner, xcorr_fft,
    xcorr_fft_accumulate, xcorr_fft_with_planner,
};
pub use grid::frequency_grid;
pub use window::{WindowType, apply_window};
