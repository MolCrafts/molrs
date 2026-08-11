//! Signal-processing primitives for molrs analysis crates.
//!
//! Three modules:
//!
//! * [`acf`] — linear autocorrelation via Wiener–Khinchin (FFT-based).
//! * [`window`] — Hann and Blackman window functions for spectral
//!   estimation.
//! * [`grid`] — equally-spaced angular-frequency grids matched to
//!   the conventions of [`acf`] + zero-padded FFTs.
//!
//! No unit assumptions; the caller chooses dt and interprets the
//! returned ω accordingly (`rad / [time]` where `[time]` matches dt).
//!
//! Higher-level dielectric / MSD / VACF analyses live in
//! `molrs-compute` and compose these primitives.

pub mod acf;
pub mod grid;
pub mod window;

pub use acf::{acf_fft, acf_fft_with_planner, xcorr_fft, xcorr_fft_with_planner};
pub use grid::frequency_grid;
pub use window::{WindowType, apply_window};
