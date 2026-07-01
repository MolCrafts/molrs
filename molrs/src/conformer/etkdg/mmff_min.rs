//! Second-stage MMFF94 cleanup minimizer (L-BFGS).
//!
//! The L-BFGS engine that used to live here has been **sunk down** into
//! `molrs-ff` as a force-field-agnostic optimizer
//! (`molrs::ff::optimize`). The ETKDG cleanup keeps consuming it through the
//! RMS-gradient entry point [`minimize_lbfgs`], whose convergence behaviour
//! (RMS gradient, no trust region, history size 8) is preserved exactly — so
//! conformer generation is unchanged. New callers wanting `fmax`-based
//! geometry optimization should use `molrs::ff::minimize` /
//! `molrs::ff::minimize_batch` instead.

/// RMS-gradient L-BFGS, the historical ETKDG cleanup entry point.
/// Returns `(energy, grad_rms, steps, converged)`.
pub use molrs::optimize::minimize_lbfgs_rms as minimize_lbfgs;
