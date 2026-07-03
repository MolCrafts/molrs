//! Ionic conductivity from the charge-current autocorrelation (Green–Kubo).
//!
//! The DC ionic conductivity follows from the Green–Kubo relation for the
//! collective charge current `J(t) = Σ_a q_a v_a(t)`:
//!
//! ```text
//!     σ = 1 / (3·V·k_B·T) · ∫₀^∞ ⟨J(0)·J(t)⟩ dt .
//! ```
//!
//! As of compute-fit-03-cleanup this is the explicit composition of two
//! orthogonal steps, keeping "what was measured" separate from "how the analyst
//! processes it":
//!
//! 1. [`GreenKuboConductivity`](crate::compute::GreenKuboConductivity) — the raw
//!    current autocorrelation function `C(τ) = ⟨J(0)·J(τ)⟩` over all time origins
//!    (no fitted σ).
//! 2. [`RunningIntegral`](crate::compute::fit::RunningIntegral) — the cumulative
//!    trapezoidal integral of that ACF, which the caller scales by the
//!    `1/(3·V·k_B·T)` MD→SI prefactor to obtain σ(τ) and σ.
//!
//! It is the molrs port of the `jacf` recipe from the *tame* library
//! (<https://github.com/Roy-Kid/tame>, `tame/recipes/jacf.py`). The *tame*
//! original is non-functional as published (it never actually evaluates the
//! autocorrelation before integrating); this port implements the intended
//! algorithm correctly. The collective current `J = Σ v_cation − Σ v_anion`
//! (unit charges ±1) is assembled by the caller (Python wrapper); arbitrary
//! per-ion charges are supported by pre-scaling the velocities.
//!
//! The legacy bundled `JacfResult` + `green_kubo_conductivity` free function
//! (which baked the trapezoid integral and σ/σ_running into the raw result) were
//! removed in compute-fit-03-cleanup; the raw-ACF + `RunningIntegral`
//! composition above replaces them.
//!
//! # Units
//!
//! LAMMPS *real* units, matching [`crate::compute::dielectric`]:
//!
//! | quantity     | unit              |
//! |--------------|-------------------|
//! | current `J`  | e · Å · ps⁻¹      |
//! | time / `dt`  | ps                |
//! | volume       | Å³                |
//! | temperature  | K                 |
//! | output `σ`   | S · m⁻¹ (SI)      |
//!
//! The conversion prefactor folds in `e²`, `Å→m`, and `ps→s` so the caller
//! does no unit bookkeeping (Green–Kubo `1/3` factor).
