//! Pair Mode Fourier Transform (PMFT) analyzers.
//!
//! Family of 2-D / 3-D pair distribution histograms in fixed reference
//! frames, ported from `freud.pmft`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/)).
//!
//! | Method | Bins pair vectors in |
//! |--------|----------------------|
//! | [`PmftR12`] | `(r, θ₁, θ₂)` — distance + the two body-frame angles (2-D systems) |
//! | [`PmftXy`] | `(x, y)` — the reference particle's body frame (2-D) |
//! | [`PmftXyt`] | `(x, y, θ)` — body frame + relative orientation (2-D) |
//! | [`PmftXyz`] | `(x, y, z)` — the reference particle's body frame (3-D) |
//!
//! Each takes per-frame neighbor lists plus per-particle orientations via its
//! `Args` struct and produces a binned free-energy surface
//! `-ln g(...)` (see each `*Result`).

mod r12;
mod xy;
mod xyt;
mod xyz;

pub use r12::{PmftR12, PmftR12Args, PmftR12Result};
pub use xy::{PmftXy, PmftXyArgs, PmftXyResult};
pub use xyt::{PmftXyt, PmftXytArgs, PmftXytResult};
pub use xyz::{PmftXyz, PmftXyzArgs, PmftXyzResult};
