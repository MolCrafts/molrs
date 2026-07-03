//! Pair Mode Fourier Transform (PMFT) analyzers.
//!
//! Family of 2-D / 3-D pair distribution histograms in fixed reference
//! frames, ported from `freud.pmft`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/)).
//!
//! | Method | Bins pair vectors in |
//! |--------|----------------------|
//! | [`PMFTR12`] | `(r, θ₁, θ₂)` — distance + the two body-frame angles (2-D systems) |
//! | [`PMFTXY`] | `(x, y)` — the reference particle's body frame (2-D) |
//! | [`PMFTXYT`] | `(x, y, θ)` — body frame + relative orientation (2-D) |
//! | [`PMFTXYZ`] | `(x, y, z)` — the reference particle's body frame (3-D) |
//!
//! Each takes per-frame neighbor lists plus per-particle orientations via its
//! `Args` struct and produces a binned free-energy surface
//! `-ln g(...)` (see each `*Result`).

pub mod r12;
pub mod xy;
pub mod xyt;
pub mod xyz;

pub use r12::{PMFTR12, PMFTR12Args, PMFTR12Result};
pub use xy::{PMFTXY, PMFTXYArgs, PMFTXYResult};
pub use xyt::{PMFTXYT, PMFTXYTArgs, PMFTXYTResult};
pub use xyz::{PMFTXYZ, PMFTXYZArgs, PMFTXYZResult};
