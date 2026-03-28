//! Precision-stable random helpers for packing paths.

use molrs::types::F;
use rand::{Rng, RngCore};

/// Draw a uniform random number in `[0, 1)` from an f64 stream, then cast to `F`.
///
/// This keeps the RNG trajectory identical across `f32`/`f64` builds and isolates
/// true numeric-precision differences from type-dependent random draws.
#[inline]
pub fn uniform01(rng: &mut impl Rng) -> F {
    rng.random::<f64>() as F
}

/// Same as [`uniform01`], but for trait-object RNGs used by hook runners.
#[inline]
pub fn uniform01_core(rng: &mut dyn RngCore) -> F {
    let unit = (rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
    unit as F
}
