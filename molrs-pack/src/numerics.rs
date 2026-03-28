//! Precision-aware numeric control helpers.

use molrs::types::F;

#[derive(Clone, Copy)]
pub(crate) struct NumericControls {
    pub(crate) steabs: F,
    pub(crate) sterel: F,
    pub(crate) epsabs: F,
    pub(crate) epsrel: F,
}

#[inline]
pub(crate) fn numeric_controls() -> NumericControls {
    // Packmol calibrates these constants for double precision. `molrs-pack`
    // defaults to `f32`, so any control threshold tied to finite differences
    // or "same point" detection must stay above the active machine epsilon.
    let eps = F::EPSILON;
    NumericControls {
        steabs: (1.0e-10 as F).max(eps),
        sterel: (1.0e-7 as F).max(eps.sqrt()),
        epsabs: (1.0e-20 as F).max(eps * eps),
        epsrel: (1.0e-10 as F).max(eps),
    }
}

#[inline]
pub(crate) fn objective_small_floor() -> F {
    (1.0e-10 as F).max(F::EPSILON)
}

#[inline]
pub(crate) fn residual_small_floor() -> F {
    (1.0e-10 as F).max(F::EPSILON)
}

#[inline]
pub(crate) fn near_zero_norm_floor() -> F {
    F::EPSILON.sqrt()
}

#[inline]
pub(crate) fn positive_norm_floor() -> F {
    F::MIN_POSITIVE
}
