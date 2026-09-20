//! Arithmetic on `[F; 3]` for the geometry kernels that stay on stack arrays
//! ([`mesh`](super::mesh), [`bvh`](super::bvh), the mesh and sphere-union
//! regions). ndarray is the API type; these are the inner-loop primitives.

use crate::types::F;

#[inline]
pub(crate) fn sub(a: [F; 3], b: [F; 3]) -> [F; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn add(a: [F; 3], b: [F; 3]) -> [F; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn scale(a: [F; 3], s: F) -> [F; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn dot(a: [F; 3], b: [F; 3]) -> F {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn cross(a: [F; 3], b: [F; 3]) -> [F; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn norm(a: [F; 3]) -> F {
    dot(a, a).sqrt()
}
