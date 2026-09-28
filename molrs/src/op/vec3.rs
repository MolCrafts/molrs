//! Arithmetic on [`Vec3`] (`[F; 3]`): the inner-loop vector primitives.
//!
//! ndarray is the API type of the column stores; these stack kernels are what
//! geometry code computes on. Plain free functions over plain values, by the
//! operator ruling that scopes a functional style to `molrs::op`.

use crate::op::types::{F, Vec3};

/// Shortest length a vector must have to count as a direction.
///
/// A length in the caller's coordinate unit, which is Å everywhere in molrs
/// (every coordinate reader normalises to Å). The value sits between two
/// scales with orders of magnitude to spare:
///
/// - far below any bond length (H₂: 0.74 Å), so a real bond or displacement is
///   never refused;
/// - far above the rounding noise of a coordinate difference, ~1e-13 Å at
///   1e3 Å magnitude, so a difference that is zero up to rounding is never
///   mistaken for a direction.
///
/// Applied to a unit vector (a projection of a direction), the same bound is a
/// relative 1e-6.
pub const MIN_DIRECTION_LENGTH: F = 1e-6;

/// Componentwise difference `a − b`.
#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Componentwise sum `a + b`.
#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `a` with every component multiplied by `s`.
#[inline]
pub fn scale(a: Vec3, s: F) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product `a · b`.
#[inline]
pub fn dot(a: Vec3, b: Vec3) -> F {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-handed cross product `a × b`.
#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean length `|a|`.
#[inline]
pub fn norm(a: Vec3) -> F {
    dot(a, a).sqrt()
}

/// `vector` scaled to unit length, or `None` when it is not a direction: a
/// component is non-finite, its squared length overflows, or it is not longer
/// than [`MIN_DIRECTION_LENGTH`].
pub fn normalize(vector: Vec3) -> Option<Vec3> {
    let norm_sq = dot(vector, vector);
    // NaN fails both tests, so a NaN component is rejected too.
    if !(norm_sq.is_finite() && norm_sq > MIN_DIRECTION_LENGTH * MIN_DIRECTION_LENGTH) {
        return None;
    }
    let norm = norm_sq.sqrt();
    Some([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

/// A unit vector perpendicular to `axis`: `axis` crossed with the basis vector
/// least aligned with it. `None` when `axis` is not a direction ([`normalize`]).
pub fn perpendicular(axis: Vec3) -> Option<Vec3> {
    let axis = normalize(axis)?;
    let basis = if axis[0].abs() <= axis[1].abs() && axis[0].abs() <= axis[2].abs() {
        [1.0, 0.0, 0.0]
    } else if axis[1].abs() <= axis[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalize(cross(axis, basis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::types::{F, Vec3};

    const TOL: F = 1e-12;

    fn assert_vec_close(got: Vec3, want: Vec3, tol: F) {
        for d in 0..3 {
            assert!(
                (got[d] - want[d]).abs() < tol,
                "component {d}: expected {want:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn sub_takes_componentwise_difference() {
        assert_eq!(sub([5.0, 7.0, -1.0], [1.0, 2.0, 3.0]), [4.0, 5.0, -4.0]);
    }

    #[test]
    fn add_takes_componentwise_sum() {
        assert_eq!(add([5.0, 7.0, -1.0], [1.0, 2.0, 3.0]), [6.0, 9.0, 2.0]);
    }

    #[test]
    fn scale_multiplies_every_component() {
        assert_eq!(scale([1.0, -2.0, 0.5], 4.0), [4.0, -8.0, 2.0]);
    }

    #[test]
    fn dot_of_known_vectors() {
        // 1*4 + 2*(-5) + 3*6 = 12
        assert_eq!(dot([1.0, 2.0, 3.0], [4.0, -5.0, 6.0]), 12.0);
    }

    #[test]
    fn cross_follows_right_handed_basis_and_vanishes_for_parallel_vectors() {
        let i = [1.0, 0.0, 0.0];
        let j = [0.0, 1.0, 0.0];
        let k = [0.0, 0.0, 1.0];
        assert_vec_close(cross(i, j), k, TOL);
        assert_vec_close(cross(j, k), i, TOL);
        assert_vec_close(cross(k, i), j, TOL);

        let a = [2.0, 3.0, 4.0];
        let b = [4.0, 6.0, 8.0]; // 2 * a
        assert_vec_close(cross(a, b), [0.0, 0.0, 0.0], TOL);
    }

    #[test]
    fn norm_of_basis_zero_and_pythagorean_vectors() {
        assert!((norm([1.0, 0.0, 0.0]) - 1.0).abs() < TOL);
        assert!((norm([0.0, 1.0, 0.0]) - 1.0).abs() < TOL);
        assert!((norm([0.0, 0.0, 1.0]) - 1.0).abs() < TOL);
        assert_eq!(norm([0.0, 0.0, 0.0]), 0.0);
        // (3, 4, 0) -> 5
        assert!((norm([3.0, 4.0, 0.0]) - 5.0).abs() < TOL);
    }

    #[test]
    fn normalize_scales_to_unit_length() {
        let n = normalize([3.0, 4.0, 0.0]).expect("(3,4,0) is a direction");
        assert_vec_close(n, [0.6, 0.8, 0.0], TOL);
    }

    #[test]
    fn normalize_refuses_zero_vector() {
        assert_eq!(normalize([0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn normalize_refuses_vector_shorter_than_min_direction_length() {
        assert_eq!(normalize([0.1 * MIN_DIRECTION_LENGTH, 0.0, 0.0]), None);
    }

    #[test]
    fn normalize_refuses_non_finite_components() {
        assert_eq!(normalize([F::NAN, 0.0, 0.0]), None);
        assert_eq!(normalize([1.0, F::INFINITY, 0.0]), None);
        assert_eq!(normalize([0.0, 0.0, F::NEG_INFINITY]), None);
    }

    #[test]
    fn normalize_refuses_input_whose_squared_length_overflows() {
        assert_eq!(normalize([1e308, 1e308, 1e308]), None);
    }

    #[test]
    fn perpendicular_is_a_unit_vector_orthogonal_to_the_axis() {
        let axis = [1.0, 2.0, 3.0];
        let p = perpendicular(axis).expect("(1,2,3) is a direction");
        assert!((dot(p, p) - 1.0).abs() < TOL, "|p|^2 = {}", dot(p, p));
        assert!(dot(p, axis).abs() < TOL, "p . axis = {}", dot(p, axis));
    }

    #[test]
    fn perpendicular_refuses_zero_vector() {
        assert_eq!(perpendicular([0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn perpendicular_refuses_non_finite_components() {
        assert_eq!(perpendicular([F::NAN, 1.0, 0.0]), None);
        assert_eq!(perpendicular([F::INFINITY, 0.0, 0.0]), None);
    }

    #[test]
    fn perpendicular_refuses_input_whose_squared_length_overflows() {
        assert_eq!(perpendicular([1e308, 1e308, 1e308]), None);
    }
}
