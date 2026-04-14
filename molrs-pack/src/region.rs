//! Geometric `Region` trait and composition combinators.
//!
//! A `Region` is a **geometric predicate** with a signed-distance function:
//! - `contains(x) == true`  ⇔ x is inside the region
//! - `signed_distance(x) < 0` inside, `> 0` outside, `== 0` on the boundary
//!
//! Regions compose into boolean combinations via `And` / `Or` / `Not`
//! (pure type algebra — zero runtime cost beyond the component evaluations).
//! Any `Region` lifts to a soft-penalty `Restraint` via [`FromRegion`]:
//! `penalty(x) = scale2 * max(0, signed_distance(x))²`.
//!
//! # Example
//! ```
//! use molrs_pack::region::{InsideSphereRegion, Not, Region, RegionExt};
//! use molrs_pack::FromRegion;
//!
//! // Spherical shell: inside outer sphere AND NOT inside inner sphere
//! let shell = InsideSphereRegion::new([0.0; 3], 10.0)
//!     .and(Not(InsideSphereRegion::new([0.0; 3], 5.0)));
//! assert!(shell.contains(&[7.0, 0.0, 0.0]));   // in the shell
//! assert!(!shell.contains(&[3.0, 0.0, 0.0]));  // inside inner sphere
//! assert!(!shell.contains(&[15.0, 0.0, 0.0])); // outside outer sphere
//!
//! // Lift to a Restraint for packing
//! let restraint = FromRegion(shell);
//! ```
//!
//! Direction-3 rule (spec §0 bullet 9): `Region` is a **separate trait**
//! from `Restraint`; composition operators (`.and()` etc.) live on `Region`,
//! not on `Restraint`. Plugin vs built-in `Region` are type-equal via
//! user `impl Region`.

use molrs::types::F;

use crate::restraint::Restraint;

// ============================================================================
// Core trait
// ============================================================================

/// Axis-aligned bounding box, used as an optimization hint for cell-list
/// setup. Optional (default `None`).
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub min: [F; 3],
    pub max: [F; 3],
}

/// Geometric predicate with signed-distance function.
pub trait Region: Send + Sync + std::fmt::Debug {
    /// Membership test. Must be consistent with `signed_distance(x) <= 0`.
    fn contains(&self, x: &[F; 3]) -> bool;

    /// Signed distance to the region boundary.
    /// - Negative inside, positive outside, zero on the boundary.
    /// - Not required to be the *Euclidean* signed distance for arbitrary
    ///   regions; only the sign and gradient direction matter for packing.
    fn signed_distance(&self, x: &[F; 3]) -> F;

    /// Gradient of `signed_distance` at `x`.
    ///
    /// Default implementation is a 3-point central finite difference
    /// (ε = 1e-6). Concrete `Region` types should override with an
    /// analytic gradient for hot-path use.
    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
        let h: F = 1e-6;
        let mut g = [0.0; 3];
        for k in 0..3 {
            let mut xp = *x;
            xp[k] += h;
            let mut xm = *x;
            xm[k] -= h;
            g[k] = (self.signed_distance(&xp) - self.signed_distance(&xm)) / (2.0 * h);
        }
        g
    }

    /// Axis-aligned bounding box of the region. Used as an initialization
    /// hint; default `None` is always safe.
    fn bounding_box(&self) -> Option<BBox> {
        None
    }
}

// ============================================================================
// Combinators
// ============================================================================

/// Intersection of two regions: inside iff BOTH are inside.
/// Signed distance uses `max` (outside dominates — chain-rule selects the
/// larger component's gradient).
#[derive(Debug, Clone, Copy)]
pub struct And<A, B>(pub A, pub B);

impl<A: Region, B: Region> Region for And<A, B> {
    fn contains(&self, x: &[F; 3]) -> bool {
        self.0.contains(x) && self.1.contains(x)
    }
    fn signed_distance(&self, x: &[F; 3]) -> F {
        self.0.signed_distance(x).max(self.1.signed_distance(x))
    }
    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
        if self.0.signed_distance(x) >= self.1.signed_distance(x) {
            self.0.signed_distance_grad(x)
        } else {
            self.1.signed_distance_grad(x)
        }
    }
}

/// Union of two regions: inside iff EITHER is inside.
/// Signed distance uses `min` (inside dominates).
#[derive(Debug, Clone, Copy)]
pub struct Or<A, B>(pub A, pub B);

impl<A: Region, B: Region> Region for Or<A, B> {
    fn contains(&self, x: &[F; 3]) -> bool {
        self.0.contains(x) || self.1.contains(x)
    }
    fn signed_distance(&self, x: &[F; 3]) -> F {
        self.0.signed_distance(x).min(self.1.signed_distance(x))
    }
    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
        if self.0.signed_distance(x) <= self.1.signed_distance(x) {
            self.0.signed_distance_grad(x)
        } else {
            self.1.signed_distance_grad(x)
        }
    }
}

/// Complement: inside iff the inner region is NOT inside.
/// Signed distance is negated.
#[derive(Debug, Clone, Copy)]
pub struct Not<A>(pub A);

impl<A: Region> Region for Not<A> {
    fn contains(&self, x: &[F; 3]) -> bool {
        !self.0.contains(x)
    }
    fn signed_distance(&self, x: &[F; 3]) -> F {
        -self.0.signed_distance(x)
    }
    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
        let g = self.0.signed_distance_grad(x);
        [-g[0], -g[1], -g[2]]
    }
}

// ============================================================================
// Extension trait for `.and() / .or() / .not()` chaining
// ============================================================================

/// Ergonomic method-chaining for `Region`.
///
/// ```
/// use molrs_pack::region::{InsideSphereRegion, RegionExt};
/// let outer = InsideSphereRegion::new([0.0; 3], 10.0);
/// let inner = InsideSphereRegion::new([0.0; 3], 5.0);
/// let shell = outer.and(inner.not());
/// # let _ = shell;
/// ```
pub trait RegionExt: Region + Sized {
    fn and<B: Region>(self, other: B) -> And<Self, B> {
        And(self, other)
    }
    fn or<B: Region>(self, other: B) -> Or<Self, B> {
        Or(self, other)
    }
    fn not(self) -> Not<Self> {
        Not(self)
    }
}

impl<R: Region + Sized> RegionExt for R {}

// ============================================================================
// FromRegion — lift any Region to a quadratic-exterior-penalty Restraint
// ============================================================================

/// Wraps a `Region` as a soft-penalty `Restraint` with quadratic
/// exterior penalty:
///
/// ```text
/// penalty(x) = scale2 * max(0, signed_distance(x))²
/// ```
///
/// Gradient uses the analytic chain rule
/// `2 * scale2 * max(0, d) * ∂d/∂x`, where `∂d/∂x` comes from
/// `Region::signed_distance_grad`.
///
/// # Example
/// ```
/// use molrs_pack::region::{InsideSphereRegion, RegionExt};
/// use molrs_pack::{FromRegion, Restraint};
///
/// let shell = InsideSphereRegion::new([0.0; 3], 10.0)
///     .and(InsideSphereRegion::new([0.0; 3], 5.0).not());
/// let restraint = FromRegion(shell);
/// // At x=(7,0,0) the shell is satisfied → f == 0
/// assert_eq!(restraint.f(&[7.0, 0.0, 0.0], 1.0, 1.0), 0.0);
/// // At x=(15,0,0) we are outside the outer sphere → f > 0
/// assert!(restraint.f(&[15.0, 0.0, 0.0], 1.0, 1.0) > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FromRegion<R: Region>(pub R);

impl<R: Region + 'static> Restraint for FromRegion<R> {
    fn f(&self, x: &[F; 3], _scale: F, scale2: F) -> F {
        let d = self.0.signed_distance(x);
        let v = d.max(0.0);
        scale2 * v * v
    }

    fn fg(&self, x: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let d = self.0.signed_distance(x);
        if d > 0.0 {
            let grad = self.0.signed_distance_grad(x);
            let coeff = 2.0 * scale2 * d;
            g[0] += coeff * grad[0];
            g[1] += coeff * grad[1];
            g[2] += coeff * grad[2];
        }
        self.f(x, scale, scale2)
    }
}

// ============================================================================
// Concrete regions (starting set — more can be added incrementally)
// ============================================================================

/// Axis-aligned box region (inside test).
#[derive(Debug, Clone, Copy)]
pub struct InsideBoxRegion {
    pub min: [F; 3],
    pub max: [F; 3],
}

impl InsideBoxRegion {
    pub fn new(min: [F; 3], max: [F; 3]) -> Self {
        Self { min, max }
    }
}

impl Region for InsideBoxRegion {
    fn contains(&self, x: &[F; 3]) -> bool {
        (0..3).all(|k| x[k] >= self.min[k] && x[k] <= self.max[k])
    }

    fn signed_distance(&self, x: &[F; 3]) -> F {
        // Signed distance to an axis-aligned box:
        // d = max_k max(min_k - x_k, x_k - max_k)
        // Negative inside (both terms negative), positive outside.
        let mut d = F::NEG_INFINITY;
        for ((xk, &lo_k), &hi_k) in x.iter().zip(self.min.iter()).zip(self.max.iter()) {
            d = d.max(lo_k - xk).max(xk - hi_k);
        }
        d
    }

    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
        // The max is attained on one face; gradient points outward from that face.
        let mut best_d = F::NEG_INFINITY;
        let mut best_axis = 0usize;
        let mut best_sign = 0.0 as F;
        for (k, ((xk, &lo_k), &hi_k)) in x
            .iter()
            .zip(self.min.iter())
            .zip(self.max.iter())
            .enumerate()
        {
            let lo = lo_k - xk; // gradient component -1 on axis k
            let hi = xk - hi_k; // gradient component +1 on axis k
            if lo > best_d {
                best_d = lo;
                best_axis = k;
                best_sign = -1.0;
            }
            if hi > best_d {
                best_d = hi;
                best_axis = k;
                best_sign = 1.0;
            }
        }
        let mut g = [0.0; 3];
        g[best_axis] = best_sign;
        g
    }

    fn bounding_box(&self) -> Option<BBox> {
        Some(BBox {
            min: self.min,
            max: self.max,
        })
    }
}

/// Spherical region (inside test).
#[derive(Debug, Clone, Copy)]
pub struct InsideSphereRegion {
    pub center: [F; 3],
    pub radius: F,
}

impl InsideSphereRegion {
    pub fn new(center: [F; 3], radius: F) -> Self {
        Self { center, radius }
    }
}

impl Region for InsideSphereRegion {
    fn contains(&self, x: &[F; 3]) -> bool {
        let c = self.center;
        let d2 = (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2);
        d2 <= self.radius.powi(2)
    }

    fn signed_distance(&self, x: &[F; 3]) -> F {
        let c = self.center;
        let d = ((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2)).sqrt();
        d - self.radius
    }

    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
        let c = self.center;
        let (dx, dy, dz) = (x[0] - c[0], x[1] - c[1], x[2] - c[2]);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d < 1e-12 {
            [0.0; 3]
        } else {
            [dx / d, dy / d, dz / d]
        }
    }

    fn bounding_box(&self) -> Option<BBox> {
        let r = self.radius;
        let c = self.center;
        Some(BBox {
            min: [c[0] - r, c[1] - r, c[2] - r],
            max: [c[0] + r, c[1] + r, c[2] + r],
        })
    }
}

/// Outside-sphere region: `!InsideSphereRegion` with a bespoke impl so
/// `signed_distance` avoids the `Not(...)` double negation at call time.
#[derive(Debug, Clone, Copy)]
pub struct OutsideSphereRegion {
    pub center: [F; 3],
    pub radius: F,
}

impl OutsideSphereRegion {
    pub fn new(center: [F; 3], radius: F) -> Self {
        Self { center, radius }
    }
}

impl Region for OutsideSphereRegion {
    fn contains(&self, x: &[F; 3]) -> bool {
        let c = self.center;
        let d2 = (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2);
        d2 >= self.radius.powi(2)
    }

    fn signed_distance(&self, x: &[F; 3]) -> F {
        let c = self.center;
        let d = ((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2)).sqrt();
        self.radius - d
    }

    fn signed_distance_grad(&self, x: &[F; 3]) -> [F; 3] {
        let c = self.center;
        let (dx, dy, dz) = (x[0] - c[0], x[1] - c[1], x[2] - c[2]);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d < 1e-12 {
            [0.0; 3]
        } else {
            [-dx / d, -dy / d, -dz / d]
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: F = 1e-6;

    // ── boolean algebra laws ────────────────────────────────────────────────

    #[test]
    fn and_is_intersection() {
        let a = InsideBoxRegion::new([0.0; 3], [10.0; 3]);
        let b = InsideSphereRegion::new([5.0; 3], 4.0);
        let c = And(a, b);
        // Inside both
        assert!(c.contains(&[5.0, 5.0, 5.0]));
        // Inside box only
        assert!(!c.contains(&[1.0, 1.0, 1.0]));
        // Inside sphere only — impossible here since sphere ⊂ box
        // Outside both
        assert!(!c.contains(&[-1.0, -1.0, -1.0]));
    }

    #[test]
    fn or_is_union() {
        let a = InsideSphereRegion::new([0.0; 3], 5.0);
        let b = InsideSphereRegion::new([20.0, 0.0, 0.0], 5.0);
        let u = Or(a, b);
        assert!(u.contains(&[0.0, 0.0, 0.0]));
        assert!(u.contains(&[20.0, 0.0, 0.0]));
        assert!(!u.contains(&[10.0, 0.0, 0.0]));
    }

    #[test]
    fn not_is_complement() {
        let a = InsideSphereRegion::new([0.0; 3], 5.0);
        let n = Not(a);
        assert!(!n.contains(&[0.0, 0.0, 0.0]));
        assert!(n.contains(&[10.0, 0.0, 0.0]));
    }

    #[test]
    fn de_morgan_and() {
        // !(A ∧ B) == !A ∨ !B
        let a = InsideSphereRegion::new([0.0; 3], 5.0);
        let b = InsideBoxRegion::new([-3.0; 3], [3.0; 3]);
        let lhs = Not(And(a, b));
        let rhs = Or(Not(a), Not(b));
        for pt in &[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ] {
            assert_eq!(
                lhs.contains(pt),
                rhs.contains(pt),
                "de Morgan mismatch at {pt:?}"
            );
        }
    }

    // ── signed_distance sign correctness ─────────────────────────────────────

    #[test]
    fn sphere_signed_distance_sign() {
        let s = InsideSphereRegion::new([0.0; 3], 5.0);
        assert!(s.signed_distance(&[0.0, 0.0, 0.0]) < 0.0);
        assert!((s.signed_distance(&[5.0, 0.0, 0.0])).abs() < TOL);
        assert!(s.signed_distance(&[10.0, 0.0, 0.0]) > 0.0);
    }

    #[test]
    fn box_signed_distance_sign() {
        let b = InsideBoxRegion::new([0.0; 3], [10.0; 3]);
        assert!(b.signed_distance(&[5.0, 5.0, 5.0]) < 0.0);
        assert!(b.signed_distance(&[15.0, 5.0, 5.0]) > 0.0);
        assert!(b.signed_distance(&[-5.0, 5.0, 5.0]) > 0.0);
    }

    #[test]
    fn contains_matches_signed_distance() {
        let regions: Vec<Box<dyn Region>> = vec![
            Box::new(InsideSphereRegion::new([0.0; 3], 3.0)),
            Box::new(InsideBoxRegion::new([0.0; 3], [5.0; 3])),
            Box::new(OutsideSphereRegion::new([0.0; 3], 2.0)),
        ];
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [-1.0, 0.0, 0.0],
        ];
        for r in &regions {
            for pt in &pts {
                let sd = r.signed_distance(pt);
                // contains ⇔ sd <= 0 (within tolerance for boundary)
                if sd < -TOL {
                    assert!(r.contains(pt), "sd={sd}<0 but !contains at {pt:?}");
                }
                if sd > TOL {
                    assert!(!r.contains(pt), "sd={sd}>0 but contains at {pt:?}");
                }
            }
        }
    }

    // ── FromRegion lifts to Restraint ────────────────────────────────────────

    #[test]
    fn from_region_penalty_zero_inside() {
        let r = FromRegion(InsideSphereRegion::new([0.0; 3], 5.0));
        assert_eq!(r.f(&[0.0, 0.0, 0.0], 1.0, 1.0), 0.0);
    }

    #[test]
    fn from_region_penalty_positive_outside() {
        let r = FromRegion(InsideSphereRegion::new([0.0; 3], 5.0));
        assert!(r.f(&[10.0, 0.0, 0.0], 1.0, 1.0) > 0.0);
    }

    #[test]
    fn from_region_gradient_matches_finite_diff() {
        let r = FromRegion(
            InsideBoxRegion::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
                .and(Not(InsideSphereRegion::new([5.0, 5.0, 5.0], 2.0))),
        );
        // Point outside the box in x → composite region reports "outside" via box face
        let x = [15.0, 5.0, 5.0];
        let mut g = [0.0; 3];
        let _ = r.fg(&x, 1.0, 1.0, &mut g);

        // central finite difference
        let h: F = 1e-5;
        for k in 0..3 {
            let mut xp = x;
            xp[k] += h;
            let mut xm = x;
            xm[k] -= h;
            let fd = (r.f(&xp, 1.0, 1.0) - r.f(&xm, 1.0, 1.0)) / (2.0 * h);
            assert!(
                (g[k] - fd).abs() < 1e-3,
                "gradient mismatch axis {k}: analytic={} fd={} err={}",
                g[k],
                fd,
                (g[k] - fd).abs()
            );
        }
    }

    #[test]
    fn region_ext_chain() {
        // RegionExt provides ergonomic .and() / .not()
        let shell = InsideSphereRegion::new([0.0; 3], 10.0)
            .and(InsideSphereRegion::new([0.0; 3], 5.0).not());
        assert!(shell.contains(&[7.0, 0.0, 0.0]));
        assert!(!shell.contains(&[3.0, 0.0, 0.0]));
        assert!(!shell.contains(&[15.0, 0.0, 0.0]));
    }

    #[test]
    fn gradient_accumulates_not_overwrite() {
        let r = FromRegion(InsideSphereRegion::new([0.0; 3], 1.0));
        let mut g = [100.0; 3];
        let _ = r.fg(&[2.0, 0.0, 0.0], 1.0, 1.0, &mut g);
        assert!(g[0] > 100.0, "should accumulate");
        assert!((g[1] - 100.0).abs() < TOL);
        assert!((g[2] - 100.0).abs() < TOL);
    }
}
