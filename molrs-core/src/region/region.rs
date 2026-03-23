//! Geometric regions and spatial predicates.
//!
//! This module provides a lightweight trait [`Region`] for geometric regions and a
//! concrete [`Sphere`] implementation. It also defines array-based type aliases
//! (points as an N×3 NdArray, bounds as a 3×2 NdArray, etc.).
//!
//! Type layout conventions:
//! - [`FNx3`]: N×3 row-major array, each row is a point (x, y, z).
//! - [`FNx3`] (3×2): 3×2 array where column 0 is the min corner, column 1 is the max corner;
//!   rows correspond to x/y/z respectively.

use crate::types::{F, F3, FNx3};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Axis-aligned bounding box (AABB) as a 3×2 matrix.
///
/// Column 0 is the minimum corner, column 1 is the maximum corner.
/// Rows correspond to x, y, z respectively:
///
/// [ [min_x, max_x],
///   [min_y, max_y],
///   [min_z, max_z] ]
///
/// Region trait for geometric queries.
pub trait Region: Send + Sync {
    /// Returns the axis-aligned bounding box of the region.
    ///
    /// Layout: rows = x/y/z; col 0 = min, col 1 = max.
    fn bounds(&self) -> FNx3;

    /// Batched containment test for a set of 3D points.
    ///
    /// Returns a boolean NdArray of shape [N] where each entry indicates whether
    /// the corresponding row in the N×3 array lies inside the region.
    ///
    /// Panics
    /// - If `points` does not have exactly 3 columns.
    fn contains(&self, points: &FNx3) -> Array1<bool>;

    /// Single-point containment test.
    ///
    /// Returns true if the point at `[x, y, z]` lies inside the region.
    /// This is more efficient than `contains()` for single-point checks as it
    /// avoids array allocations.
    ///
    /// Default implementation delegates to `contains()`.
    /// Implementations should override with optimized versions.
    fn contains_point(&self, point: &[F; 3]) -> bool {
        let arr = Array2::from_shape_vec((1, 3), vec![point[0], point[1], point[2]]).unwrap();
        self.contains(&arr)[0]
    }
}

/// A solid sphere region.
#[derive(Debug, Clone)]
pub struct Sphere {
    /// Center of the sphere.
    pub center: F3,
    /// Radius of the sphere.
    pub radius: F,
}

impl Sphere {
    /// Creates a sphere with a given center and radius.
    pub fn new(center: F3, radius: F) -> Self {
        Self { center, radius }
    }

    /// Creates a sphere centered at the origin with the given radius.
    pub fn with_radius(radius: F) -> Self {
        Self {
            center: Array1::zeros(3),
            radius,
        }
    }
}

impl Region for Sphere {
    fn bounds(&self) -> FNx3 {
        let r = self.radius;
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            b[[d, 0]] = self.center[d] - r;
            b[[d, 1]] = self.center[d] + r;
        }
        b
    }

    fn contains(&self, points: &FNx3) -> Array1<bool> {
        assert_eq!(points.ncols(), 3, "points must have shape (N, 3)");
        let r2 = self.radius * self.radius;
        let mut mask = Array1::from_elem(points.nrows(), false);
        for (row, m) in points.rows().into_iter().zip(mask.iter_mut()) {
            let dx = row[0] - self.center[0];
            let dy = row[1] - self.center[1];
            let dz = row[2] - self.center[2];
            *m = (dx * dx + dy * dy + dz * dz) <= r2;
        }
        mask
    }

    fn contains_point(&self, point: &[F; 3]) -> bool {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz) <= self.radius * self.radius
    }
}

/// A hollow sphere (spherical shell) region.
///
/// This region represents the space between two concentric spheres:
/// points inside the outer sphere but outside the inner sphere.
#[derive(Debug, Clone)]
pub struct HollowSphere {
    /// Center of the spheres.
    pub center: F3,
    /// Outer radius (points must be within this distance from center).
    pub outer_radius: F,
    /// Inner radius (points must be beyond this distance from center).
    pub inner_radius: F,
}

impl HollowSphere {
    /// Creates a hollow sphere with given center and radii.
    ///
    /// # Panics
    /// - If `inner_radius >= outer_radius`
    /// - If `inner_radius < 0` or `outer_radius <= 0`
    pub fn new(center: F3, inner_radius: F, outer_radius: F) -> Self {
        assert!(
            inner_radius >= 0.0,
            "inner_radius must be non-negative, got {}",
            inner_radius
        );
        assert!(
            outer_radius > inner_radius,
            "outer_radius must be greater than inner_radius, got outer={}, inner={}",
            outer_radius,
            inner_radius
        );
        Self {
            center,
            outer_radius,
            inner_radius,
        }
    }

    /// Creates a hollow sphere centered at the origin.
    pub fn with_radii(inner_radius: F, outer_radius: F) -> Self {
        Self::new(Array1::zeros(3), inner_radius, outer_radius)
    }
}

impl Region for HollowSphere {
    fn bounds(&self) -> FNx3 {
        // Bounds are the same as the outer sphere
        let r = self.outer_radius;
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            b[[d, 0]] = self.center[d] - r;
            b[[d, 1]] = self.center[d] + r;
        }
        b
    }

    fn contains(&self, points: &FNx3) -> Array1<bool> {
        assert_eq!(points.ncols(), 3, "points must have shape (N, 3)");
        let outer_r2 = self.outer_radius * self.outer_radius;
        let inner_r2 = self.inner_radius * self.inner_radius;
        let mut mask = Array1::from_elem(points.nrows(), false);
        for (row, m) in points.rows().into_iter().zip(mask.iter_mut()) {
            let dx = row[0] - self.center[0];
            let dy = row[1] - self.center[1];
            let dz = row[2] - self.center[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            // Point is inside if: inner_r^2 < dist^2 <= outer_r^2
            *m = dist_sq > inner_r2 && dist_sq <= outer_r2;
        }
        mask
    }

    fn contains_point(&self, point: &[F; 3]) -> bool {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let outer_r2 = self.outer_radius * self.outer_radius;
        let inner_r2 = self.inner_radius * self.inner_radius;
        dist_sq > inner_r2 && dist_sq <= outer_r2
    }
}

/// Intersection of two regions (AND operation).
///
/// A point is inside the intersection if it is inside both regions.
#[derive(Clone)]
pub struct AndRegion {
    a: Arc<dyn Region + Send + Sync>,
    b: Arc<dyn Region + Send + Sync>,
}

impl AndRegion {
    /// Creates an intersection of two regions.
    pub fn new(a: Arc<dyn Region + Send + Sync>, b: Arc<dyn Region + Send + Sync>) -> Self {
        Self { a, b }
    }
}

impl Region for AndRegion {
    fn bounds(&self) -> FNx3 {
        // Intersection bounds: max of mins, min of maxs
        let a_bounds = self.a.bounds();
        let b_bounds = self.b.bounds();
        let mut result = Array2::zeros((3, 2));
        for d in 0..3 {
            result[[d, 0]] = a_bounds[[d, 0]].max(b_bounds[[d, 0]]); // max of mins
            result[[d, 1]] = a_bounds[[d, 1]].min(b_bounds[[d, 1]]); // min of maxs
        }
        result
    }

    fn contains(&self, points: &FNx3) -> Array1<bool> {
        let a_mask = self.a.contains(points);
        let b_mask = self.b.contains(points);
        // Point is inside if it's inside both regions
        a_mask
            .iter()
            .zip(b_mask.iter())
            .map(|(a, b)| *a && *b)
            .collect()
    }

    fn contains_point(&self, point: &[F; 3]) -> bool {
        self.a.contains_point(point) && self.b.contains_point(point)
    }
}

/// Complement of a region (NOT operation).
///
/// A point is inside the complement if it is NOT inside the original region.
#[derive(Clone)]
pub struct NotRegion {
    a: Arc<dyn Region + Send + Sync>,
}

impl NotRegion {
    /// Creates a complement of a region.
    pub fn new(a: Arc<dyn Region + Send + Sync>) -> Self {
        Self { a }
    }
}

impl Region for NotRegion {
    fn bounds(&self) -> FNx3 {
        // Complement is unbounded, but we return the original bounds for practicality
        // (the actual constraint will be enforced by contains())
        self.a.bounds()
    }

    fn contains(&self, points: &FNx3) -> Array1<bool> {
        let a_mask = self.a.contains(points);
        // Point is inside if it's NOT inside the original region
        a_mask.iter().map(|x| !x).collect()
    }

    fn contains_point(&self, point: &[F; 3]) -> bool {
        !self.a.contains_point(point)
    }
}

/// Union of two regions (OR operation).
///
/// A point is inside the union if it is inside either region.
#[derive(Clone)]
pub struct OrRegion {
    a: Arc<dyn Region + Send + Sync>,
    b: Arc<dyn Region + Send + Sync>,
}

impl OrRegion {
    /// Creates a union of two regions.
    pub fn new(a: Arc<dyn Region + Send + Sync>, b: Arc<dyn Region + Send + Sync>) -> Self {
        Self { a, b }
    }
}

impl Region for OrRegion {
    fn bounds(&self) -> FNx3 {
        // Union bounds: min of mins, max of maxs
        let a_bounds = self.a.bounds();
        let b_bounds = self.b.bounds();
        let mut result = Array2::zeros((3, 2));
        for d in 0..3 {
            result[[d, 0]] = a_bounds[[d, 0]].min(b_bounds[[d, 0]]); // min of mins
            result[[d, 1]] = a_bounds[[d, 1]].max(b_bounds[[d, 1]]); // max of maxs
        }
        result
    }

    fn contains(&self, points: &FNx3) -> Array1<bool> {
        let a_mask = self.a.contains(points);
        let b_mask = self.b.contains(points);
        // Point is inside if it's inside either region
        a_mask
            .iter()
            .zip(b_mask.iter())
            .map(|(a, b)| *a || *b)
            .collect()
    }

    fn contains_point(&self, point: &[F; 3]) -> bool {
        self.a.contains_point(point) || self.b.contains_point(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_bounds_are_correct() {
        let s = Sphere::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), 2.0);
        let b = s.bounds();
        // Row-major [ [min_x,max_x], [min_y,max_y], [min_z,max_z] ]
        assert_eq!(b[[0, 0]], -1.0);
        assert_eq!(b[[1, 0]], 0.0);
        assert_eq!(b[[2, 0]], 1.0);
        assert_eq!(b[[0, 1]], 3.0);
        assert_eq!(b[[1, 1]], 4.0);
        assert_eq!(b[[2, 1]], 5.0);
    }

    #[test]
    fn sphere_contains_points() {
        let s = Sphere::with_radius(2.0);
        let pts: FNx3 = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, 0.0, 0.0, // inside (center)
                2.0, 0.0, 0.0, // on surface
                2.1, 0.0, 0.0, // outside
            ],
        )
        .unwrap();
        let mask = s.contains(&pts);
        assert_eq!(mask.len(), 3);
        assert!(mask[0]);
        assert!(mask[1]);
        assert!(!mask[2]);
    }

    #[test]
    fn hollow_sphere_bounds_are_correct() {
        let hs = HollowSphere::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), 2.0, 5.0);
        let b = hs.bounds();
        // Bounds should match outer sphere
        assert_eq!(b[[0, 0]], -4.0); // 1.0 - 5.0
        assert_eq!(b[[1, 0]], -3.0); // 2.0 - 5.0
        assert_eq!(b[[2, 0]], -2.0); // 3.0 - 5.0
        assert_eq!(b[[0, 1]], 6.0); // 1.0 + 5.0
        assert_eq!(b[[1, 1]], 7.0); // 2.0 + 5.0
        assert_eq!(b[[2, 1]], 8.0); // 3.0 + 5.0
    }

    #[test]
    fn hollow_sphere_contains_points() {
        let hs = HollowSphere::with_radii(2.0, 5.0);
        let pts: FNx3 = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 0.0, 0.0, // inside inner sphere (should be false)
                1.0, 0.0, 0.0, // inside inner sphere (should be false)
                3.0, 0.0, 0.0, // in shell (should be true)
                5.0, 0.0, 0.0, // on outer surface (should be true)
                5.1, 0.0, 0.0, // outside outer sphere (should be false)
            ],
        )
        .unwrap();
        let mask = hs.contains(&pts);
        assert_eq!(mask.len(), 5);
        assert!(!mask[0], "center should be outside (inside inner sphere)");
        assert!(!mask[1], "point inside inner sphere should be false");
        assert!(mask[2], "point in shell should be true");
        assert!(mask[3], "point on outer surface should be true");
        assert!(!mask[4], "point outside outer sphere should be false");
    }
}
