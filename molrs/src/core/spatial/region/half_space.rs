//! The half-space behind a plane.

use super::region::Region;
use crate::types::{F, FNx3};
use ndarray::Array2;

/// Everything on one side of a plane: inside where `n · (x − p) ≤ 0`, i.e.
/// the side the unit normal `n` points *away* from. The other side is
/// `~HalfSpace`.
///
/// `distance(x) = n · x − n · p` is the exact Euclidean distance to the plane
/// and the gradient is `n` everywhere.
#[derive(Debug, Clone)]
pub struct HalfSpace {
    normal: [F; 3],
    offset: F,
}

impl HalfSpace {
    /// The half-space behind the plane through `point` with outward normal
    /// `normal` (Å; any non-zero length, normalised here).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `normal` is zero or not finite.
    pub fn new(normal: [F; 3], point: [F; 3]) -> Result<Self, String> {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if !(len > 0.0 && len.is_finite()) {
            return Err("HalfSpace: normal must be a finite non-zero vector".to_string());
        }
        let n = [normal[0] / len, normal[1] / len, normal[2] / len];
        let offset = n[0] * point[0] + n[1] * point[1] + n[2] * point[2];
        if !offset.is_finite() {
            return Err("HalfSpace: point must be finite".to_string());
        }
        Ok(Self { normal: n, offset })
    }

    /// Unit outward normal.
    pub fn normal(&self) -> [F; 3] {
        self.normal
    }

    /// `n · p` for any point `p` on the plane, Å.
    pub fn offset(&self) -> F {
        self.offset
    }
}

impl Region for HalfSpace {
    /// Unbounded: `±∞` on every axis, except that an axis-aligned normal
    /// closes its own axis at the plane.
    fn bounds(&self) -> FNx3 {
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            b[[d, 0]] = F::NEG_INFINITY;
            b[[d, 1]] = F::INFINITY;
            if self.normal[d] == 1.0 {
                b[[d, 1]] = self.offset;
            } else if self.normal[d] == -1.0 {
                b[[d, 0]] = -self.offset;
            }
        }
        b
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2]
            - self.offset
    }

    fn distance_grad(&self, _point: &[F; 3]) -> [F; 3] {
        self.normal
    }
}

#[cfg(test)]
mod tests {
    use super::super::region::tests::check_region_contract;
    use super::*;

    #[test]
    fn below_the_plane_is_inside() {
        let h = HalfSpace::new([0.0, 0.0, 2.0], [0.0, 0.0, 5.0]).unwrap();
        assert_eq!(h.normal(), [0.0, 0.0, 1.0]);
        assert_eq!(h.distance(&[3.0, -1.0, 5.0]), 0.0);
        assert_eq!(h.distance(&[3.0, -1.0, 2.0]), -3.0);
        assert_eq!(h.distance(&[3.0, -1.0, 9.0]), 4.0);
        assert!(h.contains_point(&[0.0, 0.0, 4.9]));
        assert!(!h.contains_point(&[0.0, 0.0, 5.1]));
        let b = h.bounds();
        assert_eq!(b[[2, 1]], 5.0);
        assert_eq!(b[[2, 0]], F::NEG_INFINITY);
        assert_eq!(b[[0, 1]], F::INFINITY);
    }

    #[test]
    fn oblique_plane_distance_is_euclidean() {
        let h = HalfSpace::new([1.0, 1.0, 0.0], [1.0, 1.0, 0.0]).unwrap();
        // The plane x + y = 2; (3, 3) is 2·√2 / ... = (6 − 2)/√2 away.
        let d = h.distance(&[3.0, 3.0, 0.0]);
        assert!((d - 4.0 / 2.0_f64.sqrt()).abs() < 1e-12);
        let probes: Vec<[F; 3]> = (0..12)
            .map(|i| [i as F * 0.4 - 1.0, 2.0 - i as F * 0.3, 0.5])
            .collect();
        check_region_contract(&h, &probes);
    }

    #[test]
    fn rejects_zero_normal() {
        assert!(HalfSpace::new([0.0; 3], [0.0; 3]).is_err());
        assert!(HalfSpace::new([F::NAN, 0.0, 0.0], [0.0; 3]).is_err());
    }
}
