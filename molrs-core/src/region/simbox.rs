//! Triclinic simulation box and periodic operations based on ndarray.
//!
//! Conventions (fractional/cartesian):
//! - cart = origin + H * frac
//! - frac = H^{-1} * (cart - origin)
//! - Lattice vectors are the columns of H.

use super::region::Region;
use crate::math;
use crate::types::{F, F3, F3View, F3x3, FNx3, FNx3View, Pbc3};
use ndarray::{Array1, Array2, ArrayView1, array};

/// Box geometry kind, detected once at construction.
#[derive(Debug, Clone, PartialEq)]
pub enum BoxKind {
    /// Orthorhombic (diagonal H): lengths, inverse lengths cached.
    Ortho { len: F3, inv_len: F3 },
    /// General triclinic.
    Triclinic,
}

/// Simulation box: triclinic cell with origin and per-axis PBC mask
#[derive(Debug, Clone)]
pub struct SimBox {
    /// Triclinic cell matrix H (columns are lattice vectors)
    h: F3x3,
    /// Precomputed inverse of H
    inv: F3x3,
    /// Origin of the cell in Cartesian coordinates
    origin: F3,
    /// Per-axis periodic boundary condition flags (x, y, z)
    pbc: Pbc3,
    /// Cached geometry kind
    kind: BoxKind,
}

/// Error type for simulation box construction.
#[derive(Debug)]
pub enum BoxError {
    /// The cell matrix H is singular (determinant ≈ 0).
    SingularCell,
    /// The matrix does not have shape 3x3.
    InvalidMatrixShape { rows: usize, cols: usize },
    /// A vector does not have the expected length.
    InvalidVectorLength { len: usize },
    /// A required array is not contiguous in memory.
    NonContiguous(&'static str),
}

impl SimBox {
    /// Construct from triclinic cell matrix `H`, origin `O`, and per-axis PBC flags
    pub fn new(h: F3x3, origin: F3, pbc: Pbc3) -> Result<Self, BoxError> {
        if let Some(inv) = math::inv3(&h) {
            let kind = detect_box_kind(&h);
            Ok(Self {
                h,
                inv,
                origin,
                pbc,
                kind,
            })
        } else {
            Err(BoxError::SingularCell)
        }
    }

    pub fn try_new(h: F3x3, origin: F3, pbc: Pbc3) -> Result<Self, BoxError> {
        Self::new(h, origin, pbc)
    }

    /// Factory: cubic box with edge length `a` and origin `O`
    pub fn cube(a: F, origin: F3, pbc: Pbc3) -> Result<Self, BoxError> {
        if a <= 0.0 {
            return Err(BoxError::InvalidVectorLength { len: 0 });
        }
        let h = array![[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]];
        Self::new(h, origin, pbc)
    }

    /// Factory: ortho box with lengths (ax, ay, az) and origin `O`
    pub fn ortho(lengths: F3, origin: F3, pbc: Pbc3) -> Result<Self, BoxError> {
        if lengths.len() != 3 {
            return Err(BoxError::InvalidVectorLength { len: lengths.len() });
        }
        if lengths.iter().any(|v| *v <= 0.0) {
            return Err(BoxError::InvalidVectorLength { len: 0 });
        }
        let h = array![
            [lengths[0], 0.0, 0.0],
            [0.0, lengths[1], 0.0],
            [0.0, 0.0, lengths[2]],
        ];
        Self::new(h, origin, pbc)
    }

    /// Create a non-periodic (free-boundary) box enclosing all points.
    ///
    /// Computes the axis-aligned bounding box of `points` and adds `padding`
    /// on each side. The resulting box has `pbc = [false, false, false]`.
    ///
    /// `padding` should be >= the neighbor cutoff distance so that all
    /// particles sit well inside the box for correct cell assignment.
    ///
    /// # Errors
    /// Returns `BoxError` if padding is non-positive or the resulting box is degenerate.
    ///
    /// # Panics
    /// Panics if `padding <= 0`.
    pub fn free(points: FNx3View<'_>, padding: F) -> Result<Self, BoxError> {
        assert!(padding > 0.0, "padding must be positive");
        let n = points.nrows();
        if n == 0 {
            // Empty point set -- return a unit cube at origin
            return Self::cube(padding, array![0.0 as F, 0.0, 0.0], [false, false, false]);
        }
        let mut min = array![points[[0, 0]], points[[0, 1]], points[[0, 2]]];
        let mut max = min.clone();
        for i in 1..n {
            for d in 0..3 {
                if points[[i, d]] < min[d] {
                    min[d] = points[[i, d]];
                }
                if points[[i, d]] > max[d] {
                    max[d] = points[[i, d]];
                }
            }
        }
        let origin = array![min[0] - padding, min[1] - padding, min[2] - padding,];
        let lengths = array![
            (max[0] - min[0] + 2.0 * padding).max(padding),
            (max[1] - min[1] + 2.0 * padding).max(padding),
            (max[2] - min[2] + 2.0 * padding).max(padding),
        ];
        Self::ortho(lengths, origin, [false, false, false])
    }

    /// View of the cell matrix
    pub fn h_view(&self) -> FNx3View<'_> {
        self.h.view()
    }

    /// View of the inverse cell matrix
    pub fn inv_view(&self) -> FNx3View<'_> {
        self.inv.view()
    }

    /// View of the origin
    pub fn origin_view(&self) -> F3View<'_> {
        self.origin.view()
    }

    /// View of the PBC flags
    pub fn pbc_view(&self) -> ArrayView1<'_, bool> {
        ArrayView1::from_shape(3, &self.pbc).expect("pbc_view shape")
    }

    /// Per-axis PBC flags
    pub fn pbc(&self) -> Pbc3 {
        self.pbc
    }

    /// Cell volume (|det(H)|)
    pub fn volume(&self) -> F {
        math::det3(&self.h).abs()
    }

    /// Off-diagonal tilts [xy, xz, yz] of the cell matrix
    pub fn tilts(&self) -> F3 {
        array![self.h[[0, 1]], self.h[[0, 2]], self.h[[1, 2]]]
    }

    /// Lattice vector lengths
    pub fn lengths(&self) -> F3 {
        let a = self.lattice(0);
        let b = self.lattice(1);
        let c = self.lattice(2);
        array![math::norm3(&a), math::norm3(&b), math::norm3(&c)]
    }

    /// Nearest plane distance (half the box size along each axis)
    /// For triclinic boxes, this is the perpendicular distance to each face
    pub fn nearest_plane_distance(&self) -> F3 {
        let v = self.volume();
        let a1 = self.lattice(0);
        let a2 = self.lattice(1);
        let a3 = self.lattice(2);

        let c23 = math::cross3(&a2, &a3);
        let c31 = math::cross3(&a3, &a1);
        let c12 = math::cross3(&a1, &a2);

        array![
            v / math::norm3(&c23),
            v / math::norm3(&c31),
            v / math::norm3(&c12)
        ]
    }

    pub fn kind(&self) -> &BoxKind {
        &self.kind
    }

    /// Lattice vector by index (0,1,2) — columns of H
    pub fn lattice(&self, index: usize) -> F3 {
        assert!(index < 3, "lattice index must be 0..2");
        self.h.column(index).to_owned()
    }

    /// Convert Cartesian coordinates to fractional coordinates [0, 1)
    pub fn make_fractional(&self, r: F3View<'_>) -> F3 {
        let dr = &r - &self.origin.view();
        let mut frac = self.inv.dot(&dr);
        for f in frac.iter_mut() {
            *f -= f.floor();
        }
        frac
    }

    /// Fractional coordinates with ortho fast-path
    #[inline(always)]
    pub fn make_fractional_fast(&self, r: F3View<'_>) -> F3 {
        match &self.kind {
            BoxKind::Ortho { inv_len, .. } => {
                let mut frac = array![
                    (r[0] - self.origin[0]) * inv_len[0],
                    (r[1] - self.origin[1]) * inv_len[1],
                    (r[2] - self.origin[2]) * inv_len[2],
                ];
                for f in frac.iter_mut() {
                    *f -= f.floor();
                }
                frac
            }
            BoxKind::Triclinic => self.make_fractional(r),
        }
    }

    /// Fractional coordinates returned as `[F; 3]` (zero-alloc hot path).
    ///
    /// Equivalent to [`make_fractional_fast`](Self::make_fractional_fast) but
    /// avoids the `Array1<F>` heap allocation by returning a stack array.
    /// Use in tight inner loops (neighbor-list cell assignment, etc.).
    #[inline(always)]
    pub fn make_fractional_fast_arr(&self, r: F3View<'_>) -> [F; 3] {
        match &self.kind {
            BoxKind::Ortho { inv_len, .. } => {
                let fx = (r[0] - self.origin[0]) * inv_len[0];
                let fy = (r[1] - self.origin[1]) * inv_len[1];
                let fz = (r[2] - self.origin[2]) * inv_len[2];
                [fx - fx.floor(), fy - fy.floor(), fz - fz.floor()]
            }
            BoxKind::Triclinic => {
                let f = self.make_fractional(r);
                [f[0], f[1], f[2]]
            }
        }
    }

    /// Convert fractional coordinates to Cartesian coordinates
    pub fn make_cartesian(&self, frac: F3View<'_>) -> F3 {
        &self.origin + &self.h.dot(&frac)
    }

    /// Minimum image displacement vector from r1 to r2 (r2 - r1)
    #[inline]
    pub fn shortest_vector(&self, r1: F3View<'_>, r2: F3View<'_>) -> F3 {
        let dr = &r2 - &r1;
        let mut dr_frac = self.inv.dot(&dr);
        for d in 0..3 {
            if self.pbc[d] {
                dr_frac[d] -= dr_frac[d].round();
            }
        }
        self.h.dot(&dr_frac)
    }

    /// Shortest vector with ortho fast-path
    #[inline(always)]
    pub fn shortest_vector_fast(&self, a: F3View<'_>, b: F3View<'_>) -> F3 {
        match &self.kind {
            BoxKind::Ortho { len, inv_len } => {
                let mut dr = array![b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                for d in 0..3 {
                    if self.pbc[d] {
                        dr[d] -= (dr[d] * inv_len[d]).round() * len[d];
                    }
                }
                dr
            }
            BoxKind::Triclinic => self.shortest_vector(a, b),
        }
    }

    /// Shortest vector returned as `[F; 3]` (zero-alloc hot path).
    ///
    /// Equivalent to [`shortest_vector_fast`](Self::shortest_vector_fast) but
    /// avoids the `Array1<F>` heap allocation by returning a stack array.
    /// Use inside neighbor-list pair loops where called O(N·k) times per build.
    #[inline(always)]
    pub fn shortest_vector_fast_arr(&self, a: F3View<'_>, b: F3View<'_>) -> [F; 3] {
        match &self.kind {
            BoxKind::Ortho { len, inv_len } => {
                let mut dr = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                if self.pbc[0] {
                    dr[0] -= (dr[0] * inv_len[0]).round() * len[0];
                }
                if self.pbc[1] {
                    dr[1] -= (dr[1] * inv_len[1]).round() * len[1];
                }
                if self.pbc[2] {
                    dr[2] -= (dr[2] * inv_len[2]).round() * len[2];
                }
                dr
            }
            BoxKind::Triclinic => {
                let v = self.shortest_vector(a, b);
                [v[0], v[1], v[2]]
            }
        }
    }

    /// Shortest vector from raw `[F; 3]` inputs, returning `[F; 3]`.
    ///
    /// Both inputs and output are stack arrays — no `ArrayView` indexing.
    /// Called from neighbor-list inner loops where positions are stored in a
    /// flat `[F; 3]` slab instead of an `Array2<F>`.
    #[inline(always)]
    pub fn shortest_vector_raw(&self, a: [F; 3], b: [F; 3]) -> [F; 3] {
        match &self.kind {
            BoxKind::Ortho { len, inv_len } => {
                let mut dr = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                if self.pbc[0] {
                    dr[0] -= (dr[0] * inv_len[0]).round() * len[0];
                }
                if self.pbc[1] {
                    dr[1] -= (dr[1] * inv_len[1]).round() * len[1];
                }
                if self.pbc[2] {
                    dr[2] -= (dr[2] * inv_len[2]).round() * len[2];
                }
                dr
            }
            BoxKind::Triclinic => {
                let av = ndarray::ArrayView1::from(&a[..]);
                let bv = ndarray::ArrayView1::from(&b[..]);
                let v = self.shortest_vector(av, bv);
                [v[0], v[1], v[2]]
            }
        }
    }

    /// Calculate squared distance using MIC.
    #[inline]
    pub fn calc_distance2(&self, a: F3View<'_>, b: F3View<'_>) -> F {
        let dr = self.shortest_vector(a, b);
        dr.dot(&dr)
    }

    /// Convert Cartesian points to fractional coordinates (N×3)
    pub fn to_frac(&self, xyz: FNx3View<'_>) -> FNx3 {
        let n = xyz.nrows();
        let mut result = FNx3::zeros((n, 3));
        for i in 0..n {
            let dr = &xyz.row(i) - &self.origin.view();
            result.row_mut(i).assign(&self.inv.dot(&dr));
        }
        result
    }

    /// Convert fractional coordinates to Cartesian points (N×3)
    pub fn to_cart(&self, frac: FNx3View<'_>) -> FNx3 {
        let n = frac.nrows();
        let mut result = FNx3::zeros((n, 3));
        for i in 0..n {
            let cart = &self.origin + &self.h.dot(&frac.row(i));
            result.row_mut(i).assign(&cart);
        }
        result
    }

    /// Check if points lie within [0,1) in fractional space.
    pub fn isin(&self, xyz: FNx3View<'_>) -> Array1<bool> {
        let n = xyz.nrows();
        let mut mask = Vec::with_capacity(n);
        for i in 0..n {
            let dr = &xyz.row(i) - &self.origin.view();
            let frac = self.inv.dot(&dr);
            let inside = (0..3).all(|d| frac[d] >= 0.0 && frac[d] < 1.0);
            mask.push(inside);
        }
        Array1::from_vec(mask)
    }

    /// Batched displacement vectors row-wise (N×3).
    /// Writes result into `out` to avoid allocation.
    pub fn delta_out(
        &self,
        xyzu1: FNx3View<'_>,
        xyzu2: FNx3View<'_>,
        out: &mut FNx3,
        minimum_image: bool,
    ) {
        assert_eq!(xyzu1.nrows(), xyzu2.nrows());
        let n = xyzu1.nrows();
        if minimum_image {
            for i in 0..n {
                let dr = self.shortest_vector(xyzu1.row(i), xyzu2.row(i));
                out.row_mut(i).assign(&dr);
            }
        } else {
            for i in 0..n {
                let dr = &xyzu2.row(i) - &xyzu1.row(i);
                out.row_mut(i).assign(&dr);
            }
        }
    }

    /// Batched displacement vectors row-wise (N×3)
    pub fn delta(&self, xyzu1: FNx3View<'_>, xyzu2: FNx3View<'_>, minimum_image: bool) -> FNx3 {
        assert_eq!(xyzu1.nrows(), xyzu2.nrows());
        let n = xyzu1.nrows();
        let mut out = FNx3::zeros((n, 3));
        self.delta_out(xyzu1, xyzu2, &mut out, minimum_image);
        out
    }

    /// Wrap Cartesian points into the unit cell according to PBC
    pub fn wrap(&self, xyz: FNx3View<'_>) -> FNx3 {
        let mut frac = self.to_frac(xyz);
        let n = frac.nrows();
        for i in 0..n {
            for d in 0..3 {
                if self.pbc[d] {
                    frac[[i, d]] -= frac[[i, d]].floor();
                }
            }
        }
        self.to_cart(frac.view())
    }

    pub fn get_corners(&self) -> FNx3 {
        let l = self.lengths();
        let (ox, oy, oz) = (self.origin[0], self.origin[1], self.origin[2]);
        let (lx, ly, lz) = (l[0], l[1], l[2]);
        array![
            [ox, oy, oz],
            [ox + lx, oy, oz],
            [ox + lx, oy + ly, oz],
            [ox, oy + ly, oz],
            [ox, oy, oz + lz],
            [ox + lx, oy, oz + lz],
            [ox + lx, oy + ly, oz + lz],
            [ox, oy + ly, oz + lz],
        ]
    }
}

impl Region for SimBox {
    fn bounds(&self) -> FNx3 {
        let lengths = self.lengths();
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            b[[d, 0]] = self.origin[d];
            b[[d, 1]] = self.origin[d] + lengths[d];
        }
        b
    }

    fn contains(&self, points: &FNx3) -> Array1<bool> {
        self.isin(points.view())
    }

    fn contains_point(&self, point: &[F; 3]) -> bool {
        let r = ArrayView1::from_shape(3, point).expect("contains_point shape");
        let dr = &r - &self.origin.view();
        let frac = self.inv.dot(&dr);
        (0..3).all(|d| frac[d] >= 0.0 && frac[d] < 1.0)
    }
}

fn detect_box_kind(h: &F3x3) -> BoxKind {
    let eps: F = 1e-12;
    let is_ortho = h[[0, 1]].abs() < eps
        && h[[0, 2]].abs() < eps
        && h[[1, 0]].abs() < eps
        && h[[1, 2]].abs() < eps
        && h[[2, 0]].abs() < eps
        && h[[2, 1]].abs() < eps;
    if is_ortho {
        let len = array![h[[0, 0]], h[[1, 1]], h[[2, 2]]];
        let inv_len = array![1.0 / len[0], 1.0 / len[1], 1.0 / len[2]];
        BoxKind::Ortho { len, inv_len }
    } else {
        BoxKind::Triclinic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: F, b: F) {
        assert!((a - b).abs() < 1e-6 as F, "{} != {}", a, b);
    }

    #[test]
    fn roundtrip_frac_cart() {
        let bx = SimBox::ortho(
            array![2.0, 3.0, 4.0],
            array![0.5, -1.0, 2.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        let pts = array![[0.5, -1.0, 2.0], [2.5, 2.0, 6.0]];
        let frac = bx.to_frac(pts.view());
        let cart = bx.to_cart(frac.view());
        assert!((&pts - &cart).iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn wrap_into_cell() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[2.1, -0.1, 3.9], [-1.9, 4.2, 0.0]];
        let wrapped = bx.wrap(pts.view());
        let frac = bx.to_frac(wrapped.view());
        for i in 0..wrapped.nrows() {
            let fx = frac[[i, 0]];
            let fy = frac[[i, 1]];
            let fz = frac[[i, 2]];
            assert!((0.0..1.0).contains(&fx));
            assert!((0.0..1.0).contains(&fy));
            assert!((0.0..1.0).contains(&fz));
        }
    }

    #[test]
    fn calc_distance_matches_components() {
        let bx = SimBox::cube(3.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let a = array![0.1, 0.2, 0.3];
        let b = array![2.9, 0.2, 0.3];
        let d2 = bx.calc_distance2(a.view(), b.view());
        let dr = bx.shortest_vector(a.view(), b.view());
        let expected = dr.dot(&dr);
        assert!((d2 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_lengths_ortho() {
        let bx = SimBox::ortho(
            array![2.0, 4.0, 5.0],
            array![0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        let lengths = bx.lengths();
        assert_close(lengths[0], 2.0);
        assert_close(lengths[1], 4.0);
        assert_close(lengths[2], 5.0);
    }

    #[test]
    fn test_tilts_values() {
        let h = array![[2.0, 1.0, 2.0], [0.0, 4.0, 3.0], [0.0, 0.0, 5.0]];
        let bx = SimBox::new(h, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let tilts = bx.tilts();
        assert_close(tilts[0], 1.0);
        assert_close(tilts[1], 2.0);
        assert_close(tilts[2], 3.0);
    }

    #[test]
    fn test_volume() {
        let bx = SimBox::ortho(
            array![2.0, 3.0, 4.0],
            array![0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        assert_close(bx.volume(), 24.0);
    }

    #[test]
    fn test_wrap_single_and_multi() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[10.0, -5.0, -5.0], [0.0, 0.5, 0.0]];
        let wrapped = bx.wrap(pts.view());
        assert_close(wrapped[[0, 0]], 0.0);
        assert_close(wrapped[[0, 1]], 1.0);
        assert_close(wrapped[[0, 2]], 1.0);
        assert_close(wrapped[[1, 0]], 0.0);
        assert_close(wrapped[[1, 1]], 0.5);
        assert_close(wrapped[[1, 2]], 0.0);
    }

    #[test]
    fn test_fractional_and_cartesian() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let p = array![-1.0, -1.0, -1.0];
        let frac = bx.make_fractional(p.view());
        assert_close(frac[0], 0.5);
        assert_close(frac[1], 0.5);
        assert_close(frac[2], 0.5);
        let cart = bx.make_cartesian(frac.view());
        assert_close(cart[0], 1.0);
        assert_close(cart[1], 1.0);
        assert_close(cart[2], 1.0);
    }

    #[test]
    fn test_to_frac_to_cart_roundtrip() {
        let bx = SimBox::ortho(
            array![2.0, 3.0, 4.0],
            array![1.0, 2.0, 3.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        let pts = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let frac = bx.to_frac(pts.view());
        let cart = bx.to_cart(frac.view());
        for i in 0..pts.nrows() {
            for j in 0..3 {
                assert_close(pts[[i, j]], cart[[i, j]]);
            }
        }
    }

    #[test]
    fn test_shortest_vector_and_distance() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let a = array![0.1, 0.0, 0.0];
        let b = array![1.9, 0.0, 0.0];
        let dr = bx.shortest_vector(a.view(), b.view());
        assert_close(dr[0], -0.2);
        assert_close(dr[1], 0.0);
        assert_close(dr[2], 0.0);
        let d2 = bx.calc_distance2(a.view(), b.view());
        assert_close(d2, 0.04);
    }

    #[test]
    fn test_contains_point_non_pbc() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [false, false, false])
            .expect("invalid box length");
        assert!(bx.contains_point(&[0.5, 0.5, 0.5]));
        assert!(!bx.contains_point(&[-0.1, 0.5, 0.5]));
        assert!(!bx.contains_point(&[2.1, 0.5, 0.5]));
    }

    #[test]
    fn test_contains_mask() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [2.1, 0.0, 0.0], [-0.1, 0.0, 0.0]];
        let mask = bx.contains(&pts);
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(!mask[2]);
    }

    #[test]
    fn test_simbox_free_basic() {
        let pts = array![[1.0 as F, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let bx = SimBox::free(pts.view(), 1.0).unwrap();
        assert_eq!(bx.pbc(), [false, false, false]);
        // origin should be min - padding = [0.0, 1.0, 2.0]
        let o = bx.origin_view();
        assert!((o[0] - 0.0).abs() < 1e-5);
        assert!((o[1] - 1.0).abs() < 1e-5);
        assert!((o[2] - 2.0).abs() < 1e-5);
        // lengths should be (max-min) + 2*padding = [5.0, 5.0, 5.0]
        let l = bx.lengths();
        assert!((l[0] - 5.0).abs() < 1e-5);
        assert!((l[1] - 5.0).abs() < 1e-5);
        assert!((l[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_simbox_free_single_point() {
        let pts = array![[1.0 as F, 2.0, 3.0]];
        let bx = SimBox::free(pts.view(), 2.0).unwrap();
        assert_eq!(bx.pbc(), [false, false, false]);
        // lengths = max(0 + 4, 2) = 4 on each axis
        let l = bx.lengths();
        assert!(l[0] >= 2.0);
        assert!(l[1] >= 2.0);
        assert!(l[2] >= 2.0);
    }

    #[test]
    fn test_simbox_free_empty() {
        use ndarray::Array2;
        let pts = Array2::<F>::zeros((0, 3));
        let bx = SimBox::free(pts.view(), 1.0).unwrap();
        assert_eq!(bx.pbc(), [false, false, false]);
    }

    #[test]
    fn test_simbox_pbc_accessor() {
        let bx = SimBox::cube(1.0, array![0.0 as F, 0.0, 0.0], [true, false, true]).unwrap();
        assert_eq!(bx.pbc(), [true, false, true]);
    }
}
