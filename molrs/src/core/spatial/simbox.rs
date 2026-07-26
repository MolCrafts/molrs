//! Triclinic simulation box and periodic operations based on ndarray.
//!
//! Conventions (fractional/cartesian):
//! - cart = origin + H * frac
//! - frac = H^{-1} * (cart - origin)
//! - Lattice vectors are the columns of H.

use crate::math;
use crate::types::{F, F3, F3View, F3x3, FNx3, FNx3View, Pbc3};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, array};

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
    /// Whether the cell is geometrically defined. `false` marks a "no-cell"
    /// box (an undefined / zero-volume cell) — distinct from `pbc`, which only
    /// describes periodicity. A defined non-periodic box (e.g. a free-boundary
    /// bounding box) keeps `cell_defined = true`; only a box with no meaningful
    /// cell at all (carrying the identity matrix purely so geometry ops are
    /// no-ops) sets it `false`.
    cell_defined: bool,
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
    /// Cell lengths/angles do not describe a physical triclinic cell.
    InvalidAngles,
}

impl SimBox {
    /// Construct from triclinic cell matrix `H`, origin `O`, and per-axis PBC flags
    pub fn new(h: F3x3, origin: F3, pbc: Pbc3) -> Result<Self, BoxError> {
        Self::new_cell(h, origin, pbc, true)
    }

    /// Construct a box, explicitly marking whether the cell is geometrically
    /// defined. Pass `cell_defined = false` for a "no-cell" box (undefined /
    /// zero-volume): supply the identity matrix so geometry ops degrade to
    /// no-ops, and `volume` / `is_cell_defined` reflect the undefined cell.
    pub fn new_cell(h: F3x3, origin: F3, pbc: Pbc3, cell_defined: bool) -> Result<Self, BoxError> {
        if let Some(inv) = math::inv3(&h) {
            let kind = detect_box_kind(&h);
            Ok(Self {
                h,
                inv,
                origin,
                pbc,
                kind,
                cell_defined,
            })
        } else {
            Err(BoxError::SingularCell)
        }
    }

    /// Whether the cell is geometrically defined (`false` ⇒ a no-cell box of
    /// undefined / zero volume). Distinct from periodicity ([`is_free`]).
    ///
    /// [`is_free`]: SimBox::is_free
    pub fn is_cell_defined(&self) -> bool {
        self.cell_defined
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

    /// Restricted-triclinic matrix from edge lengths and angles in degrees.
    pub fn matrix_from_lengths_angles(lengths: [F; 3], angles: [F; 3]) -> Result<F3x3, BoxError> {
        let [a, b, c] = lengths;
        let [alpha, beta, gamma] = angles.map(F::to_radians);
        if [a, b, c].iter().any(|value| *value <= 0.0)
            || [alpha, beta, gamma]
                .iter()
                .any(|angle| !(*angle > 0.0 && *angle < std::f64::consts::PI))
        {
            return Err(BoxError::InvalidAngles);
        }
        let (cos_a, cos_b, cos_c) = (alpha.cos(), beta.cos(), gamma.cos());
        let xy = b * cos_c;
        let xz = c * cos_b;
        let ly = (b * b - xy * xy).sqrt();
        if !ly.is_finite() || ly <= 0.0 {
            return Err(BoxError::InvalidAngles);
        }
        let yz = (b * c * cos_a - xy * xz) / ly;
        let lz2 = c * c - xz * xz - yz * yz;
        if !lz2.is_finite() || lz2 <= 0.0 {
            return Err(BoxError::InvalidAngles);
        }
        Ok(array![[a, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz2.sqrt()]])
    }

    /// Restricted-triclinic matrix from diagonal sizes and `(xy, xz, yz)` tilts.
    pub fn matrix_from_lengths_tilts(lengths: [F; 3], tilts: [F; 3]) -> F3x3 {
        array![
            [lengths[0], tilts[0], tilts[1]],
            [0.0, lengths[1], tilts[2]],
            [0.0, 0.0, lengths[2]],
        ]
    }

    /// Convert a general cell matrix to LAMMPS restricted-triclinic form.
    pub fn restricted_matrix(matrix: FNx3View<'_>) -> Result<F3x3, BoxError> {
        if matrix.dim() != (3, 3) {
            return Err(BoxError::InvalidMatrixShape {
                rows: matrix.nrows(),
                cols: matrix.ncols(),
            });
        }
        let a = matrix.column(0).to_owned();
        let b = matrix.column(1).to_owned();
        let c = matrix.column(2).to_owned();
        let ax = math::norm3(&a);
        if ax <= 0.0 {
            return Err(BoxError::SingularCell);
        }
        let ua = &a / ax;
        let bx = b.dot(&ua);
        let cross_ab = math::cross3(&a, &b);
        let cross_norm = math::norm3(&cross_ab);
        if cross_norm <= 0.0 {
            return Err(BoxError::SingularCell);
        }
        let by = math::norm3(&math::cross3(&ua, &b));
        let uab = &cross_ab / cross_norm;
        let cx = c.dot(&ua);
        let cy = c.dot(&math::cross3(&uab, &ua));
        let cz = c.dot(&uab);
        Ok(array![[ax, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]])
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

    /// Create a tight orthorhombic box around a point cloud.
    ///
    /// Unlike [`free`](Self::free), padding is specified per axis and may be
    /// zero. Periodicity is supplied by the caller instead of being forced to
    /// free-boundary semantics.
    pub fn from_bounds(points: FNx3View<'_>, padding: [F; 3], pbc: Pbc3) -> Result<Self, BoxError> {
        if points.nrows() == 0 {
            return Err(BoxError::InvalidVectorLength { len: 0 });
        }
        if padding.iter().any(|value| *value < 0.0) {
            return Err(BoxError::InvalidVectorLength { len: 0 });
        }
        let mut min = [points[[0, 0]], points[[0, 1]], points[[0, 2]]];
        let mut max = min;
        for point in points.rows().into_iter().skip(1) {
            for d in 0..3 {
                min[d] = min[d].min(point[d]);
                max[d] = max[d].max(point[d]);
            }
        }
        let origin = array![
            min[0] - padding[0],
            min[1] - padding[1],
            min[2] - padding[2]
        ];
        let lengths = array![
            max[0] - min[0] + 2.0 * padding[0],
            max[1] - min[1] + 2.0 * padding[1],
            max[2] - min[2] + 2.0 * padding[2]
        ];
        Self::ortho(lengths, origin, pbc)
    }

    /// Create a non-periodic (free-boundary) box enclosing all points, reading
    /// positions from three separate `x`/`y`/`z` slices (SoA layout).
    ///
    /// Arithmetically identical to [`free`](Self::free): computes the same
    /// axis-aligned bounding box (min/max over all points) plus `padding` on
    /// each side, and returns a box byte-identical to `free` on the same
    /// points. Provided so callers holding column-major (SoA) coordinates need
    /// not interleave them into an owned `Array2` first.
    ///
    /// # Errors
    /// Returns `BoxError` if the resulting box is degenerate.
    ///
    /// # Panics
    /// Panics if `padding <= 0` or the three slices do not have equal length.
    pub fn free_columns(xs: &[F], ys: &[F], zs: &[F], padding: F) -> Result<Self, BoxError> {
        assert!(padding > 0.0, "padding must be positive");
        assert!(
            xs.len() == ys.len() && ys.len() == zs.len(),
            "x/y/z slices must have equal length"
        );
        let n = xs.len();
        if n == 0 {
            // Empty point set -- return a unit cube at origin
            return Self::cube(padding, array![0.0 as F, 0.0, 0.0], [false, false, false]);
        }
        let mut min = array![xs[0], ys[0], zs[0]];
        let mut max = min.clone();
        for i in 1..n {
            let p = [xs[i], ys[i], zs[i]];
            for d in 0..3 {
                if p[d] < min[d] {
                    min[d] = p[d];
                }
                if p[d] > max[d] {
                    max[d] = p[d];
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

    /// `true` when the box is free (non-periodic on every axis).
    pub fn is_free(&self) -> bool {
        self.pbc.iter().all(|&p| !p)
    }

    /// Geometry style label: `"free"` (no periodic axis), `"orthogonal"`
    /// (diagonal H), or `"triclinic"`.
    pub fn style(&self) -> &'static str {
        if self.is_free() {
            "free"
        } else {
            match self.kind {
                BoxKind::Ortho { .. } => "orthogonal",
                BoxKind::Triclinic => "triclinic",
            }
        }
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

    /// Lattice angles `[alpha, beta, gamma]` in degrees.
    pub fn angles(&self) -> F3 {
        let a = self.lattice(0);
        let b = self.lattice(1);
        let c = self.lattice(2);
        let angle = |u: &F3, v: &F3| {
            (u.dot(v) / (math::norm3(u) * math::norm3(v)))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        };
        array![angle(&b, &c), angle(&a, &c), angle(&a, &b)]
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

    /// Fractional coordinates **without** the wrap into `[0, 1)`.
    ///
    /// [`make_fractional_fast_arr3`](Self::make_fractional_fast_arr3) folds
    /// every axis back into the primitive cell unconditionally, which is right
    /// for a fully periodic box but destroys the information a caller needs on
    /// a **non-periodic** axis: a point above the box must stay above it, so
    /// that a cell-list assignment can clamp it to the edge cell instead of
    /// wrapping it to the opposite face. Callers that dispatch on
    /// [`pbc`](Self::pbc) per axis — see `CellGrid` — start from this raw value
    /// and apply wrap or clamp themselves.
    ///
    /// `make_fractional_fast_arr3(r)` is exactly this followed by
    /// `f - f.floor()` per component, so the two agree bit-for-bit on any point
    /// inside the cell.
    #[inline(always)]
    pub fn make_fractional_raw_arr3(&self, r: [F; 3]) -> [F; 3] {
        match &self.kind {
            BoxKind::Ortho { inv_len, .. } => [
                (r[0] - self.origin[0]) * inv_len[0],
                (r[1] - self.origin[1]) * inv_len[1],
                (r[2] - self.origin[2]) * inv_len[2],
            ],
            BoxKind::Triclinic => {
                let rv = ArrayView1::from_shape(3, &r).expect("make_fractional_raw_arr3 shape");
                let dr = &rv - &self.origin.view();
                let f = self.inv.dot(&dr);
                [f[0], f[1], f[2]]
            }
        }
    }

    /// Convert fractional coordinates to Cartesian coordinates
    pub fn make_cartesian(&self, frac: F3View<'_>) -> F3 {
        &self.origin + &self.h.dot(&frac)
    }

    /// A detached, `Copy` minimum-image kernel.
    ///
    /// [`shortest_vector_impl`](Self::shortest_vector_impl) needs a `&SimBox`,
    /// which is awkward for a caller whose hot loop already holds the owning
    /// structure mutably: it either clones the box — two `Array2` allocations,
    /// per evaluation — or restructures its borrows. `Mic` lifts the convention
    /// out as a plain value with everything it needs on the stack, so it can be
    /// captured once and carried into the loop.
    ///
    /// Bit-identical to `shortest_vector_impl`: both dispatch to the same
    /// arithmetic.
    pub fn mic(&self) -> Mic {
        match &self.kind {
            BoxKind::Ortho { len, inv_len } => Mic::Ortho {
                len: [len[0], len[1], len[2]],
                inv_len: [inv_len[0], inv_len[1], inv_len[2]],
                pbc: self.pbc(),
            },
            BoxKind::Triclinic => {
                let mut h = [0.0; 9];
                let mut inv = [0.0; 9];
                for i in 0..3 {
                    for j in 0..3 {
                        h[3 * i + j] = self.h[[i, j]];
                        inv[3 * i + j] = self.inv[[i, j]];
                    }
                }
                Mic::Triclinic {
                    h,
                    inv,
                    pbc: self.pbc(),
                }
            }
        }
    }

    /// Hot-loop MIC kernel: takes and returns `[F; 3]`, zero allocation.
    ///
    /// Ortho boxes use the `dr − round(dr / L) · L` fast path; triclinic
    /// boxes fall back to the general `H · round(H⁻¹ · dr)` form. This
    /// is the single source of truth for the minimum-image convention —
    /// both [`shortest_vector`](Self::shortest_vector) (ergonomic
    /// `F3View` / `Array1` API) and
    /// [`shortest_vector_impl`](Self::shortest_vector_impl) (zero-alloc
    /// `[F; 3]` API) route through here.
    #[inline(always)]
    fn mic_kernel(&self, a: [F; 3], b: [F; 3]) -> [F; 3] {
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
                // General triclinic path: fold the displacement through
                // fractional coords and wrap each periodic axis to
                // `[-0.5, 0.5)`.
                //
                // Written out on the stack rather than as `inv.dot(dr)` /
                // `h.dot(frac)`. Those allocate an `Array1` each, and this
                // kernel sits in the innermost pair loop of every caller — a
                // packer evaluates it millions of times per objective
                // evaluation, where two heap allocations per pair dominate the
                // arithmetic outright. Summation order matches ndarray's
                // matrix-vector product (k ascending), so the result is
                // bit-identical to the allocating form.
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let mut f = [
                    self.inv[[0, 0]] * d[0] + self.inv[[0, 1]] * d[1] + self.inv[[0, 2]] * d[2],
                    self.inv[[1, 0]] * d[0] + self.inv[[1, 1]] * d[1] + self.inv[[1, 2]] * d[2],
                    self.inv[[2, 0]] * d[0] + self.inv[[2, 1]] * d[1] + self.inv[[2, 2]] * d[2],
                ];
                for (fk, &periodic) in f.iter_mut().zip(self.pbc.iter()) {
                    if periodic {
                        *fk -= fk.round();
                    }
                }
                [
                    self.h[[0, 0]] * f[0] + self.h[[0, 1]] * f[1] + self.h[[0, 2]] * f[2],
                    self.h[[1, 0]] * f[0] + self.h[[1, 1]] * f[1] + self.h[[1, 2]] * f[2],
                    self.h[[2, 0]] * f[0] + self.h[[2, 1]] * f[1] + self.h[[2, 2]] * f[2],
                ]
            }
        }
    }

    /// Minimum image displacement vector from `r1` to `r2` (returns `r2 − r1`).
    ///
    /// Ergonomic ndarray-flavoured API: takes views and returns an owned
    /// `Array1<F>`. Inside hot loops prefer
    /// [`shortest_vector_impl`](Self::shortest_vector_impl) — it avoids the
    /// heap allocation for the output (~70% faster per call).
    #[inline]
    pub fn shortest_vector(&self, r1: F3View<'_>, r2: F3View<'_>) -> F3 {
        let dr = self.mic_kernel([r1[0], r1[1], r1[2]], [r2[0], r2[1], r2[2]]);
        array![dr[0], dr[1], dr[2]]
    }

    /// Zero-allocation MIC displacement from `a` to `b` (returns `b − a`).
    ///
    /// Stack-array in / out; the canonical hot-loop entry point. Used by
    /// [`LinkCell`](crate::spatial::neighbors::LinkCell),
    /// [`BruteForce`](crate::spatial::neighbors::BruteForce), and
    /// [`AabbQuery`](crate::spatial::neighbors::AabbQuery) inner loops.
    #[inline(always)]
    pub fn shortest_vector_impl(&self, a: [F; 3], b: [F; 3]) -> [F; 3] {
        self.mic_kernel(a, b)
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

    /// Row-wise minimum-image distances between equally sized point arrays.
    pub fn distances(&self, points1: FNx3View<'_>, points2: FNx3View<'_>) -> Array1<F> {
        assert_eq!(points1.raw_dim(), points2.raw_dim());
        let values = points1
            .rows()
            .into_iter()
            .zip(points2.rows())
            .map(|(a, b)| {
                let dr = self.shortest_vector_impl([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
                (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt()
            })
            .collect();
        Array1::from_vec(values)
    }

    /// All pairwise minimum-image displacement vectors (`points2 - points1`).
    pub fn pairwise_delta(&self, points1: FNx3View<'_>, points2: FNx3View<'_>) -> Array3<F> {
        let mut out = Array3::zeros((points1.nrows(), points2.nrows(), 3));
        for (i, a) in points1.rows().into_iter().enumerate() {
            for (j, b) in points2.rows().into_iter().enumerate() {
                let dr = self.shortest_vector_impl([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
                for d in 0..3 {
                    out[[i, j, d]] = dr[d];
                }
            }
        }
        out
    }

    /// All pairwise minimum-image distances.
    pub fn pairwise_distances(&self, points1: FNx3View<'_>, points2: FNx3View<'_>) -> Array2<F> {
        let mut out = Array2::zeros((points1.nrows(), points2.nrows()));
        for (i, a) in points1.rows().into_iter().enumerate() {
            for (j, b) in points2.rows().into_iter().enumerate() {
                let dr = self.shortest_vector_impl([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
                out[[i, j]] = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt();
            }
        }
        out
    }

    /// Return a box with its cell matrix right-multiplied by a transform.
    pub fn transformed(&self, transformation: &F3x3) -> Result<Self, BoxError> {
        Self::new_cell(
            self.h.dot(transformation),
            self.origin.clone(),
            self.pbc,
            self.cell_defined,
        )
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

    /// Integer periodic images for Cartesian points.
    pub fn images(&self, xyz: FNx3View<'_>) -> Array2<i64> {
        let frac = self.to_frac(xyz);
        let mut images = Array2::zeros((frac.nrows(), 3));
        for i in 0..frac.nrows() {
            for d in 0..3 {
                if self.pbc[d] {
                    images[[i, d]] = (frac[[i, d]] + 1e-8).floor() as i64;
                }
            }
        }
        images
    }

    /// Reconstruct unwrapped coordinates from wrapped points and image flags.
    pub fn unwrap(&self, xyz: FNx3View<'_>, images: ArrayView2<'_, i64>) -> FNx3 {
        assert_eq!(xyz.raw_dim(), images.raw_dim());
        assert_eq!(xyz.ncols(), 3);
        let mut result = xyz.to_owned();
        for i in 0..xyz.nrows() {
            let image = array![
                images[[i, 0]] as F,
                images[[i, 1]] as F,
                images[[i, 2]] as F,
            ];
            let shift = self.h.dot(&image);
            for d in 0..3 {
                result[[i, d]] += shift[d];
            }
        }
        result
    }

    pub fn get_corners(&self) -> FNx3 {
        self.to_cart(
            array![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
            .view(),
        )
    }

    /// Axis-aligned bounding box of the cell geometry.
    ///
    /// Layout: rows = x/y/z, col 0 = min, col 1 = max — the AABB of the eight
    /// corners. For triclinic cells this is larger than the true cell volume;
    /// use [`isin`](Self::isin) for membership. Geometric region types that
    /// describe the same volume live in `spatial::region`
    /// (`Cuboid` / `Parallelepiped`) — not on this type.
    pub fn bounds(&self) -> FNx3 {
        let corners = self.get_corners();
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            let mut lo = corners[[0, d]];
            let mut hi = lo;
            for i in 1..corners.nrows() {
                lo = lo.min(corners[[i, d]]);
                hi = hi.max(corners[[i, d]]);
            }
            b[[d, 0]] = lo;
            b[[d, 1]] = hi;
        }
        b
    }
}

/// Minimum-image convention as a standalone `Copy` value.
///
/// Produced by [`SimBox::mic`]. Carries no periodicity of its own beyond the
/// flags captured at construction, so a box that changes shape needs a fresh
/// one — which is the point: it is meant to be captured once per evaluation and
/// read many times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mic {
    /// No axis wraps: displacements pass through untouched.
    Free,
    Ortho {
        len: [F; 3],
        inv_len: [F; 3],
        pbc: [bool; 3],
    },
    Triclinic {
        /// Row-major 3x3 lattice, columns are the lattice vectors.
        h: [F; 9],
        /// Row-major 3x3 inverse lattice.
        inv: [F; 9],
        pbc: [bool; 3],
    },
}

impl Mic {
    /// Minimum image of a displacement.
    ///
    /// Takes the displacement rather than two points: the convention depends
    /// only on the separation, and callers in pair loops already have it.
    #[inline(always)]
    pub fn apply(&self, d: [F; 3]) -> [F; 3] {
        match self {
            Mic::Free => d,
            Mic::Ortho { len, inv_len, pbc } => {
                let mut dr = d;
                for k in 0..3 {
                    if pbc[k] {
                        dr[k] -= (dr[k] * inv_len[k]).round() * len[k];
                    }
                }
                dr
            }
            Mic::Triclinic { h, inv, pbc } => {
                let mut f = [
                    inv[0] * d[0] + inv[1] * d[1] + inv[2] * d[2],
                    inv[3] * d[0] + inv[4] * d[1] + inv[5] * d[2],
                    inv[6] * d[0] + inv[7] * d[1] + inv[8] * d[2],
                ];
                for (fk, &periodic) in f.iter_mut().zip(pbc.iter()) {
                    if periodic {
                        *fk -= fk.round();
                    }
                }
                [
                    h[0] * f[0] + h[1] * f[1] + h[2] * f[2],
                    h[3] * f[0] + h[4] * f[1] + h[5] * f[2],
                    h[6] * f[0] + h[7] * f[1] + h[8] * f[2],
                ]
            }
        }
    }

    /// Collapse to [`Mic::Free`] when no axis wraps, so the hot path is a
    /// single match arm instead of three predictable-but-present branches.
    #[inline]
    pub fn simplified(self) -> Self {
        let pbc = match self {
            Mic::Free => return self,
            Mic::Ortho { pbc, .. } | Mic::Triclinic { pbc, .. } => pbc,
        };
        if pbc.iter().any(|&p| p) {
            self
        } else {
            Mic::Free
        }
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
    fn cell_defined_distinct_from_pbc() {
        // A defined box (any pbc) reports cell_defined = true (the default).
        let defined = SimBox::ortho(
            array![2.0, 2.0, 2.0],
            array![0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        assert!(defined.is_cell_defined());
        assert!(defined.is_free()); // non-periodic => free, but cell IS defined
        assert_close(defined.volume(), 8.0); // real cell volume preserved (RDF)

        // A no-cell box carries the identity cell (so geometry is a no-op) but
        // is marked cell_defined = false.
        let nocell = SimBox::new_cell(
            ndarray::Array2::eye(3),
            array![0.0, 0.0, 0.0],
            [false, false, false],
            false,
        )
        .unwrap();
        assert!(!nocell.is_cell_defined());
        // geometry no-ops on the identity cell:
        let pts = array![[1.0, 2.0, 3.0]];
        assert_eq!(nocell.wrap(pts.view()), pts);
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
    fn test_isin_point_non_pbc() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [false, false, false])
            .expect("invalid box length");
        let pts = array![[0.5, 0.5, 0.5], [-0.1, 0.5, 0.5], [2.1, 0.5, 0.5]];
        let mask = bx.isin(pts.view());
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(!mask[2]);
    }

    #[test]
    fn test_isin_mask() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [2.1, 0.0, 0.0], [-0.1, 0.0, 0.0]];
        let mask = bx.isin(pts.view());
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

    #[test]
    fn free_columns_matches_free_bitwise() {
        let pts = array![[1.0 as F, 2.0, 3.0], [4.0, -5.0, 6.0], [-2.5, 5.5, 0.25]];
        let xs = vec![1.0 as F, 4.0, -2.5];
        let ys = vec![2.0 as F, -5.0, 5.5];
        let zs = vec![3.0 as F, 6.0, 0.25];
        let a = SimBox::free(pts.view(), 1.5).unwrap();
        let b = SimBox::free_columns(&xs, &ys, &zs, 1.5).unwrap();

        let (oa, ob) = (a.origin_view(), b.origin_view());
        let (ha, hb) = (a.h_view(), b.h_view());
        for d in 0..3 {
            assert_eq!(oa[d], ob[d], "origin bitwise");
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(ha[[i, j]], hb[[i, j]], "H bitwise");
            }
        }
        assert_eq!(a.pbc(), b.pbc());
    }

    #[test]
    fn mic_matches_the_borrowed_kernel_on_both_box_kinds() {
        // `Mic` exists so a caller can avoid holding a `&SimBox`; it must not
        // become a second, drifting definition of the convention.
        let ortho = SimBox::ortho(
            array![10.0, 11.0, 12.0],
            array![0.5, -1.0, 2.0],
            [true, false, true],
        )
        .unwrap();
        let tri = SimBox::new(
            SimBox::matrix_from_lengths_angles([10.0, 11.0, 12.0], [70.0, 80.0, 65.0]).unwrap(),
            array![0.3, -0.2, 1.1],
            [true, true, false],
        )
        .unwrap();

        for bx in [&ortho, &tri] {
            let mic = bx.mic();
            for (a, b) in [
                ([0.0 as F, 0.0, 0.0], [9.0 as F, 1.0, -3.0]),
                ([1.5, -2.5, 3.5], [-7.25, 8.125, 0.0625]),
                ([4.0, 4.0, 4.0], [4.0, 4.0, 4.0]),
            ] {
                let want = bx.shortest_vector_impl(a, b);
                let got = mic.apply([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                for d in 0..3 {
                    assert_eq!(got[d], want[d], "component {d} for {a:?} -> {b:?}");
                }
            }
        }
    }

    #[test]
    fn a_box_with_no_periodic_axis_simplifies_to_free() {
        let free = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false; 3]).unwrap();
        assert_eq!(free.mic().simplified(), Mic::Free);
        let d = [123.0 as F, -456.0, 789.0];
        assert_eq!(Mic::Free.apply(d), d);

        let periodic = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, true, false]).unwrap();
        assert_ne!(periodic.mic().simplified(), Mic::Free);
    }

    #[test]
    fn triclinic_mic_matches_the_allocating_matrix_form() {
        // The stack-arithmetic kernel must agree bit-for-bit with the
        // `inv.dot(dr)` / `h.dot(frac)` formulation it replaced; ndarray sums
        // a matrix-vector product with k ascending, and so does the kernel.
        let bx = SimBox::new(
            SimBox::matrix_from_lengths_angles([10.0, 11.0, 12.0], [70.0, 80.0, 65.0]).unwrap(),
            array![0.3, -0.2, 1.1],
            [true, true, false],
        )
        .unwrap();

        let pts = [
            ([0.0 as F, 0.0, 0.0], [9.0 as F, 1.0, -3.0]),
            ([1.5, -2.5, 3.5], [-7.25, 8.125, 0.0625]),
            ([4.0, 4.0, 4.0], [4.0, 4.0, 4.0]),
        ];
        for (a, b) in pts {
            let dr_cart = array![b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let mut dr_frac = bx.inv.dot(&dr_cart);
            for d in 0..3 {
                if bx.pbc[d] {
                    dr_frac[d] -= dr_frac[d].round();
                }
            }
            let want = bx.h.dot(&dr_frac);
            let got = bx.shortest_vector_impl(a, b);
            for d in 0..3 {
                assert_eq!(got[d], want[d], "component {d} for {a:?} -> {b:?}");
            }
        }
    }
}
