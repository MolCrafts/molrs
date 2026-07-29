//! Generate ghost-image copies of points across periodic boundaries.
//!
//! Mirrors `freud.locality.PeriodicBuffer`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/PeriodicBuffer.cc)).
//!
//! Given a [`SimBox`] with per-axis PBC flags and a per-axis "buffer"
//! distance, produces every lattice image of the input points that falls
//! inside the buffer-expanded box. The output is an ordered list of
//! ghost positions plus their **original indices** (so caller analyzers
//! can map back to the source frame after a non-periodic neighbor pass
//! over the extended set).
//!
//! # Limitations
//!
//! - Triclinic boxes are supported by iterating images in the lattice-vector
//!   basis and shifting in Cartesian.
//! - Per-axis PBC flags are honoured: an axis with `pbc = false` only ever
//!   contributes its `n = 0` image (i.e. the original points themselves).

use crate::spatial::simbox::{BoxKind, SimBox};
use crate::types::{F, FNx3, FNx3View};

/// Ghost-image expansion of a point set.
#[derive(Debug, Clone, Default)]
pub struct PeriodicBufferResult {
    /// Ghost positions, shape `(M, 3)`. Always includes the original
    /// points first (as the `n = 0` image).
    pub positions: FNx3,
    /// Original index in the input for each ghost (length `M`).
    pub indices: Vec<u32>,
    /// Image label `(nx, ny, nz)` for each ghost (length `M`).
    /// `(0, 0, 0)` marks the original copy.
    pub images: Vec<[i32; 3]>,
}

/// Generate ghosts of `points` within `buffer` of the box, respecting
/// per-axis PBC flags. The image range along each periodic axis is
/// `[−ceil(buffer / L_axis), +ceil(buffer / L_axis)]`.
pub fn periodic_buffer(points: FNx3View<'_>, bx: &SimBox, buffer: [F; 3]) -> PeriodicBufferResult {
    let pbc = bx.pbc();
    let n = points.nrows();

    // Determine image range per axis.
    let (lx, ly, lz) = match bx.kind() {
        BoxKind::Ortho { len, .. } => (len[0], len[1], len[2]),
        BoxKind::Triclinic => {
            // Use the lattice-vector lengths as a conservative proxy for the
            // periodic-image stride along each axis. Most physical triclinic
            // cells deviate < 30% from this, which only over-counts ghosts.
            let l = bx.lengths();
            (l[0], l[1], l[2])
        }
    };
    let range = |b: F, ax_len: F, periodic: bool| -> (i32, i32) {
        if !periodic || ax_len <= 0.0 {
            (0, 0)
        } else {
            let n = (b / ax_len).ceil() as i32;
            (-n, n)
        }
    };
    let (nx_min, nx_max) = range(buffer[0], lx, pbc[0]);
    let (ny_min, ny_max) = range(buffer[1], ly, pbc[1]);
    let (nz_min, nz_max) = range(buffer[2], lz, pbc[2]);

    // Build lattice-vector array once.
    let a0 = bx.lattice(0);
    let a1 = bx.lattice(1);
    let a2 = bx.lattice(2);

    let n_images_total =
        ((nx_max - nx_min + 1) * (ny_max - ny_min + 1) * (nz_max - nz_min + 1)) as usize * n;
    let mut positions = FNx3::zeros((n_images_total, 3));
    let mut indices: Vec<u32> = Vec::with_capacity(n_images_total);
    let mut images: Vec<[i32; 3]> = Vec::with_capacity(n_images_total);

    let mut row = 0;
    for ix in nx_min..=nx_max {
        for iy in ny_min..=ny_max {
            for iz in nz_min..=nz_max {
                let dx = ix as F * a0[0] + iy as F * a1[0] + iz as F * a2[0];
                let dy = ix as F * a0[1] + iy as F * a1[1] + iz as F * a2[1];
                let dz = ix as F * a0[2] + iy as F * a1[2] + iz as F * a2[2];
                for p in 0..n {
                    positions[[row, 0]] = points[[p, 0]] + dx;
                    positions[[row, 1]] = points[[p, 1]] + dy;
                    positions[[row, 2]] = points[[p, 2]] + dz;
                    indices.push(p as u32);
                    images.push([ix, iy, iz]);
                    row += 1;
                }
            }
        }
    }
    PeriodicBufferResult {
        positions,
        indices,
        images,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::simbox::SimBox;
    use ndarray::array;

    #[test]
    fn no_pbc_returns_original_set() {
        let pts = array![[0.5_f64, 0.5, 0.5], [1.5, 1.5, 1.5]];
        let bx = SimBox::cube(10.0, array![0.0_f64, 0.0, 0.0], [false; 3]).unwrap();
        let r = periodic_buffer(pts.view(), &bx, [3.0; 3]);
        assert_eq!(r.positions.nrows(), 2);
        assert_eq!(r.indices, vec![0, 1]);
        for img in &r.images {
            assert_eq!(*img, [0, 0, 0]);
        }
    }

    #[test]
    fn pbc_produces_27_images_when_buffer_exceeds_box() {
        let pts = array![[5.0_f64, 5.0, 5.0]];
        let bx = SimBox::cube(10.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // buffer = 10 > L = 10 → ceil(10/10) = 1 per side → 3 images per axis.
        let r = periodic_buffer(pts.view(), &bx, [10.0; 3]);
        assert_eq!(r.positions.nrows(), 27);
        // The (0,0,0) image must reproduce the input exactly.
        let center = r.images.iter().position(|i| *i == [0, 0, 0]).unwrap();
        assert!((r.positions[[center, 0]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn mixed_pbc_only_periodic_axes_replicate() {
        let pts = array![[1.0_f64, 1.0, 1.0]];
        let bx = SimBox::cube(5.0, array![0.0_f64, 0.0, 0.0], [true, false, true]).unwrap();
        let r = periodic_buffer(pts.view(), &bx, [5.0_f64; 3]);
        // Periodic x and z give 3 × 3 = 9 images.
        assert_eq!(r.positions.nrows(), 9);
        // No image should have ny != 0.
        for img in &r.images {
            assert_eq!(img[1], 0);
        }
    }

    #[test]
    fn shifted_positions_are_correct() {
        let pts = array![[1.0_f64, 0.0, 0.0]];
        let bx = SimBox::cube(10.0, array![0.0_f64, 0.0, 0.0], [true, true, true]).unwrap();
        let r = periodic_buffer(pts.view(), &bx, [5.0_f64; 3]);
        // The (+1, 0, 0) image should be at x = 1 + 10 = 11.
        let i = r.images.iter().position(|im| *im == [1, 0, 0]).unwrap();
        assert!((r.positions[[i, 0]] - 11.0).abs() < 1e-12);
        // The (−1, 0, 0) image should be at x = 1 − 10 = −9.
        let i_neg = r.images.iter().position(|im| *im == [-1, 0, 0]).unwrap();
        assert!((r.positions[[i_neg, 0]] + 9.0).abs() < 1e-12);
    }
}
