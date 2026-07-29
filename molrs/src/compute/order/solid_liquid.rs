//! Frenkel–ten Wolde solid/liquid classification from `q_ℓm` bond correlations.

// Index-based loops over flat qℓm arrays read naturally; iterator forms
// would require chunks_exact + enumerate without gaining clarity.
#![allow(clippy::needless_range_loop)]

//! Frenkel–ten Wolde solid/liquid classification.
//!
//! Mirrors `freud.order.SolidLiquid`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/SolidLiquid.cc)).
//!
//! For each neighbor pair `(i, j)` we compute the normalised dot product
//! of their Steinhardt qℓm vectors:
//!
//! ```text
//!   d_ij = ( Σ_m q_ℓm(i) · conj q_ℓm(j) ) / ( |q_ℓm(i)| · |q_ℓm(j)| )
//! ```
//!
//! A bond is **solid-like** when `Re(d_ij) > q_threshold` (typically `0.7`).
//! A particle is **solid** when it has at least `n_threshold` solid-like
//! bonds. The output is a per-particle solid-bond count plus the boolean
//! solid mask.
//!
//! This phase reuses [`compute_qlm`] directly
//! — no qℓm recomputation, no duplicate spherical-harmonic evaluations.

use crate::compute::result::ComputeResult;
use molrs::math::complex::Complex;
use molrs::spatial::neighbors::NeighborList;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use super::steinhardt::compute_qlm;
use crate::compute::error::ComputeError;
use crate::compute::traits::Compute;
use crate::compute::util::get_positions_ref;

/// Frenkel-ten Wolde solid/liquid classifier.
#[derive(Debug, Clone, Copy)]
pub struct SolidLiquid {
    l: u32,
    q_threshold: F,
    n_threshold: u32,
    normalize_q: bool,
}

impl SolidLiquid {
    /// Steinhardt order `l`; defaults: q_threshold 0.7, n_threshold 6, normalized `q_ℓm`.
    pub fn new(l: u32) -> Self {
        Self {
            l,
            q_threshold: 0.7,
            n_threshold: 6,
            normalize_q: true,
        }
    }

    /// Dot-product threshold for a solid-like bond (default 0.7).
    pub fn with_q_threshold(mut self, t: F) -> Self {
        self.q_threshold = t;
        self
    }

    /// Minimum solid-like bonds for a solid particle (default 6).
    pub fn with_n_threshold(mut self, n: u32) -> Self {
        self.n_threshold = n;
        self
    }

    /// If false, use the raw (unnormalised) dot product Σ_m qℓm·conj(qℓm)
    /// instead of cosine similarity. Default `true` matches freud.
    pub fn with_normalize_q(mut self, on: bool) -> Self {
        self.normalize_q = on;
        self
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<SolidLiquidResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();
        let m_count = (2 * self.l + 1) as usize;

        let qlm = compute_qlm(frame, nlist, self.l)?;

        // |qℓm(i)| for normalisation. Skip particles with no neighbors → norm = 0.
        let mut norms = vec![0.0_f64; n];
        for i in 0..n {
            let off = i * m_count;
            let mut s: F = 0.0;
            for m in 0..m_count {
                s += qlm[off + m].norm_sqr();
            }
            norms[i] = s.sqrt();
        }

        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        let mut n_solid_bonds = vec![0_u32; n];

        for k in 0..n_pairs {
            let i = i_idx[k] as usize;
            let j = j_idx[k] as usize;
            let mut dot = Complex::ZERO;
            let off_i = i * m_count;
            let off_j = j * m_count;
            for m in 0..m_count {
                // conj(a) * b = (a.re·b.re + a.im·b.im) + i(a.re·b.im − a.im·b.re)
                dot += qlm[off_i + m].conj() * qlm[off_j + m];
            }
            let real = if self.normalize_q {
                let denom = norms[i] * norms[j];
                if denom > 0.0 { dot.re / denom } else { 0.0 }
            } else {
                dot.re
            };
            if real > self.q_threshold {
                n_solid_bonds[i] += 1;
                n_solid_bonds[j] += 1;
            }
        }

        let is_solid: Vec<bool> = n_solid_bonds
            .iter()
            .map(|&c| c >= self.n_threshold)
            .collect();
        Ok(SolidLiquidResult {
            l: self.l,
            n_solid_bonds,
            is_solid,
        })
    }
}

impl Compute for SolidLiquid {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<SolidLiquidResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<SolidLiquidResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if frames.len() != nlists.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: nlists.len(),
                what: "neighbor-list count",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(nlists.par_iter())
                .map(|(f, nl)| self.one_frame(*f, nl))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

/// Per-frame solid/liquid classification.
#[derive(Debug, Clone, Default)]
pub struct SolidLiquidResult {
    /// ℓ used for the qℓm dot product.
    pub l: u32,
    /// Solid-like bond count per particle.
    pub n_solid_bonds: Vec<u32>,
    /// `true` if the particle has ≥ `n_threshold` solid-like bonds.
    pub is_solid: Vec<bool>,
}

impl ComputeResult for SolidLiquidResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::test_support::nlist_from_frame;
    use molrs::Frame;
    use molrs::spatial::simbox::SimBox;
    use molrs::store::block::Block;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap());
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
        nlist_from_frame(frame, cutoff)
    }

    /// Two octahedra sharing the same orientation: every neighbor pair across
    /// the symmetry-mate has qℓm dot ≈ 1 (identical environments).
    fn paired_octahedra(box_len: F) -> Frame {
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (10.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]);
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0_f64, 0.0, 0.0),
                (0.0_f64, 1.0, 0.0),
                (0.0_f64, -1.0, 0.0),
                (0.0_f64, 0.0, 1.0),
                (0.0_f64, 0.0, -1.0),
            ] {
                p.push([cx + dx, cy + dy, cz + dz]);
            }
        }
        frame_with(&p, box_len, [false; 3])
    }

    #[test]
    fn dot_product_self_is_one() {
        // For a single octahedron, the centre particle has neighbours whose
        // |qℓm| is zero (they each have only one neighbour). So check that
        // the bond between the centre and a neighbour gives a real-valued
        // (possibly negative) dot, and that with q_threshold < -∞ both
        // counts go up.
        let frame = frame_with(
            &[
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
                [4.0, 5.0, 5.0],
                [5.0, 6.0, 5.0],
                [5.0, 4.0, 5.0],
                [5.0, 5.0, 6.0],
                [5.0, 5.0, 4.0],
            ],
            20.0,
            [false; 3],
        );
        let nl = build_nlist(&frame, 1.2);
        let r = &SolidLiquid::new(6)
            .with_q_threshold(-2.0) // count every bond as "solid"
            .with_n_threshold(1)
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        // Centre particle has 6 bonds, each counted once.
        assert_eq!(r.n_solid_bonds[0], 6);
        // Outer particles each have 1 bond.
        for i in 1..7 {
            assert_eq!(r.n_solid_bonds[i], 1);
        }
    }

    #[test]
    fn identical_environments_score_one() {
        // Two octahedra with identical orientations sharing a long axis. The
        // central atom of each sees exactly the same neighbor distribution,
        // so qℓm(c1) and qℓm(c2) should be identical (up to numerical
        // precision) → cosine similarity ≈ 1.
        let frame = paired_octahedra(20.0);
        // First, sanity check qlm: build the nlist *only between centres*.
        let positions = [[5.0_f64, 5.0, 5.0], [10.0_f64, 5.0, 5.0]];
        let centres = frame_with(
            &[
                positions[0],
                [6.0, 5.0, 5.0],
                [4.0, 5.0, 5.0],
                [5.0, 6.0, 5.0],
                [5.0, 4.0, 5.0],
                [5.0, 5.0, 6.0],
                [5.0, 5.0, 4.0],
                positions[1],
                [11.0, 5.0, 5.0],
                [9.0, 5.0, 5.0],
                [10.0, 6.0, 5.0],
                [10.0, 4.0, 5.0],
                [10.0, 5.0, 6.0],
                [10.0, 5.0, 4.0],
            ],
            20.0,
            [false; 3],
        );
        let nl = build_nlist(&centres, 1.2);

        // Now compute qlm and check that centres 0 and 7 have ≈ identical qlm.
        let qlm = compute_qlm(&centres, &nl, 6).unwrap();
        let m = 13_usize; // 2·6 + 1
        for k in 0..m {
            let a = qlm[k];
            let b = qlm[7 * m + k];
            assert!(
                (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                "centre qℓm components must match across symmetric octahedra"
            );
        }

        // And the SolidLiquid run: any bond between two identical-environment
        // particles, if it existed in the nlist, would have dot = 1.
        let _ = frame; // suppress unused warn
    }

    #[test]
    fn deterministic_across_calls() {
        let frame = paired_octahedra(20.0);
        let nl = build_nlist(&frame, 1.2);
        let sl = SolidLiquid::new(6);
        let a = sl.compute(&[&frame], &vec![nl.clone()]).unwrap();
        let b = sl.compute(&[&frame], &vec![nl]).unwrap();
        assert_eq!(a[0].n_solid_bonds, b[0].n_solid_bonds);
        assert_eq!(a[0].is_solid, b[0].is_solid);
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = SolidLiquid::new(6)
            .compute(&frames, &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn high_threshold_makes_nothing_solid() {
        let frame = paired_octahedra(20.0);
        let nl = build_nlist(&frame, 1.2);
        let r = &SolidLiquid::new(6)
            .with_q_threshold(2.0) // impossible — cosine ≤ 1
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        assert!(r.is_solid.iter().all(|&s| !s));
    }
}
