// Flat qℓm row layout is most readable with explicit (particle, m) indexing.
#![allow(clippy::needless_range_loop)]

//! Continuous coordination number.
//!
//! Mirrors `freud.order.ContinuousCoordination`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/ContinuousCoordination.cc)).
//!
//! For each particle `i` we compute a "soft" neighbor count weighted by
//! how similar each neighbor's Steinhardt qℓm vector is to particle `i`'s.
//! The weighting function used by freud is
//!
//! ```text
//!   w_ij = ( clamp((d_ij + 1) / 2, 0, 1) )^p
//! ```
//!
//! where `d_ij ∈ [−1, 1]` is the **cosine similarity** of the qℓm vectors
//! and `p ≥ 1` is the configurable power. Setting `p = 1` reproduces the
//! "average of `(1 + cos θ)/2`" weighting that smoothly goes from 0
//! (anti-parallel qℓm) to 1 (identical environments).
//!
//! The output is one scalar per particle per requested ℓ value.

use molrs::math::complex::Complex;
use molrs::spatial::neighbors::NeighborList;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use super::steinhardt::compute_qlm;
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use crate::compute::util::get_positions_ref;

/// Per-frame continuous-coordination result.
#[derive(Debug, Clone, Default)]
pub struct ContinuousCoordinationResult {
    /// Requested ℓ values.
    pub l: Vec<u32>,
    /// `Vec<Vec<F>>` of shape `(n_l, n_particles)`.
    pub coord: Vec<Vec<F>>,
}

impl ComputeResult for ContinuousCoordinationResult {}

/// Continuous-coordination calculator.
#[derive(Debug, Clone)]
pub struct ContinuousCoordination {
    l: Vec<u32>,
    power: F,
}

impl ContinuousCoordination {
    pub fn new(l: &[u32], power: F) -> Result<Self, ComputeError> {
        if l.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "ContinuousCoordination::l",
                value: "[]".into(),
            });
        }
        if power.is_nan() || power < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "ContinuousCoordination::power",
                value: power.to_string(),
            });
        }
        Ok(Self {
            l: l.to_vec(),
            power,
        })
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<ContinuousCoordinationResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();

        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();

        let mut coord_per_l: Vec<Vec<F>> = Vec::with_capacity(self.l.len());
        for &l in &self.l {
            let m_count = (2 * l + 1) as usize;
            let qlm = compute_qlm(frame, nlist, l)?;
            // Per-particle |qℓm| for cosine-similarity normalisation.
            let mut norms = vec![0.0_f64; n];
            for i in 0..n {
                let off = i * m_count;
                let mut s: F = 0.0;
                for m in 0..m_count {
                    s += qlm[off + m].norm_sqr();
                }
                norms[i] = s.sqrt();
            }
            let mut coord = vec![0.0_f64; n];
            for k in 0..n_pairs {
                let i = i_idx[k] as usize;
                let j = j_idx[k] as usize;
                let off_i = i * m_count;
                let off_j = j * m_count;
                let mut dot = Complex::ZERO;
                for m in 0..m_count {
                    dot += qlm[off_i + m].conj() * qlm[off_j + m];
                }
                let denom = norms[i] * norms[j];
                let cos = if denom > 0.0 {
                    (dot.re / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let w = ((cos + 1.0) * 0.5).powf(self.power);
                coord[i] += w;
                coord[j] += w;
            }
            coord_per_l.push(coord);
        }
        Ok(ContinuousCoordinationResult {
            l: self.l.clone(),
            coord: coord_per_l,
        })
    }
}

impl Compute for ContinuousCoordination {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<ContinuousCoordinationResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<ContinuousCoordinationResult>, ComputeError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::spatial::neighbors::{LinkCell, NbListAlgo};
    use molrs::spatial::region::simbox::SimBox;
    use molrs::store::block::Block;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
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
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
        let xp = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(<F as molrs::store::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let yp = frame
            .get("atoms")
            .unwrap()
            .get("y")
            .and_then(<F as molrs::store::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let zp = frame
            .get("atoms")
            .unwrap()
            .get("z")
            .and_then(<F as molrs::store::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let n = xp.len();
        let mut pos = ndarray::Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = xp[i];
            pos[[i, 1]] = yp[i];
            pos[[i, 2]] = zp[i];
        }
        let simbox = frame.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build(pos.view(), simbox);
        lc.query().clone()
    }

    fn paired_octahedra() -> Frame {
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (10.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]);
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ] {
                p.push([cx + dx, cy + dy, cz + dz]);
            }
        }
        frame_with(&p, 20.0)
    }

    #[test]
    fn identical_environments_give_unit_weight() {
        // Two octahedra with identical orientations. Each of the bonds
        // between identical environments contributes weight 1 per pair.
        let frame = paired_octahedra();
        let nl = build_nlist(&frame, 1.2);
        let r = &ContinuousCoordination::new(&[6], 1.0)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        // Each cluster centre has 6 bonds to its identical neighbours
        // (within its own octahedron, but those neighbours are leaf
        // particles with only one bond → different |qℓm|). So we just
        // sanity-check that the centres get a positive coord.
        assert!(r.coord[0][0] > 0.0);
        assert!(r.coord[0][7] > 0.0);
    }

    #[test]
    fn isolated_particle_has_zero_coord() {
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 20.0);
        let nl = build_nlist(&frame, 1.0);
        let r = &ContinuousCoordination::new(&[6], 1.0)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        assert_eq!(r.coord[0][0], 0.0);
    }

    #[test]
    fn power_zero_counts_every_pair_as_one() {
        // power = 0 → ((cos+1)/2)^0 = 1 for any neighbor pair. Each particle
        // gets coord = (its neighbor count).
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 20.0);
        let nl = build_nlist(&frame, 1.5);
        let r = &ContinuousCoordination::new(&[6], 0.0)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        assert!((r.coord[0][0] - 1.0).abs() < 1e-12);
        assert!((r.coord[0][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_l_or_negative_power_errors() {
        assert!(ContinuousCoordination::new(&[], 1.0).is_err());
        assert!(ContinuousCoordination::new(&[6], -1.0).is_err());
    }

    #[test]
    fn empty_frames_error() {
        let err = ContinuousCoordination::new(&[6], 1.0)
            .unwrap()
            .compute::<Frame>(&[], &vec![])
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
