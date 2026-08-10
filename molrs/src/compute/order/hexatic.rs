//! Hexatic order parameter `ψ_k` for 2-D systems.
//!
//! Mirrors `freud.order.Hexatic`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/HexaticTranslational.cc)).
//!
//! For each particle `i` and a chosen integer rotational symmetry `k`
//! (typically 6 for hexagonal lattices, 4 for square, 3 for triangular),
//!
//! ```text
//!   ψ_k(i) = (1/N_i) Σ_{j ∈ neigh(i)} e^{i k θ_{ij}}
//! ```
//!
//! where `θ_{ij}` is the in-plane angle (atan2(dy, dx)) of the bond
//! `r_j − r_i`. Isolated particles get `|ψ_k| = 0`.
//!
//! The z-component of the bond vector is ignored — callers must arrange
//! that the configuration is genuinely planar (typically `Lz = 1`,
//! `pbc.z = false`).

use crate::compute::result::ComputeResult;
use molrs::math::complex::Complex;
use molrs::spatial::neighbors::NeighborList;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use crate::compute::error::ComputeError;
use crate::compute::traits::Compute;
use crate::compute::util::get_positions_ref;

/// Hexatic order parameter calculator.
#[derive(Debug, Clone, Copy)]
pub struct Hexatic {
    k: u32,
}

impl Hexatic {
    /// k-fold symmetry of `ψ_k` (6 = hexatic).
    pub fn new(k: u32) -> Result<Self, ComputeError> {
        if k == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Hexatic::k",
                value: "0".into(),
            });
        }
        Ok(Self { k })
    }

    pub fn k(&self) -> u32 {
        self.k
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<HexaticResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();

        let mut psi = vec![Complex::ZERO; n];
        let mut count = vec![0_u32; n];

        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let vectors = nlist.vectors();
        let n_pairs = nlist.n_pairs();
        // Empty when the list was built with `store_diff=false` — OOB → panic.
        if n_pairs > 0 && vectors.nrows() != n_pairs {
            return Err(ComputeError::BadShape {
                expected: format!(
                    "NeighborList with displacement vectors for {n_pairs} pairs (store_diff=true)"
                ),
                got: format!("{} vector rows", vectors.nrows()),
            });
        }

        // For the j-side of each self-query bond, the bond direction is
        // reversed: θ + π. `exp(i k (θ + π)) = (-1)^k exp(i k θ)`.
        let parity = if self.k & 1 == 0 { 1.0 } else { -1.0 };

        for kp in 0..n_pairs {
            let i = i_idx[kp] as usize;
            let j = j_idx[kp] as usize;
            let dx = vectors[[kp, 0]];
            let dy = vectors[[kp, 1]];
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            let theta = dy.atan2(dx);
            let phase = Complex::from_polar(1.0, self.k as F * theta);
            psi[i] += phase;
            psi[j] += phase.scale(parity);
            count[i] += 1;
            count[j] += 1;
        }
        for i in 0..n {
            if count[i] > 0 {
                let inv = 1.0 / count[i] as F;
                psi[i] = psi[i].scale(inv);
            }
        }
        Ok(HexaticResult { k: self.k, psi })
    }
}

impl Compute for Hexatic {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<HexaticResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<HexaticResult>, ComputeError> {
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

/// Per-frame hexatic order parameter result.
#[derive(Debug, Clone, Default)]
pub struct HexaticResult {
    /// Rotational symmetry order `k`.
    pub k: u32,
    /// Per-particle complex `ψ_k(i)`.
    pub psi: Vec<Complex>,
}

impl ComputeResult for HexaticResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::test_support::nlist_from_frame;
    use molrs::Frame;
    use molrs::spatial::simbox::SimBox;
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
        frame.simbox = Some(
            SimBox::ortho(
                array![box_len, box_len, 1.0_f64],
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [false, false, false],
            )
            .unwrap(),
        );
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
        nlist_from_frame(frame, cutoff)
    }

    /// Six neighbors on a regular hexagon around a centre particle.
    fn hex_environment(box_len: F) -> Frame {
        let c = box_len * 0.5;
        let mut positions = vec![[c, c, 0.0]];
        for k in 0..6 {
            let theta = 2.0_f64 * std::f64::consts::PI * k as F / 6.0;
            positions.push([c + theta.cos(), c + theta.sin(), 0.0]);
        }
        frame_with(&positions, box_len)
    }

    /// Four neighbors on a square around a centre particle.
    fn square_environment(box_len: F) -> Frame {
        let c = box_len * 0.5;
        let positions = [
            [c, c, 0.0],
            [c + 1.0, c, 0.0],
            [c - 1.0, c, 0.0],
            [c, c + 1.0, 0.0],
            [c, c - 1.0, 0.0],
        ];
        frame_with(&positions, box_len)
    }

    #[test]
    fn psi_6_on_perfect_hexagon_is_unity() {
        let frame = hex_environment(20.0);
        let nl = build_nlist(&frame, 1.2);
        let res = Hexatic::new(6)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap();
        // ψ_6 at the centre should have magnitude ≈ 1.
        let psi_center = res[0].psi[0];
        assert!(
            (psi_center.norm() - 1.0).abs() < 1e-10,
            "|ψ_6(centre)| = {} (expected ≈ 1)",
            psi_center.norm(),
        );
    }

    #[test]
    fn psi_4_on_perfect_square_is_unity() {
        let frame = square_environment(20.0);
        let nl = build_nlist(&frame, 1.2);
        let res = Hexatic::new(4)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap();
        let psi_center = res[0].psi[0];
        assert!(
            (psi_center.norm() - 1.0).abs() < 1e-10,
            "|ψ_4(square)| = {} (expected ≈ 1)",
            psi_center.norm(),
        );
    }

    #[test]
    fn psi_6_on_square_is_zero() {
        // For 4 bonds at θ = 0, π/2, π, 3π/2: Σ e^{i 6 θ} = e^0 + e^{i 3π}
        // + e^{i 6π} + e^{i 9π} = 1 − 1 + 1 − 1 = 0.
        let frame = square_environment(20.0);
        let nl = build_nlist(&frame, 1.2);
        let res = Hexatic::new(6)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap();
        let psi_center = res[0].psi[0];
        assert!(
            psi_center.norm() < 1e-10,
            "ψ_6(square) should be 0, got {}",
            psi_center.norm(),
        );
    }

    #[test]
    fn isolated_particle_psi_is_zero() {
        let frame = frame_with(&[[5.0, 5.0, 0.0]], 10.0);
        let nl = build_nlist(&frame, 1.0);
        let res = Hexatic::new(6)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap();
        assert_eq!(res[0].psi[0], Complex::ZERO);
    }

    #[test]
    fn k_zero_is_error() {
        assert!(Hexatic::new(0).is_err());
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = Hexatic::new(6)
            .unwrap()
            .compute(&frames, &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn rotation_invariant_in_magnitude() {
        // Rotate the hexagon by an arbitrary angle and check |ψ_6| unchanged.
        let frame = hex_environment(20.0);
        let nl = build_nlist(&frame, 1.2);
        let a = Hexatic::new(6)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap();
        let mag = a[0].psi[0].norm();

        let c = 10.0_f64;
        let shift = 0.3_f64;
        let mut positions = vec![[c, c, 0.0]];
        for kk in 0..6 {
            let theta = 2.0 * std::f64::consts::PI * kk as F / 6.0 + shift;
            positions.push([c + theta.cos(), c + theta.sin(), 0.0]);
        }
        let frame2 = frame_with(&positions, 20.0);
        let nl2 = build_nlist(&frame2, 1.2);
        let b = Hexatic::new(6)
            .unwrap()
            .compute(&[&frame2], &vec![nl2])
            .unwrap();
        assert!(
            (b[0].psi[0].norm() - mag).abs() < 1e-10,
            "|ψ_6| should be rotation-invariant: {} vs {}",
            mag,
            b[0].psi[0].norm(),
        );
    }

    /// The `frames.len() >= PAR_THRESHOLD` rayon branch must return exactly
    /// what the serial path does, in frame order: two identical frames →
    /// two identical results, each equal to the single-frame result.
    #[test]
    fn parallel_matches_serial() {
        let frame = hex_environment(20.0);
        let nl = build_nlist(&frame, 1.2);
        let hx = Hexatic::new(6).unwrap();
        let solo = hx.compute(&[&frame], &vec![nl.clone()]).unwrap();
        let par = hx
            .compute(&[&frame, &frame], &vec![nl.clone(), nl])
            .unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].psi, solo[0].psi);
        assert_eq!(par[1].psi, solo[0].psi);
    }
}
