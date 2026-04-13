//! Radius of gyration computation for clusters.

use molrs::frame_access::FrameAccess;
use molrs::types::F;

use super::cluster::ClusterResult;
use super::error::ComputeError;
use super::traits::Compute;
use super::util::{get_positions_generic, mic_disp};

/// Computes the radius of gyration for each cluster.
///
/// `R_g_k = sqrt( (1/M_k) * SUM_i m_i * |s_i|^2 )`
///
/// where `s_i = shortest_vector(com_k, r_i)` is the MIC displacement
/// from the center of mass.
#[derive(Debug, Clone)]
pub struct RadiusOfGyration {
    masses: Option<Vec<F>>,
}

impl RadiusOfGyration {
    pub fn new() -> Self {
        Self { masses: None }
    }

    /// Set per-particle masses. Length must match the number of atoms.
    pub fn with_masses(self, masses: &[F]) -> Self {
        Self {
            masses: Some(masses.to_vec()),
        }
    }
}

impl Default for RadiusOfGyration {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for RadiusOfGyration {
    type Args<'a> = &'a ClusterResult;
    type Output = Vec<F>;

    fn compute<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
    ) -> Result<Vec<F>, ComputeError> {
        let (xs_vec, ys_vec, zs_vec) = get_positions_generic(frame)?;
        let xs = &xs_vec[..];
        let ys = &ys_vec[..];
        let zs = &zs_vec[..];
        let n = xs.len();

        if let Some(ref ms) = self.masses
            && ms.len() != n
        {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: ms.len(),
            });
        }

        let simbox = frame.simbox_ref();
        let nc = clusters.num_clusters;

        // Compute centers of mass
        let com_calc = super::center_of_mass::CenterOfMass::new();
        let com_calc = if let Some(ref ms) = self.masses {
            com_calc.with_masses(ms)
        } else {
            com_calc
        };
        let com_result = com_calc.compute(frame, clusters)?;

        // Accumulate sum of m_i * |s_i|^2
        let mut rg_sum = vec![0.0 as F; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = self.masses.as_ref().map_or(1.0 as F, |ms| ms[i]);
            let s = mic_disp(simbox, com_result.centers_of_mass[c], pos);
            let s_sq = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
            rg_sum[c] += m * s_sq;
        }

        let mut radii = vec![0.0 as F; nc];
        for c in 0..nc {
            if com_result.cluster_masses[c] > 0.0 {
                radii[c] = (rg_sum[c] / com_result.cluster_masses[c]).sqrt();
            }
        }

        Ok(radii)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
    use crate::inertia_tensor::InertiaTensor;
    use molrs::region::simbox::SimBox;
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
            SimBox::cube(
                box_len,
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [false, false, false],
            )
            .unwrap(),
        );
        frame
    }

    fn manual_clusters(idx: &[i64]) -> ClusterResult {
        let nc = (*idx.iter().max().unwrap_or(&-1) + 1).max(0) as usize;
        let mut sizes = vec![0usize; nc];
        for &c in idx {
            if c >= 0 {
                sizes[c as usize] += 1;
            }
        }
        ClusterResult {
            cluster_idx: ndarray::Array1::from_vec(idx.to_vec()),
            num_clusters: nc,
            cluster_sizes: sizes,
        }
    }

    // --- freud: rg == 0 for coincident / single ---

    #[test]
    fn zero_for_coincident() {
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let rg = RadiusOfGyration::new()
            .with_masses(&[2.0, 5.0])
            .compute(&frame, &cl)
            .unwrap();
        assert!(rg[0].abs() < 1e-10);
    }

    #[test]
    fn single_particle_zero() {
        let rg = RadiusOfGyration::new()
            .compute(
                &frame_with(&[[5.0, 5.0, 5.0]], 10.0),
                &manual_clusters(&[0]),
            )
            .unwrap();
        assert!(rg[0].abs() < 1e-10);
    }

    // --- freud: test_cluster_props_advanced_weighted (R_g) ---

    #[test]
    fn weighted_off_center() {
        // p0(m=3) at [1,3,1], p1(m=4) at [0.9,2.9,1]
        // COM = [6.6/7, 20.6/7, 1], M=7
        // s0 = [0.4/7, 0.4/7, 0], |s0|² = 2×(0.4/7)² = 0.006531
        // s1 = [-0.3/7, -0.3/7, 0], |s1|² = 2×(0.3/7)² = 0.003673
        // R_g = sqrt( (3×0.006531 + 4×0.003673) / 7 ) = 0.06999
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let rg = RadiusOfGyration::new()
            .with_masses(&[3.0, 4.0])
            .compute(&frame, &cl)
            .unwrap();
        assert!(
            (rg[0] as f64 - 0.0699854212).abs() < 1e-4,
            "rg = {}, expected ~0.0700",
            rg[0]
        );
    }

    #[test]
    fn two_equal_mass_along_x() {
        // p0=(2,3,3), p1=(4,3,3), COM=(3,3,3), |s|²=1 each
        // R_g = sqrt( (1+1)/2 ) = 1.0
        let pos = [[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0]);
        let rg = RadiusOfGyration::new().compute(&frame, &cl).unwrap();
        assert!((rg[0] - 1.0).abs() < 1e-5);
    }

    // --- identity: R_g = sqrt(trace(I) / 2M) ---

    #[test]
    fn rg_equals_trace_of_inertia_over_2m() {
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let masses: Vec<F> = vec![3.0, 4.0];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);

        let rg = RadiusOfGyration::new()
            .with_masses(&masses)
            .compute(&frame, &cl)
            .unwrap();
        let inertia = InertiaTensor::new()
            .with_masses(&masses)
            .compute(&frame, &cl)
            .unwrap();
        let com = super::super::center_of_mass::CenterOfMass::new()
            .with_masses(&masses)
            .compute(&frame, &cl)
            .unwrap();

        let trace = inertia[0][0][0] + inertia[0][1][1] + inertia[0][2][2];
        let rg_from_trace = (trace / (2.0 * com.cluster_masses[0])).sqrt();
        assert!(
            (rg[0] - rg_from_trace).abs() < 1e-5,
            "rg={}, from_trace={}",
            rg[0],
            rg_from_trace
        );
    }
}
