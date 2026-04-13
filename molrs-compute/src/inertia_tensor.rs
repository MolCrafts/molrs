//! Moment of inertia tensor computation for clusters.

use molrs::frame_access::FrameAccess;
use molrs::types::F;

use super::cluster::ClusterResult;
use super::error::ComputeError;
use super::traits::Compute;
use super::util::{get_positions_generic, mic_disp};

/// Computes the moment of inertia tensor for each cluster.
///
/// Neither mass- nor count-normalized (matches freud convention):
///
/// `I_k[a][b] = SUM_i m_i * (|s_i|^2 * delta_ab - s_i[a] * s_i[b])`
///
/// where `s_i = shortest_vector(com_k, r_i)` is the MIC displacement
/// from the center of mass, and `delta_ab` is the Kronecker delta.
#[derive(Debug, Clone)]
pub struct InertiaTensor {
    masses: Option<Vec<F>>,
}

impl InertiaTensor {
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

impl Default for InertiaTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for InertiaTensor {
    type Args<'a> = &'a ClusterResult;
    type Output = Vec<[[F; 3]; 3]>;

    fn compute<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
    ) -> Result<Vec<[[F; 3]; 3]>, ComputeError> {
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

        // Accumulate inertia tensor
        let mut tensors = vec![[[0.0 as F; 3]; 3]; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = self.masses.as_ref().map_or(1.0 as F, |ms| ms[i]);
            let s = mic_disp(simbox, com_result.centers_of_mass[c], pos);
            let s_sq = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];

            for a in 0..3 {
                for b in 0..3 {
                    let delta = if a == b { 1.0 as F } else { 0.0 as F };
                    tensors[c][a][b] += m * (s_sq * delta - s[a] * s[b]);
                }
            }
        }

        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
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

    // --- freud: inertia == 0 for coincident / single particle ---

    #[test]
    fn zero_for_coincident() {
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let t = InertiaTensor::new()
            .with_masses(&[2.0, 5.0])
            .compute(&frame, &cl)
            .unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!(t[0][a][b].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn single_particle_zero() {
        let pos = [[5.0, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0]);
        let t = InertiaTensor::new()
            .with_masses(&[3.0])
            .compute(&frame, &cl)
            .unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!(t[0][a][b].abs() < 1e-10);
            }
        }
    }

    // --- freud: test_cluster_props_advanced_weighted (inertia tensor) ---

    #[test]
    fn weighted_off_center() {
        // p0(m=3) at [1,3,1], p1(m=4) at [0.9,2.9,1]
        // COM_x = (3+3.6)/7 = 6.6/7, COM_y = (9+11.6)/7 = 20.6/7
        // s0 = [1-6.6/7, 3-20.6/7, 0] = [0.4/7, 0.4/7, 0]
        // s1 = [0.9-6.6/7, 2.9-20.6/7, 0] = [-0.3/7, -0.3/7, 0]
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let masses: Vec<F> = vec![3.0, 4.0];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let t = InertiaTensor::new()
            .with_masses(&masses)
            .compute(&frame, &cl)
            .unwrap();

        let expected: [[f64; 3]; 3] = [
            [0.0171429, -0.0171429, 0.0],
            [-0.0171429, 0.0171429, 0.0],
            [0.0, 0.0, 0.0342857],
        ];
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (t[0][a][b] as f64 - expected[a][b]).abs() < 1e-4,
                    "I[{a}][{b}] = {}, expected {}",
                    t[0][a][b],
                    expected[a][b]
                );
            }
        }
    }

    #[test]
    fn symmetric() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0, 0]);
        let t = InertiaTensor::new().compute(&frame, &cl).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!((t[0][a][b] - t[0][b][a]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn two_along_x() {
        // p0=(2,3,3), p1=(4,3,3), COM=(3,3,3), s=(-1,0,0),(1,0,0), m=1
        // I_xx = 2*(0+0) = 0, I_yy = I_zz = 2*1 = 2
        let pos = [[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0]);
        let t = InertiaTensor::new().compute(&frame, &cl).unwrap();
        assert!(t[0][0][0].abs() < 1e-5);
        assert!((t[0][1][1] - 2.0).abs() < 1e-5);
        assert!((t[0][2][2] - 2.0).abs() < 1e-5);
    }
}
