//! Gyration tensor computation for clusters.

#![allow(clippy::needless_range_loop)]

use molrs::frame_access::FrameAccess;
use molrs::types::F;

use super::cluster::ClusterResult;
use super::error::ComputeError;
use super::traits::Compute;
use super::util::{get_positions_generic, mic_disp};

/// Computes the gyration tensor for each cluster.
///
/// The gyration tensor is particle-count-normalized:
///
/// `G_k[a][b] = (1/N_k) * SUM_i s_i[a] * s_i[b]`
///
/// where `s_i = shortest_vector(center_k, r_i)` is the MIC displacement
/// from the geometric center.
#[derive(Debug, Clone)]
pub struct GyrationTensor;

impl GyrationTensor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GyrationTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for GyrationTensor {
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
        let simbox = frame.simbox_ref();
        let nc = clusters.num_clusters;

        // Pass 1: compute geometric centers (MIC-aware)
        let centers = super::cluster_centers::ClusterCenters::new().compute(frame, clusters)?;

        // Pass 2: accumulate gyration tensor
        let mut tensors = vec![[[0.0 as F; 3]; 3]; nc];
        let mut counts = vec![0usize; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let s = mic_disp(simbox, centers[c], pos);

            for a in 0..3 {
                for b in 0..3 {
                    tensors[c][a][b] += s[a] * s[b];
                }
            }
            counts[c] += 1;
        }

        // Normalize by particle count
        for (tensor, &count) in tensors.iter_mut().zip(counts.iter()) {
            if count > 0 {
                let n = count as F;
                for row in tensor.iter_mut() {
                    for val in row.iter_mut() {
                        *val /= n;
                    }
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

    // --- freud: gyration == 0 for coincident particles ---

    #[test]
    fn zero_for_coincident() {
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let g = GyrationTensor::new().compute(&frame, &cl).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!(g[0][a][b].abs() < 1e-10);
            }
        }
    }

    // --- freud: test_cluster_props_advanced_unweighted (gyration tensor) ---

    #[test]
    fn off_center_cluster() {
        // center = mean([1,3,1],[0.9,2.9,1]) = [0.95, 2.95, 1]
        // s0 = [0.05, 0.05, 0], s1 = [-0.05, -0.05, 0]
        // G = (1/2)(s0⊗s0 + s1⊗s1) = [[.0025,.0025,0],[.0025,.0025,0],[0,0,0]]
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let g = GyrationTensor::new().compute(&frame, &cl).unwrap();
        let expected = [
            [0.0025, 0.0025, 0.0],
            [0.0025, 0.0025, 0.0],
            [0.0, 0.0, 0.0],
        ];
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (g[0][a][b] - expected[a][b] as F).abs() < 1e-5,
                    "G[{a}][{b}] = {}, expected {}",
                    g[0][a][b],
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
        let g = GyrationTensor::new().compute(&frame, &cl).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!((g[0][a][b] - g[0][b][a]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn two_particles_along_x() {
        // at (2,3,3) and (4,3,3), center=(3,3,3)
        // s=(-1,0,0) and (1,0,0)
        // G_xx = (1/2)(1+1)=1, rest=0
        let pos = [[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0]);
        let g = GyrationTensor::new().compute(&frame, &cl).unwrap();
        assert!((g[0][0][0] - 1.0).abs() < 1e-5);
        assert!(g[0][1][1].abs() < 1e-5);
        assert!(g[0][0][1].abs() < 1e-5);
    }

    #[test]
    fn single_particle_zero() {
        let pos = [[5.0, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0]);
        let g = GyrationTensor::new().compute(&frame, &cl).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!(g[0][a][b].abs() < 1e-10);
            }
        }
    }
}
