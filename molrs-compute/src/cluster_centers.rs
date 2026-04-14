//! Geometric cluster centers computed with minimum image convention.

use molrs::frame_access::FrameAccess;
use molrs::types::F;

use super::cluster::ClusterResult;
use super::error::ComputeError;
use super::traits::Compute;
use super::util::{get_positions_generic, mic_disp};

/// Computes the geometric center of each cluster using the minimum image
/// convention (MIC) for periodic systems.
///
/// Algorithm: for each cluster, pick the first particle as reference,
/// accumulate MIC-corrected displacements, then average.
///
/// `center_k = r_ref + (1/N_k) * SUM_i shortest_vector(r_ref, r_i)`
#[derive(Debug, Clone)]
pub struct ClusterCenters;

impl ClusterCenters {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ClusterCenters {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for ClusterCenters {
    type Args<'a> = &'a ClusterResult;
    type Output = Vec<[F; 3]>;

    fn compute<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
    ) -> Result<Vec<[F; 3]>, ComputeError> {
        let (xs_vec, ys_vec, zs_vec) = get_positions_generic(frame)?;
        let xs = &xs_vec[..];
        let ys = &ys_vec[..];
        let zs = &zs_vec[..];
        let simbox = frame.simbox_ref();
        let nc = clusters.num_clusters;

        let mut ref_pos = vec![[0.0 as F; 3]; nc];
        let mut sum_delta = vec![[0.0 as F; 3]; nc];
        let mut counts = vec![0usize; nc];
        let mut has_ref = vec![false; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];

            if !has_ref[c] {
                ref_pos[c] = pos;
                has_ref[c] = true;
            }

            let d = mic_disp(simbox, ref_pos[c], pos);
            sum_delta[c][0] += d[0];
            sum_delta[c][1] += d[1];
            sum_delta[c][2] += d[2];
            counts[c] += 1;
        }

        let mut centers = vec![[0.0 as F; 3]; nc];
        for c in 0..nc {
            if counts[c] > 0 {
                let n = counts[c] as F;
                centers[c][0] = ref_pos[c][0] + sum_delta[c][0] / n;
                centers[c][1] = ref_pos[c][1] + sum_delta[c][1] / n;
                centers[c][2] = ref_pos[c][2] + sum_delta[c][2] / n;
            }
        }

        Ok(centers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::Cluster;
    use crate::util::get_positions;
    use molrs::Frame;
    use molrs::block::Block;
    use molrs::neighbors::{LinkCell, NbListAlgo};
    use molrs::region::simbox::SimBox;
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

    fn clusters_via_nlist(frame: &Frame, cutoff: F) -> ClusterResult {
        let (xs, ys, zs): (&[F], &[F], &[F]) = get_positions(frame).unwrap();
        let n = xs.len();
        let mut pos = ndarray::Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = xs[i];
            pos[[i, 1]] = ys[i];
            pos[[i, 2]] = zs[i];
        }
        let simbox = frame.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build(pos.view(), simbox);
        Cluster::new(1).compute(frame, lc.query()).unwrap()
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

    // --- freud: test_cluster_props_advanced_unweighted (centers) ---

    #[test]
    fn coincident_and_offset_clusters() {
        // p0,p1 at same spot; p2,p3 nearby
        let pos = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.9, 2.9, 1.0],
        ];
        let frame = frame_with(&pos, 6.0, [false, false, false]);
        let cl = clusters_via_nlist(&frame, 0.5);
        assert_eq!(cl.num_clusters, 2);

        let centers = ClusterCenters::new().compute(&frame, &cl).unwrap();
        let (ca, cb) = if cl.cluster_idx[0] == 0 {
            (0, 1)
        } else {
            (1, 0)
        };

        // cluster A: coincident → [1, 1, 1]
        assert!((centers[ca][0] - 1.0).abs() < 1e-5);
        assert!((centers[ca][1] - 1.0).abs() < 1e-5);

        // cluster B: mean([1,3,1], [0.9,2.9,1]) = [0.95, 2.95, 1]
        assert!((centers[cb][0] - 0.95).abs() < 1e-5);
        assert!((centers[cb][1] - 2.95).abs() < 1e-5);
    }

    // --- freud: test_cluster_com_periodic ---

    #[test]
    fn mic_wrapping_across_boundary() {
        // box [0, 10), PBC. p0 at 0.5, p1 at 9.5 → MIC dist = 1.0
        let pos = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0, [true, true, true]);
        let cl = clusters_via_nlist(&frame, 2.0);
        assert_eq!(cl.num_clusters, 1);

        let centers = ClusterCenters::new().compute(&frame, &cl).unwrap();
        // ref=0.5, MIC(0.5→9.5) = -1.0, mean_delta = (-1+0)/2 = -0.5
        // center = 0.5 + (-0.5) = 0.0  (or 10.0, same in PBC)
        let cx = centers[0][0];
        assert!(
            !(1.0..=9.0).contains(&cx),
            "center should wrap near boundary, got {cx}"
        );
    }

    // --- single particle ---

    #[test]
    fn single_particle() {
        let pos = [[3.0, 4.0, 5.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0]);
        let centers = ClusterCenters::new().compute(&frame, &cl).unwrap();
        assert!((centers[0][0] - 3.0).abs() < 1e-5);
        assert!((centers[0][1] - 4.0).abs() < 1e-5);
        assert!((centers[0][2] - 5.0).abs() < 1e-5);
    }

    // --- empty ---

    #[test]
    fn empty() {
        let pos: Vec<[F; 3]> = vec![];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[]);
        let centers = ClusterCenters::new().compute(&frame, &cl).unwrap();
        assert!(centers.is_empty());
    }

    // --- no simbox (free boundary) ---

    #[test]
    fn free_boundary() {
        let pos = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]];
        let mut frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        frame.simbox = None;
        let centers = ClusterCenters::new().compute(&frame, &cl).unwrap();
        assert!((centers[0][0] - 2.0).abs() < 1e-5);
    }
}
