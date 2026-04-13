//! Mass-weighted cluster centers (center of mass) with MIC.

use molrs::frame_access::FrameAccess;
use molrs::types::F;

use super::cluster::ClusterResult;
use super::error::ComputeError;
use super::traits::Compute;
use super::util::{get_positions_generic, mic_disp};

/// Result of center-of-mass computation.
#[derive(Debug, Clone)]
pub struct CenterOfMassResult {
    /// Mass-weighted center per cluster (num_clusters x 3).
    pub centers_of_mass: Vec<[F; 3]>,
    /// Total mass per cluster.
    pub cluster_masses: Vec<F>,
}

/// Computes the center of mass of each cluster using MIC.
///
/// `com_k = r_ref + (1/M_k) * SUM_i m_i * shortest_vector(r_ref, r_i)`
///
/// Masses are optional — defaults to 1.0 for all particles (uniform).
#[derive(Debug, Clone)]
pub struct CenterOfMass {
    masses: Option<Vec<F>>,
}

impl CenterOfMass {
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

impl Default for CenterOfMass {
    fn default() -> Self {
        Self::new()
    }
}

impl Compute for CenterOfMass {
    type Args<'a> = &'a ClusterResult;
    type Output = CenterOfMassResult;

    fn compute<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
    ) -> Result<CenterOfMassResult, ComputeError> {
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

        let mut ref_pos = vec![[0.0 as F; 3]; nc];
        let mut sum_m_delta = vec![[0.0 as F; 3]; nc];
        let mut total_mass = vec![0.0 as F; nc];
        let mut has_ref = vec![false; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = self.masses.as_ref().map_or(1.0 as F, |ms| ms[i]);

            if !has_ref[c] {
                ref_pos[c] = pos;
                has_ref[c] = true;
            }

            let d = mic_disp(simbox, ref_pos[c], pos);
            sum_m_delta[c][0] += m * d[0];
            sum_m_delta[c][1] += m * d[1];
            sum_m_delta[c][2] += m * d[2];
            total_mass[c] += m;
        }

        let mut centers_of_mass = vec![[0.0 as F; 3]; nc];
        for c in 0..nc {
            if total_mass[c] > 0.0 {
                let m = total_mass[c];
                centers_of_mass[c][0] = ref_pos[c][0] + sum_m_delta[c][0] / m;
                centers_of_mass[c][1] = ref_pos[c][1] + sum_m_delta[c][1] / m;
                centers_of_mass[c][2] = ref_pos[c][2] + sum_m_delta[c][2] / m;
            }
        }

        Ok(CenterOfMassResult {
            centers_of_mass,
            cluster_masses: total_mass,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
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

    // --- freud: test_cluster_props_advanced_unweighted (uniform = geometric) ---

    #[test]
    fn uniform_mass_equals_geometric_center() {
        let pos = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        let r = CenterOfMass::new().compute(&frame, &cl).unwrap();
        assert!((r.centers_of_mass[0][0] - 2.0).abs() < 1e-5);
        assert!((r.cluster_masses[0] - 2.0).abs() < 1e-5);
    }

    // --- freud: test_cluster_props_advanced_weighted (COM + masses) ---

    #[test]
    fn weighted_shifts_toward_heavy() {
        // p0(m=3) at [1,1,1], p1(m=1) at [3,1,1]
        // COM_x = (3×1 + 1×3) / 4 = 1.5
        let pos = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        let r = CenterOfMass::new()
            .with_masses(&[3.0, 1.0])
            .compute(&frame, &cl)
            .unwrap();
        assert!((r.centers_of_mass[0][0] - 1.5).abs() < 1e-5);
        assert!((r.cluster_masses[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn two_clusters_weighted() {
        // freud data: p0(m=1),p1(m=2) at [1,1,1]; p2(m=3),p3(m=4) at [1,3,1],[0.9,2.9,1]
        let pos = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.9, 2.9, 1.0],
        ];
        let masses: Vec<F> = vec![1.0, 2.0, 3.0, 4.0];
        let frame = frame_with(&pos, 6.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0, 1, 1]);
        let r = CenterOfMass::new()
            .with_masses(&masses)
            .compute(&frame, &cl)
            .unwrap();

        // cluster 0: COM = [1,1,1], M=3
        assert!((r.centers_of_mass[0][0] - 1.0).abs() < 1e-5);
        assert!((r.cluster_masses[0] - 3.0).abs() < 1e-5);
        // cluster 1: COM_x = (3×1 + 4×0.9)/7 = 6.6/7
        assert!((r.centers_of_mass[1][0] - 6.6 / 7.0).abs() < 1e-4);
        assert!((r.cluster_masses[1] - 7.0).abs() < 1e-5);
    }

    // --- freud: test_cluster_com_periodic (PBC wrapping) ---

    #[test]
    fn periodic_wrapping() {
        // box [0,10), p0(m=1) at 1.0, p1(m=3) at 9.0
        // ref=1.0, MIC(1→9)= -2.0 (wrap: 9-1=8 > 5, so 8-10=-2)
        // COM = 1.0 + (1×0 + 3×(-2))/4 = 1.0 - 1.5 = -0.5 → wrap to 9.5
        let pos = [[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0, [true, true, true]);
        let cl = manual_clusters(&[0, 0]);
        let r = CenterOfMass::new()
            .with_masses(&[1.0, 3.0])
            .compute(&frame, &cl)
            .unwrap();
        // -0.5 mod 10 = 9.5; unrwapped is -0.5. Either is fine.
        let cx = r.centers_of_mass[0][0];
        assert!(
            (cx - 9.5).abs() < 0.5 || (cx - (-0.5)).abs() < 0.5,
            "COM should be near 9.5 (or -0.5), got {cx}"
        );
    }

    #[test]
    fn masses_dimension_mismatch() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        let err = CenterOfMass::new()
            .with_masses(&[1.0]) // wrong length
            .compute(&frame, &cl)
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
