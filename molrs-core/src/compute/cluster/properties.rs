use super::result::{ClusterPropsResult, ClusterResult};
use crate::Frame;
use crate::types::F;

use super::super::error::ComputeError;
use super::super::util::get_f_slice;

/// Computes per-cluster properties (center of mass, size) from a
/// [`ClusterResult`].
///
/// This is an example of inter-compute dependency: it takes the output
/// of `Cluster` as an explicit argument, not through framework wiring.
#[derive(Debug, Clone)]
pub struct ClusterProperties;

impl ClusterProperties {
    pub fn new() -> Self {
        Self
    }

    /// Compute center of mass and size for each cluster.
    pub fn compute(
        &self,
        frame: &Frame,
        clusters: &ClusterResult,
    ) -> Result<ClusterPropsResult, ComputeError> {
        let atoms = frame
            .get("atoms")
            .ok_or(ComputeError::MissingBlock { name: "atoms" })?;
        let xs = get_f_slice(atoms, "atoms", "x")?;
        let ys = get_f_slice(atoms, "atoms", "y")?;
        let zs = get_f_slice(atoms, "atoms", "z")?;

        let nc = clusters.num_clusters;
        let mut centers = vec![[0.0 as F; 3]; nc];
        let mut counts = vec![0usize; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            centers[c][0] += xs[i];
            centers[c][1] += ys[i];
            centers[c][2] += zs[i];
            counts[c] += 1;
        }

        for (c, count) in counts.iter().enumerate() {
            if *count > 0 {
                let n = *count as F;
                centers[c][0] /= n;
                centers[c][1] /= n;
                centers[c][2] /= n;
            }
        }

        Ok(ClusterPropsResult {
            centers,
            sizes: counts,
        })
    }
}

impl Default for ClusterProperties {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::traits::PairCompute;
    use super::super::Cluster;
    use super::*;
    use crate::block::Block;
    use crate::neighbors::{LinkCell, NbListAlgo};
    use crate::region::simbox::SimBox;
    use ndarray::{Array1 as A1, array};

    #[test]
    fn center_of_mass_matches_manual() {
        let positions: Vec<[F; 3]> = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [10.0, 12.0, 10.0],
        ];

        let mut block = Block::new();
        block
            .insert(
                "x",
                A1::from_iter(positions.iter().map(|p| p[0])).into_dyn(),
            )
            .unwrap();
        block
            .insert(
                "y",
                A1::from_iter(positions.iter().map(|p| p[1])).into_dyn(),
            )
            .unwrap();
        block
            .insert(
                "z",
                A1::from_iter(positions.iter().map(|p| p[2])).into_dyn(),
            )
            .unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox = Some(
            SimBox::cube(
                30.0,
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [false, false, false],
            )
            .unwrap(),
        );

        let n = positions.len();
        let mut pos = ndarray::Array2::<F>::zeros((n, 3));
        for (i, p) in positions.iter().enumerate() {
            pos[[i, 0]] = p[0];
            pos[[i, 1]] = p[1];
            pos[[i, 2]] = p[2];
        }
        let simbox = frame.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(5.0);
        lc.build(pos.view(), simbox);
        let nbrs = lc.query();

        let cluster = Cluster::new(1);
        let cluster_result = cluster.compute(&frame, nbrs).unwrap();
        assert_eq!(cluster_result.num_clusters, 2);

        let props = ClusterProperties::new()
            .compute(&frame, &cluster_result)
            .unwrap();
        assert_eq!(props.sizes.len(), 2);

        let (c0, c1) = if props.centers[0][0] < 5.0 {
            (0, 1)
        } else {
            (1, 0)
        };

        assert!((props.centers[c0][0] - 1.0).abs() < 1e-5);
        assert!((props.centers[c0][1] - 0.0).abs() < 1e-5);
        assert!((props.centers[c0][2] - 0.0).abs() < 1e-5);
        assert_eq!(props.sizes[c0], 2);

        assert!((props.centers[c1][0] - 10.0).abs() < 1e-5);
        assert!((props.centers[c1][1] - 11.0).abs() < 1e-5);
        assert!((props.centers[c1][2] - 10.0).abs() < 1e-5);
        assert_eq!(props.sizes[c1], 2);
    }
}
