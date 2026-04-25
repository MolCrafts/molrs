mod result;

pub use result::ClusterResult;

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use ndarray::Array1;

use crate::error::ComputeError;
use crate::traits::Compute;

/// Distance-based cluster analysis using BFS on the neighbor graph.
///
/// Two particles belong to the same cluster if they are connected (directly
/// or transitively) within the neighbor cutoff. Uses CSR adjacency for
/// cache-friendly traversal. One [`ClusterResult`] per input frame.
#[derive(Debug, Clone)]
pub struct Cluster {
    min_cluster_size: usize,
}

impl Cluster {
    pub fn new(min_cluster_size: usize) -> Self {
        Self { min_cluster_size }
    }

    fn cluster_one<FA: FrameAccess>(
        &self,
        frame: &FA,
        neighbors: &NeighborList,
    ) -> Result<ClusterResult, ComputeError> {
        let n = frame
            .visit_block("atoms", |b| b.nrows().unwrap_or(0))
            .ok_or(ComputeError::MissingBlock { name: "atoms" })?;

        if n == 0 {
            return Ok(ClusterResult {
                cluster_idx: Array1::zeros(0),
                num_clusters: 0,
                cluster_sizes: vec![],
            });
        }

        let n_pairs = neighbors.n_pairs();
        let query_indices = neighbors.query_point_indices();
        let point_indices = neighbors.point_indices();

        let mut degree = vec![0u32; n];
        for k in 0..n_pairs {
            degree[query_indices[k] as usize] += 1;
            degree[point_indices[k] as usize] += 1;
        }

        let mut offsets = vec![0usize; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + degree[i] as usize;
        }

        let mut flat_adj = vec![0u32; 2 * n_pairs];
        let mut cursor = offsets[..n].to_vec();
        for k in 0..n_pairs {
            let i = query_indices[k] as usize;
            let j = point_indices[k] as usize;
            flat_adj[cursor[i]] = j as u32;
            cursor[i] += 1;
            flat_adj[cursor[j]] = i as u32;
            cursor[j] += 1;
        }

        let mut cluster_idx = vec![-1_i64; n];
        let mut current_id: i64 = 0;
        let mut cluster_sizes: Vec<usize> = Vec::new();
        let mut queue: Vec<usize> = Vec::new();

        for start in 0..n {
            if cluster_idx[start] >= 0 {
                continue;
            }

            queue.clear();
            queue.push(start);
            cluster_idx[start] = current_id;
            let mut size = 0;
            let mut head = 0;

            while head < queue.len() {
                let node = queue[head];
                head += 1;
                size += 1;

                for &nbr in &flat_adj[offsets[node]..offsets[node + 1]] {
                    let neighbor = nbr as usize;
                    if cluster_idx[neighbor] < 0 {
                        cluster_idx[neighbor] = current_id;
                        queue.push(neighbor);
                    }
                }
            }

            cluster_sizes.push(size);
            current_id += 1;
        }

        if self.min_cluster_size > 1 {
            let mut remap = vec![-1_i64; cluster_sizes.len()];
            let mut new_id: i64 = 0;
            let mut new_sizes = Vec::new();

            for (old_id, &size) in cluster_sizes.iter().enumerate() {
                if size >= self.min_cluster_size {
                    remap[old_id] = new_id;
                    new_sizes.push(size);
                    new_id += 1;
                }
            }

            for cid in cluster_idx.iter_mut() {
                if *cid >= 0 {
                    *cid = remap[*cid as usize];
                }
            }

            cluster_sizes = new_sizes;
        }

        let num_clusters = cluster_sizes.len();

        Ok(ClusterResult {
            cluster_idx: Array1::from_vec(cluster_idx),
            num_clusters,
            cluster_sizes,
        })
    }
}

impl Compute for Cluster {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<ClusterResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        neighbors: &'a Vec<NeighborList>,
    ) -> Result<Vec<ClusterResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if neighbors.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: neighbors.len(),
                what: "neighbor-list count",
            });
        }
        // Cluster has heavy per-frame work (CSR build + BFS, ~100 µs per
        // 5k-atom frame), so rayon pays off from 2 frames onward.
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(neighbors.par_iter())
                .map(|(frame, nlist)| self.cluster_one(*frame, nlist))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (frame, nlist) in frames.iter().zip(neighbors.iter()) {
            out.push(self.cluster_one(*frame, nlist)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
    use molrs::neighbors::{LinkCell, NbListAlgo};
    use molrs::region::simbox::SimBox;
    use molrs::types::F;
    use ndarray::{Array1 as A1, array};

    fn make_frame_with_positions(positions: &[[F; 3]], box_len: F) -> Frame {
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

    fn build_neighbors(frame: &Frame, cutoff: F) -> NeighborList {
        let atoms = frame.get("atoms").unwrap();
        let xs = super::super::util::get_f_slice(atoms, "atoms", "x").unwrap();
        let ys = super::super::util::get_f_slice(atoms, "atoms", "y").unwrap();
        let zs = super::super::util::get_f_slice(atoms, "atoms", "z").unwrap();
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
        lc.query().clone()
    }

    fn cluster_single(frame: &Frame, nlist: NeighborList, min: usize) -> ClusterResult {
        let out = Cluster::new(min).compute(&[frame], &vec![nlist]).unwrap();
        assert_eq!(out.len(), 1);
        out.into_iter().next().unwrap()
    }

    #[test]
    fn two_separated_groups() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [1.0, 1.5, 1.0],
            [8.0, 8.0, 8.0],
            [8.5, 8.0, 8.0],
            [8.0, 8.5, 8.0],
        ];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 1);

        assert_eq!(result.num_clusters, 2);
        assert_eq!(result.cluster_idx[0], result.cluster_idx[1]);
        assert_eq!(result.cluster_idx[0], result.cluster_idx[2]);
        assert_eq!(result.cluster_idx[3], result.cluster_idx[4]);
        assert_eq!(result.cluster_idx[3], result.cluster_idx[5]);
        assert_ne!(result.cluster_idx[0], result.cluster_idx[3]);
    }

    #[test]
    fn min_cluster_size_filters_small() {
        let positions = [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [8.0, 8.0, 8.0]];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 2);

        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_idx[2], -1);
        assert!(result.cluster_idx[0] >= 0);
    }

    #[test]
    fn single_cluster() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [1.0, 1.5, 1.0],
            [1.5, 1.5, 1.0],
        ];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 1);

        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn collinear_four_particles() {
        let positions = [
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
        ];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 2.01);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn all_isolated() {
        let positions = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 0.5);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.num_clusters, 3);
        for &s in &result.cluster_sizes {
            assert_eq!(s, 1);
        }
        assert_ne!(result.cluster_idx[0], result.cluster_idx[1]);
        assert_ne!(result.cluster_idx[1], result.cluster_idx[2]);
    }

    #[test]
    fn coincident_particles() {
        let positions = [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 0.5);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 3);
    }

    #[test]
    fn empty_frame() {
        let frame = make_frame_with_positions(&[], 10.0);
        let nbrs = build_neighbors(&frame, 1.0);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.num_clusters, 0);
        assert!(result.cluster_idx.is_empty());
    }

    #[test]
    fn single_particle() {
        let positions = [[5.0, 5.0, 5.0]];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 1.0);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_idx[0], 0);
    }

    #[test]
    fn transitive_chain() {
        let positions = [
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
        ];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 1.5);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn pbc_wrapping_cluster() {
        let positions = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
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
                10.0,
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [true, true, true],
            )
            .unwrap(),
        );
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.num_clusters, 1);
    }

    #[test]
    fn multi_frame_runs_per_frame() {
        let f1 = make_frame_with_positions(&[[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]], 10.0);
        let f2 = make_frame_with_positions(&[[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]], 10.0);
        let n1 = build_neighbors(&f1, 1.0);
        let n2 = build_neighbors(&f2, 1.0);
        let out = Cluster::new(1).compute(&[&f1, &f2], &vec![n1, n2]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].num_clusters, 1);
        assert_eq!(out[1].num_clusters, 2);
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = Cluster::new(1)
            .compute(&frames, &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
