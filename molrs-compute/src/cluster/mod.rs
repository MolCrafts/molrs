mod result;

pub use result::ClusterResult;

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use ndarray::Array1;

use super::error::ComputeError;
use super::traits::Compute;

/// Distance-based cluster analysis using BFS on the neighbor graph.
///
/// Two particles belong to the same cluster if they are connected
/// (directly or transitively) within the neighbor cutoff.
/// Uses CSR (Compressed Sparse Row) adjacency for cache-friendly traversal.
///
/// Only supports self-query neighbor lists (will return an error for cross-query).
#[derive(Debug, Clone)]
pub struct Cluster {
    min_cluster_size: usize,
}

impl Cluster {
    pub fn new(min_cluster_size: usize) -> Self {
        Self { min_cluster_size }
    }
}

impl Compute for Cluster {
    type Args<'a> = &'a NeighborList;
    type Output = ClusterResult;

    fn compute<FA: FrameAccess>(
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

        // Build CSR adjacency (3 flat allocations instead of N Vec allocations)
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

        // BFS to assign cluster IDs
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

        // Relabel: clusters smaller than min_cluster_size get ID = -1
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

        let cluster = Cluster::new(1);
        let result = cluster.compute(&frame, &nbrs).unwrap();

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

        let cluster = Cluster::new(2);
        let result = cluster.compute(&frame, &nbrs).unwrap();

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

        let cluster = Cluster::new(1);
        let result = cluster.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    // --- freud: test_query_ball — 4 collinear particles, count neighbors ---

    #[test]
    fn collinear_four_particles() {
        // freud data: [0,0,0], [1,0,0], [3,0,0], [2,0,0], cutoff=2.01
        // p0 neighbors: p1(d=1), p3(d=2) → 2 bonds (self-query i<j)
        // p1 neighbors: p3(d=1), p2(d=2) → but as i<j pairs:
        // Pairs: (0,1)=1, (0,3)=2, (1,3)=1, (1,2)=2, (2,3)=1 → 5 pairs
        // All connected transitively → 1 cluster
        let positions = [
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
        ];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 2.01);

        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    // --- freud: all isolated → N clusters of size 1 ---

    #[test]
    fn all_isolated() {
        let positions = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 0.5);

        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        assert_eq!(result.num_clusters, 3);
        for &s in &result.cluster_sizes {
            assert_eq!(s, 1);
        }
        // all different IDs
        assert_ne!(result.cluster_idx[0], result.cluster_idx[1]);
        assert_ne!(result.cluster_idx[1], result.cluster_idx[2]);
    }

    // --- freud: coincident particles → same cluster ---

    #[test]
    fn coincident_particles() {
        let positions = [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 0.5);

        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 3);
    }

    // --- empty frame ---

    #[test]
    fn empty_frame() {
        let frame = make_frame_with_positions(&[], 10.0);
        let nbrs = build_neighbors(&frame, 1.0);
        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        assert_eq!(result.num_clusters, 0);
        assert!(result.cluster_idx.is_empty());
    }

    // --- single particle ---

    #[test]
    fn single_particle() {
        let positions = [[5.0, 5.0, 5.0]];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 1.0);
        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_idx[0], 0);
    }

    // --- transitive connectivity (chain) ---

    #[test]
    fn transitive_chain() {
        // A-B-C-D: each pair within cutoff, but A and D not directly connected
        // A=1, B=2, C=3, D=4 along x, cutoff=1.5
        let positions = [
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
        ];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 1.5);

        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        assert_eq!(result.num_clusters, 1, "chain should form 1 cluster");
        assert_eq!(result.cluster_sizes[0], 4);
    }

    // --- PBC wrapping: two particles near opposite boundaries ---

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
        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        // MIC distance = 1.0 < 2.0 → same cluster
        assert_eq!(result.num_clusters, 1);
    }

    // --- cluster_sizes sum == total assigned particles ---

    #[test]
    fn sizes_sum_to_n() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [9.0, 9.0, 9.0],
        ];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 1.0);

        let result = Cluster::new(1).compute(&frame, &nbrs).unwrap();
        let total: usize = result.cluster_sizes.iter().sum();
        assert_eq!(total, 5);
    }
}
