// Tight 3-coord AABB loops read naturally with index-based access.
#![allow(clippy::needless_range_loop)]

//! Axis-Aligned Bounding-Box tree neighbor search.
//!
//! Mirrors `freud.locality.AABBQuery`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/AABBQuery.cc)).
//!
//! Builds a balanced binary BVH over input points: each leaf wraps a
//! single point, each internal node owns the union AABB of its subtree.
//! A radial query descends the tree, pruning subtrees whose closest box
//! point is farther than the cutoff. Average query cost is
//! `O(log N + n_hits)`.
//!
//! # PBC handling — MIC-based, no ghost atoms
//!
//! Periodicity is handled the same way [`LinkCell`](super::linkcell::LinkCell)
//! does it: the tree is built **only on the original `N` points**, never
//! on a ghost-expanded set. For each query point we enumerate the small
//! set of lattice-image shifts that could bring a tree point within
//! `cutoff` of the query, run a (non-periodic) ball query for each shift,
//! then pin every hit's displacement to the canonical minimum-image vector
//! returned by [`SimBox::shortest_vector_impl`].
//!
//! Number of image shifts per axis is `ceil(cutoff / L_axis)`, so:
//! - `cutoff < L_axis / 2` (the typical MD case): only the trivial
//!   `(0, 0, 0)` shift — exactly one tree query per particle.
//! - `cutoff ≥ L_axis / 2`: up to `27` shifts in 3-D.
//!
//! This keeps memory bounded by the original `N` points (no
//! `O(N · n_images)` ghost copies) and aligns the PBC story with the rest
//! of `molrs-core::neighbors`: every algorithm gets its periodicity from
//! `SimBox`, never from `PeriodicBuffer`. `PeriodicBuffer` remains a
//! standalone user-facing utility for explicit ghost-atom workflows
//! (visualisation, exports).

use std::collections::HashSet;

use crate::spatial::neighbors::{NbListAlgo, NeighborList, QueryMode};
use crate::spatial::simbox::{BoxKind, SimBox};
use crate::types::{F, FNx3, FNx3View};

#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: [F; 3],
    max: [F; 3],
}

impl Aabb {
    fn point(p: [F; 3]) -> Self {
        Self { min: p, max: p }
    }

    fn union(a: Aabb, b: Aabb) -> Self {
        Self {
            min: [
                a.min[0].min(b.min[0]),
                a.min[1].min(b.min[1]),
                a.min[2].min(b.min[2]),
            ],
            max: [
                a.max[0].max(b.max[0]),
                a.max[1].max(b.max[1]),
                a.max[2].max(b.max[2]),
            ],
        }
    }

    /// Squared distance from `p` to the closest point on the AABB
    /// (0 if `p` is inside).
    #[inline]
    fn dist_sq_to(&self, p: [F; 3]) -> F {
        let mut s: F = 0.0;
        for d in 0..3 {
            let v = if p[d] < self.min[d] {
                self.min[d] - p[d]
            } else if p[d] > self.max[d] {
                p[d] - self.max[d]
            } else {
                0.0
            };
            s += v * v;
        }
        s
    }
}

#[derive(Debug, Clone)]
enum Node {
    Leaf { point: u32, aabb: Aabb },
    Inner { left: u32, right: u32, aabb: Aabb },
}

impl Node {
    fn aabb(&self) -> Aabb {
        match *self {
            Node::Leaf { aabb, .. } => aabb,
            Node::Inner { aabb, .. } => aabb,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AabbTree {
    nodes: Vec<Node>,
    root: u32,
}

impl AabbTree {
    fn build(points: FNx3View<'_>) -> Self {
        let n = points.nrows();
        if n == 0 {
            return Self::default();
        }
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let mut tree = AabbTree {
            nodes: Vec::with_capacity(2 * n),
            root: 0,
        };
        tree.root = tree.build_recursive(&mut indices, points);
        tree
    }

    fn build_recursive(&mut self, idx: &mut [u32], points: FNx3View<'_>) -> u32 {
        if idx.len() == 1 {
            let p = [
                points[[idx[0] as usize, 0]],
                points[[idx[0] as usize, 1]],
                points[[idx[0] as usize, 2]],
            ];
            self.nodes.push(Node::Leaf {
                point: idx[0],
                aabb: Aabb::point(p),
            });
            return (self.nodes.len() - 1) as u32;
        }
        let mut amin = [F::INFINITY; 3];
        let mut amax = [F::NEG_INFINITY; 3];
        for &i in idx.iter() {
            for d in 0..3 {
                let v = points[[i as usize, d]];
                if v < amin[d] {
                    amin[d] = v;
                }
                if v > amax[d] {
                    amax[d] = v;
                }
            }
        }
        let (mut ax, mut ext) = (0usize, amax[0] - amin[0]);
        for d in 1..3 {
            let e = amax[d] - amin[d];
            if e > ext {
                ax = d;
                ext = e;
            }
        }
        idx.sort_unstable_by(|a, b| {
            points[[*a as usize, ax]]
                .partial_cmp(&points[[*b as usize, ax]])
                .unwrap()
        });
        let mid = idx.len() / 2;
        let (left_idx, right_idx) = idx.split_at_mut(mid);
        let left = self.build_recursive(left_idx, points);
        let right = self.build_recursive(right_idx, points);
        let aabb = Aabb::union(
            self.nodes[left as usize].aabb(),
            self.nodes[right as usize].aabb(),
        );
        self.nodes.push(Node::Inner { left, right, aabb });
        (self.nodes.len() - 1) as u32
    }

    fn query(&self, p: [F; 3], radius_sq: F, mut hit: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(idx) = stack.pop() {
            match self.nodes[idx as usize] {
                Node::Leaf { point, aabb } => {
                    if aabb.dist_sq_to(p) <= radius_sq {
                        hit(point);
                    }
                }
                Node::Inner { left, right, aabb } => {
                    if aabb.dist_sq_to(p) <= radius_sq {
                        stack.push(left);
                        stack.push(right);
                    }
                }
            }
        }
    }

    /// Visit up to `k` nearest leaves to `p`, written into `top` as
    /// `(d_sq, point_idx)` sorted ascending by `d_sq`. Uses branch-and-bound
    /// with the worst-so-far distance as the prune radius.
    fn knn(&self, p: [F; 3], k: usize, top: &mut Vec<(F, u32)>) {
        if self.nodes.is_empty() || k == 0 {
            return;
        }
        self.knn_dfs(self.root, p, k, top);
    }

    fn knn_dfs(&self, node: u32, p: [F; 3], k: usize, top: &mut Vec<(F, u32)>) {
        let worst = if top.len() >= k {
            top[k - 1].0
        } else {
            F::INFINITY
        };
        let aabb_d = self.nodes[node as usize].aabb().dist_sq_to(p);
        if aabb_d > worst {
            return;
        }
        match self.nodes[node as usize] {
            Node::Leaf { point, aabb } => {
                let d = aabb.dist_sq_to(p);
                // Sorted insertion into top-k.
                let mut pos = top.len();
                while pos > 0 && top[pos - 1].0 > d {
                    pos -= 1;
                }
                if pos < k {
                    top.insert(pos, (d, point));
                    if top.len() > k {
                        top.truncate(k);
                    }
                }
            }
            Node::Inner { left, right, .. } => {
                // Visit closer child first to tighten `worst` earlier.
                let l_d = self.nodes[left as usize].aabb().dist_sq_to(p);
                let r_d = self.nodes[right as usize].aabb().dist_sq_to(p);
                if l_d <= r_d {
                    self.knn_dfs(left, p, k, top);
                    self.knn_dfs(right, p, k, top);
                } else {
                    self.knn_dfs(right, p, k, top);
                    self.knn_dfs(left, p, k, top);
                }
            }
        }
    }
}

/// AABB-tree neighbor query.
#[derive(Debug, Clone)]
pub struct AabbQuery {
    cutoff: F,
    bx: Option<SimBox>,
    result: NeighborList,
    tree: AabbTree,
    stored_pos: FNx3,
}

impl AabbQuery {
    pub fn new(cutoff: F) -> Self {
        Self {
            cutoff,
            bx: None,
            result: NeighborList::empty(),
            tree: AabbTree::default(),
            stored_pos: FNx3::zeros((0, 3)),
        }
    }

    pub fn cutoff(&self) -> F {
        self.cutoff
    }

    /// Enumerate lattice-image shifts whose magnitude could bring a tree
    /// point within `cutoff` of a query point. For each periodic axis,
    /// the image range is `[−ceil(cutoff/L), +ceil(cutoff/L)]`; non-PBC
    /// axes contribute only the zero shift.
    fn enumerate_shifts(bx: &SimBox, cutoff: F) -> Vec<[F; 3]> {
        let pbc = bx.pbc();
        let lengths = match bx.kind() {
            BoxKind::Ortho { len, .. } => [len[0], len[1], len[2]],
            BoxKind::Triclinic => {
                let l = bx.lengths();
                [l[0], l[1], l[2]]
            }
        };
        let range = |ax_len: F, periodic: bool| -> (i32, i32) {
            if !periodic || ax_len <= 0.0 {
                (0, 0)
            } else {
                let n = (cutoff / ax_len).ceil() as i32;
                (-n, n)
            }
        };
        let (nxn, nxp) = range(lengths[0], pbc[0]);
        let (nyn, nyp) = range(lengths[1], pbc[1]);
        let (nzn, nzp) = range(lengths[2], pbc[2]);

        // For ortho boxes the shift is axis-aligned and trivial. For
        // triclinic we use the lattice-vector basis.
        let mut shifts: Vec<[F; 3]> = Vec::new();
        let lat = match bx.kind() {
            BoxKind::Ortho { .. } => None,
            BoxKind::Triclinic => Some([bx.lattice(0), bx.lattice(1), bx.lattice(2)]),
        };
        for ix in nxn..=nxp {
            for iy in nyn..=nyp {
                for iz in nzn..=nzp {
                    let (dx, dy, dz) = match &lat {
                        None => (
                            ix as F * lengths[0],
                            iy as F * lengths[1],
                            iz as F * lengths[2],
                        ),
                        Some(a) => (
                            ix as F * a[0][0] + iy as F * a[1][0] + iz as F * a[2][0],
                            ix as F * a[0][1] + iy as F * a[1][1] + iz as F * a[2][1],
                            ix as F * a[0][2] + iy as F * a[1][2] + iz as F * a[2][2],
                        ),
                    };
                    shifts.push([dx, dy, dz]);
                }
            }
        }
        shifts
    }

    fn compute_pairs(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.result.clear();
        self.tree = AabbTree::build(points);
        let n = points.nrows();
        let cutoff_sq = self.cutoff * self.cutoff;
        let shifts = Self::enumerate_shifts(bx, self.cutoff);

        // Track which (i, j) pairs we've already emitted; the MIC
        // displacement gives the canonical d² regardless of which image
        // produced the hit, so we only need to emit each pair once.
        let mut seen: HashSet<(u32, u32)> = HashSet::new();

        for i in 0..n {
            let r_i = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            for shift in &shifts {
                let shifted = [r_i[0] + shift[0], r_i[1] + shift[1], r_i[2] + shift[2]];
                self.tree.query(shifted, cutoff_sq, |j| {
                    let i_u = i as u32;
                    if i_u >= j {
                        return; // self-query half-shell: i < j only
                    }
                    let key = (i_u, j);
                    if seen.contains(&key) {
                        return;
                    }
                    let r_j = [
                        points[[j as usize, 0]],
                        points[[j as usize, 1]],
                        points[[j as usize, 2]],
                    ];
                    let dr = bx.shortest_vector_impl(r_i, r_j);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff_sq {
                        seen.insert(key);
                        self.result.push(i_u, j, d2, dr);
                    }
                });
            }
        }

        // Sort the emitted pairs lexicographically for deterministic output
        // (matches LinkCell semantics).
        let n_pairs = self.result.n_pairs();
        let mut order: Vec<usize> = (0..n_pairs).collect();
        order.sort_unstable_by_key(|&k| {
            (
                self.result.query_point_indices()[k],
                self.result.point_indices()[k],
            )
        });
        let mut sorted = NeighborList::with_mode(QueryMode::SelfQuery, n, n);
        for k in order {
            sorted.push(
                self.result.query_point_indices()[k],
                self.result.point_indices()[k],
                self.result.dist_sq()[k],
                [
                    self.result.vectors()[[k, 0]],
                    self.result.vectors()[[k, 1]],
                    self.result.vectors()[[k, 2]],
                ],
            );
        }
        self.result = sorted;
        self.bx = Some(bx.clone());
        self.stored_pos = points.to_owned();
    }

    /// Find the `k` nearest neighbors of `query` in the most-recent build,
    /// honoring PBC via the same image-shift enumeration as
    /// [`build`](NbListAlgo::build).
    ///
    /// Returns up to `k` pairs `(j_index, mic_dist_sq)`, sorted ascending
    /// by distance. Fewer than `k` may be returned if the system has fewer
    /// than `k` points (or `k = 0`).
    ///
    /// # Panics
    ///
    /// Panics if [`build`](NbListAlgo::build) hasn't been called yet.
    pub fn query_knn(&self, query: [F; 3], k: usize) -> Vec<(u32, F)> {
        if k == 0 || self.stored_pos.nrows() == 0 {
            return Vec::new();
        }
        let bx = self.bx.as_ref().expect("query_knn before build");
        // For k-NN we don't have an a-priori cutoff; the MIC bound on
        // any pair-distance is at most half the box diagonal, so
        // `diag * 0.5` is a safe and tight image-shift radius.
        let diag = match bx.kind() {
            BoxKind::Ortho { len, .. } => {
                (len[0] * len[0] + len[1] * len[1] + len[2] * len[2]).sqrt()
            }
            BoxKind::Triclinic => {
                let l = bx.lengths();
                (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt()
            }
        };
        let shifts = Self::enumerate_shifts(bx, diag * 0.5);

        // Collect per-original-index minimum-distance candidate.
        let mut best_per_j: std::collections::HashMap<u32, F> = std::collections::HashMap::new();
        let mut top: Vec<(F, u32)> = Vec::with_capacity(k + 1);
        let pts = self.stored_pos.view();

        for shift in &shifts {
            let shifted = [
                query[0] + shift[0],
                query[1] + shift[1],
                query[2] + shift[2],
            ];
            top.clear();
            self.tree.knn(shifted, k, &mut top);
            for &(_, j) in &top {
                let r_j = [
                    pts[[j as usize, 0]],
                    pts[[j as usize, 1]],
                    pts[[j as usize, 2]],
                ];
                let dr = bx.shortest_vector_impl(query, r_j);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                best_per_j
                    .entry(j)
                    .and_modify(|prev| {
                        if d2 < *prev {
                            *prev = d2;
                        }
                    })
                    .or_insert(d2);
            }
        }
        let mut out: Vec<(u32, F)> = best_per_j.into_iter().collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out.truncate(k);
        out
    }
}

impl NbListAlgo for AabbQuery {
    fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.compute_pairs(points, bx);
    }

    fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.build(points, bx);
    }

    fn query(&self) -> &NeighborList {
        &self.result
    }

    fn box_ref(&self) -> &SimBox {
        self.bx.as_ref().expect("box_ref called before build")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::neighbors::{BruteForce, NbListAlgo};
    use ndarray::array;

    fn cube_bx(l: F, pbc: [bool; 3]) -> SimBox {
        SimBox::cube(l, array![0.0_f64, 0.0, 0.0], pbc).unwrap()
    }

    #[test]
    fn empty_input_is_empty_output() {
        let pts: FNx3 = ndarray::Array2::zeros((0, 3));
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        assert_eq!(aabb.query().n_pairs(), 0);
    }

    #[test]
    fn matches_brute_force_no_pbc() {
        let pts = array![
            [1.0_f64, 1.0, 1.0],
            [1.4, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [5.0, 5.0, 5.0],
            [5.4, 5.0, 5.0],
        ];
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let mut bf = BruteForce::new(1.0);
        bf.build(pts.view(), &bx);
        let mut a_pairs: Vec<(u32, u32)> = (0..aabb.query().n_pairs())
            .map(|k| {
                (
                    aabb.query().query_point_indices()[k],
                    aabb.query().point_indices()[k],
                )
            })
            .collect();
        let mut b_pairs: Vec<(u32, u32)> = (0..bf.query().n_pairs())
            .map(|k| {
                (
                    bf.query().query_point_indices()[k],
                    bf.query().point_indices()[k],
                )
            })
            .collect();
        a_pairs.sort_unstable();
        b_pairs.sort_unstable();
        assert_eq!(a_pairs, b_pairs);
    }

    #[test]
    fn matches_brute_force_with_pbc() {
        let pts = array![[0.5_f64, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let bx = cube_bx(10.0, [true, true, true]);
        let mut aabb = AabbQuery::new(2.0);
        aabb.build(pts.view(), &bx);
        let mut bf = BruteForce::new(2.0);
        bf.build(pts.view(), &bx);
        assert_eq!(aabb.query().n_pairs(), bf.query().n_pairs());
        assert_eq!(aabb.query().n_pairs(), 1);
        // PBC distance is 1.0, not 9.0.
        let d2_a = aabb.query().dist_sq()[0];
        let d2_b = bf.query().dist_sq()[0];
        assert!((d2_a - 1.0).abs() < 1e-12);
        assert!((d2_a - d2_b).abs() < 1e-12);
    }

    #[test]
    fn cutoff_below_box_size_enumerates_27_pbc_images() {
        // Tree is built on raw positions (no wrapping), so the +1 and −1
        // image shifts are needed to catch wrap-pairs near each boundary,
        // even when cutoff < L/2.
        let bx = cube_bx(10.0, [true; 3]);
        let shifts = AabbQuery::enumerate_shifts(&bx, 2.0);
        assert_eq!(shifts.len(), 27);
    }

    #[test]
    fn non_pbc_box_has_only_zero_shift() {
        let bx = cube_bx(10.0, [false; 3]);
        let shifts = AabbQuery::enumerate_shifts(&bx, 2.0);
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn cutoff_above_full_box_enumerates_more_images() {
        // With cutoff > L, the second image ring is needed too.
        let bx = cube_bx(10.0, [true; 3]);
        let shifts = AabbQuery::enumerate_shifts(&bx, 12.0);
        // ceil(12/10) = 2 → [-2, 2] = 5 per axis → 125 total
        assert_eq!(shifts.len(), 125);
    }

    #[test]
    fn larger_random_system_matches_brute_force() {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(7);
        let n = 200;
        let mut pts = FNx3::zeros((n, 3));
        for i in 0..n {
            pts[[i, 0]] = rng.random::<F>() * 10.0;
            pts[[i, 1]] = rng.random::<F>() * 10.0;
            pts[[i, 2]] = rng.random::<F>() * 10.0;
        }
        let bx = cube_bx(10.0, [true, true, true]);
        let mut aabb = AabbQuery::new(1.5);
        aabb.build(pts.view(), &bx);
        let mut bf = BruteForce::new(1.5);
        bf.build(pts.view(), &bx);
        assert_eq!(aabb.query().n_pairs(), bf.query().n_pairs());
        let mut a: Vec<(u32, u32)> = (0..aabb.query().n_pairs())
            .map(|k| {
                (
                    aabb.query().query_point_indices()[k],
                    aabb.query().point_indices()[k],
                )
            })
            .collect();
        let mut b: Vec<(u32, u32)> = (0..bf.query().n_pairs())
            .map(|k| {
                (
                    bf.query().query_point_indices()[k],
                    bf.query().point_indices()[k],
                )
            })
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn dist_sq_to_aabb() {
        let a = Aabb {
            min: [0.0_f64, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        assert!(a.dist_sq_to([0.5, 0.5, 0.5]).abs() < 1e-12);
        assert!((a.dist_sq_to([2.0, 0.5, 0.5]) - 1.0).abs() < 1e-12);
        assert!((a.dist_sq_to([2.0, 2.0, 2.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn knn_finds_k_closest_on_line() {
        // Points at x = 0, 1, 2, 3, 4. Query at x = 0.2 with k = 3 should
        // return {0, 1, 2} sorted by distance.
        let pts = array![
            [0.0_f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let bx = cube_bx(100.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.2, 0.0, 0.0], 3);
        assert_eq!(knn.len(), 3);
        assert_eq!(knn[0].0, 0);
        assert_eq!(knn[1].0, 1);
        assert_eq!(knn[2].0, 2);
        // Distances ascending: 0.2² < 0.8² < 1.8².
        assert!(knn[0].1 < knn[1].1);
        assert!(knn[1].1 < knn[2].1);
    }

    #[test]
    fn knn_with_pbc_finds_wrap_neighbor() {
        // x = 0.1 and 9.9 in a PBC box of length 10 → wrap distance 0.2.
        // Query at x = 0 should find 0 (distance 0.1) and 1 (wrap distance 0.1).
        let pts = array![[0.1_f64, 0.0, 0.0], [9.9, 0.0, 0.0]];
        let bx = cube_bx(10.0, [true, true, true]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.0, 0.0, 0.0], 2);
        assert_eq!(knn.len(), 2);
        // Both at distance² = 0.01 (approximately).
        for &(_, d2) in &knn {
            assert!(
                (d2 - 0.01).abs() < 1e-9,
                "expected wrap distance ~0.1; got d²={d2}"
            );
        }
    }

    #[test]
    fn knn_k_larger_than_n_returns_all() {
        let pts = array![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.5, 0.0, 0.0], 10);
        assert_eq!(knn.len(), 2);
    }

    #[test]
    fn knn_zero_k_returns_empty() {
        let pts = array![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.5, 0.0, 0.0], 0);
        assert_eq!(knn.len(), 0);
    }

    #[test]
    fn knn_matches_brute_force_random() {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(11);
        let n = 100;
        let mut pts = FNx3::zeros((n, 3));
        for i in 0..n {
            pts[[i, 0]] = rng.random::<F>() * 10.0;
            pts[[i, 1]] = rng.random::<F>() * 10.0;
            pts[[i, 2]] = rng.random::<F>() * 10.0;
        }
        let bx = cube_bx(10.0, [true; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);

        let q = [5.0_f64, 5.0, 5.0];
        let k = 7;
        let aabb_knn = aabb.query_knn(q, k);

        // Brute-force reference.
        let mut bf: Vec<(u32, F)> = (0..n)
            .map(|j| {
                let r_j = [pts[[j, 0]], pts[[j, 1]], pts[[j, 2]]];
                let dr = bx.shortest_vector_impl(q, r_j);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                (j as u32, d2)
            })
            .collect();
        bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        bf.truncate(k);

        assert_eq!(aabb_knn.len(), bf.len());
        for (a, b) in aabb_knn.iter().zip(bf.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-12);
        }
    }
}
