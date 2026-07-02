// Union-Find traversal and per-bucket O(b²) compare loops read more
// clearly with explicit indexing than iterator combinators.
#![allow(clippy::needless_range_loop, clippy::if_same_then_else)]

//! Environment matching by neighbor-bond fingerprint, with optional
//! rotation-invariant registration.
//!
//! Mirrors `freud.environment.EnvironmentCluster` / `MatchEnv`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/MatchEnv.cc)).
//! Two modes:
//!
//! - **No-rotation** (`with_registration(false)`, the default): two
//!   particles match when their **sorted bond magnitudes** agree
//!   pair-wise within `rmsd_threshold`.
//! - **Registration** (`with_registration(true)`): two particles match
//!   when there is a **rotation** and a **permutation** of one bond
//!   set that minimises the RMSD vs the other to within
//!   `rmsd_threshold`. Optimal rotation per permutation is found by
//!   Horn's quaternion method (largest eigenvector of a 4×4 symmetric
//!   `N` matrix built from the cross-covariance; eigensolver lives in
//!   [`molrs::math::diagonalize::eigh_largest_sym_4x4`]).
//!   Permutations are enumerated by Heap's algorithm — viable for the
//!   typical neighborhood sizes `n ≤ 12` (12! ≈ 4.8 × 10⁸ but with
//!   early-exit on `rmsd > threshold` the practical count is much lower).
//!
//! After per-pair match decisions, particles are clustered into
//! environment classes by union-find.
//!
//! # Performance
//!
//! Per-particle bond fingerprints are built in a **single O(n_pairs)
//! pass** over the neighbor list — linear in the total neighbor count
//! (≈ O(N·k) for `N` particles with `k` neighbors each). Particles are
//! then bucketed by neighbor count, and matching compares every pair
//! *within* a bucket: O(b²) per bucket of size `b`. That quadratic
//! comparison is intrinsic to environment matching (freud does the same)
//! and is the dominant cost at scale. In registration mode each
//! comparison additionally enumerates up to `n!` permutations, bounded by
//! [`MatchEnv::with_max_neighbors_for_registration`].

use std::collections::HashMap;

use molrs::math::diagonalize::eigh_largest_sym_4x4;
use molrs::spatial::neighbors::NeighborList;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use crate::compute::util::get_positions_ref;

/// Per-frame environment-matching result.
#[derive(Debug, Clone, Default)]
pub struct MatchEnvResult {
    /// Particle → environment-class label (`0..n_clusters`).
    pub cluster_idx: Vec<u32>,
    /// Number of distinct environment classes.
    pub n_clusters: usize,
    /// Per-particle sorted-bond fingerprint (the raw feature).
    pub fingerprints: Vec<Vec<F>>,
}

impl ComputeResult for MatchEnvResult {}

/// `MatchEnv` analyzer.
///
/// See the [module documentation](self) for the two matching modes and
/// their complexity (linear fingerprint build; quadratic per-bucket
/// comparison).
#[derive(Debug, Clone, Copy)]
pub struct MatchEnv {
    rmsd_threshold: F,
    registration: bool,
    /// Hard cap on `n_neighbors` accepted in registration mode — guards
    /// against accidentally invoking O(n!) brute force for huge
    /// neighborhoods. Defaults to 8 (matches freud's safety bound).
    max_neighbors_for_registration: usize,
}

impl MatchEnv {
    pub fn new(rmsd_threshold: F) -> Result<Self, ComputeError> {
        if rmsd_threshold.is_nan() || rmsd_threshold < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "MatchEnv::rmsd_threshold",
                value: rmsd_threshold.to_string(),
            });
        }
        Ok(Self {
            rmsd_threshold,
            registration: false,
            max_neighbors_for_registration: 8,
        })
    }

    /// Toggle rotation-invariant registration (Horn's quaternion +
    /// brute-force permutation matching).
    pub fn with_registration(mut self, on: bool) -> Self {
        self.registration = on;
        self
    }

    /// Set the hard cap on neighborhood size in registration mode
    /// (`n!` cost grows fast — default 8 → 40 320 perms).
    pub fn with_max_neighbors_for_registration(mut self, n: usize) -> Self {
        self.max_neighbors_for_registration = n;
        self
    }

    pub fn rmsd_threshold(&self) -> F {
        self.rmsd_threshold
    }
}

/// Bond-vector fingerprints for every particle, built in a single
/// O(n_pairs) pass over the neighbor list. In self-query mode each pair
/// contributes to both endpoints (the `j`-side bond reversed); in
/// cross-query only the query-side (`i`) endpoint. Within each
/// particle's list bonds land in ascending pair-index order — the same
/// order a per-particle scan would produce — so downstream fingerprints,
/// magnitude sorting, and registration RMSD are unchanged. Pair indices
/// outside `0..n` are ignored, matching the old per-particle scan.
fn build_all_bond_vectors(n: usize, nlist: &NeighborList) -> Vec<Vec<[F; 3]>> {
    let mut bonds: Vec<Vec<[F; 3]>> = vec![Vec::new(); n];
    let i_idx = nlist.query_point_indices();
    let j_idx = nlist.point_indices();
    let vectors = nlist.vectors();
    let symmetric = matches!(
        nlist.mode(),
        molrs::spatial::neighbors::QueryMode::SelfQuery
    );
    for k in 0..nlist.n_pairs() {
        let i = i_idx[k] as usize;
        let v = [vectors[[k, 0]], vectors[[k, 1]], vectors[[k, 2]]];
        if i < n {
            bonds[i].push(v);
        }
        if symmetric {
            let j = j_idx[k] as usize;
            // Skip self-pairs so the reversed bond is not double-counted
            // (matches the old `else if` that fired only when `i != j`).
            if j != i && j < n {
                bonds[j].push([-v[0], -v[1], -v[2]]);
            }
        }
    }
    bonds
}

/// Magnitude-only fingerprint derived from bond vectors (for the
/// no-rotation mode).
fn magnitudes_sorted(bonds: &[[F; 3]]) -> Vec<F> {
    let mut mags: Vec<F> = bonds
        .iter()
        .map(|b| (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt())
        .collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mags
}

/// RMSD between sorted-magnitude fingerprints (no rotation).
fn rmsd_no_rotation(a: &[F], b: &[F]) -> F {
    if a.len() != b.len() {
        return F::INFINITY;
    }
    if a.is_empty() {
        return 0.0;
    }
    let mut s: F = 0.0;
    for k in 0..a.len() {
        let d = a[k] - b[k];
        s += d * d;
    }
    (s / a.len() as F).sqrt()
}

/// Optimal-rotation RMSD between two point sets `a[i] ↔ b[i]` via
/// Horn's quaternion method. Assumes `a.len() == b.len()`.
fn rmsd_horn(a: &[[F; 3]], b: &[[F; 3]]) -> F {
    let n = a.len() as F;
    // Cross-covariance H_kl = Σ_i a_i,k · b_i,l
    let mut h = [[0.0_f64; 3]; 3];
    let mut sa: F = 0.0;
    let mut sb: F = 0.0;
    for i in 0..a.len() {
        for k in 0..3 {
            sa += a[i][k] * a[i][k];
            sb += b[i][k] * b[i][k];
            for l in 0..3 {
                h[k][l] += a[i][k] * b[i][l];
            }
        }
    }
    // Horn's N matrix.
    let n_mat = [
        [
            h[0][0] + h[1][1] + h[2][2],
            h[1][2] - h[2][1],
            h[2][0] - h[0][2],
            h[0][1] - h[1][0],
        ],
        [
            h[1][2] - h[2][1],
            h[0][0] - h[1][1] - h[2][2],
            h[0][1] + h[1][0],
            h[2][0] + h[0][2],
        ],
        [
            h[2][0] - h[0][2],
            h[0][1] + h[1][0],
            -h[0][0] + h[1][1] - h[2][2],
            h[1][2] + h[2][1],
        ],
        [
            h[0][1] - h[1][0],
            h[2][0] + h[0][2],
            h[1][2] + h[2][1],
            -h[0][0] - h[1][1] + h[2][2],
        ],
    ];
    let (lambda_max, _) = eigh_largest_sym_4x4(&n_mat);
    // Standard Horn identity: min Σ |a − R b|² = (|a|² + |b|² − 2 λ_max).
    let sq_err = (sa + sb - 2.0 * lambda_max).max(0.0);
    (sq_err / n).sqrt()
}

/// Heap's algorithm: enumerate every permutation of `b`, callback per perm.
/// Early-terminates when `cb` returns `false`.
fn for_each_permutation(b: &mut [[F; 3]], cb: &mut impl FnMut(&[[F; 3]]) -> bool) -> bool {
    let n = b.len();
    let mut c = vec![0usize; n];
    if !cb(b) {
        return false;
    }
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i & 1 == 0 {
                b.swap(0, i);
            } else {
                b.swap(c[i], i);
            }
            if !cb(b) {
                return false;
            }
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    true
}

/// Best RMSD over all permutations of `b`, using Horn's optimal rotation
/// per permutation. Early-exits when a perm meets `threshold`.
fn rmsd_with_registration(a: &[[F; 3]], b: &[[F; 3]], threshold: F) -> F {
    if a.len() != b.len() {
        return F::INFINITY;
    }
    if a.is_empty() {
        return 0.0;
    }
    let mut b_work = b.to_vec();
    let mut best = F::INFINITY;
    for_each_permutation(&mut b_work, &mut |perm| {
        let r = rmsd_horn(a, perm);
        if r < best {
            best = r;
        }
        // Continue while we haven't yet matched.
        best > threshold
    });
    best
}

/// Single-pass disjoint-set union for environment clustering.
struct Dsu {
    parent: Vec<u32>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let next = self.parent[x as usize];
            self.parent[x as usize] = self.parent[next as usize];
            x = next;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
}

impl MatchEnv {
    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &NeighborList,
    ) -> Result<MatchEnvResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();

        let bonds = build_all_bond_vectors(n, nlist);

        // Group particles by neighbor count first — only same-length
        // bond sets can match.
        let mut by_len: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, b) in bonds.iter().enumerate() {
            by_len.entry(b.len()).or_default().push(i);
        }

        let mut dsu = Dsu::new(n);
        for bucket in by_len.values() {
            if bucket.is_empty() {
                continue;
            }
            let bucket_len = bonds[bucket[0]].len();
            // Decide mode for this bucket: registration is off iff
            // (a) caller disabled it, or
            // (b) bucket_len exceeds the safety cap.
            let use_registration =
                self.registration && bucket_len <= self.max_neighbors_for_registration;
            // Pre-compute sorted magnitudes for non-registration path.
            let mags: Vec<Vec<F>> = if !use_registration {
                bucket
                    .iter()
                    .map(|&i| magnitudes_sorted(&bonds[i]))
                    .collect()
            } else {
                Vec::new()
            };
            for ai in 0..bucket.len() {
                let a = bucket[ai];
                for bi in (ai + 1)..bucket.len() {
                    let b = bucket[bi];
                    let r = if use_registration {
                        rmsd_with_registration(&bonds[a], &bonds[b], self.rmsd_threshold)
                    } else {
                        rmsd_no_rotation(&mags[ai], &mags[bi])
                    };
                    if r <= self.rmsd_threshold {
                        dsu.union(a as u32, b as u32);
                    }
                }
            }
        }

        // Compact root labels into 0..n_clusters.
        let mut label_for: HashMap<u32, u32> = HashMap::new();
        let mut cluster_idx = vec![0_u32; n];
        for i in 0..n {
            let r = dsu.find(i as u32);
            let next = label_for.len() as u32;
            let lbl = *label_for.entry(r).or_insert(next);
            cluster_idx[i] = lbl;
        }
        let n_clusters = label_for.len();

        // Expose magnitude fingerprints for inspection (same shape as
        // before; bond-vector fingerprints are internal-only).
        let fingerprints: Vec<Vec<F>> = bonds.iter().map(|b| magnitudes_sorted(b)).collect();

        Ok(MatchEnvResult {
            cluster_idx,
            n_clusters,
            fingerprints,
        })
    }
}

impl Compute for MatchEnv {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = Vec<MatchEnvResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a Vec<NeighborList>,
    ) -> Result<Vec<MatchEnvResult>, ComputeError> {
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
        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::spatial::neighbors::{LinkCell, NbListAlgo};
    use molrs::spatial::region::simbox::SimBox;
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
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
        let xp = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(<F as molrs::store::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let yp = frame
            .get("atoms")
            .unwrap()
            .get("y")
            .and_then(<F as molrs::store::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let zp = frame
            .get("atoms")
            .unwrap()
            .get("z")
            .and_then(<F as molrs::store::block::BlockDtype>::from_column)
            .unwrap()
            .as_slice()
            .unwrap()
            .to_vec();
        let n = xp.len();
        let mut pos = ndarray::Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = xp[i];
            pos[[i, 1]] = yp[i];
            pos[[i, 2]] = zp[i];
        }
        let simbox = frame.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build(pos.view(), simbox);
        lc.query().clone()
    }

    /// Two identical octahedra at different centres should be classed
    /// together; their satellite atoms (each with a single bond) form a
    /// second class.
    fn paired_octahedra() -> Frame {
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (10.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]); // centre (6 bonds of length 1)
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ] {
                p.push([cx + dx, cy + dy, cz + dz]); // satellite (1 bond)
            }
        }
        frame_with(&p, 20.0)
    }

    #[test]
    fn two_octahedra_centres_share_a_class() {
        let frame = paired_octahedra();
        let nl = build_nlist(&frame, 1.2);
        let r = &MatchEnv::new(1e-9)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        // Centres are particles 0 and 7. Their fingerprints are both
        // [1,1,1,1,1,1] → same class.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[7]);
        // Each satellite has fingerprint [1] → another (single) class
        // shared by all 12 satellites.
        for k in (1..=6).chain(8..=13) {
            assert_eq!(r.cluster_idx[k], r.cluster_idx[1]);
        }
        // Centres vs satellites must differ (different fingerprint length).
        assert_ne!(r.cluster_idx[0], r.cluster_idx[1]);
        assert_eq!(r.n_clusters, 2);
    }

    #[test]
    fn distinct_fingerprints_get_distinct_classes() {
        // Three particles forming a chain: 0 has 1 neighbour, 1 has 2, 2 has 1.
        let frame = frame_with(
            &[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            10.0,
        );
        let nl = build_nlist(&frame, 1.5);
        let r = &MatchEnv::new(1e-9)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        // 0 and 2 each have one bond of length 1 → same class.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[2]);
        // 1 has two bonds → different class.
        assert_ne!(r.cluster_idx[1], r.cluster_idx[0]);
        assert_eq!(r.n_clusters, 2);
    }

    #[test]
    fn threshold_merges_near_matches() {
        // Two centres with slightly perturbed bond lengths: 1.00 vs 1.01.
        // Threshold 0.05 merges them; threshold 0.005 keeps them apart.
        let mut p: Vec<[F; 3]> = Vec::new();
        for (cx, off) in &[(5.0_f64, 1.00_f64), (10.0_f64, 1.01_f64)] {
            p.push([*cx, 5.0, 5.0]);
            for &(dx, dy, dz) in &[
                (*off, 0.0, 0.0),
                (-*off, 0.0, 0.0),
                (0.0, *off, 0.0),
                (0.0, -*off, 0.0),
            ] {
                p.push([*cx + dx, 5.0 + dy, 5.0 + dz]);
            }
        }
        let frame = frame_with(&p, 20.0);
        let nl = build_nlist(&frame, 1.5);
        let merged = &MatchEnv::new(0.05)
            .unwrap()
            .compute(&[&frame], &vec![nl.clone()])
            .unwrap()[0];
        assert_eq!(merged.cluster_idx[0], merged.cluster_idx[5]);

        let split = &MatchEnv::new(0.001)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        assert_ne!(split.cluster_idx[0], split.cluster_idx[5]);
    }

    /// Regression guard for the single-pass `build_all_bond_vectors`
    /// dispatch: an asymmetric chain with two *distinct* bond lengths
    /// pins each particle's collected bond set via the public
    /// `fingerprints`. A mis-dispatch — dropping the reversed `j`-side
    /// bond, assigning to the wrong endpoint, or double-counting a pair —
    /// changes at least one fingerprint here.
    ///
    /// Geometry: three collinear atoms at x = 0.0, 1.0, 2.5 with cutoff
    /// 1.6 Å. Neighbor pairs: (0,1) at 1.0 and (1,2) at 1.5; (0,2) at 2.5
    /// is outside the cutoff. Expected sorted magnitudes:
    ///   atom 0 → [1.0]   atom 1 → [1.0, 1.5]   atom 2 → [1.5]
    #[test]
    fn fingerprints_dispatch_forward_and_reversed_bonds() {
        let frame = frame_with(
            &[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            10.0,
        );
        let nl = build_nlist(&frame, 1.6);
        let r = &MatchEnv::new(1e-9)
            .unwrap()
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];

        let approx = |a: &[F], b: &[F]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
        };
        assert!(
            approx(&r.fingerprints[0], &[1.0]),
            "atom 0 fingerprint = {:?}",
            r.fingerprints[0]
        );
        assert!(
            approx(&r.fingerprints[1], &[1.0, 1.5]),
            "atom 1 fingerprint = {:?}",
            r.fingerprints[1]
        );
        assert!(
            approx(&r.fingerprints[2], &[1.5]),
            "atom 2 fingerprint = {:?}",
            r.fingerprints[2]
        );
    }

    #[test]
    fn invalid_threshold_errors() {
        assert!(MatchEnv::new(-1.0).is_err());
    }

    #[test]
    fn empty_frames_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = MatchEnv::new(0.1)
            .unwrap()
            .compute(&frames, &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    /// Two octahedra at different orientations: the magnitude-only mode
    /// already merges them (their sorted magnitudes are equal). Add a
    /// case that ONLY registration mode catches: equal bond-magnitude
    /// sets in two different relative orientations whose lab-frame
    /// bond vectors differ but RMSD-with-rotation is zero.
    #[test]
    fn registration_matches_rotated_octahedra() {
        // Particle 0: octahedron, axes (±x, ±y, ±z).
        // Particle 7: same octahedron rotated 45° about z. Lab-frame bond
        // vectors differ from particle 0's; registration finds the
        // rotation that aligns them.
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (12.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]);
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ] {
                if cx > 10.0 {
                    // Rotate 45° about z for the second octahedron.
                    let c = std::f64::consts::FRAC_PI_4.cos();
                    let s = std::f64::consts::FRAC_PI_4.sin();
                    p.push([cx + c * dx - s * dy, cy + s * dx + c * dy, cz + dz]);
                } else {
                    p.push([cx + dx, cy + dy, cz + dz]);
                }
            }
        }
        let frame = frame_with(&p, 30.0);
        let nl = build_nlist(&frame, 1.2);
        let r = &MatchEnv::new(1e-6)
            .unwrap()
            .with_registration(true)
            .compute(&[&frame], &vec![nl])
            .unwrap()[0];
        // The two centres (indices 0 and 7) must share a class under
        // registration mode.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[7]);
    }
}
