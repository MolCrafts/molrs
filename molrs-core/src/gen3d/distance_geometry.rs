//! Distance-geometry embedding backend (stage-1).
//!
//! First-principles improvements over basic DG:
//! 1. Exact 1-3 / 1-4 distance bounds from VSEPR hybridization angles
//! 2. Ring closure constraints for tighter intra-ring bounds
//! 3. Convergent triangle-inequality smoothing (iterate until stable)
//! 4. Metrized (correlated) distance sampling
//! 5. Metric-matrix eigenvalue embedding (Crippen-Havel)

use std::collections::{HashMap, HashSet, VecDeque};

use rand::Rng;

use super::builder::BuildSummary;
use super::geom::{add, dot, norm, random_unit, scale, sub};
use crate::element::Element;
use crate::error::MolRsError;
use crate::molgraph::{AtomId, MolGraph};
use crate::rings::find_rings;

/// Bond-length tolerance (Å) already used for 1-2 bounds; propagated into
/// higher-order bounds via geometry.
const BOND_TOL: f64 = 0.06;

/// Embed initial 3D coordinates with an improved distance-geometry workflow.
pub(crate) fn embed_distance_geometry(
    mol: &mut MolGraph,
    rng: &mut impl Rng,
) -> Result<BuildSummary, MolRsError> {
    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    if atom_ids.is_empty() {
        return Err(MolRsError::validation(
            "cannot build 3D coordinates for empty molecule",
        ));
    }
    let n = atom_ids.len();
    let id_to_idx: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let adjacency = build_adjacency(mol, &atom_ids, &id_to_idx);
    let topo_dist = topological_distances(&adjacency);
    let cov_r: Vec<f64> = atom_ids
        .iter()
        .map(|&id| covalent_radius(mol, id))
        .collect();
    let vdw_r: Vec<f64> = atom_ids.iter().map(|&id| vdw_radius(mol, id)).collect();

    // --- Improvement 1: precise geometric bounds ---
    let (mut lower, mut upper) = build_geometric_bounds(
        mol, &atom_ids, &id_to_idx, &adjacency, &topo_dist, &cov_r, &vdw_r,
    );

    // --- Improvement 2: ring closure constraints ---
    apply_ring_constraints(mol, &atom_ids, &id_to_idx, &mut lower, &mut upper);

    // --- Improvement 3: convergent smoothing ---
    smooth_bounds_converged(&mut lower, &mut upper, 50);

    let mut warnings = Vec::new();
    fix_inconsistencies(&mut lower, &mut upper, &mut warnings);

    // --- Improvement 4: metrized sampling ---
    let target = metrize_sample(&lower, &upper, rng);

    // --- Improvement 5: eigenvalue embedding ---
    let mut coords = eigenvalue_embed(&target, n);
    // Fallback: if eigenvalue embedding is degenerate, use random init.
    let max_coord = coords
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0_f64, |m, &x| m.max(x.abs()));
    if max_coord < 0.01 {
        coords = random_init_coords(n, rng);
    }

    // Refine with stress minimization.
    optimize_stress(&mut coords, &target, &lower);
    recenter(&mut coords);

    for (i, atom_id) in atom_ids.iter().copied().enumerate() {
        let atom = mol.get_atom_mut(atom_id)?;
        atom.set("x", coords[i][0]);
        atom.set("y", coords[i][1]);
        atom.set("z", coords[i][2]);
    }

    Ok(BuildSummary {
        placed_atoms: n,
        warnings,
    })
}

// ───────────────────── adjacency / topology ─────────────────────

fn build_adjacency(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); atom_ids.len()];
    for (i, &aid) in atom_ids.iter().enumerate() {
        let mut nlist: Vec<usize> = mol
            .neighbors(aid)
            .filter_map(|nid| id_to_idx.get(&nid).copied())
            .collect();
        nlist.sort_unstable();
        adj[i] = nlist;
    }
    adj
}

#[allow(clippy::needless_range_loop)]
fn topological_distances(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for start in 0..n {
        let mut q = VecDeque::new();
        dist[start][start] = 0;
        q.push_back(start);
        while let Some(i) = q.pop_front() {
            let d = dist[start][i];
            for &j in &adjacency[i] {
                if dist[start][j] == usize::MAX {
                    dist[start][j] = d + 1;
                    q.push_back(j);
                }
            }
        }
    }
    dist
}

// ───────────────────── precise geometric bounds ─────────────────────

#[allow(clippy::needless_range_loop)]
fn build_geometric_bounds(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    _id_to_idx: &HashMap<AtomId, usize>,
    adjacency: &[Vec<usize>],
    topo_dist: &[Vec<usize>],
    _cov_r: &[f64],
    vdw_r: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = atom_ids.len();
    let mut lower = vec![vec![0.0_f64; n]; n];
    let mut upper = vec![vec![1.0e6_f64; n]; n];
    for i in 0..n {
        upper[i][i] = 0.0;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let ai = atom_ids[i];
            let aj = atom_ids[j];
            let dij = topo_dist[i][j];
            let bonded = bond_order_between(mol, ai, aj);

            let (lo, hi) = if bonded > 0.0 {
                // 1-2 (bonded): tight bounds from ideal bond length.
                let r0 = ideal_bond_length(mol, ai, aj);
                ((r0 - BOND_TOL).max(0.4), r0 + BOND_TOL)
            } else if dij == 2 {
                // 1-3: exact from VSEPR angle via law of cosines.
                bounds_13_pair(mol, atom_ids, adjacency, i, j)
            } else if dij == 3 {
                // 1-4: exact from chain geometry with free torsion.
                bounds_14_pair(mol, atom_ids, adjacency, topo_dist, i, j)
            } else if dij < usize::MAX {
                // Longer topological distances.
                let vdw_lo = 0.35 * (vdw_r[i] + vdw_r[j]);
                let chain_hi = 1.7 * dij as f64 + 0.9;
                (vdw_lo.max(1.2), chain_hi.max(vdw_lo + 0.5))
            } else {
                // Disconnected components.
                let lo = (0.30 * (vdw_r[i] + vdw_r[j])).max(1.0);
                let hi = (2.0 * (n as f64).sqrt() + 6.0).max(lo + 1.0);
                (lo, hi)
            };

            lower[i][j] = lo;
            lower[j][i] = lo;
            upper[i][j] = hi;
            upper[j][i] = hi;
        }
    }

    (lower, upper)
}

/// Compute 1-3 distance bounds using the VSEPR ideal angle at the bridging atom.
fn bounds_13_pair(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    adjacency: &[Vec<usize>],
    i: usize,
    j: usize,
) -> (f64, f64) {
    // Find common neighbour(s) — the bridging atom(s).
    let i_set: HashSet<usize> = adjacency[i].iter().copied().collect();
    let centers: Vec<usize> = adjacency[j]
        .iter()
        .filter(|&&n| i_set.contains(&n))
        .copied()
        .collect();

    if centers.is_empty() {
        // Fallback (shouldn't happen for topo_dist == 2).
        return (1.1, 3.0);
    }

    // Take the tightest intersection over all bridging atoms.
    let mut best_lo = 0.0_f64;
    let mut best_hi = f64::MAX;

    for &c in &centers {
        let ai = atom_ids[i];
        let ac = atom_ids[c];
        let aj = atom_ids[j];
        let r1 = ideal_bond_length(mol, ai, ac);
        let r2 = ideal_bond_length(mol, ac, aj);
        let theta = hybridization_angle(mol, ac);

        // Exact distance from law of cosines.
        let d0 = (r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * theta.cos())
            .max(0.01)
            .sqrt();

        // Tolerance: propagated from bond length uncertainty + angle flexibility.
        let tol = 2.0 * BOND_TOL + angle_flexibility(theta);
        let lo = (d0 - tol).max(0.4);
        let hi = d0 + tol;

        best_lo = best_lo.max(lo);
        best_hi = best_hi.min(hi);
    }

    if best_hi < best_lo + 0.01 {
        best_hi = best_lo + 0.01;
    }
    (best_lo, best_hi)
}

/// Distance tolerance from angle flexibility, derived from hybridization stiffness.
/// sp → very rigid, sp3 → most flexible.
fn angle_flexibility(theta: f64) -> f64 {
    let pi = std::f64::consts::PI;
    if (theta - pi).abs() < 0.05 {
        0.02 // sp (linear)
    } else if (theta - 2.094).abs() < 0.1 {
        0.05 // sp2 (120°)
    } else {
        0.10 // sp3 (109.47°)
    }
}

/// Compute 1-4 distance bounds from chain geometry i-a-b-j with free torsion.
///
/// d²(τ) = r1²+r2²+r3² − 2r1r2cos θ₁ − 2r2r3cos θ₂
///         + 2r1r3[cos θ₁ cos θ₂ − sin θ₁ sin θ₂ cos τ]
///
/// Minimum at τ = 0 (cis), maximum at τ = π (anti).
fn bounds_14_pair(
    mol: &MolGraph,
    atom_ids: &[AtomId],
    adjacency: &[Vec<usize>],
    topo_dist: &[Vec<usize>],
    i: usize,
    j: usize,
) -> (f64, f64) {
    let mut best_lo = 0.0_f64;
    let mut best_hi = f64::MAX;
    let mut found_path = false;

    for &a in &adjacency[i] {
        if topo_dist[a][j] != 2 {
            continue;
        }
        for &b in &adjacency[a] {
            if b == i {
                continue;
            }
            if !adjacency[b].contains(&j) {
                continue;
            }
            found_path = true;
            let r1 = ideal_bond_length(mol, atom_ids[i], atom_ids[a]);
            let r2 = ideal_bond_length(mol, atom_ids[a], atom_ids[b]);
            let r3 = ideal_bond_length(mol, atom_ids[b], atom_ids[j]);
            let theta1 = hybridization_angle(mol, atom_ids[a]);
            let theta2 = hybridization_angle(mol, atom_ids[b]);

            let (lo, hi) = exact_14_bounds(r1, r2, r3, theta1, theta2);
            best_lo = best_lo.max(lo);
            best_hi = best_hi.min(hi);
        }
    }

    if !found_path {
        return (1.4, 4.5);
    }

    // Propagate bond-length tolerance (3 bonds in chain).
    let tol = 3.0 * BOND_TOL;
    best_lo = (best_lo - tol).max(0.5);
    best_hi += tol;

    if best_hi < best_lo + 0.01 {
        best_hi = best_lo + 0.01;
    }
    (best_lo, best_hi)
}

/// Analytical 1-4 distance extremes over all torsion angles.
fn exact_14_bounds(r1: f64, r2: f64, r3: f64, theta1: f64, theta2: f64) -> (f64, f64) {
    let common =
        r1 * r1 + r2 * r2 + r3 * r3 - 2.0 * r1 * r2 * theta1.cos() - 2.0 * r2 * r3 * theta2.cos();
    // τ = 0 (cis): cos τ = +1  →  last term = 2r1r3 cos(θ1+θ2)
    let d_cis_sq = (common + 2.0 * r1 * r3 * (theta1 + theta2).cos()).max(0.01);
    // τ = π (anti): cos τ = −1 →  last term = 2r1r3 cos(θ1−θ2)
    let d_anti_sq = (common + 2.0 * r1 * r3 * (theta1 - theta2).cos()).max(0.01);

    let d_min = d_cis_sq.sqrt().min(d_anti_sq.sqrt());
    let d_max = d_cis_sq.sqrt().max(d_anti_sq.sqrt());
    (d_min, d_max)
}

// ───────────────────── ring closure constraints ─────────────────────

fn apply_ring_constraints(
    mol: &MolGraph,
    _atom_ids: &[AtomId],
    id_to_idx: &HashMap<AtomId, usize>,
    lower: &mut [Vec<f64>],
    upper: &mut [Vec<f64>],
) {
    let ring_info = find_rings(mol);
    for ring in ring_info.rings() {
        let n_ring = ring.len();
        if n_ring < 3 {
            continue;
        }
        let idxs: Vec<usize> = ring
            .iter()
            .filter_map(|&aid| id_to_idx.get(&aid).copied())
            .collect();
        if idxs.len() != n_ring {
            continue;
        }

        // Bond lengths around the ring.
        let ring_bonds: Vec<f64> = (0..n_ring)
            .map(|k| ideal_bond_length(mol, ring[k], ring[(k + 1) % n_ring]))
            .collect();

        // Check if all ring atoms are sp2 (aromatic / planar ring).
        let all_sp2 = ring.iter().all(|&aid| {
            let theta = hybridization_angle(mol, aid);
            (theta - 2.094).abs() < 0.1
        });

        let avg_bond = ring_bonds.iter().sum::<f64>() / n_ring as f64;

        for a in 0..n_ring {
            for b in (a + 1)..n_ring {
                let i = idxs[a];
                let j = idxs[b];
                let k_short = b - a;

                // Upper bound: sum of bond lengths along each path.
                let sum_short: f64 = (a..b).map(|p| ring_bonds[p]).sum();
                let sum_long: f64 = (b..n_ring).chain(0..a).map(|p| ring_bonds[p]).sum();
                let path_upper = sum_short.min(sum_long);
                if path_upper < upper[i][j] {
                    upper[i][j] = path_upper;
                    upper[j][i] = path_upper;
                }

                if all_sp2 {
                    // Planar ring: chord distance from regular polygon geometry.
                    let chord = planar_chord(avg_bond, n_ring, k_short);
                    // Tolerance for bond-length variation in heteroaromatics.
                    let tol = 0.15;
                    let lo = chord - tol;
                    let hi = chord + tol;
                    if lo > lower[i][j] {
                        lower[i][j] = lo;
                        lower[j][i] = lo;
                    }
                    if hi < upper[i][j] {
                        upper[i][j] = hi;
                        upper[j][i] = hi;
                    }
                } else if n_ring <= 6 {
                    // Small aliphatic rings: puckering reduces chord by ≤ ~20%.
                    let chord = planar_chord(avg_bond, n_ring, k_short);
                    let lo = chord * 0.80;
                    if lo > lower[i][j] {
                        lower[i][j] = lo;
                        lower[j][i] = lo;
                    }
                }
            }
        }
    }
}

/// Chord distance in a planar regular polygon with edge length `r`, `n` vertices,
/// for vertices separated by `k` edges.
///
/// d = r · sin(k π / n) / sin(π / n)
fn planar_chord(r: f64, n: usize, k: usize) -> f64 {
    let pi = std::f64::consts::PI;
    r * (pi * k as f64 / n as f64).sin() / (pi / n as f64).sin()
}

// ───────────────────── convergent bounds smoothing ─────────────────────

fn smooth_bounds_converged(lower: &mut [Vec<f64>], upper: &mut [Vec<f64>], max_rounds: usize) {
    let n = lower.len();
    for _ in 0..max_rounds {
        let mut changed = false;

        // Upper-bound tightening.
        for k in 0..n {
            for i in 0..n {
                if i == k {
                    continue;
                }
                for j in (i + 1)..n {
                    if j == k {
                        continue;
                    }
                    let u = upper[i][k] + upper[k][j];
                    if u < upper[i][j] - 1.0e-10 {
                        upper[i][j] = u;
                        upper[j][i] = u;
                        changed = true;
                    }
                }
            }
        }

        // Lower-bound tightening.
        for k in 0..n {
            for i in 0..n {
                if i == k {
                    continue;
                }
                for j in (i + 1)..n {
                    if j == k {
                        continue;
                    }
                    let l = (lower[i][k] - upper[k][j]).max(lower[k][j] - upper[i][k]);
                    if l > lower[i][j] + 1.0e-10 {
                        lower[i][j] = l;
                        lower[j][i] = l;
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn fix_inconsistencies(lower: &mut [Vec<f64>], upper: &mut [Vec<f64>], warnings: &mut Vec<String>) {
    let n = lower.len();
    let mut inconsistent = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if upper[i][j] < lower[i][j] + 1.0e-6 {
                inconsistent += 1;
                let mid = 0.5 * (upper[i][j] + lower[i][j]);
                lower[i][j] = (mid - 0.005).max(0.0);
                upper[i][j] = mid + 0.005;
                lower[j][i] = lower[i][j];
                upper[j][i] = upper[i][j];
            }
        }
    }
    if inconsistent > 0 {
        warnings.push(format!(
            "distance-geometry bounds had {} inconsistent pair(s); softened during smoothing",
            inconsistent
        ));
    }
}

// ───────────────────── metrized distance sampling ─────────────────────

/// Sample a distance matrix using metrization: after choosing each d(i,j),
/// re-smooth the bounds for all pairs involving i or j before sampling the
/// next pair.  Produces much more geometrically consistent distance matrices
/// than independent uniform sampling.
fn metrize_sample(lower: &[Vec<f64>], upper: &[Vec<f64>], rng: &mut impl Rng) -> Vec<Vec<f64>> {
    let n = lower.len();
    let mut d = vec![vec![0.0_f64; n]; n];

    // Work on copies so original bounds are preserved.
    let mut lo: Vec<Vec<f64>> = lower.to_vec();
    let mut hi: Vec<Vec<f64>> = upper.to_vec();
    let mut sampled = vec![vec![false; n]; n];

    // Shuffle pair order for better metrization.
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    for k in (1..pairs.len()).rev() {
        let idx = rng.random_range(0..(k + 1));
        pairs.swap(k, idx);
    }

    for &(i, j) in &pairs {
        let lo_ij = lo[i][j].max(1.0e-5);
        let hi_ij = hi[i][j].max(lo_ij + 1.0e-5);
        let v = if (hi_ij - lo_ij) < 1.0e-8 {
            lo_ij
        } else {
            rng.random_range(lo_ij..hi_ij)
        };
        d[i][j] = v;
        d[j][i] = v;
        sampled[i][j] = true;
        sampled[j][i] = true;

        // Fix this pair's bounds to the sampled value.
        lo[i][j] = v;
        lo[j][i] = v;
        hi[i][j] = v;
        hi[j][i] = v;

        // Locally re-smooth bounds for unsampled pairs involving i or j.
        for k in 0..n {
            if k == i || k == j {
                continue;
            }
            if !sampled[i][k] {
                let new_hi = v + hi[j][k];
                if new_hi < hi[i][k] {
                    hi[i][k] = new_hi;
                    hi[k][i] = new_hi;
                }
                let new_lo = (v - hi[j][k]).max(lo[j][k] - v);
                if new_lo > lo[i][k] {
                    lo[i][k] = new_lo;
                    lo[k][i] = new_lo;
                }
            }
            if !sampled[j][k] {
                let new_hi = v + hi[i][k];
                if new_hi < hi[j][k] {
                    hi[j][k] = new_hi;
                    hi[k][j] = new_hi;
                }
                let new_lo = (v - hi[i][k]).max(lo[i][k] - v);
                if new_lo > lo[j][k] {
                    lo[j][k] = new_lo;
                    lo[k][j] = new_lo;
                }
            }
        }
    }

    d
}

// ───────────────────── eigenvalue embedding (Crippen-Havel) ─────────────────────

/// Embed a sampled distance matrix into 3D coordinates via the metric-matrix
/// method: D² → Gram matrix G → eigendecompose → take top-3 eigenvectors.
fn eigenvalue_embed(target: &[Vec<f64>], n: usize) -> Vec<[f64; 3]> {
    if n <= 1 {
        return vec![[0.0; 3]; n];
    }

    // Distance-squared matrix.
    let mut d2 = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            d2[i][j] = target[i][j] * target[i][j];
        }
    }

    // Gram matrix: G = -½ J D² J  where J = I − 11ᵀ/n.
    // G[i][j] = -½ (D²[i][j] − row_mean[i] − col_mean[j] + grand_mean)
    let grand_mean: f64 = d2.iter().flat_map(|row| row.iter()).sum::<f64>() / (n * n) as f64;
    let row_means: Vec<f64> = d2
        .iter()
        .map(|row| row.iter().sum::<f64>() / n as f64)
        .collect();

    let mut gram = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let val = -0.5 * (d2[i][j] - row_means[i] - row_means[j] + grand_mean);
            gram[i][j] = val;
            gram[j][i] = val;
        }
    }

    // Eigendecompose.
    let (eigenvalues, eigenvectors) = jacobi_eigendecomp(&mut gram);

    // Sort eigenvalues descending; take top 3 positive ones.
    let mut indexed: Vec<(usize, f64)> = eigenvalues.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut coords = vec![[0.0_f64; 3]; n];
    for dim in 0..3.min(indexed.len()) {
        let (idx, eval) = indexed[dim];
        let sf = if eval > 1.0e-10 { eval.sqrt() } else { 0.0 };
        for i in 0..n {
            coords[i][dim] = eigenvectors[i][idx] * sf;
        }
    }

    coords
}

/// Jacobi cyclic-sweep eigenvalue algorithm for real symmetric matrices.
/// Returns (eigenvalues, eigenvectors) where `eigenvectors[i][j]` is the
/// j-th component of the i-th *row* vector (i.e. eigenvector columns are
/// `eigenvectors[·][j]`).
#[allow(clippy::needless_range_loop)]
fn jacobi_eigendecomp(a: &mut [Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_sweeps = 100 * n.max(10);
    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1.0e-24 {
            break;
        }

        // Cyclic sweep.
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1.0e-15 {
                    continue;
                }
                let diff = a[q][q] - a[p][p];
                let t = if diff.abs() < 1.0e-15 {
                    1.0_f64
                } else {
                    let tau = diff / (2.0 * a[p][q]);
                    if tau.abs() > 1.0e15 {
                        0.5 / tau
                    } else {
                        tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
                    }
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_val = s / (1.0 + c);
                let a_pq = a[p][q];
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                a[p][p] -= t * a_pq;
                a[q][q] += t * a_pq;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let a_rp = a[r][p];
                    let a_rq = a[r][q];
                    a[r][p] = a_rp - s * (a_rq + tau_val * a_rp);
                    a[p][r] = a[r][p];
                    a[r][q] = a_rq + s * (a_rp - tau_val * a_rq);
                    a[q][r] = a[r][q];
                }

                for r in 0..n {
                    let v_rp = v[r][p];
                    let v_rq = v[r][q];
                    v[r][p] = v_rp - s * (v_rq + tau_val * v_rp);
                    v[r][q] = v_rq + s * (v_rp - tau_val * v_rq);
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

// ───────────────────── stress optimisation (refinement) ─────────────────────

fn random_init_coords(n: usize, rng: &mut impl Rng) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            let r = rng.random_range(0.6..1.8);
            scale(random_unit(rng), r)
        })
        .collect()
}

fn optimize_stress(coords: &mut [[f64; 3]], target: &[Vec<f64>], lower: &[Vec<f64>]) {
    let n = coords.len();
    if n < 2 {
        return;
    }

    let mut grad = vec![[0.0_f64; 3]; n];
    let mut trial = coords.to_vec();
    let mut step = 0.04_f64;
    let mut energy = stress_energy_grad(coords, target, lower, &mut grad);

    for _ in 0..900 {
        let rms = (grad.iter().map(|g| dot(*g, *g)).sum::<f64>() / n as f64).sqrt();
        if rms < 1.0e-4 {
            break;
        }

        for i in 0..n {
            trial[i] = sub(coords[i], scale(grad[i], step));
        }

        let e_trial = stress_energy(&trial, target, lower);
        if e_trial < energy {
            coords.copy_from_slice(&trial);
            energy = e_trial;
            let _ = stress_energy_grad(coords, target, lower, &mut grad);
            step = (step * 1.05).min(0.12);
        } else {
            step *= 0.5;
            if step < 1.0e-5 {
                break;
            }
        }
    }
}

fn stress_energy(coords: &[[f64; 3]], target: &[Vec<f64>], lower: &[Vec<f64>]) -> f64 {
    let n = coords.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let rij = sub(coords[i], coords[j]);
            let d = norm(rij).max(1.0e-8);
            let t = target[i][j].max(1.0e-6);
            let w = 1.0 / (t * t);
            let err = d - t;
            e += w * err * err;

            let lo = lower[i][j];
            if d < lo {
                let o = lo - d;
                e += 0.2 * o * o;
            }
        }
    }
    e
}

fn stress_energy_grad(
    coords: &[[f64; 3]],
    target: &[Vec<f64>],
    lower: &[Vec<f64>],
    grad: &mut [[f64; 3]],
) -> f64 {
    for g in grad.iter_mut() {
        *g = [0.0, 0.0, 0.0];
    }

    let n = coords.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let rij = sub(coords[i], coords[j]);
            let d = norm(rij).max(1.0e-8);
            let inv_d = 1.0 / d;
            let t = target[i][j].max(1.0e-6);
            let w = 1.0 / (t * t);
            let err = d - t;
            e += w * err * err;
            let gscale = 2.0 * w * err * inv_d;
            let g = scale(rij, gscale);
            grad[i] = add(grad[i], g);
            grad[j] = sub(grad[j], g);

            let lo = lower[i][j];
            if d < lo {
                let o = lo - d;
                e += 0.2 * o * o;
                let gs2 = -0.4 * o * inv_d;
                let g2 = scale(rij, gs2);
                grad[i] = add(grad[i], g2);
                grad[j] = sub(grad[j], g2);
            }
        }
    }
    e
}

fn recenter(coords: &mut [[f64; 3]]) {
    if coords.is_empty() {
        return;
    }
    let n = coords.len() as f64;
    let c = [
        coords.iter().map(|p| p[0]).sum::<f64>() / n,
        coords.iter().map(|p| p[1]).sum::<f64>() / n,
        coords.iter().map(|p| p[2]).sum::<f64>() / n,
    ];
    for p in coords {
        p[0] -= c[0];
        p[1] -= c[1];
        p[2] -= c[2];
    }
}

// ───────────────────── element helpers ─────────────────────

fn covalent_radius(mol: &MolGraph, atom_id: AtomId) -> f64 {
    mol.get_atom(atom_id)
        .ok()
        .and_then(|a| a.get_str("element"))
        .and_then(Element::by_symbol)
        .map(|e| e.covalent_radius() as f64)
        .unwrap_or(0.77)
}

fn vdw_radius(mol: &MolGraph, atom_id: AtomId) -> f64 {
    mol.get_atom(atom_id)
        .ok()
        .and_then(|a| a.get_str("element"))
        .and_then(Element::by_symbol)
        .map(|e| e.vdw_radius() as f64)
        .unwrap_or(1.8)
}

fn ideal_bond_length(mol: &MolGraph, a: AtomId, b: AtomId) -> f64 {
    let r_a = covalent_radius(mol, a);
    let r_b = covalent_radius(mol, b);
    let order = bond_order_between(mol, a, b);

    let mut length = r_a + r_b;
    if order >= 2.5 {
        length *= 0.86;
    } else if order >= 1.5 {
        length *= 0.92;
    }
    length.clamp(0.9, 2.2)
}

fn bond_order_between(mol: &MolGraph, a: AtomId, b: AtomId) -> f64 {
    mol.neighbor_bonds(a)
        .find_map(|(nbr, order)| (nbr == b).then_some(order))
        .unwrap_or(0.0)
}

/// VSEPR ideal angle at a center atom based on coordination number and
/// maximum bond order among its neighbours.
fn hybridization_angle(mol: &MolGraph, center: AtomId) -> f64 {
    let degree = mol.neighbors(center).count();
    let max_order = mol
        .neighbor_bonds(center)
        .map(|(_, order)| order)
        .fold(1.0_f64, f64::max);

    if max_order >= 2.5 || (degree == 2 && max_order >= 2.0) {
        std::f64::consts::PI // 180° — sp
    } else if max_order >= 1.5 || degree == 3 {
        2.094_f64 // 120° — sp2
    } else {
        1.911_f64 // 109.47° — sp3
    }
}
