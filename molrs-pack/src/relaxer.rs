//! Per-target in-loop relaxer system for molecular packing.
//!
//! Relaxers modify the reference geometry (shape of the molecule itself) during
//! packing, complementing the constraint system which modifies the objective
//! function. The two-level split (`Relaxer` → `RelaxerRunner`) separates immutable
//! configuration (stored on `Target`) from mutable runtime state (created
//! inside `pack()`).
//!
//! # Built-in relaxers
//!
//! - [`TorsionMcRelaxer`]: Monte Carlo torsion angle sampling for flexible molecules.

use molrs::molgraph::MolGraph;
use molrs::rotatable::{RotatableBond, atom_id_to_index, detect_rotatable_bonds_with_downstream};
use molrs::types::F;
use rand::RngCore;
use std::collections::HashSet;
use std::f64::consts::PI;

use crate::numerics::near_zero_norm_floor;
use crate::random::uniform01_core;

// ── Traits ──────────────────────────────────────────────────────────────────

/// Per-target in-loop relaxer. Stored on `Target` (immutable config).
/// Creates a stateful runner inside `pack()`.
///
/// Analogous to `Restraint` (objective-function penalties) but operates on
/// a different axis: a `Relaxer` modifies the **reference geometry** of the
/// molecule (e.g. torsion MC on a polymer backbone), while a `Restraint`
/// adds a penalty term to the per-atom objective.
pub trait Relaxer: Send + Sync + CloneRelaxer {
    /// Create a stateful runner for this relaxer.
    /// Called once at the start of `pack()`.
    fn build(&self, ref_coords: &[[F; 3]]) -> Box<dyn RelaxerRunner>;
}

/// Clone-box helper for trait objects.
pub trait CloneRelaxer {
    fn clone_box(&self) -> Box<dyn Relaxer>;
}

impl<T: Relaxer + Clone + 'static> CloneRelaxer for T {
    fn clone_box(&self) -> Box<dyn Relaxer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Relaxer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for Box<dyn Relaxer> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Box<dyn Relaxer>")
    }
}

/// Runtime state for a relaxer. Created by `Relaxer::build()`, used inside `pack()`.
///
/// Analogous to calling `Restraint::f()`/`fg()` during evaluation,
/// but stateful (MC acceptance counters, temperature schedule, etc.).
pub trait RelaxerRunner: Send {
    /// Called between movebad and pgencan in each iteration.
    ///
    /// - `coords`: current reference coords for this molecule type
    /// - `f_current`: current total objective (fdist + frest)
    /// - `evaluate`: closure to test trial coords against full objective
    /// - `rng`: shared random number generator
    ///
    /// Returns `Some(new_coords)` if any modification accepted, `None` otherwise.
    fn on_iter(
        &mut self,
        coords: &[[F; 3]],
        f_current: F,
        evaluate: &mut dyn FnMut(&[[F; 3]]) -> F,
        rng: &mut dyn RngCore,
    ) -> Option<Vec<[F; 3]>>;

    /// Acceptance statistics (for progress reporting).
    fn acceptance_rate(&self) -> F {
        0.0
    }
}

// ── TorsionMcRelaxer ───────────────────────────────────────────────────────────

/// MC torsion angle sampling relaxer.
///
/// Analogous to `InsideSphereConstraint` — a concrete implementation of `Relaxer`.
/// Picks random rotatable bonds and applies random torsion rotations,
/// accepting or rejecting via Metropolis criterion.
///
/// # Self-avoidance
///
/// By default, the packer objective skips same-molecule pairs, so torsion MC
/// can fold a chain into self-overlapping configurations. Use
/// [`with_self_avoidance`][Self::with_self_avoidance] to add an intra-molecular
/// repulsion penalty. Atom pairs that are 1-2 or 1-3 bonded neighbors are
/// excluded from the penalty (they are naturally close due to bond geometry).
#[derive(Debug, Clone)]
pub struct TorsionMcRelaxer {
    /// Pre-detected rotatable bonds with downstream atom sets.
    pub bonds: Vec<RotatableBond>,
    /// Max rotation per step (radians). Default: π/6.
    pub max_delta: F,
    /// MC steps per iteration. Default: 10.
    pub steps: usize,
    /// Metropolis temperature. Default: 1.0.
    pub temperature: F,
    /// Self-avoidance cutoff radius. Pairs closer than `2 * radius` get a
    /// quadratic penalty. `0.0` disables (default).
    pub self_avoidance_radius: F,
    /// Excluded atom pairs (1-2 and 1-3 bonded neighbors), stored as
    /// `(min_idx, max_idx)` for efficient lookup.
    excluded_pairs: HashSet<(usize, usize)>,
}

impl TorsionMcRelaxer {
    /// Create from a `MolGraph`.
    ///
    /// Rotatable bonds are detected automatically from the graph topology
    /// (single, acyclic, non-terminal). Downstream atom sets are computed
    /// via BFS for each rotatable bond.
    pub fn new(graph: &MolGraph) -> Self {
        let bonds = detect_rotatable_bonds_with_downstream(graph);
        let excluded_pairs = compute_excluded_pairs(graph);
        Self {
            bonds,
            max_delta: (PI / 6.0) as F,
            steps: 10,
            temperature: 1.0,
            self_avoidance_radius: 0.0,
            excluded_pairs,
        }
    }

    pub fn with_temperature(mut self, t: F) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_steps(mut self, n: usize) -> Self {
        self.steps = n;
        self
    }

    pub fn with_max_delta(mut self, rad: F) -> Self {
        self.max_delta = rad;
        self
    }

    /// Enable intra-molecular self-avoidance with the given cutoff radius.
    ///
    /// Atom pairs (excluding 1-2 and 1-3 bonded neighbors) closer than
    /// `2 * radius` receive a quadratic overlap penalty:
    /// `(2*radius - dist)^2`. This prevents the chain from folding through
    /// itself during torsion MC.
    pub fn with_self_avoidance(mut self, radius: F) -> Self {
        self.self_avoidance_radius = radius;
        self
    }
}

impl Relaxer for TorsionMcRelaxer {
    fn build(&self, _ref_coords: &[[F; 3]]) -> Box<dyn RelaxerRunner> {
        Box::new(TorsionMcRelaxerRunner {
            bonds: self.bonds.clone(),
            max_delta: self.max_delta,
            steps: self.steps,
            temperature: self.temperature,
            self_avoidance_radius: self.self_avoidance_radius,
            excluded_pairs: self.excluded_pairs.clone(),
            attempts: 0,
            accepts: 0,
        })
    }
}

/// Compute the set of excluded atom pairs (1-2, 1-3, and 1-4 bonded neighbors).
///
/// In molecular mechanics, 1-4 interactions are always treated specially (scaled
/// or excluded). For self-avoidance, we exclude them because their distances are
/// geometrically constrained by bond lengths, angles, and torsions — penalizing
/// them creates a huge baseline penalty that masks real overlaps.
///
/// Uses positional indices (0-based), stored as `(min, max)` for canonical order.
pub fn compute_excluded_pairs(graph: &MolGraph) -> HashSet<(usize, usize)> {
    let id_to_idx = atom_id_to_index(graph);

    // Build adjacency list (AtomId → Vec<AtomId>) for BFS.
    let atom_ids: Vec<_> = graph.atoms().map(|(id, _)| id).collect();
    let mut adj: std::collections::HashMap<_, Vec<_>> = std::collections::HashMap::new();
    for &id in &atom_ids {
        adj.insert(id, graph.neighbors(id).collect());
    }

    let mut excluded = HashSet::new();

    // For each atom, BFS up to depth 3 to find all 1-2, 1-3, and 1-4 neighbors.
    for &root in &atom_ids {
        let root_idx = id_to_idx[&root];
        // depth 1 neighbors
        for &n1 in adj.get(&root).unwrap_or(&Vec::new()) {
            let n1_idx = id_to_idx[&n1];
            excluded.insert((root_idx.min(n1_idx), root_idx.max(n1_idx)));
            // depth 2 neighbors
            for &n2 in adj.get(&n1).unwrap_or(&Vec::new()) {
                if n2 == root {
                    continue;
                }
                let n2_idx = id_to_idx[&n2];
                excluded.insert((root_idx.min(n2_idx), root_idx.max(n2_idx)));
                // depth 3 neighbors (1-4 pairs)
                for &n3 in adj.get(&n2).unwrap_or(&Vec::new()) {
                    if n3 == root || n3 == n1 {
                        continue;
                    }
                    let n3_idx = id_to_idx[&n3];
                    excluded.insert((root_idx.min(n3_idx), root_idx.max(n3_idx)));
                }
            }
        }
    }

    excluded
}

// ── TorsionMcRelaxerRunner ─────────────────────────────────────────────────────────

struct TorsionMcRelaxerRunner {
    bonds: Vec<RotatableBond>,
    max_delta: F,
    steps: usize,
    temperature: F,
    self_avoidance_radius: F,
    excluded_pairs: HashSet<(usize, usize)>,
    attempts: usize,
    accepts: usize,
}

/// Compute intra-molecular self-avoidance penalty.
///
/// Uses the same quartic form as the packer's `fparc`: `(d² - cutoff²)²` when
/// `d < cutoff`. This ensures intra-molecular overlap penalties have comparable
/// magnitude to inter-molecular penalties, preventing the optimizer from trading
/// self-intersection for inter-molecular gains.
pub fn self_avoidance_penalty(
    coords: &[[F; 3]],
    radius: F,
    excluded: &HashSet<(usize, usize)>,
) -> F {
    let cutoff = 2.0 * radius;
    let cutoff_sq = cutoff * cutoff;
    let n = coords.len();
    let mut penalty: F = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            if excluded.contains(&(i, j)) {
                continue;
            }
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < cutoff_sq {
                // Quartic form matching packer's fparc: (d² - r²)²
                let gap = dist_sq - cutoff_sq; // negative when overlapping
                penalty += gap * gap;
            }
        }
    }

    penalty
}

impl RelaxerRunner for TorsionMcRelaxerRunner {
    fn on_iter(
        &mut self,
        coords: &[[F; 3]],
        f_current: F,
        evaluate: &mut dyn FnMut(&[[F; 3]]) -> F,
        rng: &mut dyn RngCore,
    ) -> Option<Vec<[F; 3]>> {
        if self.bonds.is_empty() {
            return None;
        }

        let use_sa = self.self_avoidance_radius > 0.0;

        let mut best = coords.to_vec();
        let mut best_f = if use_sa {
            f_current
                + self_avoidance_penalty(&best, self.self_avoidance_radius, &self.excluded_pairs)
        } else {
            f_current
        };
        let mut any_accepted = false;

        for _ in 0..self.steps {
            let bond_idx = rng_usize(rng, self.bonds.len());
            let bond = &self.bonds[bond_idx];
            let delta = (rng_f(rng) * 2.0 - 1.0) * self.max_delta;

            let mut trial = best.clone();
            rotate_around_bond(&mut trial, bond, delta);
            recenter(&mut trial);

            self.attempts += 1;
            let f_packer = evaluate(&trial);
            let f_trial = if use_sa {
                f_packer
                    + self_avoidance_penalty(
                        &trial,
                        self.self_avoidance_radius,
                        &self.excluded_pairs,
                    )
            } else {
                f_packer
            };

            if metropolis_accept(f_trial, best_f, self.temperature, rng) {
                best = trial;
                best_f = f_trial;
                self.accepts += 1;
                any_accepted = true;
            }
        }

        if any_accepted { Some(best) } else { None }
    }

    fn acceptance_rate(&self) -> F {
        if self.attempts == 0 {
            0.0
        } else {
            self.accepts as F / self.attempts as F
        }
    }
}

// ── Geometry helpers ────────────────────────────────────────────────────────

/// Rodrigues' rotation: rotate downstream atoms around the bond axis (j→k).
fn rotate_around_bond(coords: &mut [[F; 3]], bond: &RotatableBond, angle: F) {
    let j = &coords[bond.j];
    let k = &coords[bond.k];

    // Axis direction (j → k), normalized
    let axis = [k[0] - j[0], k[1] - j[1], k[2] - j[2]];
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if len < near_zero_norm_floor() {
        return;
    }
    let u = [axis[0] / len, axis[1] / len, axis[2] / len];

    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let origin = *j;

    for &idx in &bond.downstream {
        let p = [
            coords[idx][0] - origin[0],
            coords[idx][1] - origin[1],
            coords[idx][2] - origin[2],
        ];

        // Rodrigues: v' = v cos θ + (u × v) sin θ + u (u·v)(1 − cos θ)
        let udotp = u[0] * p[0] + u[1] * p[1] + u[2] * p[2];
        let cross = [
            u[1] * p[2] - u[2] * p[1],
            u[2] * p[0] - u[0] * p[2],
            u[0] * p[1] - u[1] * p[0],
        ];

        coords[idx] = [
            p[0] * cos_a + cross[0] * sin_a + u[0] * udotp * (1.0 - cos_a) + origin[0],
            p[1] * cos_a + cross[1] * sin_a + u[1] * udotp * (1.0 - cos_a) + origin[1],
            p[2] * cos_a + cross[2] * sin_a + u[2] * udotp * (1.0 - cos_a) + origin[2],
        ];
    }
}

/// Re-center coordinates at their geometric center.
fn recenter(coords: &mut [[F; 3]]) {
    let n = coords.len() as F;
    if n < 1.0 {
        return;
    }
    let cx: F = coords.iter().map(|p| p[0]).sum::<F>() / n;
    let cy: F = coords.iter().map(|p| p[1]).sum::<F>() / n;
    let cz: F = coords.iter().map(|p| p[2]).sum::<F>() / n;
    for p in coords.iter_mut() {
        p[0] -= cx;
        p[1] -= cy;
        p[2] -= cz;
    }
}

/// Metropolis acceptance criterion.
fn metropolis_accept(f_trial: F, f_current: F, temperature: F, rng: &mut dyn RngCore) -> bool {
    if f_trial <= f_current {
        return true;
    }
    if temperature <= 0.0 {
        return false;
    }
    let delta = (f_trial - f_current) / temperature;
    let prob = (-delta).exp();
    rng_f(rng) < prob
}

/// Generate a random F in [0, 1).
fn rng_f(rng: &mut dyn RngCore) -> F {
    uniform01_core(rng)
}

/// Generate a random usize in [0, max).
fn rng_usize(rng: &mut dyn RngCore, max: usize) -> usize {
    (rng.next_u32() as usize) % max
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::molgraph::Atom;
    use molrs::rotatable::RotatableBond;

    /// Build a chain MolGraph (topology only, no coords needed) + zigzag coords.
    fn chain(n: usize) -> (MolGraph, Vec<[F; 3]>) {
        let mut g = MolGraph::new();
        let mut ids = Vec::new();
        for _ in 0..n {
            ids.push(g.add_atom(Atom::new()));
        }
        for i in 0..n - 1 {
            g.add_bond(ids[i], ids[i + 1]).expect("add chain bond");
        }
        (g, zigzag_coords(n))
    }

    /// Zigzag coords with tetrahedral bond angles in the xz-plane.
    fn zigzag_coords(n: usize) -> Vec<[F; 3]> {
        let bond_len: F = 1.54;
        let theta = (109.5_f64 * std::f64::consts::PI / 180.0) as F;
        // Half-angle: α = (π − θ)/2 gives correct bond projections for zigzag.
        let alpha = (std::f64::consts::PI as F - theta) / 2.0;
        let dx = bond_len * alpha.cos();
        let dz = bond_len * alpha.sin();

        let mut coords = Vec::with_capacity(n);
        coords.push([0.0, 0.0, 0.0]);
        for i in 1..n {
            let prev = coords[i - 1];
            let sign: F = if i % 2 == 0 { 1.0 } else { -1.0 };
            coords.push([prev[0] + dx, 0.0, prev[2] + sign * dz]);
        }
        coords
    }

    #[test]
    fn test_rotate_around_bond_preserves_distance() {
        let coords = zigzag_coords(3);
        let mut trial = coords.clone();
        let bond = RotatableBond {
            j: 0,
            k: 1,
            downstream: vec![1, 2],
        };

        rotate_around_bond(&mut trial, &bond, PI as F / 4.0);

        assert!((distance(&coords[0], &coords[1]) - distance(&trial[0], &trial[1])).abs() < 1e-6);
        assert!((distance(&coords[1], &coords[2]) - distance(&trial[1], &trial[2])).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_180_flips_off_axis_atom() {
        // Bond axis along x, downstream atom off-axis at (1,1,0)
        let mut coords: Vec<[F; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let bond = RotatableBond {
            j: 0,
            k: 1,
            downstream: vec![2],
        };

        rotate_around_bond(&mut coords, &bond, PI as F);

        assert!((coords[2][0] - 1.0).abs() < 1e-6);
        assert!((coords[2][1] + 1.0).abs() < 1e-6);
        assert!(coords[2][2].abs() < 1e-6);
    }

    #[test]
    fn test_rotation_actually_moves_zigzag_atoms() {
        let coords = zigzag_coords(5);
        let mut trial = coords.clone();
        let bond = RotatableBond {
            j: 1,
            k: 2,
            downstream: vec![2, 3, 4],
        };

        rotate_around_bond(&mut trial, &bond, PI as F / 3.0);

        // Upstream atoms unchanged
        assert!((trial[0][0] - coords[0][0]).abs() < 1e-10);
        assert!((trial[1][0] - coords[1][0]).abs() < 1e-10);

        // Downstream atoms moved
        let moved = (2..5).any(|i| distance(&trial[i], &coords[i]) > 0.1);
        assert!(moved, "downstream atoms must move for zigzag geometry");
    }

    #[test]
    fn test_recenter() {
        let mut coords: Vec<[F; 3]> = vec![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
        recenter(&mut coords);

        let cx: F = coords.iter().map(|p| p[0]).sum::<F>() / 2.0;
        assert!(cx.abs() < 1e-10);
    }

    #[test]
    fn test_metropolis_always_accepts_lower() {
        let mut rng = rand::rng();
        assert!(metropolis_accept(5.0, 10.0, 1.0, &mut rng));
    }

    #[test]
    fn test_metropolis_zero_temp_rejects_higher() {
        let mut rng = rand::rng();
        assert!(!metropolis_accept(10.0, 5.0, 0.0, &mut rng));
    }

    #[test]
    fn test_new_detects_rotatable_bonds() {
        let (g, _) = chain(5);
        let hook = TorsionMcRelaxer::new(&g);
        assert_eq!(hook.bonds.len(), 2);
        assert_eq!(hook.steps, 10);
    }

    #[test]
    fn test_runner_modifies_zigzag() {
        let (g, coords) = chain(5);
        let hook = TorsionMcRelaxer::new(&g)
            .with_temperature(1.0)
            .with_steps(5);

        let mut runner = hook.build(&coords);
        let mut rng = rand::rng();

        let result = runner.on_iter(&coords, 100.0, &mut |_| 50.0, &mut rng);
        assert!(result.is_some());

        let new_coords = result.unwrap();
        let changed = new_coords
            .iter()
            .zip(coords.iter())
            .any(|(a, b)| distance(a, b) > 1e-6);
        assert!(changed, "zigzag coords must change after torsion MC");
    }

    fn distance(a: &[F; 3], b: &[F; 3]) -> F {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[test]
    fn test_self_avoidance_penalty_overlapping_atoms() {
        // Two atoms at distance 1.0, cutoff = 2*1.0 = 2.0 → overlap
        // Quartic form: (d² - cutoff²)² = (1.0 - 4.0)² = 9.0
        let coords: Vec<[F; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let excluded = HashSet::new();
        let penalty = self_avoidance_penalty(&coords, 1.0, &excluded);
        assert!((penalty - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_self_avoidance_penalty_no_overlap() {
        // Two atoms at distance 3.0, cutoff = 2*1.0 = 2.0 → no overlap
        let coords: Vec<[F; 3]> = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let excluded = HashSet::new();
        let penalty = self_avoidance_penalty(&coords, 1.0, &excluded);
        assert!(penalty.abs() < 1e-10);
    }

    #[test]
    fn test_self_avoidance_penalty_excluded_pairs_skipped() {
        // Overlapping atoms, but pair is excluded → no penalty
        let coords: Vec<[F; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut excluded = HashSet::new();
        excluded.insert((0, 1));
        let penalty = self_avoidance_penalty(&coords, 1.0, &excluded);
        assert!(penalty.abs() < 1e-10);
    }

    #[test]
    fn test_self_avoidance_penalty_disabled_when_radius_zero() {
        let (g, coords) = chain(5);
        let hook = TorsionMcRelaxer::new(&g)
            .with_temperature(0.0)
            .with_steps(5);

        // radius=0.0 (default), evaluate always returns 0 → all moves accepted
        let mut runner = hook.build(&coords);
        let mut rng = rand::rng();
        let result = runner.on_iter(&coords, 0.0, &mut |_| 0.0, &mut rng);
        assert!(result.is_some());
    }

    #[test]
    fn test_runner_with_self_avoidance_rejects_overlapping_moves() {
        // Build a 5-atom chain. With temperature=0, only improvements accepted.
        // The evaluate closure returns 0 (no external penalty), so only
        // self-avoidance penalty drives acceptance.
        let (g, coords) = chain(5);
        let hook = TorsionMcRelaxer::new(&g)
            .with_self_avoidance(1.0)
            .with_temperature(0.0) // greedy: only accept improvements
            .with_steps(50);

        let mut runner = hook.build(&coords);
        let mut rng = rand::rng();

        // Run with evaluate returning 0 — self-avoidance is the only signal.
        let result = runner.on_iter(&coords, 0.0, &mut |_| 0.0, &mut rng);

        // With greedy acceptance, the self-avoidance penalty should only
        // decrease or stay the same. Any accepted move must not increase
        // the penalty.
        if let Some(new_coords) = &result {
            let excluded = compute_excluded_pairs(&g);
            let p_old = self_avoidance_penalty(&coords, 1.0, &excluded);
            let p_new = self_avoidance_penalty(new_coords, 1.0, &excluded);
            assert!(
                p_new <= p_old + 1e-6,
                "greedy MC must not increase penalty: {p_new} > {p_old}"
            );
        }
    }
}
