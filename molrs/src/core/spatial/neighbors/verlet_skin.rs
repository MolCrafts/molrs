//! A [`NeighborList`] with Verlet skin — `VerletSkin(search)`.
//!
//! This **is** a neighbour list: same job as [`NeighborList`] (find pairs under
//! a cutoff), plus the LAMMPS `neigh_modify every/delay/check` gate and a skin
//! margin so the index can stay valid across several steps. Pair *search* is
//! delegated to the wrapped [`NeighborList`] (LinkCell, BruteForce, …).
//!
//! Constructed as `VerletSkin::new(NeighborList::new(cutoff + skin), …)`.
//!
//! The list is built at `r_build = cutoff + skin` and stays complete out to
//! `cutoff` while no atom has moved more than `skin/2` since the last build
//! (strict `>`). Displacements use the **raw** difference `x - x_hold`, never
//! a minimum image. Crossing half the smallest perpendicular cell width
//! raises — positions must stay unwrapped.
//!
//! Force / analysis callers must use [`for_each_pair_at`](VerletSkin::for_each_pair_at)
//! (or the stored `(i, j)` edges plus a fresh MIC). Never stream the inner
//! [`NeighborList::for_each_pair`] for live forces — that reads geometry
//! frozen at the last rebuild.
//!
//! Edge *order* is not part of the contract. The *set* of half-shell `(i, j)`
//! pairs (`i < j` from the backend) plus the live count is.
//!
//! References:
//!     LAMMPS `neigh_modify` / `Neighbor::decide` / `Neighbor::check_distance`
//!     / `Neighbor::init`. Nordlund lecture notes for the two-atom skin
//!     criterion. Allen & Tildesley for cell lists.

use std::fmt;

use ndarray::{Array2, ArrayView2};

use crate::spatial::simbox::SimBox;
use crate::types::{F, FNx3};

use super::NeighborList;

/// Failures from Verlet-skin construction or the unwrapped-position guard.
#[derive(Debug)]
pub enum SkinError {
    /// A constructor or update argument is out of domain.
    Invalid(String),
    /// Completeness / unwrapped-position guard failed.
    Guard(String),
}

impl fmt::Display for SkinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) | Self::Guard(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for SkinError {}

/// Rebuild policy for [`VerletSkin`].
///
/// The search backend is **not** selected here — pass a ready [`NeighborList`]
/// (`NeighborList::new` or [`NeighborList::brute_force`]) into
/// [`VerletSkin::new`].
#[derive(Clone, Copy, Debug)]
pub struct NeighborPolicy {
    /// Verlet skin `s` in Å. The list is built at `cutoff + s`.
    pub skin: F,
    /// Attempt a rebuild only when `ago` is a multiple of this (steps).
    pub every: usize,
    /// Attempt no rebuild until at least this many steps since the last build.
    /// Must be a multiple of `every`. `0` is always legal.
    pub delay: usize,
    /// Rebuild only when the max displacement exceeds `skin/2`.
    /// `false` rebuilds on cadence alone and skips the unwrapped-position guard.
    pub check: bool,
}

impl Default for NeighborPolicy {
    fn default() -> Self {
        Self {
            skin: 0.0,
            every: 1,
            delay: 0,
            check: true,
        }
    }
}

/// One half-shell neighbour edge (backend `i < j` convention).
#[derive(Clone, Copy, Debug)]
pub struct SkinPair {
    /// Source atom index.
    pub i: u32,
    /// Target atom index.
    pub j: u32,
}

/// Minimum-image Verlet-skin neighbour list over a core [`NeighborList`].
///
/// Two entry points: [`rebuild`](Self::rebuild) forces a build,
/// [`update`](Self::update) applies the `every` / `delay` / `check` policy.
pub struct VerletSkin {
    /// Interaction cutoff `r_cut` in Å.
    pub cutoff: F,
    /// Verlet skin in Å.
    pub skin: F,
    /// `every` gate (steps).
    pub every: usize,
    /// `delay` gate (steps).
    pub delay: usize,
    /// Displacement check switch.
    pub check: bool,
    r_build: F,
    simbox: SimBox,
    search: NeighborList,
    edges: Vec<SkinPair>,
    /// Live edge count (`== edges.len()`).
    pub num_edges: usize,
    /// Rebuilds since construction; the initial build is not one of them.
    pub rebuild_count: usize,
    /// Steps since the last build (LAMMPS `ago`).
    pub ago: usize,
    /// Rebuilds that fired at the first permitted opportunity.
    pub ndanger: usize,
    half_skin_sq: F,
    wrap_guard_sq: F,
    danger_ago: usize,
    x_hold: FNx3,
}

impl VerletSkin {
    /// Wrap `search` (must be built at `cutoff + policy.skin`) with Verlet policy.
    ///
    /// The initial build is not counted in [`Self::rebuild_count`].
    pub fn new(
        mut search: NeighborList,
        cutoff: F,
        policy: NeighborPolicy,
        positions: ArrayView2<'_, F>,
        simbox: SimBox,
    ) -> Result<Self, SkinError> {
        if cutoff <= 0.0 {
            return Err(SkinError::Invalid(format!(
                "cutoff must be > 0 Å, got {cutoff} Å"
            )));
        }
        if policy.skin < 0.0 {
            return Err(SkinError::Invalid(format!(
                "skin must be >= 0 Å, got {} Å",
                policy.skin
            )));
        }
        if policy.every < 1 {
            return Err(SkinError::Invalid(format!(
                "every must be an integer >= 1 step, got {}",
                policy.every
            )));
        }
        if !policy.delay.is_multiple_of(policy.every) {
            return Err(SkinError::Invalid(format!(
                "delay {} steps must be a multiple of every {} steps; LAMMPS Neighbor::init \
                 rejects the same pair, because the danger threshold max(every, delay) would \
                 be an ago the gate never permits, leaving ndanger silently dead.",
                policy.delay, policy.every
            )));
        }
        if positions.ncols() != 3 {
            return Err(SkinError::Invalid(format!(
                "positions must have shape (N, 3), got {:?}",
                positions.shape()
            )));
        }
        if !simbox.is_cell_defined() {
            return Err(SkinError::Invalid(
                "VerletSkin requires a geometrically defined cell".into(),
            ));
        }

        let r_build = cutoff + policy.skin;
        let want = search.cutoff();
        if (want - r_build).abs() > 1e-12 {
            return Err(SkinError::Invalid(format!(
                "NeighborList cutoff {want} Å must equal cutoff + skin = {r_build} Å \
                 (cutoff {cutoff} Å + skin {} Å)",
                policy.skin
            )));
        }

        let widths = simbox.nearest_plane_distance();
        let min_width = widths.iter().copied().fold(F::INFINITY, F::min);
        let half_width = 0.5 * min_width;
        if !simbox.is_free() && r_build > half_width {
            return Err(SkinError::Invalid(format!(
                "r_build {r_build} Å (cutoff {cutoff} Å + skin {} Å) exceeds half the \
                 minimum perpendicular cell width ({half_width:.3} Å); the minimum-image \
                 reduction would silently drop pairs inside the cutoff.",
                policy.skin
            )));
        }

        let n_atoms = positions.nrows();
        search.build(positions, &simbox);

        let mut skin = Self {
            cutoff,
            skin: policy.skin,
            every: policy.every,
            delay: policy.delay,
            check: policy.check,
            r_build,
            simbox,
            search,
            edges: Vec::new(),
            num_edges: 0,
            rebuild_count: 0,
            ago: 0,
            ndanger: 0,
            half_skin_sq: (0.5 * policy.skin) * (0.5 * policy.skin),
            wrap_guard_sq: half_width * half_width,
            danger_ago: policy.every.max(policy.delay),
            x_hold: Array2::zeros((n_atoms, 3)),
        };
        skin.write_edges();
        skin.hold(positions);
        Ok(skin)
    }

    /// Build radius `cutoff + skin` in Å (derived, never settable).
    pub fn r_build(&self) -> F {
        self.r_build
    }

    /// Half-shell edges from the last rebuild.
    pub fn edges(&self) -> &[SkinPair] {
        &self.edges
    }

    /// The cell this list was built against.
    pub fn simbox(&self) -> &SimBox {
        &self.simbox
    }

    /// Core pair search, built at [`Self::r_build`]. Interaction cutoff is
    /// the caller's (`Potential` still masks the skin shell).
    pub fn search(&self) -> &NeighborList {
        &self.search
    }

    fn write_edges(&mut self) {
        let r_build_sq = self.r_build * self.r_build;
        self.edges.clear();
        self.search.for_each_pair(|pair| {
            if pair.dist_sq <= 0.0 || pair.dist_sq > r_build_sq {
                return;
            }
            self.edges.push(SkinPair {
                i: pair.i,
                j: pair.j,
            });
        });
        self.num_edges = self.edges.len();
    }

    fn hold(&mut self, positions: ArrayView2<'_, F>) {
        self.ago = 0;
        self.x_hold.assign(&positions);
    }

    fn build_at(&mut self, positions: ArrayView2<'_, F>) -> Result<(), SkinError> {
        self.require_shape(positions)?;
        self.search.build(positions, &self.simbox);
        self.write_edges();
        self.hold(positions);
        Ok(())
    }

    fn require_shape(&self, positions: ArrayView2<'_, F>) -> Result<(), SkinError> {
        let n = self.x_hold.nrows();
        if positions.shape() != [n, 3] {
            return Err(SkinError::Invalid(format!(
                "positions must have shape ({n}, 3) — this list was constructed for {n} atoms \
                 — but got {:?}",
                positions.shape()
            )));
        }
        Ok(())
    }

    /// Recompute the neighbour list at `positions` (unconditional).
    pub fn rebuild(&mut self, positions: ArrayView2<'_, F>) -> Result<(), SkinError> {
        self.build_at(positions)?;
        self.rebuild_count += 1;
        Ok(())
    }

    /// Rebuild at `positions` if the `every`/`delay`/`check` gate says so.
    ///
    /// Call once per force evaluation, at the positions being evaluated.
    /// Returns whether a rebuild happened.
    pub fn update(&mut self, positions: ArrayView2<'_, F>) -> Result<bool, SkinError> {
        self.require_shape(positions)?;
        self.ago += 1;
        let permitted = self.ago >= self.delay && self.ago.is_multiple_of(self.every);
        if !permitted {
            return Ok(false);
        }
        if !self.check {
            self.rebuild(positions)?;
            return Ok(true);
        }
        let mut max_d2 = 0.0;
        for i in 0..positions.nrows() {
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = positions[[i, k]] - self.x_hold[[i, k]];
                d2 += d * d;
            }
            if d2 > max_d2 {
                max_d2 = d2;
            }
        }
        if max_d2 >= self.wrap_guard_sq && !self.simbox.is_free() {
            return Err(SkinError::Guard(format!(
                "positions are no longer unwrapped: displacement {:.3} Å \
                 >= half min perpendicular cell width ({:.3} Å). \
                 The frozen neighbour index holds only for continuously drifting \
                 coordinates. Wrap, cell change, or blow-up are fatal.",
                max_d2.sqrt(),
                self.wrap_guard_sq.sqrt()
            )));
        }
        if max_d2 > self.half_skin_sq {
            if self.ago == self.danger_ago {
                self.ndanger += 1;
            }
            self.rebuild(positions)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Stream stored edges with **current** minimum-image geometry.
    ///
    /// `disp` is MIC `r_j − r_i` at `positions`; `r2 = |disp|²`. Pairs outside
    /// the interaction `cutoff` are still visited (caller may mask).
    pub fn for_each_pair_at<G>(&self, positions: ArrayView2<'_, F>, mut f: G)
    where
        G: FnMut(u32, u32, F, [F; 3]),
    {
        debug_assert_eq!(positions.ncols(), 3);
        let mic = self.simbox.mic();
        let Some(pos) = positions.as_slice() else {
            // Fall back for non-contiguous views (rare in MD).
            for edge in &self.edges {
                let i = edge.i as usize;
                let j = edge.j as usize;
                let pi = [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]];
                let pj = [positions[[j, 0]], positions[[j, 1]], positions[[j, 2]]];
                let disp = mic.apply([pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]]);
                let r2 = disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];
                f(edge.i, edge.j, r2, disp);
            }
            return;
        };
        for edge in &self.edges {
            let i = edge.i as usize;
            let j = edge.j as usize;
            let bi = 3 * i;
            let bj = 3 * j;
            let disp = mic.apply([
                pos[bj] - pos[bi],
                pos[bj + 1] - pos[bi + 1],
                pos[bj + 2] - pos[bi + 2],
            ]);
            let r2 = disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];
            f(edge.i, edge.j, r2, disp);
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;

    fn cube(a: F) -> SimBox {
        SimBox::cube(a, ndarray::array![0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    fn two_atoms(dx: F) -> Array2<F> {
        array![[0.0, 0.0, 0.0], [dx, 0.0, 0.0]]
    }

    fn skin_link(
        cutoff: F,
        policy: NeighborPolicy,
        pos: ArrayView2<'_, F>,
        box_a: F,
    ) -> VerletSkin {
        let search = NeighborList::new(cutoff + policy.skin);
        VerletSkin::new(search, cutoff, policy, pos, cube(box_a)).unwrap()
    }

    #[test]
    fn delay_must_be_multiple_of_every() {
        let Err(err) = VerletSkin::new(
            NeighborList::new(2.0),
            2.0,
            NeighborPolicy {
                every: 2,
                delay: 3,
                ..NeighborPolicy::default()
            },
            two_atoms(1.0).view(),
            cube(20.0),
        ) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("multiple"));
    }

    #[test]
    fn search_cutoff_must_match_r_build() {
        let Err(err) = VerletSkin::new(
            NeighborList::new(2.0),
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            two_atoms(1.0).view(),
            cube(20.0),
        ) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("cutoff + skin"));
    }

    #[test]
    fn r_build_beyond_half_width_is_rejected() {
        let Err(err) = VerletSkin::new(
            NeighborList::new(2.5),
            2.5,
            NeighborPolicy::default(),
            two_atoms(1.0).view(),
            cube(4.0),
        ) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("r_build"));
    }

    #[test]
    fn pair_inside_cutoff_is_half_shell() {
        let nl = skin_link(2.5, NeighborPolicy::default(), two_atoms(1.0).view(), 20.0);
        assert_eq!(nl.num_edges, 1);
        assert_eq!(nl.edges[0].i, 0);
        assert_eq!(nl.edges[0].j, 1);
    }

    #[test]
    fn pair_outside_cutoff_is_absent() {
        let nl = skin_link(1.0, NeighborPolicy::default(), two_atoms(1.5).view(), 20.0);
        assert_eq!(nl.num_edges, 0);
    }

    #[test]
    fn half_skin_exactly_does_not_rebuild() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            20.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let rebuilt = nl.update(pos1.view()).unwrap();
        assert!(!rebuilt);
        assert_eq!(nl.rebuild_count, 0);
    }

    #[test]
    fn half_skin_beyond_does_rebuild() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            20.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]];
        let rebuilt = nl.update(pos1.view()).unwrap();
        assert!(rebuilt);
        assert_eq!(nl.rebuild_count, 1);
    }

    #[test]
    fn for_each_pair_at_uses_live_geometry() {
        let pos0 = two_atoms(1.0);
        let nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            20.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
        let mut r2_live = 0.0;
        nl.for_each_pair_at(pos1.view(), |_i, _j, r2, disp| {
            r2_live = r2;
            assert!((disp[0] - 1.4).abs() < 1e-12);
        });
        assert!((r2_live - 1.4 * 1.4).abs() < 1e-12);
    }

    #[test]
    fn wrap_guard_fires_on_large_raw_displacement() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 0.5,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            10.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
        let err = nl.update(pos1.view()).unwrap_err();
        assert!(format!("{err}").contains("unwrapped"));
    }

    #[test]
    fn check_false_skips_wrap_guard_and_rebuilds_on_cadence() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 0.5,
                check: false,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            10.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
        let rebuilt = nl.update(pos1.view()).unwrap();
        assert!(rebuilt);
    }

    #[test]
    fn linkcell_and_bruteforce_agree_on_the_edge_set() {
        let mut pos = Array2::<F>::zeros((8, 3));
        for i in 0..8 {
            pos[[i, 0]] = (i % 2) as F * 2.0;
            pos[[i, 1]] = ((i / 2) % 2) as F * 2.0;
            pos[[i, 2]] = (i / 4) as F * 2.0;
        }
        let policy = NeighborPolicy {
            skin: 0.5,
            ..NeighborPolicy::default()
        };
        let cutoff = 3.5;
        let a = VerletSkin::new(
            NeighborList::new(cutoff + policy.skin),
            cutoff,
            policy,
            pos.view(),
            cube(20.0),
        )
        .unwrap();
        let b = VerletSkin::new(
            NeighborList::brute_force(cutoff + policy.skin),
            cutoff,
            policy,
            pos.view(),
            cube(20.0),
        )
        .unwrap();
        assert_eq!(edge_set(&a), edge_set(&b));
    }

    fn edge_set(nl: &VerletSkin) -> std::collections::BTreeSet<(u32, u32)> {
        nl.edges.iter().map(|e| (e.i, e.j)).collect()
    }
}
