//! Soft (penetrable) packing **potential** — a [`Potential`] *form*.
//!
//! [`SoftPotential`] is a pure, distance-based energy: given coordinates and a
//! **pre-resolved pair list**, it returns energy + forces. It does NOT build a
//! neighbour list and does NOT know about periodicity — exactly like
//! [`PairLJCut`](crate::ff::potential::pair::PairLJCut), every pair `(i, j)` is
//! resolved once (by a neighbour list) and stored, together with the per-pair
//! minimum-image **shift** the neighbour list reported. The kernel only ever
//! evaluates `d = x_i - x_j - shift`.
//!
//! Periodicity lives in the **builder** [`SoftSpec::build_potential`], which uses
//! molrs's own [`NeighborQuery`](crate::core::spatial::neighbors::NeighborQuery)
//! to resolve the non-bonded pairs (excluding 1-2 / 1-3 neighbours) for a given
//! configuration + box. A minimizer rebuilds the potential periodically as the
//! atoms move.

use std::collections::HashSet;

use molrs::core::spatial::neighbors::NeighborQuery;
use molrs::core::spatial::region::simbox::SimBox;
use molrs::ff::potential::Potential;
use molrs::store::frame::Frame;
use molrs::types::F;
use ndarray::{ArrayView2, array};

/// Harmonic distance term: atoms `(i, j)`, equilibrium `t`, per-pair image
/// `shift` (so `d = x_i - x_j - shift`).
pub type HarmTerm = (usize, usize, F, [F; 3]);
/// Non-bonded pair: atoms `(i, j)` with per-pair image `shift`.
pub type NbTerm = (usize, usize, [F; 3]);

// SoftPotential (pure: pairs in, energy/forces out)

/// Pure soft packing potential over pre-resolved pairs. No box, no neighbour
/// list — build it with [`SoftSpec::build_potential`].
#[derive(Debug, Clone)]
pub struct SoftPotential {
    bonds: Vec<HarmTerm>,
    angles: Vec<HarmTerm>,
    nb: Vec<NbTerm>,
    sigma: F,
    a_rep: F,
    b_attract: F,
    rcut: F,
    k_bond: F,
    k_ang: F,
}

impl SoftPotential {
    /// Construct directly from resolved terms (the builder is the usual entry).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bonds: Vec<HarmTerm>,
        angles: Vec<HarmTerm>,
        nb: Vec<NbTerm>,
        sigma: F,
        a_rep: F,
        b_attract: F,
        rcut: F,
        k_bond: F,
        k_ang: F,
    ) -> Self {
        Self {
            bonds,
            angles,
            nb,
            sigma,
            a_rep,
            b_attract,
            rcut,
            k_bond,
            k_ang,
        }
    }

    /// Number of stored pair terms (bonds + angles + non-bonded).
    pub fn n_pairs(&self) -> usize {
        self.bonds.len() + self.angles.len() + self.nb.len()
    }
}

impl Potential for SoftPotential {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let mut forces = vec![0.0; coords.len()];
        let mut e: F = 0.0;
        for &(i, j, t, shift) in &self.bonds {
            e += harmonic(coords, &mut forces, i, j, t, self.k_bond, shift);
        }
        for &(i, j, t, shift) in &self.angles {
            e += harmonic(coords, &mut forces, i, j, t, self.k_ang, shift);
        }
        for &(i, j, shift) in &self.nb {
            let d = disp(coords, i, j, shift);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < 1e-18 {
                continue;
            }
            let r = r2.sqrt();
            let dedr = if r < self.sigma {
                e += self.a_rep * (self.sigma - r) * (self.sigma - r);
                -2.0 * self.a_rep * (self.sigma - r)
            } else if self.b_attract > 0.0 && r < self.rcut {
                e += -self.b_attract * (r - self.sigma) * (self.rcut - r);
                -self.b_attract * (self.rcut + self.sigma - 2.0 * r)
            } else {
                continue;
            };
            let c = -dedr / r;
            for ax in 0..3 {
                forces[3 * i + ax] += c * d[ax];
                forces[3 * j + ax] -= c * d[ax];
            }
        }
        (e, forces)
    }
}

/// `d = x_i - x_j - shift`.
#[inline]
fn disp(coords: &[F], i: usize, j: usize, shift: [F; 3]) -> [F; 3] {
    [
        coords[3 * i] - coords[3 * j] - shift[0],
        coords[3 * i + 1] - coords[3 * j + 1] - shift[1],
        coords[3 * i + 2] - coords[3 * j + 2] - shift[2],
    ]
}

fn harmonic(coords: &[F], forces: &mut [F], i: usize, j: usize, t: F, k: F, shift: [F; 3]) -> F {
    let d = disp(coords, i, j, shift);
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if r2 < 1e-18 {
        return 0.0;
    }
    let r = r2.sqrt();
    let e = k * (r - t) * (r - t);
    let dedr = 2.0 * k * (r - t);
    let c = -dedr / r;
    for ax in 0..3 {
        forces[3 * i + ax] += c * d[ax];
        forces[3 * j + ax] -= c * d[ax];
    }
    e
}

// SoftSpec (config + builder; owns the neighbour list / periodicity)

/// Topology + parameters for a [`SoftPotential`]. [`SoftSpec::build_potential`]
/// resolves the non-bonded pairs with molrs's neighbour list for a given
/// configuration + box and produces the pure potential.
#[derive(Debug, Clone)]
pub struct SoftSpec {
    bonds: Vec<(usize, usize)>,
    angles: Vec<(usize, usize)>,
    excluded: HashSet<(usize, usize)>,
    sigma: F,
    a_rep: F,
    b_attract: F,
    rcut: F,
    k_bond: F,
    k_ang: F,
}

impl SoftSpec {
    /// Detect 1-2 (bonds) and 1-3 (angles) topology by reading the frame's
    /// **bonds** and **angles** blocks directly (no neighbour-graph round-trip):
    /// 1-2 = bonds (`atomi`,`atomj`), 1-3 = angle ends (`atomi`,`atomk`). This is
    /// the robust path for a packed/assembled frame whose connectivity lives in
    /// its topology blocks.
    pub fn from_frame(frame: &Frame) -> Self {
        let bonds = read_pairs(frame, "bonds", "atomi", "atomj");
        let angles = read_pairs(frame, "angles", "atomi", "atomk");
        let mut excluded: std::collections::HashSet<(usize, usize)> =
            bonds.iter().copied().collect();
        for &k in &angles {
            excluded.insert(k);
        }
        Self {
            bonds,
            angles,
            excluded,
            sigma: 2.6,
            a_rep: 8.0,
            b_attract: 0.0,
            rcut: 5.0,
            k_bond: 50.0,
            k_ang: 8.0,
        }
    }

    pub fn with_sigma(mut self, s: F) -> Self {
        self.sigma = s;
        self
    }
    pub fn with_repulsion(mut self, a: F) -> Self {
        self.a_rep = a;
        self
    }
    pub fn with_attraction(mut self, b: F) -> Self {
        self.b_attract = b;
        self
    }
    pub fn with_rcut(mut self, r: F) -> Self {
        self.rcut = r;
        self
    }
    pub fn with_bond_k(mut self, k: F) -> Self {
        self.k_bond = k;
        self
    }
    pub fn with_angle_k(mut self, k: F) -> Self {
        self.k_ang = k;
        self
    }

    /// Resolve the pairs for `coords` (+ optional periodic cubic box) with molrs's
    /// neighbour list and return the pure [`SoftPotential`]. Non-bonded pairs use
    /// cutoff `max(rcut, sigma)`, exclude 1-2 / 1-3 neighbours, and carry the
    /// neighbour list's minimum-image shift; bond / 1-3 `r0`/`a0` (and shift) come
    /// from the minimum-image displacement in this configuration.
    /// Build the FIXED bonded terms (bonds + 1-3) from a reference configuration.
    /// `r0`/`a0` and per-pair image shift come from the minimum-image
    /// displacement here and must NOT be recomputed as the system relaxes — a
    /// minimizer computes these ONCE, then rebuilds only the non-bonded pairs.
    pub fn build_bonded(
        &self,
        ref_coords: &[[F; 3]],
        box_edge: Option<F>,
    ) -> (Vec<HarmTerm>, Vec<HarmTerm>) {
        let bonds = self
            .bonds
            .iter()
            .map(|&(i, j)| harm_term(ref_coords, i, j, box_edge))
            .collect();
        let angles = self
            .angles
            .iter()
            .map(|&(i, j)| harm_term(ref_coords, i, j, box_edge))
            .collect();
        (bonds, angles)
    }

    /// Resolve the NON-bonded pairs (with minimum-image shift) for `coords` via
    /// molrs's neighbour list, excluding 1-2 / 1-3 neighbours.
    pub fn build_nb(&self, coords: &[[F; 3]], box_edge: Option<F>) -> Vec<NbTerm> {
        let n = coords.len();
        let mut flat = Vec::with_capacity(n * 3);
        for c in coords {
            flat.extend_from_slice(c);
        }
        let view = ArrayView2::from_shape((n, 3), &flat).expect("(n,3)");
        let cutoff = self.rcut.max(self.sigma);
        let nl = match box_edge {
            Some(l) => {
                let sb = SimBox::cube(l, array![0.0, 0.0, 0.0], [true, true, true]).expect("cube");
                NeighborQuery::new(&sb, view, cutoff).query_self()
            }
            None => NeighborQuery::free(view, cutoff).query_self(),
        };
        let qi = nl.query_point_indices();
        let qj = nl.point_indices();
        let vecs = nl.vectors(); // minimum-image displacement (j - i)
        let mut nb = Vec::with_capacity(nl.n_pairs());
        for k in 0..nl.n_pairs() {
            let i = qi[k] as usize;
            let j = qj[k] as usize;
            if i == j || self.excluded.contains(&(i.min(j), i.max(j))) {
                continue;
            }
            // kernel d = x_i - x_j - shift must equal the i->j minimum image
            // (= -vectors). shift = (x_i - x_j) + vectors.
            let shift = [
                coords[i][0] - coords[j][0] + vecs[[k, 0]],
                coords[i][1] - coords[j][1] + vecs[[k, 1]],
                coords[i][2] - coords[j][2] + vecs[[k, 2]],
            ];
            nb.push((i, j, shift));
        }
        nb
    }

    /// One-shot build (bonded `r0`/`a0` AND non-bonded pairs from `coords`).
    /// Convenience for single evaluations; a minimizer should instead fix the
    /// bonded terms via [`build_bonded`](Self::build_bonded) and rebuild only the
    /// non-bonded pairs via [`build_nb`](Self::build_nb).
    pub fn build_potential(&self, coords: &[[F; 3]], box_edge: Option<F>) -> SoftPotential {
        let (bonds, angles) = self.build_bonded(coords, box_edge);
        let nb = self.build_nb(coords, box_edge);
        SoftPotential::new(
            bonds,
            angles,
            nb,
            self.sigma,
            self.a_rep,
            self.b_attract,
            self.rcut,
            self.k_bond,
            self.k_ang,
        )
    }

    /// Soft-potential parameters, in the order [`SoftPotential::new`] expects:
    /// `(sigma, a_rep, b_attract, rcut, k_bond, k_ang)`.
    pub fn params(&self) -> (F, F, F, F, F, F) {
        (
            self.sigma,
            self.a_rep,
            self.b_attract,
            self.rcut,
            self.k_bond,
            self.k_ang,
        )
    }
}

/// Harmonic term `(i, j, r0, shift)` from the minimum-image displacement of
/// `(i, j)` in `coords` (raw when `box_edge` is `None`).
fn harm_term(coords: &[[F; 3]], i: usize, j: usize, box_edge: Option<F>) -> HarmTerm {
    let mut d = [
        coords[i][0] - coords[j][0],
        coords[i][1] - coords[j][1],
        coords[i][2] - coords[j][2],
    ];
    let mut shift = [0.0; 3];
    if let Some(l) = box_edge {
        for a in 0..3 {
            let image = l * (d[a] / l).round();
            shift[a] = image; // d_kernel = raw - shift = minimum image
            d[a] -= image;
        }
    }
    let r0 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    (i, j, r0, shift)
}

/// Read a topology block's two atom-index columns as canonical `(min, max)`
/// pairs (empty if the block/columns are absent).
fn read_pairs(frame: &Frame, block: &str, col_a: &str, col_b: &str) -> Vec<(usize, usize)> {
    let Some(b) = frame.get(block) else {
        return Vec::new();
    };
    let (Some(a_col), Some(b_col)) = (b.get_uint(col_a), b.get_uint(col_b)) else {
        return Vec::new();
    };
    a_col
        .iter()
        .zip(b_col.iter())
        .map(|(&i, &j)| {
            let (i, j) = (i as usize, j as usize);
            (i.min(j), i.max(j))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::store::block::Block;
    use molrs::types::U;
    use ndarray::Array1;

    /// A frame carrying the bonds/angles topology blocks of a linear n-atom chain
    /// (1-2 = (i, i+1); 1-3 angle ends = (i, i+2)). No Atomistic involved.
    fn chain_frame(n: usize) -> Frame {
        let mut frame = Frame::new();
        let (bi, bj): (Vec<U>, Vec<U>) = (0..n - 1).map(|i| (i as U, (i + 1) as U)).unzip();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(bi).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(bj).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);
        let (ai, ak): (Vec<U>, Vec<U>) = (0..n - 2).map(|i| (i as U, (i + 2) as U)).unzip();
        let mut angles = Block::new();
        angles
            .insert("atomi", Array1::from_vec(ai).into_dyn())
            .unwrap();
        angles
            .insert("atomk", Array1::from_vec(ak).into_dyn())
            .unwrap();
        frame.insert("angles", angles);
        frame
    }

    #[test]
    fn spec_detects_bonds_and_angles() {
        let s = SoftSpec::from_frame(&chain_frame(5));
        assert_eq!(s.bonds.len(), 4);
        assert_eq!(s.angles.len(), 3);
    }

    fn fd_check(box_edge: Option<F>) {
        let xv: Vec<[F; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.4, 0.3, 0.0],
            [2.9, -0.2, 0.4],
            [4.0, 0.5, 0.1],
            [5.2, 0.0, -0.3],
        ];
        let pot = SoftSpec::from_frame(&chain_frame(5))
            .with_attraction(1.0)
            .build_potential(&xv, box_edge);
        let flat: Vec<F> = xv.iter().flatten().copied().collect();
        let (_e, f) = pot.calc_energy_forces(&flat);
        let h = 1e-6;
        for k in 0..flat.len() {
            let mut xp = flat.clone();
            let mut xm = flat.clone();
            xp[k] += h;
            xm[k] -= h;
            let num = (pot.calc_energy_forces(&xp).0 - pot.calc_energy_forces(&xm).0) / (2.0 * h);
            assert!(
                (f[k] + num).abs() < 1e-3,
                "force[{k}]={} -num={}",
                f[k],
                -num
            );
        }
    }

    #[test]
    fn forces_match_finite_difference_free() {
        fd_check(None);
    }

    #[test]
    fn forces_match_finite_difference_periodic() {
        fd_check(Some(50.0));
    }
}
