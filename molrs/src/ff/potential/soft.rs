//! Soft (penetrable) packing **potential** — a [`Potential`] *form*.
//!
//! [`SoftPotential`] is a pure, distance-based energy: given coordinates and a
//! **pre-resolved pair list**, it returns energy + forces. It does NOT build a
//! neighbour list and does NOT know about periodicity — exactly like
//! [`LJCut`](crate::ff::potential::pair::LJCut), every pair `(i, j)` is
//! resolved once (by a neighbour list) and stored, together with the per-pair
//! minimum-image **shift** the neighbour list reported. The kernel only ever
//! evaluates `d = x_i - x_j - shift`.
//!
//! Periodicity lives in the **builder**, `molrs::optimize::SoftSpec`, which
//! resolves the non-bonded pairs (excluding 1-2 / 1-3 neighbours) for a given
//! configuration + box and rebuilds them as the atoms move. This module holds
//! only the kernel: no box, no neighbour list, no `optimize`.

use molrs::ff::potential::Potential;
use molrs::types::F;

/// Harmonic distance term: atoms `(i, j)`, equilibrium `t`, per-pair image
/// `shift` (so `d = x_i - x_j - shift`).
pub type HarmTerm = (usize, usize, F, [F; 3]);
/// Non-bonded pair: atoms `(i, j)` with per-pair image `shift`.
pub type NbTerm = (usize, usize, [F; 3]);

// SoftPotential (pure: pairs in, energy/forces out)

/// Pure soft packing potential over pre-resolved pairs. No box, no neighbour
/// list — build it with `molrs::optimize::SoftSpec::build_potential`.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel's forces are the analytic gradient of its energy, checked by
    /// central difference on hand-written terms.
    ///
    /// The builder that normally produces these terms lives in `optimize`, so
    /// this constructs `SoftPotential` directly — which is also the point: the
    /// kernel owes a correct gradient without a box, a neighbour list, or
    /// anything from `optimize`. Both branches of the non-bonded term are
    /// exercised: one pair inside `sigma` (repulsive) and one between `sigma`
    /// and `rcut` (attractive).
    #[test]
    fn forces_match_finite_difference() {
        let coords = [
            [0.0_f64, 0.0, 0.0],
            [1.4, 0.3, -0.2], // < sigma from atom 0  -> repulsive branch
            [3.6, 0.1, 0.4],  // sigma..rcut from 0   -> attractive branch
            [5.0, 1.1, 0.7],
        ];
        let flat: Vec<F> = coords.iter().flatten().copied().collect();

        let bonds: Vec<HarmTerm> = vec![(0, 1, 1.5, [0.0; 3]), (2, 3, 1.6, [0.0; 3])];
        let angles: Vec<HarmTerm> = vec![(0, 2, 2.9, [0.0; 3])];
        let nb: Vec<NbTerm> = vec![(0, 1, [0.0; 3]), (0, 2, [0.0; 3]), (1, 3, [0.0; 3])];

        let pot = SoftPotential::new(bonds, angles, nb, 2.6, 8.0, 0.5, 5.0, 50.0, 8.0);
        assert_eq!(pot.n_pairs(), 6, "2 bonds + 1 angle + 3 non-bonded terms");

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
                "component {k}: analytic {} vs -dE/dx {num}",
                f[k]
            );
        }
    }
}
