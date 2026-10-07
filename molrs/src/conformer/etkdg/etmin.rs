//! ETKDG distance-geometry objectives (first stage and torsion refinement).
//!
//! Port of RDKit's distance-geometry error function and its experimental-torsion
//! refinement, assembled from:
//!   * `$RDBASE/Code/DistGeom/DistViolationContribs.cpp` — distance-bound
//!     violation energy + gradient,
//!   * `$RDBASE/Code/DistGeom/ChiralViolationContribs.cpp` — signed
//!     chiral-volume violation energy + gradient (`calcChiralVolume`),
//!   * `$RDBASE/Code/DistGeom/FourthDimContribs.h` — fourth-dimension penalty
//!     used to squeeze a 4D embedding back to 3D,
//!   * `$RDBASE/Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionAngleM6.cpp` —
//!     the CrystalFF M6 experimental-torsion potential.
//!
//! BSD-3, Copyright (C) 2004-2025 Greg Landrum / Sereina Riniker and other
//! RDKit contributors.
//!
//! RDKit runs two minimizations: a 4D "first minimization" over distance +
//! chiral + fourth-dimension terms, then a 3D experimental-torsion refinement.
//! This module supplies the two objectives (energy + gradient on a flat
//! `n*dim` coordinate buffer); the minimizer is the crate's one L-BFGS,
//! [`crate::optimize::minimize_lbfgs_rms`], as for RDKit's own BFGS.
//!
//! The distance-geometry terms (distance bounds, chirality, the fourth
//! dimension) are internal to `conformer`; the torsion and planarity terms of
//! the second stage are `ff::potential` kernels.

use crate::conformer::distgeom::{
    BoundsMatrix, ChiralConstraint, ImproperConstraint, TorsionConstraint,
};
use crate::ff::forcefield::Params;
use crate::ff::potential::improper::ImproperDistance;
use crate::ff::potential::{ExplicitTerms, Potential, Potentials};
use crate::op::vec3::{cross, dot};

/// Per-atom energy threshold above which the first minimization is rejected
/// (RDKit `MAX_MINIMIZED_E_PER_ATOM`).
pub(super) const MAX_MINIMIZED_E_PER_ATOM: f64 = 0.05;

/// Distance-violation contribution (squared bounds form, RDKit
/// `DistViolationContribs`).
#[derive(Clone, Copy)]
struct DistContrib {
    i: usize,
    j: usize,
    ub2: f64,
    lb2: f64,
    weight: f64,
}

/// Chiral-volume contribution (RDKit `ChiralViolationContribs`).
#[derive(Clone, Copy)]
struct ChiralContrib {
    idx: [usize; 4],
    vol_lower: f64,
    vol_upper: f64,
    weight: f64,
}

/// The 4D first-stage force field: distance + chiral + fourth-dim penalties.
pub(super) struct FirstStageField {
    n: usize,
    dim: usize,
    dist: Vec<DistContrib>,
    chiral: Vec<ChiralContrib>,
    fourth_weight: f64,
}

impl FirstStageField {
    /// Build the first-stage field over all atom pairs (RDKit
    /// `constructForceField` with `weightChiral`, `weightFourthDim`).
    pub(super) fn build(
        bounds: &BoundsMatrix,
        chiral: &[ChiralConstraint],
        dim: usize,
        weight_chiral: f64,
        weight_fourth: f64,
    ) -> Self {
        let n = bounds.len();
        let mut dist = Vec::new();
        for i in 1..n {
            for j in 0..i {
                let u = bounds.upper(i, j);
                let l = bounds.lower(i, j);
                dist.push(DistContrib {
                    i,
                    j,
                    ub2: u * u,
                    lb2: l * l,
                    weight: 1.0,
                });
            }
        }
        let mut cc = Vec::new();
        if weight_chiral > 1e-8 {
            for c in chiral {
                cc.push(ChiralContrib {
                    idx: c.neighbors,
                    vol_lower: c.volume_lower,
                    vol_upper: c.volume_upper,
                    weight: weight_chiral,
                });
            }
        }
        Self {
            n,
            dim,
            dist,
            chiral: cc,
            fourth_weight: if dim == 4 { weight_fourth } else { 0.0 },
        }
    }

    fn dist2(&self, p: &[f64], a: usize, b: usize) -> f64 {
        let mut d2 = 0.0;
        for k in 0..self.dim {
            let d = p[a * self.dim + k] - p[b * self.dim + k];
            d2 += d * d;
        }
        d2
    }

    /// Energy + gradient (gradient written into `grad`, which must be
    /// length `n*dim` and is overwritten).
    pub(super) fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let mut energy = 0.0;
        let dim = self.dim;

        // Distance violations.
        for c in &self.dist {
            let d2 = self.dist2(p, c.i, c.j);
            let mut val = 0.0;
            if d2 > c.ub2 {
                val = d2 / c.ub2 - 1.0;
            } else if d2 < c.lb2 {
                val = 2.0 * c.lb2 / (c.lb2 + d2) - 1.0;
            }
            if val > 0.0 {
                energy += c.weight * val * val;
            }
            // Gradient (RDKit DistViolationContribs::getGrad).
            let mut pre = 0.0;
            let mut d = 0.0;
            if d2 > c.ub2 {
                d = d2.sqrt();
                pre = 4.0 * (d2 / c.ub2 - 1.0) * (d / c.ub2);
            } else if d2 < c.lb2 {
                d = d2.sqrt();
                let l2d2 = d2 + c.lb2;
                pre = 8.0 * c.lb2 * d * (1.0 - 2.0 * c.lb2 / l2d2) / (l2d2 * l2d2);
            }
            if pre != 0.0 {
                for k in 0..dim {
                    let p1 = c.i * dim + k;
                    let p2 = c.j * dim + k;
                    let dgrad = if d > 0.0 {
                        c.weight * pre * (p[p1] - p[p2]) / d
                    } else {
                        c.weight * pre * (p[p1] - p[p2])
                    };
                    grad[p1] += dgrad;
                    grad[p2] -= dgrad;
                }
            }
        }

        // Chiral-volume violations (computed using only the first 3 dims).
        for c in &self.chiral {
            let (e, _) = self.chiral_energy_grad(p, c, grad);
            energy += e;
        }

        // Fourth-dimension penalty.
        if self.fourth_weight > 1e-8 && dim == 4 {
            for i in 0..self.n {
                let pid = i * dim + 3;
                energy += self.fourth_weight * p[pid] * p[pid];
                grad[pid] += self.fourth_weight * p[pid];
            }
        }
        energy
    }

    fn chiral_energy_grad(&self, p: &[f64], c: &ChiralContrib, grad: &mut [f64]) -> (f64, ()) {
        let dim = self.dim;
        let [i1, i2, i3, i4] = c.idx;
        // v1 = p1 - p4, v2 = p2 - p4, v3 = p3 - p4 (first 3 dims).
        let v1 = [
            p[i1 * dim] - p[i4 * dim],
            p[i1 * dim + 1] - p[i4 * dim + 1],
            p[i1 * dim + 2] - p[i4 * dim + 2],
        ];
        let v2 = [
            p[i2 * dim] - p[i4 * dim],
            p[i2 * dim + 1] - p[i4 * dim + 1],
            p[i2 * dim + 2] - p[i4 * dim + 2],
        ];
        let v3 = [
            p[i3 * dim] - p[i4 * dim],
            p[i3 * dim + 1] - p[i4 * dim + 1],
            p[i3 * dim + 2] - p[i4 * dim + 2],
        ];
        let vol = dot(v1, cross(v2, v3));

        let (energy, pre) = if vol < c.vol_lower {
            (
                c.weight * (vol - c.vol_lower) * (vol - c.vol_lower),
                c.weight * (vol - c.vol_lower),
            )
        } else if vol > c.vol_upper {
            (
                c.weight * (vol - c.vol_upper) * (vol - c.vol_upper),
                c.weight * (vol - c.vol_upper),
            )
        } else {
            return (0.0, ());
        };

        // Gradient (RDKit ChiralViolationContribs::getGrad, 12 components).
        grad[dim * i1] += pre * (v2[1] * v3[2] - v3[1] * v2[2]);
        grad[dim * i1 + 1] += pre * (v3[0] * v2[2] - v2[0] * v3[2]);
        grad[dim * i1 + 2] += pre * (v2[0] * v3[1] - v3[0] * v2[1]);

        grad[dim * i2] += pre * (v3[1] * v1[2] - v3[2] * v1[1]);
        grad[dim * i2 + 1] += pre * (v3[2] * v1[0] - v3[0] * v1[2]);
        grad[dim * i2 + 2] += pre * (v3[0] * v1[1] - v3[1] * v1[0]);

        grad[dim * i3] += pre * (v2[2] * v1[1] - v2[1] * v1[2]);
        grad[dim * i3 + 1] += pre * (v2[0] * v1[2] - v2[2] * v1[0]);
        grad[dim * i3 + 2] += pre * (v2[1] * v1[0] - v2[0] * v1[1]);

        grad[dim * i4] += pre
            * (p[i1 * dim + 2] * (p[i2 * dim + 1] - p[i3 * dim + 1])
                + p[i2 * dim + 2] * (p[i3 * dim + 1] - p[i1 * dim + 1])
                + p[i3 * dim + 2] * (p[i1 * dim + 1] - p[i2 * dim + 1]));
        grad[dim * i4 + 1] += pre
            * (p[i1 * dim] * (p[i2 * dim + 2] - p[i3 * dim + 2])
                + p[i2 * dim] * (p[i3 * dim + 2] - p[i1 * dim + 2])
                + p[i3 * dim] * (p[i1 * dim + 2] - p[i2 * dim + 2]));
        grad[dim * i4 + 2] += pre
            * (p[i1 * dim + 1] * (p[i2 * dim] - p[i3 * dim])
                + p[i2 * dim + 1] * (p[i3 * dim] - p[i1 * dim])
                + p[i3 * dim + 1] * (p[i1 * dim] - p[i2 * dim]));
        (energy, ())
    }
}

/// Signed chiral volume of four points using the first 3 dimensions (RDKit
/// `DistGeom::calcChiralVolume`). `dim` is the coordinate stride.
pub(super) fn calc_chiral_volume(p: &[f64], idx: [usize; 4], dim: usize) -> f64 {
    let [i1, i2, i3, i4] = idx;
    let v1 = [
        p[i1 * dim] - p[i4 * dim],
        p[i1 * dim + 1] - p[i4 * dim + 1],
        p[i1 * dim + 2] - p[i4 * dim + 2],
    ];
    let v2 = [
        p[i2 * dim] - p[i4 * dim],
        p[i2 * dim + 1] - p[i4 * dim + 1],
        p[i2 * dim + 2] - p[i4 * dim + 2],
    ];
    let v3 = [
        p[i3 * dim] - p[i4 * dim],
        p[i3 * dim + 1] - p[i4 * dim + 1],
        p[i3 * dim + 2] - p[i4 * dim + 2],
    ];
    dot(v1, cross(v2, v3))
}

/// Second-stage (3D) objective: distance constraints from the bounds matrix,
/// the experimental (CrystalFF M6) and flat-ring basic-knowledge torsions and
/// the sp2 planarity terms —
/// RDKit's `construct3DForceField`, which together carry the torsion bias
/// plus the bonded skeleton.
///
/// Only the distance constraints are distance-geometry terms. The torsions and
/// the planarity terms are force-field kernels, priced by `ff::potential`:
///
/// * the M6 torsion `Σₘ Vₘ (1 + sₘ cos mφ)` is LAMMPS `dihedral_style fourier`
///   (`dihedral periodic` here) with `kₘ = Vₘ`, `nₘ = m` and phase 0° for
///   `sₘ = +1`, 180° for `sₘ = −1`;
/// * the planarity term `10 · h²`, `h` the height of the sp2 centre over the
///   plane of its three neighbours (RDKit's out-of-plane term at
///   `oobForceScalingFactor = 10`), is
///   [`ImproperDistance`] with `K₂ = 10`, `K₄ = 0`.
pub(super) struct ExpTorsionField {
    dist: Vec<DistContrib>,
    /// The M6 torsions, as one `dihedral periodic` kernel (empty when there
    /// is none).
    torsions: Potentials,
    /// The sp2 planarity terms.
    planarity: ImproperDistance,
}

/// Force constant of the planarity term (RDKit `oobForceScalingFactor`).
const IMPROPER_FORCE: f64 = 10.0;

impl ExpTorsionField {
    /// Build over the bounds (distance constraints), experimental torsions, and
    /// improper (sp2 planarity) constraints.
    pub(super) fn build<'a>(
        bounds: &BoundsMatrix,
        torsions: impl IntoIterator<Item = &'a TorsionConstraint>,
        impropers: &[ImproperConstraint],
    ) -> Self {
        let n = bounds.len();
        let mut dist = Vec::new();
        for i in 1..n {
            for j in 0..i {
                let u = bounds.upper(i, j);
                let l = bounds.lower(i, j);
                dist.push(DistContrib {
                    i,
                    j,
                    ub2: u * u,
                    lb2: l * l,
                    weight: 1.0,
                });
            }
        }
        let mut fourier = ExplicitTerms::new("dihedral", "periodic");
        let mut any_torsion = false;
        for t in torsions {
            any_torsion = true;
            let mut row = Params::new();
            for m in 0..6 {
                row.set(&format!("k{}", m + 1), t.force_constants[m]);
                row.set(&format!("periodicity{}", m + 1), (m + 1) as f64);
                let phase = if t.signs[m] < 0 { 180.0 } else { 0.0 };
                row.set(&format!("phase{}", m + 1), phase);
            }
            fourier = fourier.term(&t.atoms, row);
        }
        let torsions = if any_torsion {
            fourier
                .compile()
                .expect("the built-in `dihedral periodic` style prices any six-term row")
        } else {
            Potentials::new()
        };
        // RDKit packs improper atoms as [n0, center, n2, n3]: index 1 is the
        // centre, which `ImproperDistance` takes first.
        let planarity = impropers.iter().fold(ImproperDistance::new(), |k, im| {
            k.term(
                [im.atoms[1], im.atoms[0], im.atoms[2], im.atoms[3]],
                IMPROPER_FORCE,
                0.0,
            )
        });
        Self {
            dist,
            torsions,
            planarity,
        }
    }

    fn dist2_3d(&self, p: &[f64], a: usize, b: usize) -> f64 {
        let mut d2 = 0.0;
        for k in 0..3 {
            let d = p[a * 3 + k] - p[b * 3 + k];
            d2 += d * d;
        }
        d2
    }

    /// Energy + gradient over a flat `n*3` coordinate buffer.
    pub(super) fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let mut energy = 0.0;
        // Distance constraints (3D).
        for c in &self.dist {
            let d2 = self.dist2_3d(p, c.i, c.j);
            let mut val = 0.0;
            if d2 > c.ub2 {
                val = d2 / c.ub2 - 1.0;
            } else if d2 < c.lb2 {
                val = 2.0 * c.lb2 / (c.lb2 + d2) - 1.0;
            }
            if val > 0.0 {
                energy += c.weight * val * val;
            }
            let mut pre = 0.0;
            let mut d = 0.0;
            if d2 > c.ub2 {
                d = d2.sqrt();
                pre = 4.0 * (d2 / c.ub2 - 1.0) * (d / c.ub2);
            } else if d2 < c.lb2 {
                d = d2.sqrt();
                let l2d2 = d2 + c.lb2;
                pre = 8.0 * c.lb2 * d * (1.0 - 2.0 * c.lb2 / l2d2) / (l2d2 * l2d2);
            }
            if pre != 0.0 {
                for k in 0..3 {
                    let p1 = c.i * 3 + k;
                    let p2 = c.j * 3 + k;
                    let dgrad = if d > 0.0 {
                        c.weight * pre * (p[p1] - p[p2]) / d
                    } else {
                        c.weight * pre * (p[p1] - p[p2])
                    };
                    grad[p1] += dgrad;
                    grad[p2] -= dgrad;
                }
            }
        }
        // Experimental torsions and planarity: forces are −gradient.
        for kernel in [&self.torsions as &dyn Potential, &self.planarity] {
            let (e, forces) = kernel.calc_energy_forces(p);
            energy += e;
            for (g, f) in grad.iter_mut().zip(forces) {
                *g -= f;
            }
        }
        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A right-handed tetrahedron: apex above the origin, base on the axes.
    fn tetrahedron(dim: usize) -> Vec<f64> {
        let pts = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ];
        let mut p = vec![0.0; 4 * dim];
        for (a, xyz) in pts.iter().enumerate() {
            p[a * dim..a * dim + 3].copy_from_slice(xyz);
        }
        p
    }

    #[test]
    fn the_chiral_volume_is_the_triple_product_about_the_fourth_point() {
        // v1 = e_x, v2 = e_y, v3 = e_z relative to the origin: e_x · (e_y × e_z) = 1.
        assert_eq!(calc_chiral_volume(&tetrahedron(3), [0, 1, 2, 3], 3), 1.0);
        // Swapping two substituents flips the sign.
        assert_eq!(calc_chiral_volume(&tetrahedron(3), [1, 0, 2, 3], 3), -1.0);
    }

    #[test]
    fn the_stride_skips_the_fourth_dimension() {
        let mut p = tetrahedron(4);
        for a in 0..4 {
            p[a * 4 + 3] = 9.0 * a as f64; // anything in dimension 4 is ignored
        }
        assert_eq!(calc_chiral_volume(&p, [0, 1, 2, 3], 4), 1.0);
    }

    #[test]
    fn coplanar_points_have_no_volume() {
        let mut p = tetrahedron(3);
        p[2] = 0.0; // put the apex into the z = 0 plane
        p[8] = 0.0;
        assert_eq!(calc_chiral_volume(&p, [0, 1, 2, 3], 3), 0.0);
    }
}
