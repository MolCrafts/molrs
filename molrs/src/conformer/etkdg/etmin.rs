//! First-stage ETKDG error function + minimizer.
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
//! We reproduce both as steepest-descent / gradient minimizations on a flat
//! `n*dim` coordinate buffer.

use crate::conformer::distgeom::{
    BoundsMatrix, ChiralConstraint, ImproperConstraint, TorsionConstraint,
};

/// Per-atom energy threshold above which the first minimization is rejected
/// (RDKit `MAX_MINIMIZED_E_PER_ATOM`).
pub const MAX_MINIMIZED_E_PER_ATOM: f64 = 0.05;

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
pub struct FirstStageField {
    n: usize,
    dim: usize,
    dist: Vec<DistContrib>,
    chiral: Vec<ChiralContrib>,
    fourth_weight: f64,
}

impl FirstStageField {
    /// Build the first-stage field over all atom pairs (RDKit
    /// `constructForceField` with `weightChiral`, `weightFourthDim`).
    pub fn build(
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
    pub fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
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
        let v2xv3 = [
            v2[1] * v3[2] - v2[2] * v3[1],
            v2[2] * v3[0] - v2[0] * v3[2],
            v2[0] * v3[1] - v2[1] * v3[0],
        ];
        let vol = v1[0] * v2xv3[0] + v1[1] * v2xv3[1] + v1[2] * v2xv3[2];

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
pub fn calc_chiral_volume(p: &[f64], idx: [usize; 4], dim: usize) -> f64 {
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
    let v2xv3 = [
        v2[1] * v3[2] - v2[2] * v3[1],
        v2[2] * v3[0] - v2[0] * v3[2],
        v2[0] * v3[1] - v2[1] * v3[0],
    ];
    v1[0] * v2xv3[0] + v1[1] * v2xv3[1] + v1[2] * v2xv3[2]
}

/// Second-stage 3D experimental-torsion field (CrystalFF M6 + distance
/// constraints derived from the smoothed bounds). This is a lighter port of
/// RDKit's `construct3DForceField`: we keep the experimental-torsion M6 terms
/// and the long-range distance constraints from the bounds matrix, which
/// together carry the torsion bias plus the bonded skeleton.
pub struct ExpTorsionField {
    dist: Vec<DistContrib>,
    torsions: Vec<TorsionM6>,
    impropers: Vec<Improper>,
}

#[derive(Clone)]
struct TorsionM6 {
    atoms: [usize; 4],
    signs: [i8; 6],
    v: [f64; 6],
}

/// A planarity (out-of-plane inversion) term: the sp2 centre `center` and its
/// three neighbours `[n0, n1, n2]`. Penalizes displacement of the centre out
/// of the plane of its neighbours (the physical effect of RDKit's UFF
/// inversion term, `addImproperTorsionTerms`, force scaling 10.0).
#[derive(Clone, Copy)]
struct Improper {
    center: usize,
    n: [usize; 3],
}

/// Force scaling for the planarity inversion term (RDKit `oobForceScalingFactor`).
const IMPROPER_FORCE: f64 = 10.0;

impl ExpTorsionField {
    /// Build over the bounds (distance constraints), experimental torsions, and
    /// improper (sp2 planarity) constraints.
    pub fn build(
        bounds: &BoundsMatrix,
        torsions: &[TorsionConstraint],
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
        let torsions = torsions
            .iter()
            .map(|t| TorsionM6 {
                atoms: t.atoms,
                signs: t.signs,
                v: t.force_constants,
            })
            .collect();
        // RDKit packs improper atoms as [n0, center, n2, n3]: index 1 is centre.
        let impropers = impropers
            .iter()
            .map(|im| Improper {
                center: im.atoms[1],
                n: [im.atoms[0], im.atoms[2], im.atoms[3]],
            })
            .collect();
        let _ = n;
        Self {
            dist,
            torsions,
            impropers,
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
    pub fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
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
        // Experimental torsion M6 terms.
        for t in &self.torsions {
            energy += torsion_m6_energy_grad(p, t, grad);
        }
        // Improper (sp2 planarity) inversion terms.
        for im in &self.impropers {
            energy += improper_energy_grad(p, im, grad);
        }
        energy
    }
}

/// Out-of-plane (inversion) energy: `IMPROPER_FORCE · h²` where `h` is the
/// signed perpendicular distance of the centre from the plane of its three
/// neighbours. Gradient by central finite difference over the four atoms.
fn improper_energy_grad(p: &[f64], im: &Improper, grad: &mut [f64]) -> f64 {
    let e = improper_energy(p, im);
    let h = 1e-5;
    let atoms = [im.center, im.n[0], im.n[1], im.n[2]];
    for &a in &atoms {
        for k in 0..3 {
            let idx = a * 3 + k;
            let orig = p[idx];
            let mut pp = p.to_vec();
            pp[idx] = orig + h;
            let ep = improper_energy(&pp, im);
            pp[idx] = orig - h;
            let em = improper_energy(&pp, im);
            grad[idx] += (ep - em) / (2.0 * h);
        }
    }
    e
}

/// Squared out-of-plane height of the centre above the neighbour plane.
fn improper_energy(p: &[f64], im: &Improper) -> f64 {
    let c = [p[im.center * 3], p[im.center * 3 + 1], p[im.center * 3 + 2]];
    let a = [p[im.n[0] * 3], p[im.n[0] * 3 + 1], p[im.n[0] * 3 + 2]];
    let b = [p[im.n[1] * 3], p[im.n[1] * 3 + 1], p[im.n[1] * 3 + 2]];
    let d = [p[im.n[2] * 3], p[im.n[2] * 3 + 1], p[im.n[2] * 3 + 2]];
    let ab = sub(b, a);
    let ad = sub(d, a);
    let mut normal = cross(ab, ad);
    let nlen = norm(normal);
    if nlen < 1e-9 {
        return 0.0;
    }
    for x in normal.iter_mut() {
        *x /= nlen;
    }
    let ac = sub(c, a);
    let height = dot(ac, normal);
    IMPROPER_FORCE * height * height
}

/// M6 torsion energy `V = Σ_m Vm·(1 + sm·cos(m·x))` (RDKit `calcTorsionEnergyM6`)
/// plus a numerical-gradient contribution into `grad` (3D).
fn torsion_m6_energy_grad(p: &[f64], t: &TorsionM6, grad: &mut [f64]) -> f64 {
    let cos_phi = torsion_cos_phi(p, t.atoms);
    let e = torsion_energy_m6(&t.v, &t.signs, cos_phi);
    // Analytical dE/dcos is awkward through cosPhi; use a central finite
    // difference on the four atoms' Cartesians for the gradient. The torsion
    // force constants are small (≤ 8 kcal) so this is a stable, low-cost path.
    let h = 1e-5;
    for &a in &t.atoms {
        for k in 0..3 {
            let idx = a * 3 + k;
            let orig = p[idx];
            // Mutate a scratch copy of the four-atom-relevant coordinate.
            let mut pp = p.to_vec();
            pp[idx] = orig + h;
            let ep = torsion_energy_m6(&t.v, &t.signs, torsion_cos_phi(&pp, t.atoms));
            pp[idx] = orig - h;
            let em = torsion_energy_m6(&t.v, &t.signs, torsion_cos_phi(&pp, t.atoms));
            grad[idx] += (ep - em) / (2.0 * h);
        }
    }
    e
}

/// M6 torsion energy form (RDKit `calcTorsionEnergyM6`).
fn torsion_energy_m6(v: &[f64; 6], signs: &[i8; 6], cos_phi: f64) -> f64 {
    let c = cos_phi;
    let c2 = c * c;
    let c3 = c * c2;
    let c4 = c * c3;
    let c5 = c * c4;
    let c6 = c * c5;
    let cos2 = 2.0 * c2 - 1.0;
    let cos3 = 4.0 * c3 - 3.0 * c;
    let cos4 = 8.0 * c4 - 8.0 * c2 + 1.0;
    let cos5 = 16.0 * c5 - 20.0 * c3 + 5.0 * c;
    let cos6 = 32.0 * c6 - 48.0 * c4 + 18.0 * c2 - 1.0;
    v[0] * (1.0 + signs[0] as f64 * c)
        + v[1] * (1.0 + signs[1] as f64 * cos2)
        + v[2] * (1.0 + signs[2] as f64 * cos3)
        + v[3] * (1.0 + signs[3] as f64 * cos4)
        + v[4] * (1.0 + signs[4] as f64 * cos5)
        + v[5] * (1.0 + signs[5] as f64 * cos6)
}

/// cos(dihedral) for atoms `i-j-k-l` in a flat `n*3` buffer (RDKit
/// `Utils::calcTorsionCosPhi`).
fn torsion_cos_phi(p: &[f64], atoms: [usize; 4]) -> f64 {
    let [i, j, k, l] = atoms;
    let pi = [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]];
    let pj = [p[j * 3], p[j * 3 + 1], p[j * 3 + 2]];
    let pk = [p[k * 3], p[k * 3 + 1], p[k * 3 + 2]];
    let pl = [p[l * 3], p[l * 3 + 1], p[l * 3 + 2]];
    let r1 = sub(pi, pj);
    let r2 = sub(pk, pj);
    let r3 = sub(pj, pk);
    let r4 = sub(pl, pk);
    let t1 = cross(r1, r2);
    let t2 = cross(r3, r4);
    let d1 = norm(t1);
    let d2 = norm(t2);
    if d1 < 1e-12 || d2 < 1e-12 {
        return 1.0;
    }
    (dot(t1, t2) / (d1 * d2)).clamp(-1.0, 1.0)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Generic gradient minimizer (steepest descent with adaptive step + a simple
/// backtracking line search). `closure` computes energy and fills the gradient.
/// Returns `(final_energy, converged, steps)`.
pub fn minimize<F>(
    coords: &mut [f64],
    max_iters: usize,
    force_tol: f64,
    mut closure: F,
) -> (f64, bool, usize)
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = coords.len();
    let mut grad = vec![0.0; n];
    let mut energy = closure(coords, &mut grad);
    let mut step = 0.01;
    let mut converged = false;
    let mut iters = 0;
    for it in 0..max_iters {
        iters = it + 1;
        let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < force_tol {
            converged = true;
            break;
        }
        // Backtracking line search along -grad.
        let mut trial = coords.to_vec();
        let mut new_grad = vec![0.0; n];
        let mut accepted = false;
        let mut local_step = step;
        for _ in 0..20 {
            for k in 0..n {
                trial[k] = coords[k] - local_step * grad[k];
            }
            let e_trial = closure(&trial, &mut new_grad);
            if e_trial < energy {
                coords.copy_from_slice(&trial);
                energy = e_trial;
                grad.copy_from_slice(&new_grad);
                step = (local_step * 1.2).min(0.1);
                accepted = true;
                break;
            }
            local_step *= 0.5;
        }
        if !accepted {
            // Could not make progress; treat as converged at a local min.
            converged = true;
            break;
        }
    }
    (energy, converged, iters)
}
