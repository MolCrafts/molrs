//! Conjugate Gradient inner solver for the Truncated Newton direction.
//! Exact algorithmic port of `cg` from `gencan.f`.

use crate::constraints::EvalMode;
use crate::numerics::{near_zero_norm_floor, positive_norm_floor, residual_small_floor};
use crate::objective::Objective;
use molrs::types::F;

/// Reusable CG work vectors, matching packmol `cg` workspace roles.
pub struct CgScratch {
    gfree: Vec<F>,
    s: Vec<F>,
    sprev: Vec<F>,
    r: Vec<F>,
    d: Vec<F>,
    w: Vec<F>,
    y: Vec<F>,
    gy: Vec<F>,
}

impl CgScratch {
    pub fn new(n: usize) -> Self {
        Self {
            gfree: vec![0.0; n],
            s: vec![0.0; n],
            sprev: vec![0.0; n],
            r: vec![0.0; n],
            d: vec![0.0; n],
            w: vec![0.0; n],
            y: vec![0.0; n],
            gy: vec![0.0; n],
        }
    }

    fn ensure_len(&mut self, n: usize) {
        if self.gfree.len() < n {
            self.gfree.resize(n, 0.0);
        }
        if self.s.len() < n {
            self.s.resize(n, 0.0);
        }
        if self.sprev.len() < n {
            self.sprev.resize(n, 0.0);
        }
        if self.r.len() < n {
            self.r.resize(n, 0.0);
        }
        if self.d.len() < n {
            self.d.resize(n, 0.0);
        }
        if self.w.len() < n {
            self.w.resize(n, 0.0);
        }
        if self.y.len() < n {
            self.y.resize(n, 0.0);
        }
        if self.gy.len() < n {
            self.gy.resize(n, 0.0);
        }
    }
}

pub struct CgResult {
    pub iter: usize,
    pub q: F,
    pub inform: i32,
    /// Boundary info returned when `inform == 2` (box boundary reached).
    pub rbdind: Option<usize>,
    /// 1 = lower, 2 = upper, 0 = not set.
    pub rbdtype: i32,
}

/// Solve the trust-region quadratic subproblem by CG.
/// Operates on the free-variable set `ind[0..nind)`.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
pub fn cg_solve(
    nind: usize,
    ind: &[usize],
    n: usize,
    x: &[F],
    g: &[F],
    delta: F,
    l: &[F],
    u: &[F],
    eps: F,
    epsnqmp: F,
    maxitnqmp: usize,
    maxit: usize,
    nearlyq: bool,
    trtype: i32,
    theta: F,
    sterel: F,
    steabs: F,
    epsrel: F,
    epsabs: F,
    _infrel: F,
    infabs: F,
    d_out: &mut [F],
    scratch: &mut CgScratch,
    obj: &mut dyn Objective,
) -> CgResult {
    let nind = nind.min(ind.len());
    scratch.ensure_len(n);

    for &ii in &ind[..nind] {
        d_out[ii] = 0.0;
    }

    if nind == 0 {
        return CgResult {
            iter: 0,
            q: 0.0,
            inform: 0,
            rbdind: None,
            rbdtype: 0,
        };
    }

    let mut gnorm2 = 0.0;
    {
        let gfree = &mut scratch.gfree[..nind];
        for (j, &ii) in ind[..nind].iter().enumerate() {
            gfree[j] = g[ii];
            gnorm2 += gfree[j] * gfree[j];
        }
    }
    if gnorm2 <= 0.0 {
        return CgResult {
            iter: 0,
            q: 0.0,
            inform: 0,
            rbdind: None,
            rbdtype: 0,
        };
    }

    let mut iter = 0usize;
    let mut itnqmp = 0usize;
    let mut qprev = infabs;
    let mut bestprog = 0.0 as F;

    let s = &mut scratch.s[..nind];
    let sprev = &mut scratch.sprev[..nind];
    let r = &mut scratch.r[..nind];
    let d = &mut scratch.d[..nind];
    let w = &mut scratch.w[..nind];
    let y = &mut scratch.y[..n];
    let gy = &mut scratch.gy[..n];
    let gfree = &scratch.gfree[..nind];

    s.fill(0.0);
    sprev.fill(0.0);
    r.copy_from_slice(gfree);

    let mut q = 0.0 as F;
    let mut gts = 0.0 as F;
    let mut snorm2 = 0.0 as F;
    let mut snorm2prev = 0.0 as F;
    let mut rnorm2 = gnorm2;
    let mut rnorm2prev = rnorm2;
    let mut dnorm2 = 0.0 as F;
    let mut dtr = 0.0 as F;
    let mut dtw = 0.0 as F;
    let mut alpha = 0.0 as F;
    let mut inform = 0i32;

    let mut rbdind: Option<usize> = None;
    let mut rbdtype = 0i32;

    loop {
        // Residual convergence
        if rnorm2 <= near_zero_norm_floor() * near_zero_norm_floor()
            || (((rnorm2 <= eps * eps * gnorm2) || (rnorm2 <= residual_small_floor() && iter != 0))
                && iter >= 4)
        {
            inform = 0;
            break;
        }

        // Iteration limit (minimum 4 iterations as in Fortran)
        if iter >= usize::max(4, maxit) {
            inform = 8;
            break;
        }

        // Compute direction
        if iter == 0 {
            for j in 0..nind {
                d[j] = -r[j];
            }
            dnorm2 = rnorm2;
            dtr = -rnorm2;
        } else {
            let beta = rnorm2 / rnorm2prev;
            for j in 0..nind {
                d[j] = -r[j] + beta * d[j];
            }
            dnorm2 = rnorm2 - 2.0 * beta * (dtr + alpha * dtw) + beta * beta * dnorm2;
            dtr = -rnorm2 + beta * (dtr + alpha * dtw);
        }

        // Force descent direction if needed
        if dtr > 0.0 {
            for dj in d[..nind].iter_mut() {
                *dj = -*dj;
            }
            dtr = -dtr;
        }

        // Compute w = H d by finite differences
        hessian_times_vec_diff(nind, ind, n, x, d, g, sterel, steabs, w, y, gy, obj);
        dtw = (0..nind).map(|j| d[j] * w[j]).sum();

        // Maximum trust-region step
        let dts: F = (0..nind).map(|j| d[j] * s[j]).sum();

        let (amax1, amax1n) = if trtype == 0 {
            // Euclidian trust region
            let aa = dnorm2;
            let bb = 2.0 * dts;
            let cc = snorm2 - delta * delta;
            let dd = (bb * bb - 4.0 * aa * cc).max(0.0).sqrt();
            ((-bb + dd) / (2.0 * aa), (-bb - dd) / (2.0 * aa))
        } else {
            // Sup-norm trust region
            let mut amax1 = infabs;
            let mut amax1n = -infabs;
            for j in 0..nind {
                if d[j] > 0.0 {
                    amax1 = amax1.min((delta - s[j]) / d[j]);
                    amax1n = amax1n.max((-delta - s[j]) / d[j]);
                } else if d[j] < 0.0 {
                    amax1 = amax1.min((-delta - s[j]) / d[j]);
                    amax1n = amax1n.max((delta - s[j]) / d[j]);
                }
            }
            (amax1, amax1n)
        };

        // Maximum box step and corresponding boundary variable/type
        let mut amax2 = infabs;
        let mut amax2n = -infabs;
        let mut rbdposaind = 0usize;
        let mut rbdnegaind = 0usize;
        let mut rbdposatype = 0i32;
        let mut rbdnegatype = 0i32;

        for j in 0..nind {
            let ii = ind[j];
            if d[j] > 0.0 {
                let amax2x = (u[ii] - x[ii] - s[j]) / d[j];
                if amax2x < amax2 {
                    amax2 = amax2x;
                    rbdposaind = j;
                    rbdposatype = 2;
                }
                let amax2nx = (l[ii] - x[ii] - s[j]) / d[j];
                if amax2nx > amax2n {
                    amax2n = amax2nx;
                    rbdnegaind = j;
                    rbdnegatype = 1;
                }
            } else if d[j] < 0.0 {
                let amax2x = (l[ii] - x[ii] - s[j]) / d[j];
                if amax2x < amax2 {
                    amax2 = amax2x;
                    rbdposaind = j;
                    rbdposatype = 1;
                }
                let amax2nx = (u[ii] - x[ii] - s[j]) / d[j];
                if amax2nx > amax2n {
                    amax2n = amax2nx;
                    rbdnegaind = j;
                    rbdnegatype = 2;
                }
            }
        }

        let amax = amax1.min(amax2);
        let amaxn = amax1n.max(amax2n);

        qprev = q;

        // Step selection
        if dtw > 0.0 {
            alpha = amax.min(rnorm2 / dtw);
            q = q + 0.5 * alpha * alpha * dtw + alpha * dtr;
        } else {
            let qamax = q + 0.5 * amax * amax * dtw + amax * dtr;
            if iter == 0 {
                alpha = amax;
                q = qamax;
            } else {
                let qamaxn = q + 0.5 * amaxn * amaxn * dtw + amaxn * dtr;
                if nearlyq && (qamax < q || qamaxn < q) {
                    if qamax < qamaxn {
                        alpha = amax;
                        q = qamax;
                    } else {
                        alpha = amaxn;
                        q = qamaxn;
                    }
                } else {
                    inform = 7;
                    break;
                }
            }
        }

        // Update s
        sprev[..nind].copy_from_slice(&s[..nind]);
        for j in 0..nind {
            s[j] += alpha * d[j];
        }
        snorm2prev = snorm2;
        snorm2 = snorm2 + alpha * alpha * dnorm2 + 2.0 * alpha * dts;

        // Update residual r = r + alpha * w
        rnorm2prev = rnorm2;
        for j in 0..nind {
            r[j] += alpha * w[j];
        }
        rnorm2 = r[..nind].iter().map(|v| v * v).sum();

        iter += 1;

        // Angle condition
        gts = (0..nind).map(|j| gfree[j] * s[j]).sum();
        if gts > 0.0 || gts * gts < theta * theta * gnorm2 * snorm2 {
            s.copy_from_slice(sprev);
            snorm2 = snorm2prev;
            q = qprev;
            inform = 3;
            break;
        }

        // Box boundary
        if alpha == amax2 || alpha == amax2n {
            if alpha == amax2 {
                rbdind = Some(ind[rbdposaind]);
                rbdtype = rbdposatype;
            } else {
                rbdind = Some(ind[rbdnegaind]);
                rbdtype = rbdnegatype;
            }
            inform = 2;
            break;
        }

        // Trust-region boundary
        if alpha == amax1 || alpha == amax1n {
            inform = 1;
            break;
        }

        // Very similar consecutive iterates
        let mut samep = true;
        for j in 0..nind {
            if (alpha * d[j]).abs() > (epsrel * s[j].abs()).max(epsabs) {
                samep = false;
                break;
            }
        }
        if samep {
            inform = 6;
            break;
        }

        // Not enough progress in quadratic model
        let currprog = qprev - q;
        bestprog = bestprog.max(currprog);
        if currprog <= epsnqmp * bestprog {
            itnqmp += 1;
            if itnqmp >= maxitnqmp {
                inform = 4;
                break;
            }
        } else {
            itnqmp = 0;
        }
    }

    for j in 0..nind {
        d_out[ind[j]] = s[j];
    }

    CgResult {
        iter,
        q,
        inform,
        rbdind,
        rbdtype,
    }
}

/// Approximate Hessian-vector product by incremental quotients.
/// Port of `calchddiff` from `gencan.f`.
#[allow(clippy::too_many_arguments)]
fn hessian_times_vec_diff(
    nind: usize,
    ind: &[usize],
    n: usize,
    x: &[F],
    d: &[F],
    g: &[F],
    sterel: F,
    steabs: F,
    w: &mut [F],
    y: &mut [F],
    gy: &mut [F],
    obj: &mut dyn Objective,
) {
    let xsupn = ind[..nind]
        .iter()
        .map(|&ii| x[ii].abs())
        .fold(0.0 as F, F::max);
    let dsupn = d[..nind].iter().map(|v| v.abs()).fold(0.0 as F, F::max);
    let step = (sterel * xsupn).max(steabs) / dsupn.max(positive_norm_floor());

    y[..n].copy_from_slice(&x[..n]);
    for j in 0..nind {
        let ii = ind[j];
        y[ii] = x[ii] + step * d[j];
    }

    obj.evaluate(y, EvalMode::GradientOnly, Some(gy));

    for j in 0..nind {
        let ii = ind[j];
        w[j] = (gy[ii] - g[ii]) / step;
    }
}
