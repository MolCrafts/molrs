//! GENCAN optimizer — faithful Rust port of `gencan.f` and `pgencan.f90`.
//!
//! Reference: Birgin & Martinez, Comp.Opt.Appl. 23:101-125, 2002.

use molrs::types::F;
pub mod cg;
pub mod spg;

use crate::constraints::EvalMode;
use crate::context::PackContext;
use crate::numerics::{numeric_controls, positive_norm_floor};
use crate::objective::{compute_f, compute_g};

/// Parameters for the GENCAN call (matches `easygencan` defaults from `pgencan.f90`).
pub struct GencanParams {
    pub epsgpsn: F,
    pub maxit: usize,
    pub maxfc: usize,
    pub delmin: F,
    pub iprint: i32,
    pub ncomp: usize,
}

impl Default for GencanParams {
    fn default() -> Self {
        Self {
            epsgpsn: 1.0e-6,
            maxit: 20,
            maxfc: 200, // 10 * maxit
            delmin: 2.0,
            iprint: 0,
            ncomp: 50,
        }
    }
}

/// Result of a GENCAN run.
pub struct GencanResult {
    pub f: F,
    pub gpsupn: F,
    pub iter: usize,
    pub fcnt: usize,
    pub gcnt: usize,
    pub cgcnt: usize,
    /// 0=converged(eucl), 1=converged(sup), 2=noFprogress, 3=noGprogress,
    /// 4=fSmall, 7=maxIter, 8=maxFeval, <0=error
    pub inform: i32,
}

/// Reusable work buffers for repeated `pgencan` calls.
pub struct GencanWorkspace {
    g: Vec<F>,
    ind: Vec<usize>,
    d: Vec<F>,
    s: Vec<F>,
    y: Vec<F>,
    cg_scratch: cg::CgScratch,
    spg_scratch: spg::SpgScratch,
    tnls_scratch: TnLsScratch,
}

impl GencanWorkspace {
    pub fn new() -> Self {
        Self {
            g: Vec::new(),
            ind: Vec::new(),
            d: Vec::new(),
            s: Vec::new(),
            y: Vec::new(),
            cg_scratch: cg::CgScratch::new(0),
            spg_scratch: spg::SpgScratch::new(0),
            tnls_scratch: TnLsScratch::new(0),
        }
    }

    fn ensure_len(&mut self, n: usize) {
        if self.g.len() != n {
            self.g.resize(n, 0.0);
        }
        if self.d.len() != n {
            self.d.resize(n, 0.0);
        }
        if self.s.len() != n {
            self.s.resize(n, 0.0);
        }
        if self.y.len() != n {
            self.y.resize(n, 0.0);
        }
        if self.ind.capacity() < n {
            self.ind.reserve(n - self.ind.capacity());
        }
    }
}

impl Default for GencanWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry point — mirrors `pgencan.f90` → `easygencan` → `gencan`.
///
/// The variable layout is:
///   x[0..3N]   = COM positions (free molecules)
///   x[3N..6N]  = Euler angles (free molecules)
///
/// Bounds: COM variables are unbounded; Euler angles may be bounded by
/// `constrain_rotation` constraints (Packmol pgencan.f90).
pub fn pgencan(
    x: &mut [F],
    sys: &mut PackContext,
    params: &GencanParams,
    precision: F,
    workspace: &mut GencanWorkspace,
) -> GencanResult {
    let n = x.len();

    let (l, u) = build_bounds(n, sys);

    gencan(x, &l, &u, sys, params, precision, workspace)
}

fn build_bounds(n: usize, sys: &PackContext) -> (Vec<F>, Vec<F>) {
    let mut l = vec![-1.0e20 as F; n];
    let mut u = vec![1.0e20 as F; n];
    let mut i = n / 2;

    for itype in 0..sys.ntype {
        if !sys.comptype[itype] {
            continue;
        }
        for _imol in 0..sys.nmols[itype] {
            for axis in 0..3 {
                if sys.constrain_rot[itype][axis] {
                    let center = sys.rot_bound[itype][axis][0];
                    let half_width = sys.rot_bound[itype][axis][1].abs();
                    l[i] = center - half_width;
                    u[i] = center + half_width;
                } else {
                    l[i] = -1.0e20;
                    u[i] = 1.0e20;
                }
                i += 1;
            }
        }
    }
    debug_assert_eq!(i, n);

    (l, u)
}

/// Main GENCAN loop.
/// Port of `gencan.f` subroutine with Packmol-specific additions.
#[allow(unused_assignments)]
pub fn gencan(
    x: &mut [F],
    l: &[F],
    u: &[F],
    sys: &mut PackContext,
    params: &GencanParams,
    precision: F,
    workspace: &mut GencanWorkspace,
) -> GencanResult {
    let n = x.len();
    workspace.ensure_len(n);

    // Constants (from easygencan parameters section)
    const INFREL: F = 1.0e20;
    const INFABS: F = F::MAX;
    const BETA_LS: F = 0.5;
    const GAMMA: F = 1.0e-4;
    const THETA: F = 1.0e-6;
    const SIGMA1: F = 0.1;
    const SIGMA2: F = 0.9;
    const MAXEXTRAP: usize = 100;
    const MININTERP: usize = 4;
    const NINT: F = 2.0;
    const NEXT: F = 2.0;
    const ETA: F = 0.9;
    const LSPGMA: F = 1.0e10;
    const LSPGMI: F = 1.0e-10;
    const FMIN: F = 1.0e-5;
    const EPSGPEN: F = 0.0;
    let numeric = numeric_controls();

    let cgepsi = 0.1 as F;
    let cgepsf = 1.0e-5 as F;
    let cggpnf = F::max(1.0e-4, params.epsgpsn);
    let epsnqmp = 1.0e-4 as F;
    let maxitnqmp = 5usize;
    let epsnfp = 0.0 as F;
    let maxitnfp = params.maxit;
    let maxitngp = 1000usize;

    // Project initial point
    for i in 0..n {
        x[i] = x[i].clamp(l[i], u[i]);
    }

    // Initial function value + gradient.
    let mut fcnt = 0usize;
    let g = &mut workspace.g;
    g.fill(0.0);
    let mut f = sys.evaluate(x, EvalMode::FAndGradient, Some(g)).f_total;

    // Packmol behavior: check packmolprecision before counting this first eval.
    if packmolprecision(sys, precision) {
        return GencanResult {
            f,
            gpsupn: 0.0,
            iter: 0,
            fcnt,
            gcnt: 0,
            cgcnt: 0,
            inform: 0,
        };
    }
    fcnt += 1;
    let mut gcnt = 1usize;
    let mut cgcnt = 0usize;

    // Compute xnorm
    let mut xnorm = x.iter().map(|xi| xi * xi).sum::<F>().sqrt();

    let ind = &mut workspace.ind;
    let cg_scratch = &mut workspace.cg_scratch;
    let spg_scratch = &mut workspace.spg_scratch;
    let tnls_scratch = &mut workspace.tnls_scratch;

    // Compute projected gradient
    let (mut gpsupn, mut gpeucn2, mut gieucn2, mut nind) =
        projected_gradient_info(n, x, g.as_slice(), l, u, ind);

    // CG epsilon scaling
    let (acgeps, bcgeps) = gp_ieee_signal(gpsupn, cgepsf, cgepsi, cggpnf);

    // Track initial projected gradient for kappa computation in cgmaxit
    // (Packmol gp_ieee_signal2: cgscre=2, uses sup-norm)
    let gpsupn0 = gpsupn;

    let mut iter = 0usize;
    let mut inform = 7i32; // default: max iterations

    // Trust radius — computed fresh at the start of each TN iteration (see below).
    // Fortran: iter==1 → max(delmin, 0.1*xnorm); iter>1 → max(delmin, 10*sqrt(sts)).
    let mut delta = 0.0 as F; // placeholder; set before each cg_solve call

    // BB spectral step
    let mut sts = 0.0 as F;
    let mut sty = 0.0 as F;
    let ometa2 = (1.0 - ETA).powi(2);

    // No-progress tracking
    let mut fprev = INFABS;
    let mut bestprog = 0.0 as F;
    let mut itnfp = 0usize;
    let mut lastgpns = vec![INFABS; maxitngp];

    // Working vectors
    let d = &mut workspace.d;
    let s = &mut workspace.s;
    let y = &mut workspace.y;

    // Main loop
    loop {
        // Packmol behavior: recompute precision test with computef at each iteration.
        if packmolprecision(sys, precision) {
            break;
        }

        if gpeucn2 <= EPSGPEN * EPSGPEN {
            inform = 0;
            break;
        }
        // Check convergence: sup-norm of projected gradient
        if gpsupn <= params.epsgpsn {
            inform = 1;
            break;
        }

        // No function progress
        let currprog = fprev - f;
        bestprog = bestprog.max(currprog);
        if currprog <= epsnfp * bestprog {
            itnfp += 1;
            if itnfp >= maxitnfp {
                inform = 2;
                break;
            }
        } else {
            itnfp = 0;
        }

        // No gradient progress
        let gpnmax = lastgpns.iter().copied().fold(0.0 as F, F::max);
        lastgpns[iter % maxitngp] = gpeucn2;
        if gpeucn2 >= gpnmax {
            inform = 3;
            break;
        }

        if f <= FMIN {
            inform = 4;
            break;
        }
        if iter >= params.maxit {
            inform = 7;
            break;
        }
        if fcnt >= params.maxfc {
            inform = 8;
            break;
        }

        // New iteration
        iter += 1;
        fprev = f;

        // Save x → s, g → y
        s.copy_from_slice(x);
        y.copy_from_slice(g.as_slice());

        if gieucn2 <= ometa2 * gpeucn2 {
            // SPG iteration: abandon current face
            let lamspg = if iter == 1 || sty <= 0.0 {
                F::max(1.0, xnorm) / gpeucn2.sqrt().max(positive_norm_floor())
            } else {
                sts / sty
            };
            let lamspg = lamspg.clamp(LSPGMI, LSPGMA);

            let spg_res = spg::spgls(
                n,
                x,
                g.as_slice(),
                l,
                u,
                lamspg,
                f,
                NINT,
                MININTERP,
                FMIN,
                params.maxfc,
                fcnt,
                GAMMA,
                SIGMA1,
                SIGMA2,
                numeric.sterel,
                numeric.steabs,
                numeric.epsrel,
                numeric.epsabs,
                spg_scratch,
                sys,
            );
            f = spg_res.f;
            fcnt = spg_res.fcnt;
            x.copy_from_slice(&spg_scratch.xtrial);

            if spg_res.inform < 0 {
                inform = spg_res.inform;
                break;
            }

            sys.evaluate(x, EvalMode::GradientOnly, Some(g.as_mut_slice()));
            gcnt += 1;
        } else {
            // TN iteration: compute Newton direction via CG

            // Compute trust-region radius (Fortran gencan.f lines 2120-2128):
            //   iter==1: delta = max(delmin, 0.1 * max(1, xnorm))
            //   iter>1:  delta = max(delmin, 10 * sqrt(sts))
            delta = if iter == 1 {
                F::max(params.delmin, 0.1 * F::max(1.0, xnorm))
            } else {
                F::max(params.delmin, 10.0 * sts.sqrt())
            };

            let cgeps = compute_cgeps(gpsupn, acgeps, bcgeps, cgepsf, cgepsi);

            // Packmol gp_ieee_signal2 formula (cgscre=2, nearlyq=false):
            //   kappa = clamp(log10(gpsupn/gpsupn0) / log10(epsgpsn/gpsupn0), 0, 1)
            //   cgmaxit = min(20, (1-kappa)*max(1, 10*log10(nind)) + kappa*nind)
            let cgmaxit = {
                let mut kappa = (gpsupn / gpsupn0).log10() / (params.epsgpsn / gpsupn0).log10();
                kappa = F::max(0.0, F::min(1.0, kappa));

                let nind_f = nind as F;
                let base = (1.0 - kappa) * F::max(1.0, 10.0 * nind_f.log10()) + kappa * nind_f;
                usize::min(20, base as usize)
            };

            let cg_res = cg::cg_solve(
                nind,
                ind,
                n,
                x,
                g.as_slice(),
                delta,
                l,
                u,
                cgeps,
                epsnqmp,
                maxitnqmp,
                cgmaxit,
                false, // nearlyq = .false. in packmol easygencan defaults
                1,     // trtype=1 (sup-norm)
                THETA,
                numeric.sterel,
                numeric.steabs,
                numeric.epsrel,
                numeric.epsabs,
                INFREL,
                INFABS,
                d,
                cg_scratch,
                sys,
            );
            cgcnt += cg_res.iter;

            // Compute maximum feasible step along d (packmol gencan.f lines 2204-2225).
            let mut amax = INFABS;
            let mut rbdtype = 0i32;
            let mut rbdind = if nind > 0 { ind[0] } else { 0 };

            if cg_res.inform == 2 {
                amax = 1.0;
                rbdtype = cg_res.rbdtype;
                rbdind = cg_res.rbdind.unwrap_or(rbdind);
            } else {
                for &ii in &ind[..nind] {
                    if d[ii] > 0.0 {
                        let amaxx = (u[ii] - x[ii]) / d[ii];
                        if amaxx < amax {
                            amax = amaxx;
                            rbdind = ii;
                            rbdtype = 2;
                        }
                    } else if d[ii] < 0.0 {
                        let amaxx = (l[ii] - x[ii]) / d[ii];
                        if amaxx < amax {
                            amax = amaxx;
                            rbdind = ii;
                            rbdtype = 1;
                        }
                    }
                }
            }

            // TN line search (full port of tnls behavior).
            let ls_res = tn_linesearch(
                nind,
                ind,
                n,
                x,
                g.as_slice(),
                d,
                l,
                u,
                f,
                amax,
                rbdtype,
                rbdind,
                NINT,
                NEXT,
                MININTERP,
                MAXEXTRAP,
                FMIN,
                params.maxfc,
                fcnt,
                gcnt,
                GAMMA,
                BETA_LS,
                numeric.sterel,
                numeric.steabs,
                SIGMA1,
                SIGMA2,
                numeric.epsrel,
                numeric.epsabs,
                tnls_scratch,
                sys,
            );
            f = ls_res.f;
            fcnt = ls_res.fcnt;
            gcnt = ls_res.gcnt;
            x.copy_from_slice(&tnls_scratch.xret);
            g.copy_from_slice(&tnls_scratch.gret);

            if ls_res.inform < 0 {
                inform = ls_res.inform;
                break;
            }
            inform = ls_res.inform;

            // packmol behavior: if tnls stops with inform=6, discard TN step and force SPG.
            if ls_res.inform == 6 {
                let lamspg = if iter == 1 || sty <= 0.0 {
                    F::max(1.0, xnorm) / gpeucn2.sqrt().max(positive_norm_floor())
                } else {
                    sts / sty
                };
                let lamspg = lamspg.clamp(LSPGMI, LSPGMA);

                let spg_res = spg::spgls(
                    n,
                    x,
                    g.as_slice(),
                    l,
                    u,
                    lamspg,
                    f,
                    NINT,
                    MININTERP,
                    FMIN,
                    params.maxfc,
                    fcnt,
                    GAMMA,
                    SIGMA1,
                    SIGMA2,
                    numeric.sterel,
                    numeric.steabs,
                    numeric.epsrel,
                    numeric.epsabs,
                    spg_scratch,
                    sys,
                );
                f = spg_res.f;
                fcnt = spg_res.fcnt;
                x.copy_from_slice(&spg_scratch.xtrial);

                if spg_res.inform < 0 {
                    inform = spg_res.inform;
                    break;
                }

                let infotmp = spg_res.inform;
                sys.evaluate(x, EvalMode::GradientOnly, Some(g.as_mut_slice()));
                gcnt += 1;
                inform = infotmp;
            }
        }

        // Adjust to bounds near machine precision (packmol gencan.f lines 2363-2371).
        for i in 0..n {
            if x[i] <= l[i] + (numeric.epsrel * l[i].abs()).max(numeric.epsabs) {
                x[i] = l[i];
            } else if x[i] >= u[i] - (numeric.epsrel * u[i].abs()).max(numeric.epsabs) {
                x[i] = u[i];
            }
        }

        // Update x norm.
        xnorm = x.iter().map(|xi| xi * xi).sum::<F>().sqrt();

        // Update BB steplength: sts = (x-s)^T(x-s), sty = (x-s)^T(g-y)
        sts = 0.0;
        sty = 0.0;
        for i in 0..n {
            let ds = x[i] - s[i];
            let dy = g[i] - y[i];
            sts += ds * ds;
            sty += ds * dy;
        }

        // Update projected gradient info (reuses pre-allocated ind buffer)
        let pg = projected_gradient_info(n, x, g.as_slice(), l, u, ind);
        gpsupn = pg.0;
        gpeucn2 = pg.1;
        gieucn2 = pg.2;
        nind = pg.3;
    }

    // No extra compute_f here — packer.rs calls compute_f with unscaled radii
    // immediately after pgencan returns, so this would be redundant.

    GencanResult {
        f,
        gpsupn,
        iter,
        fcnt,
        gcnt,
        cgcnt,
        inform,
    }
}

/// Compute projected gradient info into pre-allocated `ind` buffer.
/// Returns (gpsupn, gpeucn2, gieucn2, nind).
fn projected_gradient_info(
    n: usize,
    x: &[F],
    g: &[F],
    l: &[F],
    u: &[F],
    ind: &mut Vec<usize>,
) -> (F, F, F, usize) {
    let mut gpsupn = 0.0 as F;
    let mut gpeucn2 = 0.0 as F;
    let mut gieucn2 = 0.0 as F;

    ind.clear();
    for i in 0..n {
        let gpi = (x[i] - g[i]).clamp(l[i], u[i]) - x[i];
        gpsupn = gpsupn.max(gpi.abs());
        gpeucn2 += gpi * gpi;
        if x[i] > l[i] && x[i] < u[i] {
            gieucn2 += gpi * gpi;
            ind.push(i);
        }
    }

    (gpsupn, gpeucn2, gieucn2, ind.len())
}

/// Packmol precision check (`packmolprecision` in `pgencan.f90`):
/// recompute objective-side violations and test `fdist/frest`.
fn packmolprecision(sys: &PackContext, precision: F) -> bool {
    sys.fdist < precision && sys.frest < precision
}

/// Compute scaling for CG epsilon.
fn gp_ieee_signal(gpsupn: F, cgepsf: F, cgepsi: F, cggpnf: F) -> (F, F) {
    if gpsupn > 0.0 {
        let acgeps = (cgepsf / cgepsi).log10() / (cggpnf / gpsupn).log10();
        let bcgeps = cgepsi.log10() - acgeps * gpsupn.log10();
        (acgeps, bcgeps)
    } else {
        (0.0, cgepsf)
    }
}

fn compute_cgeps(gpsupn: F, acgeps: F, bcgeps: F, cgepsf: F, cgepsi: F) -> F {
    let cgeps = (10.0 as F).powf(acgeps * gpsupn.log10() + bcgeps);
    cgeps.clamp(cgepsf, cgepsi)
}

/// Truncated-Newton line search result.
struct LsResult {
    pub f: F,
    pub fcnt: usize,
    pub gcnt: usize,
    pub inform: i32,
}

/// Reusable buffers for TN line search (`tnls` in Packmol `gencan.f`).
struct TnLsScratch {
    xret: Vec<F>,
    gret: Vec<F>,
    xplus: Vec<F>,
    xtmp: Vec<F>,
    gplus: Vec<F>,
}

impl TnLsScratch {
    fn new(n: usize) -> Self {
        Self {
            xret: vec![0.0; n],
            gret: vec![0.0; n],
            xplus: vec![0.0; n],
            xtmp: vec![0.0; n],
            gplus: vec![0.0; n],
        }
    }

    fn ensure_len(&mut self, n: usize) {
        if self.xret.len() != n {
            self.xret.resize(n, 0.0);
        }
        if self.gret.len() != n {
            self.gret.resize(n, 0.0);
        }
        if self.xplus.len() != n {
            self.xplus.resize(n, 0.0);
        }
        if self.xtmp.len() != n {
            self.xtmp.resize(n, 0.0);
        }
        if self.gplus.len() != n {
            self.gplus.resize(n, 0.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
fn tn_linesearch(
    nind: usize,
    ind: &[usize],
    n: usize,
    x: &[F],
    g: &[F],
    d: &[F],
    l: &[F],
    u: &[F],
    f0: F,
    amax: F,
    rbdtype: i32,
    rbdind: usize,
    nint: F,
    next: F,
    mininterp: usize,
    maxextrap: usize,
    fmin: F,
    maxfc: usize,
    mut fcnt: usize,
    mut gcnt: usize,
    gamma: F,
    beta: F,
    _sterel: F,
    _steabs: F,
    sigma1: F,
    sigma2: F,
    epsrel: F,
    epsabs: F,
    scratch: &mut TnLsScratch,
    sys: &mut PackContext,
) -> LsResult {
    scratch.ensure_len(n);
    let nind = nind.min(ind.len());
    let xret = &mut scratch.xret;
    let gret = &mut scratch.gret;
    let xplus = &mut scratch.xplus;
    let xtmp = &mut scratch.xtmp;
    let gplus = &mut scratch.gplus;
    xret.copy_from_slice(x);
    gret.copy_from_slice(g);
    let mut fret = f0;

    let mut gplus_valid = false;

    // gtd = <g,d> in the free-variable subspace.
    let mut gtd = 0.0 as F;
    for &ii in &ind[..nind] {
        gtd += g[ii] * d[ii];
    }

    // First trial alpha = min(1, amax)
    let mut alpha = (1.0 as F).min(amax);
    xplus.copy_from_slice(x);
    for &ii in &ind[..nind] {
        xplus[ii] = x[ii] + alpha * d[ii];
    }
    if alpha == amax && rbdtype != 0 {
        if rbdtype == 1 {
            xplus[rbdind] = l[rbdind];
        } else {
            xplus[rbdind] = u[rbdind];
        }
    }

    let mut fplus = compute_f(xplus, sys);
    fcnt += 1;

    let mut do_extrap = false;

    // Decide between extrapolation and interpolation.
    if amax > 1.0 {
        if fplus <= f0 + gamma * alpha * gtd {
            compute_g(xplus, sys, gplus);
            gcnt += 1;
            gplus_valid = true;

            let mut gptd = 0.0 as F;
            for &ii in &ind[..nind] {
                gptd += gplus[ii] * d[ii];
            }

            if gptd < beta * gtd {
                do_extrap = true;
            } else {
                xret.copy_from_slice(xplus);
                gret.copy_from_slice(gplus);
                fret = fplus;
                return LsResult {
                    f: fret,
                    fcnt,
                    gcnt,
                    inform: 0,
                };
            }
        }
    } else if fplus < f0 {
        do_extrap = true;
    }

    // ------------------------------------------------------------------
    // Extrapolation
    // ------------------------------------------------------------------
    if do_extrap {
        let mut extrap = 0usize;

        loop {
            if fplus <= fmin {
                xret.copy_from_slice(xplus);
                fret = fplus;
                if extrap != 0 || amax <= 1.0 {
                    compute_g(xret, sys, gret);
                    gcnt += 1;
                } else if gplus_valid {
                    gret.copy_from_slice(gplus);
                }
                return LsResult {
                    f: fret,
                    fcnt,
                    gcnt,
                    inform: 4,
                };
            }

            if fcnt >= maxfc {
                xret.copy_from_slice(xplus);
                fret = fplus;
                if extrap != 0 || amax <= 1.0 {
                    compute_g(xret, sys, gret);
                    gcnt += 1;
                } else if gplus_valid {
                    gret.copy_from_slice(gplus);
                }
                return LsResult {
                    f: fret,
                    fcnt,
                    gcnt,
                    inform: 8,
                };
            }

            if extrap >= maxextrap {
                xret.copy_from_slice(xplus);
                fret = fplus;
                if extrap != 0 || amax <= 1.0 {
                    compute_g(xret, sys, gret);
                    gcnt += 1;
                } else if gplus_valid {
                    gret.copy_from_slice(gplus);
                }
                return LsResult {
                    f: fret,
                    fcnt,
                    gcnt,
                    inform: 7,
                };
            }

            let atmp = if alpha < amax && next * alpha > amax {
                amax
            } else {
                next * alpha
            };

            xtmp.copy_from_slice(x);
            for &ii in &ind[..nind] {
                xtmp[ii] = x[ii] + atmp * d[ii];
            }
            if atmp == amax && rbdtype != 0 {
                if rbdtype == 1 {
                    xtmp[rbdind] = l[rbdind];
                } else {
                    xtmp[rbdind] = u[rbdind];
                }
            }
            if atmp > amax {
                for &ii in &ind[..nind] {
                    xtmp[ii] = xtmp[ii].clamp(l[ii], u[ii]);
                }
            }

            if alpha > amax {
                let mut samep = true;
                for &ii in &ind[..nind] {
                    if (xtmp[ii] - xplus[ii]).abs() > (epsrel * xplus[ii].abs()).max(epsabs) {
                        samep = false;
                        break;
                    }
                }

                if samep {
                    xret.copy_from_slice(xplus);
                    fret = fplus;
                    if extrap != 0 || amax <= 1.0 {
                        compute_g(xret, sys, gret);
                        gcnt += 1;
                    } else if gplus_valid {
                        gret.copy_from_slice(gplus);
                    }
                    return LsResult {
                        f: fret,
                        fcnt,
                        gcnt,
                        inform: 0,
                    };
                }
            }

            let ftmp = compute_f(xtmp, sys);
            fcnt += 1;

            if ftmp < fplus {
                alpha = atmp;
                fplus = ftmp;
                xplus.copy_from_slice(xtmp);
                gplus_valid = false;
                extrap += 1;
                continue;
            }

            xret.copy_from_slice(xplus);
            fret = fplus;
            if extrap != 0 || amax <= 1.0 {
                compute_g(xret, sys, gret);
                gcnt += 1;
            } else if gplus_valid {
                gret.copy_from_slice(gplus);
            } else {
                compute_g(xret, sys, gret);
                gcnt += 1;
            }
            return LsResult {
                f: fret,
                fcnt,
                gcnt,
                inform: 0,
            };
        }
    }

    // ------------------------------------------------------------------
    // Interpolation
    // ------------------------------------------------------------------
    let mut interp = 0usize;
    loop {
        if fplus <= fmin {
            xret.copy_from_slice(xplus);
            fret = fplus;
            compute_g(xret, sys, gret);
            gcnt += 1;
            return LsResult {
                f: fret,
                fcnt,
                gcnt,
                inform: 4,
            };
        }

        if fcnt >= maxfc {
            if fplus < f0 {
                xret.copy_from_slice(xplus);
                fret = fplus;
                compute_g(xret, sys, gret);
                gcnt += 1;
            }
            return LsResult {
                f: fret,
                fcnt,
                gcnt,
                inform: 8,
            };
        }

        if fplus <= f0 + gamma * alpha * gtd {
            xret.copy_from_slice(xplus);
            fret = fplus;
            compute_g(xret, sys, gret);
            gcnt += 1;
            return LsResult {
                f: fret,
                fcnt,
                gcnt,
                inform: 0,
            };
        }

        interp += 1;
        if alpha < sigma1 {
            alpha /= nint;
        } else {
            let denom = 2.0 * (fplus - f0 - alpha * gtd);
            let atmp = if denom != 0.0 {
                (-gtd * alpha * alpha) / denom
            } else {
                alpha / nint
            };
            if atmp < sigma1 || atmp > sigma2 * alpha {
                alpha /= nint;
            } else {
                alpha = atmp;
            }
        }

        xplus.copy_from_slice(x);
        for &ii in &ind[..nind] {
            xplus[ii] = x[ii] + alpha * d[ii];
        }

        fplus = compute_f(xplus, sys);
        fcnt += 1;

        let mut samep = true;
        for &ii in &ind[..nind] {
            if (alpha * d[ii]).abs() > (epsrel * x[ii].abs()).max(epsabs) {
                samep = false;
                break;
            }
        }
        if interp >= mininterp && samep {
            return LsResult {
                f: fret,
                fcnt,
                gcnt,
                inform: 6,
            };
        }
    }
}
