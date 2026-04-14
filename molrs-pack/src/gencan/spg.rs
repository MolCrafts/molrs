//! Spectral Projected Gradient line search.
//! Port of `spgls` from `gencan.f`.

use crate::constraints::EvalMode;
use crate::objective::Objective;
use molrs::types::F;

/// Reusable buffers for SPG line search.
///
/// This mirrors Packmol's reuse of work arrays inside GENCAN while keeping
/// Rust ownership explicit.
pub struct SpgScratch {
    pub xtrial: Vec<F>,
    pub d: Vec<F>,
}

impl SpgScratch {
    pub fn new(n: usize) -> Self {
        Self {
            xtrial: vec![0.0; n],
            d: vec![0.0; n],
        }
    }

    pub fn ensure_len(&mut self, n: usize) {
        if self.xtrial.len() != n {
            self.xtrial.resize(n, 0.0);
        }
        if self.d.len() != n {
            self.d.resize(n, 0.0);
        }
    }
}

pub struct SpgResult {
    pub f: F,
    pub fcnt: usize,
    pub inform: i32,
}

/// Safeguarded quadratic interpolation along the spectral projected gradient direction.
/// Port of Fortran `spgls`.
#[allow(clippy::too_many_arguments)]
pub fn spgls(
    n: usize,
    x: &[F],
    g: &[F],
    l: &[F],
    u: &[F],
    lamspg: F,
    f0: F,
    nint: F,
    mininterp: usize,
    fmin: F,
    maxfc: usize,
    mut fcnt: usize,
    gamma: F,
    sigma1: F,
    sigma2: F,
    _sterel: F,
    _steabs: F,
    epsrel: F,
    epsabs: F,
    scratch: &mut SpgScratch,
    obj: &mut dyn Objective,
) -> SpgResult {
    scratch.ensure_len(n);
    let xtrial = &mut scratch.xtrial;
    let d = &mut scratch.d;

    // Compute first trial point and directional derivative
    for i in 0..n {
        xtrial[i] = (x[i] - lamspg * g[i]).clamp(l[i], u[i]);
        d[i] = xtrial[i] - x[i];
    }
    let gtd: F = g.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();

    let mut ftrial = obj.evaluate(xtrial, EvalMode::FOnly, None).f_total;
    fcnt += 1;

    let mut alpha = 1.0 as F;
    let mut interp = 0usize;
    let f = f0;

    loop {
        // Armijo criterion
        if ftrial <= f + gamma * alpha * gtd {
            return SpgResult {
                f: ftrial,
                fcnt,
                inform: 0,
            };
        }

        if ftrial <= fmin {
            return SpgResult {
                f: ftrial,
                fcnt,
                inform: 4,
            };
        }

        if fcnt >= maxfc {
            let fret = if ftrial < f {
                ftrial
            } else {
                xtrial.copy_from_slice(x);
                f
            };
            return SpgResult {
                f: fret,
                fcnt,
                inform: 8,
            };
        }

        // Safeguarded quadratic interpolation
        interp += 1;
        let atmp = if alpha < sigma1 {
            alpha / nint
        } else {
            let candidate = (-gtd * alpha * alpha) / (2.0 * (ftrial - f - alpha * gtd));
            if candidate < sigma1 || candidate > sigma2 * alpha {
                alpha / nint
            } else {
                candidate
            }
        };
        alpha = atmp;

        // New trial point: x + alpha * d
        for i in 0..n {
            xtrial[i] = x[i] + alpha * d[i];
        }

        ftrial = obj.evaluate(xtrial, EvalMode::FOnly, None).f_total;
        fcnt += 1;

        // Check for too-small step
        let samep = d
            .iter()
            .zip(x)
            .all(|(di, xi)| (alpha * di).abs() <= (epsrel * xi.abs()).max(epsabs));

        if interp >= mininterp && samep {
            let fret = if ftrial < f {
                ftrial
            } else {
                xtrial.copy_from_slice(x);
                f
            };
            return SpgResult {
                f: fret,
                fcnt,
                inform: 6,
            };
        }
    }
}
