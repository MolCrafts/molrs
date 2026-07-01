//! Legendre reorientational time-correlation functions `C_ℓ(t)`.
//!
//! For a molecular **unit vector** `û` (an O–H bond, a dipole axis, …) the
//! reorientational TCFs are
//!
//! ```text
//!   C_1(t) = ⟨ P_1(û(0)·û(t)) ⟩ = ⟨ cos θ(t) ⟩
//!   C_2(t) = ⟨ P_2(û(0)·û(t)) ⟩ = ⟨ (3 cos²θ(t) − 1) / 2 ⟩
//! ```
//!
//! averaged over molecules and (multi-)time origins. `C_1` is the dielectric /
//! IR reorientation observable; the `C_2` decay gives the NMR rotational
//! correlation time τ_c (Berne & Pecora, *Dynamic Light Scattering*; standard
//! NMR relaxation theory).
//!
//! # Provenance
//!
//! Ported from TRAVIS's reorientation-dynamics analyzer `CReorDyn`
//! (`src/reordyn.cpp`): per-molecule unit vectors (cf. `src/order_vector.cpp`),
//! the 1st/2nd-Legendre selection (`m_bLegendre2`), and the multi-time-origin
//! correlation-depth accumulation (`m_iDepth`, the ACF origin sums in
//! `src/acf.cpp`). TRAVIS evaluates the origin average via FFT; molrs uses the
//! algebraically-identical **direct** multi-origin double sum here (exact,
//! O(T²·N)), which the spec permits — the Legendre polynomials are applied to
//! the dot product *before* averaging, so an FFT factorization does not apply to
//! `C_2` anyway.
//!
//! # Relation to [`RotationalAutocorrelation`](super::rotational_autocorrelation)
//!
//! This analyzer is **distinct** from
//! [`RotationalAutocorrelation`](super::rotational_autocorrelation::RotationalAutocorrelation):
//! that one is freud's rigid-body **quaternion** autocorrelation (a Wigner-D
//! character of the full orientation), whereas `LegendreReorientation` correlates
//! a single molecular **vector** via Legendre polynomials. They measure different
//! observables and both remain independently usable.
//!
//! # Fitting τ_c
//!
//! The raw `C_2(t)` curve is `Fit`-ready: feed it to
//! [`DebyeFit`](crate::compute::fit::DebyeFit) (which fits a normalized
//! `Φ(t) → τ` decay) to extract the rotational correlation time — no new fitting
//! code is needed.

use molrs::store::frame_access::FrameAccess;
use molrs::types::F;
use ndarray::Array1;

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use crate::compute::util::{get_positions_ref, mic_disp};

/// Legendre reorientational TCF analyzer.
///
/// Stateless parameter bag: the maximum lag (correlation depth, in frames) and
/// the time-origin stride. The molecular vectors are selected by the `Args`
/// passed to [`compute`](Compute::compute): one `(a, b)` atom-index pair per
/// molecule, defining `û = normalize(r_b − r_a)` (minimum-image).
#[derive(Debug, Clone)]
pub struct LegendreReorientation {
    max_lag: usize,
    stride: usize,
}

impl LegendreReorientation {
    /// New analyzer evaluating `C_1`/`C_2` at lags `0..=max_lag` (clamped to the
    /// trajectory length at compute time).
    pub fn new(max_lag: usize) -> Self {
        Self { max_lag, stride: 1 }
    }

    /// Set the time-origin stride (default 1).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride.max(1);
        self
    }
}

fn p1(x: F) -> F {
    x
}
fn p2(x: F) -> F {
    0.5 * (3.0 * x * x - 1.0)
}

impl Compute for LegendreReorientation {
    /// One `(atom_a, atom_b)` index pair per molecular vector.
    type Args<'a> = &'a [(u32, u32)];
    type Output = LegendreReorientationResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        pairs: &'a [(u32, u32)],
    ) -> Result<LegendreReorientationResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if pairs.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "LegendreReorientation::pairs",
                value: "empty".to_string(),
            });
        }
        let n_frames = frames.len();
        let n_vec = pairs.len();

        // Per-frame unit vectors [frame][mol] = [x,y,z].
        let mut vecs: Vec<Vec<[F; 3]>> = Vec::with_capacity(n_frames);
        for frame in frames {
            let (xp, yp, zp) = get_positions_ref(*frame)?;
            let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
            let simbox = frame.simbox_ref();
            let mut fv = Vec::with_capacity(n_vec);
            for &(a, b) in pairs {
                let (a, b) = (a as usize, b as usize);
                if a >= xs.len() || b >= xs.len() {
                    return Err(ComputeError::DimensionMismatch {
                        expected: xs.len(),
                        got: a.max(b) + 1,
                        what: "LegendreReorientation atom index",
                    });
                }
                let d = mic_disp(simbox, [xs[a], ys[a], zs[a]], [xs[b], ys[b], zs[b]]);
                let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    return Err(ComputeError::NonFinite {
                        where_: "LegendreReorientation unit vector (zero-length)",
                        index: 0,
                    });
                }
                fv.push([d[0] / norm, d[1] / norm, d[2] / norm]);
            }
            vecs.push(fv);
        }

        let max_lag = self.max_lag.min(n_frames - 1);
        let mut c1 = Array1::<F>::zeros(max_lag + 1);
        let mut c2 = Array1::<F>::zeros(max_lag + 1);

        for t in 0..=max_lag {
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut count: usize = 0;
            let mut tau = 0usize;
            while tau + t < n_frames {
                let a = &vecs[tau];
                let b = &vecs[tau + t];
                for m in 0..n_vec {
                    let x = a[m][0] * b[m][0] + a[m][1] * b[m][1] + a[m][2] * b[m][2];
                    s1 += p1(x);
                    s2 += p2(x);
                    count += 1;
                }
                tau += self.stride;
            }
            let inv = if count > 0 { 1.0 / count as F } else { 0.0 };
            c1[t] = s1 * inv;
            c2[t] = s2 * inv;
        }

        Ok(LegendreReorientationResult {
            lags: (0..=max_lag).collect(),
            c1,
            c2,
        })
    }
}

/// Result of a [`LegendreReorientation`] computation.
#[derive(Debug, Clone)]
pub struct LegendreReorientationResult {
    /// Lag times (frame units), aligned with `c1`/`c2`.
    pub lags: Vec<usize>,
    /// `C_1(t) = ⟨cos θ(t)⟩`.
    pub c1: Array1<F>,
    /// `C_2(t) = ⟨(3cos²θ(t) − 1)/2⟩`.
    pub c2: Array1<F>,
}

impl ComputeResult for LegendreReorientationResult {}
