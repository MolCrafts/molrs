//! Hydrogen-bond lifetime correlation functions.
//!
//! Continuous `S_HB(t)` and intermittent `C_HB(t)` TCFs (Luzar & Chandler,
//! *Phys. Rev. Lett.* **1996**, 76, 928): with `h(t) ∈ {0,1}` the per-bond
//! presence indicator,
//!
//! ```text
//! C_HB(t) = ⟨h(0) h(t)⟩ / ⟨h⟩            (intermittent — gaps allowed)
//! S_HB(t) = ⟨h(0) Θ(t) h(t)⟩ / ⟨h⟩       (continuous — bond never broke in [0,t])
//! ```
//!
//! The continuous / intermittent accumulation mirrors molrs's
//! [`pair_survival_tcf`](crate::compute::persist::pair_survival_tcf) exactly
//! (`SurvivalMethod::Continuous` walks forward until the first absence;
//! `Intermittent` counts every lag where the bond is present).
//!
//! **Documented deviation:** `pair_survival_tcf`'s presence model is a *distance*
//! cutoff and cannot express the D–H···A *angle* criterion, so its API cannot be
//! called directly here. This module applies the identical continuous /
//! intermittent definitions to the geometric presence series produced by
//! [`HBonds`](super::detect::HBonds), rather than re-deriving a different TCF.

use molrs::types::F;
use ndarray::Array1;
use std::collections::HashMap;

use super::detect::HBondsResult;
use crate::compute::error::ComputeError;

/// Lifetime TCFs for a set of bond presence series.
#[derive(Debug, Clone)]
pub struct LifetimeResult {
    /// Lag times (`tau * dt`), length `max_lag + 1`.
    pub lag_times: Array1<F>,
    /// Continuous `S_HB(t)`, normalized to `S_HB(0) = 1`.
    pub continuous: Array1<F>,
    /// Intermittent `C_HB(t)`, normalized to `C_HB(0) = 1`.
    pub intermittent: Array1<F>,
    /// Continuous lifetime `τ_c = ∫ S_HB(t) dt` (trapezoidal).
    pub tau_continuous: F,
    /// Intermittent lifetime `τ_i = ∫ C_HB(t) dt` (trapezoidal).
    pub tau_intermittent: F,
}

fn trapz(y: &Array1<F>, dt: F) -> F {
    if y.len() < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    for w in y.windows(2).into_iter() {
        s += 0.5 * (w[0] + w[1]) * dt;
    }
    s
}

/// Compute continuous / intermittent lifetime TCFs from per-bond binary presence
/// series `present[bond][frame]`.
///
/// All series must share the same frame count. `dt` is the frame spacing
/// (> 0); `max_lag` is clamped to `n_frames − 1`.
pub fn hbond_lifetimes(
    present: &[Vec<bool>],
    dt: F,
    max_lag: usize,
) -> Result<LifetimeResult, ComputeError> {
    if present.is_empty() {
        return Err(ComputeError::EmptyInput);
    }
    let n_frames = present[0].len();
    if n_frames < 2 {
        return Err(ComputeError::EmptyInput);
    }
    for s in present {
        if s.len() != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: s.len(),
                what: "hbond presence series length",
            });
        }
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "hbond_lifetimes::dt",
            value: dt.to_string(),
        });
    }
    let max_lag = max_lag.min(n_frames - 1);
    let n_series = present.len();

    // acc_*[tau] = Σ over (bond, origin t0) of the survival indicator at lag tau.
    let mut acc_c = vec![0.0_f64; max_lag + 1];
    let mut acc_i = vec![0.0_f64; max_lag + 1];
    for h in present {
        for t0 in 0..n_frames {
            if !h[t0] {
                continue;
            }
            let lmax = max_lag.min(n_frames - 1 - t0);
            acc_c[0] += 1.0;
            acc_i[0] += 1.0;
            // Continuous: stop at the first absence.
            for tau in 1..=lmax {
                if h[t0 + tau] {
                    acc_c[tau] += 1.0;
                } else {
                    break;
                }
            }
            // Intermittent: count every present lag, gaps allowed.
            for tau in 1..=lmax {
                if h[t0 + tau] {
                    acc_i[tau] += 1.0;
                }
            }
        }
    }

    // Per-lag normalization: divide by the number of valid (bond, origin) pairs
    // at that lag → ⟨h(0)h(tau)⟩, then divide by the tau=0 value so C(0)=1.
    let mut cont = Array1::<F>::zeros(max_lag + 1);
    let mut inter = Array1::<F>::zeros(max_lag + 1);
    let mean = |acc: &[f64], tau: usize| -> F {
        let n_pairs = (n_series * (n_frames - tau)) as F;
        if n_pairs > 0.0 {
            acc[tau] / n_pairs
        } else {
            0.0
        }
    };
    let c0 = mean(&acc_c, 0);
    let i0 = mean(&acc_i, 0);
    for tau in 0..=max_lag {
        cont[tau] = if c0 > 0.0 {
            mean(&acc_c, tau) / c0
        } else {
            0.0
        };
        inter[tau] = if i0 > 0.0 {
            mean(&acc_i, tau) / i0
        } else {
            0.0
        };
    }

    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as F * dt));
    let tau_continuous = trapz(&cont, dt);
    let tau_intermittent = trapz(&inter, dt);

    Ok(LifetimeResult {
        lag_times,
        continuous: cont,
        intermittent: inter,
        tau_continuous,
        tau_intermittent,
    })
}

/// Build per-bond presence series from an [`HBondsResult`], keyed by the ordered
/// `(donor, acceptor)` pair. Returns the bond keys and the
/// `present[bond][frame]` matrix, ready for [`hbond_lifetimes`].
pub fn presence_from_hbonds(res: &HBondsResult) -> (Vec<(u32, u32)>, Vec<Vec<bool>>) {
    let n_frames = res.per_frame.len();
    let mut index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut keys: Vec<(u32, u32)> = Vec::new();
    for frame in &res.per_frame {
        for b in frame {
            let key = (b.donor, b.acceptor);
            if let std::collections::hash_map::Entry::Vacant(e) = index.entry(key) {
                e.insert(keys.len());
                keys.push(key);
            }
        }
    }
    let mut present = vec![vec![false; n_frames]; keys.len()];
    for (t, frame) in res.per_frame.iter().enumerate() {
        for b in frame {
            let bi = index[&(b.donor, b.acceptor)];
            present[bi][t] = true;
        }
    }
    (keys, present)
}
