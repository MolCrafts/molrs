//! Pair-survival (persistence) time-correlation functions.
//!
//! Measures how long pairs of particles remain "bonded" — i.e. within a
//! distance cutoff — as a function of time lag. For a reference species `i` and
//! a partner species `j`, the persistence correlation at lag `τ` is the average
//! number of surviving partners per reference particle:
//!
//! ```text
//!     C(τ) = ⟨ (1/N_i) Σ_i Σ_j S_{ij}(t, t+τ) ⟩_t ,
//! ```
//!
//! where `S_{ij}(t, t+τ) ∈ {0, 1}` is the survival indicator for the pair
//! `(i, j)` born at time `t` and observed at `t + τ`. `C(0)` is the mean
//! coordination number (partners within the birth cutoff).
//!
//! Three survival definitions are supported, following the residence-time /
//! hydrogen-bond-dynamics literature (Rapaport 1983; Luzar & Chandler 1996):
//!
//! - **[`SurvivalMethod::Continuous`]** — `S = 1` only if the pair stayed
//!   within the survival cutoff `r1` at *every* frame in `[t, t+τ]` (continuous
//!   survival; the pair is removed the first time it breaks).
//! - **[`SurvivalMethod::Intermittent`]** — `S = 1` if the pair is within `r1`
//!   at `t + τ`, regardless of whether it left in between (re-formation allowed).
//! - **[`SurvivalMethod::Ssp`]** — stable-state picture: born within the inner
//!   cutoff `r0` and continuously within the outer cutoff `r1` ever since
//!   (`r1 ≥ r0`). A pair must leave `r1` to be considered broken, which
//!   suppresses rattling across a single cutoff.
//!
//! In all cases a pair is *born* at `t` only if it is within the inner cutoff
//! `r0` at `t`. For `Continuous` / `Intermittent` set `r1 = r0` for the usual
//! single-cutoff behaviour.
//!
//! This is the molrs port of the `persist` recipe / `tpairsurvive` operator
//! from the *tame* library (<https://github.com/Roy-Kid/tame>). The *tame*
//! `persist.py` recipe is non-functional as published (undefined names); this
//! port implements the intended pair-survival correlation with explicit,
//! well-defined survival criteria. Minimum-image distances use the
//! orthorhombic convention `d −= round(d / L)·L` per axis, matching *tame*'s
//! `tpairsurvive`.
//!
//! # Units
//!
//! Unit-agnostic for the correlation (a dimensionless count). Cutoffs and
//! coordinates share the same length unit (Å); `dt` sets the `lag_times` axis.

use ndarray::{Array1, Array2, Array3};

use crate::compute::error::ComputeError;

/// Pair-survival criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvivalMethod {
    /// Within `r1` at every frame since birth (breaks permanently on first exit).
    Continuous,
    /// Within `r1` at the observation frame, intermediate exits allowed.
    Intermittent,
    /// Stable-state picture: born within `r0`, continuously within `r1` since.
    Ssp,
}

impl SurvivalMethod {
    /// Parse a method name (case-insensitive). Accepts `continuous`/`cr`,
    /// `intermittent`/`imm`, `ssp`.
    pub fn parse(s: &str) -> Result<Self, ComputeError> {
        match s.to_ascii_lowercase().as_str() {
            "continuous" | "cr" | "rf" => Ok(SurvivalMethod::Continuous),
            "intermittent" | "imm" => Ok(SurvivalMethod::Intermittent),
            "ssp" => Ok(SurvivalMethod::Ssp),
            other => Err(ComputeError::OutOfRange {
                field: "method (expected continuous|intermittent|ssp)",
                value: other.to_string(),
            }),
        }
    }
}

/// Result of a pair-survival correlation computation.
#[derive(Debug, Clone)]
pub struct PersistResult {
    /// Lag times τ = i·dt, length `max_lag + 1` (same unit as `dt`).
    pub lag_times: Array1<f64>,
    /// Persistence correlation `C(τ)` (mean surviving partners per reference
    /// particle), length `max_lag + 1`. `C(0)` is the mean coordination number.
    pub correlation: Array1<f64>,
}

#[inline]
fn mic_dist2(a: [f64; 3], b: [f64; 3], l: [f64; 3]) -> f64 {
    let mut s = 0.0;
    for d in 0..3 {
        let mut dx = b[d] - a[d];
        if l[d] > 0.0 {
            dx -= (dx / l[d]).round() * l[d];
        }
        s += dx * dx;
    }
    s
}

/// Pair-survival time-correlation function between two species.
///
/// # Arguments
/// * `coords_i` — reference-species coordinates, shape `(n_frames, n_i, 3)`.
/// * `coords_j` — partner-species coordinates, shape `(n_frames, n_j, 3)`.
///   Must have the same `n_frames` as `coords_i`.
/// * `box_lengths` — per-frame orthorhombic box edge lengths, shape
///   `(n_frames, 3)`. A non-positive edge disables wrapping on that axis.
/// * `r0` — inner (birth) cutoff (> 0).
/// * `r1` — outer (survival) cutoff (≥ `r0`). For single-cutoff behaviour pass
///   `r1 = r0`.
/// * `method` — survival criterion, see [`SurvivalMethod`].
/// * `dt` — frame spacing (> 0), sets the `lag_times` axis.
/// * `max_correlation_time` — longest lag in **frames**, clamped to
///   `n_frames − 1`.
/// * `exclude_self` — when the two species are identical (`coords_i` and
///   `coords_j` index the same atoms), set `true` to drop the `i == j`
///   self-pair.
///
/// # Errors
/// * `DimensionMismatch` for wrong rank/shape or mismatched frame counts.
/// * `EmptyInput` if fewer than two frames or either species is empty.
/// * `OutOfRange` if `r0 ≤ 0`, `r1 < r0`, or `dt ≤ 0`.
#[allow(clippy::too_many_arguments)]
pub fn pair_survival_tcf(
    coords_i: &Array3<f64>,
    coords_j: &Array3<f64>,
    box_lengths: &Array2<f64>,
    r0: f64,
    r1: f64,
    method: SurvivalMethod,
    dt: f64,
    max_correlation_time: usize,
    exclude_self: bool,
) -> Result<PersistResult, ComputeError> {
    let si = coords_i.shape();
    let sj = coords_j.shape();
    if si[2] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: si[2],
            what: "coords_i (expected (n_frames, n_i, 3))",
        });
    }
    if sj[2] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: sj[2],
            what: "coords_j (expected (n_frames, n_j, 3))",
        });
    }
    let n_frames = si[0];
    if sj[0] != n_frames {
        return Err(ComputeError::DimensionMismatch {
            expected: n_frames,
            got: sj[0],
            what: "coords_i / coords_j frame count",
        });
    }
    let bl = box_lengths.shape();
    if bl[0] != n_frames || bl[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: n_frames,
            got: bl[0],
            what: "box_lengths (expected (n_frames, 3))",
        });
    }
    let n_i = si[1];
    let n_j = sj[1];
    if n_frames < 2 || n_i == 0 || n_j == 0 {
        return Err(ComputeError::EmptyInput);
    }
    if r0 <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "r0",
            value: r0.to_string(),
        });
    }
    if r1 < r0 {
        return Err(ComputeError::OutOfRange {
            field: "r1 (require r1 >= r0)",
            value: r1.to_string(),
        });
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    let same_species = exclude_self && n_i == n_j;

    let max_lag = max_correlation_time.min(n_frames - 1);
    let r0_2 = r0 * r0;
    let r1_2 = r1 * r1;

    // acc[τ] = Σ over (origin t0, i, j) of the survival indicator at lag τ.
    let mut acc = vec![0.0_f64; max_lag + 1];

    let pos = |coords: &Array3<f64>, t: usize, a: usize| -> [f64; 3] {
        [coords[[t, a, 0]], coords[[t, a, 1]], coords[[t, a, 2]]]
    };

    for t0 in 0..n_frames {
        let lmax = max_lag.min(n_frames - 1 - t0);
        let l0 = [
            box_lengths[[t0, 0]],
            box_lengths[[t0, 1]],
            box_lengths[[t0, 2]],
        ];
        for i in 0..n_i {
            let pi0 = pos(coords_i, t0, i);
            for j in 0..n_j {
                if same_species && i == j {
                    continue;
                }
                // Birth test at t0 (inner cutoff r0).
                if mic_dist2(pi0, pos(coords_j, t0, j), l0) > r0_2 {
                    continue;
                }
                acc[0] += 1.0; // born ⇒ alive at τ = 0
                match method {
                    SurvivalMethod::Continuous | SurvivalMethod::Ssp => {
                        // Walk forward; stop the first time the pair leaves r1.
                        // `tau` indexes both `acc` and the frame `t0 + tau`.
                        #[allow(clippy::needless_range_loop)]
                        for tau in 1..=lmax {
                            let t = t0 + tau;
                            let lt = [
                                box_lengths[[t, 0]],
                                box_lengths[[t, 1]],
                                box_lengths[[t, 2]],
                            ];
                            let d2 = mic_dist2(pos(coords_i, t, i), pos(coords_j, t, j), lt);
                            if d2 <= r1_2 {
                                acc[tau] += 1.0;
                            } else {
                                break;
                            }
                        }
                    }
                    SurvivalMethod::Intermittent => {
                        // Alive at τ if within r1 at t0+τ, gaps allowed.
                        // `tau` indexes both `acc` and the frame `t0 + tau`.
                        #[allow(clippy::needless_range_loop)]
                        for tau in 1..=lmax {
                            let t = t0 + tau;
                            let lt = [
                                box_lengths[[t, 0]],
                                box_lengths[[t, 1]],
                                box_lengths[[t, 2]],
                            ];
                            let d2 = mic_dist2(pos(coords_i, t, i), pos(coords_j, t, j), lt);
                            if d2 <= r1_2 {
                                acc[tau] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalize: mean over valid origins (n_frames − τ) and over reference
    // particles (n_i). The result is "surviving partners per reference atom".
    let mut correlation = Array1::<f64>::zeros(max_lag + 1);
    for tau in 0..=max_lag {
        let n_origins = (n_frames - tau) as f64;
        correlation[tau] = acc[tau] / (n_origins * n_i as f64);
    }

    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));

    Ok(PersistResult {
        lag_times,
        correlation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// Build a (n_frames, n, 3) array from per-frame lists of [x,y,z].
    fn coords(frames: &[Vec<[f64; 3]>]) -> Array3<f64> {
        let n_frames = frames.len();
        let n = frames[0].len();
        let mut a = Array3::<f64>::zeros((n_frames, n, 3));
        for (t, fr) in frames.iter().enumerate() {
            for (k, p) in fr.iter().enumerate() {
                for d in 0..3 {
                    a[[t, k, d]] = p[d];
                }
            }
        }
        a
    }

    fn boxes(n_frames: usize, l: f64) -> Array2<f64> {
        let mut b = Array2::<f64>::zeros((n_frames, 3));
        for t in 0..n_frames {
            for d in 0..3 {
                b[[t, d]] = l;
            }
        }
        b
    }

    #[test]
    fn permanently_bonded_pair_survives_fully() {
        // One i and one j, always 0.5 Å apart, cutoff 1.0 over 4 frames.
        let ci = coords(&[
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0]],
        ]);
        let cj = coords(&[
            vec![[0.5, 0.0, 0.0]],
            vec![[0.5, 0.0, 0.0]],
            vec![[0.5, 0.0, 0.0]],
            vec![[0.5, 0.0, 0.0]],
        ]);
        let b = boxes(4, 100.0);
        let r = pair_survival_tcf(
            &ci,
            &cj,
            &b,
            1.0,
            1.0,
            SurvivalMethod::Continuous,
            1.0,
            3,
            false,
        )
        .unwrap();
        // Always bonded ⇒ C(τ) = 1 for all τ.
        for tau in 0..=3 {
            assert!(
                (r.correlation[tau] - 1.0).abs() < 1e-12,
                "C({tau})={}",
                r.correlation[tau]
            );
        }
    }

    #[test]
    fn continuous_breaks_permanently_intermittent_reforms() {
        // Pair within cutoff at frames 0,1, leaves at 2, returns at 3.
        let ci = coords(&[
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0]],
        ]);
        let cj = coords(&[
            vec![[0.5, 0.0, 0.0]],
            vec![[0.5, 0.0, 0.0]],
            vec![[5.0, 0.0, 0.0]], // out of cutoff
            vec![[0.5, 0.0, 0.0]], // back in
        ]);
        let b = boxes(4, 100.0);

        // Only the origin t0=0 is a long-lived birth we track here, but every
        // frame is a candidate origin. Use intermittent vs continuous contrast
        // at origin 0 specifically by making just one pair.
        let cont = pair_survival_tcf(
            &ci,
            &cj,
            &b,
            1.0,
            1.0,
            SurvivalMethod::Continuous,
            1.0,
            3,
            false,
        )
        .unwrap();
        let imm = pair_survival_tcf(
            &ci,
            &cj,
            &b,
            1.0,
            1.0,
            SurvivalMethod::Intermittent,
            1.0,
            3,
            false,
        )
        .unwrap();

        // Continuous from origin 0: alive τ=0,1, dead from τ=2 onward.
        // Intermittent from origin 0: alive τ=0,1, dead τ=2, alive again τ=3.
        // At τ=3 only origin 0 is valid (n_frames-τ = 1).
        assert!(
            cont.correlation[3].abs() < 1e-12,
            "continuous C(3)={}",
            cont.correlation[3]
        );
        assert!(
            (imm.correlation[3] - 1.0).abs() < 1e-12,
            "intermittent C(3)={}",
            imm.correlation[3]
        );
    }

    #[test]
    fn ssp_uses_inner_birth_outer_survival_cutoff() {
        // Born requires r<=r0=1.0; survives while r<=r1=2.0.
        // Pair at 0.5 (born) then drifts to 1.5 (still < r1, alive under SSP).
        let ci = coords(&[vec![[0.0, 0.0, 0.0]], vec![[0.0, 0.0, 0.0]]]);
        let cj = coords(&[vec![[0.5, 0.0, 0.0]], vec![[1.5, 0.0, 0.0]]]);
        let b = boxes(2, 100.0);
        let ssp =
            pair_survival_tcf(&ci, &cj, &b, 1.0, 2.0, SurvivalMethod::Ssp, 1.0, 1, false).unwrap();
        // Born only at frame 0 (at frame 1 the pair is at 1.5 > r0=1.0), so the
        // coordination C(0) averaged over both origin frames is 0.5.
        assert!(
            (ssp.correlation[0] - 0.5).abs() < 1e-12,
            "C(0)={}",
            ssp.correlation[0]
        );
        // τ=1 has a single valid origin (t0=0); 1.5 < r1=2.0 ⇒ still alive ⇒ 1.0.
        assert!((ssp.correlation[1] - 1.0).abs() < 1e-12); // 1.5 < r1=2.0 still alive
        // With a single cutoff 1.0 it would have died at τ=1.
        let cont = pair_survival_tcf(
            &ci,
            &cj,
            &b,
            1.0,
            1.0,
            SurvivalMethod::Continuous,
            1.0,
            1,
            false,
        )
        .unwrap();
        assert!(cont.correlation[1].abs() < 1e-12);
    }

    #[test]
    fn coordination_number_at_lag_zero() {
        // One i, two j both within cutoff ⇒ C(0) = 2 partners / 1 reference.
        let ci = coords(&[vec![[0.0, 0.0, 0.0]], vec![[0.0, 0.0, 0.0]]]);
        let cj = coords(&[
            vec![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            vec![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        ]);
        let b = boxes(2, 100.0);
        let r = pair_survival_tcf(
            &ci,
            &cj,
            &b,
            1.0,
            1.0,
            SurvivalMethod::Intermittent,
            1.0,
            1,
            false,
        )
        .unwrap();
        assert!(
            (r.correlation[0] - 2.0).abs() < 1e-12,
            "C(0)={}",
            r.correlation[0]
        );
    }

    #[test]
    fn minimum_image_wraps_across_boundary() {
        // i at 0.2, j at 9.8 in a box of length 10: MIC distance = 0.4 < 1.0.
        let ci = coords(&[vec![[0.2, 0.0, 0.0]], vec![[0.2, 0.0, 0.0]]]);
        let cj = coords(&[vec![[9.8, 0.0, 0.0]], vec![[9.8, 0.0, 0.0]]]);
        let b = boxes(2, 10.0);
        let r = pair_survival_tcf(
            &ci,
            &cj,
            &b,
            1.0,
            1.0,
            SurvivalMethod::Continuous,
            1.0,
            1,
            false,
        )
        .unwrap();
        assert!((r.correlation[0] - 1.0).abs() < 1e-12); // bonded via PBC
    }

    #[test]
    fn exclude_self_drops_diagonal() {
        // Same species, 2 atoms far apart; only cross-pairs counted.
        let c = coords(&[
            vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        ]);
        let b = boxes(2, 100.0);
        let r = pair_survival_tcf(
            &c,
            &c,
            &b,
            1.0,
            1.0,
            SurvivalMethod::Intermittent,
            1.0,
            1,
            true,
        )
        .unwrap();
        // Each atom has exactly 1 partner (the other) within cutoff ⇒ C(0)=1.
        assert!(
            (r.correlation[0] - 1.0).abs() < 1e-12,
            "C(0)={}",
            r.correlation[0]
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let c = coords(&[vec![[0.0, 0.0, 0.0]], vec![[0.0, 0.0, 0.0]]]);
        let b = boxes(2, 10.0);
        assert!(
            pair_survival_tcf(&c, &c, &b, 0.0, 1.0, SurvivalMethod::Ssp, 1.0, 1, false).is_err()
        );
        assert!(
            pair_survival_tcf(&c, &c, &b, 2.0, 1.0, SurvivalMethod::Ssp, 1.0, 1, false).is_err()
        );
        assert!(SurvivalMethod::parse("bogus").is_err());
        assert_eq!(SurvivalMethod::parse("SSP").unwrap(), SurvivalMethod::Ssp);
    }
}
