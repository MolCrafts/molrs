//! Domain-validation checks for computed dielectric / conductivity spectra.
//!
//! Pure numerical routines (no Python, no I/O) that verify physical
//! self-consistency of computed spectra:
//!
//! * [`KramersKronig`] — recover ε'(ω) from ε''(ω) via a discrete
//!   Kramers-Kronig integral and compare to the supplied real part.
//! * [`ConductivitySumRule`] — verify ∫₀^∞ σ(ω) dω = (π/2)·⟨J²⟩/(3 V k_B T).
//! * [`RouteAgreement`] — pairwise RMS distance between named ε(ω) arrays.
//!
//! These live here (not in the Python binding) so the physics is testable in
//! Rust and shares the single Boltzmann constant with the dielectric module.
//!
//! Each criterion is a [`Check`] impl: an immutable parameter bag whose
//! [`check`](Check::check) consumes an upstream spectrum and returns a
//! [`Verdict`]. **A violated criterion is `Ok(verdict)` with
//! `passed() == false`** — `Err` is reserved for inputs the criterion cannot
//! be evaluated on at all.

use ndarray::Array1;

use crate::compute::dielectric::K_B;
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::{Check, Verdict};

/// Outcome of the Kramers-Kronig consistency check.
#[derive(Debug, Clone)]
pub struct KramersKronigCheck {
    /// ε'(ω) recovered from ε''(ω).
    pub recovered: Array1<f64>,
    /// Mean absolute error vs the supplied real part.
    pub mae: f64,
    /// Whether `mae` is within the (dynamic-range-scaled) tolerance.
    pub passed: bool,
}

/// Recover ε'(ω) from ε''(ω) by a discrete Kramers-Kronig integral with
/// trapezoidal weights, offset by ε∞ (the principal value skips j == i).
fn discrete_kramers_kronig(
    omega: &Array1<f64>,
    eps_imag: &Array1<f64>,
    eps_inf: f64,
) -> Array1<f64> {
    let n = omega.len();
    let mut recovered = Array1::<f64>::from_elem(n, eps_inf);
    for i in 0..n {
        let omega_i = omega[i];
        let mut acc = 0.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let omega_j = omega[j];
            let denom = omega_j * omega_j - omega_i * omega_i;
            if denom.abs() < 1e-30 {
                continue;
            }
            let dw = if j == 0 {
                omega[1] - omega[0]
            } else if j == n - 1 {
                omega[n - 1] - omega[n - 2]
            } else {
                0.5 * (omega[j + 1] - omega[j - 1])
            };
            acc += eps_imag[j] * omega_j / denom * dw;
        }
        recovered[i] += (2.0 / std::f64::consts::PI) * acc;
    }
    recovered
}

/// Kramers-Kronig causality of a dielectric spectrum.
///
/// `eps_inf` is the high-frequency limit ε∞ that offsets the recovered real
/// part. Requires ≥ 3 frequency points and equal-length arrays.
#[derive(Debug, Clone, Copy)]
pub struct KramersKronig {
    /// High-frequency limit ε∞.
    pub eps_inf: f64,
}

impl Check for KramersKronig {
    /// `(omega, eps_real, eps_imag)`.
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, &'a Array1<f64>);
    type Output = KramersKronigCheck;

    fn check<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (omega, eps_real, eps_imag) = input;
        let eps_inf = self.eps_inf;
        require_same_len(
            "kramers_kronig eps_real vs frequency",
            omega.len(),
            eps_real.len(),
        )?;
        require_same_len(
            "kramers_kronig eps_imag vs frequency",
            omega.len(),
            eps_imag.len(),
        )?;
        if omega.len() < 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: omega.len(),
                what: "kramers_kronig needs at least 3 frequency points",
            });
        }

        let recovered = discrete_kramers_kronig(omega, eps_imag, eps_inf);
        let sum_abs: f64 = recovered
            .iter()
            .zip(eps_real.iter())
            .map(|(r, e)| (r - e).abs())
            .sum();
        let mae = sum_abs / (eps_real.len() as f64);
        // Tolerance scales with the dynamic range of the real part so small
        // absolute residuals on bounded dielectrics are forgiven.
        let dynamic_range = eps_real.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - eps_real.iter().copied().fold(f64::INFINITY, f64::min);
        let tol = (dynamic_range.abs() * 0.1).max(1e-2);
        Ok(KramersKronigCheck {
            recovered,
            mae,
            passed: mae < tol,
        })
    }
}

/// Outcome of the conductivity sum-rule check.
#[derive(Debug, Clone, Copy)]
pub struct SumRuleCheck {
    /// Trapezoidal ∫ σ(ω) dω.
    pub integral: f64,
    /// (π/2)·⟨J²⟩/(3 V k_B T).
    pub expected: f64,
    /// `(integral - expected) / |expected|`.
    pub relative_error: f64,
    /// Whether `|relative_error| < 0.05`.
    pub passed: bool,
}

/// Conductivity sum rule ∫₀^∞ σ(ω) dω = (π/2)·⟨J²⟩/(3 V k_B T).
///
/// Requires ≥ 2 frequency points, positive finite `volume` / `temperature`,
/// and a finite non-negative `current_sq_mean`.
#[derive(Debug, Clone, Copy)]
pub struct ConductivitySumRule {
    /// ⟨J²⟩, the mean squared current.
    pub current_sq_mean: f64,
    /// Cell volume.
    pub volume: f64,
    /// Temperature.
    pub temperature: f64,
}

impl Check for ConductivitySumRule {
    /// `(omega, sigma)`.
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>);
    type Output = SumRuleCheck;

    fn check<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (omega, sigma) = input;
        let Self {
            current_sq_mean,
            volume,
            temperature,
        } = *self;
        require_same_len(
            "sum_rule conductivity vs frequency",
            omega.len(),
            sigma.len(),
        )?;
        require_positive("volume", volume)?;
        require_positive("temperature", temperature)?;
        if !current_sq_mean.is_finite() || current_sq_mean < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "current_sq_mean",
                value: current_sq_mean.to_string(),
            });
        }
        if omega.len() < 2 {
            return Err(ComputeError::DimensionMismatch {
                expected: 2,
                got: omega.len(),
                what: "conductivity sum rule needs at least 2 frequency points",
            });
        }

        let mut integral = 0.0;
        for i in 1..omega.len() {
            let dw = omega[i] - omega[i - 1];
            integral += 0.5 * (sigma[i] + sigma[i - 1]) * dw;
        }
        let expected =
            std::f64::consts::PI * 0.5 * current_sq_mean / (3.0 * volume * K_B * temperature);
        let denom = expected.abs().max(1e-30);
        let relative_error = (integral - expected) / denom;
        Ok(SumRuleCheck {
            integral,
            expected,
            relative_error,
            passed: relative_error.abs() < 0.05,
        })
    }
}

/// Outcome of the route-agreement check.
#[derive(Debug, Clone)]
pub struct RouteAgreementCheck {
    /// `(pair_label, relative_rms)` for each unordered pair, in input order.
    pub pairwise: Vec<(String, f64)>,
    /// Largest pairwise relative RMS.
    pub max_rms: f64,
    /// Whether `max_rms < 0.10`.
    pub passed: bool,
}

/// Pairwise relative-RMS agreement between named ε(ω) arrays — e.g. the same
/// quantity computed by the Einstein-Helfand and Green-Kubo routes.
///
/// Requires ≥ 2 equal-length arrays.
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteAgreement;

impl Check for RouteAgreement {
    /// Named result arrays, one per route.
    type Input<'a> = &'a [(String, Array1<f64>)];
    type Output = RouteAgreementCheck;

    fn check<'a>(&self, entries: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        if entries.len() < 2 {
            return Err(ComputeError::DimensionMismatch {
                expected: 2,
                got: entries.len(),
                what: "route agreement needs at least two named result arrays",
            });
        }
        let expected_len = entries[0].1.len();
        for (_name, arr) in entries {
            require_same_len("route_agreement array lengths", expected_len, arr.len())?;
        }

        let mut pairwise = Vec::new();
        let mut max_rms = 0.0_f64;
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let (name_i, arr_i) = &entries[i];
                let (name_j, arr_j) = &entries[j];
                let mut sum_sq = 0.0;
                let mut norm = 0.0;
                for k in 0..expected_len {
                    let diff = arr_i[k] - arr_j[k];
                    sum_sq += diff * diff;
                    norm += 0.5 * (arr_i[k].abs() + arr_j[k].abs());
                }
                let rms_abs = (sum_sq / (expected_len as f64)).sqrt();
                let scale = (norm / (expected_len as f64)).max(1e-30);
                let rms_rel = rms_abs / scale;
                max_rms = max_rms.max(rms_rel);
                pairwise.push((format!("{name_i}_vs_{name_j}"), rms_rel));
            }
        }
        Ok(RouteAgreementCheck {
            pairwise,
            max_rms,
            passed: max_rms < 0.10,
        })
    }
}

fn require_same_len(what: &'static str, expected: usize, got: usize) -> Result<(), ComputeError> {
    if expected != got {
        return Err(ComputeError::DimensionMismatch {
            expected,
            got,
            what,
        });
    }
    Ok(())
}

fn require_positive(field: &'static str, value: f64) -> Result<(), ComputeError> {
    if !(value.is_finite() && value > 0.0) {
        return Err(ComputeError::OutOfRange {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

impl Verdict for KramersKronigCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}
impl ComputeResult for KramersKronigCheck {}

impl Verdict for SumRuleCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}
impl ComputeResult for SumRuleCheck {}

impl Verdict for RouteAgreementCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}
impl ComputeResult for RouteAgreementCheck {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kk_recovers_eps_inf_for_zero_loss() {
        // ε'' ≡ 0 ⇒ the KK integral is zero, so the recovered real part is ε∞
        // everywhere; matching eps_real ⇒ mae 0, passed.
        let omega = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let eps_imag = Array1::zeros(4);
        let eps_real = Array1::from_elem(4, 2.5);
        let out = KramersKronig { eps_inf: 2.5 }
            .check((&omega, &eps_real, &eps_imag))
            .unwrap();
        assert!(out.mae < 1e-12, "mae {}", out.mae);
        assert!(out.passed);
    }

    #[test]
    fn kk_requires_three_points() {
        let omega = Array1::from_vec(vec![1.0, 2.0]);
        let z = Array1::zeros(2);
        assert!(
            KramersKronig { eps_inf: 1.0 }
                .check((&omega, &z, &z))
                .is_err()
        );
    }

    #[test]
    fn sum_rule_matches_constructed_integral() {
        // Constant σ over [0, 4]: trapezoidal integral = σ·4 = 8. Pick the
        // physical inputs so `expected` equals 8 ⇒ relative_error 0, passed.
        let omega = Array1::from_vec(vec![0.0, 2.0, 4.0]);
        let sigma = Array1::from_elem(3, 2.0);
        let volume = 100.0;
        let temperature = 300.0;
        let expected = 8.0;
        // expected = π/2 · ⟨J²⟩ / (3 V k_B T)  ⇒  ⟨J²⟩ = expected·3 V k_B T / (π/2)
        let current_sq_mean =
            expected * 3.0 * volume * K_B * temperature / (std::f64::consts::PI * 0.5);
        let out = ConductivitySumRule {
            current_sq_mean,
            volume,
            temperature,
        }
        .check((&omega, &sigma))
        .unwrap();
        assert!((out.integral - 8.0).abs() < 1e-12);
        assert!(out.relative_error.abs() < 1e-12, "{}", out.relative_error);
        assert!(out.passed);
    }

    #[test]
    fn sum_rule_rejects_nonpositive_volume() {
        let omega = Array1::from_vec(vec![0.0, 1.0]);
        let sigma = Array1::from_elem(2, 1.0);
        assert!(
            ConductivitySumRule {
                current_sq_mean: 1.0,
                volume: 0.0,
                temperature: 300.0
            }
            .check((&omega, &sigma))
            .is_err()
        );
    }

    #[test]
    fn route_agreement_identical_is_zero() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let entries = vec![("r1".to_string(), a.clone()), ("r2".to_string(), a)];
        let out = RouteAgreement.check(&entries).unwrap();
        assert_eq!(out.pairwise.len(), 1);
        assert_eq!(out.pairwise[0].0, "r1_vs_r2");
        assert!(out.max_rms < 1e-12);
        assert!(out.passed);
    }

    #[test]
    fn route_agreement_flags_divergence() {
        let a = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let b = Array1::from_vec(vec![2.0, 2.0, 2.0]);
        let entries = vec![("a".to_string(), a), ("b".to_string(), b)];
        let out = RouteAgreement.check(&entries).unwrap();
        assert!(out.max_rms > 0.1);
        assert!(!out.passed);
    }
}
