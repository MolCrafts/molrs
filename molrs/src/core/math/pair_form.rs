//! Closed-form conversions between parameterisations of one pair potential.
//!
//! The same 12-6 interaction is written two ways. Amber files carry
//! `E = A/r¹² − B/r⁶`; molrs kernels take `E = 4ε[(σ/r)¹² − (σ/r)⁶]`. The
//! identity between them belongs to neither the reader that parses `A`/`B` nor
//! the force field that consumes σ/ε, so it lives here, where both can reach
//! it without `ff` importing `io`.

use crate::types::F;

/// Lennard-Jones `A`/`B` coefficients to `(σ, ε)`.
///
/// `r_min = (2A/B)^{1/6}`, `ε = B²/(4A)`, `σ = 2^{−1/6} r_min`. A row with
/// either coefficient zero is a non-interacting type — Amber writes those for
/// types that carry no vdW term — and yields `(1.0, 0.0)`: ε of zero switches
/// the term off, and σ is given a benign non-zero value so that no consumer
/// divides by it.
///
/// ```
/// # use molrs::math::pair_form::lj_ab_to_sigma_epsilon;
/// // GAFF c3: R* = 1.9080 Å, ε = 0.1094 kcal/mol.
/// let (sigma, epsilon) = lj_ab_to_sigma_epsilon(1.043080230e6, 6.75612248e2);
/// assert!((epsilon - 0.1094).abs() < 1e-6);
/// assert!((sigma - 3.399669).abs() < 1e-5);
/// assert_eq!(lj_ab_to_sigma_epsilon(0.0, 1.0), (1.0, 0.0));
/// ```
#[must_use]
pub fn lj_ab_to_sigma_epsilon(a: F, b: F) -> (F, F) {
    if a == 0.0 || b == 0.0 {
        return (1.0, 0.0);
    }
    let r_min = (2.0 * a / b).powf(1.0 / 6.0);
    let epsilon = 0.25 * b * b / a;
    let sigma = (2.0 as F).powf(-1.0 / 6.0) * r_min;
    (sigma, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_coefficient_is_a_non_interacting_type() {
        assert_eq!(lj_ab_to_sigma_epsilon(0.0, 1.0), (1.0, 0.0));
        assert_eq!(lj_ab_to_sigma_epsilon(1.0, 0.0), (1.0, 0.0));
    }

    #[test]
    fn round_trips_the_gaff_c3_row() {
        let (sigma, epsilon) = lj_ab_to_sigma_epsilon(1.043080230e6, 6.75612248e2);
        assert!((epsilon - 0.1094).abs() < 1e-6, "{epsilon}");
        // sigma = 2^(-1/6) * R* * 2, with R* the Amber half-distance 1.9080 Å.
        assert!((sigma - 3.399669).abs() < 1e-5, "{sigma}");
    }

    #[test]
    fn inverts_the_molrs_form() {
        // Build A/B from a known (sigma, epsilon) and recover it.
        let (sigma, epsilon) = (3.4_f64, 0.238_f64);
        let a = 4.0 * epsilon * sigma.powi(12);
        let b = 4.0 * epsilon * sigma.powi(6);
        let (got_sigma, got_epsilon) = lj_ab_to_sigma_epsilon(a, b);
        assert!((got_sigma - sigma).abs() < 1e-12, "{got_sigma}");
        assert!((got_epsilon - epsilon).abs() < 1e-12, "{got_epsilon}");
    }
}
