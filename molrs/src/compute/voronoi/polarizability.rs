//! Finite-field molecular polarizability from Voronoi dipoles.
//!
//! Ported from TRAVIS's polarizability workflow (`src/dpol.cpp`): three external
//! field directions, each a ±E pair of Voronoi-integrated dipole sets, combined
//! by central difference. The `2.0` denominator here is the same central-finite
//! difference TRAVIS uses (`dpol.cpp`, the `... / estrength ... * 2.0 ...`
//! prefactor, stripped of TRAVIS's Debye/SI unit conversions — molrs stays in
//! `e·Å` / `e·Å²·V⁻¹`-style natural units).
//!
//! # Definition
//!
//! `α_ij = ∂μ_i/∂E_j ≈ (μ_i(+E ê_j) − μ_i(−E ê_j)) / (2E)`.
//!
//! One call handles **one** field direction `j` and returns the column
//! `∂μ/∂E_j` (a 3-vector per molecule); assemble the full 3×3 tensor from three
//! orthogonal field runs (see the test). For a linear response
//! `μ(E) = μ₀ + α E`, the central difference is exact.

use molrs::types::F;
use ndarray::Array2;

use super::integrate::MolecularMoments;
use crate::compute::error::ComputeError;

/// Per-molecule polarizability column `∂μ/∂E_j` (e·Å per field unit), shape
/// `(n_mol, 3)`, from the `+E` and `−E` dipole sets at field magnitude `field`.
///
/// `moments_zero` (the field-off set) is accepted for API symmetry with
/// TRAVIS's three-point workflow and to validate molecule alignment; the
/// central difference itself uses only `plus`/`minus`.
pub fn polarizability_finite_field(
    moments_zero: &MolecularMoments,
    plus: &MolecularMoments,
    minus: &MolecularMoments,
    field: F,
) -> Result<Array2<F>, ComputeError> {
    if field == 0.0 || !field.is_finite() {
        return Err(ComputeError::OutOfRange {
            field: "polarizability::field",
            value: field.to_string(),
        });
    }
    let n = moments_zero.charges.len();
    if plus.charges.len() != n || minus.charges.len() != n {
        return Err(ComputeError::DimensionMismatch {
            expected: n,
            got: plus.charges.len().min(minus.charges.len()),
            what: "moment set molecule count",
        });
    }
    let mut col = Array2::<F>::zeros((n, 3));
    let inv = 1.0 / (2.0 * field);
    for m in 0..n {
        for c in 0..3 {
            col[[m, c]] = (plus.dipoles[[m, c]] - minus.dipoles[[m, c]]) * inv;
        }
    }
    Ok(col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn moments(dipoles: Array2<F>) -> MolecularMoments {
        let n = dipoles.nrows();
        MolecularMoments {
            charges: vec![0.0; n],
            dipoles,
            references: Array2::zeros((n, 3)),
        }
    }

    #[test]
    fn recovers_known_alpha_tensor() {
        // Known α (one molecule), anisotropic + with off-diagonal coupling.
        let alpha = [[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]];
        let mu0 = [0.4, -0.2, 0.05];
        let e = 1e-3;
        let zero = moments(Array2::from_shape_vec((1, 3), mu0.to_vec()).unwrap());

        let mut recovered = [[0.0; 3]; 3];
        for j in 0..3 {
            // μ(±E ê_j) = μ0 ± E α_{:,j}
            let plus_v: Vec<F> = (0..3).map(|i| mu0[i] + e * alpha[i][j]).collect();
            let minus_v: Vec<F> = (0..3).map(|i| mu0[i] - e * alpha[i][j]).collect();
            let plus = moments(Array2::from_shape_vec((1, 3), plus_v).unwrap());
            let minus = moments(Array2::from_shape_vec((1, 3), minus_v).unwrap());
            let col = polarizability_finite_field(&zero, &plus, &minus, e).unwrap();
            for i in 0..3 {
                recovered[i][j] = col[[0, i]];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recovered[i][j] - alpha[i][j]).abs() < 1e-9,
                    "α[{i}][{j}] = {} expected {}",
                    recovered[i][j],
                    alpha[i][j]
                );
            }
        }
    }

    #[test]
    fn zero_field_is_error() {
        let z = moments(Array2::zeros((1, 3)));
        assert!(polarizability_finite_field(&z, &z, &z, 0.0).is_err());
    }
}
