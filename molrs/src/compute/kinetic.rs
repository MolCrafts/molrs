//! Kinetic observables of one configuration: kinetic energy, kinetic
//! temperature and the centre-of-mass velocity.

use ndarray::{Array1, ArrayView1, ArrayView2, Zip};

use crate::compute::ComputeError;
use crate::op::F;

fn check_rows(mass: ArrayView1<'_, F>, vel: ArrayView2<'_, F>) -> Result<(), ComputeError> {
    if vel.ncols() != 3 {
        return Err(ComputeError::BadShape {
            expected: "velocities of shape (n_atoms, 3)".into(),
            got: format!("{:?}", vel.shape()),
        });
    }
    if mass.len() != vel.nrows() {
        return Err(ComputeError::DimensionMismatch {
            expected: vel.nrows(),
            got: mass.len(),
            what: "mass length against the velocity rows",
        });
    }
    Ok(())
}

/// Kinetic energy `K = ½ Σᵢ mᵢ |vᵢ|²` of the `(n_atoms, 3)` velocities `vel`
/// with per-atom masses `mass`.
///
/// An instantaneous thermodynamic reading of an MD state — what a `thermo`
/// line prints — and a pure function of the arrays handed in. Units are the
/// caller's: with masses in g/mol and velocities in Å/fs the energy is in
/// g·Å²/(mol·fs²), and [`kinetic_temperature`] comes out in K only when `kb`
/// is Boltzmann's constant in that same energy unit.
///
/// # Errors
///
/// [`ComputeError::BadShape`] when `vel` is not `(n, 3)`;
/// [`ComputeError::DimensionMismatch`] when `mass` does not have one entry per
/// row of `vel`.
pub fn kinetic_energy(mass: ArrayView1<'_, F>, vel: ArrayView2<'_, F>) -> Result<F, ComputeError> {
    check_rows(mass, vel)?;
    let mut twice = 0.0;
    Zip::from(mass).and(vel.rows()).for_each(|&m, v| {
        twice += m * v.dot(&v);
    });
    Ok(0.5 * twice)
}

/// Kinetic temperature `T = 2K / (n_dof · k_B)` by equipartition: `n_dof` is
/// the number of unconstrained degrees of freedom (`3N` less any removed by
/// constraints or a fixed centre of mass) and `kb` Boltzmann's constant in the
/// unit of `kinetic_energy`.
///
/// # Errors
///
/// [`ComputeError::OutOfRange`] when `n_dof` is zero or `kb` is not a positive
/// finite number.
pub fn kinetic_temperature(kinetic_energy: F, n_dof: usize, kb: F) -> Result<F, ComputeError> {
    if n_dof == 0 {
        return Err(ComputeError::OutOfRange {
            field: "n_dof",
            value: n_dof.to_string(),
        });
    }
    if !(kb.is_finite() && kb > 0.0) {
        return Err(ComputeError::OutOfRange {
            field: "kb",
            value: kb.to_string(),
        });
    }
    Ok(2.0 * kinetic_energy / (n_dof as F * kb))
}

/// The mass-weighted centre-of-mass velocity `Σᵢ mᵢ vᵢ / Σᵢ mᵢ` of the
/// `(n_atoms, 3)` velocities `vel`; zero when the total mass is zero.
///
/// # Errors
///
/// As [`kinetic_energy`].
pub fn center_of_mass_velocity(
    mass: ArrayView1<'_, F>,
    vel: ArrayView2<'_, F>,
) -> Result<Array1<F>, ComputeError> {
    check_rows(mass, vel)?;
    let mut p = [0.0; 3];
    let mut total = 0.0;
    Zip::from(vel.rows()).and(mass).for_each(|v, &m| {
        total += m;
        p[0] += m * v[0];
        p[1] += m * v[1];
        p[2] += m * v[2];
    });
    if total == 0.0 {
        return Ok(Array1::zeros(3));
    }
    Ok(Array1::from_vec(vec![
        p[0] / total,
        p[1] / total,
        p[2] / total,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn kinetic_energy_is_half_m_v_squared() {
        let mass = array![2.0, 1.0];
        let vel = array![[1.0, 0.0, 0.0], [0.0, 2.0, 2.0]];
        // ½(2·1 + 1·8) = 5
        assert_eq!(kinetic_energy(mass.view(), vel.view()).unwrap(), 5.0);
        assert!(kinetic_energy(array![1.0].view(), vel.view()).is_err());
    }

    #[test]
    fn temperature_is_equipartition() {
        // 2K/(n_dof kb) with K = 3, n_dof = 6, kb = 0.5 → 2
        assert_eq!(kinetic_temperature(3.0, 6, 0.5).unwrap(), 2.0);
        assert!(kinetic_temperature(3.0, 0, 0.5).is_err());
        assert!(kinetic_temperature(3.0, 6, 0.0).is_err());
    }

    #[test]
    fn center_of_mass_velocity_is_mass_weighted() {
        let mass = array![3.0, 1.0];
        let vel = array![[1.0, 0.0, 0.0], [-3.0, 4.0, 0.0]];
        let v = center_of_mass_velocity(mass.view(), vel.view()).unwrap();
        assert_eq!(v.to_vec(), vec![0.0, 1.0, 0.0]);
        let zero = center_of_mass_velocity(array![0.0, 0.0].view(), vel.view()).unwrap();
        assert_eq!(zero.to_vec(), vec![0.0; 3]);
    }
}
