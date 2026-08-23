//! Maxwell-Boltzmann velocity distribution. Not a hook — draw only.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use molrs::types::{F, FNx3};

use super::error::MdError;
use super::units::kb_md;

/// Maxwell-Boltzmann velocity distribution at a target temperature.
///
/// Attach it from the runner / a hook; this type only draws:
///
/// ```ignore
/// let mb = MaxwellBoltzmann::new(300.0, 0)?;
/// let vel = mb.velocities(pos.view(), mass.view())?;
/// ```
pub struct MaxwellBoltzmann {
    /// Target temperature in kelvin.
    pub temperature: F,
    /// RNG seed.
    pub seed: u64,
    /// Subtract the centre-of-mass velocity after the draw.
    pub remove_com: bool,
}

impl MaxwellBoltzmann {
    /// Draw at `temperature` kelvin. `remove_com` defaults to `true`.
    pub fn new(temperature: F, seed: u64) -> Result<Self, MdError> {
        Self::with_com(temperature, seed, true)
    }

    /// Draw at `temperature` kelvin, optionally leaving the COM velocity.
    pub fn with_com(temperature: F, seed: u64, remove_com: bool) -> Result<Self, MdError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(MdError::Invalid(
                "MaxwellBoltzmann temperature must be strictly positive".into(),
            ));
        }
        Ok(Self {
            temperature,
            seed,
            remove_com,
        })
    }

    /// Draw `(N, 3)` velocities in Å/fs at [`Self::temperature`].
    pub fn velocities(
        &self,
        pos: ArrayView2<'_, F>,
        mass: ArrayView1<'_, F>,
    ) -> Result<FNx3, MdError> {
        let n = pos.nrows();
        if pos.ncols() != 3 {
            return Err(MdError::Invalid(format!(
                "pos must have shape (N, 3), got {:?}",
                pos.shape()
            )));
        }
        if mass.len() != n {
            return Err(MdError::Invalid(format!(
                "atoms.mass length {} disagrees with n_atoms={n}",
                mass.len()
            )));
        }
        if mass.iter().any(|&m| !m.is_finite() || m <= 0.0) {
            return Err(MdError::Invalid("mass must be strictly positive".into()));
        }
        let kb = kb_md()?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut vel = Array2::<F>::zeros((n, 3));
        Zip::from(vel.rows_mut()).and(mass).for_each(|mut row, &m| {
            let scale = (kb * self.temperature / m).sqrt();
            for x in row.iter_mut() {
                *x = standard_normal(&mut rng) * scale;
            }
        });
        if self.remove_com {
            remove_com_velocity(&mut vel, mass);
        }
        Ok(vel)
    }
}

fn standard_normal(rng: &mut StdRng) -> F {
    let u1 = rng.random::<F>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<F>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn remove_com_velocity(vel: &mut Array2<F>, mass: ArrayView1<'_, F>) {
    let mut p = [0.0; 3];
    let mut mtot = 0.0;
    Zip::from(vel.rows()).and(mass).for_each(|v, &m| {
        mtot += m;
        p[0] += m * v[0];
        p[1] += m * v[1];
        p[2] += m * v[2];
    });
    if mtot == 0.0 {
        return;
    }
    let com = [p[0] / mtot, p[1] / mtot, p[2] / mtot];
    Zip::from(vel.rows_mut()).for_each(|mut v| {
        v[0] -= com[0];
        v[1] -= com[1];
        v[2] -= com[2];
    });
}

/// Total mass-weighted COM velocity of `vel`.
pub fn com_velocity(vel: ArrayView2<'_, F>, mass: ArrayView1<'_, F>) -> Array1<F> {
    let mut p = [0.0; 3];
    let mut mtot = 0.0;
    Zip::from(vel.rows()).and(mass).for_each(|v, &m| {
        mtot += m;
        p[0] += m * v[0];
        p[1] += m * v[1];
        p[2] += m * v[2];
    });
    if mtot == 0.0 {
        return Array1::zeros(3);
    }
    Array1::from_vec(vec![p[0] / mtot, p[1] / mtot, p[2] / mtot])
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::*;

    #[test]
    fn same_seed_is_reproducible() {
        let pos = Array2::zeros((8, 3));
        let mass = Array1::from_elem(8, 1.0);
        let a = MaxwellBoltzmann::new(300.0, 7)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap();
        let b = MaxwellBoltzmann::new(300.0, 7)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn remove_com_leaves_zero_com() {
        let pos = Array2::zeros((6, 3));
        let mass = Array1::from_elem(6, 2.0);
        let vel = MaxwellBoltzmann::new(200.0, 1)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap();
        let com = com_velocity(vel.view(), mass.view());
        assert!(com.iter().all(|c| c.abs() < 1e-12));
    }

    #[test]
    fn rejects_nonpositive_temperature() {
        assert!(MaxwellBoltzmann::new(0.0, 0).is_err());
    }

    #[test]
    fn mass_length_must_match() {
        let pos = Array2::zeros((4, 3));
        let mass = Array1::from_elem(3, 1.0);
        let err = MaxwellBoltzmann::new(100.0, 0)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap_err();
        assert!(format!("{err}").contains("n_atoms"));
    }
}
