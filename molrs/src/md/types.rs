//! Typed array containers for the MD engine.
//!
//! [`ForceOutput`], [`MDState`] and [`MDObservables`] are the only data
//! contract crossing component boundaries (force provider → integrator →
//! runner → hook). Frame topology stays at the composer; the hot step sees
//! these three structs.

use molrs::types::{F, FNx3};

/// Energy + forces from a [`super::integrators::PotentialLike`].
///
/// `energy` is a scalar in amu·Å²/fs². `forces` is `(N, 3)` `= -∂E/∂pos`
/// in the same unit per Å.
#[derive(Clone, Debug)]
pub struct ForceOutput {
    /// Scalar total energy `()` in amu·Å²/fs².
    pub energy: F,
    /// Per-atom forces `(N, 3)`.
    pub forces: FNx3,
}

/// Dynamical state advanced one step by an integrator.
///
/// Carries the force-cache (`forces` / `energy` at `pos`, both from the same
/// force-field evaluation) so the loop does one evaluation per step.
#[derive(Clone, Debug)]
pub struct MDState {
    /// Positions `(N, 3)` in Å.
    pub pos: FNx3,
    /// Velocities `(N, 3)` in Å/fs.
    pub vel: FNx3,
    /// Cached forces `(N, 3)` at `pos`.
    pub forces: FNx3,
    /// Cached scalar energy at `pos`.
    pub energy: F,
}

/// Per-observation thermodynamic snapshot handed to MD hooks.
#[derive(Clone, Debug)]
pub struct MDObservables {
    /// Positions `(N, 3)` in Å.
    pub pos: FNx3,
    /// Velocities `(N, 3)` in Å/fs.
    pub vel: FNx3,
    /// Forces `(N, 3)` at `pos`.
    pub forces: FNx3,
    /// Scalar potential energy.
    pub potential: F,
    /// Scalar kinetic energy.
    pub kinetic: F,
    /// `potential + kinetic`.
    pub total: F,
    /// Instantaneous temperature in kelvin (for the runner's default `k_B`).
    pub temperature: F,
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn forces_field_is_plural() {
        let out = ForceOutput {
            energy: 1.0,
            forces: array![[0.0, 0.0, 0.0]],
        };
        assert_eq!(out.forces.nrows(), 1);
        let state = MDState {
            pos: array![[0.0, 0.0, 0.0]],
            vel: array![[0.0, 0.0, 0.0]],
            forces: array![[1.0, 0.0, 0.0]],
            energy: 0.0,
        };
        assert_eq!(state.forces[[0, 0]], 1.0);
    }

    #[test]
    fn observables_cover_the_hook_contract() {
        let obs = MDObservables {
            pos: array![[0.0, 0.0, 0.0]],
            vel: array![[0.0, 0.0, 0.0]],
            forces: array![[0.0, 0.0, 0.0]],
            potential: 1.0,
            kinetic: 2.0,
            total: 3.0,
            temperature: 300.0,
        };
        assert!((obs.total - (obs.potential + obs.kinetic)).abs() < 1e-15);
    }
}
