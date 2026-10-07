//! Geometry optimization of a [`Frame`].
//!
//! The optimizer contract lives here and is always compiled: the
//! [`Optimizer`] trait, its [`OptReport`] and the `atoms.free` mask
//! ([`set_free_mask`]). A purely geometric optimizer (e.g. a packer's
//! torsion Monte Carlo) implements [`Optimizer`] without enabling `ff`. The
//! force-field-agnostic L-BFGS engine behind `LBFGS` is crate-internal; the
//! ETKDG conformer stages drive it on their own distance-geometry energies.
//!
//! The optimizer that minimizes a force-field potential
//! (`ff::potential::Potential`), `LBFGS`, is gated on `ff`. It is the one
//! front door for every potential — a potential that rebuilds its pairs as
//! the atoms move (`ff::potential::soft::SoftPotential`) does so itself.
//! The dependency points one way: `optimize` consumes `ff`, never the
//! reverse.

#[cfg(feature = "ff")]
mod lbfgs;
#[cfg(feature = "ff")]
mod potential;

#[cfg(feature = "ff")]
pub use potential::LBFGS;

use crate::core::Frame;
use crate::core::keys::FREE;
use crate::core::schema::block_names::ATOMS;
use crate::op::types::F;
use ndarray::Array1;

#[cfg(feature = "conformer")]
pub(crate) use lbfgs::{MinResult, minimize_lbfgs_rms};

/// Outcome of a single minimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptReport {
    /// Whether `fmax` convergence was reached within `max_steps`.
    pub converged: bool,
    /// Number of outer L-BFGS iterations performed.
    pub n_steps: usize,
    /// Potential energy at the returned point (kcal/mol).
    pub final_energy: F,
    /// Maximum per-atom force magnitude at the returned point (kcal/mol/Å).
    pub final_fmax: F,
}

/// Geometry optimizer: minimize a [`Frame`] in place.
///
/// Optional free mask: bool column `atoms.free` (missing ⇒ every atom free).
/// Fixed atoms stay in the potential evaluation but are not optimizable DOFs.
pub trait Optimizer: Send + Sync {
    /// Relax `frame` in place. Coordinates are `atoms.{x,y,z}`.
    fn run(&mut self, frame: &mut Frame) -> Result<OptReport, String>;
}

/// Ensure `atoms.free` exists with the given mask (helper for callers assembling Frames).
pub fn set_free_mask(frame: &mut Frame, free: &[bool]) -> Result<(), String> {
    let atoms = frame
        .get_mut(ATOMS)
        .ok_or_else(|| "Frame has no atoms block".to_string())?;
    let n = atoms.nrows().unwrap_or(0);
    if free.len() != n {
        return Err(format!(
            "free mask length {} != atoms nrows {n}",
            free.len()
        ));
    }
    atoms
        .insert(FREE, Array1::from_vec(free.to_vec()).into_dyn())
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A purely geometric optimizer: translates the atoms so their centroid
    /// sits at the origin. Needs no force field, so it builds without `ff`.
    struct Center;

    impl Optimizer for Center {
        fn run(&mut self, frame: &mut Frame) -> Result<OptReport, String> {
            let mut xyz = frame.coords().map_err(|e| e.to_string())?;
            let centroid = xyz.mean_axis(ndarray::Axis(0)).unwrap_or_default();
            xyz -= &centroid;
            frame.set_coords(xyz.view()).map_err(|e| e.to_string())?;
            Ok(OptReport {
                converged: true,
                n_steps: 1,
                final_energy: 0.0,
                final_fmax: 0.0,
            })
        }
    }

    #[test]
    fn geometric_optimizer_runs_on_a_frame_without_ff() {
        let mut frame = Frame::new();
        frame
            .set_coords(array![[1.0 as F, 0.0, 0.0], [3.0, 2.0, 0.0]].view())
            .unwrap();
        let report = Center.run(&mut frame).unwrap();
        assert!(report.converged);
        assert_eq!(
            frame.coords().unwrap(),
            array![[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]
        );
    }

    #[test]
    fn set_free_mask_rejects_a_length_mismatch() {
        let mut frame = Frame::new();
        frame
            .set_coords(array![[0.0 as F, 0.0, 0.0], [1.0, 0.0, 0.0]].view())
            .unwrap();
        assert!(set_free_mask(&mut frame, &[true]).is_err());
        set_free_mask(&mut frame, &[false, true]).unwrap();
        let free = frame
            .get(ATOMS)
            .unwrap()
            .get(FREE)
            .and_then(|c| c.as_bool())
            .unwrap();
        assert_eq!(free.iter().copied().collect::<Vec<_>>(), vec![false, true]);
    }
}
