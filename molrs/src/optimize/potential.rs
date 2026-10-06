//! The optimizer over a force-field [`Potential`]: [`LBFGS`].
//!
//! One front door for every potential, including one that rebuilds its own
//! pairs as the atoms move (`ff::potential::soft::SoftPotential`).
//!
//! Gated on `ff` (the potential trait lives there); the [`Optimizer`] trait
//! they implement is always compiled.

use std::sync::Arc;

use super::lbfgs::{Converge, fmax_from_grad, minimize_core};
use super::{OptReport, Optimizer};
use crate::ff::potential::Potential;
use crate::op::types::F;
use crate::store::Frame;
use crate::store::keys::FREE;
use crate::store::schema::block_names::ATOMS;

/// Limited-memory BFGS over a molecule-bound [`Potential`].
///
/// Construct with [`LBFGS::new`] (potential + knobs). Primary call is
/// [`Optimizer::run`] on a [`Frame`]; [`run_coords`](LBFGS::run_coords) relaxes a
/// flat coordinate buffer when a Frame is not available, and the associated
/// [`LBFGS::minimize`] / [`LBFGS::minimize_batch`] work on a borrowed potential.
pub struct LBFGS {
    potential: Arc<dyn Potential>,
    fmax: F,
    max_steps: usize,
    max_step: F,
    memory: usize,
}

impl LBFGS {
    /// Bind `potential` with L-BFGS knobs.
    ///
    /// Defaults match the former config defaults when callers use the values
    /// `fmax = 0.05`, `max_steps = 500`, `max_step = 0.2`, `memory = 8`.
    pub fn new(
        potential: Arc<dyn Potential>,
        fmax: F,
        max_steps: usize,
        max_step: F,
        memory: usize,
    ) -> Self {
        Self {
            potential,
            fmax,
            max_steps,
            max_step,
            memory,
        }
    }

    /// One-shot minimize of flat coordinates under a borrowed potential.
    ///
    /// Use when the potential is not owned as an [`Arc`] (e.g. a temporary
    /// `Potentials` behind a binder borrow). For a storeable optimizer, prefer
    /// [`LBFGS::new`] + [`run_coords`](Self::run_coords) / [`Optimizer::run`].
    pub fn minimize(
        potential: &dyn Potential,
        coords: &mut [F],
        fmax: F,
        max_steps: usize,
        max_step: F,
        memory: usize,
    ) -> Result<OptReport, String> {
        if coords.is_empty() {
            return Ok(OptReport {
                converged: true,
                n_steps: 0,
                final_energy: 0.0,
                final_fmax: 0.0,
            });
        }
        if !coords.len().is_multiple_of(3) {
            return Err(format!(
                "coords length {} is not a multiple of 3 (expected 3·n_atoms)",
                coords.len()
            ));
        }
        let (final_energy, grad, n_steps, converged) = minimize_core(
            coords,
            max_steps,
            Converge::Fmax(fmax),
            max_step,
            memory,
            |c| potential.calc_energy_forces(c),
        );
        Ok(OptReport {
            converged,
            n_steps,
            final_energy,
            final_fmax: fmax_from_grad(&grad),
        })
    }

    /// One-shot homogeneous batch under a borrowed potential.
    #[allow(clippy::too_many_arguments)]
    pub fn minimize_batch(
        potential: &dyn Potential,
        coords: &mut [F],
        n_atoms: usize,
        n_structs: usize,
        fmax: F,
        max_steps: usize,
        max_step: F,
        memory: usize,
    ) -> Result<Vec<OptReport>, String> {
        let stride = n_atoms * 3;
        let expected = n_structs * stride;
        if coords.len() != expected {
            return Err(format!(
                "coords length {} != n_structs ({}) · n_atoms ({}) · 3 = {}",
                coords.len(),
                n_structs,
                n_atoms,
                expected
            ));
        }
        if n_structs == 0 {
            return Ok(Vec::new());
        }
        if stride == 0 {
            return Err(format!(
                "n_atoms must be > 0 for a batch of {n_structs} structures"
            ));
        }
        let mut reports = Vec::with_capacity(n_structs);
        for block in coords.chunks_mut(stride) {
            reports.push(Self::minimize(
                potential, block, fmax, max_steps, max_step, memory,
            )?);
        }
        Ok(reports)
    }

    /// Relax flat `3·n_atoms` coordinates in place.
    ///
    /// # Errors
    /// Returns `Err` if `coords.len()` is not a multiple of three.
    pub fn run_coords(&self, coords: &mut [F]) -> Result<OptReport, String> {
        if coords.is_empty() {
            return Ok(OptReport {
                converged: true,
                n_steps: 0,
                final_energy: 0.0,
                final_fmax: 0.0,
            });
        }
        if !coords.len().is_multiple_of(3) {
            return Err(format!(
                "coords length {} is not a multiple of 3 (expected 3·n_atoms)",
                coords.len()
            ));
        }
        Ok(self.run_one(coords))
    }

    fn run_one(&self, coords: &mut [F]) -> OptReport {
        let (final_energy, grad, n_steps, converged) = minimize_core(
            coords,
            self.max_steps,
            Converge::Fmax(self.fmax),
            self.max_step,
            self.memory,
            |c| self.potential.calc_energy_forces(c),
        );
        OptReport {
            converged,
            n_steps,
            final_energy,
            final_fmax: fmax_from_grad(&grad),
        }
    }

    /// Minimize free DOFs only, evaluating the potential on the full system.
    fn run_masked(&self, full: &mut [F], free: &[bool]) -> Result<OptReport, String> {
        let n = free.len();
        if full.len() != n * 3 {
            return Err(format!(
                "free mask length {n} does not match coords atom count {}",
                full.len() / 3
            ));
        }
        let free_idx: Vec<usize> = free
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f { Some(i) } else { None })
            .collect();
        if free_idx.is_empty() {
            let (e, forces) = self.potential.calc_energy_forces(full);
            let grad: Vec<F> = forces.iter().map(|f| -f).collect();
            return Ok(OptReport {
                converged: true,
                n_steps: 0,
                final_energy: e,
                final_fmax: fmax_from_grad(&grad),
            });
        }
        if free_idx.len() == n {
            return self.run_coords(full);
        }

        let mut x_free = Vec::with_capacity(free_idx.len() * 3);
        for &i in &free_idx {
            x_free.extend_from_slice(&full[3 * i..3 * i + 3]);
        }

        let pot = Arc::clone(&self.potential);
        let free_idx_c = free_idx.clone();
        let mut full_buf = full.to_vec();

        let (final_energy, grad_free, n_steps, converged) = minimize_core(
            &mut x_free,
            self.max_steps,
            Converge::Fmax(self.fmax),
            self.max_step,
            self.memory,
            |xf| {
                for (k, &i) in free_idx_c.iter().enumerate() {
                    full_buf[3 * i] = xf[3 * k];
                    full_buf[3 * i + 1] = xf[3 * k + 1];
                    full_buf[3 * i + 2] = xf[3 * k + 2];
                }
                let (e, forces) = pot.calc_energy_forces(&full_buf);
                let mut f_free = vec![0.0; xf.len()];
                for (k, &i) in free_idx_c.iter().enumerate() {
                    f_free[3 * k] = forces[3 * i];
                    f_free[3 * k + 1] = forces[3 * i + 1];
                    f_free[3 * k + 2] = forces[3 * i + 2];
                }
                (e, f_free)
            },
        );

        for (k, &i) in free_idx.iter().enumerate() {
            full[3 * i] = x_free[3 * k];
            full[3 * i + 1] = x_free[3 * k + 1];
            full[3 * i + 2] = x_free[3 * k + 2];
        }

        Ok(OptReport {
            converged,
            n_steps,
            final_energy,
            final_fmax: fmax_from_grad(&grad_free),
        })
    }
}

impl Optimizer for LBFGS {
    fn run(&mut self, frame: &mut Frame) -> Result<OptReport, String> {
        let mut xyz = frame.coords().map_err(|e| e.to_string())?;
        let free = frame_free_mask(frame, xyz.nrows())?;
        // `Frame::coords` builds a fresh row-major N×3 array, so its buffer is
        // the flat `[x0, y0, z0, x1, …]` the minimizer works on.
        let coords = xyz
            .as_slice_mut()
            .expect("Frame::coords returns a standard-layout array");
        let report = match free {
            None => self.run_coords(coords)?,
            Some(mask) => self.run_masked(coords, &mask)?,
        };
        frame.set_coords(xyz.view()).map_err(|e| e.to_string())?;
        Ok(report)
    }
}

// ── Frame helpers ────────────────────────────────────────────────────────────

/// `None` ⇒ all free. `Some(mask)` length = n_atoms.
fn frame_free_mask(frame: &Frame, n_atoms: usize) -> Result<Option<Vec<bool>>, String> {
    let Some(atoms) = frame.get(ATOMS) else {
        return Ok(None);
    };
    let Some(col) = atoms.get(FREE).and_then(|c| c.as_bool()) else {
        return Ok(None);
    };
    if col.len() != n_atoms {
        return Err(format!(
            "atoms.free length {} != n_atoms {n_atoms}",
            col.len()
        ));
    }
    Ok(Some(col.iter().copied().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::set_free_mask;
    use crate::store::Block;
    use ndarray::Array1;
    use std::sync::Arc;

    struct HarmonicBond {
        k: F,
        r0: F,
    }

    impl Potential for HarmonicBond {
        fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
            let d = [
                coords[3] - coords[0],
                coords[4] - coords[1],
                coords[5] - coords[2],
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let e = 0.5 * self.k * (r - self.r0) * (r - self.r0);
            let mut f = vec![0.0; 6];
            if r > 1e-12 {
                let coeff = self.k * (r - self.r0) / r;
                for i in 0..3 {
                    let fi = coeff * d[i];
                    f[i] = fi;
                    f[3 + i] = -fi;
                }
            }
            (e, f)
        }
    }

    fn opt(pot: impl Potential + 'static) -> LBFGS {
        LBFGS::new(Arc::new(pot), 0.05, 500, 0.2, 8)
    }

    fn frame_from_coords(coords: &[F]) -> Frame {
        let n = coords.len() / 3;
        let mut atoms = Block::new();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            x.push(coords[3 * i]);
            y.push(coords[3 * i + 1]);
            z.push(coords[3 * i + 2]);
        }
        atoms.insert("x", Array1::from_vec(x).into_dyn()).unwrap();
        atoms.insert("y", Array1::from_vec(y).into_dyn()).unwrap();
        atoms.insert("z", Array1::from_vec(z).into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    #[test]
    fn relaxes_harmonic_bond_to_equilibrium() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let report = opt(pot).run_coords(&mut coords).unwrap();
        assert!(report.converged, "should converge: {report:?}");
        let r = coords[3] - coords[0];
        assert!((r.abs() - 1.0).abs() < 1e-6, "bond length got {r}");
        assert!(report.final_energy < 1e-9);
        assert!(report.final_fmax <= 0.05);
    }

    #[test]
    fn run_frame_updates_xyz() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut frame = frame_from_coords(&[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]);
        let report = opt(pot).run(&mut frame).unwrap();
        assert!(report.converged);
        let x = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.as_float())
            .unwrap();
        assert!((x[[1]] - x[[0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn free_mask_freezes_fixed_atom() {
        // Atom 0 fixed at origin; atom 1 free. Bond wants r0=1.
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut frame = frame_from_coords(&[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]);
        set_free_mask(&mut frame, &[false, true]).unwrap();
        opt(pot).run(&mut frame).unwrap();
        let x = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.as_float())
            .unwrap();
        assert!(x[[0]].abs() < 1e-9, "fixed atom moved: {}", x[[0]]);
        assert!((x[[1]] - 1.0).abs() < 1e-5, "free atom should sit at r0");
    }

    #[test]
    fn fmax_convergence_semantics() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let r = LBFGS::new(Arc::new(pot), 0.05, 1, 0.2, 8)
            .run_coords(&mut coords)
            .unwrap();
        assert!(!r.converged);
        assert_eq!(r.n_steps, 1);
    }

    #[test]
    fn idempotent_at_minimum() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let r = opt(pot).run_coords(&mut coords).unwrap();
        assert!(r.converged);
        assert!(r.n_steps <= 1);
    }

    #[test]
    fn single_atom_converges_immediately() {
        struct Free;
        impl Potential for Free {
            fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
                (0.0, vec![0.0; coords.len()])
            }
        }
        let mut coords = vec![0.3, -0.2, 0.1];
        let r = opt(Free).run_coords(&mut coords).unwrap();
        assert!(r.converged);
        assert!(r.n_steps <= 1);
    }

    #[test]
    fn rejects_non_multiple_of_three() {
        struct Free;
        impl Potential for Free {
            fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
                (0.0, vec![0.0; coords.len()])
            }
        }
        let mut coords = vec![0.0, 0.0, 0.0, 1.0];
        assert!(opt(Free).run_coords(&mut coords).is_err());
    }

    #[test]
    fn empty_coords_is_converged_noop() {
        struct Free;
        impl Potential for Free {
            fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
                (0.0, vec![0.0; coords.len()])
            }
        }
        let mut coords: Vec<F> = vec![];
        let r = opt(Free).run_coords(&mut coords).unwrap();
        assert!(r.converged);
        assert_eq!(r.n_steps, 0);
    }

    #[test]
    fn trust_region_caps_step() {
        let pot = HarmonicBond { k: 500.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let before = coords.clone();
        LBFGS::new(Arc::new(pot), 0.05, 1, 0.01, 8)
            .run_coords(&mut coords)
            .unwrap();
        for (a, b) in coords.iter().zip(&before) {
            assert!((a - b).abs() <= 0.01 + 1e-12);
        }
    }

    #[test]
    fn batch_equals_serial() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let single_start = vec![0.0, 0.0, 0.0, 1.4, 0.0, 0.0];
        let mut single = single_start.clone();
        let single_report = opt(HarmonicBond { k: 100.0, r0: 1.0 })
            .run_coords(&mut single)
            .unwrap();

        let b = 4;
        let mut batch: Vec<F> = Vec::new();
        for _ in 0..b {
            batch.extend_from_slice(&single_start);
        }
        let reports = LBFGS::minimize_batch(&pot, &mut batch, 2, b, 0.05, 500, 0.2, 8).unwrap();
        assert_eq!(reports.len(), b);
        for (i, rep) in reports.iter().enumerate() {
            assert!((rep.final_energy - single_report.final_energy).abs() < 1e-10);
            let block = &batch[i * 6..i * 6 + 6];
            for (a, s) in block.iter().zip(&single) {
                assert!((a - s).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn batch_rejects_size_mismatch() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0; 6 * 3 + 1];
        assert!(LBFGS::minimize_batch(&pot, &mut coords, 2, 3, 0.05, 500, 0.2, 8).is_err());
    }

    #[test]
    fn batch_zero_structs_is_empty() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords: Vec<F> = vec![];
        let reports = LBFGS::minimize_batch(&pot, &mut coords, 2, 0, 0.05, 500, 0.2, 8).unwrap();
        assert!(reports.is_empty());
    }

    #[test]
    fn batch_zero_atoms_errors_not_panics() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords: Vec<F> = vec![];
        assert!(LBFGS::minimize_batch(&pot, &mut coords, 0, 3, 0.05, 500, 0.2, 8).is_err());
    }

    /// The soft packing potential minimizes through the one front door: two
    /// overlapping atoms are pushed to the soft core's edge, and a pair that
    /// was out of range when the run started still counts once it comes in.
    #[test]
    fn lbfgs_relaxes_the_soft_potential_that_rebuilds_its_own_pairs() {
        use crate::ff::potential::soft::SoftSpec;
        let mut frame = frame_from_coords(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 30.0, 0.0, 0.0]);
        let pot = SoftSpec::from_frame(&frame).potential(None);
        let report = LBFGS::new(Arc::new(pot), 1e-4, 500, 0.2, 8)
            .run(&mut frame)
            .unwrap();
        assert!(report.converged, "{report:?}");
        assert!(report.final_energy < 1e-6, "{report:?}");
        let x = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.as_float())
            .unwrap();
        assert!(x[[1]] - x[[0]] > 2.59, "pushed apart to sigma: {x:?}");
    }
}
