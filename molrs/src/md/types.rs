//! Typed array containers for the MD engine.
//!
//! [`ForceOutput`], [`MDState`] and [`MDObservables`] are the only data
//! contract crossing component boundaries (potential → integrator →
//! runner → hook). Frame topology stays at the composer; the hot step sees
//! these three structs.

use ndarray::Array2;

use ndarray::Array1;

use molrs::spatial::simbox::SimBox;
use molrs::store::frame::Frame;
use molrs::store::keys;
use molrs::types::{F, FNx3, I};

use super::error::MdError;
use molrs::math::Virial;

/// Energy + forces from one integrator force evaluation.
///
/// `energy` is a scalar in amu·Å²/fs². `forces` is `(N, 3)` `= -∂E/∂pos`
/// in the same unit per Å.
#[derive(Clone, Debug)]
pub struct ForceOutput {
    /// Scalar total energy `()` in amu·Å²/fs².
    pub energy: F,
    /// Per-atom forces `(N, 3)`.
    pub forces: FNx3,
    /// `Σ f ⊗ r` at `pos`, when the provider tallies one.
    ///
    /// `None` is not zero. A provider that does not tally says so, because a
    /// pressure computed from a fabricated zero is wrong and looks entirely
    /// plausible — the same distinction `Neighbors::disp` draws, for the same
    /// reason.
    pub virial: Option<Virial>,
}

/// Dynamical state advanced one step by an integrator.
///
/// Carries the force-cache (`forces` / `energy` at `pos`, both from the same
/// force-field evaluation) so the loop does one evaluation per step.
///
/// # Canonical coordinates: wrapped, plus image flags
///
/// `pos` is **wrapped** — it stays in the primary cell — and `images` records
/// how many cells each atom has crossed to get there. The continuous
/// trajectory is a derived view, `r^u = pos + H·images`
/// ([`SimBox::unwrap`](molrs::spatial::simbox::SimBox::unwrap)), never a second
/// float array integrated alongside the first: two independently-advanced
/// copies of the same quantity drift apart, and the one that drifts is
/// whichever the reader did not check.
///
/// Physics reads `pos`. History reads the reconstruction: mean-squared
/// displacement, diffusion, and any other quantity that must not see a
/// boundary crossing as a jump.
#[derive(Clone, Debug)]
pub struct MDState {
    /// Wrapped positions `(N, 3)` in Å — inside the primary cell.
    pub pos: FNx3,
    /// Accumulated box crossings `(N, 3)`, one signed count per lattice
    /// vector. `pos + H·images` is the continuous position.
    pub images: Array2<i64>,
    /// Velocities `(N, 3)` in Å/fs.
    pub vel: FNx3,
    /// Cached forces `(N, 3)` at `pos`.
    pub forces: FNx3,
    /// Cached scalar energy at `pos`.
    pub energy: F,
    /// Cached virial at `pos`, from the same evaluation as `forces`.
    ///
    /// It belongs to the force cache under the same invariant: the forces, the
    /// energy and the virial at `pos` all come from one evaluation, or the
    /// pressure and the trajectory describe different configurations.
    pub virial: Option<Virial>,
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
            virial: None,
        };
        assert_eq!(out.forces.nrows(), 1);
        let state = MDState {
            pos: array![[0.0, 0.0, 0.0]],
            images: Array2::zeros((1, 3)),
            vel: array![[0.0, 0.0, 0.0]],
            forces: array![[1.0, 0.0, 0.0]],
            energy: 0.0,
            virial: None,
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

impl MDState {
    /// Write the canonical state into a frame's `atoms` block: wrapped
    /// positions **and** the image flags that go with them.
    ///
    /// The two travel together and this is why the method exists. A wrapped
    /// coordinate on its own has lost the atom's history — how many times it
    /// crossed, and therefore where it has actually been — and a trajectory
    /// written without the flags cannot answer a mean-squared displacement, a
    /// diffusion coefficient, or any other question that reads a path rather
    /// than a configuration. Nothing about such a file looks wrong; the
    /// analysis simply comes back with a number that is too small, because
    /// every crossing was read as the atom teleporting back.
    ///
    /// Writing them one at a time is the mistake this closes off: a caller with
    /// two calls will eventually make one of them.
    ///
    /// The columns are the canonical [`keys::COORDS`] and [`keys::IMAGES`], so
    /// a writer that emits whatever the block holds — the LAMMPS dump writer
    /// does — carries them without being told.
    pub fn write_to(&self, frame: &mut Frame) -> Result<(), MdError> {
        let n = self.pos.nrows();
        if self.images.nrows() != n {
            return Err(MdError::Invalid(format!(
                "state has {n} positions but {} image flags",
                self.images.nrows()
            )));
        }
        let atoms = frame.get_mut("atoms").ok_or_else(|| {
            MdError::Invalid("frame has no \"atoms\" block to write into".to_string())
        })?;
        if let Some(rows) = atoms.nrows()
            && rows != n
        {
            return Err(MdError::Invalid(format!(
                "state has {n} atoms but the frame's atoms block has {rows}"
            )));
        }

        for (axis, key) in keys::COORDS.iter().enumerate() {
            let col: Array1<F> = self.pos.column(axis).to_owned();
            atoms
                .insert(*key, col.into_dyn())
                .map_err(|e| MdError::Invalid(format!("atoms.{key}: {e}")))?;
        }
        for (axis, key) in keys::IMAGES.iter().enumerate() {
            let col: Array1<I> = self.images.column(axis).mapv(|m| m as I);
            atoms
                .insert(*key, col.into_dyn())
                .map_err(|e| MdError::Invalid(format!("atoms.{key}: {e}")))?;
        }
        Ok(())
    }

    /// The continuous positions, `pos + H·images`.
    ///
    /// This is the view history-dependent analysis wants — mean-squared
    /// displacement, diffusion, any measurement of a path. Force evaluation
    /// must not use it: the numbers grow without bound, and a potential handed
    /// them is being asked to subtract two large coordinates to recover a small
    /// separation.
    pub fn unwrapped(&self, bx: &SimBox) -> FNx3 {
        bx.unwrap(self.pos.view(), self.images.view())
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use molrs::store::block::Block;
    use ndarray::{Array2, array};

    fn frame_with(n: usize) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                "id",
                Array1::from((0..n as u64).collect::<Vec<_>>()).into_dyn(),
            )
            .unwrap();
        atoms
            .insert("x", Array1::from(vec![0.0_f64; n]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from(vec![0.0_f64; n]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from(vec![0.0_f64; n]).into_dyn())
            .unwrap();
        let mut f = Frame::new();
        f.insert("atoms", atoms);
        f
    }

    /// Positions and image flags reach the frame together, under the canonical
    /// keys, so a writer that emits what the block holds carries both.
    #[test]
    fn the_state_writes_its_flags_beside_its_coordinates() {
        let state = MDState {
            pos: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            images: array![[1_i64, 0, -2], [0, 3, 0]],
            vel: Array2::zeros((2, 3)),
            forces: Array2::zeros((2, 3)),
            energy: 0.0,
            virial: None,
        };
        let mut frame = frame_with(2);
        state.write_to(&mut frame).unwrap();

        let atoms = frame.get("atoms").unwrap();
        for (axis, key) in keys::COORDS.iter().enumerate() {
            let col = atoms.get_float(key).unwrap();
            for i in 0..2 {
                assert_eq!(col[[i]], state.pos[[i, axis]], "{key}[{i}]");
            }
        }
        for (axis, key) in keys::IMAGES.iter().enumerate() {
            let col = atoms
                .get_int(key)
                .unwrap_or_else(|| panic!("atoms block is missing {key}"));
            for i in 0..2 {
                assert_eq!(col[[i]] as i64, state.images[[i, axis]], "{key}[{i}]");
            }
        }
    }

    /// The point of persisting the flags: the continuous trajectory survives a
    /// round trip through a frame, and a file written without them would not
    /// fail — it would quietly answer a displacement question with the wrong
    /// number.
    #[test]
    fn a_trajectory_written_with_flags_can_be_unwrapped_again() {
        let bx = SimBox::cube(10.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // An atom that has crossed +x three times: stored at 2.0, really at 32.
        let state = MDState {
            pos: array![[2.0, 5.0, 5.0]],
            images: array![[3_i64, 0, 0]],
            vel: Array2::zeros((1, 3)),
            forces: Array2::zeros((1, 3)),
            energy: 0.0,
            virial: None,
        };
        assert!((state.unwrapped(&bx)[[0, 0]] - 32.0).abs() < 1e-12);

        let mut frame = frame_with(1);
        state.write_to(&mut frame).unwrap();

        // Read it back the way an analysis would, from the frame alone.
        let atoms = frame.get("atoms").unwrap();
        let x = atoms.get_float("x").unwrap()[[0]];
        let ix = atoms.get_int("ix").unwrap()[[0]] as F;
        assert!(
            (x + ix * 10.0 - 32.0).abs() < 1e-12,
            "reconstructed {} from the frame, expected 32",
            x + ix * 10.0
        );

        // Without the flags the same frame says 2.0 — off by three cells, and
        // nothing about it looks wrong.
        assert!((x - 2.0).abs() < 1e-12);
    }

    /// A state and a frame that disagree about how many atoms they have is a
    /// caller error, and saying so beats writing a column of the wrong length.
    #[test]
    fn a_size_mismatch_is_refused() {
        let state = MDState {
            pos: array![[0.0, 0.0, 0.0]],
            images: Array2::zeros((1, 3)),
            vel: Array2::zeros((1, 3)),
            forces: Array2::zeros((1, 3)),
            energy: 0.0,
            virial: None,
        };
        let mut frame = frame_with(3);
        let err = state.write_to(&mut frame).unwrap_err();
        assert!(format!("{err}").contains("1 atoms"), "{err}");
    }
}
