//! In-process molecular-dynamics engine.
//!
//! Neighbour lists: [`crate::spatial::neighbors`] (`NeighborList`,
//! `VerletSkin`). Science here:
//!
//! * [`Potential`] / [`Potentials`] — the one force seam (re-exported from
//!   [`crate::ff::potential`]): a potential produces energy and forces from
//!   flat coordinates, and `Potentials` merges the results of its members
//!   (itself a `Potential`, so collections nest). Potentials come in
//!   categories — nonbond, bond, angle, dihedral, improper — as
//!   implementations of the one concept, never as separate seams.
//! * [`PairPotential`] / [`LJCut`] — the internal pair-kernel abstraction
//!   (`pair_energy` / `pair_force` / `pair_eval → (E, F)` per pair, plus the
//!   `eval*` drivers) and the `lj/cut` kernel. [`LJCut`] is md's nonbond
//!   [`Potential`]: the loop feeds it the current neighbour pairs
//!   ([`Potential::set_pairs`]); it never owns or updates the skin.
//! * [`VelocityVerlet`] / [`Langevin`] — constructed with
//!   `(dt, potential, neighbors, mass)` (Langevin also `gamma`, `kbt`,
//!   `seed`). The integrator owns the optional `VerletSkin`, runs its update
//!   policy each force evaluation, and feeds fresh pairs to the potential
//!   after every rebuild.
//!
//! Pair-kernel types are named after the LAMMPS `pair_style` vocabulary:
//! [`LJCut`] ↔ `lj/cut`, and future kernels follow the same mapping
//! (`coul/cut` → `CoulCut`, `lj/cut/coul/cut` → `LJCutCoulCut`, …).
//!
//! No `bind_*` façades. Compose required pieces in the constructor.

pub mod error;
pub mod integrators;
pub mod maxwell;
pub mod types;

pub use crate::ff::potential::pair::{LJCut, PairPotential};
pub use crate::ff::potential::{Potential, Potentials};
pub use error::MdError;
pub use integrators::{Langevin, VelocityVerlet, kinetic_energy, scalar_mass};
pub use maxwell::MaxwellBoltzmann;
pub use types::{ForceOutput, MDObservables, MDState};
