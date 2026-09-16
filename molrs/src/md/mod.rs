//! In-process molecular-dynamics engine.
//!
//! Neighbour lists: [`crate::spatial::neighbors`] (`NeighborList`,
//! `VerletSkin`). Science here:
//!
//! * [`crate::ff::potential::Potential`] produces energy and forces from flat
//!   coordinates. [`crate::ff::potential::Potentials`] sums the results of its
//!   members, including nonbonded, bonded, and external terms.
//! * [`crate::ff::potential::pair::PairPotential`] defines pair-energy and
//!   pair-force evaluation. [`crate::ff::potential::pair::LJCut`] implements
//!   the `lj/cut` kernel. The integrator supplies current neighbour pairs via
//!   [`crate::ff::potential::Potential::calc_energy_forces_with_pairs`];
//!   potentials do not own or update the skin.
//! * [`crate::md::VelocityVerlet`] / [`crate::md::Langevin`] — constructed with
//!   `(dt, potential, neighbors, mass)` (Langevin also `gamma`, `kbt`,
//!   `seed`). The integrator owns the optional `VerletSkin`, runs its update
//!   policy each force evaluation, and feeds current pairs to the potential.
//!
//! Pair-kernel types are named after the LAMMPS `pair_style` vocabulary:
//! [`crate::ff::potential::pair::LJCut`] ↔ `lj/cut`; other kernels follow the same mapping
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
