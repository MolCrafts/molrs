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
//! * [`crate::md::ForceProvider`] is the force-field seam: the potential, the
//!   neighbour bookkeeping and the periodic régime all live behind it, and the
//!   integrator sees only energy, forces and virial. [`crate::md::Direct`],
//!   [`crate::md::MicPairs`] and [`crate::md::GhostPairs`] are the three that
//!   ship; a new way to make a force is a new implementor, not a new variant.
//! * [`crate::md::VelocityVerlet`] / [`crate::md::Langevin`] — constructed with
//!   `(dt, forces, mass, simbox)` (Langevin also `gamma`, `kbt`, `seed`). The
//!   provider owns the neighbour state, so the integrator neither holds a skin
//!   nor knows whether one exists.
//!
//! Pair-kernel types are named after the LAMMPS `pair_style` vocabulary:
//! [`crate::ff::potential::pair::LJCut`] ↔ `lj/cut`; other kernels follow the same mapping
//! (`coul/cut` → `CoulCut`, `lj/cut/coul/cut` → `LJCutCoulCut`, …).
//!
//! No `bind_*` façades. Compose required pieces in the constructor.

pub mod error;
pub mod forces;
pub mod integrators;
pub mod maxwell;
pub mod pairs;
pub mod types;

pub use crate::ff::potential::pair::{LJCut, PairPotential};
pub use crate::ff::potential::{Potential, Potentials};
/// The virial tensor. It lives in [`crate::math`] because it is a property of
/// a force evaluation, not of the loop that runs one — a pair kernel tallies
/// it, and a kernel may not name `md`.
pub use crate::math::Virial;
pub use error::MdError;
pub use forces::{Direct, ForceProvider, GhostPairs, MicPairs, NeighborStats};
pub use integrators::{Langevin, VelocityVerlet, kinetic_energy, scalar_mass};
pub use maxwell::MaxwellBoltzmann;
pub use pairs::{BondedLists, Comm};
pub use types::{ForceOutput, MDObservables, MDState};
