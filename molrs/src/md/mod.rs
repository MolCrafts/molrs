//! In-process molecular-dynamics engine.
//!
//! Neighbour lists: [`crate::spatial::neighbors`] (`NeighborList`,
//! `VerletSkin`). Science here:
//!
//! * [`Potential`] / [`LJ`] — `calc_energy` / `calc_force` / `eval → (E, F)`
//! * [`VelocityVerlet`] / [`Langevin`] — constructed with
//!   `(dt, potential, neighbors, mass)` (Langevin also `gamma`, `kbt`, `seed`)
//!
//! No `bind_*` façades. Compose required pieces in the constructor.

pub mod error;
pub mod integrators;
pub mod lj;
pub mod maxwell;
pub mod types;
pub mod units;

pub use error::MdError;
pub use integrators::{Langevin, VelocityVerlet, kinetic_energy, scalar_mass};
pub use lj::{LJ, Potential};
pub use maxwell::MaxwellBoltzmann;
pub use types::{ForceOutput, MDObservables, MDState};
pub use units::{MD_ENERGY, energy_to_md, kb_md, preset_energy_to_md};
