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

// No re-exports of `ff` or `core` types here. `LJCut`, `PairPotential`,
// `Potential`, `Potentials` and `Virial` are owned by the modules that define
// them, and a second public spelling is a second name to keep true — the
// module doc above says where each lives, which is the pointer a reader needs.
pub use error::MdError;
pub use forces::{Direct, ForceProvider, GhostPairs, MicPairs, NeighborStats};
pub use integrators::{Langevin, VelocityVerlet, kinetic_energy, scalar_mass};
pub use maxwell::MaxwellBoltzmann;
pub use pairs::{BondedLists, Comm, SpecialWeights};
pub use types::{ForceOutput, MDState};
