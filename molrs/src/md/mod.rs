//! In-process molecular-dynamics engine: integrators and force providers.
//! Energy kernels are not part of `md`; they live in
//! [`crate::ff::potential`] and reach the integrators through a
//! [`ForceProvider`](crate::md::ForceProvider).
//!
//! Neighbour lists: [`crate::spatial::neighbors`] (`NeighborList`,
//! `VerletSkin`). Science here:
//!
//! * [`crate::ff::potential::Potential`] produces energy and forces from flat
//!   coordinates. [`crate::ff::potential::Member`] is one term of a force
//!   evaluation with the part it plays already chosen — `Indexed` for a bonded
//!   term that takes an index table, `Pair` for one summed over a neighbour
//!   table, `Plain` for an external field. The providers below hold a
//!   `Vec<Member>` and match on it, so no step re-derives which is which.
//!   [`crate::ff::potential::Potentials`] sums them as one potential.
//! * Pair kernels implement [`crate::ff::potential::pair::PairPotential`];
//!   the force provider supplies current neighbour pairs via
//!   [`crate::ff::potential::Potential::calc_energy_forces_with_pairs`], so a
//!   potential neither owns nor updates the skin.
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
//! No `bind_*` façades. Compose required pieces in the constructor.

mod error;
mod forces;
mod integrators;
mod maxwell;
mod pairs;
mod types;

// No re-exports of `ff` or `core` types here. `LJCut`, `PairPotential`,
// `Potential`, `Potentials` and `Virial` are owned by the modules that define
// them, and a second public spelling is a second name to keep true — the
// module doc above says where each lives, which is the pointer a reader needs.
pub use error::MdError;
pub use forces::{Direct, ForceProvider, GhostPairs, MicPairs, NeighborStats};
pub use integrators::{Langevin, VelocityVerlet, kinetic_energy, scalar_mass};
pub use maxwell::{MaxwellBoltzmann, com_velocity};
pub use pairs::{BondedLists, Comm};
pub use types::{ForceOutput, MDState};
