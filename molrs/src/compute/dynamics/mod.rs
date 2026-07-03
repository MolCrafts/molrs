//! Space–time correlation dynamics: the van Hove function and pair-persistence
//! survival correlators.
//!
//! | Method | Input | Output |
//! |--------|-------|--------|
//! | [`VanHove`] | trajectory frames | [`VanHoveResult`] — self/distinct G(r, t) |
//! | [`pair_survival_tcf`] | per-frame pair presence | [`PersistResult`] — survival time-correlation |
//!
//! Complements [`transport`](crate::compute::transport) (which reduces motion
//! to scalar coefficients): these methods keep the full space- and/or
//! time-resolved correlation structure.
//!
//! ```ignore
//! let gvh = VanHove::new(n_rbins, r_max, vec![0, 10, 100])?.compute(&frames, ())?;
//! ```

pub mod persist;
pub mod van_hove;

pub use persist::{PersistResult, SurvivalMethod, pair_survival_tcf};
pub use van_hove::{VanHove, VanHoveResult};
