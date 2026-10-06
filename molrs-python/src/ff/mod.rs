//! Python bindings for `molrs::ff`, one file (or directory) per Rust
//! submodule, each the binding of its owner and exposed at the Python module
//! of the same name:
//!
//! | binding            | Rust owner           | Python module          |
//! |--------------------|----------------------|------------------------|
//! | [`forcefield`]     | `ff::forcefield`     | `molrs.ff.forcefield`  |
//! | [`potential`]      | `ff::potential`      | `molrs.ff.potential`   |
//! | [`typifier`]       | `ff::typifier`       | `molrs.ff.typifier`    |
//! | [`charge`]         | `ff::charge`         | `molrs.ff.charge`      |
//! | [`ir`]             | `ff::ir`             | `molrs.ff.ir`          |
//! | [`params`]         | `ff::params`         | `molrs.ff.params`      |
//! | [`scale_lj`]       | `ff::scale_lj`       | `molrs.ff.scale_lj`    |
//!
//! The force-field file formats are `ff::forcefield`'s (they map files onto
//! the force-field IR); structure and trajectory formats are `molrs.io`'s.

pub mod charge;
pub mod forcefield;
pub mod ir;
pub mod params;
pub mod potential;
pub mod scale_lj;
pub mod typifier;

use pyo3::prelude::*;

/// Register every `molrs::ff` binding: the flat classes and functions on the
/// native module, and `molrs.ff.ir`'s registry as its `ir` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    forcefield::register(m)?;
    potential::register(m)?;
    typifier::register(m)?;
    charge::register(m)?;
    params::register(m)?;
    scale_lj::register(m)?;
    crate::add_submodule(m, "ir", "molrs.ff.ir", ir::register)
}
