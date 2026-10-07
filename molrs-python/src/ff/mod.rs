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
//! | [`clpol_scaling`]  | `ff::clpol_scaling`  | `molrs.ff.clpol_scaling` |
//!
//! No file format is here: force-field files, like every other file, are
//! `molrs.io`'s (`crate::io`).

pub mod charge;
pub mod clpol_scaling;
pub mod forcefield;
pub mod ir;
pub mod params;
pub mod potential;
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
    clpol_scaling::register(m)?;
    crate::add_submodule(m, "ir", "molrs.ff.ir", ir::register)
}
