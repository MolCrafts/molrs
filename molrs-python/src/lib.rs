//! Python bindings for the molrs molecular simulation library.
//!
//! This crate builds the private native module `molrs._lib`; the Python
//! package `molrs` (`python/molrs`) gives every symbol its one public path,
//! the Python module named after the symbol's Rust owner:
//!
//! | binding                 | Rust owner          | Python module        |
//! |-------------------------|---------------------|----------------------|
//! | [`core::store`]         | `molrs::store`      | `molrs.store`        |
//! | [`core::spatial`]       | `molrs::spatial`    | `molrs.spatial`      |
//! | [`core::system`]        | `molrs::system`     | `molrs.system`       |
//! | [`core::units`]         | `molrs::units`      | `molrs.units`        |
//! | [`op`]                  | `molrs::op`         | `molrs.op`           |
//! | [`perceive`]            | `molrs::perceive`   | `molrs.perceive`     |
//! | [`io`]                  | `molrs::io`         | `molrs.io`           |
//! | [`ff`]                  | `molrs::ff`         | `molrs.ff.*`         |
//! | [`optimize`]            | `molrs::optimize`   | `molrs.optimize`     |
//! | [`md`]                  | `molrs::md`         | `molrs.md`           |
//! | [`conformer`]           | `molrs::conformer`  | `molrs.conformer`    |
//! | [`builder`]             | `molrs::builder`    | `molrs.builder`      |
//! | [`compute`]             | `molrs::compute`    | `molrs.compute`      |
//! | [`signal`]              | `molrs::signal`     | `molrs.signal`       |
//! | [`stream`]              | `molrs::stream`     | `molrs.stream`       |
//!
//! Every subsystem binding registers its own classes and functions through
//! its `register`. Most land flat on `_lib`; a namespace with vocabulary of
//! its own is a `_lib` submodule (`op`, `md`, `ff.ir` as `ir`, and the store's
//! `keys` / `schema`). Cross-cutting plumbing has one home each: exceptions
//! and Rust-error mapping in [`error`], the pickle protocol in [`pickle`],
//! path arguments in [`path`].
//!
//! # Float Precision
//!
//! Every floating-point array crosses as `f64` (numpy `float64`): molrs fixes
//! `F = f64`, and there is no precision feature.

use pyo3::prelude::*;

mod error;
mod path;
mod pickle;

mod builder;
mod compute;
mod conformer;
mod core;
mod ff;
mod io;
mod md;
mod op;
mod optimize;
mod perceive;
mod signal;
mod stream;

/// The FFI ABI handshake token of this build.
///
/// Returns ``(abi_line, version, frameref_capsule_name, forcefield_capsule_name,
/// regionref_capsule_name)``
/// — e.g. ``("0.14", "0.14.0", "molrs.FrameRef/0.14", "molrs.ForceFieldRef/0.14")``.
///
/// A downstream extension that exchanges ``molrs_ffi`` handle capsules with
/// this wheel (e.g. molpack) calls this once at import and compares
/// ``abi_line`` against the line of the molrs it statically embeds; a mismatch
/// is raised as a clear ``ImportError`` instead of surfacing later as a
/// capsule-name ``ValueError`` (or, before capsule names were versioned,
/// undefined behavior). Patch versions may differ — layout is frozen within a
/// minor line (see `molrs-ffi`'s layout snapshot gate).
#[pyfunction]
fn _ffi_abi_token() -> (&'static str, &'static str, String, String, String) {
    (
        molrs_ffi::abi::abi_line(),
        ::molrs::VERSION,
        molrs_ffi::abi::frameref_capsule_name()
            .to_string_lossy()
            .into_owned(),
        molrs_ffi::abi::forcefield_capsule_name()
            .to_string_lossy()
            .into_owned(),
        molrs_ffi::abi::regionref_capsule_name()
            .to_string_lossy()
            .into_owned(),
    )
}

/// The native module, `molrs._lib`: every subsystem's bindings.
#[pymodule]
#[pyo3(name = "_lib")]
fn molrs_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_ffi_abi_token, m)?)?;
    m.add_function(wrap_pyfunction!(pickle::_restore_pickled_state, m)?)?;

    core::register(m)?;
    perceive::register(m)?;
    io::register(m)?;
    ff::register(m)?;
    optimize::register(m)?;
    conformer::register(m)?;
    builder::register(m)?;
    compute::register(m)?;
    signal::register(m)?;
    stream::register(m)?;

    let op_module = PyModule::new(m.py(), "op")?;
    op::register(&op_module)?;
    m.add_submodule(&op_module)?;
    let md_module = PyModule::new(m.py(), "md")?;
    md::register(&md_module)?;
    m.add_submodule(&md_module)?;
    Ok(())
}
