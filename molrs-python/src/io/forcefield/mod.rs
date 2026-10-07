//! Python bindings for `molrs::io::forcefield`: force-field files read into
//! and written from a `molrs.ff.forcefield.ForceField`, every function flat on
//! `molrs.io` as `read_<fmt>_…` / `write_<fmt>_…`.

mod readers;
mod writers;

use pyo3::prelude::*;

/// Register the force-field file readers and writers on the native module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    readers::register(m)?;
    writers::register(m)
}
