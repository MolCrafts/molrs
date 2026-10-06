//! Python bindings for molrs's core subsystems — `molrs::store`,
//! `molrs::spatial`, `molrs::system` and `molrs::units` — each exposed as the
//! Python module of the same name (`molrs.store`, `molrs.spatial`,
//! `molrs.system`, `molrs.units`).

pub mod spatial;
pub mod store;
pub mod system;
pub mod units;

use pyo3::prelude::*;

/// Register the four core subsystems on the native module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    store::register(m)?;
    spatial::register(m)?;
    system::register(m)?;
    units::register(m)?;
    Ok(())
}
