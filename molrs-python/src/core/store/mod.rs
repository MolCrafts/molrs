//! Python bindings for `molrs::store` (`molrs.store`): the column store
//! (`Block`), the frame of named blocks with its metadata (`Frame`,
//! `FrameMeta`, `MetaValue`, `MetaDocument`), the frame sequence and its
//! observables (`Trajectory`, `ScalarObservable`, `VectorObservable`), and
//! the column vocabulary (`molrs.store.keys`, `molrs.store.schema`).

pub mod block;
pub mod frame;
pub mod schema;
pub mod trajectory;

use pyo3::prelude::*;

/// Register `molrs.store`: its classes, `BlockDtypeError`, and the `keys` /
/// `schema` vocabulary submodules.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "BlockDtypeError",
        m.py().get_type::<crate::error::BlockDtypeError>(),
    )?;
    m.add_class::<block::PyBlock>()?;
    m.add_class::<frame::PyMetaValue>()?;
    m.add_class::<frame::PyMetaDocument>()?;
    m.add_class::<frame::PyFrameMeta>()?;
    m.add_class::<frame::PyFrame>()?;
    m.add_class::<trajectory::PyTrajectory>()?;
    m.add_class::<trajectory::PyScalarObservable>()?;
    m.add_class::<trajectory::PyVectorObservable>()?;
    crate::add_submodule(m, "keys", "molrs.store.keys", schema::register_keys)?;
    crate::add_submodule(m, "schema", "molrs.store.schema", schema::register_schema)?;
    Ok(())
}
