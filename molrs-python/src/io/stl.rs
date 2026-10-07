//! STL, stereolithography triangle meshes, into a `molrs.core.TriMesh`
//! (`molrs.io.read_stl`).

use crate::core::mesh::PyTriMesh;
use crate::error::io_error_to_pyerr;
use crate::path::path_str;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Read an STL surface mesh (ASCII or binary) into a [`PyTriMesh`].
///
/// The file's numbers are taken as they are; convert with
/// ``TriMesh.scaled`` when the file is not in the length unit you work in.
#[pyfunction]
pub fn read_stl(path: PathBuf) -> PyResult<PyTriMesh> {
    let path = path_str(&path)?;
    let mesh = molrs::io::read_stl(path).map_err(io_error_to_pyerr)?;
    Ok(PyTriMesh { inner: mesh })
}

/// Register this module's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_stl, m)?)?;
    Ok(())
}
