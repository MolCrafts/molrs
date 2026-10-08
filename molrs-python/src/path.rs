//! Path arguments: a ``str`` or any ``os.PathLike``, as the core readers and
//! writers take them.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A path argument (a ``str`` or any ``os.PathLike``, extracted as a
/// [`std::path::PathBuf`]) as the `&str` the core readers and writers take.
pub(crate) fn path_str(path: &std::path::Path) -> PyResult<&str> {
    path.to_str().ok_or_else(|| {
        PyValueError::new_err(format!("path is not valid UTF-8: {}", path.display()))
    })
}
