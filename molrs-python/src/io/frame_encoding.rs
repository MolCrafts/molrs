//! A frame in the wire encodings of `molrs::stream`: MessagePack bytes
//! (`molrs.io.read_msgpack_frame_bytes` / `write_msgpack_frame_bytes`) and
//! JSON text (`read_json_frame_str` / `write_json_frame_str`). The encoding
//! is the stream's — [`molrs::stream::Publisher`] puts it on the wire — and
//! reading or writing a frame in it is `molrs.io`'s, as for every other
//! format.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::core::frame::PyFrame;
use crate::error::py_value_err;

/// Rebuild a :class:`Frame` from MessagePack bytes — what a
/// :class:`molrs.stream.Publisher` streams by default.
///
/// Parameters
/// ----------
/// data : bytes
///     A payload written by :func:`write_msgpack_frame_bytes` or a publisher.
#[pyfunction]
pub fn read_msgpack_frame_bytes(data: &[u8]) -> PyResult<PyFrame> {
    let frame = molrs::io::read_msgpack_frame_bytes(data).map_err(py_value_err)?;
    PyFrame::from_core_frame(frame)
}

/// Encode a :class:`Frame` as MessagePack bytes — the inverse of
/// :func:`read_msgpack_frame_bytes`.
#[pyfunction]
pub fn write_msgpack_frame_bytes<'py>(
    py: Python<'py>,
    frame: &PyFrame,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = frame
        .with_frame(molrs::io::write_msgpack_frame_bytes)?
        .map_err(py_value_err)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Rebuild a :class:`Frame` from its JSON text — the debugging / interop
/// wire encoding.
///
/// Parameters
/// ----------
/// text : str
///     A document written by :func:`write_json_frame_str` or a JSON publisher.
#[pyfunction]
pub fn read_json_frame_str(text: &str) -> PyResult<PyFrame> {
    let frame = molrs::io::read_json_frame_str(text).map_err(py_value_err)?;
    PyFrame::from_core_frame(frame)
}

/// Encode a :class:`Frame` as JSON text — the inverse of
/// :func:`read_json_frame_str`.
#[pyfunction]
pub fn write_json_frame_str(frame: &PyFrame) -> PyResult<String> {
    frame
        .with_frame(molrs::io::write_json_frame_str)?
        .map_err(py_value_err)
}

/// Register the frame-encoding doors on the native module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for door in [
        wrap_pyfunction!(read_msgpack_frame_bytes, m)?,
        wrap_pyfunction!(write_msgpack_frame_bytes, m)?,
        wrap_pyfunction!(read_json_frame_str, m)?,
        wrap_pyfunction!(write_json_frame_str, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
