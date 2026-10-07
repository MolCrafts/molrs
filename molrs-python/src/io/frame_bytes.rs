//! The frame-bytes doors, `molrs.io.read_frame_bytes` /
//! `molrs.io.write_frame_bytes`: a [`PyFrame`] to and from the wire encoding
//! of `molrs::stream` (MessagePack or JSON). The encoding is the stream's —
//! [`molrs::stream::Publisher`] puts it on the wire — and reading or writing a
//! frame in it is `molrs.io`'s, as for every other format.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::core::frame::PyFrame;
use crate::error::py_value_err;
use crate::stream::message_format;

/// Rebuild a :class:`Frame` from streaming wire bytes.
///
/// This is the encoding ``molrs::stream::Publisher`` puts on the wire, so a
/// consumer decodes a live stream with this and never re-derives the layout.
///
/// Parameters
/// ----------
/// data : bytes
///     A payload produced by :func:`write_frame_bytes` or by a Rust
///     ``Publisher``.
/// format : {"msgpack", "json"}
///     Wire encoding the payload was written with.
#[pyfunction]
#[pyo3(signature = (data, format = "msgpack"))]
pub fn read_frame_bytes(data: &[u8], format: &str) -> PyResult<PyFrame> {
    let fmt = message_format(format)?;
    let frame = molrs::stream::bytes_to_frame(data, fmt).map_err(py_value_err)?;
    PyFrame::from_core_frame(frame)
}

/// Encode a :class:`Frame` as streaming wire bytes. The inverse of
/// :func:`read_frame_bytes`.
#[pyfunction]
#[pyo3(signature = (frame, format = "msgpack"))]
pub fn write_frame_bytes<'py>(
    py: Python<'py>,
    frame: &PyFrame,
    format: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let fmt = message_format(format)?;
    let bytes = frame
        .with_frame(|f| molrs::stream::frame_to_bytes(f, fmt))?
        .map_err(py_value_err)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Register the frame-bytes doors on the native module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_frame_bytes, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_frame_bytes, m)?)
}
