//! Shared helper functions and type aliases for PyO3 bindings.
//!
//! This module provides error conversion functions that map Rust error types
//! to appropriate Python exceptions, and the [`NpF`] type alias that matches
//! the crate's float precision setting.

use molrs::spatial::simbox::BoxError;
use molrs::types::F;
use ndarray::{Array1, array};
use numpy::PyReadonlyArray1;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Numpy float type matching the `F` alias — always `f64`.
pub type NpF = f64;

/// Pickle protocol: reconstruct with ``type(self)(*args)``.
pub fn reduce_via_type<'py>(
    slf: &Bound<'py, PyAny>,
    args: impl IntoPyObject<'py>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
    let py = slf.py();
    let packed = args.into_bound_py_any(py)?;
    let tuple = if let Ok(tuple) = packed.cast::<PyTuple>() {
        tuple.to_owned()
    } else {
        PyTuple::new(py, [packed])?
    };
    Ok((slf.get_type().into_any(), tuple))
}

/// Parse an optional origin array, defaulting to `[0, 0, 0]`.
///
/// # Errors
///
/// Returns `PyValueError` if the array does not have exactly 3 elements.
pub fn parse_origin(origin: Option<PyReadonlyArray1<'_, NpF>>) -> PyResult<Array1<F>> {
    match origin {
        Some(o) => {
            let s = o.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("origin must have length 3"));
            }
            Ok(array![s[0], s[1], s[2]])
        }
        None => Ok(array![0.0 as F, 0.0 as F, 0.0 as F]),
    }
}

/// Parse an optional PBC flag array, defaulting to `[true, true, true]`.
///
/// # Errors
///
/// Returns `PyValueError` if the array does not have exactly 3 elements.
pub fn parse_pbc(pbc: Option<PyReadonlyArray1<'_, bool>>) -> PyResult<[bool; 3]> {
    match pbc {
        Some(p) => {
            let s = p.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("pbc must have 3 elements"));
            }
            Ok([s[0], s[1], s[2]])
        }
        None => Ok([true, true, true]),
    }
}

/// Convert a [`BoxError`] to a Python `ValueError`.
pub fn box_error_to_pyerr(e: BoxError) -> PyErr {
    PyValueError::new_err(format!("{:?}", e))
}

/// Convert a [`std::io::Error`] to a Python `IOError`.
pub fn io_error_to_pyerr(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

/// Convert a [`molrs::MolRsError`] to a Python `ValueError`.
pub fn molrs_error_to_pyerr(e: molrs::MolRsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert a [`molrs::io::smiles::SmilesError`] to a Python
/// [`SmilesError`](crate::error::SmilesError).
///
/// The rendered Rust message stays `args[0]`, so `str(e)` is unchanged and the
/// class subclasses `ValueError`; on top of it the four facts the Rust error
/// owns cross as plain attributes — `kind`, `span`, `input`, `notation` — so a
/// caller can branch on the reason instead of matching message text.
///
/// The helper takes no [`Python`] token (it is used as a bare `map_err`
/// argument all over the io bindings), so it attaches to the interpreter
/// itself. Nothing here can panic across the seam: if the interpreter refuses
/// to carry the attributes — only reachable under allocation failure or
/// interpreter shutdown — that failure is returned as the raised error rather
/// than unwrapped.
pub fn smiles_error_to_pyerr(e: molrs::io::smiles::SmilesError) -> PyErr {
    let message = e.to_string();
    // A span is a byte range into `input`, and the scanner reports
    // end-of-input one byte past the text (`Span::new(len, len + 1)`), which
    // is not a valid slice bound of the string published alongside it — so the
    // end is clamped here. The start is clamped to that end so the pair stays
    // an ordered range for errors raised away from the scanner, which carry a
    // span into text they do not publish (`input` is then empty).
    let end = e.span.end.min(e.input.len());
    let start = e.span.start.min(end);

    Python::attach(|py| {
        let err = crate::error::SmilesError::new_err(message);
        let value = err.value(py);
        let attached = (|| -> PyResult<()> {
            value.setattr("kind", smiles_error_kind_name(&e.kind))?;
            value.setattr("span", (start, end))?;
            value.setattr("input", e.input.as_str())?;
            value.setattr("notation", notation_name(e.notation))?;
            Ok(())
        })();
        match attached {
            Ok(()) => err,
            Err(failed) => failed,
        }
    })
}

/// The bare variant name of a [`SmilesErrorKind`], payload dropped.
///
/// Derived from the derived [`Debug`] rendering rather than from a hand-written
/// table: a table would have to be extended every time the Rust enum grows a
/// variant, and a forgotten row is a wrong `kind` at the seam rather than a
/// compile error. `Debug` writes the variant name first in all three shapes —
/// `UnexpectedEnd`, `UnexpectedChar('X')`, `RingBondConflict { rnum: 1 }` — so
/// cutting at the first `(` or space leaves exactly the name.
///
/// [`SmilesErrorKind`]: molrs::io::smiles::SmilesErrorKind
fn smiles_error_kind_name(kind: &molrs::io::smiles::SmilesErrorKind) -> String {
    let rendered = format!("{kind:?}");
    rendered
        .split(['(', ' '])
        .next()
        .unwrap_or(rendered.as_str())
        .to_owned()
}

/// The lowercase name of a [`Notation`], matching how every other enum crosses
/// this seam (bond kinds, pair ends).
///
/// Matched totally: a new notation in Rust must be spelled here, not silently
/// rendered as some fallback.
///
/// [`Notation`]: molrs::io::smiles::Notation
fn notation_name(notation: molrs::io::smiles::Notation) -> &'static str {
    match notation {
        molrs::io::smiles::Notation::Smiles => "smiles",
        molrs::io::smiles::Notation::Smarts => "smarts",
        molrs::io::smiles::Notation::CGsmiles => "cgsmiles",
    }
}

/// Convert any `Display` error (typically a `molrs-compute` / `molrs-signal`
/// analysis error) to a Python `ValueError`. Shared by the analysis bindings.
pub fn py_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Resolve a wire-encoding name onto [`MessageFormat`].
///
/// The two spellings are the only ones the Rust side can produce, so an
/// unknown name is an error rather than a silent fall back to MessagePack —
/// a caller who writes `"messagepack"` must find out, not stream bytes the
/// peer will read as JSON.
pub(crate) fn message_format(name: &str) -> PyResult<molrs::stream::MessageFormat> {
    match name {
        "msgpack" => Ok(molrs::stream::MessageFormat::MessagePack),
        "json" => Ok(molrs::stream::MessageFormat::Json),
        other => Err(PyValueError::new_err(format!(
            "unknown wire format {other:?}; expected 'msgpack' or 'json'"
        ))),
    }
}

/// Collect owned core [`Frame`]s from a single `Frame` or a list of them.
/// Used by every batch-`compute` binding to accept both shapes.
///
/// [`Frame`]: molrs::store::frame::Frame
pub(crate) fn collect_frames(
    frames: &Bound<'_, PyAny>,
) -> PyResult<Vec<molrs::store::frame::Frame>> {
    use crate::core::store::frame::PyFrame;
    if let Ok(single) = frames.extract::<PyRef<'_, PyFrame>>() {
        return Ok(vec![single.clone_core_frame()?]);
    }
    let list: Vec<PyRef<'_, PyFrame>> = frames.extract()?;
    list.iter().map(|f| f.clone_core_frame()).collect()
}

/// Collect owned [`Neighbors`] tables from a single wrapper or a list of them.
///
/// The analyses take one materialized table per frame, so this is where the
/// binder accepts either shape. The engine that produced a table
/// (`NeighborList`) is deliberately not accepted: it would have to guess a
/// column policy, and a guess that drops `disp` is exactly the silent failure
/// this chain removed.
///
/// [`Neighbors`]: molrs::spatial::neighbors::Neighbors
pub(crate) fn collect_neighbors(
    arg: &Bound<'_, PyAny>,
) -> PyResult<Vec<molrs::spatial::neighbors::Neighbors>> {
    use crate::core::spatial::neighborlist::PyNeighbors;
    if let Ok(single) = arg.extract::<PyRef<'_, PyNeighbors>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyNeighbors>> = arg.extract()?;
    Ok(list.iter().map(|n| n.inner.clone()).collect())
}

/// A path argument (a ``str`` or any ``os.PathLike``, extracted as a
/// [`std::path::PathBuf`]) as the `&str` the core readers and writers take.
pub(crate) fn path_str(path: &std::path::Path) -> PyResult<&str> {
    path.to_str().ok_or_else(|| {
        PyValueError::new_err(format!("path is not valid UTF-8: {}", path.display()))
    })
}
