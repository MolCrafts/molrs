//! Scientific-record (`*.mrec`) path doors.
//!
//! `Frame` and `Trajectory` are the in-memory objects. These functions and
//! [`PyMrecTrajectoryReader`] are the filesystem doors, exported to Python as
//! `molrs.io.mrec`. The class is `MrecTrajectoryReader` on `_lib` so it does
//! not collide with the dump concatenator `molrs.io.TrajectoryReader`.

use std::path::Path;

use crate::core::store::frame::PyFrame;
use crate::core::store::trajectory::PyTrajectory;
use crate::helpers::molrs_error_to_pyerr;
use molrs::io::mrec::{FrameSequence, FrameSequenceWriter, SequenceSchema};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use serde_json::{Map as JsonMap, Value as JsonValue};

/// Write a snapshot as a record whose only state section is ``frame``.
///
/// Args:
///     path: Destination filesystem path.
///     frame: In-memory :class:`~molrs.Frame` to persist.
///     system: Optional system-definition :class:`~molrs.Frame` written
///         beside the snapshot as the ``system/`` section.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or the
///         frame fails to encode.
#[pyfunction]
#[pyo3(signature = (path, frame, system=None, meta=None))]
pub fn write_frame(
    path: &str,
    frame: &Bound<'_, PyFrame>,
    system: Option<&Bound<'_, PyFrame>>,
    meta: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let core = frame.borrow().clone_core_frame()?;
    let system_core = system
        .map(|sys| sys.borrow().clone_core_frame())
        .transpose()?;
    let meta_map = meta.map(dict_to_json_map).transpose()?;
    molrs::io::mrec::write_frame_file(path, &core, system_core.as_ref(), meta_map.as_ref())
        .map_err(molrs_error_to_pyerr)
}

/// Write a topology as a record whose only state section is ``system``.
///
/// Args:
///     path: Destination filesystem path.
///     system: In-memory :class:`~molrs.Frame` to persist as ``system/``.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or the
///         frame fails to encode.
#[pyfunction]
#[pyo3(signature = (path, system, meta=None))]
pub fn write_system(
    path: &str,
    system: &Bound<'_, PyFrame>,
    meta: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let meta_map = meta.map(dict_to_json_map).transpose()?;
    molrs::io::mrec::write_system_file(
        path,
        &system.borrow().clone_core_frame()?,
        meta_map.as_ref(),
    )
    .map_err(molrs_error_to_pyerr)
}

/// Write a trajectory as a record whose only state section is ``trajectory``.
///
/// Args:
///     path: Destination filesystem path.
///     traj: In-memory :class:`~molrs.Trajectory` to persist.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or a frame
///         fails to encode.
#[pyfunction]
pub fn write_trajectory(path: &str, traj: PyRef<'_, PyTrajectory>) -> PyResult<()> {
    molrs::io::mrec::write_trajectory_file(path, &traj.inner).map_err(molrs_error_to_pyerr)
}

/// Read the ``frame`` section of a ``*.mrec`` store.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The in-memory :class:`~molrs.Frame`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, the store
///         has no ``frame`` section, or a section fails to decode.
#[pyfunction]
pub fn read_frame(path: &str) -> PyResult<PyFrame> {
    let frame = molrs::io::mrec::read_frame_file(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read the ``system`` section of a ``*.mrec`` store.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The in-memory :class:`~molrs.Frame`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, the store
///         has no ``system`` section, or a section fails to decode.
#[pyfunction]
pub fn read_system(path: &str) -> PyResult<PyFrame> {
    let frame = molrs::io::mrec::read_system_file(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read the ``trajectory`` section of a ``*.mrec`` store.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The in-memory :class:`~molrs.Trajectory`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or a
///         section fails to decode.
#[pyfunction]
pub fn read_trajectory(path: &str) -> PyResult<PyTrajectory> {
    let inner = molrs::io::mrec::read_trajectory_file(path).map_err(molrs_error_to_pyerr)?;
    Ok(PyTrajectory { inner })
}

/// Read the mandatory ``meta`` document of a ``*.mrec`` store.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The record-level metadata mapping, including the stamped brand keys.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or ``meta``
///         is missing or does not match the mrec contract.
#[pyfunction]
pub fn read_meta(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    let map = molrs::io::mrec::read_meta_file(path).map_err(molrs_error_to_pyerr)?;
    Ok(json_map_to_dict(py, &map)?.unbind())
}

/// Child group names at the record root (``meta``, ``frame``, ``system``, …).
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     Sorted section names. Callers ask which sections are present instead
///     of probing ``read_frame`` / ``read_system``.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or the store
///         is not a readable record.
#[pyfunction]
pub fn section_names(path: &str) -> PyResult<Vec<String>> {
    molrs::io::mrec::section_names(path).map_err(molrs_error_to_pyerr)
}

/// Lazy one-frame cursor over a ``*.mrec`` trajectory.
///
/// Wraps the Rust ``FrameSequence`` store cursor: construction opens the
/// index, and :meth:`read_frame` decodes exactly the asked-for frame.
///
/// Args:
///     path: Filesystem path of the record store.
#[pyclass(module = "molrs.io.mrec", name = "MrecTrajectoryReader", unsendable)]
pub struct PyMrecTrajectoryReader {
    inner: FrameSequence,
}

#[pymethods]
impl PyMrecTrajectoryReader {
    #[new]
    fn py_new(path: &str) -> PyResult<Self> {
        let inner =
            molrs::io::mrec::open_trajectory_sequence(path).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Decode one committed frame.
    ///
    /// Args:
    ///     index: Zero-based frame index.
    ///
    /// Returns:
    ///     The frame at ``index``.
    ///
    /// Raises:
    ///     IndexError: If ``index`` is past the commit marker.
    ///     ValueError: If a section fails to decode.
    fn read_frame(&mut self, index: isize) -> PyResult<PyFrame> {
        let resolved = self.resolve_index(index)?;
        let frame = self
            .inner
            .frame(resolved)
            .map_err(molrs_error_to_pyerr)?
            .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
        PyFrame::from_core_frame(frame)
    }

    /// Number of committed frames.
    fn __len__(&self) -> usize {
        self.inner.steps().len()
    }

    /// Frame at ``index``, with Python negative-from-the-end indexing.
    fn __getitem__(&mut self, index: isize) -> PyResult<PyFrame> {
        self.read_frame(index)
    }

    /// Step numbers of the committed frames.
    #[getter]
    fn step(&self) -> Vec<i64> {
        self.inner.steps().to_vec()
    }

    /// Physical times (fs) of the committed frames, when the run wrote any.
    #[getter]
    fn time(&self) -> Option<Vec<f64>> {
        self.inner.times().map(<[f64]>::to_vec)
    }

    /// Whether the store carries a block section of this name (e.g. ``"bonds"``).
    ///
    /// Lets a caller decide once — instead of probing every frame — whether a
    /// per-frame section such as dynamic bonds was written.
    fn has_block(&self, name: &str) -> bool {
        self.inner.has_block(name)
    }

    /// Names of every block section present in the store.
    fn block_names(&self) -> Vec<String> {
        self.inner.block_names().map(str::to_string).collect()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> bool {
        false
    }
}

impl PyMrecTrajectoryReader {
    /// Resolve a possibly-negative Python index against the committed length.
    fn resolve_index(&self, index: isize) -> PyResult<u64> {
        let len = self.inner.steps().len() as isize;
        let normalized = if index < 0 { index + len } else { index };
        if normalized < 0 || normalized >= len {
            return Err(PyIndexError::new_err("trajectory index out of range"));
        }
        Ok(normalized as u64)
    }
}

/// A frame-sequence schema pinned before a run's frames are written.
///
/// The schema fixes every block, column and dtype the run may carry;
/// :class:`MrecTrajectoryWriter` refuses a frame that steps outside it. Derive
/// one from a representative frame, or from the union of several.
#[pyclass(module = "molrs.io.mrec", name = "MrecSequenceSchema")]
pub struct PyMrecSequenceSchema {
    inner: SequenceSchema,
}

#[pymethods]
impl PyMrecSequenceSchema {
    /// Derive a schema from one representative frame.
    #[staticmethod]
    fn from_frame(frame: &Bound<'_, PyFrame>) -> PyResult<Self> {
        let core = frame.borrow().clone_core_frame()?;
        let inner = SequenceSchema::from_frame(&core).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Derive a schema from the union of several frames' blocks and columns.
    #[staticmethod]
    fn from_frames(frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<Self> {
        let cores = frames
            .iter()
            .map(|frame| frame.clone_core_frame())
            .collect::<PyResult<Vec<_>>>()?;
        let inner = SequenceSchema::from_frames(&cores).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }
}

/// Append-first writer for a ``*.mrec`` trajectory store.
///
/// Wraps the Rust ``FrameSequenceWriter`` over the fast positional-write store,
/// so a growing run is written frame by frame without holding the whole
/// trajectory in memory. Use as a context manager, or call :meth:`close`.
///
/// Args:
///     path: Destination filesystem path for the store.
///     schema: The :class:`MrecSequenceSchema` every appended frame is checked
///         against.
#[pyclass(module = "molrs.io.mrec", name = "MrecTrajectoryWriter", unsendable)]
pub struct PyMrecTrajectoryWriter {
    inner: Option<FrameSequenceWriter>,
}

#[pymethods]
impl PyMrecTrajectoryWriter {
    #[new]
    fn py_new(path: &str, schema: &Bound<'_, PyMrecSequenceSchema>) -> PyResult<Self> {
        let schema = schema.borrow().inner.clone();
        let writer = FrameSequenceWriter::create_at(path, schema).map_err(molrs_error_to_pyerr)?;
        Ok(Self {
            inner: Some(writer),
        })
    }

    /// Buffer a frame. With no ``step`` the writer numbers frames 0, 1, 2, …;
    /// pass ``step`` (and optionally ``time`` in fs) for real MD numbering.
    ///
    /// Raises:
    ///     ValueError: If the writer is closed, the frame steps outside the
    ///         pinned schema, or ``step`` does not strictly increase.
    #[pyo3(signature = (frame, step=None, time=None))]
    fn append(
        &mut self,
        frame: &Bound<'_, PyFrame>,
        step: Option<i64>,
        time: Option<f64>,
    ) -> PyResult<()> {
        let core = frame.borrow().clone_core_frame()?;
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer is closed"))?;
        match step {
            Some(step) => writer
                .append_at(&core, step, time)
                .map_err(molrs_error_to_pyerr),
            None if time.is_some() => Err(PyValueError::new_err("time requires an explicit step")),
            None => writer.append(&core).map_err(molrs_error_to_pyerr),
        }
    }

    /// Land every buffered frame on disk.
    fn flush(&mut self) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer is closed"))?;
        writer.flush().map_err(molrs_error_to_pyerr)
    }

    /// Flush and seal the store. Idempotent; appends after close raise.
    fn close(&mut self) -> PyResult<()> {
        if let Some(writer) = self.inner.take() {
            writer.close().map_err(molrs_error_to_pyerr)?;
        }
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Refuse the retired ``.zarr`` / ``.zarr.zip`` scientific suffixes.
///
/// Args:
///     path: Filesystem path of a record store.
///
/// Raises:
///     ValueError: If the path uses a retired suffix.
#[pyfunction]
pub fn mrec_validate_path(path: &str) -> PyResult<()> {
    molrs::io::mrec::schema::validate_path(Path::new(path)).map_err(molrs_error_to_pyerr)
}

/// Validate the mandatory ``meta`` version key against the mrec contract.
///
/// Args:
///     meta: Record-level metadata mapping.
///
/// Raises:
///     ValueError: If ``molrec_version`` is missing or not a version this
///         reader supports.
#[pyfunction]
pub fn mrec_validate_meta(meta: &Bound<'_, PyDict>) -> PyResult<()> {
    let map = dict_to_json_map(meta)?;
    molrs::io::mrec::schema::validate_meta(&map).map_err(molrs_error_to_pyerr)
}

/// Judge a snapshot or system-definition frame against the Frame vocabulary.
///
/// Args:
///     frame: In-memory :class:`~molrs.Frame`.
///
/// Raises:
///     ValueError: If the frame fails the canonical vocabulary.
#[pyfunction]
pub fn mrec_validate_frame(frame: &Bound<'_, PyFrame>) -> PyResult<()> {
    molrs::io::mrec::schema::validate_frame(&frame.borrow().clone_core_frame()?)
        .map_err(molrs_error_to_pyerr)
}

fn json_map_to_dict<'py>(
    py: Python<'py>,
    map: &JsonMap<String, JsonValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in map {
        dict.set_item(key, json_to_py(py, value)?)?;
    }
    Ok(dict)
}

fn json_to_py(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        JsonValue::Null => py.None(),
        JsonValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any().unbind()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any().unbind()
            } else {
                n.as_f64()
                    .unwrap_or(f64::NAN)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        JsonValue::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        JsonValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        JsonValue::Object(map) => json_map_to_dict(py, map)?.into_any().unbind(),
    })
}

fn dict_to_json_map(dict: &Bound<'_, PyDict>) -> PyResult<JsonMap<String, JsonValue>> {
    let mut map = JsonMap::new();
    for (key, value) in dict.iter() {
        let key: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("record metadata keys must be strings"))?;
        map.insert(key, py_to_json(&value)?);
    }
    Ok(map)
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if let Ok(b) = value.cast::<PyBool>() {
        return Ok(JsonValue::Bool(b.is_true()));
    }
    if let Ok(i) = value.cast::<PyInt>() {
        return Ok(JsonValue::from(i.extract::<i64>()?));
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        return Ok(serde_json::Number::from_f64(f.extract::<f64>()?)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null));
    }
    if let Ok(s) = value.cast::<PyString>() {
        return Ok(JsonValue::String(s.extract::<String>()?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return Ok(JsonValue::Object(dict_to_json_map(dict)?));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_json(&item)?);
        }
        return Ok(JsonValue::Array(items));
    }
    Err(PyTypeError::new_err(format!(
        "unsupported metadata value type: {}",
        value.get_type().name()?
    )))
}
