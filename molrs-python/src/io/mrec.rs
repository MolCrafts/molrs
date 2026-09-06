//! Scientific-record (`*.mrec`) path doors.
//!
//! `Frame` and `Trajectory` are the in-memory objects. These functions and
//! [`PyMrecTrajectoryReader`] are the filesystem doors, exported to Python as
//! `molrs.io.mrec`. The class is `MrecTrajectoryReader` on `_lib` so it does
//! not collide with the dump concatenator `molrs.io.TrajectoryReader`.

use std::path::Path;

use crate::core::spatial::simbox::PyBox;
use crate::core::store::frame::PyFrame;
use crate::core::store::frame::PyMetaValue;
use crate::core::store::trajectory::PyTrajectory;
use crate::helpers::molrs_error_to_pyerr;
use molrs::io::mrec::{
    Compression, FrameSequence, FrameSequenceWriter, SequenceSchema, column_dtype, open_packed,
};
use molrs::store::meta::MetaValue;
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
    /// Open a ``*.mrec`` directory or a packed ``*.mrec.zip`` for reading.
    #[new]
    fn py_new(path: &str) -> PyResult<Self> {
        let inner = if path.ends_with(".zip") {
            let store = open_packed(path).map_err(molrs_error_to_pyerr)?;
            FrameSequence::open(store).map_err(molrs_error_to_pyerr)?
        } else {
            molrs::io::mrec::open_trajectory_sequence(path).map_err(molrs_error_to_pyerr)?
        };
        Ok(Self { inner })
    }

    /// Decode one committed frame.
    ///
    /// Args:
    ///     index: Zero-based frame index; negative counts from the end.
    ///
    /// Returns:
    ///     The frame at ``index``.
    ///
    /// Raises:
    ///     IndexError: If ``index`` is past the commit marker.
    ///     ValueError: If a section fails to decode.
    fn read_frame(&self, index: isize) -> PyResult<PyFrame> {
        let resolved = self.resolve_index(index)?;
        let frame = self
            .inner
            .frame(resolved)
            .map_err(molrs_error_to_pyerr)?
            .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
        PyFrame::from_core_frame(frame)
    }

    /// Decode one frame carrying only the named ``(block, column)`` pairs.
    ///
    /// A viewer that needs coordinates decodes ``("atoms", "x")``,
    /// ``("atoms", "y")``, ``("atoms", "z")`` and nothing else. Blocks none of
    /// whose columns are named are left out; the cell and per-step metadata
    /// always come along.
    fn read_columns(&self, index: isize, columns: Vec<(String, String)>) -> PyResult<PyFrame> {
        let resolved = self.resolve_index(index)?;
        let pairs: Vec<(&str, &str)> = columns
            .iter()
            .map(|(block, column)| (block.as_str(), column.as_str()))
            .collect();
        let frame = self
            .inner
            .frame_columns(resolved, &pairs)
            .map_err(molrs_error_to_pyerr)?
            .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
        PyFrame::from_core_frame(frame)
    }

    /// The update of block ``name`` that frame ``index`` resolves to, or
    /// ``None`` when the block is absent there.
    ///
    /// Two consecutive frames resolving to the same update carry the same
    /// rows, so a consumer can skip re-uploading a block whose update did not
    /// change without comparing a single value.
    fn block_update_at(&self, name: &str, index: isize) -> PyResult<Option<u64>> {
        let resolved = self.resolve_index(index)?;
        self.inner
            .block_update_at(name, resolved)
            .map_err(molrs_error_to_pyerr)
    }

    /// The cell at frame ``index``, or ``None`` when no cell has been written
    /// at or before it.
    fn box_at(&self, index: isize) -> PyResult<Option<PyBox>> {
        let resolved = self.resolve_index(index)?;
        Ok(self
            .inner
            .box_at(resolved)
            .map_err(molrs_error_to_pyerr)?
            .map(|inner| PyBox { inner }))
    }

    /// Number of committed frames.
    fn __len__(&self) -> usize {
        self.inner.steps().len()
    }

    /// Frame at ``index``, with Python negative-from-the-end indexing.
    fn __getitem__(&self, index: isize) -> PyResult<PyFrame> {
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

    /// Whether the store carries a block section of this name (e.g. ``"bonds"``)
    /// with at least one committed update.
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
/// :class:`MrecTrajectoryWriter` refuses a frame that steps outside it.
/// Declare it column by column, or derive it from representative frames.
#[pyclass(module = "molrs.io.mrec", name = "MrecSequenceSchema")]
pub struct PyMrecSequenceSchema {
    inner: SequenceSchema,
}

#[pymethods]
impl PyMrecSequenceSchema {
    /// An empty declaration, to be filled with ``declare_*``.
    #[new]
    fn py_new() -> Self {
        Self {
            inner: SequenceSchema::new(),
        }
    }

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

    /// Declare a block, with the row count a typical frame of it carries.
    ///
    /// ``rows`` sizes the block's frame-aligned inner chunk; ``None`` leaves
    /// the byte floor to decide.
    #[pyo3(signature = (name, rows=None))]
    fn declare_block(&mut self, name: &str, rows: Option<u64>) -> PyResult<()> {
        self.inner
            .declare_block(name, rows)
            .map_err(molrs_error_to_pyerr)
    }

    /// Declare a column of ``block`` by dtype tag (``"f64"``, ``"u64"``,
    /// ``"string"``, …) and trailing shape (``[3]`` for an xyz column).
    #[pyo3(signature = (block, column, dtype, trailing=None))]
    fn declare_column(
        &mut self,
        block: &str,
        column: &str,
        dtype: &str,
        trailing: Option<Vec<u64>>,
    ) -> PyResult<()> {
        let dtype = column_dtype(dtype).map_err(molrs_error_to_pyerr)?;
        self.inner
            .declare_column(block, column, dtype, &trailing.unwrap_or_default())
            .map_err(molrs_error_to_pyerr)
    }

    /// Declare the structural shape of ``block`` (a volumetric grid); every
    /// update then carries exactly ``prod(shape)`` rows.
    fn declare_structural_shape(&mut self, block: &str, shape: Vec<usize>) -> PyResult<()> {
        self.inner
            .declare_structural_shape(block, &shape)
            .map_err(molrs_error_to_pyerr)
    }

    /// Declare a per-step metadata key by dtype tag (``"f64"``, ``"i64"``,
    /// ``"f64x3"``, ``"string"``, ``"json"``, …). A frame that omits it is an
    /// error unless a fill is declared.
    fn declare_meta(&mut self, key: &str, dtype: &str) -> PyResult<()> {
        self.inner
            .declare_meta(key, dtype)
            .map_err(molrs_error_to_pyerr)
    }

    /// Declare a per-step metadata key with the value written for steps that
    /// omit it. The dtype is ``dtype`` when given (``"f64x3"``, ``"f32"``, …;
    /// the fill is read at that width), else the fill's own.
    #[pyo3(signature = (key, fill, dtype=None))]
    fn declare_meta_with_fill(
        &mut self,
        key: &str,
        fill: &Bound<'_, PyAny>,
        dtype: Option<&str>,
    ) -> PyResult<()> {
        let value = if let Ok(meta) = fill.extract::<PyRef<'_, PyMetaValue>>() {
            meta.inner.clone()
        } else if let Some(dtype) = dtype {
            MetaValue::from_json_value(&serde_json::json!({
                "dtype": dtype,
                "value": py_to_json(fill)?,
            }))
            .map_err(|e| PyValueError::new_err(format!("meta key {key:?} fill: {e}")))?
        } else {
            MetaValue::from_attr_value(&py_to_json(fill)?)
        };
        self.inner
            .declare_meta_with_fill(key, value)
            .map_err(molrs_error_to_pyerr)
    }

    /// The declared block names.
    fn block_names(&self) -> Vec<String> {
        self.inner.block_names().map(str::to_string).collect()
    }

    /// The declared column names of ``block``, or ``None`` when undeclared.
    fn column_names(&self, block: &str) -> Option<Vec<String>> {
        self.inner
            .column_names(block)
            .map(|names| names.map(str::to_string).collect())
    }

    /// The declared per-step metadata keys with their dtype tags.
    fn meta_keys(&self) -> Vec<(String, String)> {
        self.inner
            .meta_keys()
            .map(|(key, dtype)| (key.to_string(), dtype.to_string()))
            .collect()
    }
}

/// Parse a compression spec: ``None`` / ``"none"``, ``"gzip"`` (level 1),
/// ``"gzip:5"``, ``"zstd"`` (level 3), ``"zstd:7"``.
fn parse_compression(spec: Option<&str>) -> PyResult<Compression> {
    let Some(spec) = spec else {
        return Ok(Compression::None);
    };
    let (name, level) = match spec.split_once(':') {
        Some((name, level)) => (name, Some(level)),
        None => (spec, None),
    };
    let level = |default: i64| -> PyResult<i64> {
        match level {
            Some(text) => text.parse::<i64>().map_err(|_| {
                PyValueError::new_err(format!("compression level {text:?} is not an integer"))
            }),
            None => Ok(default),
        }
    };
    match name {
        "none" => Ok(Compression::None),
        "gzip" => Ok(Compression::Gzip(level(1)? as u32)),
        "zstd" => Ok(Compression::Zstd(level(3)? as i32)),
        other => Err(PyValueError::new_err(format!(
            "unknown compression {other:?}; use None, \"gzip[:level]\" or \"zstd[:level]\""
        ))),
    }
}

/// Append-first writer for a ``*.mrec`` trajectory store.
///
/// Wraps the Rust ``FrameSequenceWriter`` over the positional-write store.
/// Frames are buffered and landed whole inner chunks at a time on a cadence
/// derived from the frame size (or ``flush_every``); ``flush()`` and
/// ``close()`` commit whatever is buffered, durably unless ``durable=False``.
///
/// Args:
///     path: Destination filesystem path for the store.
///     schema: The :class:`MrecSequenceSchema` every appended frame is checked
///         against.
///     flush_every: Land every this many frames instead of the derived cadence.
///     compression: How floating-point columns are compressed: ``None``,
///         ``"gzip[:level]"`` or ``"zstd[:level]"``. Everything else always
///         carries gzip level 1.
///     durable: Whether ``flush()`` / ``close()`` fsync the touched files.
///     meta: The record's identity document, written to ``meta/``.
#[pyclass(module = "molrs.io.mrec", name = "MrecTrajectoryWriter", unsendable)]
pub struct PyMrecTrajectoryWriter {
    inner: Option<FrameSequenceWriter>,
}

#[pymethods]
impl PyMrecTrajectoryWriter {
    #[new]
    #[pyo3(signature = (path, schema, flush_every=None, compression=None, durable=true, meta=None))]
    fn py_new(
        path: &str,
        schema: &Bound<'_, PyMrecSequenceSchema>,
        flush_every: Option<u64>,
        compression: Option<&str>,
        durable: bool,
        meta: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let schema = schema.borrow().inner.clone();
        let mut writer = FrameSequenceWriter::create_at(path, schema)
            .map_err(molrs_error_to_pyerr)?
            .with_compression(parse_compression(compression)?)
            .map_err(molrs_error_to_pyerr)?
            .with_durable(durable);
        if let Some(frames) = flush_every {
            writer = writer
                .with_flush_every(frames)
                .map_err(molrs_error_to_pyerr)?;
        }
        if let Some(meta) = meta {
            writer = writer
                .with_meta(&dict_to_json_map(meta)?)
                .map_err(molrs_error_to_pyerr)?;
        }
        Ok(Self {
            inner: Some(writer),
        })
    }

    /// Reattach to an existing store and continue appending after its last
    /// committed frame. Anything a crash left past the commit marker is
    /// rolled back first.
    #[staticmethod]
    #[pyo3(signature = (path, flush_every=None, durable=true))]
    fn open(path: &str, flush_every: Option<u64>, durable: bool) -> PyResult<Self> {
        let mut writer = FrameSequenceWriter::open_at(path)
            .map_err(molrs_error_to_pyerr)?
            .with_durable(durable);
        if let Some(frames) = flush_every {
            writer = writer
                .with_flush_every(frames)
                .map_err(molrs_error_to_pyerr)?;
        }
        Ok(Self {
            inner: Some(writer),
        })
    }

    /// Buffer a frame. With no ``step`` the writer numbers frames 0, 1, 2, …;
    /// pass ``step`` (and optionally ``time`` in fs) for real MD numbering, or
    /// ``time`` alone to keep the automatic numbering and still record times.
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
        match (step, time) {
            (Some(step), time) => writer
                .append_at(&core, step, time)
                .map_err(molrs_error_to_pyerr),
            (None, Some(time)) => writer
                .append_timed(&core, time)
                .map_err(molrs_error_to_pyerr),
            (None, None) => writer.append(&core).map_err(molrs_error_to_pyerr),
        }
    }

    /// Commit every buffered frame (durably, unless ``durable=False``).
    fn flush(&mut self) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer is closed"))?;
        writer.flush().map_err(molrs_error_to_pyerr)
    }

    /// Commit and release the store. Idempotent; appends after close raise.
    fn close(&mut self) -> PyResult<()> {
        if let Some(writer) = self.inner.take() {
            writer.close().map_err(molrs_error_to_pyerr)?;
        }
        Ok(())
    }

    /// The landing cadence in force, in frames.
    #[getter]
    fn flush_every(&self) -> PyResult<u64> {
        self.inner
            .as_ref()
            .map(FrameSequenceWriter::flush_every)
            .ok_or_else(|| PyValueError::new_err("writer is closed"))
    }

    /// Frames committed so far.
    #[getter]
    fn committed(&self) -> PyResult<u64> {
        self.inner
            .as_ref()
            .map(FrameSequenceWriter::committed)
            .ok_or_else(|| PyValueError::new_err("writer is closed"))
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

/// Pack a closed ``*.mrec`` directory into a sibling ``*.mrec.zip`` (every
/// entry stored, byte-identical to the files it replaces) and remove the
/// directory.
///
/// Returns:
///     The path of the archive.
#[pyfunction]
pub fn pack(path: &str) -> PyResult<String> {
    let archive = molrs::io::mrec::pack(path).map_err(molrs_error_to_pyerr)?;
    Ok(archive.to_string_lossy().into_owned())
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
