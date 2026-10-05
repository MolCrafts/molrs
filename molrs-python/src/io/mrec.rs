//! Scientific-record (`*.mrec`) path doors.
//!
//! `Frame` and `Trajectory` are the in-memory objects. The whole-record
//! functions are exported at `molrs.io` (`read_mrec`, `write_mrec`, …,
//! `read_mrec_meta`, `mrec_sections`); the store cursor and writer classes
//! ([`PyMrecTrajectoryReader`], [`PyMrecSequenceSchema`],
//! [`PyMrecTrajectoryWriter`]) and [`pack`] are `molrs.io.mrec`'s.

use std::path::PathBuf;

use crate::core::spatial::simbox::PyBox;
use crate::core::store::frame::{
    PyFrame, PyMetaValue, infer_meta_value, json_map_to_plain_dict, meta_document_arg,
    meta_value_from_dtype,
};
use crate::core::store::trajectory::PyTrajectory;
use crate::ff::section::PyForceFieldSection;
use crate::helpers::{molrs_error_to_pyerr, path_str};
use molrs::io::mrec::{
    Compression, FrameSequence, FrameSequenceWriter, SequenceSchema, column_dtype, open_packed,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFrozenSet};

/// Write a snapshot as a record whose only state section is ``frame``.
///
/// Args:
///     path: Destination filesystem path.
///     frame: In-memory :class:`~molrs.Frame` to persist.
///     system: Optional system-definition :class:`~molrs.Frame` written
///         beside the snapshot as the ``system/`` section.
///     meta: The record's identity document, written to ``meta/`` with
///         ``molrec_version`` stamped in: a ``dict``, a
///         :class:`~molrs.MetaDocument`, or any mapping (``frame.meta``
///         included). Nested tuples and documents are JSON arrays and objects.
///     forcefield: Optional force field written beside them as the
///         ``forcefield/`` section: a :class:`~molrs.io.mrec.ForceFieldSection`
///         as given, or a :class:`~molrs.ff.ForceField` through its
///         :meth:`~molrs.ff.ForceField.to_section`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, a section
///         fails to encode, or the force field is invalid or has no section
///         form.
///     TypeError: If ``meta`` is not a mapping or holds a value JSON cannot,
///         or ``forcefield`` is neither a ``ForceField`` nor a
///         ``ForceFieldSection``.
#[pyfunction]
#[pyo3(signature = (path, frame, system=None, meta=None, forcefield=None))]
pub fn write_mrec(
    path: PathBuf,
    frame: &Bound<'_, PyFrame>,
    system: Option<&Bound<'_, PyFrame>>,
    meta: Option<&Bound<'_, PyAny>>,
    forcefield: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut record = record_arg(meta, forcefield)?;
    record.frame = Some(frame.borrow().clone_core_frame()?);
    record.system = system.map(|s| s.borrow().clone_core_frame()).transpose()?;
    write_record(&path, &record)
}

/// Write a topology as a record whose only state section is ``system``.
///
/// Args:
///     path: Destination filesystem path.
///     system: In-memory :class:`~molrs.Frame` to persist as ``system/``.
///     meta: The record's identity document, written to ``meta/`` with
///         ``molrec_version`` stamped in: a ``dict``, a
///         :class:`~molrs.MetaDocument`, or any mapping (``frame.meta``
///         included). Nested tuples and documents are JSON arrays and objects.
///     forcefield: Optional force field the system's types link into,
///         written as the ``forcefield/`` section: a
///         :class:`~molrs.io.mrec.ForceFieldSection` or a
///         :class:`~molrs.ff.ForceField`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, a section
///         fails to encode, or the force field is invalid or has no section
///         form.
///     TypeError: If ``meta`` is not a mapping or holds a value JSON cannot,
///         or ``forcefield`` is neither a ``ForceField`` nor a
///         ``ForceFieldSection``.
#[pyfunction]
#[pyo3(signature = (path, system, meta=None, forcefield=None))]
pub fn write_mrec_system(
    path: PathBuf,
    system: &Bound<'_, PyFrame>,
    meta: Option<&Bound<'_, PyAny>>,
    forcefield: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut record = record_arg(meta, forcefield)?;
    record.system = Some(system.borrow().clone_core_frame()?);
    write_record(&path, &record)
}

/// Write a force field as a record whose only state section is
/// ``forcefield``: a force-field package (``meta`` + ``forcefield/``).
///
/// Args:
///     path: Destination filesystem path.
///     forcefield: A :class:`~molrs.io.mrec.ForceFieldSection`, written as
///         given, or a :class:`~molrs.ff.ForceField`, written through its
///         :meth:`~molrs.ff.ForceField.to_section`.
///     meta: The record's identity document (see :func:`write_mrec`).
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or the force
///         field is invalid or has no section form.
///     TypeError: If ``meta`` is not a mapping, or ``forcefield`` is neither a
///         ``ForceField`` nor a ``ForceFieldSection``.
#[pyfunction]
#[pyo3(signature = (path, forcefield, meta=None))]
pub fn write_mrec_forcefield(
    path: PathBuf,
    forcefield: &Bound<'_, PyAny>,
    meta: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let record = record_arg(meta, Some(forcefield))?;
    write_record(&path, &record)
}

/// A record holding only `meta` and the `forcefield` argument, for the
/// writing doors to add their state sections to.
fn record_arg(
    meta: Option<&Bound<'_, PyAny>>,
    forcefield: Option<&Bound<'_, PyAny>>,
) -> PyResult<molrs::MolRec> {
    let mut record = molrs::MolRec::new();
    if let Some(meta) = meta {
        record.meta = meta_document_arg(meta)?;
    }
    record.forcefield = forcefield.map(PyForceFieldSection::from_arg).transpose()?;
    Ok(record)
}

fn write_record(path: &std::path::Path, record: &molrs::MolRec) -> PyResult<()> {
    molrs::io::mrec::write_record_file(path_str(path)?, record).map_err(molrs_error_to_pyerr)
}

/// Write a trajectory as a record whose only state section is ``trajectory``.
///
/// Args:
///     path: Destination filesystem path.
///     traj: In-memory :class:`~molrs.Trajectory` to persist.
///     meta: The record's identity document, written to ``meta/`` with
///         ``molrec_version`` stamped in: a ``dict``, a
///         :class:`~molrs.MetaDocument`, or any mapping (``frame.meta``
///         included). Nested tuples and documents are JSON arrays and objects.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or a frame
///         fails to encode.
///     TypeError: If ``meta`` is not a mapping or holds a value JSON cannot.
#[pyfunction]
#[pyo3(signature = (path, traj, meta=None))]
pub fn write_mrec_trajectory(
    path: PathBuf,
    traj: PyRef<'_, PyTrajectory>,
    meta: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let path = path_str(&path)?;
    let meta_map = meta.map(meta_document_arg).transpose()?;
    molrs::io::mrec::write_trajectory_file(path, &traj.inner, meta_map.as_ref())
        .map_err(molrs_error_to_pyerr)
}

/// Read the ``frame`` section of a ``*.mrec`` store.
///
/// Only ``meta`` (for its version) and the ``frame`` section are decoded, so
/// another section — a trajectory, observables, one this build does not know —
/// cannot fail the read.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The in-memory :class:`~molrs.Frame`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, ``meta``
///         carries an unsupported ``molrec_version``, the store has no
///         ``frame`` section, or that section fails to decode.
#[pyfunction]
pub fn read_mrec(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::mrec::read_frame_file(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read the ``system`` section of a ``*.mrec`` store.
///
/// Only ``meta`` (for its version) and the ``system`` section are decoded.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The in-memory :class:`~molrs.Frame`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, ``meta``
///         carries an unsupported ``molrec_version``, the store has no
///         ``system`` section, or that section fails to decode.
#[pyfunction]
pub fn read_mrec_system(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::mrec::read_system_file(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read the ``trajectory`` section of a ``*.mrec`` store.
///
/// Only ``meta`` (for its version) and the ``trajectory`` section are
/// decoded. A store without one reads as an empty trajectory.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The in-memory :class:`~molrs.Trajectory`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, ``meta``
///         carries an unsupported ``molrec_version``, or the ``trajectory``
///         section fails to decode.
#[pyfunction]
pub fn read_mrec_trajectory(path: PathBuf) -> PyResult<PyTrajectory> {
    let path = path_str(&path)?;
    let inner = molrs::io::mrec::read_trajectory_file(path).map_err(molrs_error_to_pyerr)?;
    Ok(PyTrajectory { inner })
}

/// Read the ``meta`` document of a ``*.mrec`` store (empty when the producer
/// wrote none).
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The record-level metadata mapping, including ``molrec_version`` when
///     the writer stamped it.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or ``meta``
///         carries a ``molrec_version`` this reader does not support.
#[pyfunction]
pub fn read_mrec_meta(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyDict>> {
    let path = path_str(&path)?;
    let map = molrs::io::mrec::read_meta_file(path).map_err(molrs_error_to_pyerr)?;
    Ok(json_map_to_plain_dict(py, &map)?.unbind())
}

/// Read the ``forcefield`` section of a ``*.mrec`` store.
///
/// Only ``meta`` (for its version) and the ``forcefield`` section are
/// decoded. The section comes back whole — every document key, every table,
/// units as stored; :meth:`molrs.ff.ForceField.from_section` turns it into a
/// force field molrs can compile.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The :class:`~molrs.io.mrec.ForceFieldSection`, or ``None`` when the
///     record carries no force field.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, ``meta``
///         carries an unsupported ``molrec_version``, or the section is
///         malformed (the ``forcefield`` chapter's refusals).
#[pyfunction]
pub fn read_mrec_forcefield(path: PathBuf) -> PyResult<Option<PyForceFieldSection>> {
    let path = path_str(&path)?;
    let section = molrs::io::mrec::read_forcefield_file(path).map_err(molrs_error_to_pyerr)?;
    Ok(section.map(|inner| PyForceFieldSection { inner }))
}

/// Child group names at the record root (``meta``, ``frame``, ``system``, …).
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The section names, as a ``frozenset``. Callers ask which sections are
///     present instead of probing ``read_mrec`` / ``read_mrec_system`` and
///     catching a missing-section error.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, or the store
///         is not a readable record.
#[pyfunction]
pub fn mrec_sections(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyFrozenSet>> {
    let path = path_str(&path)?;
    let names = molrs::io::mrec::section_names(path).map_err(molrs_error_to_pyerr)?;
    PyFrozenSet::new(py, names)
}

/// Lazy one-frame cursor over a ``*.mrec`` trajectory (directory or ``.zip``).
///
/// Wraps the Rust ``FrameSequence`` store cursor: construction opens the
/// index only, and each read decodes exactly the asked-for frame, keeping the
/// last decoded chunk of every column so playback through consecutive frames
/// is a slice rather than a decode. Supports ``len(reader)``, ``reader[i]``
/// (negative indices included), iteration, and the ``with`` statement.
///
/// Args:
///     path: Filesystem path of the record store, or of a packed
///         ``*.mrec.zip``.
#[pyclass(module = "molrs.io.mrec", name = "TrajectoryReader", unsendable)]
pub struct PyMrecTrajectoryReader {
    inner: FrameSequence,
}

#[pymethods]
impl PyMrecTrajectoryReader {
    /// Open a ``*.mrec`` directory or a packed ``*.mrec.zip`` for reading.
    #[new]
    fn py_new(path: PathBuf) -> PyResult<Self> {
        let path = path_str(&path)?;
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

    /// Time of each committed frame, when the run wrote any. The record
    /// carries no unit for it; the producer's convention applies.
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
/// :class:`TrajectoryWriter` refuses a frame that steps outside it.
/// Declare it column by column, or derive it from representative frames.
#[pyclass(module = "molrs.io.mrec", name = "SequenceSchema")]
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
        let inner = frame
            .borrow()
            .with_frame(SequenceSchema::from_frame)?
            .map_err(molrs_error_to_pyerr)?;
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
    fn declare_block<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        rows: Option<u64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_block(name, rows)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare a column of ``block`` by dtype tag (``"f64"``, ``"u64"``,
    /// ``"string"``, …) and trailing shape (``[3]`` for an xyz column).
    #[pyo3(signature = (block, column, dtype, trailing=None))]
    fn declare_column<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        column: &str,
        dtype: &str,
        trailing: Option<Vec<u64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let dtype = column_dtype(dtype).map_err(molrs_error_to_pyerr)?;
        slf.inner
            .declare_column(block, column, dtype, &trailing.unwrap_or_default())
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare the precision of ``column`` of ``block``: an absolute tolerance
    /// in the column's units (``1e-3`` keeps Å coordinates to a thousandth).
    ///
    /// Every frame's values are rounded to the largest power of two not above
    /// it (ties to even) before the change check and before they land, so a
    /// change below half that step is no change, and the column is stored
    /// shuffled and compressed. Pinned with the schema. A schema derived with
    /// :meth:`from_frames` takes each column's :meth:`Block.precision`.
    fn declare_precision<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        column: &str,
        precision: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_precision(block, column, precision)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare that ``column`` of ``block`` (``u64``) holds row indices into
    /// *target*: ``"<block>"`` of the same resolved frame, or
    /// ``"/<section>/<block>"``. Pinned with the schema; the writer refuses a
    /// frame whose resolved blocks break it. :meth:`from_frames` takes each
    /// column's :meth:`Block.target`.
    fn declare_target<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        column: &str,
        target: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_target(block, column, target)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// The declared target of ``column`` of ``block``, or ``None``.
    fn target(&self, block: &str, column: &str) -> Option<String> {
        self.inner.target(block, column).map(str::to_string)
    }

    /// Declare ``block`` aligned with ``target``: its rows are ``target``'s
    /// rows, one for one, at every frame after carry-forward. A frame whose
    /// ``target`` changes row count must restate ``block``; one that keeps
    /// it may let ``block`` carry forward. The two blocks keep disjoint
    /// columns, alignments do not chain, and the aligned block declares no
    /// structural shape. A reader hands back the two blocks.
    fn declare_aligned<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        target: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_aligned(block, target)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// The block ``block`` is aligned with, or ``None``.
    fn aligned_with(&self, block: &str) -> Option<String> {
        self.inner.aligned_with(block).map(str::to_string)
    }

    /// The declared precision of ``column`` of ``block``, or ``None``.
    fn precision(&self, block: &str, column: &str) -> Option<f64> {
        self.inner.precision(block, column)
    }

    /// Declare the structural shape of ``block`` (a volumetric grid); every
    /// update then carries exactly ``prod(shape)`` rows.
    fn declare_structural_shape<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        shape: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_structural_shape(block, &shape)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare a per-step metadata key by dtype tag (``"f64"``, ``"i64"``,
    /// ``"f64x3"``, ``"string"``, ``"json"``, …). A frame that omits it is an
    /// error unless a fill is declared.
    fn declare_meta<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        dtype: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_meta(key, dtype)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare a per-step metadata key with the value written for steps that
    /// omit it. The dtype is ``dtype`` when given (``"f64x3"``, ``"f64"``, …;
    /// the fill is read at that width), else the fill's own.
    #[pyo3(signature = (key, fill, dtype=None))]
    fn declare_meta_with_fill<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        fill: &Bound<'_, PyAny>,
        dtype: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let value = match dtype {
            Some(dtype) if fill.extract::<PyRef<'_, PyMetaValue>>().is_err() => {
                meta_value_from_dtype(dtype, fill)
                    .map_err(|e| PyValueError::new_err(format!("meta key {key:?} fill: {e}")))?
            }
            _ => infer_meta_value(fill)?,
        };
        slf.inner
            .declare_meta_with_fill(key, value)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
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
///     schema: The :class:`SequenceSchema` every appended frame is checked
///         against.
///     flush_every: Land every this many frames instead of the derived cadence.
///     compression: How floating-point columns are compressed: ``None``,
///         ``"gzip[:level]"`` or ``"zstd[:level]"``. Everything else always
///         carries gzip level 1. A column with a declared precision is
///         byte-shuffled and compressed whatever this says: ``None`` means
///         zstd level 3, and a named compressor replaces it.
///     durable: Whether ``flush()`` / ``close()`` fsync the touched files.
///     meta: The record's identity document, written to ``meta/`` with
///         ``molrec_version`` stamped in: a ``dict``, a
///         :class:`~molrs.MetaDocument`, or any mapping (``frame.meta``
///         included). Nested tuples and documents are JSON arrays and objects.
#[pyclass(module = "molrs.io.mrec", name = "TrajectoryWriter", unsendable)]
pub struct PyMrecTrajectoryWriter {
    inner: Option<FrameSequenceWriter>,
}

#[pymethods]
impl PyMrecTrajectoryWriter {
    #[new]
    #[pyo3(signature = (path, schema, *, flush_every=None, compression=None, durable=true, meta=None))]
    fn py_new(
        path: PathBuf,
        schema: &Bound<'_, PyMrecSequenceSchema>,
        flush_every: Option<u64>,
        compression: Option<&str>,
        durable: bool,
        meta: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let path = path_str(&path)?;
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
                .with_meta(&meta_document_arg(meta)?)
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
    #[pyo3(signature = (path, *, flush_every=None, durable=true))]
    fn open(path: PathBuf, flush_every: Option<u64>, durable: bool) -> PyResult<Self> {
        let path = path_str(&path)?;
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
    /// pass ``step`` (and optionally ``time``, in the producer's own unit — the
    /// record stores none) for real MD numbering, or
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
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer is closed"))?;
        frame
            .borrow()
            .with_frame(|core| match (step, time) {
                (Some(step), time) => writer.append_at(core, step, time),
                (None, Some(time)) => writer.append_timed(core, time),
                (None, None) => writer.append(core),
            })?
            .map_err(molrs_error_to_pyerr)
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
pub fn pack(path: PathBuf) -> PyResult<String> {
    let path = path_str(&path)?;
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
pub fn mrec_validate_path(path: PathBuf) -> PyResult<()> {
    molrs::io::mrec::schema::validate_path(&path).map_err(molrs_error_to_pyerr)
}

/// Validate the ``meta`` version key against the mrec contract.
///
/// Every record is stamped on write, but a reader validates
/// ``molrec_version`` only when it is present: an absent key is no version
/// check, and a present one must be an integer in ``1..=MOLREC_VERSION``.
///
/// Args:
///     meta: Record-level metadata mapping.
///
/// Raises:
///     ValueError: If ``molrec_version`` is present and not a version this
///         reader supports.
#[pyfunction]
pub fn mrec_validate_meta(meta: &Bound<'_, PyAny>) -> PyResult<()> {
    molrs::io::mrec::schema::validate_meta(&meta_document_arg(meta)?).map_err(molrs_error_to_pyerr)
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
    frame
        .borrow()
        .with_frame(molrs::io::mrec::schema::validate_frame)?
        .map_err(molrs_error_to_pyerr)
}
