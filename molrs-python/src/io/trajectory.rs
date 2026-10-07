//! Trajectory formats (`molrs::io::trajectory`): the lazy, seekable native
//! readers (LAMMPS dump, DCD, TRR, XTC, and multi-frame XYZ) and the
//! multi-frame writers.
//!
//! The readers are private plumbing (`molrs._lib`): Python reaches them
//! through `molrs.io.read_*_trajectory`, which wraps one or several of them
//! in `molrs.io.trajectory.TrajectoryReader`.

use crate::core::frame::PyFrame;
use crate::error::io_error_to_pyerr;
use crate::path::path_str;
use molrs::io::data::xyz::XYZReader;
use molrs::io::reader::{FrameReader, ReadSeek, TrajectoryReader, open_seekable};
use molrs::io::trajectory::dcd::{DcdReader, open_dcd, write_dcd as write_dcd_rs};
use molrs::io::trajectory::lammps_dump::{
    LAMMPSTrajReader, open_lammps_dump, write_lammps_dump,
    write_lammps_dump_local as write_lammps_dump_local_rs,
};
use molrs::io::trajectory::trr::{TrrReader, open_trr, write_trr as write_trr_rs};
use molrs::io::trajectory::xtc::{XtcReader, open_xtc, write_xtc as write_xtc_rs};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};
use std::path::PathBuf;

// ============================================================================
// Shared trajectory-reader helpers
//
// `LAMMPSTrajReader` and `DCDTrajReader` wrap different concrete readers that
// both implement [`TrajectoryReader`]. These generics give both classes one
// consistent, molpy-aligned behaviour set (negative indexing, slicing, batch
// reads) without duplicating the logic per class.
// ============================================================================

/// Number of frames in the trajectory.
fn traj_len<R: TrajectoryReader>(inner: &mut R) -> PyResult<usize> {
    inner.len().map_err(io_error_to_pyerr)
}

/// Read a known in-bounds, non-negative index. Used by slice iteration where
/// the bounds are already resolved.
fn traj_read_idx<R: TrajectoryReader>(inner: &mut R, idx: isize) -> PyResult<PyFrame> {
    let frame = inner
        .read_step(idx as usize)
        .map_err(io_error_to_pyerr)?
        .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
    PyFrame::from_core_frame(frame)
}

/// Read a single frame, resolving Python-style negative indices and raising
/// `IndexError` if out of range.
fn traj_read_frame<R: TrajectoryReader>(inner: &mut R, index: isize) -> PyResult<PyFrame> {
    let n = traj_len(inner)? as isize;
    let idx = if index < 0 { index + n } else { index };
    if idx < 0 || idx >= n {
        return Err(PyIndexError::new_err("trajectory index out of range"));
    }
    traj_read_idx(inner, idx)
}

/// Read the frame at `step`, or `None` past the end — what `__next__` needs to
/// end an iteration.
fn traj_read_step<R: TrajectoryReader>(inner: &mut R, step: usize) -> PyResult<Option<PyFrame>> {
    match inner.read_step(step).map_err(io_error_to_pyerr)? {
        Some(f) => Ok(Some(PyFrame::from_core_frame(f)?)),
        None => Ok(None),
    }
}

/// Read an explicit list of (possibly negative) indices.
fn traj_read_frames<R: TrajectoryReader>(
    inner: &mut R,
    indices: Vec<isize>,
) -> PyResult<Vec<PyFrame>> {
    indices
        .into_iter()
        .map(|i| traj_read_frame(inner, i))
        .collect()
}

/// Iterate already-resolved `[start, stop)` bounds with `step` (the semantics
/// produced by Python's ``slice.indices``).
fn traj_slice<R: TrajectoryReader>(
    inner: &mut R,
    start: isize,
    stop: isize,
    step: isize,
) -> PyResult<Vec<PyFrame>> {
    let mut frames = Vec::new();
    let mut i = start;
    if step > 0 {
        while i < stop {
            frames.push(traj_read_idx(inner, i)?);
            i += step;
        }
    } else {
        while i > stop {
            frames.push(traj_read_idx(inner, i)?);
            i += step;
        }
    }
    Ok(frames)
}

/// `read_range(start, stop, step)` with Python-like normalization. A `None`
/// stop means "to the end" (or "to the start" for a negative step).
fn traj_read_range<R: TrajectoryReader>(
    inner: &mut R,
    start: isize,
    stop: Option<isize>,
    step: isize,
) -> PyResult<Vec<PyFrame>> {
    if step == 0 {
        return Err(PyValueError::new_err("read_range step must not be zero"));
    }
    let n = traj_len(inner)? as isize;
    let norm = |v: isize| -> isize { if v < 0 { (v + n).max(0) } else { v.min(n) } };
    let start = norm(start);
    let stop = match stop {
        Some(s) => norm(s),
        None => {
            if step > 0 {
                n
            } else {
                -1
            }
        }
    };
    traj_slice(inner, start, stop, step)
}

/// Read every frame.
fn traj_read_all<R: TrajectoryReader>(inner: &mut R) -> PyResult<Vec<PyFrame>> {
    let n = traj_len(inner)? as isize;
    traj_slice(inner, 0, n, 1)
}

/// `__getitem__` supporting both integer indices and slices. Returns a single
/// `Frame` for an integer key, or a `list[Frame]` for a slice key.
fn traj_getitem<R: TrajectoryReader>(inner: &mut R, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = key.py();
    if let Ok(slice) = key.cast::<PySlice>() {
        let n = traj_len(inner)?;
        let indices = slice.indices(n as isize)?;
        let frames = traj_slice(inner, indices.start, indices.stop, indices.step)?;
        Ok(PyList::new(py, frames)?.into_any().unbind())
    } else {
        let index: isize = key.extract()?;
        let frame = traj_read_frame(inner, index)?;
        Ok(Py::new(py, frame)?.into_any())
    }
}

/// Lazy, indexed reader for LAMMPS dump trajectory files.
///
/// Unlike :func:`read_lammps_trajectory`, this does **not** parse every frame
/// upfront. The underlying file stays open and frames are parsed on demand
/// via byte-offset seeks. Random access (``reader[i]``, ``read_step(i)``)
/// triggers a one-time index scan for ``ITEM: TIMESTEP`` markers; subsequent
/// accesses are O(1) seeks plus one frame parse.
///
/// Use this for long trajectories where you only need a subset of frames
/// or want to walk lazily without holding all frames in memory.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS dump file (``.lammpstrj``). Gzip files are
///     auto-detected by extension and decompressed into memory.
///
/// Private: Python reaches it through :func:`molrs.io.read_lammps_trajectory`, which
/// wraps one per file in a :class:`molrs.io.trajectory.TrajectoryReader`.
#[pyclass(module = "molrs._lib", name = "LAMMPSTrajReader", unsendable)]
pub struct PyLAMMPSTrajReader {
    inner: Option<LAMMPSTrajReader<Box<dyn ReadSeek>>>,
    cursor: usize,
}

impl PyLAMMPSTrajReader {
    fn reader(&mut self) -> PyResult<&mut LAMMPSTrajReader<Box<dyn ReadSeek>>> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("operation on a closed LAMMPSTrajReader"))
    }
}

#[pymethods]
impl PyLAMMPSTrajReader {
    #[new]
    fn py_new(path: PathBuf) -> PyResult<Self> {
        let path = path_str(&path)?;
        let inner = open_lammps_dump(path).map_err(io_error_to_pyerr)?;
        Ok(Self {
            inner: Some(inner),
            cursor: 0,
        })
    }

    /// Number of frames in the trajectory (triggers index construction).
    #[getter]
    fn n_frames(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    /// Force the byte-offset index to be built now.
    ///
    /// The index is built lazily on the first call to ``__len__``,
    /// ``__getitem__``, or ``read_step``. Call this explicitly to amortize
    /// the cost upfront — useful when timing only the random-access path.
    fn build_index(&mut self) -> PyResult<()> {
        self.reader()?.build_index().map_err(io_error_to_pyerr)
    }

    /// Read a single frame by index (supports negative indexing).
    ///
    /// Raises ``IndexError`` if out of range. molpy-aligned.
    fn read_frame(&mut self, index: isize) -> PyResult<PyFrame> {
        traj_read_frame(self.reader()?, index)
    }

    /// Read an explicit list of frame indices (each may be negative).
    fn read_frames(&mut self, indices: Vec<isize>) -> PyResult<Vec<PyFrame>> {
        traj_read_frames(self.reader()?, indices)
    }

    /// Read a contiguous range of frames, Python-slice style.
    #[pyo3(signature = (start=0, stop=None, step=1))]
    fn read_range(
        &mut self,
        start: isize,
        stop: Option<isize>,
        step: isize,
    ) -> PyResult<Vec<PyFrame>> {
        traj_read_range(self.reader()?, start, stop, step)
    }

    /// Eagerly read every frame into a list.
    fn read_all(&mut self) -> PyResult<Vec<PyFrame>> {
        traj_read_all(self.reader()?)
    }

    /// Release the underlying file handle. Further reads raise ``ValueError``.
    fn close(&mut self) {
        self.inner = None;
    }

    fn __len__(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    fn __getitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        traj_getitem(self.reader()?, key)
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        // Reset cursor each time iter() is requested so re-iteration works.
        let mut slf = slf;
        slf.cursor = 0;
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
        let cursor = self.cursor;
        let frame = traj_read_step(self.reader()?, cursor)?;
        if frame.is_some() {
            self.cursor += 1;
        }
        Ok(frame)
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
    ) -> bool {
        self.inner = None;
        false
    }

    fn __repr__(&mut self) -> String {
        match self.inner.as_mut() {
            Some(r) => match r.len() {
                Ok(n) => format!("LAMMPSTrajReader(n_frames={})", n),
                Err(_) => "LAMMPSTrajReader(<unread>)".to_string(),
            },
            None => "LAMMPSTrajReader(<closed>)".to_string(),
        }
    }
}

/// Lazy, indexed reader for DCD trajectory files.
///
/// Unlike :func:`read_dcd_trajectory`, this does **not** load every frame upfront. The
/// underlying file stays open and frames are parsed on demand via byte-offset
/// seeks computed from the DCD header. The header is parsed lazily on the
/// first call to ``__len__``, ``__getitem__``, or ``read_step`` (or eagerly
/// via ``build_index()``); subsequent random access (``reader[i]``,
/// ``read_step(i)``) is an O(1) seek plus one frame parse.
///
/// Use this for long trajectories where you only need a subset of frames or
/// want to walk lazily without holding all frames in memory.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.dcd`` file.
///
/// Private: Python reaches it through :func:`molrs.io.read_dcd_trajectory`, which
/// wraps one per file in a :class:`molrs.io.trajectory.TrajectoryReader`.
#[pyclass(module = "molrs._lib", name = "DCDTrajReader", unsendable)]
pub struct PyDcdTrajReader {
    inner: Option<DcdReader<Box<dyn ReadSeek>>>,
    cursor: usize,
}

impl PyDcdTrajReader {
    fn reader(&mut self) -> PyResult<&mut DcdReader<Box<dyn ReadSeek>>> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("operation on a closed DCDTrajReader"))
    }
}

#[pymethods]
impl PyDcdTrajReader {
    #[new]
    fn py_new(path: PathBuf) -> PyResult<Self> {
        let path = path_str(&path)?;
        let inner = open_dcd(path).map_err(io_error_to_pyerr)?;
        Ok(Self {
            inner: Some(inner),
            cursor: 0,
        })
    }

    /// Number of frames in the trajectory (triggers header parsing).
    #[getter]
    fn n_frames(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    /// Force the DCD header to be parsed now.
    ///
    /// The header is parsed lazily on the first call to ``__len__``,
    /// ``__getitem__``, or ``read_step``. Call this explicitly to amortize
    /// the cost upfront — useful when timing only the random-access path.
    fn build_index(&mut self) -> PyResult<()> {
        self.reader()?.build_index().map_err(io_error_to_pyerr)
    }

    /// Read a single frame by index (supports negative indexing).
    ///
    /// Raises ``IndexError`` if out of range. molpy-aligned.
    fn read_frame(&mut self, index: isize) -> PyResult<PyFrame> {
        traj_read_frame(self.reader()?, index)
    }

    /// Read an explicit list of frame indices (each may be negative).
    fn read_frames(&mut self, indices: Vec<isize>) -> PyResult<Vec<PyFrame>> {
        traj_read_frames(self.reader()?, indices)
    }

    /// Read a contiguous range of frames, Python-slice style.
    #[pyo3(signature = (start=0, stop=None, step=1))]
    fn read_range(
        &mut self,
        start: isize,
        stop: Option<isize>,
        step: isize,
    ) -> PyResult<Vec<PyFrame>> {
        traj_read_range(self.reader()?, start, stop, step)
    }

    /// Eagerly read every frame into a list.
    fn read_all(&mut self) -> PyResult<Vec<PyFrame>> {
        traj_read_all(self.reader()?)
    }

    /// Release the underlying file handle. Further reads raise ``ValueError``.
    fn close(&mut self) {
        self.inner = None;
    }

    fn __len__(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    fn __getitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        traj_getitem(self.reader()?, key)
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        // Reset cursor each time iter() is requested so re-iteration works.
        let mut slf = slf;
        slf.cursor = 0;
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
        let cursor = self.cursor;
        let frame = traj_read_step(self.reader()?, cursor)?;
        if frame.is_some() {
            self.cursor += 1;
        }
        Ok(frame)
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
    ) -> bool {
        self.inner = None;
        false
    }

    fn __repr__(&mut self) -> String {
        match self.inner.as_mut() {
            Some(r) => match r.len() {
                Ok(n) => format!("DCDTrajReader(n_frames={})", n),
                Err(_) => "DCDTrajReader(<unread>)".to_string(),
            },
            None => "DCDTrajReader(<closed>)".to_string(),
        }
    }
}

/// Lazy, indexed reader for multi-frame XYZ trajectory files.
///
/// Frames are parsed on demand; the frame-offset index is built lazily on
/// first random access or eagerly via ``build_index()``. Exposes the same
/// surface as the DCD and LAMMPS dump readers: ``read_frame``,
/// ``read_frames``, ``read_range``, ``read_all``, ``n_frames``, slicing,
/// ``close()``, and context-manager use.
///
/// Parameters
/// ----------
/// path : str
///     Path to a multi-frame ``.xyz`` file.
///
/// Private: Python reaches it through :func:`molrs.io.read_xyz_trajectory`,
/// which wraps one per file in a :class:`molrs.io.trajectory.TrajectoryReader`.
#[pyclass(module = "molrs._lib", name = "XYZTrajReader", unsendable)]
pub struct PyXYZTrajReader {
    inner: Option<XYZReader<Box<dyn ReadSeek>>>,
    cursor: usize,
}

impl PyXYZTrajReader {
    fn reader(&mut self) -> PyResult<&mut XYZReader<Box<dyn ReadSeek>>> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("operation on a closed XYZTrajReader"))
    }
}

#[pymethods]
impl PyXYZTrajReader {
    #[new]
    fn py_new(path: PathBuf) -> PyResult<Self> {
        let path = path_str(&path)?;
        let reader = open_seekable(path).map_err(io_error_to_pyerr)?;
        Ok(Self {
            inner: Some(XYZReader::new(reader)),
            cursor: 0,
        })
    }

    /// Number of frames in the trajectory (triggers index construction).
    #[getter]
    fn n_frames(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    /// Force the frame-offset index to be built now.
    fn build_index(&mut self) -> PyResult<()> {
        self.reader()?.build_index().map_err(io_error_to_pyerr)
    }

    /// Read a single frame by index (supports negative indexing).
    ///
    /// Raises ``IndexError`` if out of range. molpy-aligned.
    fn read_frame(&mut self, index: isize) -> PyResult<PyFrame> {
        traj_read_frame(self.reader()?, index)
    }

    /// Read an explicit list of frame indices (each may be negative).
    fn read_frames(&mut self, indices: Vec<isize>) -> PyResult<Vec<PyFrame>> {
        traj_read_frames(self.reader()?, indices)
    }

    /// Read a contiguous range of frames, Python-slice style.
    #[pyo3(signature = (start=0, stop=None, step=1))]
    fn read_range(
        &mut self,
        start: isize,
        stop: Option<isize>,
        step: isize,
    ) -> PyResult<Vec<PyFrame>> {
        traj_read_range(self.reader()?, start, stop, step)
    }

    /// Eagerly read every frame into a list.
    fn read_all(&mut self) -> PyResult<Vec<PyFrame>> {
        traj_read_all(self.reader()?)
    }

    /// Release the underlying file handle. Further reads raise ``ValueError``.
    fn close(&mut self) {
        self.inner = None;
    }

    fn __len__(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    fn __getitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        traj_getitem(self.reader()?, key)
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        let mut slf = slf;
        slf.cursor = 0;
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
        let cursor = self.cursor;
        let frame = traj_read_step(self.reader()?, cursor)?;
        if frame.is_some() {
            self.cursor += 1;
        }
        Ok(frame)
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
    ) -> bool {
        self.inner = None;
        false
    }

    fn __repr__(&mut self) -> String {
        match self.inner.as_mut() {
            Some(r) => match r.len() {
                Ok(n) => format!("XYZTrajReader(n_frames={})", n),
                Err(_) => "XYZTrajReader(<unread>)".to_string(),
            },
            None => "XYZTrajReader(<closed>)".to_string(),
        }
    }
}

/// Write Frames to a LAMMPS dump trajectory file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
///     Frames to write.
/// columns : list[str], optional
///     The ``dump custom`` column line, e.g. ``["id", "element", "mol", "x",
///     "y", "z"]``. Written in the order given; a name the frame's ``atoms``
///     block cannot supply raises. Default writes every column it holds.
///
/// Notes
/// -----
/// The dump's ``type`` field is ``type_id`` when the block has it, otherwise
/// the string ``type`` labels (which read back as ``type``). With both, only
/// ``type_id`` is written. Values are formatted from each column's stored
/// dtype; a complex column, one with more than one value per row, or a string
/// that is empty or contains whitespace raises.
#[pyfunction]
#[pyo3(signature = (path, frames, columns = None))]
pub fn write_lammps_trajectory(
    path: PathBuf,
    frames: Vec<PyRef<'_, PyFrame>>,
    columns: Option<Vec<String>>,
) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    let chosen: Option<Vec<&str>> = columns
        .as_ref()
        .map(|c| c.iter().map(String::as_str).collect());
    write_lammps_dump(path, &core_frames, chosen.as_deref()).map_err(io_error_to_pyerr)
}

/// Write Frames as LAMMPS ``dump local`` (OVITO Load Trajectory bonds).
///
/// Emits ``ITEM: NUMBER OF ENTRIES`` + ``ITEM: ENTRIES batom1 batom2 [btype]``.
/// Rows come from ``entries`` if present, otherwise from canonical ``bonds``.
#[pyfunction]
pub fn write_lammps_dump_local(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_lammps_dump_local_rs(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Write Frames to a DCD trajectory file.
///
/// Produces a NAMD-compatible little-endian DCD. Every frame must have the
/// same atom count and the same box presence as the first frame. The box, if
/// any, is taken from each ``frame.box``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
///     Frames to write. Must be non-empty and homogeneous in atom count.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be written, or a frame uses an unsupported feature
///     (e.g. 4D dynamics / fixed atoms).
#[pyfunction]
pub fn write_dcd_trajectory(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_dcd_rs(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Lazy, indexed reader for GROMACS TRR trajectory files.
///
/// Builds a per-frame byte-offset index on first random access (or eagerly via
/// ``build_index()``); subsequent ``reader[i]`` / ``read_step(i)`` is an O(1)
/// seek plus one frame parse. Exposes the same surface as the DCD reader.
/// Private: Python reaches it through :func:`molrs.io.read_trr_trajectory`.
#[pyclass(module = "molrs._lib", name = "TRRTrajReader", unsendable)]
pub struct PyTrrTrajReader {
    inner: Option<TrrReader<Box<dyn ReadSeek>>>,
    cursor: usize,
}

impl PyTrrTrajReader {
    fn reader(&mut self) -> PyResult<&mut TrrReader<Box<dyn ReadSeek>>> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("operation on a closed TRRTrajReader"))
    }
}

#[pymethods]
impl PyTrrTrajReader {
    #[new]
    fn py_new(path: PathBuf) -> PyResult<Self> {
        let path = path_str(&path)?;
        let inner = open_trr(path).map_err(io_error_to_pyerr)?;
        Ok(Self {
            inner: Some(inner),
            cursor: 0,
        })
    }

    /// Number of frames in the trajectory (triggers index construction).
    #[getter]
    fn n_frames(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    /// Force the frame-offset index to be built now.
    fn build_index(&mut self) -> PyResult<()> {
        self.reader()?.build_index().map_err(io_error_to_pyerr)
    }

    /// Read a single frame by index (supports negative indexing).
    fn read_frame(&mut self, index: isize) -> PyResult<PyFrame> {
        traj_read_frame(self.reader()?, index)
    }

    /// Read an explicit list of frame indices (each may be negative).
    fn read_frames(&mut self, indices: Vec<isize>) -> PyResult<Vec<PyFrame>> {
        traj_read_frames(self.reader()?, indices)
    }

    /// Read a contiguous range of frames, Python-slice style.
    #[pyo3(signature = (start=0, stop=None, step=1))]
    fn read_range(
        &mut self,
        start: isize,
        stop: Option<isize>,
        step: isize,
    ) -> PyResult<Vec<PyFrame>> {
        traj_read_range(self.reader()?, start, stop, step)
    }

    /// Eagerly read every frame into a list.
    fn read_all(&mut self) -> PyResult<Vec<PyFrame>> {
        traj_read_all(self.reader()?)
    }

    /// Release the underlying file handle. Further reads raise ``ValueError``.
    fn close(&mut self) {
        self.inner = None;
    }

    fn __len__(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    fn __getitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        traj_getitem(self.reader()?, key)
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let mut slf = slf;
        slf.cursor = 0;
        // True sequential pass: rewind without forcing a full index scan.
        if let Some(r) = slf.inner.as_mut() {
            r.rewind().map_err(io_error_to_pyerr)?;
        }
        Ok(slf)
    }

    fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
        // FrameReader::read streams without building the offset index.
        // (traj_read_step would force ensure_index → double I/O.)
        match self.reader()?.read().map_err(io_error_to_pyerr)? {
            Some(f) => {
                self.cursor += 1;
                Ok(Some(PyFrame::from_core_frame(f)?))
            }
            None => Ok(None),
        }
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
    ) -> bool {
        self.inner = None;
        false
    }

    fn __repr__(&mut self) -> String {
        match self.inner.as_mut() {
            // Avoid forcing a full-file index scan just for repr.
            Some(_) => "TRRTrajReader(<lazy>)".to_string(),
            None => "TRRTrajReader(<closed>)".to_string(),
        }
    }
}

/// Lazy, indexed reader for GROMACS XTC trajectory files.
///
/// Like the TRR reader but for the compressed XTC format. Frame sizes vary
/// (compression), so the byte-offset index is built by a single scan; random
/// access is O(1) thereafter. Private: Python reaches it through
/// :func:`molrs.io.read_xtc_trajectory`.
#[pyclass(module = "molrs._lib", name = "XTCTrajReader", unsendable)]
pub struct PyXtcTrajReader {
    inner: Option<XtcReader<Box<dyn ReadSeek>>>,
    cursor: usize,
}

impl PyXtcTrajReader {
    fn reader(&mut self) -> PyResult<&mut XtcReader<Box<dyn ReadSeek>>> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("operation on a closed XTCTrajReader"))
    }
}

#[pymethods]
impl PyXtcTrajReader {
    #[new]
    fn py_new(path: PathBuf) -> PyResult<Self> {
        let path = path_str(&path)?;
        let inner = open_xtc(path).map_err(io_error_to_pyerr)?;
        Ok(Self {
            inner: Some(inner),
            cursor: 0,
        })
    }

    /// Number of frames in the trajectory (triggers index construction).
    #[getter]
    fn n_frames(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    /// Force the frame-offset index to be built now.
    fn build_index(&mut self) -> PyResult<()> {
        self.reader()?.build_index().map_err(io_error_to_pyerr)
    }

    /// Read a single frame by index (supports negative indexing).
    fn read_frame(&mut self, index: isize) -> PyResult<PyFrame> {
        traj_read_frame(self.reader()?, index)
    }

    /// Read an explicit list of frame indices (each may be negative).
    fn read_frames(&mut self, indices: Vec<isize>) -> PyResult<Vec<PyFrame>> {
        traj_read_frames(self.reader()?, indices)
    }

    /// Read a contiguous range of frames, Python-slice style.
    #[pyo3(signature = (start=0, stop=None, step=1))]
    fn read_range(
        &mut self,
        start: isize,
        stop: Option<isize>,
        step: isize,
    ) -> PyResult<Vec<PyFrame>> {
        traj_read_range(self.reader()?, start, stop, step)
    }

    /// Eagerly read every frame into a list.
    fn read_all(&mut self) -> PyResult<Vec<PyFrame>> {
        traj_read_all(self.reader()?)
    }

    /// Release the underlying file handle. Further reads raise ``ValueError``.
    fn close(&mut self) {
        self.inner = None;
    }

    fn __len__(&mut self) -> PyResult<usize> {
        traj_len(self.reader()?)
    }

    fn __getitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        traj_getitem(self.reader()?, key)
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let mut slf = slf;
        slf.cursor = 0;
        if let Some(r) = slf.inner.as_mut() {
            r.rewind().map_err(io_error_to_pyerr)?;
        }
        Ok(slf)
    }

    fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
        match self.reader()?.read().map_err(io_error_to_pyerr)? {
            Some(f) => {
                self.cursor += 1;
                Ok(Some(PyFrame::from_core_frame(f)?))
            }
            None => Ok(None),
        }
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
    ) -> bool {
        self.inner = None;
        false
    }

    fn __repr__(&mut self) -> String {
        match self.inner.as_mut() {
            Some(_) => "XTCTrajReader(<lazy>)".to_string(),
            None => "XTCTrajReader(<closed>)".to_string(),
        }
    }
}

/// Write Frames to a GROMACS TRR trajectory file (single precision).
///
/// Each frame's ``"atoms"`` block must have ``x``/``y``/``z`` (nm); optional
/// ``vx``/``vy``/``vz`` and ``fx``/``fy``/``fz`` are written when present. The
/// box, if any, is taken from each ``frame.box``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
#[pyfunction]
pub fn write_trr_trajectory(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_trr_rs(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Write Frames to a GROMACS XTC trajectory file (lossy compression).
///
/// Each frame's ``"atoms"`` block must have ``x``/``y``/``z`` (nm). The
/// quantization precision is taken from ``frame.meta["precision"]`` when
/// present, else defaults to 1000 (i.e. 0.001 nm resolution). The box, if any,
/// is taken from each ``frame.box``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
#[pyfunction]
pub fn write_xtc_trajectory(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_xtc_rs(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Register this module's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLAMMPSTrajReader>()?;
    m.add_class::<PyDcdTrajReader>()?;
    m.add_class::<PyXYZTrajReader>()?;
    m.add_class::<PyTrrTrajReader>()?;
    m.add_class::<PyXtcTrajReader>()?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_lammps_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_lammps_dump_local, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_dcd_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_trr_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xtc_trajectory, m)?)?;
    Ok(())
}
