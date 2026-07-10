//! I/O functions for reading and writing molecular data files, and parsing
//! SMILES notation.
//!
//! ## Supported formats
//!
//! | Format | Read | Write |
//! |--------|------|-------|
//! | PDB | [`read_pdb`] | [`write_pdb`] |
//! | XYZ | [`read_xyz`], [`read_xyz_traj`] | [`write_xyz`] |
//! | LAMMPS data | [`read_lammps`] | [`write_lammps`] |
//! | LAMMPS dump | [`read_lammps_traj`] | [`write_lammps_traj`] |
//! | DCD | [`read_dcd`], [`PyDcdTrajReader`] | [`write_dcd`] |
//! | GRO | [`read_gro`] | [`write_gro`] |

use crate::core::store::frame::PyFrame;
use crate::core::system::molgraph::PyAtomistic;
use crate::helpers::{io_error_to_pyerr, molrs_error_to_pyerr, smiles_error_to_pyerr};
use molrs::io::data::chgcar::read_chgcar;
use molrs::io::data::cube::{read_cube, write_cube};
use molrs::io::data::gro::{read_gro as read_gro_rs, write_gro as write_gro_rs};
use molrs::io::data::lammps_data::{read_lammps_data, write_lammps_data};
use molrs::io::data::pdb::{read_pdb_frame, read_pdb_traj, write_pdb_frame, write_pdb_traj};
use molrs::io::data::xyz::{XYZReader, read_xyz_frame, read_xyz_traj, write_xyz_frame};
use molrs::io::reader::{ReadSeek, TrajReader, open_seekable};
use molrs::io::trajectory::dcd::{
    DcdReader, open_dcd, read_dcd as read_dcd_rs, write_dcd as write_dcd_rs,
};
use molrs::io::trajectory::lammps_dump::{
    LAMMPSTrajReader, open_lammps_dump, read_lammps_dump, write_lammps_dump,
};
use molrs::io::trajectory::trr::{
    TrrReader, open_trr, read_trr as read_trr_rs, write_trr as write_trr_rs,
};
use molrs::io::trajectory::xtc::{
    XtcReader, open_xtc, read_xtc as read_xtc_rs, write_xtc as write_xtc_rs,
};
use molrs::store::frame::Frame as CoreFrame;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};
use std::fs::File;
use std::io::BufWriter;

/// Read a PDB file and return a Frame.
///
/// The resulting frame contains an ``"atoms"`` block with columns ``symbol``
/// (str), ``x``/``y``/``z`` (float), ``name`` (str), ``resname`` (str), and
/// ``resid`` (int). If CRYST1 records are present a ``Box`` is also attached.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.pdb`` file on disk.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frame = molrs.read_pdb("molecule.pdb")
/// >>> atoms = frame["atoms"]
/// >>> symbols = atoms.view("symbol")
#[pyfunction]
pub fn read_pdb(path: &str) -> PyResult<PyFrame> {
    let frame = read_pdb_frame(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read every MODEL of a PDB file as a trajectory (one Frame per MODEL).
///
/// A single-model (or MODEL-less) PDB returns a one-element list.
///
/// Parameters
/// ----------
/// path : str
///     Path to a (possibly multi-MODEL) ``.pdb`` file.
///
/// Returns
/// -------
/// list[Frame]
#[pyfunction]
pub fn read_pdb_trajectory(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_pdb_traj(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Read an XYZ file and return a single Frame.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xyz`` file on disk.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_xyz(path: &str) -> PyResult<PyFrame> {
    let frame = read_xyz_frame(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read all frames from an XYZ trajectory file.
///
/// Parameters
/// ----------
/// path : str
///     Path to a multi-frame ``.xyz`` file.
///
/// Returns
/// -------
/// list[Frame]
#[pyfunction]
pub fn read_xyz_trajectory(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_xyz_traj(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Read a LAMMPS data file and return a Frame.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS data file on disk.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data with atoms, bonds, and box metadata.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frame = molrs.read_lammps_data("system.data")
/// >>> atoms = frame["atoms"]
#[pyfunction]
pub fn read_lammps(path: &str) -> PyResult<PyFrame> {
    let frame = read_lammps_data(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a LAMMPS dump trajectory file and return a list of Frames.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS dump file (e.g. ``.lammpstrj``) on disk.
///
/// Returns
/// -------
/// list[Frame]
///     All frames in the trajectory.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frames = molrs.read_lammps_dump("trajectory.lammpstrj")
/// >>> len(frames)
/// 100
#[pyfunction]
pub fn read_lammps_traj(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_lammps_dump(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

// ============================================================================
// Shared trajectory-reader helpers
//
// `LAMMPSTrajReader` and `DCDTrajReader` wrap different concrete readers that
// both implement [`TrajReader`]. These generics give both classes one
// consistent, molpy-aligned behaviour set (negative indexing, slicing, batch
// reads) without duplicating the logic per class.
// ============================================================================

/// Number of frames in the trajectory.
fn traj_len<R: TrajReader<Frame = CoreFrame>>(inner: &mut R) -> PyResult<usize> {
    inner.len().map_err(io_error_to_pyerr)
}

/// Read a known in-bounds, non-negative index. Used by slice iteration where
/// the bounds are already resolved.
fn traj_read_idx<R: TrajReader<Frame = CoreFrame>>(inner: &mut R, idx: isize) -> PyResult<PyFrame> {
    let frame = inner
        .read_step(idx as usize)
        .map_err(io_error_to_pyerr)?
        .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
    PyFrame::from_core_frame(frame)
}

/// Read a single frame, resolving Python-style negative indices and raising
/// `IndexError` if out of range.
fn traj_read_frame<R: TrajReader<Frame = CoreFrame>>(
    inner: &mut R,
    index: isize,
) -> PyResult<PyFrame> {
    let n = traj_len(inner)? as isize;
    let idx = if index < 0 { index + n } else { index };
    if idx < 0 || idx >= n {
        return Err(PyIndexError::new_err("trajectory index out of range"));
    }
    traj_read_idx(inner, idx)
}

/// Read a frame by step index, returning ``None`` when out of bounds (lenient
/// variant retained for backward compatibility).
fn traj_read_step<R: TrajReader<Frame = CoreFrame>>(
    inner: &mut R,
    step: usize,
) -> PyResult<Option<PyFrame>> {
    match inner.read_step(step).map_err(io_error_to_pyerr)? {
        Some(f) => Ok(Some(PyFrame::from_core_frame(f)?)),
        None => Ok(None),
    }
}

/// Read an explicit list of (possibly negative) indices.
fn traj_read_frames<R: TrajReader<Frame = CoreFrame>>(
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
fn traj_slice<R: TrajReader<Frame = CoreFrame>>(
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
fn traj_read_range<R: TrajReader<Frame = CoreFrame>>(
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
fn traj_read_all<R: TrajReader<Frame = CoreFrame>>(inner: &mut R) -> PyResult<Vec<PyFrame>> {
    let n = traj_len(inner)? as isize;
    traj_slice(inner, 0, n, 1)
}

/// `__getitem__` supporting both integer indices and slices. Returns a single
/// `Frame` for an integer key, or a `list[Frame]` for a slice key.
fn traj_getitem<R: TrajReader<Frame = CoreFrame>>(
    inner: &mut R,
    key: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
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
/// Unlike :func:`read_lammps_traj`, this does **not** parse every frame
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
/// Examples
/// --------
/// >>> reader = molrs.LAMMPSTrajReader("trajectory.lammpstrj")
/// >>> len(reader)
/// 1000
/// >>> frame = reader[42]
/// >>> for frame in reader:
/// ...     pass
#[pyclass(name = "LAMMPSTrajReader", unsendable)]
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
    fn py_new(path: &str) -> PyResult<Self> {
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

/// Read every frame of a DCD trajectory file and return a list of Frames.
///
/// DCD is the binary trajectory format used by CHARMM, NAMD, and LAMMPS.
/// Each frame contains an ``"atoms"`` block with ``x``/``y``/``z`` columns
/// (Å). Unit-cell information, when present, is stored in ``frame.simbox``;
/// per-frame ``timestep``/``delta`` (and the file ``title``) are recorded in
/// ``frame.meta``.
///
/// For long trajectories where only a subset of frames is needed, prefer the
/// lazy :class:`DCDTrajReader`, which seeks frame-by-frame instead of loading
/// everything into memory.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.dcd`` file on disk.
///
/// Returns
/// -------
/// list[Frame]
///     All frames in the trajectory.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frames = molrs.read_dcd("trajectory.dcd")
/// >>> len(frames)
/// 100
/// >>> frames[0]["atoms"].view("x")
#[pyfunction]
pub fn read_dcd(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_dcd_rs(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Lazy, indexed reader for DCD trajectory files.
///
/// Unlike :func:`read_dcd`, this does **not** load every frame upfront. The
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
/// Examples
/// --------
/// >>> reader = molrs.DCDTrajReader("trajectory.dcd")
/// >>> len(reader)
/// 1000
/// >>> frame = reader[42]
/// >>> for frame in reader:
/// ...     pass
#[pyclass(name = "DCDTrajReader", unsendable)]
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
    fn py_new(path: &str) -> PyResult<Self> {
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
/// The molrs-native counterpart to :func:`read_xyz_trajectory` (which eagerly
/// returns ``list[Frame]``). Frames are parsed on demand; the frame-offset
/// index is built lazily on first random access or eagerly via
/// ``build_index()``.
///
/// Exposes the same molpy ``BaseTrajectoryReader`` surface as
/// :class:`DCDTrajReader` and :class:`LAMMPSTrajReader`: ``read_frame``,
/// ``read_frames``, ``read_range``, ``read_all``, ``n_frames``, slicing,
/// ``close()``, and context-manager use.
///
/// Parameters
/// ----------
/// path : str
///     Path to a multi-frame ``.xyz`` file.
///
/// Examples
/// --------
/// >>> reader = molrs.XYZTrajReader("traj.xyz")
/// >>> reader.n_frames
/// 50
/// >>> reader[-1]["atoms"].view("x")
#[pyclass(name = "XYZTrajReader", unsendable)]
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
    fn py_new(path: &str) -> PyResult<Self> {
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

/// Read all frames from a GROMACS GRO file.
///
/// GRO is a fixed-column text format used by GROMACS for input structures and
/// single-precision trajectories. Each frame contains an ``"atoms"`` block
/// with columns ``resid``, ``resname``, ``atom_name``, ``atom_id``,
/// ``x``/``y``/``z`` (in nm), and optional ``vx``/``vy``/``vz``. The
/// simulation box is stored in ``frame.simbox``.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.gro`` file on disk.
///
/// Returns
/// -------
/// list[Frame]
///     All frames in the file (single-frame files return a one-element list).
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frames = molrs.read_gro("system.gro")
/// >>> frame = frames[0]
/// >>> atoms = frame["atoms"]
/// >>> atoms.view("atom_name")
#[pyfunction]
pub fn read_gro(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_gro_rs(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Write a Frame to a GROMACS GRO file.
///
/// The Frame must contain an ``"atoms"`` block with at least ``x``, ``y``,
/// ``z`` columns (in nm). Optional columns: ``resid``, ``resname``,
/// ``atom_name``, ``atom_id``, ``vx``, ``vy``, ``vz``. The box is taken
/// from ``frame.simbox``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be written.
/// ValueError
///     If the frame is missing the ``"atoms"`` block or coordinate columns.
#[pyfunction]
pub fn write_gro(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_gro_rs(path, &core_frame).map_err(io_error_to_pyerr)
}

/// Read a VASP CHGCAR or CHGDIF file.
///
/// Returns a Frame containing:
///
/// - ``"atoms"`` block with ``symbol``, ``x``, ``y``, ``z`` (Cartesian Å)
/// - ``simbox``: triclinic periodic box
/// - grid ``"chgcar"``: a :class:`Grid` with at least ``"total"`` (and
///   ``"diff"`` for spin-polarised ISPIN=2 calculations)
///
/// The volumetric values are stored **raw** (ρ × V_cell, units e).
/// Divide by ``simbox.volume()`` to get charge density in e/Å³.
///
/// Parameters
/// ----------
/// path : str
///     Path to a CHGCAR or CHGDIF file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// ValueError
///     On parse errors.
/// IOError
///     If the file cannot be opened.
///
/// Examples
/// --------
/// >>> frame = molrs.read_chgcar("CHGCAR")
/// >>> grid = frame["chgcar"]
/// >>> total = grid["total"]          # shape (nx, ny, nz)
/// >>> density = total / frame.simbox.volume()
#[pyfunction]
pub fn read_chgcar_file(path: &str) -> PyResult<PyFrame> {
    let frame = read_chgcar(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a Gaussian Cube file.
///
/// Returns a Frame containing:
///
/// - ``"atoms"`` block with ``element``, ``x``, ``y``, ``z``,
///   ``atomic_number``, ``charge``
/// - grid ``"cube"``: a :class:`Grid` with ``"density"`` (scalar field)
///   or ``"mo_<idx>"`` arrays (MO variant)
///
/// Values are stored as-is from the file (no unit conversion).
/// The unit system is recorded in ``frame.meta["cube_units"]``
/// (``"bohr"`` or ``"angstrom"``).
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.cube`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// ValueError
///     On parse errors.
/// IOError
///     If the file cannot be opened.
///
/// Examples
/// --------
/// >>> frame = molrs.read_cube_file("density.cube")
/// >>> grid = frame["cube"]
/// >>> density = grid["density"]       # shape (nx, ny, nz)
#[pyfunction]
pub fn read_cube_file(path: &str) -> PyResult<PyFrame> {
    let frame = read_cube(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame to a Gaussian Cube file.
///
/// The Frame must contain a ``"cube"`` grid and an ``"atoms"`` block.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_cube_file(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_cube(path, &core_frame).map_err(molrs_error_to_pyerr)
}

// ============================================================================
// Writers
// ============================================================================

/// Write a Frame to a PDB file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_pdb(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    let file = File::create(path).map_err(io_error_to_pyerr)?;
    let mut buf = BufWriter::new(file);
    write_pdb_frame(&mut buf, &core_frame).map_err(io_error_to_pyerr)
}

/// Write a list of Frames to a multi-MODEL PDB trajectory.
///
/// Each frame becomes one ``MODEL``/``ENDMDL`` block; a shared ``CRYST1`` is
/// written once from the first frame. Inverse of :func:`read_pdb_trajectory`.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
///     Frames to write, in order.
#[pyfunction]
pub fn write_pdb_trajectory(path: &str, frames: Vec<PyFrame>) -> PyResult<()> {
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    let file = File::create(path).map_err(io_error_to_pyerr)?;
    let mut buf = BufWriter::new(file);
    write_pdb_traj(&mut buf, &core_frames).map_err(io_error_to_pyerr)
}

/// Write a Frame to an XYZ file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_xyz(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    let file = File::create(path).map_err(io_error_to_pyerr)?;
    let mut buf = BufWriter::new(file);
    write_xyz_frame(&mut buf, &core_frame).map_err(io_error_to_pyerr)
}

/// Write a Frame to a LAMMPS data file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_lammps(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_lammps_data(path, &core_frame).map_err(io_error_to_pyerr)
}

/// Write Frames to a LAMMPS dump trajectory file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
///     Frames to write.
#[pyfunction]
pub fn write_lammps_traj(path: &str, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_lammps_dump(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Write Frames to a DCD trajectory file.
///
/// Produces a NAMD-compatible little-endian DCD. Every frame must have the
/// same atom count and the same box presence as the first frame. The box, if
/// any, is taken from each ``frame.simbox``.
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
pub fn write_dcd(path: &str, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_dcd_rs(path, &core_frames).map_err(io_error_to_pyerr)
}

// ============================================================================
// GROMACS TRR / XTC trajectories
// ============================================================================

/// Read every frame of a GROMACS TRR trajectory and return a list of Frames.
///
/// TRR is the full-precision GROMACS format. Each frame's ``"atoms"`` block has
/// ``id`` and ``x``/``y``/``z`` (nm), plus ``vx``/``vy``/``vz`` and
/// ``fx``/``fy``/``fz`` when the frame carries velocities / forces. The box is
/// in ``frame.simbox``; ``step``/``time``/``lambda`` in ``frame.meta``.
///
/// For long trajectories prefer the lazy :class:`TRRTrajReader`.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.trr`` file.
///
/// Returns
/// -------
/// list[Frame]
///
/// Examples
/// --------
/// >>> frames = molrs.read_trr("traj.trr")
#[pyfunction]
pub fn read_trr(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_trr_rs(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Read every frame of a GROMACS XTC trajectory and return a list of Frames.
///
/// XTC is the compressed GROMACS format (lossy, ``1/precision`` nm resolution).
/// Each frame's ``"atoms"`` block has ``id`` and ``x``/``y``/``z`` (nm); the box
/// is in ``frame.simbox``; ``step``/``time``/``precision`` in ``frame.meta``.
/// Both the classic (1995) and 2023 magic numbers are accepted.
///
/// For long trajectories prefer the lazy :class:`XTCTrajReader`.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xtc`` file.
///
/// Returns
/// -------
/// list[Frame]
///
/// Examples
/// --------
/// >>> frames = molrs.read_xtc("traj.xtc")
#[pyfunction]
pub fn read_xtc(path: &str) -> PyResult<Vec<PyFrame>> {
    let frames = read_xtc_rs(path).map_err(io_error_to_pyerr)?;
    frames.into_iter().map(PyFrame::from_core_frame).collect()
}

/// Lazy, indexed reader for GROMACS TRR trajectory files.
///
/// Builds a per-frame byte-offset index on first random access (or eagerly via
/// ``build_index()``); subsequent ``reader[i]`` / ``read_step(i)`` is an O(1)
/// seek plus one frame parse. Exposes the same surface as
/// :class:`DCDTrajReader`.
#[pyclass(name = "TRRTrajReader", unsendable)]
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
    fn py_new(path: &str) -> PyResult<Self> {
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
                Ok(n) => format!("TRRTrajReader(n_frames={})", n),
                Err(_) => "TRRTrajReader(<unread>)".to_string(),
            },
            None => "TRRTrajReader(<closed>)".to_string(),
        }
    }
}

/// Lazy, indexed reader for GROMACS XTC trajectory files.
///
/// Like :class:`TRRTrajReader` but for the compressed XTC format. Frame sizes
/// vary (compression), so the byte-offset index is built by a single scan;
/// random access is O(1) thereafter.
#[pyclass(name = "XTCTrajReader", unsendable)]
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
    fn py_new(path: &str) -> PyResult<Self> {
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
                Ok(n) => format!("XTCTrajReader(n_frames={})", n),
                Err(_) => "XTCTrajReader(<unread>)".to_string(),
            },
            None => "XTCTrajReader(<closed>)".to_string(),
        }
    }
}

/// Write Frames to a GROMACS TRR trajectory file (single precision).
///
/// Each frame's ``"atoms"`` block must have ``x``/``y``/``z`` (nm); optional
/// ``vx``/``vy``/``vz`` and ``fx``/``fy``/``fz`` are written when present. The
/// box, if any, is taken from each ``frame.simbox``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
#[pyfunction]
pub fn write_trr(path: &str, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
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
/// is taken from each ``frame.simbox``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
#[pyfunction]
pub fn write_xtc(path: &str, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    write_xtc_rs(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Intermediate representation of a parsed SMILES or SMARTS string.
///
/// This is the raw syntax tree produced by the parser. Convert it to a
/// molecular graph via :meth:`to_atomistic`.
///
/// Attributes
/// ----------
/// n_components : int
///     Number of disconnected components (fragments separated by ``'.'``
///     in the SMILES string).
///
/// Examples
/// --------
/// >>> ir = molrs.parse_smiles("CCO")
/// >>> ir.n_components
/// 1
/// >>> mol = ir.to_atomistic()
/// >>> mol.n_atoms
/// 3
#[pyclass(name = "SmilesIR")]
pub struct PySmilesIR {
    inner: molrs::io::smiles::SmilesIR,
    input: String,
}

#[pymethods]
impl PySmilesIR {
    /// Number of disconnected molecular components.
    ///
    /// Fragments separated by ``'.'`` in the SMILES string are counted as
    /// separate components.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.components.len()
    }

    /// Convert the SMILES intermediate representation to an all-atom
    /// molecular graph.
    ///
    /// Hydrogen atoms that are implicit in the SMILES string are **not**
    /// added here; use :class:`Conformer` with ``add_hydrogens=True`` for
    /// that.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     Molecular graph with atoms and bonds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the IR contains invalid ring-closure or stereochemistry data.
    ///
    /// Examples
    /// --------
    /// >>> mol = parse_smiles("c1ccccc1").to_atomistic()
    /// >>> mol.n_atoms
    /// 6
    fn to_atomistic(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        let mol = molrs::io::smiles::to_atomistic(&self.inner).map_err(smiles_error_to_pyerr)?;
        PyAtomistic::from_core(py, mol)
    }

    fn __repr__(&self) -> String {
        format!(
            "SmilesIR('{}', components={})",
            self.input,
            self.inner.components.len()
        )
    }
}

/// Parse a SMILES string into an intermediate representation.
///
/// The returned :class:`SmilesIR` can be converted to an :class:`Atomistic`
/// molecular graph via :meth:`SmilesIR.to_atomistic`.
///
/// Parameters
/// ----------
/// smiles : str
///     SMILES string (e.g. ``"CCO"`` for ethanol, ``"c1ccccc1"`` for benzene).
///
/// Returns
/// -------
/// SmilesIR
///     Parsed intermediate representation.
///
/// Raises
/// ------
/// ValueError
///     If the SMILES string is syntactically invalid.
///
/// Examples
/// --------
/// >>> ir = molrs.parse_smiles("CCO")
/// >>> mol = ir.to_atomistic()
/// >>> mol.n_atoms
/// 3
#[pyfunction]
pub fn parse_smiles(smiles: &str) -> PyResult<PySmilesIR> {
    let ir = molrs::io::smiles::parse_smiles(smiles).map_err(smiles_error_to_pyerr)?;
    Ok(PySmilesIR {
        inner: ir,
        input: smiles.to_owned(),
    })
}
