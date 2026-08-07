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
//! | XSF | [`read_xsf`] | [`write_xsf`] |
//! | AMBER inpcrd | [`read_amber_inpcrd`] | — |
//! | AMBER prmtop (structure) | [`read_amber_prmtop`] | — |

use crate::core::store::block::PyBlock;
use crate::core::store::frame::PyFrame;
use pyo3::types::PyBytes;
use crate::core::system::molgraph::PyAtomistic;
use crate::helpers::{io_error_to_pyerr, molrs_error_to_pyerr, smiles_error_to_pyerr};
use molrs::io::data::chgcar::read_chgcar;
use molrs::io::data::cube::{read_cube, write_cube};
use molrs::io::data::gro::{read_gro as read_gro_rs, write_gro as write_gro_rs};
use molrs::io::data::inpcrd::read_amber_inpcrd as read_amber_inpcrd_rs;
use molrs::io::data::ac::read_ac as read_ac_rs;
use molrs::io::data::frcmod::{
    parse_frcmod as parse_frcmod_rs, read_frcmod as read_frcmod_rs,
    write_frcmod as write_frcmod_rs, FrcmodFile,
};
use molrs::io::data::prep::{
    read_prep as read_prep_rs, write_prep as write_prep_rs, PrepAtom, PrepResidue,
};
use molrs::io::data::prmtop::{
    read_amber_prmtop as read_amber_prmtop_rs,
    read_amber_prmtop_sections as read_amber_prmtop_sections_rs,
};
use molrs::io::data::prmtop_tables::{
    decode_angle_params as decode_angle_params_rs,
    decode_bond_params as decode_bond_params_rs,
    decode_dihedral_params as decode_dihedral_params_rs,
    decode_nonbond_params as decode_nonbond_params_rs, parse_a4_names as parse_a4_names_rs,
    parse_pointers as parse_pointers_rs,
};
use molrs::io::data::lammps_data::{read_lammps_data, write_lammps_data};
use molrs::io::data::mol2::{read_mol2 as read_mol2_rs, write_mol2 as write_mol2_rs};
use molrs::io::data::top::{read_top as read_top_rs, write_top as write_top_rs};
use molrs::io::data::lammps_molecule::{
    read_lammps_molecule as read_lammps_molecule_rs,
    write_lammps_molecule as write_lammps_molecule_rs,
};
use molrs::io::data::pdb::{read_pdb_frame, read_pdb_traj, write_pdb_frame, write_pdb_traj};
use molrs::io::data::xsf::{read_xsf as read_xsf_rs, write_xsf as write_xsf_rs};
use molrs::io::data::xyz::{XYZReader, read_xyz_frame, read_xyz_traj, write_xyz_frame};
use molrs::io::log::lammps::{
    parse_lammps_log_text as parse_lammps_log_text_rs,
    read_lammps_log_with_style as read_lammps_log_rs,
};
use molrs::io::reader::{ReadSeek, TrajectoryReader, open_seekable};
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
use pyo3::exceptions::{PyFileNotFoundError, PyIndexError, PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyType};
use serde_json::Value as JsonValue;
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

/// Read a frame by step index, returning ``None`` when out of bounds (lenient
/// variant retained for backward compatibility).
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
#[pyclass(module = "molrs.io.raw", name = "LAMMPSTrajReader", unsendable)]
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
/// (Å). Unit-cell information, when present, is stored in ``frame.box``;
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
#[pyclass(module = "molrs.io.raw", name = "DCDTrajReader", unsendable)]
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
#[pyclass(module = "molrs.io.raw", name = "XYZTrajReader", unsendable)]
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
/// simulation box is stored in ``frame.box``.
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
/// from ``frame.box``.
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
/// - ``box``: triclinic periodic box
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
/// >>> density = total / frame.box.volume()
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

/// Read a Tripos MOL2 file and return the first molecule as a Frame.
///
/// Format-native columns on atoms: ``id``, ``name``, ``x``/``y``/``z``,
/// ``atom_type``, optional ``subst_id``/``subst_name``/``charge``. Bonds carry
/// ``atomi``/``atomj`` (0-based), ``sybyl_bond_type``, and canonical
/// ``bond_type``/``bond_number``.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.mol2`` file.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_mol2(path: &str) -> PyResult<PyFrame> {
    let frame = read_mol2_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an AMBER ASCII inpcrd / restrt coordinate file.
///
/// Fixed-width Fortran ``6F12.7`` layout. Returns a Frame with
/// ``id``, ``name``, ``x``/``y``/``z``, optional ``vel`` (shape ``[n, 3]``),
/// optional box, and meta ``title`` / ``timestep``.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.inpcrd`` or restart file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_amber_inpcrd(path: &str) -> PyResult<PyFrame> {
    let frame = read_amber_inpcrd_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Alias for [`read_amber_inpcrd`].
#[pyfunction]
pub fn read_inpcrd(path: &str) -> PyResult<PyFrame> {
    read_amber_inpcrd(path)
}

/// Read an AMBER prmtop **structure** file into a Frame.
///
/// Structure / connectivity only: atoms (name, type, charge in electron units,
/// mass, optional atomic_number/element, res_id), bonds/angles/dihedrals with
/// 0-based indices and type labels, plus POINTERS meta. Force-field parameter
/// tables are not assembled.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.prmtop`` / ``.parm7`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_amber_prmtop(path: &str) -> PyResult<PyFrame> {
    let frame = read_amber_prmtop_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Alias for [`read_amber_prmtop`].
#[pyfunction]
pub fn read_prmtop(path: &str) -> PyResult<PyFrame> {
    read_amber_prmtop(path)
}

/// Read raw prmtop ``%FLAG`` sections as ``{flag: [lines...]}``.
#[pyfunction]
pub fn read_amber_prmtop_sections(
    path: &str,
) -> PyResult<std::collections::HashMap<String, Vec<String>>> {
    read_amber_prmtop_sections_rs(path).map_err(io_error_to_pyerr)
}

/// Parse POINTERS lines into the historical meta map (raw + derived counts).
#[pyfunction]
pub fn prmtop_parse_pointers(
    lines: Vec<String>,
) -> PyResult<std::collections::HashMap<String, i64>> {
    parse_pointers_rs(&lines).map_err(PyValueError::new_err)
}

/// Parse Fortran ``20a4`` name fields from section lines.
#[pyfunction]
pub fn prmtop_parse_a4_names(lines: Vec<String>) -> Vec<String> {
    parse_a4_names_rs(&lines)
}

/// Decode bond pointer tables → ``(type, i, j, K, r0)`` (atoms 1-based).
#[pyfunction]
pub fn prmtop_decode_bond_params(
    pointers: Vec<i64>,
    force_k: Vec<f64>,
    equil: Vec<f64>,
) -> PyResult<Vec<(i64, i64, i64, f64, f64)>> {
    decode_bond_params_rs(&pointers, &force_k, &equil).map_err(PyValueError::new_err)
}

/// Decode angle pointer tables → ``(type, i, j, k, K, theta0_deg)`` (1-based).
#[pyfunction]
pub fn prmtop_decode_angle_params(
    pointers: Vec<i64>,
    force_k: Vec<f64>,
    equil_rad: Vec<f64>,
) -> PyResult<Vec<(i64, i64, i64, i64, f64, f64)>> {
    decode_angle_params_rs(&pointers, &force_k, &equil_rad).map_err(PyValueError::new_err)
}

/// Decode dihedral pointer tables → ``(type, i, j, k, l, K, phase, n)`` (1-based).
#[pyfunction]
pub fn prmtop_decode_dihedral_params(
    pointers: Vec<i64>,
    force_k: Vec<f64>,
    phase: Vec<f64>,
    periodicity: Vec<f64>,
) -> PyResult<Vec<(i64, i64, i64, i64, i64, f64, f64, i64)>> {
    decode_dihedral_params_rs(&pointers, &force_k, &phase, &periodicity)
        .map_err(PyValueError::new_err)
}

/// Per-atom LJ ``(atom_1based, sigma, epsilon)`` from ICO + A/B.
#[pyfunction]
#[pyo3(signature = (
    n_atom,
    n_types,
    atom_type_index,
    nonbonded_parm_index,
    acoef,
    bcoef,
    hbond_a = vec![],
    hbond_b = vec![],
))]
pub fn prmtop_decode_nonbond_params(
    n_atom: usize,
    n_types: usize,
    atom_type_index: Vec<i64>,
    nonbonded_parm_index: Vec<i64>,
    acoef: Vec<f64>,
    bcoef: Vec<f64>,
    hbond_a: Vec<f64>,
    hbond_b: Vec<f64>,
) -> PyResult<Vec<(i64, f64, f64)>> {
    decode_nonbond_params_rs(
        n_atom,
        n_types,
        &atom_type_index,
        &nonbonded_parm_index,
        &acoef,
        &bcoef,
        &hbond_a,
        &hbond_b,
    )
    .map_err(PyValueError::new_err)
}

/// Read an Antechamber ``.ac`` file into a Frame.
#[pyfunction]
pub fn read_ac(path: &str) -> PyResult<PyFrame> {
    let frame = read_ac_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an Amber prep file into a nested dict (serde JSON shape).
#[pyfunction]
pub fn read_prep<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let res = read_prep_rs(path).map_err(io_error_to_pyerr)?;
    prep_residue_to_pydict(py, &res)
}

/// Write an Amber prep residue from a nested dict.
#[pyfunction]
pub fn write_prep(path: &str, residue: &Bound<'_, PyAny>) -> PyResult<()> {
    let res = py_to_prep_residue(residue)?;
    write_prep_rs(path, &res).map_err(io_error_to_pyerr)
}

fn py_to_prep_residue(residue: &Bound<'_, PyAny>) -> PyResult<PrepResidue> {
    let name: String = residue.get_item("name")?.extract()?;
    let atoms_list = residue.get_item("atoms")?;
    let mut atoms = Vec::new();
    for item in atoms_list.try_iter()? {
        let d = item?;
        atoms.push(PrepAtom {
            index: d.get_item("index")?.extract()?,
            name: d.get_item("name")?.extract()?,
            atom_type: d.get_item("atom_type")?.extract()?,
            tree_type: d
                .get_item("tree_type")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(|| "M".into()),
            na: d.get_item("na").ok().and_then(|v| v.extract().ok()).unwrap_or(0),
            nb: d.get_item("nb").ok().and_then(|v| v.extract().ok()).unwrap_or(0),
            nc: d.get_item("nc").ok().and_then(|v| v.extract().ok()).unwrap_or(0),
            r: d.get_item("r").ok().and_then(|v| v.extract().ok()).unwrap_or(0.0),
            theta: d
                .get_item("theta")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            phi: d.get_item("phi").ok().and_then(|v| v.extract().ok()).unwrap_or(0.0),
            charge: d
                .get_item("charge")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            element: d
                .get_item("element")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_default(),
        });
    }
    let mut impropers = Vec::new();
    if let Ok(imps) = residue.get_item("impropers") {
        for item in imps.try_iter()? {
            let row: Vec<String> = item?.extract()?;
            impropers.push(row);
        }
    }
    Ok(PrepResidue {
        name,
        atoms,
        head_atom: None,
        tail_atom: None,
        impropers,
    })
}

fn prep_residue_to_pydict<'py>(
    py: Python<'py>,
    res: &PrepResidue,
) -> PyResult<Bound<'py, PyDict>> {
    let value = serde_json::to_value(res).map_err(|e| {
        PyValueError::new_err(format!("failed to serialize prep residue: {e}"))
    })?;
    match value {
        JsonValue::Object(map) => json_object_to_pydict(py, &map),
        _ => Err(PyValueError::new_err(
            "internal error: prep residue did not serialize to an object",
        )),
    }
}

/// Read an AMBER FRCMOD file into a section dict.
#[pyfunction]
pub fn read_frcmod(path: &str) -> PyResult<std::collections::HashMap<String, String>> {
    let file = read_frcmod_rs(path).map_err(io_error_to_pyerr)?;
    Ok(frcmod_to_map(file))
}

/// Parse FRCMOD text into a section dict.
#[pyfunction]
pub fn parse_frcmod(text: &str) -> PyResult<std::collections::HashMap<String, String>> {
    Ok(frcmod_to_map(parse_frcmod_rs(text)))
}

/// Write FRCMOD sections (dict with remark/mass/bond/…) to a path.
#[pyfunction]
pub fn write_frcmod(
    path: &str,
    sections: std::collections::HashMap<String, String>,
) -> PyResult<()> {
    let file = map_to_frcmod(sections);
    write_frcmod_rs(path, &file).map_err(io_error_to_pyerr)
}

fn frcmod_to_map(file: FrcmodFile) -> std::collections::HashMap<String, String> {
    let mut m = std::collections::HashMap::new();
    m.insert("remark".into(), file.remark);
    m.insert("raw_text".into(), file.raw_text);
    for key in ["mass", "bond", "angle", "dihe", "improper", "nonbon"] {
        m.insert(
            key.into(),
            file.sections.get(key).cloned().unwrap_or_default(),
        );
    }
    m
}

fn map_to_frcmod(sections: std::collections::HashMap<String, String>) -> FrcmodFile {
    let mut file = FrcmodFile {
        remark: sections.get("remark").cloned().unwrap_or_default(),
        raw_text: sections.get("raw_text").cloned().unwrap_or_default(),
        sections: Default::default(),
    };
    for key in ["mass", "bond", "angle", "dihe", "improper", "nonbon"] {
        if let Some(v) = sections.get(key) {
            if !v.is_empty() {
                file.sections.insert(key.into(), v.clone());
            }
        }
    }
    file
}


/// Write a Frame to a Tripos MOL2 file.
///
/// Expects format-native atom columns (``atom_type``, optional
/// ``subst_id``/``subst_name``). Canonical renames are applied by the
/// :mod:`molrs.io` façade before calling this binding.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_mol2(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_mol2_rs(path, &core_frame).map_err(io_error_to_pyerr)
}

/// Read a GROMACS topology (``.top`` / ``.itp``) **structure** file.
///
/// Structure only (``[ atoms ]``, ``[ bonds ]``, ``[ pairs ]``,
/// ``[ angles ]``, ``[ dihedrals ]``). Force-field parameter tables and
/// ``#include`` expansion are not handled.
///
/// Connectivity atom indices are **1-based** as written in the file.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.top`` or ``.itp`` file.
///
/// Returns
/// -------
/// Frame
///     Blocks for atoms and any connectivity sections present.
#[pyfunction]
pub fn read_top(path: &str) -> PyResult<PyFrame> {
    let frame = read_top_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame as a minimal GROMACS topology structure file.
///
/// Emits ``[ moleculetype ]``, ``[ atoms ]``, optional connectivity
/// sections, then ``[ system ]`` / ``[ molecules ]``. Molecule name is
/// taken from ``frame.meta["name"]`` (fallback ``"MOL"``).
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write (atoms + optional bonds/pairs/angles/dihedrals).
#[pyfunction]
pub fn write_top(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_top_rs(path, &core_frame).map_err(io_error_to_pyerr)
}

/// Read a LAMMPS molecule template (native ``.mol`` or JSON).
#[pyfunction]
pub fn read_lammps_molecule(path: &str) -> PyResult<PyFrame> {
    let frame = read_lammps_molecule_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an XSF (XCrySDen Structure File) and return a Frame.
///
/// Crystal structures (`CRYSTAL` + `PRIMVEC`/`CONVVEC`) yield a periodic box;
/// molecular structures (`MOLECULE`) yield a free box. Atoms carry
/// ``atomic_number``, ``element``, and ``x``/``y``/``z`` (Å).
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xsf`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_xsf(path: &str) -> PyResult<PyFrame> {
    let frame = read_xsf_rs(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame to an XSF (XCrySDen Structure File).
///
/// A defined periodic box produces `CRYSTAL` + `PRIMVEC`/`CONVVEC`; otherwise
/// the structure is written as `MOLECULE`. Atoms need ``atomic_number`` and
/// ``x``/``y``/``z``.
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
#[pyfunction]
pub fn write_xsf(path: &str, frame: &PyFrame) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_xsf_rs(path, &core_frame).map_err(io_error_to_pyerr)
}

/// Read a LAMMPS log file into a nested plain-Python dict.
///
/// The shape matches molpy's ``LAMMPSLog.to_dict()`` payload so higher layers
/// can hydrate dataclasses without re-parsing. Thermo rows are
/// ``list[list[float]]`` (not a NumPy structured array).
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS log file (e.g. ``log.lammps``).
/// style : str, optional
///     Thermo style. Only ``"default"`` is currently parsed.
///
/// Returns
/// -------
/// dict
///     Nested mapping with ``path``, ``version``, ``header``, ``runs``,
///     ``total_wall_time``, ``warnings``, ``raw_text``, and ``style``.
///
/// Raises
/// ------
/// FileNotFoundError
///     If ``path`` does not exist.
/// OSError
///     On other I/O failures.
#[pyfunction]
#[pyo3(signature = (path, style = "default"))]
pub fn read_lammps_log<'py>(
    py: Python<'py>,
    path: &str,
    style: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let log = read_lammps_log_rs(path, style).map_err(lammps_log_io_error)?;
    lammps_log_to_pydict(py, &log)
}

/// Parse a LAMMPS log from an in-memory string (no filesystem access).
///
/// Parameters
/// ----------
/// text : str
///     Full log file contents.
/// path : str, optional
///     Recorded on the result as ``path`` (default ``"<string>"``).
/// style : str, optional
///     Thermo style. Only ``"default"`` is currently parsed.
///
/// Returns
/// -------
/// dict
///     Same nested shape as :func:`read_lammps_log`.
#[pyfunction]
#[pyo3(signature = (text, path = "<string>", style = "default"))]
pub fn parse_lammps_log_text<'py>(
    py: Python<'py>,
    text: &str,
    path: &str,
    style: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let log = parse_lammps_log_text_rs(text, path, style);
    lammps_log_to_pydict(py, &log)
}

fn lammps_log_io_error(e: std::io::Error) -> PyErr {
    if e.kind() == std::io::ErrorKind::NotFound {
        PyFileNotFoundError::new_err(e.to_string())
    } else {
        PyIOError::new_err(e.to_string())
    }
}

fn lammps_log_to_pydict<'py>(
    py: Python<'py>,
    log: &molrs::io::log::LammpsLog,
) -> PyResult<Bound<'py, PyDict>> {
    let value = serde_json::to_value(log).map_err(|e| {
        PyValueError::new_err(format!("failed to serialize LAMMPS log: {e}"))
    })?;
    match value {
        JsonValue::Object(map) => json_object_to_pydict(py, &map),
        _ => Err(PyValueError::new_err(
            "internal error: LAMMPS log did not serialize to an object",
        )),
    }
}

fn json_object_to_pydict<'py>(
    py: Python<'py>,
    map: &serde_json::Map<String, JsonValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in map {
        dict.set_item(key, json_value_to_py(py, value)?)?;
    }
    Ok(dict)
}

fn json_value_to_py(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
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
                list.append(json_value_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        JsonValue::Object(map) => json_object_to_pydict(py, map)?.into_any().unbind(),
    })
}

/// Write a Frame as a LAMMPS molecule template.
///
/// Parameters
/// ----------
/// path : str
///     Output path.
/// frame : Frame
///     Molecule frame.
/// format : str
///     ``"native"`` or ``"json"`` (default ``"native"``).
#[pyfunction]
#[pyo3(signature = (path, frame, format = "native"))]
pub fn write_lammps_molecule(path: &str, frame: &PyFrame, format: &str) -> PyResult<()> {
    let core_frame = frame.clone_core_frame()?;
    write_lammps_molecule_rs(path, &core_frame, format).map_err(io_error_to_pyerr)
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
/// in ``frame.box``; ``step``/``time``/``lambda`` in ``frame.meta``.
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
/// is in ``frame.box``; ``step``/``time``/``precision`` in ``frame.meta``.
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
#[pyclass(module = "molrs.io.raw", name = "TRRTrajReader", unsendable)]
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
#[pyclass(module = "molrs.io.raw", name = "XTCTrajReader", unsendable)]
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
/// box, if any, is taken from each ``frame.box``.
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
/// is taken from each ``frame.box``.
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
#[pyclass(module = "molrs.io", name = "SmilesIR")]
pub struct PySmilesIR {
    inner: molrs::io::smiles::SmilesIR,
    input: String,
}

#[pymethods]
impl PySmilesIR {
    /// Parse `smiles` into its intermediate representation.
    ///
    /// Parameters
    /// ----------
    /// smiles : str
    ///     SMILES string (e.g. ``"CCO"`` for ethanol, ``"c1ccccc1"``).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the SMILES string is syntactically invalid.
    ///
    /// Examples
    /// --------
    /// >>> molrs.SmilesIR("CCO").to_atomistic().n_atoms
    /// 3
    #[new]
    fn new(smiles: &str) -> PyResult<Self> {
        let inner = molrs::io::smiles::parse_smiles(smiles).map_err(smiles_error_to_pyerr)?;
        Ok(Self {
            inner,
            input: smiles.to_owned(),
        })
    }

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

    /// One graph per disconnected component, in input order.
    ///
    /// ``to_atomistic`` returns a *single* graph holding every component;
    /// this returns them separately. Splitting happens on the parsed
    /// components, not by cutting the string on ``'.'`` — a separator is only
    /// a separator once the parser says so.
    ///
    /// Returns
    /// -------
    /// list of Atomistic
    ///
    /// Examples
    /// --------
    /// >>> len(molrs.SmilesIR("CCO.O").components())
    /// 2
    fn components(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAtomistic>>> {
        self.inner
            .components
            .iter()
            .map(|chain| {
                let one = molrs::io::smiles::SmilesIR {
                    components: vec![chain.clone()],
                    span: self.inner.span,
                };
                let mol = molrs::io::smiles::to_atomistic(&one).map_err(smiles_error_to_pyerr)?;
                PyAtomistic::from_core(py, mol)
            })
            .collect()
    }

    /// Write this IR as a SMILES string (concrete atoms only).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the IR contains SMARTS query atoms or SMARTS-only bond operators.
    fn write_smiles(&self) -> PyResult<String> {
        molrs::io::smiles::write_smiles(&self.inner).map_err(smiles_error_to_pyerr)
    }

    /// Write this IR as a SMARTS string (queries allowed).
    fn write_smarts(&self) -> PyResult<String> {
        molrs::io::smiles::write_smarts(&self.inner).map_err(smiles_error_to_pyerr)
    }

    /// Build a concrete IR from an :class:`~molrs.Atomistic` graph.
    ///
    /// All science/representation choices are **keyword-only flags** forwarded
    /// to ``molrs::io::smiles::SmilesEmitOptions``. This is an *io* alternate
    /// constructor — it does **not** live on ``Atomistic`` (no dependency
    /// inversion into core).
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Molecular graph to serialise.
    /// canonical : bool, default True
    /// root : int or None
    ///     Optional atom handle to use as the SMILES root.
    /// aromatic : {"as_marked", "kekule_only"}, default "as_marked"
    /// hydrogens : {"organic_subset", "explicit_all", "as_stored"}, default "organic_subset"
    /// include_stereo : bool, default False
    /// multi_component : {"error_if_multiple", "join_dot", "first_only"}, default "error_if_multiple"
    /// organic_subset : bool, default True
    #[classmethod]
    #[pyo3(signature = (
        mol,
        *,
        canonical = true,
        root = None,
        aromatic = "as_marked",
        hydrogens = "organic_subset",
        include_stereo = false,
        multi_component = "error_if_multiple",
        organic_subset = true,
    ))]
    fn from_atomistic(
        _cls: &Bound<'_, PyType>,
        mol: &PyAtomistic,
        canonical: bool,
        root: Option<u64>,
        aromatic: &str,
        hydrogens: &str,
        include_stereo: bool,
        multi_component: &str,
        organic_subset: bool,
    ) -> PyResult<Self> {
        let opts = build_smiles_emit_options(
            canonical,
            root,
            aromatic,
            hydrogens,
            include_stereo,
            multi_component,
            organic_subset,
        )?;
        let ir = molrs::io::smiles::from_atomistic(mol.core(), &opts)
            .map_err(smiles_error_to_pyerr)?;
        let input = molrs::io::smiles::write_smiles(&ir)
            .unwrap_or_else(|_| "<from_atomistic>".to_owned());
        Ok(Self { inner: ir, input })
    }

    fn __repr__(&self) -> String {
        format!(
            "SmilesIR('{}', components={})",
            self.input,
            self.inner.components.len()
        )
    }
}

fn build_smiles_emit_options(
    canonical: bool,
    root: Option<u64>,
    aromatic: &str,
    hydrogens: &str,
    include_stereo: bool,
    multi_component: &str,
    organic_subset: bool,
) -> PyResult<molrs::io::smiles::SmilesEmitOptions> {
    use molrs::io::smiles::{AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions};
    use molrs::system::molgraph::node_from_u64;

    let aromatic = match aromatic {
        "as_marked" => AromaticEmit::AsMarked,
        "kekule_only" => AromaticEmit::KekuleOnly,
        other => {
            return Err(PyValueError::new_err(format!(
                "aromatic must be 'as_marked' or 'kekule_only', got {other:?}"
            )));
        }
    };
    let hydrogens = match hydrogens {
        "organic_subset" => HydrogenEmit::OrganicSubset,
        "explicit_all" => HydrogenEmit::ExplicitAll,
        "as_stored" => HydrogenEmit::AsStored,
        other => {
            return Err(PyValueError::new_err(format!(
                "hydrogens must be 'organic_subset', 'explicit_all', or 'as_stored', got {other:?}"
            )));
        }
    };
    let multi_component = match multi_component {
        "error_if_multiple" => MultiComponentEmit::ErrorIfMultiple,
        "join_dot" => MultiComponentEmit::JoinDot,
        "first_only" => MultiComponentEmit::FirstOnly,
        other => {
            return Err(PyValueError::new_err(format!(
                "multi_component must be 'error_if_multiple', 'join_dot', or 'first_only', got {other:?}"
            )));
        }
    };
    Ok(SmilesEmitOptions {
        canonical,
        root: root.map(node_from_u64),
        aromatic,
        hydrogens,
        include_stereo,
        multi_component,
        organic_subset,
    })
}

/// Write an :class:`~molrs.Atomistic` to a SMILES string (io surface, not a core method).
///
/// Equivalent to ``SmilesIR.from_atomistic(mol, **flags).write_smiles()``.
/// Keyword flags match :meth:`SmilesIR.from_atomistic`.
///
/// Public name is **``write_smiles``** (not ``write_atomistic_smiles``).
#[pyfunction]
#[pyo3(name = "write_smiles", signature = (
    mol,
    *,
    canonical = true,
    root = None,
    aromatic = "as_marked",
    hydrogens = "organic_subset",
    include_stereo = false,
    multi_component = "error_if_multiple",
    organic_subset = true,
))]
pub fn write_smiles_from_atomistic(
    mol: &PyAtomistic,
    canonical: bool,
    root: Option<u64>,
    aromatic: &str,
    hydrogens: &str,
    include_stereo: bool,
    multi_component: &str,
    organic_subset: bool,
) -> PyResult<String> {
    let opts = build_smiles_emit_options(
        canonical,
        root,
        aromatic,
        hydrogens,
        include_stereo,
        multi_component,
        organic_subset,
    )?;
    // Compose from_atomistic + write_smiles (preferred public path).
    let ir = molrs::io::smiles::from_atomistic(mol.core(), &opts).map_err(smiles_error_to_pyerr)?;
    molrs::io::smiles::write_smiles(&ir).map_err(smiles_error_to_pyerr)
}

/// Encode the local topology around ``center`` as a SMARTS string.
///
/// All science knobs are keyword-only flags (see molrs ``LocalSmartsOptions``).
/// Lives on the **io** surface — not on ``Atomistic`` / ``Atom``.
///
/// Public name is **``write_smarts``** only (no ``write_local_smarts`` alias).
/// For IR → SMARTS string use :meth:`SmilesIR.write_smarts`.
#[pyfunction]
#[pyo3(signature = (
    mol,
    center,
    *,
    reach = 1,
    atomic_number = true,
    include_degree = true,
    include_h_count = true,
    include_charge = true,
    include_aromatic = true,
    include_ring_membership = false,
    include_ring_size = false,
    include_explicit_h_atoms = false,
    include_bond_orders = true,
    neighbor_style = "chain",
    canonical_neighbor_order = true,
))]
pub fn write_smarts(
    mol: &PyAtomistic,
    center: u64,
    reach: u32,
    atomic_number: bool,
    include_degree: bool,
    include_h_count: bool,
    include_charge: bool,
    include_aromatic: bool,
    include_ring_membership: bool,
    include_ring_size: bool,
    include_explicit_h_atoms: bool,
    include_bond_orders: bool,
    neighbor_style: &str,
    canonical_neighbor_order: bool,
) -> PyResult<String> {
    use molrs::io::smiles::{LocalSmartsOptions, NeighborStyle};
    use molrs::system::molgraph::node_from_u64;

    let neighbor_style = match neighbor_style {
        "chain" => NeighborStyle::Chain,
        "recursive" => NeighborStyle::Recursive,
        other => {
            return Err(PyValueError::new_err(format!(
                "neighbor_style must be 'chain' or 'recursive', got {other:?}"
            )));
        }
    };
    let opts = LocalSmartsOptions {
        reach,
        atomic_number,
        include_degree,
        include_h_count,
        include_charge,
        include_aromatic,
        include_ring_membership,
        include_ring_size,
        include_explicit_h_atoms,
        include_bond_orders,
        neighbor_style,
        canonical_neighbor_order,
    };
    // Rust graph→SMARTS entry (local environment). IR-only write is
    // ``SmilesIR.write_smarts()`` / ``molrs::io::smiles::write_smarts(&ir)``.
    molrs::io::smiles::write_local_smarts(mol.core(), node_from_u64(center), &opts)
        .map_err(smiles_error_to_pyerr)
}

// ---------------------------------------------------------------------------
// Store-type serialization.
//
// These read and write the store containers themselves rather than a molecular
// file format, but they are still IO and they live here for the same reason
// `read_pdb` does: turning bytes into a container is a reader's job. `Block`
// and `Frame` deliberately carry no `from_csv` / `from_bytes` constructors —
// a container that parses its own wire formats grows one entry point per
// format and duplicates this module.
// ---------------------------------------------------------------------------

/// Parse CSV ``text`` into a :class:`Block`.
///
/// Each column's dtype is inferred int → float → str. When ``header`` is given
/// the text is treated as headerless and those names are used; otherwise the
/// first non-empty line provides the column names.
#[pyfunction]
#[pyo3(signature = (text, delimiter = ',', header = None))]
pub fn read_block_csv(
    text: &str,
    delimiter: char,
    header: Option<Vec<String>>,
) -> PyResult<PyBlock> {
    let block = molrs::io::store::csv::block_from_csv(text, delimiter, header.as_deref())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    PyBlock::from_core_block(block)
}

/// Serialize a :class:`Block` to CSV text. The inverse of :func:`read_block_csv`.
#[pyfunction]
#[pyo3(signature = (block, delimiter = ',', header = true))]
pub fn write_block_csv(block: &PyBlock, delimiter: char, header: bool) -> PyResult<String> {
    PyBlock::with_block(block, |b| {
        molrs::io::store::csv::block_to_csv(b, delimiter, header)
    })
}

/// Rebuild a :class:`Frame` from streaming wire bytes.
///
/// This is the encoding ``molrs::stream::FrameServer`` puts on the wire, so a
/// consumer decodes a live stream with this and never re-derives the layout.
///
/// Parameters
/// ----------
/// data : bytes
///     A payload produced by :func:`write_frame_bytes` or by a Rust
///     ``FrameServer``.
/// format : {"msgpack", "json"}
///     Wire encoding the payload was written with.
#[pyfunction]
#[pyo3(signature = (data, format = "msgpack"))]
pub fn read_frame_bytes(data: &[u8], format: &str) -> PyResult<PyFrame> {
    let fmt = crate::helpers::message_format(format)?;
    let frame = molrs::stream::bytes_to_frame(data, fmt).map_err(crate::helpers::py_value_err)?;
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
    let fmt = crate::helpers::message_format(format)?;
    let bytes = frame
        .with_frame(|f| molrs::stream::frame_to_bytes(f, fmt))?
        .map_err(crate::helpers::py_value_err)?;
    Ok(PyBytes::new(py, &bytes))
}
