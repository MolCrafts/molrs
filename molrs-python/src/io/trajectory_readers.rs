//! The lazy trajectory readers of `molrs.io` — one class per format
//! (`molrs.io.pdb.PdbReader`, `molrs.io.xyz.XyzReader`,
//! `molrs.io.gro.GroReader`, `molrs.io.lammps.LammpsDumpReader`,
//! `molrs.io.dcd.DcdReader`, `molrs.io.trr.TrrReader`,
//! `molrs.io.xtc.XtcReader`) — the `read_<fmt>_trajectory` doors that open
//! them, and the multi-frame writers.
//!
//! Every reader takes one path or a sequence of paths (their frames are
//! concatenated) and reads frames on demand through the format's Rust
//! reader: `reader[i]`, slices, `read_frame`, `read_frames`, `read_range`,
//! `read_all`, iteration, `len()`, `close()` and use as a context manager.

use std::path::PathBuf;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};

use crate::core::frame::PyFrame;
use crate::error::io_error_to_pyerr;
use crate::path::path_str;
use molrs::core::Frame;
use molrs::io::reader::{ReadSeek, TrajectoryReader};

/// One trajectory over the files of one format, read in file order.
struct Concatenated<R> {
    readers: Vec<R>,
    /// Each file's frame count, once asked for.
    counts: Vec<Option<usize>>,
}

impl<R: TrajectoryReader> Concatenated<R> {
    fn new(readers: Vec<R>) -> Self {
        let counts = vec![None; readers.len()];
        Self { readers, counts }
    }

    fn count(&mut self, file: usize) -> std::io::Result<usize> {
        if let Some(n) = self.counts[file] {
            return Ok(n);
        }
        let n = self.readers[file].len()?;
        self.counts[file] = Some(n);
        Ok(n)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        (0..self.readers.len()).map(|file| self.count(file)).sum()
    }

    fn build_index(&mut self) -> std::io::Result<()> {
        self.readers
            .iter_mut()
            .try_for_each(TrajectoryReader::build_index)
    }

    /// Frame `step` of the whole sequence, or `None` past its end.
    fn read_step(&mut self, mut step: usize) -> std::io::Result<Option<Frame>> {
        for file in 0..self.readers.len() {
            let n = self.count(file)?;
            if step < n {
                return self.readers[file].read_step(step);
            }
            step -= n;
        }
        Ok(None)
    }
}

/// A Python index (negative from the end) as an in-range step.
fn resolve(index: isize, n: usize) -> PyResult<usize> {
    let n = n as isize;
    let i = if index < 0 { index + n } else { index };
    if i < 0 || i >= n {
        return Err(PyIndexError::new_err("trajectory index out of range"));
    }
    Ok(i as usize)
}

/// Every path `paths` names: one `str` / `os.PathLike`, or a sequence of them.
fn path_list(paths: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(one) = paths.extract::<PathBuf>() {
        return Ok(vec![path_str(&one)?.to_owned()]);
    }
    let mut out = Vec::new();
    for item in paths.try_iter()? {
        let path: PathBuf = item?.extract()?;
        out.push(path_str(&path)?.to_owned());
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("no trajectory file given"));
    }
    Ok(out)
}

/// Read an in-range step as a Python frame.
fn frame_at<R: TrajectoryReader>(inner: &mut Concatenated<R>, step: usize) -> PyResult<PyFrame> {
    let frame = inner
        .read_step(step)
        .map_err(io_error_to_pyerr)?
        .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
    PyFrame::from_core_frame(frame)
}

/// The steps a Python slice over `n` frames selects.
fn slice_steps(start: isize, stop: isize, step: isize) -> Vec<usize> {
    let mut steps = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        steps.push(i as usize);
        i += step;
    }
    steps
}

macro_rules! lazy_reader {
    (
        $py:ident, $name:literal, $module:literal, $inner:ty, $open:path,
        door = $door:ident, format = $format:literal
    ) => {
        #[doc = concat!(
            "Lazy, indexed reader of ", $format, " files.\n\n",
            "Frames are parsed on demand: random access (``reader[i]``,\n",
            "``read_frame(i)``) indexes a file once, then seeks; iteration walks\n",
            "the files in order. Several files read as one trajectory, their\n",
            "frames concatenated.\n\n",
            "Parameters\n",
            "----------\n",
            "paths : str | os.PathLike | Sequence[str | os.PathLike]\n",
            "    One file, or several read in order. Gzip files (``.gz``) are\n",
            "    decompressed into memory.\n\n",
            "Examples\n",
            "--------\n",
            ">>> traj = molrs.io.", stringify!($door), "(\"run.", $format, "\")  # doctest: +SKIP\n",
            ">>> last = traj[-1]  # doctest: +SKIP\n"
        )]
        #[pyclass(module = $module, name = $name, unsendable)]
        pub struct $py {
            inner: Option<Concatenated<$inner>>,
            /// The iteration cursor: `(file, frame within it)`.
            cursor: (usize, usize),
        }

        impl $py {
            fn open_paths(paths: &Bound<'_, PyAny>) -> PyResult<Self> {
                let readers = path_list(paths)?
                    .iter()
                    .map(|path| $open(path))
                    .collect::<std::io::Result<Vec<_>>>()
                    .map_err(io_error_to_pyerr)?;
                Ok(Self {
                    inner: Some(Concatenated::new(readers)),
                    cursor: (0, 0),
                })
            }

            fn reader(&mut self) -> PyResult<&mut Concatenated<$inner>> {
                self.inner.as_mut().ok_or_else(|| {
                    PyValueError::new_err(concat!("operation on a closed ", $name))
                })
            }

            fn n(&mut self) -> PyResult<usize> {
                self.reader()?.len().map_err(io_error_to_pyerr)
            }
        }

        #[pymethods]
        impl $py {
            #[new]
            fn py_new(paths: &Bound<'_, PyAny>) -> PyResult<Self> {
                Self::open_paths(paths)
            }

            /// Number of frames over every file (indexes each file once).
            #[getter]
            fn n_frames(&mut self) -> PyResult<usize> {
                self.n()
            }

            /// Index every file now rather than on first random access.
            fn build_index(&mut self) -> PyResult<()> {
                self.reader()?.build_index().map_err(io_error_to_pyerr)
            }

            /// Read one frame by index (negative counts from the end).
            ///
            /// Raises ``IndexError`` if out of range.
            fn read_frame(&mut self, index: isize) -> PyResult<PyFrame> {
                let n = self.n()?;
                let step = resolve(index, n)?;
                frame_at(self.reader()?, step)
            }

            /// Read an explicit list of frame indices (each may be negative).
            fn read_frames(&mut self, indices: Vec<isize>) -> PyResult<Vec<PyFrame>> {
                let n = self.n()?;
                indices
                    .into_iter()
                    .map(|i| {
                        let step = resolve(i, n)?;
                        frame_at(self.reader()?, step)
                    })
                    .collect()
            }

            /// Read a range of frames, Python-slice style.
            #[pyo3(signature = (start=0, stop=None, step=1))]
            fn read_range(
                &mut self,
                py: Python<'_>,
                start: isize,
                stop: Option<isize>,
                step: isize,
            ) -> PyResult<Vec<PyFrame>> {
                if step == 0 {
                    return Err(PyValueError::new_err("read_range step must not be zero"));
                }
                let n = self.n()?;
                let slice = PySlice::new(py, start, stop.unwrap_or(if step > 0 { isize::MAX } else { isize::MIN }), step);
                let ix = slice.indices(n as isize)?;
                slice_steps(ix.start, ix.stop, ix.step)
                    .into_iter()
                    .map(|s| frame_at(self.reader()?, s))
                    .collect()
            }

            /// Read every frame into a list.
            fn read_all(&mut self) -> PyResult<Vec<PyFrame>> {
                let n = self.n()?;
                (0..n).map(|s| frame_at(self.reader()?, s)).collect()
            }

            /// Release every file handle. Further reads raise ``ValueError``.
            fn close(&mut self) {
                self.inner = None;
            }

            fn __len__(&mut self) -> PyResult<usize> {
                self.n()
            }

            fn __getitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let py = key.py();
                let n = self.n()?;
                if let Ok(slice) = key.cast::<PySlice>() {
                    let ix = slice.indices(n as isize)?;
                    let frames = slice_steps(ix.start, ix.stop, ix.step)
                        .into_iter()
                        .map(|s| frame_at(self.reader()?, s))
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PyList::new(py, frames)?.into_any().unbind())
                } else {
                    let step = resolve(key.extract()?, n)?;
                    Ok(Py::new(py, frame_at(self.reader()?, step)?)?.into_any())
                }
            }

            /// Walk the files in order, one frame at a time, without
            /// indexing a file before reading it.
            fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
                slf.cursor = (0, 0);
                slf
            }

            fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
                loop {
                    let (file, local) = self.cursor;
                    let inner = self.reader()?;
                    if file >= inner.readers.len() {
                        return Ok(None);
                    }
                    match inner.readers[file].read_step(local).map_err(io_error_to_pyerr)? {
                        Some(frame) => {
                            self.cursor = (file, local + 1);
                            return Ok(Some(PyFrame::from_core_frame(frame)?));
                        }
                        None => self.cursor = (file + 1, 0),
                    }
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
                    Some(inner) => {
                        let files = inner.readers.len();
                        match inner.len() {
                            Ok(n) => format!(concat!($name, "(n_frames={}, files={})"), n, files),
                            Err(_) => format!(concat!($name, "(<unread>, files={})"), files),
                        }
                    }
                    None => concat!($name, "(<closed>)").to_owned(),
                }
            }
        }

        #[doc = concat!(
            "Open one ", $format, " trajectory, or several whose frames are\n",
            "concatenated, as a lazy :class:`", $module, ".", $name, "`.\n\n",
            "Parameters\n",
            "----------\n",
            "paths : str | os.PathLike | Sequence[str | os.PathLike]\n",
            "    One file, or several read in order.\n\n",
            "Returns\n",
            "-------\n",
            $module, ".", $name, "\n"
        )]
        #[pyfunction]
        pub fn $door(paths: &Bound<'_, PyAny>) -> PyResult<$py> {
            $py::open_paths(paths)
        }
    };
}

lazy_reader!(
    PyPdbReader,
    "PdbReader",
    "molrs.io.pdb",
    molrs::io::pdb::PdbReader<Box<dyn ReadSeek>>,
    molrs::io::pdb::PdbReader::open,
    door = read_pdb_trajectory,
    format = "pdb"
);
lazy_reader!(
    PyXyzReader,
    "XyzReader",
    "molrs.io.xyz",
    molrs::io::xyz::XyzReader<Box<dyn ReadSeek>>,
    molrs::io::xyz::XyzReader::open,
    door = read_xyz_trajectory,
    format = "xyz"
);
lazy_reader!(
    PyGroReader,
    "GroReader",
    "molrs.io.gro",
    molrs::io::gro::GroReader<Box<dyn ReadSeek>>,
    molrs::io::gro::GroReader::open,
    door = read_gro_trajectory,
    format = "gro"
);
lazy_reader!(
    PyLammpsDumpReader,
    "LammpsDumpReader",
    "molrs.io.lammps",
    molrs::io::lammps::LammpsDumpReader<Box<dyn ReadSeek>>,
    molrs::io::lammps::LammpsDumpReader::open,
    door = read_lammps_dump_trajectory,
    format = "lammpstrj"
);
lazy_reader!(
    PyDcdReader,
    "DcdReader",
    "molrs.io.dcd",
    molrs::io::dcd::DcdReader<Box<dyn ReadSeek>>,
    molrs::io::dcd::DcdReader::open,
    door = read_dcd_trajectory,
    format = "dcd"
);
lazy_reader!(
    PyTrrReader,
    "TrrReader",
    "molrs.io.trr",
    molrs::io::trr::TrrReader<Box<dyn ReadSeek>>,
    molrs::io::trr::TrrReader::open,
    door = read_trr_trajectory,
    format = "trr"
);
lazy_reader!(
    PyXtcReader,
    "XtcReader",
    "molrs.io.xtc",
    molrs::io::xtc::XtcReader<Box<dyn ReadSeek>>,
    molrs::io::xtc::XtcReader::open,
    door = read_xtc_trajectory,
    format = "xtc"
);

/// Every frame of a Python list, as core frames.
fn core_frames(frames: &[PyRef<'_, PyFrame>]) -> PyResult<Vec<Frame>> {
    frames.iter().map(|f| f.clone_core_frame()).collect()
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
pub fn write_lammps_dump_trajectory(
    path: PathBuf,
    frames: Vec<PyRef<'_, PyFrame>>,
    columns: Option<Vec<String>>,
) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames = core_frames(&frames)?;
    let chosen: Option<Vec<&str>> = columns
        .as_ref()
        .map(|c| c.iter().map(String::as_str).collect());
    molrs::io::write_lammps_dump_trajectory(path, &core_frames, chosen.as_deref())
        .map_err(io_error_to_pyerr)
}

/// Write Frames as LAMMPS ``dump local`` (OVITO Load Trajectory bonds).
///
/// Emits ``ITEM: NUMBER OF ENTRIES`` + ``ITEM: ENTRIES batom1 batom2 [btype]``.
/// Rows come from ``entries`` if present, otherwise from canonical ``bonds``.
#[pyfunction]
pub fn write_lammps_dump_local(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames = core_frames(&frames)?;
    molrs::io::write_lammps_dump_local(path, &core_frames).map_err(io_error_to_pyerr)
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
    let core_frames = core_frames(&frames)?;
    molrs::io::write_dcd_trajectory(path, &core_frames).map_err(io_error_to_pyerr)
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
    let core_frames = core_frames(&frames)?;
    molrs::io::write_trr_trajectory(path, &core_frames).map_err(io_error_to_pyerr)
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
    let core_frames = core_frames(&frames)?;
    molrs::io::write_xtc_trajectory(path, &core_frames).map_err(io_error_to_pyerr)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPdbReader>()?;
    m.add_class::<PyXyzReader>()?;
    m.add_class::<PyGroReader>()?;
    m.add_class::<PyLammpsDumpReader>()?;
    m.add_class::<PyDcdReader>()?;
    m.add_class::<PyTrrReader>()?;
    m.add_class::<PyXtcReader>()?;
    for door in [
        wrap_pyfunction!(read_pdb_trajectory, m)?,
        wrap_pyfunction!(read_xyz_trajectory, m)?,
        wrap_pyfunction!(read_gro_trajectory, m)?,
        wrap_pyfunction!(read_lammps_dump_trajectory, m)?,
        wrap_pyfunction!(read_dcd_trajectory, m)?,
        wrap_pyfunction!(read_trr_trajectory, m)?,
        wrap_pyfunction!(read_xtc_trajectory, m)?,
        wrap_pyfunction!(write_lammps_dump_trajectory, m)?,
        wrap_pyfunction!(write_lammps_dump_local, m)?,
        wrap_pyfunction!(write_dcd_trajectory, m)?,
        wrap_pyfunction!(write_trr_trajectory, m)?,
        wrap_pyfunction!(write_xtc_trajectory, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
