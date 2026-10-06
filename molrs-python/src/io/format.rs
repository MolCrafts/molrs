//! `molrs.io.read_frame` / `write_frame`: the path-dispatching door of
//! `molrs::io::format`.

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::core::store::frame::PyFrame;
use crate::error::io_error_to_pyerr;
use crate::path::path_str;

/// Read one structure from a file, picking the format from its name.
///
/// Formats: ``pdb``, ``xyz`` (extended XYZ), ``sdf`` (``.mol``), ``mol2``,
/// ``gro``, ``cif``, ``poscar`` (``.vasp``, ``POSCAR*``, ``CONTCAR*``),
/// ``xsf``, ``cube``, ``inpcrd`` (``.rst7``, ``.restrt``, ``.crd``),
/// ``lammps_data`` (``.data``, ``.lmp``) and ``lammps_dump``
/// (``.lammpstrj``, ``.dump``). A multi-structure file gives its first
/// structure. Columns carry the canonical names, as each format's own
/// ``read_*`` gives them.
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     File to read.
/// format : str, optional
///     Format name or extension (``"xyz"``, ``"lammpstrj"``, …), matched
///     case-insensitively; overrides the file name.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// OSError
///     The format cannot be told or is unknown, or the file does not read.
#[pyfunction]
#[pyo3(signature = (path, format = None))]
pub fn read_frame(path: PathBuf, format: Option<&str>) -> PyResult<PyFrame> {
    let frame = molrs::io::read_frame(path_str(&path)?, format).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a frame to a file, picking the format from its name.
///
/// Writable formats: ``pdb``, ``xyz``, ``mol2``, ``gro``, ``cif``,
/// ``poscar``, ``xsf``, ``cube``, ``lammps_data`` and ``lammps_dump`` (one
/// snapshot, every ``atoms`` column). ``sdf`` and ``inpcrd`` are read-only.
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     File to write (replaced if present).
/// frame : Frame
///     Frame to write.
/// format : str, optional
///     Format name or extension; overrides the file name.
///
/// Raises
/// ------
/// OSError
///     The format cannot be told, is unknown or read-only, or the writer
///     refuses the frame.
#[pyfunction]
#[pyo3(signature = (path, frame, format = None))]
pub fn write_frame(path: PathBuf, frame: &PyFrame, format: Option<&str>) -> PyResult<()> {
    let path = path_str(&path)?;
    frame
        .with_frame(|f| molrs::io::write_frame(path, f, format))?
        .map_err(io_error_to_pyerr)
}

/// Register the format-dispatching doors.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_frame, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_frame, m)?)?;
    Ok(())
}
