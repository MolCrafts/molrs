//! Scientific-record (`*.mrec`) path doors.
//!
//! `Record` and `Trajectory` are in-memory carriers. These functions and
//! [`PyMrecTrajectoryReader`] are the filesystem doors, exported to Python as
//! `molrs.io.mrec`. The class is `MrecTrajectoryReader` on `_lib` so it does
//! not collide with the dump concatenator `molrs.io.TrajectoryReader`.

use crate::core::store::frame::PyFrame;
use crate::core::store::record::PyMolRec;
use crate::core::store::trajectory::PyTrajectory;
use crate::helpers::molrs_error_to_pyerr;
use molrs::io::mrec::FrameSequence;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// Read a scientific record from a ``*.mrec`` directory.
///
/// Args:
///     path: Filesystem path of the record store.
///
/// Returns:
///     The in-memory :class:`~molrs.Record`.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, the store is
///         not a readable record, or a section fails to decode.
#[pyfunction]
pub fn read_record(path: &str) -> PyResult<PyMolRec> {
    let inner = molrs::io::mrec::read_record_file(path).map_err(molrs_error_to_pyerr)?;
    Ok(PyMolRec { inner })
}

/// Write a scientific record to a ``*.mrec`` directory.
///
/// Args:
///     path: Destination filesystem path.
///     record: In-memory :class:`~molrs.Record` to persist.
///
/// Raises:
///     ValueError: If ``path`` uses a retired ``.zarr`` suffix, the record
///         has no state section, or a section fails to encode.
#[pyfunction]
pub fn write_record(path: &str, record: PyRef<'_, PyMolRec>) -> PyResult<()> {
    molrs::io::mrec::write_record_file(path, &record.inner).map_err(molrs_error_to_pyerr)
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
        let frame = self
            .inner
            .frame(index as u64)
            .map_err(molrs_error_to_pyerr)?
            .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
        PyFrame::from_core_frame(frame)
    }
}
